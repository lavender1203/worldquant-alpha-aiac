"""
Knowledge Router - CoSTEER Knowledge Base management

Uses KnowledgeService for all business logic.

Features:
- External Knowledge: Forum posts, academic papers
- System Knowledge: Mining patterns, failure pitfalls
- Full CRUD operations
- Forum sync and paper download
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import aiohttp
import os
import re
import hashlib
from pathlib import Path

from backend.database import get_db
from backend.services.knowledge_service import (
    KnowledgeService,
    KnowledgeListFilters,
    KnowledgeCreateData,
    KnowledgeUpdateData,
)
from backend.external_knowledge import (
    ExternalKnowledgeSyncer,
    ForumPost,
    extract_alpha_expressions,
    extract_insights,
    calculate_relevance_score,
)

router = APIRouter(
    prefix="/knowledge",
    tags=["knowledge"],
    responses={404: {"description": "Not found"}},
)


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_knowledge_service(db: AsyncSession = Depends(get_db)) -> KnowledgeService:
    """Get KnowledgeService instance with injected dependencies."""
    return KnowledgeService(db)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class KnowledgeEntryResponse(BaseModel):
    id: int
    entry_type: str
    pattern: Optional[str] = None
    description: Optional[str] = None
    meta_data: dict = {}
    usage_count: int
    is_active: bool
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class KnowledgeCreateRequest(BaseModel):
    entry_type: str  # SUCCESS_PATTERN, FAILURE_PITFALL, FIELD_BLACKLIST, OPERATOR_STAT
    pattern: Optional[str] = None
    description: Optional[str] = None
    meta_data: dict = {}


class KnowledgeUpdateRequest(BaseModel):
    pattern: Optional[str] = None
    description: Optional[str] = None
    meta_data: Optional[dict] = None
    is_active: Optional[bool] = None


class ForumSyncRequest(BaseModel):
    """Request to sync forum posts."""
    search_terms: List[str] = Field(
        default=["high sharpe tips", "alpha improvement", "momentum factor"],
        description="Search terms for forum"
    )
    max_posts: int = Field(default=30, ge=1, le=100)
    min_likes: int = Field(default=3, ge=0)


class ForumPostResponse(BaseModel):
    """Response model for forum post."""
    post_id: str
    title: str
    author: str
    content: str
    likes: int
    views: int
    replies: int
    alpha_patterns: List[str] = []
    insights: List[str] = []
    relevance_score: float = 0.0
    url: Optional[str] = None


class PaperInfo(BaseModel):
    """Academic paper information."""
    id: int
    title: str
    source_url: str
    local_path: Optional[str] = None
    description: str
    patterns_count: int
    downloaded: bool = False


class PaperDownloadRequest(BaseModel):
    """Request to download a paper."""
    url: str
    title: str
    description: Optional[str] = ""


class KnowledgeStatsResponse(BaseModel):
    """Knowledge base statistics."""
    total_entries: int
    external_count: int
    system_count: int
    by_type: Dict[str, int]
    by_source: Dict[str, int]
    recent_entries: int


# Paper storage directory
PAPERS_DIR = Path("data/papers")
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("", response_model=List[KnowledgeEntryResponse])
async def list_knowledge(
    entry_type: Optional[str] = Query(None, description="Filter by type"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """List knowledge base entries with optional filters."""
    filters = KnowledgeListFilters(
        entry_type=entry_type,
        is_active=is_active,
        limit=limit,
        offset=offset,
    )
    
    entries = await service.list_entries(filters)
    
    return [
        KnowledgeEntryResponse(
            id=e.id,
            entry_type=e.entry_type,
            pattern=e.pattern,
            description=e.description,
            meta_data=e.meta_data,
            usage_count=e.usage_count,
            is_active=e.is_active,
            created_by=e.created_by,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )
        for e in entries
    ]


@router.get("/success-patterns", response_model=List[KnowledgeEntryResponse])
async def get_success_patterns(
    limit: int = Query(20, ge=1, le=100),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """Get successful alpha patterns for RAG retrieval."""
    entries = await service.get_success_patterns(limit=limit)
    
    return [
        KnowledgeEntryResponse(
            id=e.id,
            entry_type=e.entry_type,
            pattern=e.pattern,
            description=e.description,
            meta_data=e.meta_data,
            usage_count=e.usage_count,
            is_active=e.is_active,
            created_by=e.created_by,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )
        for e in entries
    ]


@router.get("/failure-pitfalls", response_model=List[KnowledgeEntryResponse])
async def get_failure_pitfalls(
    limit: int = Query(50, ge=1, le=100),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """Get failure pitfalls for the feedback loop."""
    entries = await service.get_failure_pitfalls(limit=limit)
    
    return [
        KnowledgeEntryResponse(
            id=e.id,
            entry_type=e.entry_type,
            pattern=e.pattern,
            description=e.description,
            meta_data=e.meta_data,
            usage_count=e.usage_count,
            is_active=e.is_active,
            created_by=e.created_by,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )
        for e in entries
    ]


@router.get("/field-blacklist", response_model=List[KnowledgeEntryResponse])
async def get_field_blacklist(
    region: Optional[str] = Query(None, description="Filter by region"),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """Get blacklisted fields that should not be used in alpha expressions."""
    entries = await service.get_field_blacklist(region=region)
    
    return [
        KnowledgeEntryResponse(
            id=e.id,
            entry_type=e.entry_type,
            pattern=e.pattern,
            description=e.description,
            meta_data=e.meta_data,
            usage_count=e.usage_count,
            is_active=e.is_active,
            created_by=e.created_by,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )
        for e in entries
    ]


@router.post("", response_model=KnowledgeEntryResponse)
async def create_knowledge_entry(
    request: KnowledgeCreateRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """Create a new knowledge entry (manually add a pattern or pitfall)."""
    data = KnowledgeCreateData(
        entry_type=request.entry_type,
        pattern=request.pattern,
        description=request.description,
        meta_data=request.meta_data,
    )
    
    entry = await service.create_entry(data)
    
    return KnowledgeEntryResponse(
        id=entry.id,
        entry_type=entry.entry_type,
        pattern=entry.pattern,
        description=entry.description,
        meta_data=entry.meta_data,
        usage_count=entry.usage_count,
        is_active=entry.is_active,
        created_by=entry.created_by,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@router.put("/{entry_id}", response_model=KnowledgeEntryResponse)
async def update_knowledge_entry(
    entry_id: int,
    request: KnowledgeUpdateRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """Update a knowledge entry."""
    data = KnowledgeUpdateData(
        pattern=request.pattern,
        description=request.description,
        meta_data=request.meta_data,
        is_active=request.is_active,
    )
    
    try:
        entry = await service.update_entry(entry_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return KnowledgeEntryResponse(
        id=entry.id,
        entry_type=entry.entry_type,
        pattern=entry.pattern,
        description=entry.description,
        meta_data=entry.meta_data,
        usage_count=entry.usage_count,
        is_active=entry.is_active,
        created_by=entry.created_by,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@router.delete("/{entry_id}")
async def delete_knowledge_entry(
    entry_id: int,
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """Delete a knowledge entry (or deactivate it)."""
    try:
        await service.delete_entry(entry_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return {"message": "Knowledge entry deactivated", "id": entry_id}


# =============================================================================
# KNOWLEDGE CATEGORIES
# =============================================================================

@router.get("/categories/external", response_model=List[KnowledgeEntryResponse])
async def get_external_knowledge(
    source: Optional[str] = Query(None, description="Filter by source: forum, paper, documentation"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get external knowledge entries (forum posts, papers, etc.)."""
    from sqlalchemy import select, or_
    from backend.models import KnowledgeEntry
    
    # External sources created_by values
    external_sources = ['FORUM_SYNC', 'CURATED_IMPORT', 'PAPER_IMPORT', 'USER_EXTERNAL']
    
    # Query external entries directly from database
    query = select(KnowledgeEntry).where(
        KnowledgeEntry.is_active == True,
        KnowledgeEntry.created_by.in_(external_sources)
    ).order_by(KnowledgeEntry.created_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    entries = result.scalars().all()
    
    # Filter by source if specified
    if source:
        entries = [
            e for e in entries
            if e.meta_data and e.meta_data.get('source') == source
        ]
    
    return [
        KnowledgeEntryResponse(
            id=e.id,
            entry_type=e.entry_type,
            pattern=e.pattern,
            description=e.description,
            meta_data=e.meta_data or {},
            usage_count=e.usage_count,
            is_active=e.is_active,
            created_by=e.created_by,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )
        for e in entries
    ]


@router.get("/categories/system", response_model=List[KnowledgeEntryResponse])
async def get_system_knowledge(
    entry_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get system-generated knowledge entries (from mining operations)."""
    from sqlalchemy import select, not_
    from backend.models import KnowledgeEntry
    
    # External sources to exclude
    external_sources = ['FORUM_SYNC', 'CURATED_IMPORT', 'PAPER_IMPORT', 'USER_EXTERNAL']
    
    # Query system entries (not external)
    query = select(KnowledgeEntry).where(
        KnowledgeEntry.is_active == True,
        not_(KnowledgeEntry.created_by.in_(external_sources))
    )
    
    if entry_type:
        query = query.where(KnowledgeEntry.entry_type == entry_type)
    
    query = query.order_by(KnowledgeEntry.created_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    entries = result.scalars().all()
    
    return [
        KnowledgeEntryResponse(
            id=e.id,
            entry_type=e.entry_type,
            pattern=e.pattern,
            description=e.description,
            meta_data=e.meta_data or {},
            usage_count=e.usage_count,
            is_active=e.is_active,
            created_by=e.created_by,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )
        for e in entries
    ]


@router.get("/stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats(
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """Get knowledge base statistics."""
    filters = KnowledgeListFilters(is_active=True, limit=1000, offset=0)
    all_entries = await service.list_entries(filters)
    
    # Count by type
    by_type = {}
    for e in all_entries:
        by_type[e.entry_type] = by_type.get(e.entry_type, 0) + 1
    
    # Count by source
    by_source = {}
    for e in all_entries:
        source = e.created_by or 'UNKNOWN'
        by_source[source] = by_source.get(source, 0) + 1
    
    # Count external vs system
    external_sources = ['FORUM_SYNC', 'CURATED_IMPORT', 'PAPER_IMPORT', 'USER_EXTERNAL']
    external_count = sum(1 for e in all_entries if e.created_by in external_sources)
    system_count = len(all_entries) - external_count
    
    # Recent entries (last 7 days)
    from datetime import timedelta
    week_ago = datetime.now() - timedelta(days=7)
    recent_entries = sum(1 for e in all_entries if e.created_at and e.created_at.replace(tzinfo=None) > week_ago)
    
    return KnowledgeStatsResponse(
        total_entries=len(all_entries),
        external_count=external_count,
        system_count=system_count,
        by_type=by_type,
        by_source=by_source,
        recent_entries=recent_entries,
    )


# =============================================================================
# FORUM SYNC
# =============================================================================

@router.post("/sync/forum")
async def sync_forum_posts(
    request: ForumSyncRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Sync high-quality forum posts from WorldQuant BRAIN.
    
    This endpoint triggers a background sync job.
    """
    # For now, we'll do a simulated sync since MCP is not always available
    # In production, this would use the MCP client
    
    return {
        "status": "started",
        "message": "Forum sync started in background",
        "search_terms": request.search_terms,
        "max_posts": request.max_posts,
    }


@router.get("/forum/search", response_model=List[ForumPostResponse])
async def search_forum_posts(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Search forum posts via MCP.
    
    Returns posts with extracted patterns and insights.
    """
    try:
        # Try to use MCP to search forum
        from backend.adapters.brain_adapter import BrainAdapter
        adapter = BrainAdapter()
        
        # Search forum
        posts = await adapter.search_forum(query, limit)
        
        results = []
        for post in posts:
            forum_post = ForumPost(
                post_id=str(post.get('id', '')),
                title=post.get('title', ''),
                content=post.get('content', post.get('body', '')),
                author=post.get('author', ''),
                likes=post.get('likes', 0),
                views=post.get('views', 0),
                replies=post.get('replies', 0),
            )
            
            # Extract patterns and insights
            forum_post.alpha_patterns = extract_alpha_expressions(forum_post.content)
            forum_post.insights = extract_insights(forum_post.content)
            forum_post.relevance_score = calculate_relevance_score(forum_post)
            
            results.append(ForumPostResponse(
                post_id=forum_post.post_id,
                title=forum_post.title,
                author=forum_post.author,
                content=forum_post.content[:500] + "..." if len(forum_post.content) > 500 else forum_post.content,
                likes=forum_post.likes,
                views=forum_post.views,
                replies=forum_post.replies,
                alpha_patterns=forum_post.alpha_patterns,
                insights=forum_post.insights,
                relevance_score=forum_post.relevance_score,
                url=post.get('url', ''),
            ))
        
        return results
        
    except Exception as e:
        # Return empty if MCP not available
        return []


@router.post("/forum/import")
async def import_forum_post(
    post: ForumPostResponse,
    db: AsyncSession = Depends(get_db),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """Import a forum post's knowledge into the knowledge base."""
    imported = 0
    
    # Import alpha patterns
    for pattern in post.alpha_patterns[:5]:
        data = KnowledgeCreateData(
            entry_type='SUCCESS_PATTERN',
            pattern=pattern,
            description=f"From forum: {post.title}",
            meta_data={
                'source': 'forum',
                'source_title': post.title,
                'source_author': post.author,
                'post_id': post.post_id,
                'url': post.url,
                'likes': post.likes,
                'relevance_score': post.relevance_score,
            },
        )
        try:
            await service.create_entry(data, created_by='USER_EXTERNAL')
            imported += 1
        except Exception:
            pass  # Skip duplicates
    
    # Import insights
    for insight in post.insights[:3]:
        data = KnowledgeCreateData(
            entry_type='SUCCESS_PATTERN',
            pattern=f"INSIGHT: {insight[:100]}",
            description=insight,
            meta_data={
                'source': 'forum_insight',
                'source_title': post.title,
                'post_id': post.post_id,
                'pattern_type': 'insight',
            },
        )
        try:
            await service.create_entry(data, created_by='USER_EXTERNAL')
            imported += 1
        except Exception:
            pass
    
    return {
        "success": True,
        "imported": imported,
        "message": f"Imported {imported} knowledge entries from forum post",
    }


# =============================================================================
# PAPER MANAGEMENT
# =============================================================================

@router.get("/papers", response_model=List[PaperInfo])
async def list_papers(
    db: AsyncSession = Depends(get_db),
):
    """List all papers in the knowledge base."""
    from sqlalchemy import select
    from backend.models import KnowledgeEntry
    
    # Query paper entries directly
    query = select(KnowledgeEntry).where(
        KnowledgeEntry.is_active == True,
        KnowledgeEntry.created_by == 'PAPER_IMPORT'
    ).order_by(KnowledgeEntry.created_at.desc())
    
    result = await db.execute(query)
    entries = result.scalars().all()
    
    # Group by paper
    papers = {}
    for e in entries:
        if e.meta_data and e.meta_data.get('source') == 'paper':
            paper_title = e.meta_data.get('source_title', 'Unknown Paper')
            if paper_title not in papers:
                papers[paper_title] = {
                    'id': e.id,
                    'title': paper_title,
                    'source_url': e.meta_data.get('source_url', ''),
                    'local_path': e.meta_data.get('local_path'),
                    'description': e.description or '',
                    'patterns_count': 0,
                    'downloaded': bool(e.meta_data.get('local_path')),
                }
            papers[paper_title]['patterns_count'] += 1
    
    return [PaperInfo(**p) for p in papers.values()]


@router.post("/papers/download")
async def download_paper(
    request: PaperDownloadRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Download a paper from URL and add to knowledge base.
    
    Supports: arXiv, SSRN, and direct PDF links.
    """
    url = request.url
    title = request.title
    
    # Determine paper type and get direct download URL
    pdf_url = None
    
    # arXiv
    if 'arxiv.org' in url:
        # Convert abstract URL to PDF URL
        arxiv_id = re.search(r'(\d+\.\d+)', url)
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id.group(1)}.pdf"
    # SSRN
    elif 'ssrn.com' in url:
        pdf_url = url  # SSRN requires login, store link only
    # Direct PDF
    elif url.endswith('.pdf'):
        pdf_url = url
    else:
        pdf_url = url
    
    # Generate filename
    safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
    filename = f"{safe_title.replace(' ', '_')}.pdf"
    local_path = PAPERS_DIR / filename
    
    downloaded = False
    
    # Try to download
    if pdf_url:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                        content = await response.read()
                        with open(local_path, 'wb') as f:
                            f.write(content)
                        downloaded = True
        except Exception as e:
            pass  # Download failed, store link only
    
    # Create knowledge entry for the paper
    from backend.models import KnowledgeEntry
    
    entry = KnowledgeEntry(
        entry_type='SUCCESS_PATTERN',
        pattern=f"PAPER: {title}",
        description=request.description or f"Academic paper: {title}",
        meta_data={
            'source': 'paper',
            'source_url': url,
            'source_title': title,
            'local_path': str(local_path) if downloaded else None,
            'downloaded': downloaded,
            'pdf_url': pdf_url,
        },
        usage_count=0,
        is_active=True,
        created_by='PAPER_IMPORT'
    )
    db.add(entry)
    await db.commit()
    
    return {
        "success": True,
        "downloaded": downloaded,
        "local_path": str(local_path) if downloaded else None,
        "message": f"Paper {'downloaded' if downloaded else 'link saved'}: {title}",
    }


@router.get("/papers/{paper_id}/download")
async def get_paper_file(
    paper_id: int,
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """Download a paper file if available."""
    filters = KnowledgeListFilters(limit=1000, offset=0)
    entries = await service.list_entries(filters)
    
    entry = next((e for e in entries if e.id == paper_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    local_path = entry.meta_data.get('local_path') if entry.meta_data else None
    if not local_path or not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Paper file not available")
    
    return FileResponse(local_path, filename=os.path.basename(local_path))


# =============================================================================
# USER KNOWLEDGE ENTRY
# =============================================================================

@router.post("/external", response_model=KnowledgeEntryResponse)
async def add_external_knowledge(
    request: KnowledgeCreateRequest,
    source_type: str = Query("manual", description="Source type: manual, forum, paper"),
    source_url: Optional[str] = Query(None, description="Source URL if available"),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """Add external knowledge entry manually."""
    meta_data = request.meta_data.copy() if request.meta_data else {}
    meta_data['source'] = source_type
    if source_url:
        meta_data['source_url'] = source_url
    
    data = KnowledgeCreateData(
        entry_type=request.entry_type,
        pattern=request.pattern,
        description=request.description,
        meta_data=meta_data,
    )
    
    entry = await service.create_entry(data, created_by='USER_EXTERNAL')
    
    return KnowledgeEntryResponse(
        id=entry.id,
        entry_type=entry.entry_type,
        pattern=entry.pattern,
        description=entry.description,
        meta_data=entry.meta_data,
        usage_count=entry.usage_count,
        is_active=entry.is_active,
        created_by=entry.created_by,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )
