"""
Knowledge Service - Business logic for knowledge base management

Provides methods for:
- Knowledge entry CRUD
- Pattern retrieval (success patterns, failure pitfalls)
- Field blacklist management
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from backend.services.base import BaseService
from backend.repositories.knowledge_repository import KnowledgeRepository
from backend.models import KnowledgeEntry

logger = logging.getLogger("services.knowledge")


@dataclass
class KnowledgeEntryInfo:
    """Knowledge entry information for responses."""
    id: int
    entry_type: str
    pattern: Optional[str]
    description: Optional[str]
    meta_data: Dict[str, Any]
    usage_count: int
    is_active: bool
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime]


@dataclass
class KnowledgeCreateData:
    """Data for creating a knowledge entry."""
    entry_type: str
    pattern: Optional[str] = None
    description: Optional[str] = None
    meta_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeUpdateData:
    """Data for updating a knowledge entry."""
    pattern: Optional[str] = None
    description: Optional[str] = None
    meta_data: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


@dataclass
class KnowledgeListFilters:
    """Filters for listing knowledge entries."""
    entry_type: Optional[str] = None
    is_active: Optional[bool] = None
    limit: int = 50
    offset: int = 0


class KnowledgeService(BaseService):
    """
    Service for knowledge base operations.
    
    Provides a clean interface for knowledge management,
    abstracting database operations from routers.
    """
    
    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self.knowledge_repo = KnowledgeRepository(db)
    
    # =========================================================================
    # List Operations
    # =========================================================================
    
    async def list_entries(
        self,
        filters: KnowledgeListFilters,
    ) -> List[KnowledgeEntryInfo]:
        """
        List knowledge entries with optional filtering.
        
        Args:
            filters: List filters
            
        Returns:
            List of KnowledgeEntryInfo
        """
        query = select(KnowledgeEntry).order_by(KnowledgeEntry.usage_count.desc())
        
        if filters.entry_type:
            query = query.where(KnowledgeEntry.entry_type == filters.entry_type)
        if filters.is_active is not None:
            query = query.where(KnowledgeEntry.is_active == filters.is_active)
        
        query = query.limit(filters.limit).offset(filters.offset)
        
        result = await self.db.execute(query)
        entries = result.scalars().all()
        
        return [self._to_entry_info(e) for e in entries]
    
    def _to_entry_info(self, e: KnowledgeEntry) -> KnowledgeEntryInfo:
        """Convert KnowledgeEntry to KnowledgeEntryInfo."""
        return KnowledgeEntryInfo(
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
    
    # =========================================================================
    # Specialized Queries
    # =========================================================================
    
    async def get_success_patterns(self, limit: int = 20) -> List[KnowledgeEntryInfo]:
        """
        Get successful alpha patterns for RAG retrieval.
        
        Args:
            limit: Maximum results
            
        Returns:
            List of success pattern entries
        """
        query = (
            select(KnowledgeEntry)
            .where(
                KnowledgeEntry.entry_type == "SUCCESS_PATTERN",
                KnowledgeEntry.is_active == True,
            )
            .order_by(KnowledgeEntry.usage_count.desc())
            .limit(limit)
        )
        
        result = await self.db.execute(query)
        entries = result.scalars().all()
        
        return [self._to_entry_info(e) for e in entries]
    
    async def get_failure_pitfalls(self, limit: int = 50) -> List[KnowledgeEntryInfo]:
        """
        Get failure pitfalls for the feedback loop.
        
        Args:
            limit: Maximum results
            
        Returns:
            List of failure pitfall entries
        """
        query = (
            select(KnowledgeEntry)
            .where(
                KnowledgeEntry.entry_type == "FAILURE_PITFALL",
                KnowledgeEntry.is_active == True,
            )
            .order_by(KnowledgeEntry.created_at.desc())
            .limit(limit)
        )
        
        result = await self.db.execute(query)
        entries = result.scalars().all()
        
        return [self._to_entry_info(e) for e in entries]
    
    async def get_field_blacklist(
        self,
        region: Optional[str] = None,
    ) -> List[KnowledgeEntryInfo]:
        """
        Get blacklisted fields.
        
        Args:
            region: Optional region filter
            
        Returns:
            List of blacklisted field entries
        """
        query = (
            select(KnowledgeEntry)
            .where(
                KnowledgeEntry.entry_type == "FIELD_BLACKLIST",
                KnowledgeEntry.is_active == True,
            )
        )
        
        result = await self.db.execute(query)
        entries = result.scalars().all()
        
        # Filter by region if specified (in-memory filter for meta_data)
        if region:
            entries = [
                e for e in entries 
                if e.meta_data and e.meta_data.get("region") == region
            ]
        
        return [self._to_entry_info(e) for e in entries]
    
    # =========================================================================
    # CRUD Operations
    # =========================================================================
    
    async def create_entry(
        self,
        data: KnowledgeCreateData,
        created_by: str = "USER",
    ) -> KnowledgeEntryInfo:
        """
        Create a new knowledge entry.
        
        Args:
            data: Entry creation data
            created_by: Source of the entry (USER, SYSTEM, FORUM_SYNC, etc.)
            
        Returns:
            Created KnowledgeEntryInfo
        """
        entry = KnowledgeEntry(
            entry_type=data.entry_type,
            pattern=data.pattern,
            description=data.description,
            meta_data=data.meta_data,
            created_by=created_by,
        )
        
        created = await self.knowledge_repo.create(entry)
        await self.commit()
        
        return self._to_entry_info(created)
    
    async def get_entry(self, entry_id: int) -> Optional[KnowledgeEntryInfo]:
        """
        Get a knowledge entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            KnowledgeEntryInfo or None
        """
        entry = await self.knowledge_repo.get_by_id(entry_id)
        if not entry:
            return None
        return self._to_entry_info(entry)
    
    async def update_entry(
        self,
        entry_id: int,
        data: KnowledgeUpdateData,
    ) -> KnowledgeEntryInfo:
        """
        Update a knowledge entry.
        
        Args:
            entry_id: Entry ID
            data: Update data
            
        Returns:
            Updated KnowledgeEntryInfo
            
        Raises:
            ValueError if entry not found
        """
        entry = await self.knowledge_repo.get_by_id(entry_id)
        if not entry:
            raise ValueError(f"Knowledge entry {entry_id} not found")
        
        update_data = {}
        if data.pattern is not None:
            update_data["pattern"] = data.pattern
        if data.description is not None:
            update_data["description"] = data.description
        if data.meta_data is not None:
            update_data["meta_data"] = data.meta_data
        if data.is_active is not None:
            update_data["is_active"] = data.is_active
        
        if update_data:
            await self.knowledge_repo.update_by_id(entry_id, update_data)
            await self.commit()
            # Refresh entry
            entry = await self.knowledge_repo.get_by_id(entry_id)
        
        return self._to_entry_info(entry)
    
    async def delete_entry(self, entry_id: int) -> bool:
        """
        Soft delete a knowledge entry (deactivate).
        
        Args:
            entry_id: Entry ID
            
        Returns:
            True if deleted
            
        Raises:
            ValueError if entry not found
        """
        entry = await self.knowledge_repo.get_by_id(entry_id)
        if not entry:
            raise ValueError(f"Knowledge entry {entry_id} not found")
        
        await self.knowledge_repo.update_by_id(entry_id, {"is_active": False})
        await self.commit()
        
        return True
