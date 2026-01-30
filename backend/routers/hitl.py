"""
Human-in-the-Loop (HITL) Router

This module provides API endpoints for human interaction with the alpha mining system.
Based on Alpha-GPT's human-AI collaboration paradigm:
1. Allow users to inject their own trading ideas/hypotheses
2. Provide feedback on generated alphas (thumbs up/down, detailed review)
3. Guide exploration direction (prefer certain datasets, fields, patterns)
4. Add knowledge entries manually
5. Override or modify generated alphas

Reference: Alpha-GPT 1.0 - Human-AI Interactive Alpha Mining
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from loguru import logger

from backend.database import get_db
from backend.models import Alpha, MiningTask, KnowledgeEntry, AlphaFailure


router = APIRouter(
    prefix="/hitl",
    tags=["human-in-the-loop"],
    responses={404: {"description": "Not found"}},
)


# =============================================================================
# Pydantic Models
# =============================================================================

class HypothesisInjectionRequest(BaseModel):
    """Request to inject a user's trading hypothesis."""
    task_id: int
    hypothesis: str = Field(..., description="Trading hypothesis statement")
    rationale: str = Field(default="", description="Economic reasoning")
    expected_signal: str = Field(default="", description="momentum | mean_reversion | value | other")
    suggested_fields: List[str] = Field(default_factory=list, description="Suggested data fields")
    priority: str = Field(default="normal", description="high | normal | low")


class AlphaFeedbackRequest(BaseModel):
    """Request to provide feedback on a generated alpha."""
    alpha_id: int
    feedback_type: str = Field(..., description="approve | reject | modify | optimize")
    feedback_text: str = Field(default="", description="Detailed feedback")
    modified_expression: Optional[str] = Field(default=None, description="Modified expression if type=modify")
    rating: Optional[int] = Field(default=None, description="1-5 rating")
    
    
class ExplorationGuidanceRequest(BaseModel):
    """Request to guide exploration direction."""
    task_id: int
    preferred_datasets: List[str] = Field(default_factory=list)
    avoid_datasets: List[str] = Field(default_factory=list)
    preferred_fields: List[str] = Field(default_factory=list)
    avoid_fields: List[str] = Field(default_factory=list)
    preferred_operators: List[str] = Field(default_factory=list)
    focus_patterns: List[str] = Field(default_factory=list)
    avoid_patterns: List[str] = Field(default_factory=list)
    notes: str = Field(default="")


class KnowledgeEntryRequest(BaseModel):
    """Request to add manual knowledge entry."""
    entry_type: str = Field(..., description="SUCCESS_PATTERN | FAILURE_PITFALL | FIELD_INSIGHT")
    pattern: str = Field(..., description="Pattern or template")
    description: str = Field(default="")
    dataset_category: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HITLResponse(BaseModel):
    """Standard HITL response."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/inject-hypothesis", response_model=HITLResponse)
async def inject_hypothesis(
    request: HypothesisInjectionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Inject a user's trading hypothesis into an active mining task.
    
    This allows humans to guide the AI towards specific trading ideas:
    - The hypothesis will be prioritized in the next mining iteration
    - The system will generate expressions to test this hypothesis
    - Feedback will be recorded for learning
    """
    # Verify task exists
    task_query = select(MiningTask).where(MiningTask.id == request.task_id)
    result = await db.execute(task_query)
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Store hypothesis as a high-priority knowledge entry
    entry = KnowledgeEntry(
        entry_type="USER_HYPOTHESIS",
        pattern=request.hypothesis,
        description=request.rationale,
        meta_data={
            "task_id": request.task_id,
            "expected_signal": request.expected_signal,
            "suggested_fields": request.suggested_fields,
            "priority": request.priority,
            "source": "human_injection",
            "injected_at": datetime.now().isoformat(),
        }
    )
    db.add(entry)
    
    # Update task to indicate human guidance
    task.metadata = task.metadata or {}
    task.metadata["has_human_guidance"] = True
    task.metadata["last_hypothesis_injection"] = datetime.now().isoformat()
    
    # Add to injection queue (will be picked up by mining agent)
    injections = task.metadata.get("hypothesis_injections", [])
    injections.append({
        "hypothesis": request.hypothesis,
        "rationale": request.rationale,
        "expected_signal": request.expected_signal,
        "suggested_fields": request.suggested_fields,
        "priority": request.priority,
        "injected_at": datetime.now().isoformat(),
    })
    task.metadata["hypothesis_injections"] = injections[-10:]  # Keep last 10
    
    await db.commit()
    
    logger.info(f"[HITL] Hypothesis injected for task {request.task_id}: {request.hypothesis[:50]}...")
    
    return HITLResponse(
        success=True,
        message=f"Hypothesis injected successfully. Will be tested in next iteration.",
        data={
            "entry_id": entry.id,
            "task_id": request.task_id,
            "priority": request.priority,
        }
    )


@router.post("/alpha-feedback", response_model=HITLResponse)
async def provide_alpha_feedback(
    request: AlphaFeedbackRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Provide feedback on a generated alpha.
    
    Feedback types:
    - approve: Mark as good, use for knowledge learning
    - reject: Mark as bad, record why
    - modify: Provide modified expression for testing
    - optimize: Request optimization with specific guidance
    """
    # Verify alpha exists
    alpha_query = select(Alpha).where(Alpha.id == request.alpha_id)
    result = await db.execute(alpha_query)
    alpha = result.scalar_one_or_none()
    
    if not alpha:
        raise HTTPException(status_code=404, detail="Alpha not found")
    
    # Update alpha with feedback
    alpha.metadata = alpha.metadata or {}
    alpha.metadata["human_feedback"] = {
        "type": request.feedback_type,
        "text": request.feedback_text,
        "rating": request.rating,
        "provided_at": datetime.now().isoformat(),
    }
    
    # Handle different feedback types
    if request.feedback_type == "approve":
        alpha.metadata["human_approved"] = True
        
        # Record as positive pattern for learning
        entry = KnowledgeEntry(
            entry_type="SUCCESS_PATTERN",
            pattern=alpha.expression,
            description=f"Human approved: {request.feedback_text}",
            meta_data={
                "alpha_id": alpha.id,
                "sharpe": alpha.is_sharpe,
                "human_approved": True,
                "source": "human_feedback",
            }
        )
        db.add(entry)
        
    elif request.feedback_type == "reject":
        alpha.metadata["human_rejected"] = True
        
        # Record as pitfall for learning
        entry = KnowledgeEntry(
            entry_type="FAILURE_PITFALL",
            pattern=alpha.expression,
            description=f"Human rejected: {request.feedback_text}",
            meta_data={
                "alpha_id": alpha.id,
                "reason": request.feedback_text,
                "human_rejected": True,
                "source": "human_feedback",
            }
        )
        db.add(entry)
        
    elif request.feedback_type == "modify" and request.modified_expression:
        # Create new alpha with modified expression
        modified_alpha = Alpha(
            task_id=alpha.task_id,
            expression=request.modified_expression,
            hypothesis=f"Human modified: {alpha.hypothesis}",
            logic_explanation=f"Modified from alpha {alpha.id}: {request.feedback_text}",
            dataset_id=alpha.dataset_id,
            region=alpha.region,
            universe=alpha.universe,
            quality_status="PENDING",
            metadata={
                "parent_alpha_id": alpha.id,
                "human_modified": True,
                "modification_reason": request.feedback_text,
            }
        )
        db.add(modified_alpha)
        
    elif request.feedback_type == "optimize":
        alpha.quality_status = "OPTIMIZE"
        alpha.metadata["optimization_guidance"] = request.feedback_text
    
    await db.commit()
    
    logger.info(f"[HITL] Feedback provided for alpha {request.alpha_id}: {request.feedback_type}")
    
    return HITLResponse(
        success=True,
        message=f"Feedback recorded: {request.feedback_type}",
        data={"alpha_id": request.alpha_id}
    )


@router.post("/exploration-guidance", response_model=HITLResponse)
async def set_exploration_guidance(
    request: ExplorationGuidanceRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Set exploration guidance for a mining task.
    
    This allows humans to:
    - Steer the system towards preferred datasets/fields
    - Avoid certain patterns that don't work
    - Focus on specific operator types
    """
    # Verify task exists
    task_query = select(MiningTask).where(MiningTask.id == request.task_id)
    result = await db.execute(task_query)
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update task metadata with guidance
    task.metadata = task.metadata or {}
    task.metadata["exploration_guidance"] = {
        "preferred_datasets": request.preferred_datasets,
        "avoid_datasets": request.avoid_datasets,
        "preferred_fields": request.preferred_fields,
        "avoid_fields": request.avoid_fields,
        "preferred_operators": request.preferred_operators,
        "focus_patterns": request.focus_patterns,
        "avoid_patterns": request.avoid_patterns,
        "notes": request.notes,
        "updated_at": datetime.now().isoformat(),
    }
    task.metadata["has_human_guidance"] = True
    
    await db.commit()
    
    logger.info(f"[HITL] Exploration guidance set for task {request.task_id}")
    
    return HITLResponse(
        success=True,
        message="Exploration guidance updated",
        data={
            "task_id": request.task_id,
            "guidance": task.metadata["exploration_guidance"]
        }
    )


@router.post("/add-knowledge", response_model=HITLResponse)
async def add_knowledge_entry(
    request: KnowledgeEntryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Manually add a knowledge entry.
    
    Allows humans to contribute:
    - Known successful patterns
    - Known pitfalls/failures
    - Field insights
    """
    if request.entry_type not in ["SUCCESS_PATTERN", "FAILURE_PITFALL", "FIELD_INSIGHT"]:
        raise HTTPException(status_code=400, detail="Invalid entry_type")
    
    entry = KnowledgeEntry(
        entry_type=request.entry_type,
        pattern=request.pattern,
        description=request.description,
        meta_data={
            **request.metadata,
            "dataset_category": request.dataset_category,
            "region": request.region,
            "source": "human_manual",
            "added_at": datetime.now().isoformat(),
        }
    )
    db.add(entry)
    await db.commit()
    
    logger.info(f"[HITL] Knowledge entry added: {request.entry_type} - {request.pattern[:50]}...")
    
    return HITLResponse(
        success=True,
        message="Knowledge entry added",
        data={"entry_id": entry.id}
    )


@router.get("/task/{task_id}/guidance")
async def get_task_guidance(
    task_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get current exploration guidance for a task."""
    task_query = select(MiningTask).where(MiningTask.id == task_id)
    result = await db.execute(task_query)
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    metadata = task.metadata or {}
    
    return {
        "task_id": task_id,
        "has_human_guidance": metadata.get("has_human_guidance", False),
        "exploration_guidance": metadata.get("exploration_guidance", {}),
        "hypothesis_injections": metadata.get("hypothesis_injections", []),
        "last_updated": metadata.get("exploration_guidance", {}).get("updated_at"),
    }


@router.get("/pending-review")
async def get_alphas_pending_review(
    task_id: Optional[int] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    Get alphas that are pending human review.
    
    Returns alphas that:
    - Have status OPTIMIZE (need guidance)
    - Were recently generated but not reviewed
    - Have high Sharpe but failed other checks
    """
    query = select(Alpha).where(
        Alpha.quality_status.in_(["OPTIMIZE", "PASS"])
    ).order_by(Alpha.created_at.desc()).limit(limit)
    
    if task_id:
        query = query.where(Alpha.task_id == task_id)
    
    result = await db.execute(query)
    alphas = result.scalars().all()
    
    # Filter to those needing review
    pending = []
    for alpha in alphas:
        metadata = alpha.metadata or {}
        if not metadata.get("human_feedback"):
            pending.append({
                "id": alpha.id,
                "expression": alpha.expression,
                "hypothesis": alpha.hypothesis,
                "status": alpha.quality_status,
                "sharpe": alpha.is_sharpe,
                "fitness": alpha.is_fitness,
                "turnover": alpha.is_turnover,
                "created_at": alpha.created_at.isoformat() if alpha.created_at else None,
            })
    
    return {
        "count": len(pending),
        "alphas": pending[:limit]
    }


@router.post("/batch-approve")
async def batch_approve_alphas(
    alpha_ids: List[int],
    db: AsyncSession = Depends(get_db)
):
    """Batch approve multiple alphas."""
    approved = 0
    for alpha_id in alpha_ids:
        query = select(Alpha).where(Alpha.id == alpha_id)
        result = await db.execute(query)
        alpha = result.scalar_one_or_none()
        
        if alpha:
            alpha.metadata = alpha.metadata or {}
            alpha.metadata["human_approved"] = True
            alpha.metadata["human_feedback"] = {
                "type": "approve",
                "provided_at": datetime.now().isoformat(),
            }
            approved += 1
    
    await db.commit()
    
    return HITLResponse(
        success=True,
        message=f"Approved {approved} alphas",
        data={"approved_count": approved}
    )
