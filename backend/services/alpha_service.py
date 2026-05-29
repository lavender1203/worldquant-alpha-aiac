"""
Alpha Service - Business logic for alpha management

Provides methods for:
- Listing and filtering alphas
- Alpha details and trace retrieval
- Human feedback submission
- Statistics and aggregations
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func

from backend.services.base import BaseService
from backend.repositories import AlphaRepository
from backend.protocols.repository_protocol import PaginationParams, PaginatedResult
from backend.models import Alpha, TraceStep

logger = logging.getLogger("services.alpha")


@dataclass
class AlphaListFilters:
    """Filters for alpha listing."""
    region: Optional[str] = None
    quality_status: Optional[str] = None
    human_feedback: Optional[str] = None
    dataset_id: Optional[str] = None
    task_id: Optional[int] = None


@dataclass
class AlphaListItem:
    """Simplified alpha for list views."""
    id: int
    alpha_id: Optional[str]
    type: str
    name: Optional[str]
    expression: str
    region: Optional[str]
    dataset_id: Optional[str]
    quality_status: str
    human_feedback: str
    sharpe: Optional[float]
    returns: Optional[float]
    turnover: Optional[float]
    drawdown: Optional[float]
    margin: Optional[float]
    fitness: Optional[float]
    created_at: Optional[datetime]


@dataclass
class AlphaDetail:
    """Full alpha details."""
    id: int
    alpha_id: Optional[str]
    task_id: Optional[int]
    expression: str
    hypothesis: Optional[str]
    logic_explanation: Optional[str]
    region: Optional[str]
    universe: Optional[str]
    dataset_id: Optional[str]
    fields_used: List[str]
    operators_used: List[str]
    status: str
    quality_status: str
    human_feedback: str
    feedback_comment: Optional[str]
    metrics: Dict[str, Any]
    is_metrics: Dict[str, Any]
    os_metrics: Dict[str, Any]
    created_at: Optional[datetime]


class AlphaService(BaseService):
    """
    Service for alpha-related operations.
    
    Provides a clean interface for alpha management,
    abstracting database operations from routers.
    """
    
    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self.alpha_repo = AlphaRepository(db)
    
    # =========================================================================
    # List Operations
    # =========================================================================
    
    async def list_alphas(
        self,
        filters: AlphaListFilters,
        sort_by: str = "date_created",
        sort_order: str = "desc",
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[List[AlphaListItem], int]:
        """
        List alphas with filtering and sorting.
        
        Args:
            filters: Filter criteria
            sort_by: Column to sort by
            sort_order: 'asc' or 'desc'
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            Tuple of (items, total_count)
        """
        # Build query
        query = select(Alpha)
        count_query = select(func.count()).select_from(Alpha)
        
        # Apply filters
        if filters.region:
            query = query.where(Alpha.region == filters.region)
            count_query = count_query.where(Alpha.region == filters.region)
        
        if filters.quality_status:
            query = query.where(Alpha.quality_status == filters.quality_status)
            count_query = count_query.where(Alpha.quality_status == filters.quality_status)
        
        if filters.human_feedback:
            query = query.where(Alpha.human_feedback == filters.human_feedback)
            count_query = count_query.where(Alpha.human_feedback == filters.human_feedback)
        
        if filters.dataset_id:
            query = query.where(Alpha.dataset_id == filters.dataset_id)
            count_query = count_query.where(Alpha.dataset_id == filters.dataset_id)
        
        if filters.task_id:
            query = query.where(Alpha.task_id == filters.task_id)
            count_query = count_query.where(Alpha.task_id == filters.task_id)
        
        # Get total count
        total = (await self.db.execute(count_query)).scalar() or 0
        
        # Apply sorting
        sort_column = getattr(Alpha, sort_by, Alpha.date_created)
        if sort_order.lower() == "desc":
            query = query.order_by(sort_column.desc().nullslast())
        else:
            query = query.order_by(sort_column.asc().nullsfirst())
        
        query = query.limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        alphas = result.scalars().all()
        
        # Convert to list items
        items = [self._to_list_item(a) for a in alphas]
        
        return items, total
    
    def _to_list_item(self, alpha: Alpha) -> AlphaListItem:
        """Convert Alpha model to AlphaListItem."""
        expression = alpha.expression or "N/A"
        if len(expression) > 100:
            expression = expression[:100] + "..."
        
        margin = None
        if alpha.is_metrics and isinstance(alpha.is_metrics, dict):
            margin = alpha.is_metrics.get("margin")
        
        return AlphaListItem(
            id=alpha.id,
            alpha_id=alpha.alpha_id,
            type=alpha.type or "REGULAR",
            name=alpha.name,
            expression=expression,
            region=alpha.region,
            dataset_id=alpha.dataset_id,
            quality_status=alpha.quality_status or "PENDING",
            human_feedback=alpha.human_feedback or "NONE",
            sharpe=alpha.is_sharpe,
            returns=alpha.is_returns,
            turnover=alpha.is_turnover,
            drawdown=alpha.is_drawdown,
            margin=margin,
            fitness=alpha.is_fitness,
            created_at=alpha.date_created or alpha.created_at,
        )
    
    # =========================================================================
    # Get Operations
    # =========================================================================
    
    async def get_alpha(self, alpha_id: int) -> Optional[AlphaDetail]:
        """
        Get detailed alpha information.
        
        Args:
            alpha_id: Database ID of the alpha
            
        Returns:
            AlphaDetail or None if not found
        """
        alpha = await self.alpha_repo.get_by_id(alpha_id)
        if not alpha:
            return None
        
        return self._to_detail(alpha)
    
    async def get_alpha_by_brain_id(self, brain_alpha_id: str) -> Optional[AlphaDetail]:
        """
        Get alpha by BRAIN platform ID.
        
        Args:
            brain_alpha_id: BRAIN alpha ID string
            
        Returns:
            AlphaDetail or None if not found
        """
        alpha = await self.alpha_repo.get_by_alpha_id(brain_alpha_id)
        if not alpha:
            return None
        
        return self._to_detail(alpha)
    
    def _to_detail(self, alpha: Alpha) -> AlphaDetail:
        """Convert Alpha model to AlphaDetail."""
        return AlphaDetail(
            id=alpha.id,
            alpha_id=alpha.alpha_id,
            task_id=alpha.task_id,
            expression=alpha.expression,
            hypothesis=alpha.hypothesis,
            logic_explanation=alpha.logic_explanation,
            region=alpha.region,
            universe=alpha.universe,
            dataset_id=alpha.dataset_id,
            fields_used=alpha.fields_used or [],
            operators_used=alpha.operators_used or [],
            status=alpha.status or "created",
            quality_status=alpha.quality_status or "PENDING",
            human_feedback=alpha.human_feedback or "NONE",
            feedback_comment=alpha.feedback_comment,
            metrics=alpha.metrics or {},
            is_metrics=alpha.is_metrics or {},
            os_metrics=alpha.os_metrics or {},
            created_at=alpha.created_at,
        )
    
    # =========================================================================
    # Trace Operations
    # =========================================================================
    
    async def get_alpha_trace(self, alpha_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the trace steps that generated an alpha.
        
        Args:
            alpha_id: Database ID of the alpha
            
        Returns:
            Dict with trace context or None if not found
        """
        alpha = await self.alpha_repo.get_by_id(alpha_id)
        if not alpha:
            return None
        
        if not alpha.trace_step_id:
            return {"message": "No trace step linked to this alpha"}
        
        # Get the trace step
        step_query = select(TraceStep).where(TraceStep.id == alpha.trace_step_id)
        step_result = await self.db.execute(step_query)
        step = step_result.scalar_one_or_none()
        
        if not step:
            return {"message": "Trace step not found"}
        
        # Get all related trace steps for context
        context_query = (
            select(TraceStep)
            .where(TraceStep.task_id == step.task_id)
            .where(TraceStep.step_order <= step.step_order)
            .order_by(TraceStep.step_order)
        )
        
        context_result = await self.db.execute(context_query)
        context_steps = context_result.scalars().all()
        
        return {
            "alpha_id": alpha_id,
            "trace_step_id": step.id,
            "task_id": step.task_id,
            "context": [
                {
                    "step_type": s.step_type,
                    "step_order": s.step_order,
                    "status": s.status,
                    "input": s.input_data,
                    "output": s.output_data,
                    "duration_ms": s.duration_ms,
                }
                for s in context_steps
            ],
        }
    
    # =========================================================================
    # Feedback Operations
    # =========================================================================
    
    async def submit_feedback(
        self,
        alpha_id: int,
        rating: str,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Submit human feedback for an alpha.
        
        Args:
            alpha_id: Database ID of the alpha
            rating: 'LIKED' or 'DISLIKED'
            comment: Optional feedback comment
            
        Returns:
            True if feedback was submitted, False if alpha not found
        """
        if rating not in ["LIKED", "DISLIKED"]:
            raise ValueError("Rating must be LIKED or DISLIKED")
        
        # Check if alpha exists
        alpha = await self.alpha_repo.get_by_id(alpha_id)
        if not alpha:
            return False
        
        # Update feedback
        await self.db.execute(
            update(Alpha)
            .where(Alpha.id == alpha_id)
            .values(human_feedback=rating, feedback_comment=comment)
        )
        await self.commit()
        
        if rating == "LIKED":
            try:
                from backend.tasks import learn_from_alpha

                learn_from_alpha.delay(alpha_id)
            except Exception as e:
                import logging

                logging.getLogger("services.alpha").warning(
                    "Failed to dispatch alpha feedback learning task: %s", e
                )
        
        return True
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    async def get_task_stats(self, task_id: int) -> Dict[str, Any]:
        """
        Get statistics for alphas in a task.
        
        Args:
            task_id: The task ID
            
        Returns:
            Statistics dict
        """
        return await self.alpha_repo.get_task_stats(task_id)
    
    async def get_region_distribution(self, task_id: Optional[int] = None) -> Dict[str, int]:
        """
        Get distribution of alphas by region.
        
        Args:
            task_id: Optional task filter
            
        Returns:
            Dict of region -> count
        """
        return await self.alpha_repo.get_region_distribution(task_id)
