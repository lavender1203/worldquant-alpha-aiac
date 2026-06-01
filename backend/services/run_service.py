"""
Run Service - Business logic for experiment run management

Provides methods for:
- Run detail retrieval
- Run trace steps
- Run alphas listing
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.services.base import BaseService
from backend.repositories.task_repository import ExperimentRunRepository
from backend.repositories.alpha_repository import AlphaRepository
from backend.models import ExperimentRun, TraceStep, Alpha
from backend.services.trace_metrics import normalize_round_summary_metrics

logger = logging.getLogger("services.run")


@dataclass
class RunDetailInfo:
    """Experiment run detail information."""
    id: int
    task_id: int
    status: str
    trigger_source: Optional[str]
    celery_task_id: Optional[str]
    config_snapshot: Dict[str, Any]
    prompt_version: Optional[str]
    thresholds_version: Optional[str]
    strategy_snapshot: Dict[str, Any]
    started_at: datetime
    finished_at: Optional[datetime]
    error_message: Optional[str]


@dataclass
class TraceStepInfo:
    """Trace step information."""
    id: int
    task_id: int
    run_id: Optional[int]
    step_type: str
    step_order: int
    iteration: int
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    duration_ms: Optional[int]
    status: str
    error_message: Optional[str]
    created_at: datetime


@dataclass
class AlphaListItem:
    """Alpha list item for run alphas."""
    id: int
    alpha_id: Optional[str]
    task_id: Optional[int]
    run_id: Optional[int]
    expression: str
    region: Optional[str]
    dataset_id: Optional[str]
    quality_status: str
    metrics: Dict[str, Any]
    created_at: datetime


@dataclass
class PaginatedResult:
    """Paginated result container."""
    total: int
    items: List[Any] = field(default_factory=list)


class RunService(BaseService):
    """
    Service for experiment run operations.
    
    Provides a clean interface for run management,
    abstracting database operations from routers.
    """
    
    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self.run_repo = ExperimentRunRepository(db)
        self.alpha_repo = AlphaRepository(db)
    
    # =========================================================================
    # Get Operations
    # =========================================================================
    
    async def get_run(self, run_id: int) -> Optional[RunDetailInfo]:
        """
        Get experiment run by ID.
        
        Args:
            run_id: Run ID
            
        Returns:
            RunDetailInfo or None
        """
        run = await self.run_repo.get_by_id(run_id)
        if not run:
            return None
        
        return RunDetailInfo(
            id=run.id,
            task_id=run.task_id,
            status=run.status,
            trigger_source=run.trigger_source,
            celery_task_id=run.celery_task_id,
            config_snapshot=run.config_snapshot or {},
            prompt_version=run.prompt_version,
            thresholds_version=run.thresholds_version,
            strategy_snapshot=run.strategy_snapshot or {},
            started_at=run.started_at,
            finished_at=run.finished_at,
            error_message=run.error_message,
        )
    
    # =========================================================================
    # Trace Operations
    # =========================================================================
    
    async def get_run_trace(self, run_id: int) -> List[TraceStepInfo]:
        """
        Get all trace steps for an experiment run.
        
        Args:
            run_id: Run ID
            
        Returns:
            List of TraceStepInfo
            
        Raises:
            ValueError if run not found
        """
        run = await self.run_repo.get_by_id(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        
        query = (
            select(TraceStep)
            .where(TraceStep.run_id == run_id)
            .order_by(TraceStep.iteration, TraceStep.step_order, TraceStep.id)
        )
        result = await self.db.execute(query)
        steps = result.scalars().all()
        
        trace_steps = [
            TraceStepInfo(
                id=s.id,
                task_id=s.task_id,
                run_id=s.run_id,
                step_type=s.step_type,
                step_order=s.step_order,
                iteration=s.iteration,
                input_data=s.input_data or {},
                output_data=s.output_data or {},
                duration_ms=s.duration_ms,
                status=s.status,
                error_message=s.error_message,
                created_at=s.created_at,
            )
            for s in steps
        ]
        normalize_round_summary_metrics(trace_steps)
        return trace_steps
    
    # =========================================================================
    # Alpha Operations
    # =========================================================================
    
    async def get_run_alphas(
        self,
        run_id: int,
        quality_status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PaginatedResult:
        """
        Get alphas for an experiment run.
        
        Args:
            run_id: Run ID
            quality_status: Optional status filter
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            PaginatedResult with AlphaListItem items
            
        Raises:
            ValueError if run not found
        """
        run = await self.run_repo.get_by_id(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        
        # Build query
        query = select(Alpha).where(Alpha.run_id == run_id)
        count_query = select(func.count()).select_from(
            select(Alpha.id).where(Alpha.run_id == run_id).subquery()
        )
        
        if quality_status:
            query = query.where(Alpha.quality_status == quality_status)
            count_query = select(func.count()).select_from(
                select(Alpha.id)
                .where(Alpha.run_id == run_id)
                .where(Alpha.quality_status == quality_status)
                .subquery()
            )
        
        # Get total
        total_result = await self.db.execute(count_query)
        total = total_result.scalar_one()
        
        # Apply pagination
        query = query.order_by(Alpha.created_at.desc()).limit(limit).offset(offset)
        result = await self.db.execute(query)
        alphas = result.scalars().all()
        
        items = [
            AlphaListItem(
                id=a.id,
                alpha_id=a.alpha_id,
                task_id=a.task_id,
                run_id=a.run_id,
                expression=a.expression,
                region=a.region,
                dataset_id=a.dataset_id,
                quality_status=a.quality_status,
                metrics=a.metrics or {},
                created_at=a.created_at,
            )
            for a in alphas
        ]
        
        return PaginatedResult(total=total, items=items)
