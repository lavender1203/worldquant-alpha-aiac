"""
Run Service - Business logic for experiment run management

P2 FIX: Enhanced with experiment versioning capabilities

Provides methods for:
- Run detail retrieval
- Run trace steps
- Run alphas listing
- Experiment version management
- Run comparison
- Configuration tracking
"""

import logging
import hashlib
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from backend.services.base import BaseService
from backend.repositories.task_repository import ExperimentRunRepository
from backend.repositories.alpha_repository import AlphaRepository
from backend.models import ExperimentRun, TraceStep, Alpha

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
            .order_by(TraceStep.step_order)
        )
        result = await self.db.execute(query)
        steps = result.scalars().all()
        
        return [
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
    
    # =========================================================================
    # P2 FIX: Experiment Versioning Operations
    # =========================================================================
    
    async def create_run(
        self,
        task_id: int,
        trigger_source: str = "API",
        config: Dict[str, Any] = None,
        prompt_version: str = None,
        thresholds_version: str = None,
        strategy: Dict[str, Any] = None,
    ) -> ExperimentRun:
        """
        Create a new experiment run with configuration snapshot.
        
        P2 FIX: Enhanced for reproducibility tracking.
        
        Args:
            task_id: ID of the mining task
            trigger_source: How the run was triggered (API, SCHEDULER, etc.)
            config: Configuration to snapshot
            prompt_version: Version hash of prompts
            thresholds_version: Version hash of quality thresholds
            strategy: Initial strategy configuration
        
        Returns:
            Created ExperimentRun
        """
        # Generate version hashes if not provided
        if config and not prompt_version:
            prompt_hash = hashlib.md5(
                json.dumps(config.get("prompts", {}), sort_keys=True).encode()
            ).hexdigest()[:8]
            prompt_version = f"p_{prompt_hash}"
        
        if config and not thresholds_version:
            thresholds_hash = hashlib.md5(
                json.dumps(config.get("thresholds", {}), sort_keys=True).encode()
            ).hexdigest()[:8]
            thresholds_version = f"t_{thresholds_hash}"
        
        run = ExperimentRun(
            task_id=task_id,
            status="RUNNING",
            trigger_source=trigger_source,
            config_snapshot=config or {},
            prompt_version=prompt_version,
            thresholds_version=thresholds_version,
            strategy_snapshot=strategy or {},
            started_at=datetime.now(),
        )
        
        self.db.add(run)
        await self.db.commit()
        await self.db.refresh(run)
        
        logger.info(
            f"[RunService] Created run {run.id} | task={task_id} "
            f"prompt={prompt_version} thresholds={thresholds_version}"
        )
        
        return run
    
    async def complete_run(
        self,
        run_id: int,
        status: str = "COMPLETED",
        error_message: str = None,
    ) -> Optional[ExperimentRun]:
        """
        Mark an experiment run as complete.
        
        Args:
            run_id: Run ID
            status: Final status (COMPLETED, FAILED, CANCELLED)
            error_message: Error message if failed
        
        Returns:
            Updated ExperimentRun or None
        """
        run = await self.run_repo.get_by_id(run_id)
        if not run:
            return None
        
        run.status = status
        run.finished_at = datetime.now()
        if error_message:
            run.error_message = error_message
        
        await self.db.commit()
        
        logger.info(
            f"[RunService] Completed run {run_id} | status={status} "
            f"duration={(run.finished_at - run.started_at).total_seconds():.1f}s"
        )
        
        return run
    
    async def get_task_runs(
        self,
        task_id: int,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> List[RunDetailInfo]:
        """
        Get all runs for a task, ordered by most recent first.
        
        P2 FIX: Added for experiment history tracking.
        
        Args:
            task_id: Task ID
            limit: Maximum results
            status: Optional status filter
        
        Returns:
            List of RunDetailInfo
        """
        query = (
            select(ExperimentRun)
            .where(ExperimentRun.task_id == task_id)
            .order_by(desc(ExperimentRun.started_at))
            .limit(limit)
        )
        
        if status:
            query = query.where(ExperimentRun.status == status)
        
        result = await self.db.execute(query)
        runs = result.scalars().all()
        
        return [
            RunDetailInfo(
                id=r.id,
                task_id=r.task_id,
                status=r.status,
                trigger_source=r.trigger_source,
                celery_task_id=r.celery_task_id,
                config_snapshot=r.config_snapshot or {},
                prompt_version=r.prompt_version,
                thresholds_version=r.thresholds_version,
                strategy_snapshot=r.strategy_snapshot or {},
                started_at=r.started_at,
                finished_at=r.finished_at,
                error_message=r.error_message,
            )
            for r in runs
        ]
    
    async def compare_runs(
        self,
        run_id_1: int,
        run_id_2: int,
    ) -> Dict[str, Any]:
        """
        Compare two experiment runs.
        
        P2 FIX: Added for A/B analysis and version comparison.
        
        Args:
            run_id_1: First run ID
            run_id_2: Second run ID
        
        Returns:
            Comparison dict with metrics diff and config diff
        """
        run1 = await self.run_repo.get_by_id(run_id_1)
        run2 = await self.run_repo.get_by_id(run_id_2)
        
        if not run1 or not run2:
            raise ValueError("One or both runs not found")
        
        # Get alpha counts and metrics for each run
        async def get_run_stats(run_id: int) -> Dict:
            # Total alphas
            count_query = select(func.count()).select_from(
                select(Alpha.id).where(Alpha.run_id == run_id).subquery()
            )
            total = (await self.db.execute(count_query)).scalar_one()
            
            # Passed alphas
            passed_query = select(func.count()).select_from(
                select(Alpha.id)
                .where(Alpha.run_id == run_id)
                .where(Alpha.quality_status == "PASS")
                .subquery()
            )
            passed = (await self.db.execute(passed_query)).scalar_one()
            
            # Average metrics for passed alphas
            metric_query = (
                select(
                    func.avg(Alpha.is_sharpe).label("avg_sharpe"),
                    func.avg(Alpha.is_fitness).label("avg_fitness"),
                    func.avg(Alpha.is_turnover).label("avg_turnover"),
                )
                .where(Alpha.run_id == run_id)
                .where(Alpha.quality_status == "PASS")
            )
            metrics = (await self.db.execute(metric_query)).first()
            
            return {
                "total_alphas": total,
                "passed_alphas": passed,
                "success_rate": passed / total if total > 0 else 0,
                "avg_sharpe": float(metrics.avg_sharpe or 0) if metrics else 0,
                "avg_fitness": float(metrics.avg_fitness or 0) if metrics else 0,
                "avg_turnover": float(metrics.avg_turnover or 0) if metrics else 0,
            }
        
        stats1 = await get_run_stats(run_id_1)
        stats2 = await get_run_stats(run_id_2)
        
        # Compare configs
        config_diff = {
            "prompt_version": {
                "run1": run1.prompt_version,
                "run2": run2.prompt_version,
                "changed": run1.prompt_version != run2.prompt_version,
            },
            "thresholds_version": {
                "run1": run1.thresholds_version,
                "run2": run2.thresholds_version,
                "changed": run1.thresholds_version != run2.thresholds_version,
            },
        }
        
        return {
            "run1_id": run_id_1,
            "run2_id": run_id_2,
            "run1_stats": stats1,
            "run2_stats": stats2,
            "improvement": {
                "success_rate": stats2["success_rate"] - stats1["success_rate"],
                "avg_sharpe": stats2["avg_sharpe"] - stats1["avg_sharpe"],
                "avg_fitness": stats2["avg_fitness"] - stats1["avg_fitness"],
            },
            "config_diff": config_diff,
            "winner": run_id_2 if stats2["avg_sharpe"] > stats1["avg_sharpe"] else run_id_1,
        }
    
    async def get_latest_run(self, task_id: int) -> Optional[ExperimentRun]:
        """
        Get the most recent run for a task.
        
        Args:
            task_id: Task ID
        
        Returns:
            Most recent ExperimentRun or None
        """
        query = (
            select(ExperimentRun)
            .where(ExperimentRun.task_id == task_id)
            .order_by(desc(ExperimentRun.started_at))
            .limit(1)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
