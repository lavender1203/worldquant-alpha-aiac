"""
Runs Router - Experiment run management

Uses RunService for all business logic.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from backend.database import get_db
from backend.services.run_service import RunService

router = APIRouter(
    prefix="/runs",
    tags=["runs"],
    responses={404: {"description": "Not found"}},
)


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_run_service(db: AsyncSession = Depends(get_db)) -> RunService:
    """Get RunService instance with injected dependencies."""
    return RunService(db)


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class ExperimentRunDetailResponse(BaseModel):
    id: int
    task_id: int
    status: str
    trigger_source: Optional[str] = None
    celery_task_id: Optional[str] = None
    config_snapshot: dict = {}
    prompt_version: Optional[str] = None
    thresholds_version: Optional[str] = None
    strategy_snapshot: dict = {}
    started_at: datetime
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class TraceStepResponse(BaseModel):
    id: int
    task_id: int
    run_id: Optional[int] = None
    step_type: str
    step_order: int
    iteration: int = 1
    input_data: dict
    output_data: dict
    duration_ms: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AlphaListItem(BaseModel):
    id: int
    alpha_id: Optional[str] = None
    task_id: Optional[int] = None
    run_id: Optional[int] = None
    expression: str
    region: Optional[str] = None
    dataset_id: Optional[str] = None
    quality_status: str
    metrics: dict = {}
    created_at: datetime

    class Config:
        from_attributes = True


class AlphaListResponse(BaseModel):
    items: List[AlphaListItem]
    total: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/{run_id}", response_model=ExperimentRunDetailResponse)
async def get_run(
    run_id: int,
    service: RunService = Depends(get_run_service),
):
    """Get experiment run details."""
    run = await service.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return ExperimentRunDetailResponse(
        id=run.id,
        task_id=run.task_id,
        status=run.status,
        trigger_source=run.trigger_source,
        celery_task_id=run.celery_task_id,
        config_snapshot=run.config_snapshot,
        prompt_version=run.prompt_version,
        thresholds_version=run.thresholds_version,
        strategy_snapshot=run.strategy_snapshot,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error_message=run.error_message,
    )


@router.get("/{run_id}/trace", response_model=List[TraceStepResponse])
async def get_run_trace(
    run_id: int,
    service: RunService = Depends(get_run_service),
):
    """Get trace steps for an experiment run."""
    try:
        steps = await service.get_run_trace(run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return [
        TraceStepResponse(
            id=s.id,
            task_id=s.task_id,
            run_id=s.run_id,
            step_type=s.step_type,
            step_order=s.step_order,
            iteration=s.iteration,
            input_data=s.input_data,
            output_data=s.output_data,
            duration_ms=s.duration_ms,
            status=s.status,
            error_message=s.error_message,
            created_at=s.created_at,
        )
        for s in steps
    ]


@router.get("/{run_id}/alphas", response_model=AlphaListResponse)
async def get_run_alphas(
    run_id: int,
    quality_status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: RunService = Depends(get_run_service),
):
    """Get alphas for an experiment run."""
    try:
        result = await service.get_run_alphas(
            run_id=run_id,
            quality_status=quality_status,
            limit=limit,
            offset=offset,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
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
            metrics=a.metrics,
            created_at=a.created_at,
        )
        for a in result.items
    ]
    
    return AlphaListResponse(items=items, total=result.total)


# =============================================================================
# P2 FIX: EXPERIMENT VERSIONING ENDPOINTS
# =============================================================================

class RunHistoryResponse(BaseModel):
    task_id: int
    runs: List[ExperimentRunDetailResponse]
    total: int


class RunComparisonResponse(BaseModel):
    run1_id: int
    run2_id: int
    run1_stats: dict
    run2_stats: dict
    improvement: dict
    config_diff: dict
    winner: int


@router.get("/task/{task_id}/history", response_model=RunHistoryResponse)
async def get_task_run_history(
    task_id: int,
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    service: RunService = Depends(get_run_service),
):
    """
    P2 FIX: Get run history for a task.
    
    Returns all experiment runs for a task, ordered by most recent.
    Useful for tracking experiment evolution and comparing versions.
    """
    runs = await service.get_task_runs(
        task_id=task_id,
        limit=limit,
        status=status,
    )
    
    return RunHistoryResponse(
        task_id=task_id,
        runs=[
            ExperimentRunDetailResponse(
                id=r.id,
                task_id=r.task_id,
                status=r.status,
                trigger_source=r.trigger_source,
                celery_task_id=r.celery_task_id,
                config_snapshot=r.config_snapshot,
                prompt_version=r.prompt_version,
                thresholds_version=r.thresholds_version,
                strategy_snapshot=r.strategy_snapshot,
                started_at=r.started_at,
                finished_at=r.finished_at,
                error_message=r.error_message,
            )
            for r in runs
        ],
        total=len(runs),
    )


@router.get("/compare/{run_id_1}/{run_id_2}", response_model=RunComparisonResponse)
async def compare_runs(
    run_id_1: int,
    run_id_2: int,
    service: RunService = Depends(get_run_service),
):
    """
    P2 FIX: Compare two experiment runs.
    
    Returns detailed comparison including:
    - Alpha metrics (success rate, avg sharpe, avg fitness)
    - Configuration differences
    - Improvement metrics
    - Overall winner
    """
    try:
        comparison = await service.compare_runs(run_id_1, run_id_2)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return RunComparisonResponse(**comparison)
