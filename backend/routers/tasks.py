"""
Tasks Router - Mining task management

Uses TaskService for all business logic.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from celery.result import AsyncResult

from backend.database import get_db
from backend.services.task_service import TaskService, TaskCreateData
from backend.celery_app import celery_app

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    responses={404: {"description": "Not found"}},
)


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_task_service(db: AsyncSession = Depends(get_db)) -> TaskService:
    """Get TaskService instance with injected dependencies."""
    return TaskService(db)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class TaskCreateRequest(BaseModel):
    name: str
    region: str = "USA"
    universe: str = "TOP3000"
    dataset_strategy: str = "AUTO"
    target_datasets: List[str] = []
    agent_mode: str = "AUTONOMOUS"
    daily_goal: int = 4
    max_iterations: int = 10
    config: dict = {}


class TaskResponse(BaseModel):
    id: int
    task_name: str
    region: str
    universe: str
    dataset_strategy: str
    agent_mode: str
    status: str
    daily_goal: int
    progress_current: int
    current_iteration: int = 0
    max_iterations: int = 10
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class TraceStepResponse(BaseModel):
    id: int
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


class TaskDetailResponse(TaskResponse):
    trace_steps: List[TraceStepResponse] = []
    alphas_count: int = 0


class ExperimentRunResponse(BaseModel):
    id: int
    task_id: int
    status: str
    trigger_source: Optional[str] = None
    celery_task_id: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class InterventionRequest(BaseModel):
    action: str  # PAUSE, RESUME, SKIP, ADJUST
    parameters: dict = {}


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: TaskService = Depends(get_task_service),
):
    """List all mining tasks with optional status filter."""
    tasks = await service.list_tasks(status=status, limit=limit, offset=offset)
    
    return [
        TaskResponse(
            id=t.id,
            task_name=t.task_name,
            region=t.region,
            universe=t.universe,
            dataset_strategy=t.dataset_strategy,
            agent_mode=t.agent_mode,
            status=t.status,
            daily_goal=t.daily_goal,
            progress_current=t.progress_current,
            current_iteration=t.current_iteration,
            max_iterations=t.max_iterations,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in tasks
    ]


@router.post("", response_model=TaskResponse)
async def create_task(
    request: TaskCreateRequest,
    service: TaskService = Depends(get_task_service),
):
    """Create a new mining task."""
    data = TaskCreateData(
        name=request.name,
        region=request.region,
        universe=request.universe,
        dataset_strategy=request.dataset_strategy,
        target_datasets=request.target_datasets,
        agent_mode=request.agent_mode,
        daily_goal=request.daily_goal,
        max_iterations=request.max_iterations,
        config=request.config,
    )
    
    task = await service.create_task(data)
    
    return TaskResponse(
        id=task.id,
        task_name=task.task_name,
        region=task.region,
        universe=task.universe,
        dataset_strategy=task.dataset_strategy,
        agent_mode=task.agent_mode,
        status=task.status,
        daily_goal=task.daily_goal,
        progress_current=task.progress_current,
        current_iteration=task.current_iteration,
        max_iterations=task.max_iterations,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@router.get("/{task_id}", response_model=TaskDetailResponse)
async def get_task(
    task_id: int,
    service: TaskService = Depends(get_task_service),
):
    """Get task details including trace steps."""
    detail = await service.get_task_detail(task_id)
    
    if not detail:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskDetailResponse(
        id=detail.id,
        task_name=detail.task_name,
        region=detail.region,
        universe=detail.universe,
        dataset_strategy=detail.dataset_strategy,
        agent_mode=detail.agent_mode,
        status=detail.status,
        daily_goal=detail.daily_goal,
        progress_current=detail.progress_current,
        current_iteration=detail.current_iteration,
        max_iterations=detail.max_iterations,
        created_at=detail.created_at,
        updated_at=detail.updated_at,
        trace_steps=[
            TraceStepResponse(
                id=s.id,
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
            for s in detail.trace_steps
        ],
        alphas_count=detail.alphas_count,
    )


@router.get("/{task_id}/trace", response_model=List[TraceStepResponse])
async def get_task_trace(
    task_id: int,
    service: TaskService = Depends(get_task_service),
):
    """Get the complete trace (all steps) for a task."""
    try:
        steps = await service.get_task_trace(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return [
        TraceStepResponse(
            id=s.id,
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


@router.post("/{task_id}/start")
async def start_task(
    task_id: int,
    service: TaskService = Depends(get_task_service),
):
    """Start a mining task."""
    try:
        result = await service.start_task(task_id)
        return {
            "message": "Task started",
            "task_id": task_id,
            "run_id": result["run_id"],
            "celery_task_id": result["celery_task_id"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{task_id}/runs", response_model=List[ExperimentRunResponse])
async def list_task_runs(
    task_id: int,
    service: TaskService = Depends(get_task_service),
):
    """List all experiment runs for a task."""
    try:
        runs = await service.list_task_runs(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return [
        ExperimentRunResponse(
            id=r.id,
            task_id=r.task_id,
            status=r.status,
            trigger_source=r.trigger_source,
            celery_task_id=r.celery_task_id,
            started_at=r.started_at,
            finished_at=r.finished_at,
            error_message=r.error_message,
        )
        for r in runs
    ]


@router.post("/{task_id}/intervene")
async def intervene_task(
    task_id: int,
    request: InterventionRequest,
    service: TaskService = Depends(get_task_service),
):
    """Human intervention endpoint - pause, resume, skip, or adjust a running task."""
    try:
        result = await service.intervene_task(
            task_id=task_id,
            action=request.action,
            parameters=request.parameters,
        )
        return {
            "message": f"Task {result['action']}",
            "task_id": task_id,
            **result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/celery/{task_id}/status")
async def get_celery_task_status(task_id: str):
    """Get status of a Celery background task by UUID."""
    result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": result.status,
    }
    
    if result.ready():
        try:
            if result.failed():
                response["error"] = str(result.result)
            else:
                response["result"] = result.result
        except Exception as e:
            response["error"] = str(e)
            
    return response
