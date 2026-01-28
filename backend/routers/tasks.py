"""
Tasks Router - Enhanced with Trace visualization and intervention support
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from backend.database import get_db
from backend.models import MiningTask, TraceStep, Alpha, ExperimentRun

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    responses={404: {"description": "Not found"}},
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class TaskCreateRequest(BaseModel):
    name: str
    region: str = "USA"
    universe: str = "TOP3000"
    dataset_strategy: str = "AUTO"  # AUTO or SPECIFIC
    target_datasets: List[str] = []
    agent_mode: str = "AUTONOMOUS"  # AUTONOMOUS or INTERACTIVE
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
    parameters: dict = {}  # For ADJUST action


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    List all mining tasks with optional status filter.
    """
    query = select(MiningTask).order_by(MiningTask.created_at.desc())
    
    if status:
        query = query.where(MiningTask.status == status)
    
    query = query.limit(limit).offset(offset)
    
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    return [TaskResponse(
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
        updated_at=t.updated_at
    ) for t in tasks]


@router.post("", response_model=TaskResponse)
async def create_task(
    request: TaskCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new mining task.
    """
    task = MiningTask(
        task_name=request.name,
        region=request.region,
        universe=request.universe,
        dataset_strategy=request.dataset_strategy,
        target_datasets=request.target_datasets,
        agent_mode=request.agent_mode,
        daily_goal=request.daily_goal,
        max_iterations=request.max_iterations,
        config=request.config,
        status="PENDING"
    )
    
    db.add(task)
    await db.commit()
    await db.refresh(task)
    
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
        updated_at=task.updated_at
    )


@router.get("/{task_id}", response_model=TaskDetailResponse)
async def get_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get task details including trace steps.
    """
    # Get task
    task_query = select(MiningTask).where(MiningTask.id == task_id)
    task_result = await db.execute(task_query)
    task = task_result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Get trace steps
    steps_query = select(TraceStep).where(
        TraceStep.task_id == task_id
    ).order_by(TraceStep.step_order)
    steps_result = await db.execute(steps_query)
    steps = steps_result.scalars().all()
    
    # Count alphas
    alphas_query = select(Alpha).where(Alpha.task_id == task_id)
    alphas_result = await db.execute(alphas_query)
    alphas_count = len(alphas_result.scalars().all())
    
    return TaskDetailResponse(
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
        trace_steps=[TraceStepResponse(
            id=s.id,
            step_type=s.step_type,
            step_order=s.step_order,
            iteration=s.iteration,
            input_data=s.input_data or {},
            output_data=s.output_data or {},
            duration_ms=s.duration_ms,
            status=s.status,
            error_message=s.error_message,
            created_at=s.created_at
        ) for s in steps],
        alphas_count=alphas_count
    )


@router.get("/{task_id}/trace", response_model=List[TraceStepResponse])
async def get_task_trace(
    task_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the complete trace (all steps) for a task.
    This is the RD-Agent style trace visualization endpoint.
    """
    # Verify task exists
    task_query = select(MiningTask).where(MiningTask.id == task_id)
    task_result = await db.execute(task_query)
    if not task_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Get all trace steps
    steps_query = select(TraceStep).where(
        TraceStep.task_id == task_id
    ).order_by(TraceStep.step_order)
    
    result = await db.execute(steps_query)
    steps = result.scalars().all()
    
    return [TraceStepResponse(
        id=s.id,
        step_type=s.step_type,
        step_order=s.step_order,
        iteration=s.iteration,
        input_data=s.input_data or {},
        output_data=s.output_data or {},
        duration_ms=s.duration_ms,
        status=s.status,
        error_message=s.error_message,
        created_at=s.created_at
    ) for s in steps]


@router.post("/{task_id}/start")
async def start_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """
    Start a mining task.
    """
    task_query = select(MiningTask).where(MiningTask.id == task_id)
    task_result = await db.execute(task_query)
    task = task_result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status not in ["PENDING", "PAUSED", "STOPPED", "FAILED", "COMPLETED"]:
        raise HTTPException(status_code=400, detail=f"Cannot start task in {task.status} status")
    
    await db.execute(
        update(MiningTask)
        .where(MiningTask.id == task_id)
        .values(status="RUNNING")
    )
    await db.commit()

    run = ExperimentRun(
        task_id=task_id,
        status="RUNNING",
        trigger_source="API",
        celery_task_id=None,
        config_snapshot={
            "task": {
                "region": task.region,
                "universe": task.universe,
                "dataset_strategy": task.dataset_strategy,
                "target_datasets": task.target_datasets,
                "daily_goal": task.daily_goal,
                "max_iterations":task.max_iterations,
                "config": task.config,
            },
        },
        strategy_snapshot={},
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    
    # Trigger actual mining via Celery
    from backend.tasks import run_mining_task
    celery_task = run_mining_task.delay(task_id, run.id)

    run.celery_task_id = celery_task.id
    await db.commit()
    
    return {"message": "Task started", "task_id": task_id, "run_id": run.id, "celery_task_id": celery_task.id}


@router.get("/{task_id}/runs", response_model=List[ExperimentRunResponse])
async def list_task_runs(task_id: int, db: AsyncSession = Depends(get_db)):
    task_query = select(MiningTask).where(MiningTask.id == task_id)
    task_result = await db.execute(task_query)
    if not task_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Task not found")

    query = select(ExperimentRun).where(ExperimentRun.task_id == task_id).order_by(ExperimentRun.started_at.desc())
    result = await db.execute(query)
    runs = result.scalars().all()
    return list(runs)


@router.post("/{task_id}/intervene")
async def intervene_task(
    task_id: int,
    request: InterventionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Human intervention endpoint - pause, resume, skip, or adjust a running task.
    This is the Human-in-the-Loop interaction point.
    """
    task_query = select(MiningTask).where(MiningTask.id == task_id)
    task_result = await db.execute(task_query)
    task = task_result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    action = request.action.upper()
    
    if action == "PAUSE":
        if task.status != "RUNNING":
            raise HTTPException(status_code=400, detail="Can only pause running tasks")
        new_status = "PAUSED"
    elif action == "RESUME":
        if task.status != "PAUSED":
            raise HTTPException(status_code=400, detail="Can only resume paused tasks")
        new_status = "RUNNING"
    elif action == "STOP":
        new_status = "STOPPED"
    elif action == "SKIP":
        # Skip current dataset - just log and continue
        # TODO: Implement skip logic in the mining loop
        return {"message": "Skip signal sent", "task_id": task_id}
    elif action == "ADJUST":
        # Update task config with new parameters
        new_config = {**task.config, **request.parameters}
        await db.execute(
            update(MiningTask)
            .where(MiningTask.id == task_id)
            .values(config=new_config)
        )
        await db.commit()
        return {"message": "Task config adjusted", "task_id": task_id, "new_config": new_config}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
    
    await db.execute(
        update(MiningTask)
        .where(MiningTask.id == task_id)
        .values(status=new_status)
    )
    await db.commit()
    
    return {"message": f"Task {action.lower()}d", "task_id": task_id, "new_status": new_status}

from celery.result import AsyncResult
from backend.celery_app import celery_app

@router.get("/celery/{task_id}/status")
async def get_celery_task_status(task_id: str):
    """
    Get status of a Celery background task by UUID.
    Used for polling sync tasks.
    """
    # AsyncResult check is sync, but fast enough.
    # We can run it directly.
    result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": result.status,
    }
    
    if result.ready():
        # If result is an exception, accessing .result might raise it, 
        # unless we check carefully or use .info (depends on configuration)
        # But usually .result is fine if we handle exception wrapping
        try:
             # If failed, result.result is the exception object
             if result.failed():
                 response["error"] = str(result.result)
             else:
                 response["result"] = result.result
        except Exception as e:
            response["error"] = str(e)
            
    return response
