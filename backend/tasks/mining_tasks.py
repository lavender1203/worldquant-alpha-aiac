"""
Mining Tasks - Background tasks for alpha mining

Contains the main mining task execution logic.
"""

from datetime import datetime
from sqlalchemy import select, update, desc
from loguru import logger
from celery.signals import task_failure

from backend.config import settings
from backend.celery_app import celery_app
from backend.database import AsyncSessionLocal
from backend.agents import MiningAgent
from backend.adapters.mcp_brain_adapter import MCPBrainAdapter
from backend.models import MiningTask, DatasetMetadata, Operator, DataField, ExperimentRun
from backend.tasks import run_async


@celery_app.task(bind=True, name="backend.tasks.run_mining_task")
def run_mining_task(self, task_id: int, run_id: int | None = None):
    """
    Run a complete mining task.
    Called when a task is started via API.
    
    Args:
        task_id: The mining task ID
        run_id: Optional experiment run ID
    """
    logger.info(f"Starting mining task {task_id} (run_id={run_id})")
    
    async def _run():
        async with AsyncSessionLocal() as db:
            run: ExperimentRun | None = None

            # Get task
            query = select(MiningTask).where(MiningTask.id == task_id)
            result = await db.execute(query)
            task = result.scalar_one_or_none()
            
            if not task:
                logger.error(f"Task {task_id} not found")
                return {"error": "Task not found"}
            
            # Update status to RUNNING
            await db.execute(
                update(MiningTask)
                .where(MiningTask.id == task_id)
                .values(status="RUNNING")
            )
            await db.commit()

            # Create or attach ExperimentRun
            run = await _get_or_create_run(db, task, run_id, self.request.id)

            try:
                try:
                    from backend.alpha_semantic_validator import load_operators_from_db
                    operators_loaded = await load_operators_from_db(db)
                    logger.info(f"Operator registry loaded in worker: {len(operators_loaded)} operators")
                except Exception as e:
                    logger.warning(f"Failed to load operator registry in worker: {e}")

                async with MCPBrainAdapter() as brain:
                    mining_agent = MiningAgent(db, brain)
                    
                    # Get datasets to mine
                    datasets = await _get_datasets_to_mine(db, task)
                    
                    if not datasets:
                        logger.warning(f"No datasets found for mining in {task.region}/{task.universe}")
                        await _mark_mining_task_failed(
                            db,
                            task_id=task_id,
                            run_id=run.id if run is not None else None,
                            celery_task_id=self.request.id,
                            error_message="No datasets found",
                        )
                        return {"warning": "No datasets found"}

                    # Get operators
                    operators = await _get_operators(db)
                    
                    # Mine each dataset
                    total_alphas = 0
                    target_reached = task.progress_current >= task.daily_goal
                    incomplete_reasons = []
                    for dataset_id in datasets:
                        # Check if task should continue
                        await db.refresh(task)
                        if task.status in ["STOPPED", "PAUSED"]:
                            logger.info(f"Task {task_id} {task.status}, stopping")
                            break
                        
                        if task.progress_current >= task.daily_goal:
                            logger.info(f"Task {task_id} reached goal")
                            break
                        
                        # Get fields
                        fields = await _get_dataset_fields(db, dataset_id, task.region, task.universe)
                        
                        if not fields:
                            logger.warning(f"No fields found for dataset {dataset_id}, skipping")
                            continue
                        
                        # Calculate remaining alphas needed
                        remaining_goal = task.daily_goal - task.progress_current
                        if remaining_goal <= 0:
                            logger.info(f"Task {task_id} already reached goal, stopping")
                            break
                        
                        # Run evolution loop
                        try:
                            result = await mining_agent.run_evolution_loop(
                                task=task,
                                dataset_id=dataset_id,
                                fields=fields,
                                operators=operators,
                                max_iterations=task.max_iterations or 10,
                                target_alphas=remaining_goal,
                                num_alphas_per_round=4,
                                run_id=run.id
                            )
                            
                            # Update progress
                            task.progress_current += result.get("total_success", 0)
                            await db.commit()
                            
                            total_alphas += len(result.get("all_alphas", []))
                            target_reached = task.progress_current >= task.daily_goal

                            if not result.get("target_reached"):
                                incomplete_reasons.append(
                                    (
                                        f"{dataset_id}: target not reached "
                                        f"({result.get('total_success', 0)}/{remaining_goal}) "
                                        f"after {result.get('iterations_completed', 0)}/"
                                        f"{task.max_iterations or 10} iterations"
                                    )
                                )
                                if result.get("dataset_stop_reason"):
                                    incomplete_reasons[-1] += (
                                        f"; stop_reason={result.get('dataset_stop_reason')}"
                                    )
                            
                            logger.info(
                                f"Evolution loop for {dataset_id} complete | "
                                f"iterations={result.get('iterations_completed')} "
                                f"success={result.get('total_success')}"
                            )
                            
                            if target_reached:
                                logger.info(f"Task {task_id} reached goal via evolution loop")
                                break
                                
                        except Exception as e:
                            logger.error(f"Evolution loop failed for {dataset_id}: {e}")
                            # Rollback any failed transaction before continuing
                            await db.rollback()
                            continue
                
                # Preserve user intervention state. A STOP/PAUSE can arrive
                # while a BRAIN multi-simulation is still polling; once that
                # call returns, this task must not overwrite it as COMPLETED.
                await db.refresh(task)
                if task.status in ["STOPPED", "PAUSED"]:
                    final_status = task.status
                    logger.info(f"Task {task_id} ended with intervention status {final_status}")
                else:
                    final_status, error_message = _resolve_final_status(
                        target_reached=target_reached,
                        incomplete_reasons=incomplete_reasons,
                        progress_current=task.progress_current,
                        daily_goal=task.daily_goal,
                    )
                    if final_status == "COMPLETED" and error_message:
                        logger.warning(
                            f"Task {task_id} completed all configured mining work "
                            f"without reaching target: {error_message}"
                        )

                    await db.execute(
                        update(MiningTask)
                        .where(MiningTask.id == task_id)
                        .values(status=final_status, updated_at=datetime.utcnow())
                    )

                if run is not None:
                    run.status = final_status
                    run.finished_at = datetime.utcnow()
                    if error_message:
                        run.error_message = error_message
                await db.commit()
                
                logger.info(f"Task {task_id} finished as {final_status}: {total_alphas} alphas mined")
                return {"success": True, "alphas_mined": total_alphas}
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                # Rollback any failed transaction before updating status
                await db.rollback()
                
                try:
                    await _mark_mining_task_failed(
                        db,
                        task_id=task_id,
                        run_id=run.id if run is not None else run_id,
                        celery_task_id=self.request.id,
                        error_message=str(e)[:500],
                    )
                except Exception as db_err:
                    logger.error(f"Failed to update task status: {db_err}")
                    await db.rollback()
                raise
    
    return run_async(_run())


def _resolve_final_status(
    *,
    target_reached: bool,
    incomplete_reasons: list[str],
    progress_current: int,
    daily_goal: int,
) -> tuple[str, str | None]:
    """Resolve task status after the mining workflow returns normally."""
    if target_reached:
        return "COMPLETED", None
    if incomplete_reasons:
        return "COMPLETED", ("; ".join(incomplete_reasons))[:500]
    return "FAILED", f"Target not reached: {progress_current}/{daily_goal}"[:500]


@task_failure.connect
def _sync_mining_task_failure_state(sender=None, task_id=None, exception=None, args=None, kwargs=None, **_):
    """Persist FAILED state for Celery-level failures such as hard time limits."""
    if getattr(sender, "name", None) != "backend.tasks.run_mining_task":
        return

    args = args or ()
    kwargs = kwargs or {}

    mining_task_id = kwargs.get("task_id")
    run_id = kwargs.get("run_id")
    if mining_task_id is None and len(args) >= 1:
        mining_task_id = args[0]
    if run_id is None and len(args) >= 2:
        run_id = args[1]

    if mining_task_id is None:
        logger.error(f"Mining task failure signal missing task id | celery_task_id={task_id}")
        return

    error_message = f"{type(exception).__name__}: {exception}"[:500]

    async def _mark_failed_from_signal():
        async with AsyncSessionLocal() as db:
            await _mark_mining_task_failed(
                db,
                task_id=int(mining_task_id),
                run_id=int(run_id) if run_id is not None else None,
                celery_task_id=task_id,
                error_message=error_message,
            )

    try:
        run_async(_mark_failed_from_signal())
    except Exception as err:
        logger.error(
            "Failed to sync mining task failure state from Celery signal | "
            f"task_id={mining_task_id} run_id={run_id} celery_task_id={task_id}: {err}"
        )


async def _mark_mining_task_failed(
    db,
    task_id: int,
    run_id: int | None,
    celery_task_id: str | None,
    error_message: str,
):
    """Mark a mining task/run as failed without overwriting user intervention states."""
    now = datetime.utcnow()
    error_message = (error_message or "Mining task failed")[:500]

    await db.execute(
        update(MiningTask)
        .where(MiningTask.id == task_id)
        .where(MiningTask.status == "RUNNING")
        .values(status="FAILED", updated_at=now)
    )

    run_query = update(ExperimentRun).where(ExperimentRun.status == "RUNNING")
    if run_id is not None:
        run_query = run_query.where(ExperimentRun.id == run_id)
    elif celery_task_id:
        run_query = run_query.where(ExperimentRun.celery_task_id == celery_task_id)
    else:
        run_query = run_query.where(ExperimentRun.task_id == task_id)

    await db.execute(
        run_query.values(
            status="FAILED",
            finished_at=now,
            error_message=error_message,
        )
    )
    await db.commit()


async def _get_or_create_run(db, task, run_id, celery_task_id):
    """Get or create an experiment run."""
    if run_id is not None:
        run_query = select(ExperimentRun).where(ExperimentRun.id == run_id)
        run_res = await db.execute(run_query)
        run = run_res.scalar_one_or_none()

        if run and run.task_id != task.id:
            raise ValueError(f"ExperimentRun {run_id} does not belong to task {task.id}")

        if run is None:
            run = ExperimentRun(
                id=run_id,
                task_id=task.id,
                status="RUNNING",
                trigger_source="API",
                celery_task_id=celery_task_id,
                config_snapshot=_create_config_snapshot(task),
                strategy_snapshot={},
            )
            db.add(run)
            await db.commit()
            await db.refresh(run)
        else:
            run.status = "RUNNING"
            run.trigger_source = "API"
            run.celery_task_id = celery_task_id
            run.error_message = None
            await db.commit()
    else:
        run = ExperimentRun(
            task_id=task.id,
            status="RUNNING",
            trigger_source="API",
            celery_task_id=celery_task_id,
            config_snapshot=_create_config_snapshot(task),
            strategy_snapshot={},
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)
    
    return run


def _create_config_snapshot(task):
    """Create a config snapshot for experiment run."""
    return {
        "task": {
            "region": task.region,
            "universe": task.universe,
            "dataset_strategy": task.dataset_strategy,
            "target_datasets": task.target_datasets,
            "daily_goal": task.daily_goal,
            "config": task.config,
        },
    }


async def _get_datasets_to_mine(db, task):
    """Get list of dataset IDs to mine."""
    if task.dataset_strategy == "SPECIFIC" and task.target_datasets:
        return task.target_datasets

    config = task.config or {}
    delay = config.get("delay")
    include_categories = _config_list(config, "include_categories", "target_categories")
    exclude_categories = _config_list(
        config,
        "exclude_categories",
        "avoid_categories",
        "lit_categories",
        "already_lit_categories",
    )
    exclude_categories.extend(_csv_list(settings.MINING_EXCLUDE_CATEGORIES))
    exclude_categories = sorted({c.lower() for c in exclude_categories if c})

    # Auto-explore: get top datasets by weight, honoring task-level pyramid/category filters.
    ds_query = (
        select(DatasetMetadata)
        .where(
            DatasetMetadata.region == task.region,
            DatasetMetadata.universe == task.universe,
            DatasetMetadata.is_active == True,
        )
        .order_by(
            DatasetMetadata.mining_weight.desc(),
            desc(DatasetMetadata.pyramid_multiplier),
            desc(DatasetMetadata.value_score),
            desc(DatasetMetadata.coverage),
            DatasetMetadata.alpha_count.asc(),
        )
        .limit(10)
    )
    if delay is not None:
        ds_query = ds_query.where(DatasetMetadata.delay == int(delay))
    if include_categories:
        ds_query = ds_query.where(DatasetMetadata.category.in_(include_categories))
    if exclude_categories:
        ds_query = ds_query.where(~DatasetMetadata.category.in_(exclude_categories))

    ds_result = await db.execute(ds_query)
    datasets_objs = ds_result.scalars().all()
    datasets = [d.dataset_id for d in datasets_objs]
    logger.info(
        "[MiningTask] AUTO dataset selection | task={} selected={} include_categories={} "
        "exclude_categories={} delay={}",
        task.id,
        datasets,
        include_categories,
        exclude_categories,
        delay,
    )
    return datasets


def _config_list(config, *keys):
    """Read a list-like config value from the first present key."""
    for key in keys:
        if key in config:
            return _csv_list(config.get(key))
    return []


def _csv_list(value):
    """Normalize comma-separated strings and JSON arrays to a string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


async def _get_operators(db):
    """Get operators for mining."""
    op_query = select(Operator).where(Operator.is_active == True)
    op_result = await db.execute(op_query)
    
    operators = []
    for op in op_result.scalars().all():
        operators.append({
            "name": op.name,
            "category": op.category,
            "description": op.description,
            "definition": op.definition,
            "scope": op.scope,
        })
    
    if not operators:
        # Fallback if DB is empty
        logger.warning("No operators found in DB, using basic set")
        operators = [
            {"name": "ts_rank", "category": "Time Series", "description": "Rank over time", "definition": "ts_rank(x, d)", "scope": ["REGULAR"]},
            {"name": "ts_mean", "category": "Time Series", "description": "Mean over time", "definition": "ts_mean(x, d)", "scope": ["REGULAR"]},
            {"name": "ts_std_dev", "category": "Time Series", "description": "Std Dev over time", "definition": "ts_std_dev(x, d)", "scope": ["REGULAR"]},
            {"name": "ts_corr", "category": "Time Series", "description": "Correlation", "definition": "ts_corr(x, y, d)", "scope": ["REGULAR"]},
            {"name": "ts_product", "category": "Time Series", "description": "Product over time", "definition": "ts_product(x, d)", "scope": ["REGULAR"]},
            {"name": "ts_sum", "category": "Time Series", "description": "Sum over time", "definition": "ts_sum(x, d)", "scope": ["REGULAR"]}
        ]
    
    return operators


async def _get_dataset_fields(db, dataset_id, region, universe):
    """Get fields for a dataset."""
    ds_meta_stmt = select(DatasetMetadata).where(
        DatasetMetadata.dataset_id == dataset_id,
        DatasetMetadata.region == region,
        DatasetMetadata.universe == universe
    )
    ds_meta_res = await db.execute(ds_meta_stmt)
    ds_meta = ds_meta_res.scalar_one_or_none()
    
    if not ds_meta:
        return []
    
    fields_stmt = select(DataField).where(DataField.dataset_id == ds_meta.id)
    fields_res = await db.execute(fields_stmt)
    fields_objs = fields_res.scalars().all()
    
    return [
        {
            "id": f.field_id,
            "name": f.field_name,
            "description": f.description,
            "type": f.field_type,
            "coverage": f.coverage,
            "date_coverage": f.date_coverage,
            "alpha_count": f.alpha_count,
            "user_count": f.user_count,
            "pyramid_multiplier": f.pyramid_multiplier,
            "category": f.category,
            "subcategory": f.subcategory,
        }
        for f in fields_objs
    ]
