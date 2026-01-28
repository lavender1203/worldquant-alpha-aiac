"""
Celery Tasks - Background mining and feedback jobs
"""

import asyncio
from datetime import datetime
from backend.celery_app import celery_app
from backend.database import AsyncSessionLocal
from backend.agents import MiningAgent, FeedbackAgent
from backend.adapters.brain_adapter import BrainAdapter
from backend.models import MiningTask, DatasetMetadata, Operator, DataField, ExperimentRun
from sqlalchemy import select, update, func
from loguru import logger


def run_async(coro):
    """Helper to run async functions in Celery."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="backend.tasks.run_mining_task")
def run_mining_task(self, task_id: int, run_id: int | None = None):
    """
    Run a complete mining task.
    Called when a task is started via API.
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

            # Create or attach ExperimentRun (Run-level artifact for reproducibility)
            if run_id is not None:
                run_query = select(ExperimentRun).where(ExperimentRun.id == run_id)
                run_res = await db.execute(run_query)
                run = run_res.scalar_one_or_none()

                if run and run.task_id != task_id:
                    raise ValueError(f"ExperimentRun {run_id} does not belong to task {task_id}")

                if run is None:
                    run = ExperimentRun(
                        id=run_id,
                        task_id=task_id,
                        status="RUNNING",
                        trigger_source="API",
                        celery_task_id=self.request.id,
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
                else:
                    run.status = "RUNNING"
                    run.trigger_source = "API"
                    run.celery_task_id = self.request.id
                    run.error_message = None
                    await db.commit()
            else:
                run = ExperimentRun(
                    task_id=task_id,
                    status="RUNNING",
                    trigger_source="API",
                    celery_task_id=self.request.id,
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

            try:
                async with BrainAdapter() as brain:
                    mining_agent = MiningAgent(db, brain)
                    
                    # Get datasets to mine (LOCAL DB)
                    if task.dataset_strategy == "SPECIFIC" and task.target_datasets:
                        datasets = task.target_datasets
                    else:
                        # Auto-explore: get top datasets by weight from LOCAL DB
                        ds_query = select(DatasetMetadata).where(
                            DatasetMetadata.region == task.region,
                            DatasetMetadata.universe == task.universe
                        ).order_by(DatasetMetadata.mining_weight.desc()).limit(10)
                        ds_result = await db.execute(ds_query)
                        datasets_objs = ds_result.scalars().all()
                        datasets = [d.dataset_id for d in datasets_objs]
                    
                    if not datasets:
                        logger.warning(f"No datasets found for mining in {task.region}/{task.universe}")
                        if run is not None:
                            run.status = "FAILED"
                            run.finished_at = datetime.utcnow()
                            run.error_message = "No datasets found"
                            await db.commit()
                        return {"warning": "No datasets found"}

                    # Get operators from LOCAL DB
                    op_query = select(Operator).where(Operator.is_active == True)
                    op_result = await db.execute(op_query)
                    # Convert to list of dicts (rich metadata) as expected by MiningAgent
                    operators = []
                    for op in op_result.scalars().all():
                        operators.append({
                            "name": op.name,
                            "category": op.category,
                            "description": op.description,
                            "definition": op.definition
                        })
                    
                    if not operators:
                        # Fallback if DB is empty - use basic dicts
                        logger.warning("No operators found in DB, using basic set")
                        operators = [
                            {"name": "ts_rank", "category": "Time Series", "description": "Rank over time", "definition": "ts_rank(x, d)"},
                            {"name": "ts_mean", "category": "Time Series", "description": "Mean over time", "definition": "ts_mean(x, d)"}, 
                            {"name": "ts_std_dev", "category": "Time Series", "description": "Std Dev over time", "definition": "ts_std_dev(x, d)"},
                            {"name": "ts_corr", "category": "Time Series", "description": "Correlation", "definition": "ts_corr(x, y, d)"},
                            {"name": "ts_product", "category": "Time Series", "description": "Product over time", "definition": "ts_product(x, d)"},
                            {"name": "ts_sum", "category": "Time Series", "description": "Sum over time", "definition": "ts_sum(x, d)"}
                        ]
                    
                    # Mine each dataset
                    total_alphas = 0
                    for dataset_id in datasets:
                        # Check if task should continue
                        await db.refresh(task)
                        if task.status in ["STOPPED", "PAUSED"]:
                            logger.info(f"Task {task_id} {task.status}, stopping")
                            break
                        
                        if task.progress_current >= task.daily_goal:
                            logger.info(f"Task {task_id} reached goal")
                            break
                        
                        # Get fields from LOCAL DB
                        # 1. Get Dataset PK
                        ds_meta_stmt = select(DatasetMetadata).where(
                            DatasetMetadata.dataset_id == dataset_id,
                            DatasetMetadata.region == task.region,
                            DatasetMetadata.universe == task.universe
                        )
                        ds_meta_res = await db.execute(ds_meta_stmt)
                        ds_meta = ds_meta_res.scalar_one_or_none()
                        
                        fields = []
                        if ds_meta:
                            # 2. Get Fields linked to this dataset
                            fields_stmt = select(DataField).where(
                                DataField.dataset_id == ds_meta.id
                            )
                            fields_res = await db.execute(fields_stmt)
                            fields_objs = fields_res.scalars().all()
                            
                            # Convert to dict format expected by MiningAgent
                            fields = [
                                {
                                    "id": f.field_id,
                                    "name": f.field_name,
                                    "description": f.description,
                                    "type": f.field_type
                                }
                                for f in fields_objs
                            ]
                        
                        if not fields:
                            logger.warning(f"No fields found for dataset {dataset_id}, skipping")
                            continue
                        
                        # Calculate remaining alphas needed
                        remaining_goal = task.daily_goal - task.progress_current
                        if remaining_goal <= 0:
                            logger.info(f"Task {task_id} already reached goal, stopping")
                            break
                        
                        # Run evolution loop (multi-round iteration)
                        try:
                            result = await mining_agent.run_evolution_loop(
                                task=task,
                                dataset_id=dataset_id,
                                fields=fields,
                                operators=operators,
                                max_iterations=task.max_iterations,  # Configurable: max rounds per dataset
                                target_alphas=remaining_goal,
                                num_alphas_per_round=4,
                                run_id=run.id
                            )
                            
                            # Update progress with actual successes
                            task.progress_current += result.get("total_success", 0)
                            await db.commit()
                            
                            total_alphas += len(result.get("all_alphas", []))
                            
                            logger.info(
                                f"Evolution loop for {dataset_id} complete | "
                                f"iterations={result.get('iterations_completed')} "
                                f"success={result.get('total_success')}"
                            )
                            
                            # Check if goal reached after this dataset
                            if result.get("target_reached"):
                                logger.info(f"Task {task_id} reached goal via evolution loop")
                                break
                                
                        except Exception as e:
                             logger.error(f"Evolution loop failed for {dataset_id}: {e}")
                             continue
                
                # Mark task complete
                await db.execute(
                    update(MiningTask)
                    .where(MiningTask.id == task_id)
                    .values(status="COMPLETED")
                )

                if run is not None:
                    run.status = "COMPLETED"
                    run.finished_at = datetime.utcnow()
                await db.commit()
                
                logger.info(f"Task {task_id} completed: {total_alphas} alphas mined")
                return {"success": True, "alphas_mined": total_alphas}
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                await db.execute(
                    update(MiningTask)
                    .where(MiningTask.id == task_id)
                    .values(status="FAILED")
                )

                if run is not None:
                    run.status = "FAILED"
                    run.finished_at = datetime.utcnow()
                    run.error_message = str(e)
                await db.commit()
                raise
    
    return run_async(_run())


@celery_app.task(name="backend.tasks.run_daily_feedback")
def run_daily_feedback():
    """Run daily feedback analysis (scheduled)."""
    logger.info("Running daily feedback analysis...")
    
    async def _run():
        async with AsyncSessionLocal() as db:
            feedback_agent = FeedbackAgent(db)
            result = await feedback_agent.run_daily_feedback()
            logger.info(f"Feedback analysis complete: {result}")
            return result
    
    return run_async(_run())


@celery_app.task(name="backend.tasks.update_operator_stats")
def update_operator_stats():
    """Update operator usage statistics (scheduled)."""
    logger.info("Updating operator stats...")
    
    async def _run():
        async with AsyncSessionLocal() as db:
            feedback_agent = FeedbackAgent(db)
            result = await feedback_agent.update_operator_stats()
            logger.info(f"Operator stats updated: {len(result)} operators")
            return {"operators_updated": len(result)}
    
    return run_async(_run())


@celery_app.task(name="backend.tasks.sync_datasets")
def sync_datasets():
    """Sync dataset metadata from BRAIN (scheduled)."""
    logger.info("Syncing datasets from BRAIN...")
    
    async def _run():
        async with AsyncSessionLocal() as db:
            async with BrainAdapter() as brain:
                regions = ["USA", "CHN", "ASI", "EUR"]
                total = 0
                
                for region in regions:
                    datasets = await brain.get_datasets(region=region)
                    
                    for ds in datasets:
                        existing = await db.execute(
                            select(DatasetMetadata).where(
                                DatasetMetadata.dataset_id == ds.get("id"),
                                DatasetMetadata.region == region
                            )
                        )
                        
                        if not existing.scalar_one_or_none():
                            category = ds.get("category")
                            if isinstance(category, dict):
                                category = category.get("id")
                                
                            subcategory = ds.get("subcategory")
                            if isinstance(subcategory, dict):
                                subcategory = subcategory.get("id")

                            metadata = DatasetMetadata(
                                dataset_id=ds.get("id"),
                                region=region,
                                category=category,
                                subcategory=subcategory,
                                description=ds.get("description"),
                                field_count=ds.get("fieldCount", 0)
                            )
                            db.add(metadata)
                            total += 1
                
                await db.commit()
                logger.info(f"Synced {total} new datasets")
                return {"new_datasets": total}
    
    return run_async(_run())


@celery_app.task(name="backend.tasks.learn_from_alpha")
def learn_from_alpha(alpha_id: int):
    """Learn from a successful/liked alpha."""
    async def _run():
        async with AsyncSessionLocal() as db:
            from backend.models import Alpha
            
            query = select(Alpha).where(Alpha.id == alpha_id)
            result = await db.execute(query)
            alpha = result.scalar_one_or_none()
            
            if not alpha:
                return {"error": "Alpha not found"}
            
            feedback_agent = FeedbackAgent(db)
            result = await feedback_agent.learn_from_success(alpha)
            return result
    
    return run_async(_run())



@celery_app.task(name="backend.tasks.sync_datasets_from_brain")
def sync_datasets_from_brain(region: str = "USA", universe: str = "TOP3000"):
    """
    Sync datasets for a specific region (Manual Trigger).
    """
    logger.info(f"Syncing datasets for region={region} universe={universe}...")
    
    async def _run():
        async with AsyncSessionLocal() as db:
            async with BrainAdapter() as brain:
                datasets = await brain.get_datasets(region=region, universe=universe)
                count = 0
                updated = 0
                
                for ds in datasets:
                    stmt = select(DatasetMetadata).where(
                        DatasetMetadata.dataset_id == ds.get("id"),
                        DatasetMetadata.region == region,
                        DatasetMetadata.universe == universe
                    )
                    result = await db.execute(stmt)
                    existing = result.scalar_one_or_none()
                    
                    category = ds.get("category")
                    if isinstance(category, dict):
                        category = category.get("id")
                        
                    subcategory = ds.get("subcategory")
                    if isinstance(subcategory, dict):
                        subcategory = subcategory.get("id")
                        
                    if existing:
                        existing.description = ds.get("description")
                        existing.category = category
                        existing.subcategory = subcategory
                        existing.field_count = ds.get("fieldCount", 0)
                        existing.last_synced_at = func.now()
                        # New fields
                        existing.date_coverage = ds.get("dateCoverage")
                        existing.themes = ds.get("themes")
                        existing.resources = ds.get("researchPapers")
                        existing.value_score = ds.get("valueScore")
                        existing.alpha_count = ds.get("alphaCount")
                        existing.pyramid_multiplier = ds.get("pyramidMultiplier")
                        existing.coverage = ds.get("coverage")
                        
                        updated += 1
                    else:
                        metadata = DatasetMetadata(
                            dataset_id=ds.get("id"),
                            region=region,
                            universe=universe,
                            category=category,
                            subcategory=subcategory,
                            description=ds.get("description"),
                            field_count=ds.get("fieldCount", 0),
                            # New fields
                            date_coverage=ds.get("dateCoverage"),
                            themes=ds.get("themes"),
                            resources=ds.get("researchPapers"),
                            value_score=ds.get("valueScore"),
                            alpha_count=ds.get("alphaCount"),
                            pyramid_multiplier=ds.get("pyramidMultiplier"),
                            coverage=ds.get("coverage")
                        )
                        db.add(metadata)
                        count += 1
                
                await db.commit()
                
                # Auto-trigger field sync for ALL datasets found
                logger.info(f"Auto-triggering field sync for {len(datasets)} datasets...")
                for ds in datasets:
                    sync_fields_from_brain.delay(
                        dataset_id=ds.get("id"),
                        region=region,
                        universe=universe, 
                        delay=1
                    )
                
                logger.info(f"Sync complete: {count} new, {updated} updated. Field syncs queued.")
                return {"new": count, "updated": updated, "field_syncs_queued": len(datasets)}
    
    return run_async(_run())

@celery_app.task(name="backend.tasks.sync_operators_from_brain")
def sync_operators_from_brain():
    """Sync operators from BRAIN platform."""
    logger.info("Syncing operators from BRAIN...")
    
    async def _run():
        async with AsyncSessionLocal() as db:
            async with BrainAdapter() as brain:
                # Get full operator details
                ops_data = await brain.get_operators(detailed=True)
                
                # If fallback returned strings, we can't do much detailed sync
                if ops_data and isinstance(ops_data[0], str):
                    logger.warning("Operator sync got simple list, skipping detailed update")
                    return {"updated": 0}
                
                count = 0
                updated = 0
                
                for op_data in ops_data:
                    name = op_data.get("name")
                    if not name:
                        continue
                        
                    stmt = select(Operator).where(Operator.name == name)
                    result = await db.execute(stmt)
                    existing = result.scalar_one_or_none()
                    
                    if existing:
                        existing.description = op_data.get("description")
                        existing.category = op_data.get("category")
                        existing.definition = op_data.get("definition")
                        existing.level = op_data.get("level")
                        existing.scope = op_data.get("scope")
                        # existing.param_count = len(op_data.get("parameters", []))
                        updated += 1
                    else:
                        new_op = Operator(
                            name=name,
                            description=op_data.get("description"),
                            category=op_data.get("category"),
                            definition=op_data.get("definition"),
                            level=op_data.get("level"),
                            scope=op_data.get("scope"),
                            # param_count=len(op_data.get("parameters", []))
                        )
                        db.add(new_op)
                        count += 1
                
                await db.commit()
                logger.info(f"Operator sync complete: {count} new, {updated} updated")
                return {"new": count, "updated": updated}
    
    return run_async(_run())

@celery_app.task(name="backend.tasks.sync_fields_from_brain")
def sync_fields_from_brain(dataset_id: str, region: str = "USA", universe: str = "TOP3000", delay: int = 1):
    """
    Sync fields for a specific dataset from BRAIN.
    """
    logger.info(f"Syncing fields for {dataset_id}...")
    
    async def _run():
        async with AsyncSessionLocal() as db:
            from backend.models import DataField
            
            # Resolve dataset PK
            stmt_ds = select(DatasetMetadata).where(
                DatasetMetadata.dataset_id == dataset_id,
                DatasetMetadata.region == region,
                DatasetMetadata.universe == universe
            )
            res_ds = await db.execute(stmt_ds)
            dataset = res_ds.scalar_one_or_none()
            
            if not dataset:
                logger.error(f"Dataset {dataset_id} not found for region {region}")
                return {"error": "Dataset not found"}
            
            async with BrainAdapter() as brain:
                fields = await brain.get_datafields(
                    dataset_id=dataset_id,
                    region=region,
                    universe=universe,
                    delay=delay
                )
                
                count = 0
                updated = 0
                
                for f_data in fields:
                    fid = f_data.get("id")
                    if not fid:
                        continue
                        
                    # Check existence
                    stmt = select(DataField).where(
                        DataField.dataset_id == dataset.id,
                        DataField.field_id == fid
                    )
                    result = await db.execute(stmt)
                    existing = result.scalar_one_or_none()
                    
                    if existing:
                        existing.description = f_data.get("description")
                        existing.field_name = f_data.get("name", fid)
                        # New fields
                        existing.field_type = f_data.get("type")
                        existing.date_coverage = f_data.get("dateCoverage")
                        existing.coverage = f_data.get("coverage")
                        existing.pyramid_multiplier = f_data.get("pyramidMultiplier")
                        existing.alpha_count = f_data.get("alphaCount", 0)
                        
                        updated += 1
                    else:
                        new_field = DataField(
                            dataset_id=dataset.id,
                            region=region,
                            universe=universe,
                            delay=delay,
                            field_id=fid,
                            field_name=f_data.get("name", fid),
                            description=f_data.get("description"),
                            # New fields
                            field_type=f_data.get("type"),
                            date_coverage=f_data.get("dateCoverage"),
                            coverage=f_data.get("coverage"),
                            pyramid_multiplier=f_data.get("pyramidMultiplier"),
                            alpha_count=f_data.get("alphaCount", 0)
                        )
                        db.add(new_field)
                        count += 1
                
                # Update dataset field count
                dataset.field_count = (dataset.field_count or 0) + count
                dataset.last_synced_at = func.now()
                
                await db.commit()
                logger.info(f"Field sync for {dataset_id}: {count} new, {updated} updated")
                return {"new": count, "updated": updated}
    
    return run_async(_run())


@celery_app.task(name="backend.tasks.sync_user_alphas")
def sync_user_alphas():
    """Sync all user alphas (IS and OS) from Brain."""
    logger.info("Syncing user alphas from Brain...")
    
    async def _run():
        async with AsyncSessionLocal() as db:
            from backend.models import Alpha
            
            async with BrainAdapter() as brain:
                count = 0
                updated = 0
                
                # Sync stages: IS first, then OS (submitted)
                stages = ["IS", "OS"]
                
                
                # Check for latest created timestamp (Incremental Sync)
                # We use date_created because Brain API 'startDate' usually filters on creation date.
                stmt_latest = select(func.max(Alpha.date_created))
                result_latest = await db.execute(stmt_latest)
                latest_date = result_latest.scalar_one_or_none()
                
                start_date_iso = None
                MIN_START_DATE = datetime(2025, 7, 5) # User requested hard floor

                if latest_date:
                    # Timezone Safety: Subtract 3 days to handle timezone diffs and ensuring overlap
                    from datetime import timedelta
                    safe_start = latest_date - timedelta(days=3)
                    
                    # Enforce floor
                    if safe_start < MIN_START_DATE:
                        safe_start = MIN_START_DATE
                    
                    start_date_iso = safe_start.strftime("%Y-%m-%d")
                    logger.info(f"Incremental Sync: Fetching alphas created since {start_date_iso} (buffer applied)")
                else:
                    start_date_iso = MIN_START_DATE.strftime("%Y-%m-%d")
                    logger.info(f"Full Sync (Filtered): Fetching all alphas since {start_date_iso}")

                for stage in stages:
                    # Pagination loop
                    offset = 0
                    limit = 100
                    while True:
                        alphas_data = await brain.get_user_alphas(
                            limit=limit, 
                            offset=offset, 
                            stage=stage,
                            start_date=start_date_iso
                        )
                        results = alphas_data.get("results", [])
                        if not results:
                            break
                            
                        logger.info(f"Syncing {len(results)} alphas from {stage} (offset {offset})...")
                        
                        for a_data in results:
                            alpha_id = a_data.get("id")
                            if not alpha_id:
                                continue
                                
                            # Check existence
                            stmt = select(Alpha).where(Alpha.alpha_id == alpha_id)
                            result = await db.execute(stmt)
                            existing = result.scalar_one_or_none()
                            
                            # Parse Dates with Beijing Time Conversion (UTC+8)
                            from datetime import timezone, timedelta
                            BEIJING_TZ = timezone(timedelta(hours=8))
                            
                            def parse_to_beijing(iso_str):
                                if not iso_str: return None
                                try:
                                    # Parse as UTC aware
                                    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
                                    # Convert to Beijing
                                    dt_bj = dt.astimezone(BEIJING_TZ)
                                    # Return as naive (strip tzinfo) so it is stored exactly as is in DB
                                    return dt_bj.replace(tzinfo=None)
                                except:
                                    return None

                            date_created = parse_to_beijing(a_data.get("dateCreated"))
                            date_submitted = parse_to_beijing(a_data.get("dateSubmitted"))

                            settings = a_data.get("settings", {})
                            is_metrics = a_data.get("is", {})
                            os_metrics = a_data.get("os", {}) or {}
                            
                            if existing:
                                existing.status = a_data.get("status")
                                # existing.stage = stage # Don't overwrite stage if we want local state, but usually Brain is truth
                                existing.stage = stage 
                                existing.settings = settings
                                existing.tags = a_data.get("tags")
                                existing.checks = a_data.get("is", {}).get("checks", [])
                                
                                # Metrics
                                existing.is_metrics = is_metrics
                                existing.os_metrics = os_metrics
                                
                                # Update flat metrics (legacy compat)
                                existing.is_sharpe = is_metrics.get("sharpe")
                                existing.is_fitness = is_metrics.get("fitness")
                                existing.is_returns = is_metrics.get("returns")
                                existing.is_turnover = is_metrics.get("turnover")
                                existing.is_drawdown = is_metrics.get("drawdown")
                                
                                existing.date_modified = datetime.now() # Use local app time (Beijing)
                                if date_submitted:
                                    existing.date_submitted = date_submitted
                                
                                # Fix: Update new schema fields for existing records
                                existing.dataset_id = settings.get("datasetId")
                                existing.metrics = is_metrics
                                
                                # Update extended flat metrics
                                existing.is_margin = is_metrics.get("margin")
                                existing.is_long_count = is_metrics.get("longCount")
                                existing.is_short_count = is_metrics.get("shortCount")
                                    
                                updated += 1
                            else:
                                new_alpha = Alpha(
                                    alpha_id=alpha_id,
                                    type=a_data.get("type"),
                                    expression=(
                                        a_data.get("regular", {}).get("code") or 
                                        a_data.get("combo", {}).get("code") or 
                                        a_data.get("selection", {}).get("code") or
                                        "N/A" # Fallback to prevent NotNullViolation
                                    ),
                                    name=a_data.get("name"),
                                    region=settings.get("region"),
                                    universe=settings.get("universe"),
                                    
                                    dataset_id=settings.get("datasetId"),
                                    
                                    status=a_data.get("status"),
                                    stage=stage,
                                    
                                    settings=settings,
                                    tags=a_data.get("tags"),
                                    checks=a_data.get("is", {}).get("checks", []),
                                    
                                    is_metrics=is_metrics,
                                    os_metrics=os_metrics,
                                    
                                    is_sharpe=is_metrics.get("sharpe"),
                                    is_fitness=is_metrics.get("fitness"),
                                    is_returns=is_metrics.get("returns"),
                                    is_turnover=is_metrics.get("turnover"),
                                    is_drawdown=is_metrics.get("drawdown"),
                                    is_margin=is_metrics.get("margin"),
                                    is_long_count=is_metrics.get("longCount"),
                                    is_short_count=is_metrics.get("shortCount"),
                                    
                                    date_created=date_created,
                                    date_submitted=date_submitted,
                                    
                                    metrics=is_metrics, # Populate main metrics JSON for API
                                )
                                db.add(new_alpha)
                                count += 1
                        
                        # Commit incrementally (per page)
                        await db.commit()
                        logger.info(f"Committed {len(results)} updates/inserts.")
                        
                        offset += limit
                        if offset >= alphas_data.get("count", 0):
                            break
                
                logger.info(f"Alpha sync complete: {count} new, {updated} updated")
                return {"new": count, "updated": updated}

    return run_async(_run())
