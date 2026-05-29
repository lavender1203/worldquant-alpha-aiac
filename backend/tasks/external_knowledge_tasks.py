"""
External knowledge tasks.

Imports curated academic alpha patterns into the knowledge base. Forum sync is
kept optional because it requires an MCP/forum client at runtime.
"""

from loguru import logger

from backend.celery_app import celery_app
from backend.database import AsyncSessionLocal
from backend.tasks import run_async


@celery_app.task(name="backend.tasks.sync_external_knowledge")
def sync_external_knowledge(force: bool = False):
    """Synchronize curated external knowledge into the knowledge base."""
    logger.info("Syncing external knowledge...")

    async def _run():
        async with AsyncSessionLocal() as db:
            from backend.external_knowledge import run_scheduled_sync

            return await run_scheduled_sync(db=db, force=force)

    return run_async(_run())
