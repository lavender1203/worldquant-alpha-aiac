"""
Tasks Module - Celery background tasks

This module organizes Celery tasks by category:
- mining_tasks: Mining task execution
- feedback_tasks: Feedback analysis and learning
- sync_tasks: Data synchronization with BRAIN
- external_knowledge_tasks: Curated external knowledge import

Common utilities are provided here.
"""

import asyncio
from backend.celery_app import celery_app

# Common utility for running async code in Celery
def run_async(coro):
    """Helper to run async functions in Celery tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Re-export all tasks for backward compatibility
from backend.tasks.mining_tasks import run_mining_task
from backend.tasks.feedback_tasks import (
    run_daily_feedback,
    update_operator_stats,
    learn_from_alpha,
)
from backend.tasks.sync_tasks import (
    sync_datasets,
    sync_datasets_from_brain,
    sync_operators_from_brain,
    sync_fields_from_brain,
    sync_user_alphas,
)
from backend.tasks.external_knowledge_tasks import sync_external_knowledge

__all__ = [
    # Utilities
    "run_async",
    "celery_app",
    # Mining
    "run_mining_task",
    # Feedback
    "run_daily_feedback",
    "update_operator_stats",
    "learn_from_alpha",
    # Sync
    "sync_datasets",
    "sync_datasets_from_brain",
    "sync_operators_from_brain",
    "sync_fields_from_brain",
    "sync_user_alphas",
    "sync_external_knowledge",
]
