import pytest
from sqlalchemy.dialects import postgresql

from backend.tasks.mining_tasks import _mark_mining_task_failed, _resolve_final_status


class FakeSession:
    def __init__(self):
        self.statements = []
        self.committed = False

    async def execute(self, statement):
        self.statements.append(statement)

    async def commit(self):
        self.committed = True


def _compile(statement):
    return str(statement.compile(dialect=postgresql.dialect()))


def _update_value(statement, column_name):
    for column, value in statement._values.items():
        if column.name == column_name:
            return value.value
    raise AssertionError(f"Missing update value for {column_name}")


def test_target_miss_after_completed_iterations_is_not_failed():
    status, message = _resolve_final_status(
        target_reached=False,
        incomplete_reasons=["fundamental94: target not reached (0/1) after 10/10 iterations"],
        progress_current=0,
        daily_goal=1,
    )

    assert status == "COMPLETED"
    assert "target not reached" in message


def test_missing_workflow_result_still_fails_when_target_not_reached():
    status, message = _resolve_final_status(
        target_reached=False,
        incomplete_reasons=[],
        progress_current=0,
        daily_goal=1,
    )

    assert status == "FAILED"
    assert message == "Target not reached: 0/1"


@pytest.mark.asyncio
async def test_mark_mining_task_failed_updates_task_and_run():
    db = FakeSession()

    await _mark_mining_task_failed(
        db,
        task_id=128,
        run_id=133,
        celery_task_id="celery-123",
        error_message="TimeLimitExceeded(3600,)",
    )

    assert db.committed is True
    assert len(db.statements) == 2
    assert db.statements[0].table.name == "mining_tasks"
    assert db.statements[1].table.name == "experiment_runs"
    assert _update_value(db.statements[0], "status") == "FAILED"
    assert _update_value(db.statements[1], "status") == "FAILED"
    assert _update_value(db.statements[1], "error_message") == "TimeLimitExceeded(3600,)"


@pytest.mark.asyncio
async def test_mark_mining_task_failed_only_overwrites_running_states():
    db = FakeSession()

    await _mark_mining_task_failed(
        db,
        task_id=128,
        run_id=None,
        celery_task_id="celery-123",
        error_message="TimeLimitExceeded(3600,)",
    )

    task_sql = _compile(db.statements[0])
    run_sql = _compile(db.statements[1])

    assert "mining_tasks.status = " in task_sql
    assert "experiment_runs.status = " in run_sql
    assert "experiment_runs.celery_task_id = " in run_sql
