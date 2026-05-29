"""Initial schema from SQLAlchemy models.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-05-29
"""

from typing import Sequence, Union

from alembic import op

from backend.database import SQLAlchemyBase
from backend.models import *  # noqa: F401,F403


revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    SQLAlchemyBase.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    SQLAlchemyBase.metadata.drop_all(bind=bind)
