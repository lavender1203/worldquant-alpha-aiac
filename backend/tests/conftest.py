"""
Pytest Configuration and Fixtures

This file provides:
- Async test configuration
- Database fixtures (in-memory SQLite for tests)
- Mock factory fixtures
- Common test utilities
"""

import asyncio
import sys
import os
from typing import AsyncGenerator, Generator
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.database import SQLAlchemyBase


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type, compiler, **kw):
    """Allow PostgreSQL JSONB models to be created in SQLite-backed unit tests."""
    return "JSON"


@compiles(ARRAY, "sqlite")
def _compile_array_for_sqlite(_type, compiler, **kw):
    """Allow PostgreSQL ARRAY models to be created in SQLite-backed unit tests."""
    return "JSON"


# =============================================================================
# Async Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create an async in-memory SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(SQLAlchemyBase.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a new database session for each test."""
    async_session_maker = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_brain_adapter():
    """Get a mock BrainAdapter for testing."""
    from backend.tests.fixtures.mock_brain import MockBrainAdapter
    return MockBrainAdapter()


@pytest.fixture
def mock_llm_service():
    """Get a mock LLMService for testing."""
    from backend.tests.fixtures.mock_llm import MockLLMService
    return MockLLMService()


# =============================================================================
# Model Factory Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def sample_task(db_session):
    """Create a sample mining task for testing."""
    from backend.models import MiningTask
    
    task = MiningTask(
        task_name="Test Task",
        region="USA",
        universe="TOP3000",
        dataset_strategy="AUTO",
        target_datasets=[],
        agent_mode="AUTONOMOUS",
        status="PENDING",
        daily_goal=4,
        progress_current=0,
        current_iteration=0,
        max_iterations=10,
        config={},
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)
    return task


@pytest_asyncio.fixture
async def sample_alpha(db_session, sample_task):
    """Create a sample alpha for testing."""
    from backend.models import Alpha
    
    alpha = Alpha(
        alpha_id="test-alpha-001",
        task_id=sample_task.id,
        expression="rank(close)",
        expression_hash="abc123",
        region="USA",
        universe="TOP3000",
        status="created",
        quality_status="PENDING",
        human_feedback="NONE",
        is_sharpe=1.5,
        is_fitness=0.8,
        is_turnover=0.3,
    )
    db_session.add(alpha)
    await db_session.commit()
    await db_session.refresh(alpha)
    return alpha


@pytest_asyncio.fixture
async def sample_knowledge_entry(db_session):
    """Create a sample knowledge entry for testing."""
    from backend.models import KnowledgeEntry
    
    entry = KnowledgeEntry(
        entry_type="SUCCESS_PATTERN",
        pattern="rank(ts_mean(close, 5))",
        description="Simple momentum pattern",
        meta_data={"category": "momentum"},
        usage_count=10,
        is_active=True,
    )
    db_session.add(entry)
    await db_session.commit()
    await db_session.refresh(entry)
    return entry


# =============================================================================
# Service Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def alpha_service(db_session):
    """Get an AlphaService instance for testing."""
    from backend.services import AlphaService
    return AlphaService(db_session)


@pytest_asyncio.fixture
async def dashboard_service(db_session):
    """Get a DashboardService instance for testing."""
    from backend.services import DashboardService
    return DashboardService(db_session)


@pytest_asyncio.fixture
async def mining_service(db_session, mock_brain_adapter):
    """Get a MiningService instance with mock brain adapter."""
    from backend.services import MiningService
    return MiningService(db_session, brain=mock_brain_adapter)


# =============================================================================
# Repository Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def alpha_repository(db_session):
    """Get an AlphaRepository instance for testing."""
    from backend.repositories import AlphaRepository
    return AlphaRepository(db_session)


@pytest_asyncio.fixture
async def task_repository(db_session):
    """Get a TaskRepository instance for testing."""
    from backend.repositories import TaskRepository
    return TaskRepository(db_session)


@pytest_asyncio.fixture
async def knowledge_repository(db_session):
    """Get a KnowledgeRepository instance for testing."""
    from backend.repositories import KnowledgeRepository
    return KnowledgeRepository(db_session)
