"""
AIAC 2.0 - Alpha-GPT Mining System Backend
Main FastAPI Application with all routers
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import settings
from backend.database import init_db

# Import all routers
from backend.routers import dashboard, tasks, alphas, knowledge, config, datasets, operators, runs, mcp


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Initialize database
    await init_db()
    
    # Load operator registry from database
    try:
        from backend.alpha_semantic_validator import load_operators_from_db
        operators = await load_operators_from_db()
        from loguru import logger
        logger.info(f"[Startup] Operator registry loaded: {len(operators)} operators")
    except Exception as e:
        from loguru import logger
        logger.error(f"[Startup] Failed to load operators from DB: {e}")
    
    yield
    # Shutdown: Cleanup if needed
    pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Alpha-GPT 2.0 Mining System - Human-AI Collaborative Alpha Mining",
    version="2.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(dashboard.router, prefix=settings.API_V1_STR)
app.include_router(tasks.router, prefix=settings.API_V1_STR)
app.include_router(alphas.router, prefix=settings.API_V1_STR)
app.include_router(runs.router, prefix=settings.API_V1_STR)
app.include_router(knowledge.router, prefix=settings.API_V1_STR)
app.include_router(config.router, prefix=settings.API_V1_STR)
app.include_router(datasets.router, prefix=settings.API_V1_STR)
app.include_router(operators.router, prefix=settings.API_V1_STR)
app.include_router(mcp.router, prefix=settings.API_V1_STR)

# Keep optional analysis router if available. The legacy mining router targets
# pre-2.0 models and conflicts with the current /tasks workflow.
try:
    from backend.routers import analysis
    app.include_router(analysis.router, prefix=settings.API_V1_STR)
except ImportError:
    pass


@app.get("/")
def read_root():
    return {
        "message": "Welcome to AIAC 2.0 - Alpha-GPT Mining System",
        "version": "2.0.0",
        "docs": f"{settings.API_V1_STR}/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "alpha-gpt-backend",
        "version": "2.0.0"
    }


# API Summary endpoint for documentation
@app.get(f"{settings.API_V1_STR}/")
def api_summary():
    return {
        "api_version": "v1",
        "endpoints": {
            "dashboard": {
                "GET /stats/daily": "Daily mining statistics",
                "GET /stats/active-tasks": "Currently active tasks",
                "GET /stats/kpi": "Key performance indicators",
                "GET /stats/live-feed": "SSE live activity feed"
            },
            "tasks": {
                "GET /tasks": "List all tasks",
                "POST /tasks": "Create new task",
                "GET /tasks/{id}": "Task details with trace",
                "GET /tasks/{id}/trace": "Full trace timeline",
                "POST /tasks/{id}/start": "Start task",
                "POST /tasks/{id}/intervene": "Human intervention"
            },
            "alphas": {
                "GET /alphas": "List alphas with filters",
                "GET /alphas/{id}": "Alpha details",
                "POST /alphas/{id}/feedback": "Submit human feedback",
                "GET /alphas/{id}/trace": "Alpha generation trace"
            },
            "knowledge": {
                "GET /knowledge": "List knowledge entries",
                "GET /knowledge/success-patterns": "Successful patterns",
                "GET /knowledge/failure-pitfalls": "Failure lessons",
                "POST /knowledge": "Add knowledge entry",
                "PUT /knowledge/{id}": "Update entry",
                "DELETE /knowledge/{id}": "Deactivate entry"
            },
            "config": {
                "GET /config": "Get all configuration",
                "PUT /config/thresholds": "Update quality thresholds",
                "PUT /config/diversity": "Update diversity config",
                "GET /config/operators": "Get operator preferences",
                "PUT /config/operators/{name}": "Update operator status"
            }
        }
    }
