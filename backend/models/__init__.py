"""
Models Module - Database entities

This module provides all SQLAlchemy models for the application.
Models are organized into separate files by domain but re-exported
here for backward compatibility.

Usage:
    from backend.models import Alpha, MiningTask, KnowledgeEntry
"""

# Enums
from backend.models.base import (
    MiningStatus,
    DatasetStrategy,
    AgentMode,
    TraceStepType,
    QualityStatus,
    HumanFeedback,
    KnowledgeEntryType,
    JobStatus,
)

# Task models
from backend.models.task import (
    MiningTask,
    ExperimentRun,
    TraceStep,
    MiningJob,
)

# Alpha models
from backend.models.alpha import (
    Alpha,
    AlphaFailure,
    AlphaPnl,
)

# Knowledge models
from backend.models.knowledge import (
    KnowledgeEntry,
    OperatorPreference,
    RLState,
    RLAction,
)

# Metadata models
from backend.models.metadata import (
    DatasetMetadata,
    DataField,
    Operator,
    OperatorBlacklist,
    Region,
    Universe,
    Neutralization,
    PyramidMultiplier,
    Template,
    TemplateVariable,
)

# Config models
from backend.models.config import (
    SystemConfig,
    BrainAuthToken,
    WQBCredential,
    LLMProvider,
    MCPServer,
    MCPTool,
)

__all__ = [
    # Enums
    "MiningStatus",
    "DatasetStrategy",
    "AgentMode",
    "TraceStepType",
    "QualityStatus",
    "HumanFeedback",
    "KnowledgeEntryType",
    "JobStatus",
    # Task
    "MiningTask",
    "ExperimentRun",
    "TraceStep",
    "MiningJob",
    # Alpha
    "Alpha",
    "AlphaFailure",
    "AlphaPnl",
    # Knowledge
    "KnowledgeEntry",
    "OperatorPreference",
    "RLState",
    "RLAction",
    # Metadata
    "DatasetMetadata",
    "DataField",
    "Operator",
    "OperatorBlacklist",
    "Region",
    "Universe",
    "Neutralization",
    "PyramidMultiplier",
    "Template",
    "TemplateVariable",
    # Config
    "SystemConfig",
    "BrainAuthToken",
    "WQBCredential",
    "LLMProvider",
    "MCPServer",
    "MCPTool",
]
