"""
LangGraph Node Functions - Modular Organization

This package contains all node functions split by responsibility:
- base: Common helpers, trace recording, constants
- generation: RAG query, hypothesis, code generation
- mcp: Runtime MCP tool registry loading
- validation: Expression validation, self-correction
- evaluation: Simulation, quality evaluation
- persistence: Save results to database

For backward compatibility, all functions are re-exported here.
"""

# Base utilities
from backend.agents.graph.nodes.base import (
    record_trace,
    _debug_log,
    EXPERIMENT_TRACKING_ENABLED,
    get_current_experiment,
)

# Generation nodes
from backend.agents.graph.nodes.generation import (
    node_rag_query,
    node_distill_context,
    node_hypothesis,
    node_code_gen,
)

# MCP runtime node
from backend.agents.graph.nodes.mcp import (
    node_load_mcp_tools,
)

# Validation nodes
from backend.agents.graph.nodes.validation import (
    node_validate,
    node_self_correct,
)

# Evaluation nodes
from backend.agents.graph.nodes.evaluation import (
    node_simulate,
    node_evaluate,
)

# Persistence nodes
from backend.agents.graph.nodes.persistence import (
    node_save_results,
)

__all__ = [
    # Base
    "record_trace",
    "_debug_log",
    "EXPERIMENT_TRACKING_ENABLED",
    "get_current_experiment",
    # Generation
    "node_rag_query",
    "node_distill_context",
    "node_hypothesis",
    "node_code_gen",
    # MCP
    "node_load_mcp_tools",
    # Validation
    "node_validate",
    "node_self_correct",
    # Evaluation
    "node_simulate",
    "node_evaluate",
    # Persistence
    "node_save_results",
]
