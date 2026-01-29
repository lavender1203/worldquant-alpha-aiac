"""
Base utilities for LangGraph nodes.

Contains:
- Debug logging helper
- Trace recording helper
- Experiment tracking imports
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from backend.agents.graph.state import (
    MiningState,
    add_trace_step,
)
from backend.agents.services.trace_service import TraceService, TraceStepRecord


# =============================================================================
# DEBUG LOGGING
# =============================================================================

def _debug_log(hypo_id: str, location: str, message: str, data: Dict = None):
    """
    Write debug log entry to file for development/debugging.
    
    Args:
        hypo_id: Identifier for the hypothesis/alpha
        location: Code location (e.g., "nodes.py:rag_query:result")
        message: Log message
        data: Optional data dict to include
    """
    try:
        repo_root = Path(__file__).resolve().parents[4]
        log_path = repo_root / ".cursor" / "debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "hypothesisId": hypo_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session"
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# =============================================================================
# EXPERIMENT TRACKING
# =============================================================================

try:
    from backend.experiment_tracker import get_current_experiment, MetricsCollector
    EXPERIMENT_TRACKING_ENABLED = True
except ImportError:
    EXPERIMENT_TRACKING_ENABLED = False
    get_current_experiment = lambda: None
    MetricsCollector = None


# =============================================================================
# TRACE RECORDING
# =============================================================================

async def record_trace(
    state: MiningState,
    trace_service: Optional[TraceService],
    step_type: str,
    input_data: Dict = None,
    output_data: Dict = None,
    duration_ms: int = 0,
    status: str = "SUCCESS",
    error_message: str = None
) -> Dict:
    """
    Helper to update state AND persist trace to DB immediately.
    
    Args:
        state: Current mining state
        trace_service: Optional trace service for persistence
        step_type: Type of step (e.g., "RAG_QUERY", "CODE_GEN")
        input_data: Input data for the step
        output_data: Output data from the step
        duration_ms: Duration in milliseconds
        status: Step status ("SUCCESS", "FAILED", etc.)
        error_message: Optional error message
        
    Returns:
        Dict with state updates (trace_steps, step_order)
    """
    # 1. Update In-Memory State (Pydantic)
    state_update = add_trace_step(
        state, step_type, input_data, output_data, duration_ms, status, error_message
    )
    
    # 2. Persist to DB (Real-Time)
    if trace_service:
        try:
            step_order = state.step_order + 1
            
            record = TraceStepRecord(
                step_type=step_type,
                step_order=step_order,
                input_data=input_data or {},
                output_data=output_data or {},
                duration_ms=duration_ms,
                status=status,
                error_message=error_message
            )
            await trace_service.persist_record(record)
        except Exception as e:
            logger.error(f"Failed to persist trace step: {e}")
            
    return state_update
