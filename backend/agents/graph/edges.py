"""
LangGraph Edge Functions
Conditional routing logic for the mining workflow
"""

from typing import Literal
from loguru import logger

from backend.agents.graph.state import MiningState
from backend.config import settings


# =============================================================================
# EDGE: After Validate
# =============================================================================

def route_after_validate(state: MiningState) -> Literal["simulate", "self_correct"]:
    """
    Route after validation step (Batch).
    
    - If ALL valid: proceed to simulate
    - If SOME invalid and retries available: go to self-correct
    - If max retries reached: proceed to simulate (only valid ones will run)
    """
    valid_count = sum(1 for a in state.pending_alphas if a.is_valid)
    invalid_alphas = [a for a in state.pending_alphas if not a.is_valid]
    any_invalid = bool(invalid_alphas)
    
    if not any_invalid:
        logger.debug("[Edge] route_after_validate -> simulate (All Valid)")
        return "simulate"

    only_duplicates = all(
        "duplicate" in (a.validation_error or "").lower()
        for a in invalid_alphas
    )
    if only_duplicates:
        logger.debug(
            "[Edge] route_after_validate -> simulate "
            f"(duplicate invalids skipped, valid={valid_count})"
        )
        return "simulate"
    
    if state.retry_count < state.max_retries:
        logger.debug(f"[Edge] route_after_validate -> self_correct (retry {state.retry_count + 1}/{state.max_retries})")
        return "self_correct"
    
    logger.debug("[Edge] route_after_validate -> simulate (Max retries, processing valid only)")
    return "simulate"


# =============================================================================
# EDGE: Error Check
# =============================================================================

def route_check_error(state: MiningState) -> Literal["continue", "error"]:
    """
    Check if there's a critical error that should stop execution.
    """
    if state.should_stop or state.error:
        logger.warning(f"[Edge] route_check_error -> error: {state.error}")
        return "error"
    
    return "continue"
