"""
Persistence nodes for LangGraph workflow.

Contains:
- node_save_results: Save alpha results to database
"""

from typing import Dict
from loguru import logger
from langchain_core.runnables import RunnableConfig

from backend.agents.graph.state import MiningState, AlphaResult, FailureRecord
from backend.agents.graph.nodes.base import record_trace


# =============================================================================
# NODE: Save Results
# =============================================================================

async def node_save_results(state: MiningState, config: RunnableConfig = None) -> Dict:
    """
    Batch process and save ALL results (Successes and Failures).
    
    Input State:
        - pending_alphas
    
    Output Updates:
        - generated_alphas (appends all simulated alphas with BRAIN ids)
        - failures (appends failures)
        - pending_alphas (cleared)
        - trace_steps
    """
    node_name = "SAVE_RESULTS"
    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    
    result_batch = []
    fail_batch = []
    
    logger.info(f"[{node_name}] Starting batch save | total={len(state.pending_alphas)}")
    
    for alpha in state.pending_alphas:
        has_simulated_alpha = bool(
            alpha.alpha_id and alpha.is_simulated and alpha.simulation_success
        )

        if has_simulated_alpha:
            res = AlphaResult(
                expression=alpha.expression,
                hypothesis=alpha.hypothesis,
                explanation=alpha.explanation,
                alpha_id=alpha.alpha_id,
                metrics=alpha.metrics,
                quality_status=alpha.quality_status,
            )
            result_batch.append(res)
            logger.info(
                f"[{node_name}] Alpha Saved | id={alpha.alpha_id} "
                f"status={alpha.quality_status}"
            )

        if not has_simulated_alpha or alpha.quality_status not in {"PASS", "OPTIMIZE"}:
            # Determine error type and message
            err_type = "UNKNOWN"
            err_msg = "Unknown error"
            
            if alpha.is_valid is False:
                err_type = "SYNTAX_ERROR"
                err_msg = alpha.validation_error or "Syntax Error"
            elif alpha.is_simulated and not alpha.simulation_success:
                err_type = "SIMULATION_ERROR"
                err_msg = alpha.simulation_error or "Simulation Failed"
            elif alpha.quality_status == "FAIL":
                err_type = "QUALITY_CHECK_FAILED"
                err_msg = "Metrics below threshold"
                strict_failures = alpha.metrics.get("_strict_gate_failures") if alpha.metrics else None
                if strict_failures:
                    err_msg = f"Metrics below threshold: {', '.join(strict_failures[:5])}"
            else:
                err_type = "OTHER"
                err_msg = "Unknown failure"
            
            rec = FailureRecord(
                expression=alpha.expression,
                error_type=err_type,
                error_message=err_msg,
                details={
                    "alpha_id": alpha.alpha_id,
                    "quality_status": alpha.quality_status,
                    "metrics": alpha.metrics,
                    "hypothesis": alpha.hypothesis,
                    "candidate_metadata": alpha.metadata,
                    "is_simulated": alpha.is_simulated,
                    "simulation_success": alpha.simulation_success,
                }
            )
            fail_batch.append(rec)
    
    # Record trace
    if trace_service:
        await record_trace(
            state, trace_service, node_name,
            {},
            {
                "saved": len(result_batch),
                "failed": len(fail_batch),
                "saved_alphas": len(result_batch),
                "saved_failures": len(fail_batch),
            },
            0,
            "SUCCESS",
            None
        )
    
    return {
        "generated_alphas": state.generated_alphas + result_batch,
        "failures": state.failures + fail_batch,
        "pending_alphas": [],
        "current_alpha_index": 0
    }
