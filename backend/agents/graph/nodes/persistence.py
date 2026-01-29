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
from backend.config import settings


def _summarize_brain_checks(metrics: Dict) -> tuple[str, str]:
    """
    Summarize BRAIN official checks into a compact (error_type, error_message).
    Falls back to local threshold heuristics when checks are missing.
    """
    metrics = metrics or {}

    checks = metrics.get("checks") or []
    check_map = {}
    if isinstance(checks, list):
        for c in checks:
            if isinstance(c, dict) and c.get("name"):
                check_map[str(c["name"])] = c

    failed = metrics.get("_brain_failed_checks") or metrics.get("failed_checks") or []
    pending = metrics.get("_brain_pending_checks") or metrics.get("pending_checks") or []

    # If not explicitly provided, infer from checks list.
    if not failed and check_map:
        failed = [n for n, c in check_map.items() if c.get("result") == "FAIL"]
    if not pending and check_map:
        pending = [n for n, c in check_map.items() if c.get("result") == "PENDING"]

    if failed:
        parts = []
        for name in failed[:5]:
            c = check_map.get(name, {})
            limit = c.get("limit")
            value = c.get("value")
            if limit is not None and value is not None:
                parts.append(f"{name}(value={value}, limit={limit})")
            else:
                parts.append(str(name))
        return str(failed[0]), "BRAIN checks failed: " + ", ".join(parts)

    if pending:
        return "PENDING_CHECKS", "BRAIN checks pending: " + ", ".join([str(x) for x in pending[:5]])

    # Fallback: local thresholds
    sharpe = float(metrics.get("sharpe", 0) or 0)
    turnover = float(metrics.get("turnover", 0) or 0)
    fitness = float(metrics.get("fitness", 0) or 0)

    if sharpe < 0:
        return "NEGATIVE_SIGNAL", f"Sharpe negative ({sharpe:.3f})"
    if sharpe < getattr(settings, "SHARPE_MIN", 1.5):
        return "LOW_SHARPE", f"Sharpe below threshold ({sharpe:.3f})"
    if fitness < getattr(settings, "FITNESS_MIN", 0.6):
        return "LOW_FITNESS", f"Fitness below threshold ({fitness:.3f})"
    if turnover > getattr(settings, "TURNOVER_MAX", 0.7):
        return "HIGH_TURNOVER", f"Turnover above threshold ({turnover:.3f})"

    return "QUALITY_CHECK_FAILED", "Metrics below threshold"


# =============================================================================
# NODE: Save Results
# =============================================================================

async def node_save_results(state: MiningState, config: RunnableConfig = None) -> Dict:
    """
    Batch process and save ALL results (Successes and Failures).
    
    Input State:
        - pending_alphas
    
    Output Updates:
        - generated_alphas (appends successes)
        - failures (appends failures)
        - pending_alphas (cleared)
        - trace_steps
    """
    node_name = "SAVE_RESULTS"
    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    
    success_batch = []
    fail_batch = []
    
    logger.info(f"[{node_name}] Starting batch save | total={len(state.pending_alphas)}")
    
    for alpha in state.pending_alphas:
        # Persist PASS + non-submit-ready but valuable candidates (PROMISING/OPTIMIZE)
        if alpha.quality_status in {"PASS", "PROMISING", "OPTIMIZE"}:
            res = AlphaResult(
                expression=alpha.expression,
                hypothesis=alpha.hypothesis,
                explanation=alpha.explanation,
                alpha_id=alpha.alpha_id,
                metrics=alpha.metrics,
                quality_status=alpha.quality_status
            )
            success_batch.append(res)
            logger.info(f"[{node_name}] Alpha Saved | id={alpha.alpha_id} status={alpha.quality_status}")
            
        else:
            # Determine error type and message
            err_type = "UNKNOWN"
            err_msg = "Unknown error"
            
            if alpha.is_valid is False:
                err_type = "SYNTAX_ERROR"
                err_msg = alpha.validation_error or "Syntax Error"
            elif alpha.is_simulated and not alpha.simulation_success:
                err_type = "SIMULATION_ERROR"
                err_msg = alpha.simulation_error or "Simulation Failed"
            elif alpha.quality_status == "OPTIMIZE":
                err_type = "OPTIMIZE"
                err_msg = (alpha.metrics or {}).get("_optimize_reason") or "Marked for optimization"
            elif alpha.quality_status == "FAIL":
                err_type, err_msg = _summarize_brain_checks(alpha.metrics or {})
            else:
                err_type = "OTHER"
                err_msg = "Unknown failure"
            
            rec = FailureRecord(
                expression=alpha.expression,
                error_type=err_type,
                error_message=err_msg,
                details={"metrics": alpha.metrics, "hypothesis": alpha.hypothesis}
            )
            fail_batch.append(rec)
    
    # Record trace
    if trace_service:
        await record_trace(
            state, trace_service, node_name,
            {},
            {"saved": len(success_batch), "failed": len(fail_batch)},
            0,
            "SUCCESS",
            None
        )
    
    return {
        "generated_alphas": state.generated_alphas + success_batch,
        "failures": state.failures + fail_batch,
        "pending_alphas": [],
        "current_alpha_index": 0
    }
