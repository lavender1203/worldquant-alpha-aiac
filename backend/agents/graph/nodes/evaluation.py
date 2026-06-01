"""
Evaluation nodes for LangGraph workflow.

Enhanced with hypothesis-implementation alignment checking:
- Verifies implementations correctly reflect hypotheses
- Attributes failures to hypothesis vs implementation
- Filters knowledge based on attribution confidence

Contains:
- node_simulate: Batch simulate alphas on BRAIN platform
- node_evaluate: Evaluate alpha quality using multi-objective scoring
"""

import time
import random
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from langchain_core.runnables import RunnableConfig

from backend.agents.graph.state import MiningState
from backend.agents.graph.nodes.base import (
    record_trace,
    _debug_log,
    EXPERIMENT_TRACKING_ENABLED,
    get_current_experiment,
)
from backend.adapters.brain_adapter import BrainAdapter
from backend.config import settings
from backend.agents.prompts import (
    quick_alignment_check,
    determine_attribution_heuristic,
)


def _expression_components(expression: str, fields: List[Dict]) -> Tuple[List[str], List[str]]:
    """Extract field/operator usage for hard complexity gates."""
    if not expression:
        return [], []

    try:
        from backend.alpha_semantic_validator import AlphaSemanticValidator

        validator = AlphaSemanticValidator(fields=fields, strict_field_check=False)
        result = validator.validate(expression)
        return sorted(result.used_fields), sorted(op.lower() for op in result.used_operators)
    except Exception:
        import re

        ops = sorted({m.group(1).lower() for m in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expression)})
        field_ids = {str(f.get("id") or f.get("name") or "").lower() for f in fields}
        used_fields = sorted(
            {
                token
                for token in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", expression)
                if token.lower() in field_ids
            }
        )
        return used_fields, ops


def _operator_call_count(expression: str) -> int:
    """Count operator calls, not unique operator names, for hard complexity gates."""
    if not expression:
        return 0
    import re

    return len(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(", expression))


def _as_float(value, default: float = 0.0) -> float:
    """Convert numeric API values defensively."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _default_thresholds() -> Dict[str, Any]:
    return {
        "sharpe_min": getattr(settings, "SHARPE_MIN", 1.58),
        "two_year_sharpe_min": getattr(settings, "TWO_YEAR_SHARPE_MIN", 1.6),
        "fitness_min": getattr(settings, "FITNESS_MIN", 1.0),
        "rn_sharpe_min": getattr(settings, "RN_SHARPE_MIN", 1.58),
        "rn_fitness_min": getattr(settings, "RN_FITNESS_MIN", 1.0),
        "margin_min": getattr(settings, "MARGIN_MIN", 0.001),
        "turnover_min": getattr(settings, "TURNOVER_MIN", 0.05),
        "turnover_max": getattr(settings, "TURNOVER_MAX", 0.30),
        "prod_corr_max": getattr(settings, "PROD_CORR_MAX", 0.7),
        "self_corr_max": getattr(settings, "SELF_CORR_MAX", 0.5),
        "ra_fails_max": getattr(settings, "RA_FAILS_MAX", 0),
        "score_pass": getattr(settings, "SCORE_PASS_THRESHOLD", 0.8),
        "score_optimize": getattr(settings, "SCORE_OPTIMIZE_THRESHOLD", 0.3),
        "corr_check": getattr(settings, "CORR_CHECK_THRESHOLD", 0.5),
        "max_operator_count": getattr(settings, "MAX_OPERATOR_COUNT", 5),
    }


async def _load_thresholds(config: RunnableConfig = None) -> Dict[str, Any]:
    thresholds = _default_thresholds()
    configurable = config.get("configurable", {}) if config else {}

    configured_quality = configurable.get("quality_thresholds")
    configured_diversity = configurable.get("diversity_thresholds")
    task_config = configurable.get("task_config") or {}

    if not configured_quality:
        try:
            from backend.database import AsyncSessionLocal
            from backend.services.config_service import ConfigService

            async with AsyncSessionLocal() as db:
                service = ConfigService(db)
                configured_quality = await service.get_thresholds()
                configured_diversity = await service.get_diversity_config()
        except Exception as e:
            logger.warning(f"[EVALUATE] Failed to load DB thresholds, using settings: {e}")

    def read_config_value(source: Any, key: str):
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    if configured_quality:
        # DB/UI thresholds are useful for dashboards, but mining deliverables must
        # not be relaxed below the hard production gates requested for this run.
        configured_sharpe = _as_float(read_config_value(configured_quality, "sharpe_min"), thresholds["sharpe_min"])
        configured_fitness = _as_float(read_config_value(configured_quality, "fitness_min"), thresholds["fitness_min"])
        configured_turnover_max = _as_float(read_config_value(configured_quality, "turnover_max"), thresholds["turnover_max"])
        thresholds["sharpe_min"] = max(thresholds["sharpe_min"], configured_sharpe)
        thresholds["fitness_min"] = max(thresholds["fitness_min"], configured_fitness)
        thresholds["turnover_max"] = min(thresholds["turnover_max"], configured_turnover_max)

    if configured_diversity:
        thresholds["prod_corr_max"] = _as_float(
            read_config_value(configured_diversity, "max_correlation"),
            thresholds["prod_corr_max"],
        )

    for key in (
        "two_year_sharpe_min",
        "rn_sharpe_min",
        "rn_fitness_min",
        "margin_min",
        "turnover_min",
        "turnover_max",
        "prod_corr_max",
        "self_corr_max",
        "max_operator_count",
    ):
        if key in task_config:
            if key in {"turnover_max", "prod_corr_max", "self_corr_max", "max_operator_count"}:
                thresholds[key] = min(thresholds[key], _as_float(task_config.get(key), thresholds[key]))
            else:
                thresholds[key] = max(thresholds[key], _as_float(task_config.get(key), thresholds[key]))

    return thresholds


def _risk_neutralized_metrics(metrics: Dict) -> Dict[str, Any]:
    """Normalize nested and flat MCP/BRAIN risk-neutralized metric shapes."""
    rn_metrics = metrics.get("riskNeutralized") or metrics.get("risk_neutralized") or {}
    if not isinstance(rn_metrics, dict):
        rn_metrics = {}

    sharpe = (
        rn_metrics.get("sharpe")
        if rn_metrics.get("sharpe") is not None
        else metrics.get("risk_neutralized_sharpe", metrics.get("rn_sharpe"))
    )
    fitness = (
        rn_metrics.get("fitness")
        if rn_metrics.get("fitness") is not None
        else metrics.get("risk_neutralized_fitness", metrics.get("rn_fitness"))
    )

    normalized = dict(rn_metrics)
    if sharpe is not None:
        normalized["sharpe"] = sharpe
    if fitness is not None:
        normalized["fitness"] = fitness
    if "sharpe" not in normalized or "fitness" not in normalized:
        sim_settings = metrics.get("_settings") or metrics.get("settings") or {}
        neutralization = str(sim_settings.get("neutralization") or "").upper()
        if neutralization and neutralization != "NONE":
            if "sharpe" not in normalized and metrics.get("sharpe") is not None:
                normalized["sharpe"] = metrics.get("sharpe")
            if "fitness" not in normalized and metrics.get("fitness") is not None:
                normalized["fitness"] = metrics.get("fitness")
            normalized.setdefault("_source", f"neutralized_setting:{neutralization}")
    return normalized


def _check_value(metrics: Dict, check_name: str) -> Any:
    """Return a BRAIN check value when present."""
    for check in metrics.get("checks") or []:
        if isinstance(check, dict) and check.get("name") == check_name:
            return check.get("value")
    return None


def _is_cancelled_sim_result(result: Dict[str, Any]) -> bool:
    """Detect children cancelled because a sibling expression broke the batch."""
    if result.get("success"):
        return False
    raw = result.get("raw_response") or result.get("raw") or {}
    if isinstance(raw, dict) and str(raw.get("status") or "").upper() == "CANCELLED":
        return True
    error = str(result.get("error") or "").upper()
    return "CANCELLED" in error


def _padding_expression() -> str:
    """Known simple expression used only to satisfy multi-sim's min batch size."""
    return "rank(close)"


def _strict_gate_failures(
    metrics: Dict,
    brain_failed_checks: List[str],
    prod_corr: Optional[float],
    self_corr: Optional[float],
    thresholds: Dict[str, Any],
    expression: str = "",
    fields: Optional[List[Dict]] = None,
) -> List[str]:
    """Production-quality gates requested for mined deliverables."""
    sharpe = _as_float(metrics.get("sharpe"))
    two_year_value = metrics.get("two_year_sharpe")
    if two_year_value is None:
        two_year_value = _check_value(metrics, "LOW_2Y_SHARPE")
    two_year_sharpe = _as_float(two_year_value, sharpe)
    fitness = _as_float(metrics.get("fitness"))
    margin = _as_float(metrics.get("margin"))
    turnover = _as_float(metrics.get("turnover"))
    rn_metrics = _risk_neutralized_metrics(metrics)
    rn_sharpe = _as_float(rn_metrics.get("sharpe"), default=None)
    rn_fitness = _as_float(rn_metrics.get("fitness"), default=None)
    ra_fails = len(brain_failed_checks or [])

    sharpe_min = thresholds["sharpe_min"]
    two_year_sharpe_min = thresholds["two_year_sharpe_min"]
    fitness_min = thresholds["fitness_min"]
    rn_sharpe_min = thresholds["rn_sharpe_min"]
    rn_fitness_min = thresholds["rn_fitness_min"]
    margin_min = thresholds["margin_min"]
    turnover_min = thresholds["turnover_min"]
    turnover_max = thresholds["turnover_max"]
    prod_corr_max = thresholds["prod_corr_max"]
    self_corr_max = thresholds["self_corr_max"]
    ra_fails_max = thresholds["ra_fails_max"]
    max_operator_count = int(thresholds.get("max_operator_count", 7))

    failures = []
    used_fields, used_ops = _expression_components(expression, fields or [])
    stage = metrics.get("stage")
    status = metrics.get("status")
    if "trade_when" in {op.lower() for op in used_ops}:
        failures.append("TRADE_WHEN_FORBIDDEN")
    if len(used_fields) > 2:
        failures.append(f"TOO_MANY_FIELDS (fields={len(used_fields)} > 2)")
    operator_calls = _operator_call_count(expression)
    if operator_calls > max_operator_count:
        failures.append(f"TOO_MANY_OPERATORS (ops={operator_calls} > {max_operator_count})")
    if stage != "IS":
        failures.append(f"NOT_IS_STAGE (stage={stage if stage is not None else 'missing'})")
    if status != "UNSUBMITTED":
        failures.append(f"NOT_UNSUBMITTED (status={status if status is not None else 'missing'})")
    if sharpe <= sharpe_min:
        failures.append(f"LOW_SHARPE (is={sharpe:.2f} <= {sharpe_min:.2f})")
    if two_year_sharpe <= two_year_sharpe_min:
        failures.append(f"LOW_2Y_SHARPE (2y={two_year_sharpe:.2f} <= {two_year_sharpe_min:.2f})")
    if fitness <= fitness_min:
        failures.append(f"LOW_FITNESS (fit={fitness:.2f} <= {fitness_min:.2f})")
    if rn_sharpe is None or rn_sharpe <= rn_sharpe_min:
        failures.append(f"LOW_RN_SHARPE (rn={rn_sharpe if rn_sharpe is not None else 'missing'} <= {rn_sharpe_min:.2f})")
    if rn_fitness is None or rn_fitness <= rn_fitness_min:
        failures.append(f"LOW_RN_FITNESS (rn={rn_fitness if rn_fitness is not None else 'missing'} <= {rn_fitness_min:.2f})")
    if margin <= margin_min:
        failures.append(f"LOW_MARGIN (margin={margin:.6f} <= {margin_min:.6f})")
    if turnover <= turnover_min:
        failures.append(f"LOW_TURNOVER (to={turnover:.3f} <= {turnover_min:.3f})")
    if turnover >= turnover_max:
        failures.append(f"HIGH_TURNOVER (to={turnover:.3f} >= {turnover_max:.3f})")
    if ra_fails > ra_fails_max:
        failures.append(f"RA_FAILS (fails={ra_fails} > {ra_fails_max})")
    if prod_corr is None:
        failures.append("PROD_CORR_MISSING")
    elif prod_corr >= prod_corr_max:
        failures.append(f"HIGH_PROD_CORR (pc={prod_corr:.3f} >= {prod_corr_max:.3f})")
    if self_corr is None:
        failures.append("SELF_CORR_MISSING")
    elif self_corr >= self_corr_max:
        failures.append(f"HIGH_SELF_CORR (sc={self_corr:.3f} >= {self_corr_max:.3f})")

    return failures


# =============================================================================
# NODE: Simulate
# =============================================================================

async def node_simulate(
    state: MiningState,
    brain: BrainAdapter,
    config: RunnableConfig = None
) -> Dict:
    """
    Batch simulate ALL valid alphas on BRAIN platform.
    
    Enhanced with DB-level deduplication:
    - Check expression hash against existing alphas before simulation
    - Skip already-simulated expressions to save API calls
    
    Input State:
        - pending_alphas, region, universe
    
    Output Updates:
        - pending_alphas (with simulation result)
        - trace_steps
    """
    start_time = time.time()
    node_name = "SIMULATE"
    
    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    task_config = config.get("configurable", {}).get("task_config", {}) if config else {}
    
    # Filter valid alphas that haven't been simulated
    valid_indices = [
        i for i, a in enumerate(state.pending_alphas)
        if a.is_valid and not a.simulation_success
    ]
    
    if not valid_indices:
        logger.warning(f"[{node_name}] No valid alphas to simulate")
        return {}
    
    # DB-level deduplication check
    db_duplicates = 0
    indices_to_simulate = []
    sim_delay = int(task_config.get("delay", 1))
    sim_decay = int(task_config.get("decay", 4))
    sim_truncation = float(task_config.get("truncation", 0.08))
    sim_neutralization = str(task_config.get("neutralization", "SUBINDUSTRY"))
    sim_test_period = str(task_config.get("test_period", "P2Y0M"))
    
    try:
        from backend.database import AsyncSessionLocal
        from backend.selection_strategy import filter_unsimulated_expressions
        
        expressions_to_check = [state.pending_alphas[i].expression for i in valid_indices]
        
        async with AsyncSessionLocal() as db:
            new_exprs, dup_exprs = await filter_unsimulated_expressions(
                db,
                expressions_to_check,
                state.region,
                state.universe,
                delay=sim_delay,
                decay=sim_decay,
                neutralization=sim_neutralization,
                truncation=sim_truncation,
            )
        
        new_expr_set = set(new_exprs)
        for idx in valid_indices:
            expr = state.pending_alphas[idx].expression
            if expr in new_expr_set:
                indices_to_simulate.append(idx)
            else:
                db_duplicates += 1
                state.pending_alphas[idx].simulation_error = "DB duplicate: already simulated"
                state.pending_alphas[idx].is_simulated = True
                state.pending_alphas[idx].simulation_success = False
        
        logger.info(
            f"[{node_name}] DB dedup: {db_duplicates} duplicates skipped, "
            f"{len(indices_to_simulate)} to simulate"
        )
        
    except Exception as e:
        logger.warning(f"[{node_name}] DB dedup check failed, proceeding with all: {e}")
        indices_to_simulate = valid_indices
    
    if not indices_to_simulate:
        logger.warning(f"[{node_name}] All expressions already in DB")
        return {"pending_alphas": state.pending_alphas}
    
    logger.info(f"[{node_name}] Starting batch simulation | count={len(indices_to_simulate)} region={state.region}")
    
    expressions = [state.pending_alphas[i].expression for i in indices_to_simulate]
    
    _debug_log("E", "nodes.py:simulate:expressions", "Expressions to simulate", {
        "count": len(expressions),
        "expressions": [e[:150] for e in expressions],
        "region": state.region,
        "universe": state.universe
    })
    
    max_batch_size = max(2, int(task_config.get("simulation_batch_size", 4)))

    async def simulate_expression_batch(expression_batch: List[str], batch_num: int, total_batches: int) -> List[Dict]:
        try:
            logger.info(
                f"[{node_name}] Simulating isolated batch "
                f"{batch_num}/{total_batches} size={len(expression_batch)}"
            )
            return await brain.simulate_batch(
                expressions=expression_batch,
                region=state.region,
                universe=state.universe,
                delay=sim_delay,
                decay=sim_decay,
                truncation=sim_truncation,
                neutralization=sim_neutralization,
                test_period=sim_test_period,
                max_wait=int(task_config.get("simulation_max_wait", 900)),
                timeout_grace_seconds=int(task_config.get("simulation_timeout_grace_seconds", 180)),
                no_child_timeout_seconds=int(task_config.get("simulation_no_child_timeout_seconds", 0)),
            )
        except Exception as e:
            error = f"{type(e).__name__}: {str(e) or repr(e)}"
            logger.error(f"[{node_name}] Batch Simulate Loop Error: {error}")
            return [{"success": False, "error": error} for _ in expression_batch]

    # Multi-sim cancels sibling expressions when one expression has a platform
    # unit/type error. Retry only the CANCELLED siblings in smaller batches so
    # one bad first-order probe does not erase the rest of the operator map.
    jobs = list(zip(indices_to_simulate, expressions))
    pending_jobs = jobs
    results_by_index: Dict[int, Dict] = {}
    retry_limit = int(task_config.get("simulation_cancel_retry_passes", 2))
    retry_pass = 0

    while pending_jobs:
        pass_batch_size = max_batch_size if retry_pass == 0 else 2
        job_batches = [
            pending_jobs[offset: offset + pass_batch_size]
            for offset in range(0, len(pending_jobs), pass_batch_size)
        ]

        next_pending: List[tuple[int, str]] = []
        for batch_num, job_batch in enumerate(job_batches, start=1):
            expression_batch = [expr for _, expr in job_batch]
            padded = False
            if len(expression_batch) == 1:
                expression_batch = [expression_batch[0], _padding_expression()]
                padded = True

            batch_results = await simulate_expression_batch(
                expression_batch,
                batch_num,
                len(job_batches),
            )
            core_results = batch_results[:len(job_batch)]
            explicit_failure_seen = any(
                (not result.get("success")) and not _is_cancelled_sim_result(result)
                for result in core_results
            )

            for (idx, _expr), result in zip(job_batch, core_results):
                if result.get("success"):
                    results_by_index[idx] = result
                    continue

                can_retry_cancelled = (
                    _is_cancelled_sim_result(result)
                    and retry_pass < retry_limit
                    and (explicit_failure_seen or len(job_batch) > 1)
                    and not padded
                )
                if can_retry_cancelled:
                    next_pending.append((idx, _expr))
                else:
                    results_by_index[idx] = result

        if not next_pending:
            break
        logger.info(
            f"[{node_name}] Retrying cancelled simulations | "
            f"pass={retry_pass + 1}/{retry_limit} count={len(next_pending)}"
        )
        pending_jobs = next_pending
        retry_pass += 1

    results = [
        results_by_index.get(idx, {"success": False, "error": "Missing simulation result"})
        for idx in indices_to_simulate
    ]
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    # Update alphas
    updated_alphas = state.pending_alphas.copy()
    success_count = 0
    
    for i, idx in enumerate(indices_to_simulate):
        res = results[i] if i < len(results) else {"success": False, "error": "Missing result"}
        
        current = updated_alphas[idx]
        updated = current.model_copy()
        
        updated.is_simulated = True
        updated.simulation_success = res.get("success", False)
        updated.alpha_id = res.get("alpha_id")
        updated.metrics = {
            **(res.get("metrics", {}) or {}),
            "checks": res.get("checks", []),
            "can_submit": res.get("can_submit", False),
            "failed_checks": res.get("failed_checks", []),
            "pending_checks": res.get("pending_checks", []),
            "passed_checks": res.get("passed_checks", []),
            "stage": res.get("stage"),
            "status": res.get("status"),
            "_settings": res.get("settings", {}),
            "_raw_response": res.get("raw_response", res.get("raw")),
            "_simulation_location": res.get("location"),
        }
        updated.simulation_error = res.get("error")
        
        if updated.simulation_success:
            success_count += 1
        
        updated_alphas[idx] = updated
    
    failed_errors = [
        {"expr": expressions[i][:80], "error": results[i].get("error", "unknown")[:200]}
        for i in range(len(results)) if not results[i].get("success")
    ]
    
    _debug_log("E", "nodes.py:simulate:result", "Simulation complete", {
        "total_to_simulate": len(indices_to_simulate),
        "success": success_count,
        "failed": len(indices_to_simulate) - success_count,
        "db_duplicates_skipped": db_duplicates,
        "duration_ms": duration_ms,
        "success_rate": round(success_count / max(1, len(indices_to_simulate)) * 100, 1),
        "failed_errors": failed_errors[:5]
    })
    
    logger.info(f"[{node_name}] Complete | success={success_count}/{len(indices_to_simulate)} db_skipped={db_duplicates}")
    
    # Experiment tracking
    if EXPERIMENT_TRACKING_ENABLED:
        exp = get_current_experiment()
        if exp:
            exp.metrics.increment("simulation_count", len(indices_to_simulate))
            exp.metrics.record("dedup_skip_rate",
                (db_duplicates / (len(indices_to_simulate) + db_duplicates) * 100)
                if (len(indices_to_simulate) + db_duplicates) > 0 else 0,
                tags={"node": node_name, "region": state.region}
            )
            exp.metrics.record("simulation_success_rate",
                (success_count / len(indices_to_simulate) * 100) if len(indices_to_simulate) > 0 else 0,
                tags={"node": node_name}
            )
    
    trace_update = await record_trace(
        state, trace_service, node_name,
        {
            "batch_size": len(indices_to_simulate),
            "db_duplicates_skipped": db_duplicates,
            "expressions": [e[:50] for e in expressions[:10]]
        },
        {
            "success_count": success_count,
            "simulated_count": len(indices_to_simulate),
            "db_duplicates": db_duplicates,
            "results": [{"id": r.get("alpha_id"), "err": r.get("error")} for r in results[:20]]
        },
        duration_ms,
        "SUCCESS" if success_count > 0 else "PARTIAL_FAILURE"
    )
    
    return {
        "pending_alphas": updated_alphas,
        **trace_update
    }


# =============================================================================
# NODE: Evaluate Quality
# =============================================================================

async def node_evaluate(
    state: MiningState,
    brain: BrainAdapter = None,
    config: RunnableConfig = None
) -> Dict:
    """
    Evaluate alpha quality using multi-objective scoring.
    
    Enhanced with:
    - Two-stage correlation checking
    - BRAIN platform official checks integration (checks 数组)
    - Pyramid multiplier consideration for prioritization
    
    Input State:
        - pending_alphas (with simulation results)
    
    Output Updates:
        - pending_alphas (with quality_status and score)
        - trace_steps
    """
    from backend.alpha_scoring import (
        calculate_alpha_score, 
        should_optimize, 
        get_failed_tests,
        evaluate_with_brain_checks,  # 新增：BRAIN官方检查
    )
    
    start_time = time.time()
    node_name = "EVALUATE"
    
    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    task_config = config.get("configurable", {}).get("task_config", {}) if config else {}
    
    updated_alphas = state.pending_alphas.copy()
    pass_count = 0
    fail_count = 0
    optimize_count = 0
    corr_checks_performed = 0
    corr_checks_skipped = 0
    
    logger.info(f"[{node_name}] Starting two-stage evaluation | count={len(state.pending_alphas)}")
    
    # Thresholds
    thresholds = await _load_thresholds(config)
    sharpe_min = thresholds["sharpe_min"]
    turnover_min = thresholds["turnover_min"]
    turnover_max = thresholds["turnover_max"]
    fitness_min = thresholds["fitness_min"]
    margin_min = thresholds["margin_min"]
    score_pass_threshold = thresholds["score_pass"]
    score_optimize_threshold = thresholds["score_optimize"]
    corr_check_threshold = thresholds["corr_check"]
    reversal_abs_sharpe_min = _as_float(
        task_config.get(
            "sign_reversal_abs_sharpe_min",
            task_config.get("optimization_reversal_abs_sharpe_min", 0.8),
        ),
        0.8,
    )
    
    eval_details = []
    failure_feedback_queue = []
    
    for i, alpha in enumerate(updated_alphas):
        if not alpha.is_simulated or not alpha.simulation_success:
            if alpha.quality_status == "PENDING":
                alpha.quality_status = "FAIL"
            continue
        
        metrics = alpha.metrics or {}
        
        train_sharpe_val = metrics.get("train_sharpe")
        train_fitness_val = metrics.get("train_fitness")
        test_sharpe_val = metrics.get("test_sharpe")
        test_fitness_val = metrics.get("test_fitness")
        
        # 构建完整的 sim_result，包含 BRAIN 返回的 checks
        sim_result = {
            "train": {
                "sharpe": train_sharpe_val if train_sharpe_val is not None else metrics.get("sharpe", 0),
                "fitness": train_fitness_val if train_fitness_val is not None else metrics.get("fitness", 0),
                "turnover": metrics.get("turnover", 0),
                "returns": metrics.get("returns", 0),
            },
            "test": {
                "sharpe": test_sharpe_val if test_sharpe_val is not None else metrics.get("sharpe", 0) * 0.8,
                "fitness": test_fitness_val if test_fitness_val is not None else metrics.get("fitness", 0),
            },
            "is": {
                "sharpe": metrics.get("sharpe", 0),
                "fitness": metrics.get("fitness", 0),
                "turnover": metrics.get("turnover", 0),
                "drawdown": metrics.get("drawdown", 0),
                "margin": metrics.get("margin", 0),
                "longCount": metrics.get("longCount"),
                "shortCount": metrics.get("shortCount"),
                "checks": metrics.get("checks", []),  # BRAIN 官方检查结果
            },
            "riskNeutralized": _risk_neutralized_metrics(metrics),
            "investabilityConstrained": metrics.get("investabilityConstrained", {}),
            "checks": metrics.get("checks", []),  # 顶层也放一份
            "can_submit": metrics.get("can_submit", False),
        }
        
        # 新增：使用 BRAIN 官方检查结果进行快速判断
        brain_eval = evaluate_with_brain_checks(sim_result)
        brain_can_submit = brain_eval.get('can_submit', False)
        brain_failed_checks = brain_eval.get('failed_checks', [])
        pyramid_info = brain_eval.get('pyramid_info', {})
        pyramid_multiplier = pyramid_info.get('multiplier', 1.0)
        
        # Stage 1: Preliminary score WITHOUT correlation
        preliminary_score = calculate_alpha_score(
            sim_result=sim_result,
            prod_corr=0.0,
            self_corr=0.0
        )
        
        sharpe = _as_float(metrics.get("sharpe"))
        turnover = _as_float(metrics.get("turnover"))
        fitness = _as_float(metrics.get("fitness"))
        margin = _as_float(metrics.get("margin"))
        
        # Local hard gates before correlation. Final PASS is decided after prod corr.
        meets_metric_thresholds = (
            sharpe > sharpe_min and
            fitness > fitness_min and
            margin > margin_min and
            turnover > turnover_min and
            turnover < turnover_max
        )
        meets_thresholds = meets_metric_thresholds and len(brain_failed_checks) <= thresholds["ra_fails_max"]
        
        # Stage 2: Correlation check for promising candidates. Keep checking
        # candidates that clear the core metrics even when a robust-universe
        # check fails; their correlation profile is needed to guide the next
        # robustification template instead of treating RA as a blind stop.
        prod_corr = None
        self_corr = None
        needs_corr_check = meets_metric_thresholds
        
        if needs_corr_check and brain and alpha.alpha_id:
            corr_checks_performed += 1
            try:
                self_corr_result = await brain.check_correlation(alpha.alpha_id, check_type="SELF")
                if isinstance(self_corr_result, dict):
                    corr_value = self_corr_result.get("max")
                    if corr_value is not None:
                        self_corr = float(corr_value)
            except Exception as e:
                logger.warning(f"[{node_name}] SELF correlation check failed for {alpha.alpha_id}: {e}")

            if self_corr is not None and self_corr < thresholds["self_corr_max"]:
                try:
                    prod_corr_result = await brain.check_correlation(alpha.alpha_id, check_type="PROD")
                    if isinstance(prod_corr_result, dict):
                        corr_value = prod_corr_result.get("max")
                        if corr_value is not None:
                            prod_corr = float(corr_value)
                except Exception as e:
                    logger.warning(f"[{node_name}] PROD correlation check failed for {alpha.alpha_id}: {e}")
        else:
            corr_checks_skipped += 1
        
        # Final score with correlation penalty
        score = calculate_alpha_score(
            sim_result=sim_result,
            prod_corr=prod_corr or 0.0,
            self_corr=self_corr or 0.0
        )
        
        should_opt, opt_reason = should_optimize(sim_result)
        failed_tests = get_failed_tests(sim_result)
        sign_reversal_candidate = (
            sharpe < 0 and abs(sharpe) >= reversal_abs_sharpe_min
        )
        if sign_reversal_candidate and not should_opt:
            should_opt = True
            opt_reason = (
                f"NEGATIVE_SIGNAL_REVERSAL: Sharpe {sharpe:.2f} has "
                f"absolute strength >= {reversal_abs_sharpe_min:.2f}; test reverse()"
            )
        
        strict_failures = _strict_gate_failures(
            metrics,
            brain_failed_checks,
            prod_corr,
            self_corr,
            thresholds,
            expression=alpha.expression,
            fields=state.fields,
        )
        hard_pass = not strict_failures

        # Determine quality status. Hard production gates are mandatory for PASS.
        if hard_pass:
            alpha.quality_status = "PASS"
            pass_count += 1
        elif should_opt and (score >= score_optimize_threshold or sign_reversal_candidate):
            alpha.quality_status = "OPTIMIZE"
            optimize_count += 1
        else:
            alpha.quality_status = "FAIL"
            fail_count += 1
            
            # Enhanced: Alignment check and attribution for failures
            # This helps distinguish hypothesis failure from implementation failure
            alignment_issues = []
            attribution = "unknown"
            
            # Get hypothesis from alpha if available
            hypothesis_dict = {}
            if hasattr(alpha, 'hypothesis') and alpha.hypothesis:
                if isinstance(alpha.hypothesis, dict):
                    hypothesis_dict = alpha.hypothesis
                else:
                    hypothesis_dict = {"statement": alpha.hypothesis}
            
            # Quick alignment check
            if hypothesis_dict and alpha.expression:
                is_aligned, alignment_issues = quick_alignment_check(
                    hypothesis_dict, alpha.expression, state.fields
                )
                
                # Determine attribution
                result_dict = {
                    "success": False,
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "turnover": turnover,
                }
                attribution = determine_attribution_heuristic(
                    result_dict, alignment_issues, alpha.validation_error
                )
                
                if not is_aligned:
                    logger.debug(
                        f"[{node_name}] Alignment issues for {alpha.alpha_id}: {alignment_issues[:2]}"
                    )
            
            # Determine error type
            error_type = "QUALITY_FAIL"
            if sign_reversal_candidate:
                error_type = "NEGATIVE_SIGNAL"
            elif sharpe < sharpe_min:
                error_type = "LOW_SHARPE"
            elif fitness < fitness_min:
                error_type = "LOW_FITNESS"
            elif margin <= margin_min:
                error_type = "LOW_MARGIN"
            elif turnover <= turnover_min:
                error_type = "LOW_TURNOVER"
            elif turnover > turnover_max:
                error_type = "HIGH_TURNOVER"
            elif brain_failed_checks:
                error_type = "RA_CHECK_FAIL"
            elif prod_corr is None or prod_corr >= thresholds["prod_corr_max"]:
                error_type = "PROD_CORR_FAIL"
            elif sharpe < 0:
                error_type = "NEGATIVE_SIGNAL"
            
            if alpha.expression:
                failure_feedback_queue.append({
                    "expression": alpha.expression,
                    "error_type": error_type,
                    "metrics": metrics,
                    "region": state.region,
                    "dataset_id": state.dataset_id,
                    # New: attribution info for knowledge filtering
                    "hypothesis": hypothesis_dict.get("statement", ""),
                    "alignment_issues": alignment_issues,
                    "attribution": attribution,
                })
        
        # Store detailed metrics with BRAIN checks info
        alpha.metrics = {
            **metrics,
            "two_year_sharpe": two_year_value if (two_year_value := _check_value(metrics, "LOW_2Y_SHARPE")) is not None else metrics.get("two_year_sharpe"),
            "riskNeutralized": _risk_neutralized_metrics(metrics),
            "_candidate_metadata": alpha.metadata or {},
            "_score": round(score, 4),
            "_preliminary_score": round(preliminary_score, 4),
            "_prod_corr": round(prod_corr, 4) if prod_corr is not None else None,
            "_self_corr": round(self_corr, 4) if self_corr is not None else None,
            "_corr_checked": needs_corr_check,
            "_should_optimize": should_opt,
            "_optimize_reason": opt_reason,
            "_sign_reversal_candidate": sign_reversal_candidate,
            "_failed_tests": failed_tests,
            "_strict_gate_failures": strict_failures,
            "_hard_pass": hard_pass,
            # BRAIN 官方检查信息
            "_brain_can_submit": brain_can_submit,
            "_brain_failed_checks": brain_failed_checks,
            "_brain_pending_checks": brain_eval.get('pending_checks', []),
            "_ra_fails_count": len(brain_failed_checks),
            "_pyramid_multiplier": pyramid_multiplier,
        }
        
        _debug_log("F", "nodes.py:evaluate:alpha_detail", f"Alpha evaluated: {alpha.quality_status}", {
            "alpha_id": alpha.alpha_id,
            "expression": alpha.expression[:80] if alpha.expression else None,
            "sharpe": round(sharpe, 3),
            "fitness": round(fitness, 3),
            "turnover": round(turnover, 3),
            "score": round(score, 3),
            "status": alpha.quality_status
        })
        
        eval_details.append({
            "id": alpha.alpha_id,
            "status": alpha.quality_status,
            "score": round(score, 4),
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
            "margin": margin,
            "prod_corr": prod_corr,
            "ra_fails": len(brain_failed_checks),
            "corr_checked": needs_corr_check,
            "optimize_reason": opt_reason if should_opt else None,
            "sign_reversal_candidate": sign_reversal_candidate,
            "strict_gate_failures": strict_failures,
        })
        
        updated_alphas[i] = alpha
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    _debug_log("E", "nodes.py:evaluate:result", "Evaluation complete", {
        "pass": pass_count,
        "optimize": optimize_count,
        "fail": fail_count,
        "corr_checked": corr_checks_performed,
        "corr_skipped": corr_checks_skipped,
        "duration_ms": duration_ms,
        "pass_rate": round(pass_count / max(1, pass_count + optimize_count + fail_count) * 100, 1)
    })
    
    logger.info(
        f"[{node_name}] Complete | pass={pass_count} optimize={optimize_count} fail={fail_count} "
        f"corr_checked={corr_checks_performed} corr_skipped={corr_checks_skipped}"
    )
    
    # Experiment tracking
    if EXPERIMENT_TRACKING_ENABLED:
        exp = get_current_experiment()
        if exp:
            exp.metrics.increment("pass_count", pass_count)
            exp.metrics.record("iteration_duration_ms", duration_ms, tags={"node": node_name})
            
            total_evaluated = pass_count + optimize_count + fail_count
            if total_evaluated > 0:
                exp.metrics.record("pass_rate", pass_count / total_evaluated * 100, tags={"region": state.region})
            
            total_corr = corr_checks_performed + corr_checks_skipped
            if total_corr > 0:
                exp.metrics.record("corr_check_skip_rate",
                    corr_checks_skipped / total_corr * 100,
                    tags={"node": node_name}
                )
    
    # Record failure feedback with attribution-aware filtering
    if failure_feedback_queue:
        rag_service = config.get("configurable", {}).get("rag_service") if config else None
        if rag_service:
            feedback_recorded = 0
            hypothesis_failures = 0
            implementation_failures = 0
            
            sample_size = min(3, len(failure_feedback_queue))
            sampled_failures = random.sample(failure_feedback_queue, sample_size)
            
            for feedback in sampled_failures:
                attribution = feedback.get("attribution", "unknown")
                
                # Track attribution stats
                if attribution == "hypothesis":
                    hypothesis_failures += 1
                elif attribution == "implementation":
                    implementation_failures += 1
                
                try:
                    # Only record to knowledge base if attribution is confident
                    # Implementation failures shouldn't teach us about hypotheses
                    should_record = attribution != "implementation"
                    
                    if should_record:
                        await rag_service.record_failure_pattern(
                            expression=feedback["expression"],
                            error_type=feedback["error_type"],
                            metrics=feedback["metrics"],
                            region=feedback["region"],
                            dataset_id=feedback["dataset_id"]
                        )
                        feedback_recorded += 1
                    else:
                        logger.debug(
                            f"[{node_name}] Skipping knowledge record for implementation failure: "
                            f"{feedback['alignment_issues'][:2] if feedback.get('alignment_issues') else 'N/A'}"
                        )
                except Exception as e:
                    logger.warning(f"[{node_name}] Failed to record feedback: {e}")
            
            logger.info(
                f"[{node_name}] Knowledge feedback | recorded={feedback_recorded}/{len(failure_feedback_queue)} "
                f"(hypothesis_fail={hypothesis_failures} impl_fail={implementation_failures})"
            )
    
    trace_update = await record_trace(
        state, trace_service, node_name,
        {
            "evaluation_mode": "two_stage_correlation",
            "thresholds": {
                "sharpe_min": sharpe_min,
                "fitness_min": fitness_min,
                "margin_min": margin_min,
                "turnover_min": turnover_min,
                "turnover_max": turnover_max,
                "prod_corr_max": thresholds["prod_corr_max"],
                "ra_fails_max": thresholds["ra_fails_max"],
                "score_pass": score_pass_threshold,
                "corr_check_threshold": corr_check_threshold,
            }
        },
        {
            "pass_count": pass_count,
            "optimize_count": optimize_count,
            "fail_count": fail_count,
            "corr_checks_performed": corr_checks_performed,
            "corr_checks_skipped": corr_checks_skipped,
            "details": eval_details[:20]
        },
        duration_ms,
        "SUCCESS"
    )
    
    return {
        "pending_alphas": updated_alphas,
        **trace_update
    }
