"""
Evaluation nodes for LangGraph workflow.

Enhanced with hypothesis-implementation alignment checking:
- Verifies implementations correctly reflect hypotheses
- Attributes failures to hypothesis vs implementation
- Filters knowledge based on attribution confidence

Enhanced with diversity constraints:
- Intra-batch similarity checking to avoid redundant alphas
- Expression structure deduplication based on operator patterns

Contains:
- node_simulate: Batch simulate alphas on BRAIN platform
- node_evaluate: Evaluate alpha quality using multi-objective scoring
"""

import time
import random
import re
from typing import Dict, List, Optional, Tuple, Set
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
    
    try:
        from backend.database import AsyncSessionLocal
        from backend.selection_strategy import filter_unsimulated_expressions
        
        expressions_to_check = [state.pending_alphas[i].expression for i in valid_indices]
        
        async with AsyncSessionLocal() as db:
            new_exprs, dup_exprs = await filter_unsimulated_expressions(
                db, expressions_to_check, state.region, state.universe
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
    
    try:
        results = await brain.simulate_batch(
            expressions=expressions,
            region=state.region,
            universe=state.universe,
            delay=1,
            decay=4,
            neutralization="SUBINDUSTRY"
        )
    except Exception as e:
        logger.error(f"[{node_name}] Batch Simulate Loop Error: {e}")
        results = [{"success": False, "error": str(e)} for _ in expressions]
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    # Update alphas
    updated_alphas = state.pending_alphas.copy()
    success_count = 0
    
    for i, idx in enumerate(indices_to_simulate):
        res = results[i] if i < len(results) else {"success": False, "error": "Missing result"}
        
        current = updated_alphas[idx]
        updated = current.model_copy()
        
        # Merge Brain response into candidate metrics so downstream evaluation can use:
        # - platform checks (PASS/FAIL/PENDING)
        # - can_submit signal
        # - pyramids / competitions metadata (optional)
        base_metrics = res.get("metrics", {}) or {}
        merged_metrics = dict(base_metrics) if isinstance(base_metrics, dict) else {}
        
        # Keep a compact subset of top-level Brain fields for evaluation/observability.
        # NOTE: we intentionally do NOT store the full raw response to avoid bloating DB/state.
        for key in (
            "checks",
            "can_submit",
            "failed_checks",
            "pending_checks",
            "passed_checks",
            "pyramids",
            "themes",
            "competitions",
            "stage",
            "status",
            "type",
        ):
            if key in res and res.get(key) is not None:
                merged_metrics[key] = res.get(key)
        
        # Also preserve settings used by the platform for this simulation if present.
        if "settings" in res and res.get("settings") is not None:
            merged_metrics["_brain_settings"] = res.get("settings")
        
        updated.is_simulated = True
        updated.simulation_success = res.get("success", False)
        updated.alpha_id = res.get("alpha_id")
        updated.metrics = merged_metrics
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
# DIVERSITY CHECKING UTILITIES
# =============================================================================

def _extract_structure_fingerprint(expression: str) -> str:
    """
    Extract structural fingerprint from an alpha expression.
    
    This normalizes the expression to capture its "shape" while ignoring
    specific field names and numeric parameters. Used for diversity checking.
    
    Example:
        "ts_decay_linear(ts_rank(analyst_est_rev_up_30d, 10), 5)"
        -> "ts_decay_linear(ts_rank(FIELD, N), N)"
    """
    if not expression:
        return ""
    
    expr = expression.lower().strip()
    
    # Replace field names (identifiers that aren't operators)
    operators = {
        'ts_rank', 'ts_delta', 'ts_zscore', 'ts_mean', 'ts_std_dev', 
        'ts_decay_linear', 'ts_decay_exp', 'ts_sum', 'ts_max', 'ts_min',
        'ts_arg_max', 'ts_arg_min', 'ts_returns', 'ts_product', 'ts_corr',
        'vec_sum', 'vec_avg', 'vec_count', 'vec_max', 'vec_min', 'vec_norm',
        'group_neutralize', 'group_rank', 'group_mean', 'group_sum',
        'rank', 'zscore', 'scale', 'truncate', 'winsorize',
        'divide', 'add', 'subtract', 'multiply', 'log', 'abs', 'sign',
        'power', 'sqrt', 'exp', 'sigmoid', 'min', 'max', 'if', 'else',
    }
    
    # Replace numbers with N placeholder
    expr = re.sub(r'\b\d+\.?\d*\b', 'N', expr)
    
    # Replace identifiers that aren't operators with FIELD
    def replace_identifier(match):
        word = match.group(0)
        if word in operators or word in ('n', 'field'):
            return word
        return 'FIELD'
    
    expr = re.sub(r'\b[a-z_][a-z0-9_]*\b', replace_identifier, expr)
    
    # Remove whitespace for consistent comparison
    expr = re.sub(r'\s+', '', expr)
    
    return expr


def _calculate_expression_similarity(expr1: str, expr2: str) -> float:
    """
    Calculate similarity between two alpha expressions.
    
    Combines structural similarity with token overlap.
    Returns a value between 0 (completely different) and 1 (identical).
    """
    if not expr1 or not expr2:
        return 0.0
    
    # Structural fingerprint similarity (weighted higher)
    fp1 = _extract_structure_fingerprint(expr1)
    fp2 = _extract_structure_fingerprint(expr2)
    
    if fp1 == fp2:
        structural_sim = 1.0
    else:
        # Token-based similarity on fingerprints
        tokens1 = set(re.findall(r'[a-z_]+', fp1))
        tokens2 = set(re.findall(r'[a-z_]+', fp2))
        
        if not tokens1 or not tokens2:
            structural_sim = 0.0
        else:
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            structural_sim = len(intersection) / len(union)
    
    # Token similarity on original expressions (for field overlap)
    tokens1 = set(re.findall(r'[a-z_][a-z0-9_]*', expr1.lower()))
    tokens2 = set(re.findall(r'[a-z_][a-z0-9_]*', expr2.lower()))
    
    if not tokens1 or not tokens2:
        token_sim = 0.0
    else:
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        token_sim = len(intersection) / len(union)
    
    # Weighted combination: structure is more important
    return 0.6 * structural_sim + 0.4 * token_sim


def _check_batch_diversity(
    alphas: List,
    similarity_threshold: float = 0.85
) -> List[Tuple[int, str]]:
    """
    Check intra-batch diversity and identify duplicates.
    
    Returns list of (index, reason) tuples for alphas that are duplicates.
    """
    duplicates = []
    n = len(alphas)
    
    # Track which expressions we've already kept
    kept_indices: Set[int] = set()
    
    for i in range(n):
        alpha_i = alphas[i]
        expr_i = getattr(alpha_i, 'expression', '') or ''
        
        if not expr_i:
            continue
        
        # Check against previously kept alphas
        is_duplicate = False
        for j in kept_indices:
            alpha_j = alphas[j]
            expr_j = getattr(alpha_j, 'expression', '') or ''
            
            similarity = _calculate_expression_similarity(expr_i, expr_j)
            
            if similarity >= similarity_threshold:
                duplicates.append((
                    i, 
                    f"Similar to alpha {j} (similarity={similarity:.2f})"
                ))
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept_indices.add(i)
    
    return duplicates


# =============================================================================
# NODE: Evaluate Quality
# =============================================================================

async def node_evaluate(
    state: MiningState,
    brain: BrainAdapter = None,
    rag_service=None,
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
    rag_service_from_config = config.get("configurable", {}).get("rag_service") if config else None
    rag_service = rag_service or rag_service_from_config
    
    updated_alphas = state.pending_alphas.copy()
    pass_count = 0
    fail_count = 0
    optimize_count = 0
    promising_count = 0
    duplicate_count = 0
    corr_checks_performed = 0
    corr_checks_skipped = 0
    
    logger.info(f"[{node_name}] Starting two-stage evaluation | count={len(state.pending_alphas)}")
    
    # === DIVERSITY CHECK (P3: Add diversity constraints) ===
    # Check for intra-batch duplicates before evaluation
    diversity_threshold = getattr(settings, 'DIVERSITY_SIMILARITY_THRESHOLD', 0.85)
    duplicates = _check_batch_diversity(updated_alphas, similarity_threshold=diversity_threshold)
    
    duplicate_indices = {idx for idx, _ in duplicates}
    
    if duplicates:
        logger.info(f"[{node_name}] Diversity check found {len(duplicates)} similar alphas")
        _debug_log("F", "nodes.py:evaluate:diversity", "Diversity check results", {
            "total_alphas": len(updated_alphas),
            "duplicates_found": len(duplicates),
            "duplicate_details": [
                {"index": idx, "reason": reason, "expr": (updated_alphas[idx].expression or "")[:80]}
                for idx, reason in duplicates[:5]
            ]
        })
    
    # Thresholds
    sharpe_min = getattr(settings, 'SHARPE_MIN', 1.5)
    turnover_max = getattr(settings, 'TURNOVER_MAX', 0.7)
    fitness_min = getattr(settings, 'FITNESS_MIN', 0.6)
    score_pass_threshold = getattr(settings, 'SCORE_PASS_THRESHOLD', 0.8)
    score_optimize_threshold = getattr(settings, 'SCORE_OPTIMIZE_THRESHOLD', 0.3)
    corr_check_threshold = getattr(settings, 'CORR_CHECK_THRESHOLD', 0.5)
    
    eval_details = []
    failure_feedback_queue = []
    
    for i, alpha in enumerate(updated_alphas):
        # Check for duplicates first
        if i in duplicate_indices:
            alpha.quality_status = "DUPLICATE"
            duplicate_count += 1
            # Find the duplicate reason
            dup_reason = next((r for idx, r in duplicates if idx == i), "Similar to another alpha")
            alpha.metrics = {
                **(alpha.metrics or {}),
                "_duplicate": True,
                "_duplicate_reason": dup_reason
            }
            updated_alphas[i] = alpha
            continue
        
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
                "longCount": metrics.get("longCount"),
                "shortCount": metrics.get("shortCount"),
                "checks": metrics.get("checks", []),  # BRAIN 官方检查结果
            },
            "riskNeutralized": metrics.get("riskNeutralized", {}),
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
        
        sharpe = metrics.get("sharpe", 0) or 0
        turnover = metrics.get("turnover", 0) or 0
        fitness = metrics.get("fitness", 0) or 0
        
        # 使用 BRAIN 官方检查或本地阈值
        if brain_eval['check_details']:
            # 有官方检查结果，以官方为准
            # NOTE: pending checks should NOT be treated as passable.
            meets_thresholds = brain_can_submit
        else:
            # Fallback: 使用本地阈值
            meets_thresholds = (
                sharpe >= sharpe_min and
                turnover <= turnover_max and
                fitness >= fitness_min
            )
        
        # Stage 2: Correlation check for promising candidates
        prod_corr = 0.0
        self_corr = 0.0
        needs_corr_check = (
            preliminary_score >= corr_check_threshold or
            meets_thresholds
        )
        
        if needs_corr_check and brain and alpha.alpha_id:
            corr_checks_performed += 1
            try:
                prod_corr_result = await brain.check_correlation(alpha.alpha_id, check_type="PROD")
                if isinstance(prod_corr_result, dict):
                    prod_corr = float(prod_corr_result.get("max", 0.0) or 0.0)
            except Exception as e:
                logger.warning(f"[{node_name}] PROD correlation check failed for {alpha.alpha_id}: {e}")
            
            try:
                self_corr_result = await brain.check_correlation(alpha.alpha_id, check_type="SELF")
                if isinstance(self_corr_result, dict):
                    self_corr = float(self_corr_result.get("max", 0.0) or 0.0)
            except Exception as e:
                logger.warning(f"[{node_name}] SELF correlation check failed for {alpha.alpha_id}: {e}")
        else:
            corr_checks_skipped += 1
        
        # Final score with correlation penalty
        score = calculate_alpha_score(
            sim_result=sim_result,
            prod_corr=prod_corr,
            self_corr=self_corr
        )
        
        should_opt, opt_reason = should_optimize(sim_result)
        failed_tests = get_failed_tests(sim_result)
        
        # ---------------------------------------------------------------------
        # Two-tier gating:
        # - PASS: submit-ready (BRAIN can_submit + correlation constraints)
        # - PROMISING: worth keeping for research/optimization, but not submit-ready
        # - OPTIMIZE: explicitly queued for optimization (should_optimize + score floor)
        # - FAIL: everything else
        # ---------------------------------------------------------------------
        max_corr = getattr(settings, "MAX_CORRELATION", 0.7)
        corr_ok = True
        if prod_corr is not None and prod_corr > max_corr:
            corr_ok = False
        if self_corr is not None and self_corr > max_corr:
            corr_ok = False

        submit_ready = bool(brain_can_submit) and corr_ok

        if submit_ready:
            alpha.quality_status = "PASS"
            pass_count += 1
        elif should_opt and score >= score_optimize_threshold:
            alpha.quality_status = "OPTIMIZE"
            optimize_count += 1
        elif meets_thresholds or score >= score_pass_threshold:
            alpha.quality_status = "PROMISING"
            # PROMISING counts as non-fail, but not as submit-ready success.
            promising_count += 1
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
            if sharpe < sharpe_min:
                error_type = "LOW_SHARPE"
            elif fitness < fitness_min:
                error_type = "LOW_FITNESS"
            elif turnover > turnover_max:
                error_type = "HIGH_TURNOVER"
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
            "_score": round(score, 4),
            "_preliminary_score": round(preliminary_score, 4),
            "_prod_corr": round(prod_corr, 4) if prod_corr else None,
            "_self_corr": round(self_corr, 4) if self_corr else None,
            "_corr_ok": corr_ok,
            "_submit_ready": submit_ready,
            "_corr_checked": needs_corr_check,
            "_should_optimize": should_opt,
            "_optimize_reason": opt_reason,
            "_failed_tests": failed_tests,
            # BRAIN 官方检查信息
            "_brain_can_submit": brain_can_submit,
            "_brain_failed_checks": brain_failed_checks,
            "_brain_pending_checks": brain_eval.get('pending_checks', []),
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
            "corr_checked": needs_corr_check,
            "optimize_reason": opt_reason if should_opt else None,
        })
        
        updated_alphas[i] = alpha
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    _debug_log("E", "nodes.py:evaluate:result", "Evaluation complete", {
        "pass": pass_count,
        "promising": promising_count,
        "optimize": optimize_count,
        "fail": fail_count,
        "duplicate": duplicate_count,
        "corr_checked": corr_checks_performed,
        "corr_skipped": corr_checks_skipped,
        "duration_ms": duration_ms,
        "pass_rate": round(pass_count / max(1, pass_count + promising_count + optimize_count + fail_count) * 100, 1),
        "diversity_efficiency": round((1 - duplicate_count / max(1, len(updated_alphas))) * 100, 1)
    })
    
    logger.info(
        f"[{node_name}] Complete | pass={pass_count} promising={promising_count} optimize={optimize_count} "
        f"fail={fail_count} duplicate={duplicate_count} corr_checked={corr_checks_performed}"
    )
    
    # Experiment tracking
    if EXPERIMENT_TRACKING_ENABLED:
        exp = get_current_experiment()
        if exp:
            exp.metrics.increment("pass_count", pass_count)
            exp.metrics.increment("promising_count", promising_count)
            exp.metrics.record("iteration_duration_ms", duration_ms, tags={"node": node_name})
            
            total_evaluated = pass_count + promising_count + optimize_count + fail_count
            if total_evaluated > 0:
                exp.metrics.record("pass_rate", pass_count / total_evaluated * 100, tags={"region": state.region})
            
            total_corr = corr_checks_performed + corr_checks_skipped
            if total_corr > 0:
                exp.metrics.record("corr_check_skip_rate",
                    corr_checks_skipped / total_corr * 100,
                    tags={"node": node_name}
                )
    
    # Record success/failure feedback to knowledge base (CoSTEER loop)
    if rag_service:
        # 1) Record a small sample of failure patterns (avoid KB spam)
        if failure_feedback_queue:
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
        
        # 2) Record success patterns (usually rare, so safe to record all)
        try:
            pass_candidates = [
                a for a in updated_alphas
                if getattr(a, "quality_status", None) == "PASS" and getattr(a, "expression", None)
            ]
            # Cap to prevent excessive writes if thresholds are lax
            for a in pass_candidates[:5]:
                m = getattr(a, "metrics", {}) or {}
                await rag_service.record_success_pattern(
                    expression=a.expression,
                    metrics=m,
                    region=state.region,
                    dataset_id=state.dataset_id,
                    alpha_id=getattr(a, "alpha_id", None),
                )
        except Exception as e:
            logger.warning(f"[{node_name}] Failed to record success patterns: {e}")
    
    trace_update = await record_trace(
        state, trace_service, node_name,
        {
            "evaluation_mode": "two_stage_correlation",
            "thresholds": {
                "sharpe_min": sharpe_min,
                "turnover_max": turnover_max,
                "fitness_min": fitness_min,
                "score_pass": score_pass_threshold,
                "corr_check_threshold": corr_check_threshold,
            }
        },
        {
            "pass_count": pass_count,
            "promising_count": promising_count,
            "optimize_count": optimize_count,
            "fail_count": fail_count,
            "duplicate_count": duplicate_count,
            "corr_checks_performed": corr_checks_performed,
            "corr_checks_skipped": corr_checks_skipped,
            "diversity_efficiency": round((1 - duplicate_count / max(1, len(updated_alphas))) * 100, 1),
            "details": eval_details[:20]
        },
        duration_ms,
        "SUCCESS"
    )
    
    return {
        "pending_alphas": updated_alphas,
        **trace_update
    }
