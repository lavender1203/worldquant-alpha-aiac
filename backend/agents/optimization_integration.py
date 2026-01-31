"""
Optimization Integration Module

This module integrates the AlphaOptimizer into the existing LangGraph workflow.
It provides enhanced nodes that replace or wrap existing nodes.

Key integrations:
1. Pre-generation field validation (P0-1)
2. Post-generation alignment enforcement (P0-2) 
3. Signal direction auto-correction (P0-3)
4. CoSTEER hard constraint checking (P1-1)
5. GP-style variant generation (P1-2)
6. Smart exploration parameters (P1-3)

Usage:
    from backend.agents.optimization_integration import (
        enhanced_pre_generation_check,
        enhanced_post_generation_validate,
        auto_correct_signal_direction,
        generate_optimization_variants,
        get_smart_exploration_params,
    )
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from backend.alpha_optimizer import (
    get_alpha_optimizer,
    AlphaOptimizer,
    FieldPreCheckResult,
    AlignmentCheckResult,
    ExpressionVariant,
)


# =============================================================================
# P0-1: Enhanced Pre-Generation Check
# =============================================================================

def enhanced_pre_generation_check(
    fields: List[Dict],
    region: str,
    universe: str,
    hypothesis_key_fields: Optional[List[str]] = None,
    dataset_id: str = ""
) -> Tuple[List[Dict], FieldPreCheckResult]:
    """
    P0-1: Enhanced pre-generation field check.
    
    Call this BEFORE code generation node to filter unavailable fields.
    
    Args:
        fields: All available fields from dataset
        region: Market region
        universe: Universe
        hypothesis_key_fields: Key fields from hypothesis (mandatory)
        dataset_id: Dataset ID for logging
        
    Returns:
        (filtered_fields, check_result)
    """
    optimizer = get_alpha_optimizer()
    
    result = optimizer.pre_generate_check(
        fields=fields,
        region=region,
        universe=universe,
        hypothesis_key_fields=hypothesis_key_fields
    )
    
    logger.info(
        f"[PreGenCheck] Dataset={dataset_id} | "
        f"available={len(result.available_fields)} blocked={len(result.blocked_fields)}"
    )
    
    if result.warning_message:
        logger.warning(f"[PreGenCheck] {result.warning_message}")
    
    # Filter fields to only available ones
    available_ids = set(result.available_fields)
    filtered_fields = [f for f in fields if f.get("id", f.get("name", "")) in available_ids]
    
    return filtered_fields, result


# =============================================================================
# P0-2 + P1-1 + P2-2: Enhanced Post-Generation Validation
# =============================================================================

def enhanced_post_generation_validate(
    expression: str,
    hypothesis: Dict[str, Any],
    all_fields: List[Dict]
) -> Tuple[bool, str, List[str]]:
    """
    P0-2 + P1-1 + P2-2: Enhanced post-generation validation.
    
    Call this AFTER code generation to validate and potentially correct expressions.
    
    This combines:
    - Hypothesis-implementation alignment check (P0-2)
    - CoSTEER hard constraint check (P1-1)
    - Multi-fidelity pre-screening (P2-2)
    
    Args:
        expression: Generated alpha expression
        hypothesis: Hypothesis dict with statement, key_fields, etc.
        all_fields: All available field dicts
        
    Returns:
        (is_valid, corrected_expression, issues)
    """
    optimizer = get_alpha_optimizer()
    
    available_field_names = [f.get("id", f.get("name", "")) for f in all_fields]
    
    is_valid, corrected, issues = optimizer.post_generate_validate(
        expression=expression,
        hypothesis=hypothesis,
        available_fields=available_field_names
    )
    
    if not is_valid:
        logger.warning(
            f"[PostGenValidate] REJECTED | expr={expression[:60]}... | issues={issues[:2]}"
        )
    elif corrected != expression:
        logger.info(
            f"[PostGenValidate] AUTO-CORRECTED | {expression[:40]}... -> {corrected[:40]}..."
        )
    
    return is_valid, corrected, issues


# =============================================================================
# P0-3: Signal Direction Auto-Correction
# =============================================================================

def should_try_signal_inversion(
    expression: str,
    sharpe: float,
    fitness: float,
    turnover: float
) -> Tuple[bool, Optional[str], str]:
    """
    P0-3: Check if signal should be inverted based on simulation metrics.
    
    Call this AFTER simulation for expressions with negative Sharpe.
    
    Args:
        expression: The alpha expression
        sharpe: In-sample Sharpe ratio
        fitness: In-sample fitness
        turnover: Turnover ratio
        
    Returns:
        (should_invert, inverted_expression, reason)
    """
    optimizer = get_alpha_optimizer()
    
    should_invert, inverted_expr, reason = optimizer.check_and_correct_signal(
        expression=expression,
        sharpe=sharpe,
        fitness=fitness,
        turnover=turnover
    )
    
    if should_invert:
        logger.info(
            f"[SignalCorrect] Recommending inversion | sharpe={sharpe:.2f} | {reason}"
        )
    
    return should_invert, inverted_expr, reason


async def try_inverted_simulation(
    brain_adapter,
    expression: str,
    region: str,
    universe: str,
    original_sharpe: float
) -> Optional[Dict]:
    """
    P0-3: Try simulating inverted expression if original had negative Sharpe.
    
    This is an optional enhancement that can rescue alphas with inverted signals.
    
    Args:
        brain_adapter: BrainAdapter instance
        expression: Original expression
        region: Market region
        universe: Universe
        original_sharpe: Original Sharpe ratio (should be negative)
        
    Returns:
        Inverted simulation result if successful and better, None otherwise
    """
    if original_sharpe >= 0:
        return None
    
    optimizer = get_alpha_optimizer()
    
    # Create inverted expression
    inverted_expr = optimizer.signal_corrector.create_inverted_expression(expression)
    
    try:
        # Simulate inverted
        results = await brain_adapter.simulate_batch(
            expressions=[inverted_expr],
            region=region,
            universe=universe,
            delay=1,
            decay=4,
            neutralization="SUBINDUSTRY"
        )
        
        if results and len(results) > 0 and results[0].get("success"):
            inverted_metrics = results[0].get("metrics", {})
            inverted_sharpe = float(inverted_metrics.get("sharpe", 0) or 0)
            
            # Check if inverted is better
            if inverted_sharpe > original_sharpe and inverted_sharpe > 0:
                logger.info(
                    f"[SignalCorrect] Inversion successful! "
                    f"{original_sharpe:.2f} -> {inverted_sharpe:.2f}"
                )
                return {
                    "expression": inverted_expr,
                    "result": results[0],
                    "improvement": inverted_sharpe - original_sharpe
                }
        
        return None
        
    except Exception as e:
        logger.warning(f"[SignalCorrect] Inversion simulation failed: {e}")
        return None


# =============================================================================
# P1-1: CoSTEER Feedback Integration
# =============================================================================

def update_costeer_from_round(
    successes: List[Dict],
    failures: List[Dict],
    dataset_id: str,
    passed_count: int,
    best_sharpe: Optional[float]
):
    """
    P1-1: Update CoSTEER constraints after a mining round.
    
    Call this after each round completes to update hard constraints.
    
    Args:
        successes: List of successful alpha records
        failures: List of failed alpha records  
        dataset_id: Dataset ID
        passed_count: Number of passed alphas
        best_sharpe: Best Sharpe in this round
    """
    optimizer = get_alpha_optimizer()
    
    optimizer.update_from_round(
        successes=successes,
        failures=failures,
        dataset_id=dataset_id,
        passed_count=passed_count,
        best_sharpe=best_sharpe
    )
    
    stats = optimizer.get_stats()
    logger.info(
        f"[CoSTEER] Updated | forbidden_fields={stats['costeer_constraints']['forbidden_fields']} "
        f"forbidden_patterns={stats['costeer_constraints']['forbidden_patterns']}"
    )


def get_costeer_hard_constraints() -> Dict[str, Any]:
    """
    P1-1: Get current CoSTEER hard constraints for prompt injection.
    
    Returns constraints that MUST be respected in code generation.
    """
    optimizer = get_alpha_optimizer()
    constraints = optimizer.costeer_enforcer.get_constraints()
    
    return {
        "forbidden_fields": list(constraints.forbidden_fields),
        "forbidden_patterns": constraints.forbidden_patterns[:10],
        "require_decay_wrapper": constraints.require_decay_wrapper,
        "min_decay_window": constraints.min_decay_window,
        "max_expression_depth": constraints.max_expression_depth,
    }


# =============================================================================
# P1-2: GP Enhancement - Variant Generation
# =============================================================================

def generate_optimization_variants(
    seed_expression: str,
    num_variants: int = 5,
    include_inversions: bool = True
) -> List[Dict]:
    """
    P1-2: Generate GP-style variants for optimization.
    
    Call this for OPTIMIZE-status alphas to generate variations.
    
    Args:
        seed_expression: Base expression to vary
        num_variants: Number of variants to generate
        include_inversions: Include signal inversions
        
    Returns:
        List of variant dicts with expression, modification, etc.
    """
    optimizer = get_alpha_optimizer()
    
    variants = optimizer.generate_optimization_variants(
        seed_expression=seed_expression,
        num_variants=num_variants
    )
    
    logger.info(f"[GPEnhance] Generated {len(variants)} variants from seed")
    
    return [
        {
            "expression": v.expression,
            "modification": v.modification,
            "parent": v.parent_expression,
            "expected_improvement": v.expected_improvement,
        }
        for v in variants
    ]


# =============================================================================
# P1-3: Smart Exploration Strategy
# =============================================================================

def get_smart_exploration_params(
    current_progress: float,
    max_iterations: int,
    iteration: int = 0
) -> Dict[str, Any]:
    """
    P1-3: Get smart exploration strategy parameters.
    
    Call this at the start of each iteration to get adaptive parameters.
    
    Args:
        current_progress: Progress towards goal (0-1)
        max_iterations: Maximum iterations in run
        iteration: Current iteration number
        
    Returns:
        Strategy parameters dict with temperature, exploration_weight, etc.
    """
    optimizer = get_alpha_optimizer()
    
    params = optimizer.get_exploration_parameters(
        current_progress=current_progress,
        max_iterations=max_iterations
    )
    
    logger.debug(
        f"[SmartExplore] iter={iteration} | action={params['recommended_action']} | "
        f"temp={params['temperature']:.2f} explore={params['exploration_weight']:.2f}"
    )
    
    return params


def should_rotate_dataset_smart(
    current_dataset: str,
    consecutive_failures: int
) -> Tuple[bool, str, Optional[str]]:
    """
    P1-3: Smart dataset rotation decision.
    
    Args:
        current_dataset: Current dataset ID
        consecutive_failures: Consecutive failures on this dataset
        
    Returns:
        (should_rotate, reason, suggested_dataset)
    """
    optimizer = get_alpha_optimizer()
    
    should_rotate, reason = optimizer.exploration_strategy.should_rotate_dataset(
        current_dataset=current_dataset,
        consecutive_failures_this_dataset=consecutive_failures
    )
    
    suggested = None
    if should_rotate:
        # Get available datasets from exploration state
        available = list(optimizer.exploration_strategy.state.dataset_attempts.keys())
        if available:
            suggested = optimizer.exploration_strategy.select_best_dataset(
                available_datasets=available,
                current_dataset=current_dataset
            )
    
    return should_rotate, reason, suggested


# =============================================================================
# P2-1: Knowledge Graph Integration
# =============================================================================

def get_pattern_context_for_generation(
    dataset_id: str,
    region: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    P2-1: Get pattern context from knowledge graph for code generation.
    
    Returns successful patterns and failure pitfalls adapted to current context.
    """
    optimizer = get_alpha_optimizer()
    
    # Get successful patterns from exploration state
    successful_patterns = optimizer.exploration_strategy.state.successful_patterns[-top_k:]
    
    # Get patterns to avoid (recently failed)
    patterns_to_avoid = list(optimizer.exploration_strategy.state.explored_patterns)[-top_k * 2:]
    
    return {
        "successful_patterns": successful_patterns,
        "patterns_to_avoid": patterns_to_avoid,
        "total_explored": len(optimizer.exploration_strategy.state.explored_patterns),
    }


# =============================================================================
# P2-2: Multi-Fidelity Pre-Screening
# =============================================================================

def pre_screen_expressions(
    expressions: List[str],
    hypothesis_key_fields: Optional[List[str]] = None
) -> List[Tuple[str, bool, List[str]]]:
    """
    P2-2: Pre-screen multiple expressions before simulation.
    
    This filters out obviously bad expressions to save simulation budget.
    
    Args:
        expressions: List of expressions to screen
        hypothesis_key_fields: Key fields that should be used
        
    Returns:
        List of (expression, should_simulate, issues) tuples
    """
    optimizer = get_alpha_optimizer()
    
    results = []
    for expr in expressions:
        screen = optimizer.multi_fidelity_screener.pre_screen(
            expression=expr,
            hypothesis_key_fields=hypothesis_key_fields or []
        )
        results.append((expr, screen.should_simulate, screen.issues))
    
    filtered_count = sum(1 for _, sim, _ in results if not sim)
    if filtered_count > 0:
        logger.info(f"[PreScreen] Filtered {filtered_count}/{len(expressions)} expressions")
    
    return results


# =============================================================================
# Unified Enhancement Entry Point
# =============================================================================

class OptimizedMiningEnhancements:
    """
    Unified class that provides all optimization enhancements.
    
    This can be used as a drop-in replacement for scattered optimization calls.
    """
    
    def __init__(self):
        self._optimizer = get_alpha_optimizer()
        
    def enhance_generation_input(
        self,
        fields: List[Dict],
        hypotheses: List[Dict],
        region: str,
        universe: str,
        dataset_id: str
    ) -> Dict[str, Any]:
        """
        Enhance inputs for code generation.
        
        Returns enhanced context dict to pass to generation node.
        """
        # Pre-check fields
        hypothesis_key_fields = []
        for h in hypotheses:
            hypothesis_key_fields.extend(h.get("key_fields", []))
        
        filtered_fields, field_check = enhanced_pre_generation_check(
            fields=fields,
            region=region,
            universe=universe,
            hypothesis_key_fields=hypothesis_key_fields,
            dataset_id=dataset_id
        )
        
        # Get CoSTEER constraints
        costeer = get_costeer_hard_constraints()
        
        # Get pattern context
        patterns = get_pattern_context_for_generation(dataset_id, region)
        
        return {
            "filtered_fields": filtered_fields,
            "field_check_result": {
                "blocked_count": len(field_check.blocked_fields),
                "warning": field_check.warning_message,
                "recommended_fields": field_check.recommended_fields,
            },
            "costeer_constraints": costeer,
            "pattern_context": patterns,
        }
    
    def validate_generated_alphas(
        self,
        alphas: List[Dict],
        hypotheses: List[Dict],
        all_fields: List[Dict]
    ) -> List[Dict]:
        """
        Validate and enhance generated alphas.
        
        Returns enhanced alpha list with validation status.
        """
        enhanced_alphas = []
        
        for alpha in alphas:
            expr = alpha.get("expression", "")
            hypo_text = alpha.get("hypothesis", "")
            
            # Find matching hypothesis
            matching_hypo = None
            for h in hypotheses:
                h_text = h.get("statement", h.get("idea", ""))
                if h_text and (h_text in hypo_text or hypo_text in h_text):
                    matching_hypo = h
                    break
            
            hypothesis_dict = matching_hypo or {"statement": hypo_text, "key_fields": []}
            
            # Validate
            is_valid, corrected, issues = enhanced_post_generation_validate(
                expression=expr,
                hypothesis=hypothesis_dict,
                all_fields=all_fields
            )
            
            enhanced_alpha = {
                **alpha,
                "expression": corrected,
                "_is_valid": is_valid,
                "_validation_issues": issues,
                "_original_expression": expr if corrected != expr else None,
            }
            
            if is_valid:
                enhanced_alphas.append(enhanced_alpha)
            else:
                logger.debug(f"[Validate] Rejected: {expr[:50]}... | {issues[:1]}")
        
        return enhanced_alphas
    
    def post_simulation_enhancements(
        self,
        simulated_alphas: List[Dict],
        brain_adapter
    ) -> List[Dict]:
        """
        Apply post-simulation enhancements.
        
        Includes signal direction correction for negative Sharpe alphas.
        """
        enhanced = []
        
        for alpha in simulated_alphas:
            metrics = alpha.get("metrics", {})
            sharpe = float(metrics.get("sharpe", 0) or 0)
            fitness = float(metrics.get("fitness", 0) or 0)
            turnover = float(metrics.get("turnover", 0) or 0)
            
            # Check if signal inversion might help
            should_invert, inverted_expr, reason = should_try_signal_inversion(
                expression=alpha.get("expression", ""),
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover
            )
            
            if should_invert and inverted_expr:
                alpha["_inversion_suggested"] = True
                alpha["_inverted_expression"] = inverted_expr
                alpha["_inversion_reason"] = reason
            
            enhanced.append(alpha)
        
        return enhanced
    
    def finalize_round(
        self,
        successes: List[Dict],
        failures: List[Dict],
        dataset_id: str,
        best_sharpe: Optional[float]
    ):
        """
        Finalize round and update all tracking systems.
        """
        update_costeer_from_round(
            successes=successes,
            failures=failures,
            dataset_id=dataset_id,
            passed_count=len(successes),
            best_sharpe=best_sharpe
        )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return self._optimizer.get_stats()


# Global instance
_enhancements: Optional[OptimizedMiningEnhancements] = None


def get_mining_enhancements() -> OptimizedMiningEnhancements:
    """Get or create global enhancements instance."""
    global _enhancements
    if _enhancements is None:
        _enhancements = OptimizedMiningEnhancements()
    return _enhancements
