"""
Optimization Chain Module

Implements the "Factor Optimization Chain" from Chain-of-Alpha methodology.
Takes weak/failed alphas + backtest feedback and generates local rewrites
to iteratively improve performance.

Design Principles:
1. Targeted Modifications: Small changes, not complete rewrites
2. Budget-Aware: Limit simulation calls per optimization target
3. Evidence-Based: Modifications driven by specific failure signals
4. Composable: Can be integrated into any workflow

Reference: 优化.md Section 3.3, 3.4
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class OptimizationType(Enum):
    """Types of optimization modifications."""
    SIGN_FLIP = "sign_flip"
    WINDOW_SWEEP = "window_sweep"
    WRAPPER_ADD = "wrapper_add"
    WRAPPER_REMOVE = "wrapper_remove"
    STRUCTURE_VARIATION = "structure_variation"
    SETTINGS_NEUTRALIZATION = "settings_neutralization"
    SETTINGS_DECAY = "settings_decay"
    SETTINGS_TRUNCATION = "settings_truncation"


# Window values for parameter sweep (trading days)
WINDOW_OPTIONS = [5, 10, 22, 44, 66, 126, 252]

# Decay values for settings sweep
DECAY_OPTIONS = [0, 2, 4, 8, 16]

# Neutralization options
NEUTRALIZATION_OPTIONS = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]

# Truncation options
TRUNCATION_OPTIONS = [0.01, 0.02, 0.05, 0.08, 0.10]

# Functions that typically have window parameters
WINDOW_FUNCTIONS = {
    'ts_mean', 'ts_delta', 'ts_zscore', 'ts_std_dev',
    'ts_rank', 'ts_ir', 'ts_returns', 'ts_decay_linear',
    'ts_sum', 'ts_min', 'ts_max', 'ts_argmax', 'ts_argmin',
    'ts_corr', 'ts_covariance', 'ts_skewness', 'ts_kurtosis'
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OptimizationVariant:
    """Represents a single optimization variant."""
    expression: str
    change_type: OptimizationType
    description: str
    rationale: str = ""
    priority: int = 0  # Higher = more likely to help
    
    def to_dict(self) -> Dict:
        return {
            "expression": self.expression,
            "change_type": self.change_type.value,
            "description": self.description,
            "rationale": self.rationale,
            "priority": self.priority,
        }


@dataclass
class SettingsVariant:
    """Represents a settings-level optimization variant."""
    neutralization: str
    decay: int
    truncation: float
    change_type: OptimizationType
    description: str
    
    def to_dict(self) -> Dict:
        return {
            "neutralization": self.neutralization,
            "decay": self.decay,
            "truncation": self.truncation,
            "change_type": self.change_type.value,
            "description": self.description,
        }


@dataclass
class OptimizationContext:
    """Context for optimization decision-making."""
    expression: str
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    rn_sharpe: float = 0.0  # Risk-neutralized Sharpe
    invest_sharpe: float = 0.0  # Investability-constrained Sharpe
    failed_tests: List[str] = field(default_factory=list)
    optimize_reason: str = ""


# =============================================================================
# CORE OPTIMIZATION FUNCTIONS
# =============================================================================

def generate_local_rewrites(
    expression: str,
    sim_result: Dict,
    feedback: Optional[str] = None,
    max_variants: int = 20
) -> List[Dict]:
    """
    Generate local rewrite variants for an alpha expression based on backtest feedback.
    
    This is the main entry point for expression-level optimization.
    
    Args:
        expression: Original alpha expression
        sim_result: Simulation result from Brain API
        feedback: Optional optimization feedback string
        max_variants: Maximum number of variants to generate
    
    Returns:
        List of variant dictionaries with 'expression', 'change_type', 'description'
    """
    from backend.alpha_scoring import should_optimize, get_failed_tests
    
    variants: List[OptimizationVariant] = []
    
    # Analyze simulation result to determine optimization strategy
    context = _build_optimization_context(expression, sim_result)
    
    # Determine which types of optimizations to prioritize
    priorities = _determine_optimization_priorities(context)
    
    # Generate variants based on priorities
    if priorities.get("sign", False):
        variants.extend(_generate_sign_variants(expression, context))
    
    if priorities.get("window", False):
        variants.extend(_generate_window_variants(expression, context))
    
    if priorities.get("wrapper", False):
        variants.extend(_generate_wrapper_variants(expression, context))
    
    if priorities.get("structure", False):
        variants.extend(_generate_structure_variants(expression, context))
    
    # Sort by priority and limit
    variants.sort(key=lambda v: v.priority, reverse=True)
    
    return [v.to_dict() for v in variants[:max_variants]]


def generate_settings_variants(
    base_settings: Dict,
    context: Optional[OptimizationContext] = None
) -> List[Dict]:
    """
    Generate simulation settings variants (neutralization, decay, truncation).
    
    These are applied at simulation time, not expression level.
    
    Args:
        base_settings: Current simulation settings
        context: Optional optimization context for smart prioritization
    
    Returns:
        List of settings variant dictionaries
    """
    variants: List[SettingsVariant] = []
    
    base_neut = base_settings.get('neutralization', 'INDUSTRY')
    base_decay = base_settings.get('decay', 4)
    base_trunc = base_settings.get('truncation', 0.02)
    
    # Neutralization variants
    for neut in NEUTRALIZATION_OPTIONS:
        if neut != base_neut:
            variants.append(SettingsVariant(
                neutralization=neut,
                decay=base_decay,
                truncation=base_trunc,
                change_type=OptimizationType.SETTINGS_NEUTRALIZATION,
                description=f'Neutralization: {base_neut} -> {neut}'
            ))
    
    # Decay variants
    for decay in DECAY_OPTIONS:
        if decay != base_decay:
            variants.append(SettingsVariant(
                neutralization=base_neut,
                decay=decay,
                truncation=base_trunc,
                change_type=OptimizationType.SETTINGS_DECAY,
                description=f'Decay: {base_decay} -> {decay}'
            ))
    
    # Truncation variants
    for trunc in TRUNCATION_OPTIONS:
        if abs(trunc - base_trunc) > 0.001:
            variants.append(SettingsVariant(
                neutralization=base_neut,
                decay=base_decay,
                truncation=trunc,
                change_type=OptimizationType.SETTINGS_TRUNCATION,
                description=f'Truncation: {base_trunc} -> {trunc}'
            ))
    
    # Prioritize based on context if available
    if context:
        variants = _prioritize_settings_variants(variants, context)
    
    return [v.to_dict() for v in variants]


# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================

def _build_optimization_context(expression: str, sim_result: Dict) -> OptimizationContext:
    """Build optimization context from simulation result."""
    from backend.alpha_scoring import should_optimize, get_failed_tests
    
    # Extract metrics with safe defaults
    train = sim_result.get('train', sim_result.get('is', {})) or {}
    test = sim_result.get('test', sim_result.get('os', {})) or {}
    rn = sim_result.get('riskNeutralized', {}) or {}
    invest = sim_result.get('investabilityConstrained', {}) or {}
    
    _, reason = should_optimize(sim_result)
    failed = get_failed_tests(sim_result)
    
    return OptimizationContext(
        expression=expression,
        train_sharpe=_safe_float(train.get('sharpe')),
        test_sharpe=_safe_float(test.get('sharpe')),
        fitness=_safe_float(train.get('fitness')),
        turnover=_safe_float(train.get('turnover')),
        rn_sharpe=_safe_float(rn.get('sharpe')),
        invest_sharpe=_safe_float(invest.get('sharpe')),
        failed_tests=failed,
        optimize_reason=reason,
    )


def _safe_float(value: Any) -> float:
    """Safely convert to float."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _determine_optimization_priorities(context: OptimizationContext) -> Dict[str, bool]:
    """Determine which optimization types to prioritize based on context."""
    priorities = {
        "sign": True,  # Always try sign flip
        "window": True,  # Always try window adjustment
        "wrapper": True,  # Always try wrapper changes
        "structure": False,  # Only for specific cases
    }
    
    reason = context.optimize_reason.lower()
    
    # Risk exposure issue -> prioritize neutralization (handled in settings)
    # but also try structure variations
    if "风险" in reason or "risk" in reason or "neutral" in reason:
        priorities["structure"] = True
    
    # Overfitting (IS >> OS) -> prioritize window/decay
    if "衰减" in reason or "overfit" in reason or "稳定" in reason:
        priorities["window"] = True
    
    # Investability issue -> prioritize wrappers (winsorize, etc.)
    if "投资" in reason or "invest" in reason or "集中" in reason:
        priorities["wrapper"] = True
    
    # Turnover too high -> prioritize window increase
    if "换手" in reason or "turnover" in reason:
        priorities["window"] = True
    
    return priorities


def _generate_sign_variants(
    expression: str, 
    context: OptimizationContext
) -> List[OptimizationVariant]:
    """Generate sign flip and monotonic transform variants."""
    variants = []
    
    # Negative sign (signal reversal)
    variants.append(OptimizationVariant(
        expression=f"-({expression})",
        change_type=OptimizationType.SIGN_FLIP,
        description='Signal reversal',
        rationale='Some signals work in the opposite direction',
        priority=5 if context.train_sharpe < 0 else 2,
    ))
    
    # Absolute value (remove direction, keep magnitude)
    if not expression.startswith('abs('):
        variants.append(OptimizationVariant(
            expression=f"abs({expression})",
            change_type=OptimizationType.SIGN_FLIP,
            description='Absolute value (magnitude only)',
            rationale='Captures magnitude regardless of direction',
            priority=1,
        ))
    
    return variants


def _generate_window_variants(
    expression: str,
    context: OptimizationContext
) -> List[OptimizationVariant]:
    """Generate window parameter sweep variants."""
    variants = []
    
    # Find all window parameters in the expression
    # Pattern: function_name(field, NUMBER) or function_name(field, NUMBER, ...)
    window_pattern = r'(\w+)\s*\(\s*([^,]+)\s*,\s*(\d+)'
    matches = list(re.finditer(window_pattern, expression))
    
    for match in matches:
        func_name = match.group(1)
        original_window = int(match.group(3))
        
        # Skip if not a window function
        if func_name not in WINDOW_FUNCTIONS:
            continue
        
        # Generate variants with different windows
        for new_window in WINDOW_OPTIONS:
            if new_window == original_window:
                continue
            
            # Build new expression with replaced window
            new_expr = (
                expression[:match.start(3)] + 
                str(new_window) + 
                expression[match.end(3):]
            )
            
            # Prioritize based on context
            priority = 3
            if context.turnover > 0.6 and new_window > original_window:
                priority = 8  # Larger window may reduce turnover
            elif context.train_sharpe > context.test_sharpe * 1.5 and new_window > original_window:
                priority = 7  # Larger window may reduce overfitting
            
            variants.append(OptimizationVariant(
                expression=new_expr,
                change_type=OptimizationType.WINDOW_SWEEP,
                description=f'{func_name} window: {original_window} -> {new_window}',
                rationale=f'Window adjustment to capture different signal frequencies',
                priority=priority,
            ))
    
    return variants


def _generate_wrapper_variants(
    expression: str,
    context: OptimizationContext
) -> List[OptimizationVariant]:
    """Generate wrapper function variants (rank, zscore, winsorize)."""
    variants = []
    
    # Check if already has rank wrapper
    if expression.startswith('rank(') and expression.endswith(')'):
        inner = expression[5:-1]
        
        # Try removing rank
        variants.append(OptimizationVariant(
            expression=inner,
            change_type=OptimizationType.WRAPPER_REMOVE,
            description='Remove rank() wrapper',
            rationale='Raw values may carry additional information',
            priority=2,
        ))
        
        # Try replacing rank with zscore
        variants.append(OptimizationVariant(
            expression=f"ts_zscore({inner}, 60)",
            change_type=OptimizationType.WRAPPER_ADD,
            description='Replace rank() with ts_zscore()',
            rationale='Z-score normalization with time-series context',
            priority=3,
        ))
    else:
        # Add rank wrapper
        variants.append(OptimizationVariant(
            expression=f"rank({expression})",
            change_type=OptimizationType.WRAPPER_ADD,
            description='Add rank() wrapper',
            rationale='Cross-sectional ranking for better comparability',
            priority=4,
        ))
        
        # Add zscore wrapper
        variants.append(OptimizationVariant(
            expression=f"ts_zscore({expression}, 60)",
            change_type=OptimizationType.WRAPPER_ADD,
            description='Add ts_zscore() wrapper',
            rationale='Time-series standardization for stability',
            priority=3,
        ))
    
    # Winsorize for concentration issues
    priority = 6 if context.invest_sharpe < context.train_sharpe * 0.7 else 2
    
    if not 'winsorize' in expression.lower():
        variants.append(OptimizationVariant(
            expression=f"winsorize({expression}, std=2)",
            change_type=OptimizationType.WRAPPER_ADD,
            description='Add winsorize() to control concentration',
            rationale='Limits extreme values that may cause investability issues',
            priority=priority,
        ))
    
    return variants


def _generate_structure_variants(
    expression: str,
    context: OptimizationContext
) -> List[OptimizationVariant]:
    """Generate structure-preserving variations."""
    variants = []
    
    # If risk-neutralized is much better, suggest explicit neutralization
    if context.rn_sharpe > context.train_sharpe + 0.3:
        # Try sector/industry neutralization within expression
        variants.append(OptimizationVariant(
            expression=f"group_neutralize({expression}, sector)",
            change_type=OptimizationType.STRUCTURE_VARIATION,
            description='Add explicit sector neutralization',
            rationale='Risk-neutralized performance suggests factor exposure',
            priority=7,
        ))
    
    return variants


def _prioritize_settings_variants(
    variants: List[SettingsVariant],
    context: OptimizationContext
) -> List[SettingsVariant]:
    """Reorder settings variants based on context."""
    # If RN Sharpe >> Train Sharpe, prioritize neutralization changes
    if context.rn_sharpe > context.train_sharpe + 0.25:
        neut_variants = [v for v in variants if v.change_type == OptimizationType.SETTINGS_NEUTRALIZATION]
        other_variants = [v for v in variants if v.change_type != OptimizationType.SETTINGS_NEUTRALIZATION]
        return neut_variants + other_variants
    
    # If high turnover, prioritize decay increase
    if context.turnover > 0.5:
        decay_variants = [v for v in variants if v.change_type == OptimizationType.SETTINGS_DECAY]
        other_variants = [v for v in variants if v.change_type != OptimizationType.SETTINGS_DECAY]
        return decay_variants + other_variants
    
    return variants


# =============================================================================
# LLM-BASED OPTIMIZATION PROMPT (For Advanced Cases)
# =============================================================================

def create_optimization_prompt(
    expression: str,
    sim_result: Dict,
    pool_corr: float = 0.0
) -> str:
    """
    Create LLM prompt for generating optimization suggestions.
    
    Used when rule-based optimization is insufficient.
    """
    from backend.alpha_scoring import get_failed_tests, should_optimize
    
    context = _build_optimization_context(expression, sim_result)
    failed = get_failed_tests(sim_result)
    _, reason = should_optimize(sim_result)
    
    prompt = f"""## Alpha Expression to Optimize

```
{expression}
```

## Backtest Results

| Metric | Value |
|--------|-------|
| Train Sharpe | {context.train_sharpe:.3f} |
| Test Sharpe | {context.test_sharpe:.3f} |
| Fitness | {context.fitness:.3f} |
| Turnover | {context.turnover:.3f} |
| Risk-Neutralized Sharpe | {context.rn_sharpe:.3f} |
| Investability-Constrained Sharpe | {context.invest_sharpe:.3f} |
| Pool Correlation | {pool_corr:.3f} |

## Issues Identified

- **Failed Tests**: {', '.join(failed) if failed else 'None'}
- **Optimization Trigger**: {reason}

## Task

Generate 5-8 targeted modifications that address the identified issues.

**Focus Areas**:
1. If Risk-Neutralized >> Raw: Reduce factor exposure via neutralization or structure
2. If Investability-Constrained << Raw: Add concentration controls (winsorize, truncation)
3. If Test << Train: Increase stability (larger windows, more decay)
4. If Turnover > 0.6: Smooth the signal (larger windows, decay)

**Output** (JSON):
```json
{{
  "analysis": "Brief analysis of what might be wrong",
  "modifications": [
    {{
      "type": "window|wrapper|sign|structure|settings",
      "expression": "Modified expression",
      "rationale": "Why this might help"
    }}
  ]
}}
```"""
    
    return prompt


# =============================================================================
# OPTIMIZATION EXECUTION (For Integration)
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    original_expression: str
    best_variant: Optional[str] = None
    best_score: float = 0.0
    improvement: float = 0.0
    variants_tested: int = 0
    successful: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "original": self.original_expression,
            "best_variant": self.best_variant,
            "best_score": self.best_score,
            "improvement": self.improvement,
            "variants_tested": self.variants_tested,
            "successful": self.successful,
        }


async def run_optimization_chain(
    expression: str,
    sim_result: Dict,
    brain_adapter: Any,
    budget: int = 20,
    settings: Optional[Dict] = None
) -> OptimizationResult:
    """
    Run complete optimization chain on an expression.
    
    This is the main integration point for the mining agent.
    
    Args:
        expression: Original expression to optimize
        sim_result: Original simulation result
        brain_adapter: Brain adapter for simulations
        budget: Maximum simulation budget
        settings: Base simulation settings
    
    Returns:
        OptimizationResult with best variant found
    """
    from backend.alpha_scoring import calculate_alpha_score
    
    result = OptimizationResult(original_expression=expression)
    
    # Calculate original score
    original_score = calculate_alpha_score(sim_result)
    
    # Generate variants
    expr_variants = generate_local_rewrites(expression, sim_result, max_variants=budget // 2)
    settings_variants = generate_settings_variants(settings or {})
    
    best_score = original_score
    best_variant = None
    tested = 0
    
    # Test expression variants first
    for variant in expr_variants[:budget]:
        if tested >= budget:
            break
        
        try:
            var_result = await brain_adapter.simulate_alpha(
                expression=variant['expression'],
                **settings or {}
            )
            
            if var_result.get('success'):
                score = calculate_alpha_score(var_result)
                if score > best_score:
                    best_score = score
                    best_variant = variant['expression']
            
            tested += 1
            
        except Exception as e:
            logger.warning(f"Variant simulation failed: {e}")
            continue
    
    result.best_variant = best_variant
    result.best_score = best_score
    result.improvement = best_score - original_score
    result.variants_tested = tested
    result.successful = best_score > original_score
    
    return result
