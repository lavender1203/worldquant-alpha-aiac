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

# Neutralization options. Keep the platform families used by non-USA regions
# here because optimization can only exploit settings it is allowed to sweep.
NEUTRALIZATION_OPTIONS = [
    "NONE",
    "MARKET",
    "SECTOR",
    "INDUSTRY",
    "SUBINDUSTRY",
    "CROWDING",
    "FAST",
    "SLOW",
    "SLOW_AND_FAST",
    "REVERSION_AND_MOMENTUM",
]

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
    variants.extend(_generate_second_order_probe_variants(expression, sim_result, context))

    if priorities.get("sign", False):
        variants.extend(_generate_sign_variants(expression, context))
    
    if priorities.get("window", False):
        variants.extend(_generate_window_variants(expression, context))
    
    if priorities.get("wrapper", False):
        variants.extend(_generate_wrapper_variants(expression, context))
    
    if priorities.get("structure", False):
        variants.extend(_generate_structure_variants(expression, context))

    if priorities.get("frequency", False):
        variants.extend(_generate_frequency_variants(expression, context))

    # Sort by priority, but keep variants that can pass the strict IND operator
    # budget ahead of cosmetically higher-priority wrappers that will be skipped
    # before simulation.
    variants = _dedupe_variants(variants)
    variants.sort(
        key=lambda v: (
            _operator_call_count(v.expression) > 5,
            -v.priority,
            _operator_call_count(v.expression),
        )
    )

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
    if any(key in sim_result for key in ("train", "is", "test", "os")):
        train = sim_result.get('train', sim_result.get('is', {})) or {}
        test = sim_result.get('test', sim_result.get('os', {})) or {}
    else:
        train = sim_result or {}
        test = {
            "sharpe": sim_result.get("test_sharpe", sim_result.get("os_sharpe")),
            "fitness": sim_result.get("test_fitness", sim_result.get("os_fitness")),
        }
    rn = sim_result.get('riskNeutralized', {}) or {}
    if not isinstance(rn, dict):
        rn = {}
    if rn.get("sharpe") is None:
        rn = {
            **rn,
            "sharpe": sim_result.get("risk_neutralized_sharpe", sim_result.get("rn_sharpe")),
        }
    if rn.get("fitness") is None:
        rn = {
            **rn,
            "fitness": sim_result.get("risk_neutralized_fitness", sim_result.get("rn_fitness")),
        }
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


def _operator_call_count(expression: str) -> int:
    """Count FASTEXPR function calls in a candidate expression."""
    return len(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", expression or ""))


def _dedupe_variants(variants: List[OptimizationVariant]) -> List[OptimizationVariant]:
    """Keep the highest-priority instance of each generated expression."""
    best_by_expr: Dict[str, OptimizationVariant] = {}
    for variant in variants:
        key = re.sub(r"\s+", "", variant.expression or "").lower()
        if not key:
            continue
        current = best_by_expr.get(key)
        if current is None or variant.priority > current.priority:
            best_by_expr[key] = variant
    return list(best_by_expr.values())


def _determine_optimization_priorities(context: OptimizationContext) -> Dict[str, bool]:
    """Determine which optimization types to prioritize based on context."""
    priorities = {
        "sign": True,  # Always try sign flip
        "window": True,  # Always try window adjustment
        "wrapper": True,  # Always try wrapper changes
        "structure": False,  # Only for specific cases
        "frequency": False,  # Low-turnover alphas need more active structures
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

    # Low turnover is a hard gate in strict mining. Wrappers usually preserve
    # low turnover, so add mechanism-level frequency variants instead.
    if 0 < context.turnover < 0.055:
        priorities["frequency"] = True
        priorities["window"] = True
    
    return priorities


def _generate_sign_variants(
    expression: str, 
    context: OptimizationContext
) -> List[OptimizationVariant]:
    """Generate sign flip and monotonic transform variants."""
    variants = []
    
    # Negative sign (signal reversal). Use arithmetic inversion rather than
    # reverse() so VECTOR-derived scalar signals can still test direction.
    reversed_expression = f"multiply(-1, {expression})"
    variants.append(OptimizationVariant(
        expression=reversed_expression,
        change_type=OptimizationType.SIGN_FLIP,
        description='Signal reversal',
        rationale='The tested relation may be directionally inverted',
        priority=10 if context.train_sharpe <= -0.30 else 2,
    ))

    if context.train_sharpe <= -0.30 and not expression.startswith("rank("):
        variants.append(OptimizationVariant(
            expression=f"rank({reversed_expression})",
            change_type=OptimizationType.SIGN_FLIP,
            description='Ranked signal reversal',
            rationale='Flip direction and normalize cross-sectionally',
            priority=8,
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
    
    # Find window parameters in known time-series functions. Matching each
    # function directly avoids a greedy outer function match such as
    # add(rank(ts_mean(x,20)), ...) swallowing the nested ts_mean window.
    matches = []
    for func in sorted(WINDOW_FUNCTIONS, key=len, reverse=True):
        window_pattern = rf'\b({re.escape(func)})\s*\(\s*([^,]+)\s*,\s*(\d+)'
        matches.extend(re.finditer(window_pattern, expression))
    matches.sort(key=lambda match: match.start())
    
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


def _generate_frequency_variants(
    expression: str,
    context: OptimizationContext
) -> List[OptimizationVariant]:
    """Generate variants that can lift too-low turnover without adding noise blindly."""
    variants = []

    field = _first_field_like_token(expression)
    if not field:
        return variants

    candidate_exprs = [
        (
            f"rank(ts_delta({field}, 1))",
            "One-day change impulse",
            "Low-turnover level signal may need a fresher change component",
            8,
        ),
        (
            f"rank(ts_delta(ts_mean({field}, 5), 1))",
            "Five-day smoothed change impulse",
            "Raises trading activity while damping one-day field noise",
            7,
        ),
        (
            f"rank(ts_zscore({field}, 10))",
            "Short-window field surprise",
            "Tests whether recent abnormal level carries more active alpha",
            6,
        ),
        (
            f"zscore(ts_mean({field}, 10))",
            "Shorter level window",
            "A shorter level window can lift turnover from sub-5% regimes",
            5,
        ),
    ]

    seen = {expression.strip()}
    for expr, description, rationale, priority in candidate_exprs:
        if expr in seen:
            continue
        seen.add(expr)
        variants.append(OptimizationVariant(
            expression=expr,
            change_type=OptimizationType.STRUCTURE_VARIATION,
            description=description,
            rationale=rationale,
            priority=priority,
        ))

    return variants


def _generate_second_order_probe_variants(
    expression: str,
    sim_result: Dict,
    context: OptimizationContext,
) -> List[OptimizationVariant]:
    """Add exactly one strengthening operator to a weak first-order probe.

    The IND mining workflow first maps standalone REGULAR operators. When a
    probe has weak signal, the next step should be a controlled second-order
    scan rather than a broad rewrite. These variants deliberately wrap the
    tested expression with one additional operator so attribution remains clear.
    """
    candidate_meta = sim_result.get("_candidate_metadata") if isinstance(sim_result, dict) else {}
    if not isinstance(candidate_meta, dict):
        candidate_meta = {}
    if candidate_meta.get("source") != "first_order_operator_probe":
        return []

    base_ops = _operator_call_count(expression)
    if base_ops >= 5:
        return []

    probe_op = str(candidate_meta.get("probe_operator") or "probe").lower()
    variants: List[OptimizationVariant] = []

    def add(expr: str, description: str, rationale: str, priority: int) -> None:
        if _operator_call_count(expr) > 5:
            return
        variants.append(OptimizationVariant(
            expression=expr,
            change_type=OptimizationType.STRUCTURE_VARIATION,
            description=description,
            rationale=rationale,
            priority=priority,
        ))

    add(
        f"group_rank({expression}, industry)",
        f"Second-order industry rank after {probe_op}",
        "Tests whether peer-relative normalization strengthens the weak first-order signal",
        12,
    )
    add(
        f"group_rank({expression}, subindustry)",
        f"Second-order subindustry rank after {probe_op}",
        "Tests whether narrower peer ranking improves signal purity",
        11,
    )
    add(
        f"rank({expression})",
        f"Second-order cross-sectional rank after {probe_op}",
        "Normalizes the standalone operator output without changing the core signal",
        10,
    )
    add(
        f"ts_delta({expression}, 5)",
        f"Second-order short change after {probe_op}",
        "Adds one turnover-lifting time-series change operator to the weak signal",
        14 if 0 < context.turnover < 0.055 else (9 if context.turnover < 0.08 else 7),
    )
    add(
        f"ts_zscore({expression}, 20)",
        f"Second-order time zscore after {probe_op}",
        "Tests whether recent surprise in the operator output is more predictive",
        8,
    )
    add(
        f"group_neutralize({expression}, industry)",
        f"Second-order industry neutralization after {probe_op}",
        "Removes broad industry exposure while preserving the operator mechanism",
        7,
    )
    add(
        f"signed_power({expression}, 2)",
        f"Second-order signed power after {probe_op}",
        "Amplifies stronger cross-sectional differences from the weak signal",
        6,
    )

    if context.train_sharpe <= -0.30:
        add(
            f"multiply(-1, {expression})",
            f"Second-order sign flip after {probe_op}",
            "The first-order probe was directionally negative, so test the inverted relation",
            13,
        )

    return variants


def _first_field_like_token(expression: str) -> Optional[str]:
    """Best-effort extraction of the first data field token from a FastExpr string."""
    function_tokens = {
        match.group(1).lower()
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression or "")
    }
    operators = {
        "abs", "add", "and", "bucket", "densify", "divide", "equal", "group_neutralize",
        "if_else", "inverse", "log", "max", "min", "multiply", "not_equal", "or",
        "rank", "reverse", "scale", "sign", "subtract", "ts_argmax", "ts_argmin",
        "ts_corr", "ts_count_nans", "ts_covariance", "ts_decay_linear", "ts_delta",
        "ts_ir", "ts_kurtosis", "ts_max", "ts_mean", "ts_min", "ts_product",
        "ts_rank", "ts_regression", "ts_returns", "ts_skewness", "ts_std_dev",
        "ts_sum", "ts_zscore", "vec_avg", "vec_count", "vec_sum", "winsorize",
        "zscore", "std", "sector", "industry", "subindustry", "market",
    }
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression or ""):
        lower = token.lower()
        if lower in function_tokens or lower in operators:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        return token
    return None


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
