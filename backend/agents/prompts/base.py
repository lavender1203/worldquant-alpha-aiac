"""
Base components for prompt building.

Contains:
- PromptContext data class
- Helper functions for building context sections
"""

from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class PromptContext:
    """Structured context for prompt rendering."""
    dataset_id: str = ""
    dataset_description: str = ""
    dataset_category: str = ""
    region: str = "USA"
    universe: str = "TOP3000"
    
    # Available data (will be JSON serialized)
    fields: List[Dict] = field(default_factory=list)
    operators: List[Dict] = field(default_factory=list)
    
    # Knowledge base context
    success_patterns: List[Dict] = field(default_factory=list)
    failure_pitfalls: List[Dict] = field(default_factory=list)
    
    # Strategy guidance (from StrategyAgent)
    preferred_fields: List[str] = field(default_factory=list)
    avoid_fields: List[str] = field(default_factory=list)
    focus_hypotheses: List[str] = field(default_factory=list)
    avoid_patterns: List[str] = field(default_factory=list)
    preferred_operators: List[str] = field(default_factory=list)
    avoid_operators: List[str] = field(default_factory=list)
    
    # Generation parameters
    num_alphas: int = 5
    exploration_weight: float = 0.5  # 0=pure exploitation, 1=pure exploration
    max_operator_count: int = 5


def build_fields_context(fields: List[Dict], max_fields: int = 30) -> str:
    """Build concise field reference with type info."""
    if not fields:
        return "No fields available."
    
    matrix_fields = []
    vector_fields = []
    other_fields = []
    
    visible_fields = fields[:max_fields]
    valid_field_ids = []

    for f in visible_fields:
        field_id = f.get("id", f.get("name", "unknown"))
        valid_field_ids.append(str(field_id))
        field_type = (f.get("type") or f.get("field_type") or "MATRIX").upper()
        field_text = " ".join(
            str(f.get(key) or "")
            for key in ("id", "name", "field_id", "field_name", "description")
        ).lower()
        group_like = (
            field_type == "GROUP"
            or any(token in field_text for token in ("group", "bucket", "cluster", "classification", "category"))
        )
        
        if group_like:
            other_fields.append(field_id)
        elif field_type == "VECTOR":
            vector_fields.append(field_id)
        elif field_type == "MATRIX":
            matrix_fields.append(field_id)
        else:
            other_fields.append(field_id)
    
    lines = []

    if valid_field_ids:
        lines.append(
            "- **Valid field IDs only**: "
            + ", ".join(valid_field_ids)
            + ". Do not invent fields outside this list."
        )
    
    if matrix_fields:
        sample = ", ".join(matrix_fields[:10])
        if len(matrix_fields) > 10:
            sample += f" ... (+{len(matrix_fields) - 10} more)"
        lines.append(f"- **MATRIX fields** (time-series, use ts_* operators directly): {sample}")
    
    if vector_fields:
        sample = ", ".join(vector_fields[:10])
        if len(vector_fields) > 10:
            sample += f" ... (+{len(vector_fields) - 10} more)"
        lines.append(f"- **VECTOR fields** (MUST use vec_* operators first!): {sample}")
        lines.append(
            "- VECTOR-safe skeleton examples: rank(vec_sum(x)); "
            "ts_delta(vec_sum(x), 5); ts_mean(vec_avg(x), 20); "
            "subtract(vec_avg(x), vec_avg(y)); "
            "subtract(vec_sum(x), ts_mean(vec_sum(x), 20)). "
            "Avoid reverse(vec_*(x)) and never put subtract/multiply/divide inside vec_*."
        )
    
    if other_fields:
        sample = ", ".join(other_fields[:5])
        lines.append(f"- Other/group-like fields (avoid numeric ts_* transforms unless explicitly using group operators): {sample}")
    
    return "\n".join(lines)


def build_operators_context(operators: List[Dict], max_ops: int = 40) -> str:
    """Build operator reference grouped by category."""
    if not operators:
        return "Use standard operators."
    
    by_category: Dict[str, List[str]] = {}
    for op in operators[:max_ops]:
        cat = op.get("category", "Other")
        if cat not in by_category:
            by_category[cat] = []
        op_name = op.get("name", op.get("id", "unknown"))
        by_category[cat].append(op_name)
    
    lines = []
    for cat, op_names in sorted(by_category.items()):
        lines.append(f"- {cat}: {', '.join(op_names[:10])}")
    
    return "\n".join(lines)


def build_patterns_context(patterns: List[Dict], label: str, max_items: int = 5) -> str:
    """Build pattern reference without implying they must be followed."""
    if not patterns:
        return f"No {label} recorded yet."
    
    lines = []
    for p in patterns[:max_items]:
        pattern = p.get("pattern", p.get("template", ""))
        desc = p.get("description", "")
        if pattern:
            lines.append(f"- `{pattern}`: {desc[:80]}")
    
    return "\n".join(lines) if lines else f"No {label} recorded yet."


def build_strategy_constraints(ctx: PromptContext) -> str:
    """Build strategy-driven constraints without being prescriptive."""
    constraints = []
    strategy_text = " ".join(ctx.focus_hypotheses + ctx.avoid_patterns).lower()
    
    if ctx.avoid_fields:
        constraints.append(
            f"Fields with recent issues (consider alternatives): {', '.join(ctx.avoid_fields[:5])}"
        )
    
    if ctx.avoid_patterns:
        constraints.append(
            f"Patterns that underperformed recently: {'; '.join(ctx.avoid_patterns[:3])}"
        )

    if ctx.preferred_fields:
        constraints.append(
            f"Prefer these fields when economically sensible: {', '.join(ctx.preferred_fields[:8])}"
        )

    if ctx.focus_hypotheses:
        constraints.append(
            f"Current economic/strategy focus: {'; '.join(ctx.focus_hypotheses[:6])}"
        )

    if any(
        token in strategy_text
        for token in (
            "high turnover",
            "turnover is high",
            "low-turnover",
            "lower-turnover",
            "smooth",
            "smoother",
            "smoothing",
            "long window",
            "longer window",
            "低换手",
            "换手高",
            "降低换手",
            "平滑",
            "更长窗口",
            "长期均值",
        )
    ):
        constraints.append(
            "**LOW-TURNOVER REFINEMENT RULE**: When the strategy asks for smoothing, "
            "lower turnover, or longer windows, do not emit raw one-day deltas such as "
            "`ts_delta(field, 1)` or raw field interactions like `multiply(field_a, field_b)`. "
            "Use smoothed structures such as `rank(ts_mean(field, 12))`, "
            "`rank(ts_zscore(field, 20))`, `rank(ts_delta(ts_mean(field, 12), 4))`, "
            "or a ranked spread of smoothed fields."
        )

    if ctx.preferred_operators:
        constraints.append(
            f"Prefer underexplored operators when they fit the idea: {', '.join(ctx.preferred_operators[:8])}"
        )

    if ctx.avoid_operators:
        constraints.append(
            f"Avoid these operators unless explicitly required: {', '.join(ctx.avoid_operators[:8])}"
        )
    
    # CRITICAL TYPE CONSTRAINTS
    constraints.append(
        "**VECTOR FIELD RULE**: VECTOR-type fields MUST be processed with vec_* operators "
        "(vec_sum, vec_avg, vec_max, vec_min, vec_count, vec_range, vec_stddev, etc.) "
        "BEFORE using ts_* operators. Example: ts_rank(vec_sum(vector_field), 20) - NOT ts_rank(vector_field, 20)"
    )
    constraints.append(
        "**EVENT/VECTOR OPERATOR RULE**: Do not wrap VECTOR/event-derived signals with reverse(). "
        "Some BRAIN event inputs reject reverse even after vector aggregation. Prefer sign-safe forms "
        "such as rank(vec_sum(x)), ts_delta(vec_sum(x), d), ts_mean(vec_avg(x), d), "
        "subtract(vec_sum(x), ts_mean(vec_sum(x), d)), or multiply(signal, -1) when inversion is essential."
    )
    constraints.append(
        "**EVENT/VECTOR ARITHMETIC RULE**: Arithmetic operators cannot consume raw event/VECTOR fields. "
        "Aggregate each event field first, then combine the aggregated MATRIX signals. "
        "Valid: subtract(vec_avg(event_a), vec_avg(event_b)). "
        "Invalid: vec_avg(subtract(event_a, event_b)) or vec_sum(multiply(event_a, event_b))."
    )
    constraints.append(
        "**MATRIX FIELD RULE**: MATRIX-type fields can use ts_* operators directly. "
        "Example: ts_rank(matrix_field, 20)"
    )
    
    # Syntax constraints (always apply)
    constraints.extend([
        "Lookback windows must be positive integers",
        "Use at most 2 distinct data fields per expression",
        f"Use fewer than {ctx.max_operator_count + 1} operator calls; target {ctx.max_operator_count} or fewer",
        "Do not use trade_when",
        "Avoid tail() unless you provide both lower and upper constants; tail(x, lower) is invalid",
        "Use scale(x) with one positional argument only; scale(x, 1) is invalid on BRAIN",
        "Use clamp(x, lower=..., upper=...) with named bounds; clamp(x, 0, 0.1) is invalid on BRAIN",
        "Use canonical operator names exactly: ts_std_dev, group_std_dev, vec_stddev; do not use ts_stddev or ts_std",
        "Do not use diagnostic/meta operators as signals: self_corr, inst_pnl, generate_stats",
        "Do not force ratio/spread templates; choose the expression family that fits the factor style",
        "Within a batch, diversify mechanism, operator skeleton, parameter window, and signal direction",
        "Ensure no look-ahead bias (no future data access)"
    ])
    
    return "\n".join(f"- {c}" for c in constraints)


def build_factor_construction_context(ctx: PromptContext) -> str:
    """Build mechanism-level construction guidance for stronger template diversity."""
    region_hint = ""
    if ctx.region.upper() == "IND":
        region_hint = (
            "\n- IND/D1 often rewards compact, risk-aware structures: group_rank or "
            "group_neutralize within industry/subindustry, residualizing a fundamental, "
            "analyst, sentiment, news, model, or other signal against returns/risk/liquidity. "
            "Prefer settings-level neutralization sweeps over adding many expression operators."
        )

    return f"""
## Factor Construction Playbook

Use this as a mechanism library, not as formulas to copy blindly:
- **Comparable scale construction**: convert raw levels into comparable quantities before ranking. Examples of ideas: `a / b`, `(a - c) / b`, `a - ts_mean(a,d)`, `a - ts_median(a,d)`.
- **Risk or crowding residual construction**: compare the primary field with a risk, return, volatility, liquidity, or crowding proxy. Keep this to one primary field plus one control field.
- **Center and extreme deviation**: test distance from a rolling center or extreme, such as mean/median/max/min deviations, when the field carries regime information.
- **Second-order change**: when the idea is acceleration or stability, test `ts_delta(ts_delta(x,d1),d2)` or a short-window change of a smoothed value.
- **State-frequency thinking**: when the signal is event-like, use counts, abnormal-state intensity, or threshold-state persistence rather than raw event magnitude.
- **Low-operator discipline**: prefer one clean transformation plus one normalization/grouping step over nested wrappers.
{region_hint}
"""
