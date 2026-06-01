"""
Generation nodes for LangGraph workflow.

Redesigned based on RD-Agent's hypothesis-driven approach:
- Each experiment tests a specific hypothesis
- Knowledge transfer from previous experiments
- Balanced exploration and exploitation
- No preconceived biases

Contains:
- node_rag_query: Retrieve patterns from knowledge base
- node_distill_context: Distill concepts from fields
- node_hypothesis: Generate investment hypotheses
- node_code_gen: Generate alpha expressions
"""

import time
import random
import re
from typing import Dict, List, Optional, Tuple
from loguru import logger
from langchain_core.runnables import RunnableConfig

from backend.agents.graph.state import MiningState, AlphaCandidate
from backend.agents.graph.nodes.base import record_trace, _debug_log
from backend.agents.services import LLMService, RAGService
from backend.agents.prompts import (
    ALPHA_GENERATION_SYSTEM,
    HYPOTHESIS_SYSTEM,
    DISTILL_SYSTEM,
    HYPOTHESIS_USER,
    DISTILL_USER,
    build_alpha_generation_prompt,
    build_hypothesis_prompt,
    build_distill_prompt,
    PromptContext,
)


# =============================================================================
# NODE: RAG Query
# =============================================================================

async def node_rag_query(
    state: MiningState,
    rag_service: RAGService,
    config: RunnableConfig = None
) -> Dict:
    """
    Retrieve success patterns and failure pitfalls from knowledge base.
    
    Input State:
        - dataset_id, region
    
    Output Updates:
        - patterns, pitfalls
        - trace_steps
    """
    start_time = time.time()
    node_name = "RAG_QUERY"
    
    logger.info(f"[{node_name}] Starting | task={state.task_id} dataset={state.dataset_id}")
    
    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    
    try:
        result = await rag_service.query(
            dataset_id=state.dataset_id,
            region=state.region,
            max_patterns=5,
            max_pitfalls=10
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        _debug_log("C", "nodes.py:rag_query:result", "RAG query complete", {
            "patterns_count": len(result.patterns),
            "pitfalls_count": len(result.pitfalls),
            "duration_ms": duration_ms,
            "dataset_id": state.dataset_id,
        })
        
        logger.info(
            f"[{node_name}] Complete | patterns={len(result.patterns)} pitfalls={len(result.pitfalls)}"
        )
        
        trace_update = await record_trace(
            state, trace_service, step_type=node_name,
            input_data={"dataset_id": state.dataset_id, "region": state.region},
            output_data={
                "patterns_count": len(result.patterns),
                "pitfalls_count": len(result.pitfalls),
                "top_patterns": [p['pattern'] for p in result.patterns[:3]],
                "top_pitfalls": [p['pattern'] for p in result.pitfalls[:3]]
            },
            duration_ms=duration_ms,
            status="SUCCESS"
        )
        
        ds_info = result.dataset_info or {}
        description = ds_info.get("description", "")
        category = ds_info.get("category", "Unknown")
        subcategory = ds_info.get("subcategory", "")
        full_category = f"{category} > {subcategory}" if subcategory else category
        
        return {
            "patterns": result.patterns,
            "pitfalls": result.pitfalls,
            "dataset_description": description,
            "dataset_category": full_category,
            **trace_update
        }
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[{node_name}] Failed | error={e}")
        
        trace_update = await record_trace(
            state, trace_service, node_name, {}, {},
            duration_ms, "FAILED", str(e)
        )
        
        return {
            "patterns": [],
            "pitfalls": [],
            "error": str(e),
            **trace_update
        }


# =============================================================================
# NODE: Distill Context
# =============================================================================

_NEWS_MECHANISM_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("fast_d1_reaction", ("pct_30sec", "pct_1min", "pct_5_min", "max_up_ret", "max_dn_ret")),
    ("delayed_news_drift", ("pct_30min", "pct_60min", "pct_120min", "indx_perf", "prev_day_ret")),
    ("abnormal_range_volatility", ("high_exc_stddev", "low_exc_stddev", "atr_ratio", "range_stddev", "session_range")),
    ("volume_liquidity_response", ("ratio_vol", "vol_stddev", "curr_vol", "mov_vol", "volume", "vwap")),
    ("crowding_context", ("short_interest", "open_gap")),
]

_INSIDER_MECHANISM_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("transaction_value_intensity", ("valueeur", "value")),
    ("position_size_impact", ("holdings", "shares")),
    ("transaction_price_pressure", ("max_trade_price", "price")),
    ("transaction_significance", ("tradesignificance", "significance")),
]

_OPTION_MECHANISM_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("implied_volatility", ("implied_volatility", "deltasurface_d1_vi", "_vi")),
    ("tenor_term_structure", ("period", "time_period", "maturity", "expiry")),
    ("moneyness_delta", ("moneyness", "delta")),
    ("forward_price_basis", ("forward", "yield")),
    ("underlying_price_volume", ("price_", "volume")),
    ("dividend_context", ("dvd", "dividend")),
    ("data_quality", ("data_quality", "isenabled")),
]


def _field_key(field: Dict) -> str:
    return str(field.get("id") or field.get("name") or field.get("field_id") or "")


def _is_news_dataset(dataset_id: str, dataset_category: str, fields: List[Dict]) -> bool:
    dataset_text = f"{dataset_id} {dataset_category}".lower()
    if "news" in dataset_text:
        return True
    return any(_field_key(f).lower().startswith("news_") for f in fields[:20])


def _classify_news_mechanism(field: Dict) -> Optional[str]:
    text = " ".join(
        str(field.get(key, ""))
        for key in ("id", "name", "description")
    ).lower()

    # Plain EOD price fields created strategy collapse in real runs; keep them
    # out of the primary mechanism focus unless the task explicitly prefers them.
    if any(token in text for token in ("eod_open", "eod_high", "eod_low", "eod_close")):
        return "plain_eod_price_context"

    for mechanism, tokens in _NEWS_MECHANISM_RULES:
        if any(token in text for token in tokens):
            return mechanism
    return None


def _dataset_family(dataset_id: str, dataset_category: str) -> str:
    text = f"{dataset_id} {dataset_category}".lower()
    if "news" in text:
        return "news"
    if "insider" in text:
        return "insiders"
    if "option" in text:
        return "option"
    return ""


def _classify_by_rules(field: Dict, rules: List[Tuple[str, Tuple[str, ...]]]) -> Optional[str]:
    text = " ".join(
        str(field.get(key, ""))
        for key in ("id", "name", "description")
    ).lower()
    for mechanism, tokens in rules:
        if any(token in text for token in tokens):
            return mechanism
    return None


def _build_mechanism_groups(
    fields: List[Dict],
    dataset_id: str,
    dataset_category: str,
) -> Dict[str, List[Dict]]:
    family = _dataset_family(dataset_id, dataset_category)
    if family == "news" or _is_news_dataset(dataset_id, dataset_category, fields):
        classifier = _classify_news_mechanism
    elif family == "insiders":
        classifier = lambda field: _classify_by_rules(field, _INSIDER_MECHANISM_RULES)
    elif family == "option":
        classifier = lambda field: _classify_by_rules(field, _OPTION_MECHANISM_RULES)
    else:
        return {}

    groups: Dict[str, List[Dict]] = {}
    for field in _prefer_numeric_fields(fields):
        mechanism = classifier(field)
        if not mechanism:
            continue
        groups.setdefault(mechanism, []).append({**field, "mechanism": mechanism})
    return groups


def _select_mechanism_diverse_fields(
    groups: Dict[str, List[Dict]],
    preferred_fields: List[str],
    max_total: int = 30,
    max_per_group: int = 5,
) -> List[Dict]:
    preferred = {str(f).lower() for f in preferred_fields}
    selected: List[Dict] = []
    seen = set()

    priority = [
        "fast_d1_reaction",
        "delayed_news_drift",
        "abnormal_range_volatility",
        "volume_liquidity_response",
        "crowding_context",
        "plain_eod_price_context",
        "transaction_value_intensity",
        "position_size_impact",
        "transaction_price_pressure",
        "transaction_significance",
        "implied_volatility",
        "moneyness_delta",
        "tenor_term_structure",
        "forward_price_basis",
        "underlying_price_volume",
        "dividend_context",
        "data_quality",
    ]

    def add(field: Dict) -> None:
        key = _field_key(field)
        if not key or key in seen or len(selected) >= max_total:
            return
        selected.append(field)
        seen.add(key)

    for mechanism in priority:
        fields = groups.get(mechanism, [])
        fields = sorted(
            fields,
            key=lambda f: (
                _field_key(f).lower() not in preferred,
                len(_field_key(f)),
                _field_key(f),
            ),
        )
        limit = 2 if mechanism == "plain_eod_price_context" else max_per_group
        for field in fields[:limit]:
            add(field)

    # Round-robin in any mechanisms not listed above.
    for mechanism, fields in groups.items():
        if mechanism in priority:
            continue
        for field in fields[:max_per_group]:
            add(field)

    return selected

async def node_distill_context(
    state: MiningState,
    llm_service: LLMService,
    config: RunnableConfig = None
) -> Dict:
    """
    Distill relevant concepts/categories from large field sets.
    
    Input State:
        - fields, dataset_description
        
    Output Updates:
        - distilled_concepts
        - focused_fields
        - trace_steps
    """
    start_time = time.time()
    node_name = "DISTILL_CONTEXT"
    
    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    task_config = config.get("configurable", {}).get("task_config", {}) if config else {}
    logger.info(f"[{node_name}] Starting | task={state.task_id} fields={len(state.fields)}")
    
    # Group fields by category
    categories = {}
    for f in state.fields:
        cat = f.get("category") or "General"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(f.get("id", f.get("name")))
    
    # Format for prompt
    categories_text = []
    for cat, f_list in categories.items():
        sample = ", ".join(f_list[:5])
        suffix = f"... ({len(f_list)-5} more)" if len(f_list) > 5 else ""
        categories_text.append(f"- **{cat}**: {sample}{suffix}")
    
    field_categories_str = "\n".join(categories_text)
    
    success_patterns_text = "\n".join([
        f"- {p.get('pattern', '')}" for p in state.patterns[:3]
    ]) or "N/A"
    
    prompt = DISTILL_USER.format(
        dataset_id=state.dataset_id,
        description=state.dataset_description or "N/A",
        category=state.dataset_category or "Unknown",
        success_patterns=success_patterns_text,
        field_categories=field_categories_str
    )
    
    try:
        response = await llm_service.call(
            system_prompt=DISTILL_SYSTEM,
            user_prompt=prompt,
            temperature=0.5,
            json_mode=True,
            max_tokens=768
        )
    except Exception as llm_err:
        logger.error(f"[{node_name}] LLM call failed: {llm_err}")
        response = type('obj', (object,), {'success': False, 'parsed': None, 'error': str(llm_err)})()
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    selected_concepts = []
    reasoning = ""
    focused_fields = []
    
    if response.success and response.parsed:
        try:
            parsed = response.parsed
            if isinstance(parsed, dict):
                selected_concepts = parsed.get("selected_concepts", []) or []
                reasoning = parsed.get("reasoning", "") or ""
        except (TypeError, AttributeError) as parse_err:
            logger.error(f"[{node_name}] Parse error: {parse_err}")
    
    if not isinstance(selected_concepts, list):
        selected_concepts = [selected_concepts] if selected_concepts else []

    mechanism_groups = _build_mechanism_groups(
        state.fields,
        dataset_id=state.dataset_id,
        dataset_category=state.dataset_category,
    )
    preferred_fields = task_config.get("preferred_fields", []) or []
    mechanism_focused_fields = _select_mechanism_diverse_fields(
        mechanism_groups,
        preferred_fields=preferred_fields,
    )

    if mechanism_groups:
        mechanism_concepts = [
            mechanism
            for mechanism, fields in mechanism_groups.items()
            if fields and mechanism != "plain_eod_price_context"
        ]
        selected_concepts = list(dict.fromkeys([*mechanism_concepts, *selected_concepts]))
    
    if selected_concepts:
        full_field_list = state.fields
        
        for f in full_field_list:
            f_cat = f.get("category") or "General"
            f_id = str(f.get("id", "")).lower()
            f_name = str(f.get("name", "")).lower()
            
            for c in selected_concepts:
                c_lower = c.lower()
                if c_lower in f_cat.lower() or f_cat.lower() in c_lower:
                    focused_fields.append(f)
                    break
                if c_lower in f_id or c_lower in f_name:
                    focused_fields.append(f)
                    break
    
    if mechanism_focused_fields:
        merged = mechanism_focused_fields + focused_fields
        seen = set()
        focused_fields = []
        for field in merged:
            key = _field_key(field)
            if key and key not in seen:
                focused_fields.append(field)
                seen.add(key)

    focused_fields = _prefer_numeric_fields(focused_fields)

    if not focused_fields:
        logger.warning(f"[{node_name}] Distillation yielded 0 fields. Falling back to top 30.")
        focused_fields = _prefer_numeric_fields(state.fields)[:30] or state.fields[:30]
    
    logger.info(
        f"[{node_name}] Complete | concepts={selected_concepts} focused={len(focused_fields)} "
        f"mechanisms={list(mechanism_groups.keys())}"
    )
    
    trace_update = await record_trace(
        state, trace_service, node_name,
        {"field_count": len(state.fields), "categories": list(categories.keys())},
        {
            "selected_concepts": selected_concepts,
            "focused_count": len(focused_fields),
            "reasoning": reasoning,
            "mechanism_groups": {
                mechanism: [_field_key(field) for field in fields[:8]]
                for mechanism, fields in mechanism_groups.items()
            },
        },
        duration_ms,
        "SUCCESS" if response.success else "FAILED",
        response.error if hasattr(response, 'error') else None
    )
    
    return {
        "distilled_concepts": selected_concepts,
        "focused_fields": focused_fields,
        **trace_update
    }


# =============================================================================
# NODE: Hypothesis Generation
# =============================================================================

async def node_hypothesis(
    state: MiningState,
    llm_service: LLMService,
    config: RunnableConfig = None
) -> Dict:
    """
    Generate investment hypotheses based on dataset using hypothesis-driven approach.
    
    Redesigned based on RD-Agent principles:
    - Each hypothesis is precise, testable, and focused on a single direction
    - Learns from previous experiment results (feedback loop)
    - Balances exploration and exploitation based on evidence
    - No preconceived biases about what works
    
    Input State:
        - dataset_id, fields, patterns, dataset_description
        - experiment_trace (optional): Previous experiment results for learning
    
    Output Updates:
        - hypotheses
        - knowledge_transfer
        - trace_steps
    """
    start_time = time.time()
    node_name = "HYPOTHESIS"
    
    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    strategy_dict = config.get("configurable", {}).get("strategy", {}) if config else {}
    task_config = config.get("configurable", {}).get("task_config", {}) if config else {}
    
    # Get experiment trace for learning (if available)
    experiment_trace = strategy_dict.get("experiment_trace", [])
    exploration_weight = strategy_dict.get("exploration_weight", 0.5)
    
    logger.info(f"[{node_name}] Starting | task={state.task_id} trace_len={len(experiment_trace)}")
    
    target_fields = _prefer_numeric_fields(state.focused_fields) if state.focused_fields else _prefer_numeric_fields(state.fields)[:20]
    if not target_fields:
        target_fields = state.focused_fields if state.focused_fields else state.fields[:20]
    
    # Build prompt context
    prompt_context = PromptContext(
        dataset_id=state.dataset_id,
        dataset_description=state.dataset_description or "",
        dataset_category=state.dataset_category or "",
        region=state.region,
        universe=state.universe,
        fields=target_fields,
        operators=state.operators[:30],
        success_patterns=state.patterns[:5],
        failure_pitfalls=state.pitfalls[:5],
        exploration_weight=exploration_weight,
    )
    
    # Use new hypothesis builder with experiment trace
    prompt = build_hypothesis_prompt(prompt_context, experiment_trace)
    
    # Adjust temperature based on exploration weight
    # Higher exploration -> higher temperature for more diverse hypotheses
    temperature = 0.7 + (exploration_weight * 0.3)  # Range: 0.7 - 1.0
    
    response = await llm_service.call(
        system_prompt=HYPOTHESIS_SYSTEM,
        user_prompt=prompt,
        temperature=temperature,
        json_mode=True,
        max_tokens=2048
    )
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    hypotheses = []
    knowledge_transfer = {}
    analysis = {}
    
    if response.success and response.parsed:
        parsed = response.parsed
        hypotheses = parsed.get("hypotheses", [])
        knowledge_transfer = parsed.get("knowledge_transfer", {})
        analysis = parsed.get("analysis", {})
        
        # Log extracted knowledge for future reference
        if knowledge_transfer:
            rules = knowledge_transfer.get("if_then_rules", [])
            if rules:
                logger.info(f"[{node_name}] Extracted {len(rules)} knowledge rules")
                for rule in rules[:3]:
                    logger.debug(f"[{node_name}] Rule: {rule}")

    if not hypotheses:
        fallback_fields = target_fields[:3] or state.fields[:3]
        hypotheses = [
            {
                "statement": f"{f.get('id', f.get('name', 'field'))} may contain cross-sectional predictive signal.",
                "rationale": f"Use {f.get('id', f.get('name', 'field'))} with ranking and time-series smoothing as a fallback hypothesis.",
                "key_fields": [f.get("id", f.get("name", ""))],
            }
            for f in fallback_fields
        ]
        analysis = {**analysis, "fallback_used": True}
    
    logger.info(f"[{node_name}] Complete | hypotheses={len(hypotheses)}")
    
    trace_update = await record_trace(
        state, trace_service, node_name,
        {
            "dataset_id": state.dataset_id,
            "mode": "hypothesis_driven",
            "exploration_weight": exploration_weight,
            "experiment_trace_length": len(experiment_trace)
        },
        {
            "hypotheses_count": len(hypotheses),
            "hypotheses": hypotheses[:3],
            "knowledge_transfer": knowledge_transfer,
            "analysis": analysis
        },
        duration_ms,
        "SUCCESS" if response.success else "FAILED",
        response.error if hasattr(response, 'error') else None
    )
    
    return {
        "hypotheses": hypotheses,
        "knowledge_transfer": knowledge_transfer,
        **trace_update
    }


# Backward compatible helper - select exploration fields
def _select_exploration_fields(
    target_fields: List[Dict],
    all_fields: List[Dict],
    count: int = 3
) -> List[Dict]:
    """
    Select fields for exploration that are not in the target set.
    
    This helps ensure diversity and prevents tunnel vision.
    """
    remaining_fields = [f for f in all_fields if f not in target_fields]
    if len(remaining_fields) >= count:
        return random.sample(remaining_fields, count)
    elif len(all_fields) > count:
        return random.sample(all_fields, min(count, len(all_fields)))
    return all_fields


def _is_group_like_field(field: Dict) -> bool:
    """Avoid fields that BRAIN treats as group-valued numeric units."""
    field_type = str(field.get("type") or field.get("field_type") or "").upper()
    if field_type == "GROUP":
        return True
    field_id = str(field.get("id") or field.get("field_id") or "").lower()
    field_name = str(field.get("name") or field.get("field_name") or "").lower()
    if field_id in {"sector", "industry", "subindustry", "exchange", "country", "market"}:
        return True
    if field_name in {"sector", "industry", "subindustry", "exchange", "country", "market"}:
        return True
    text = " ".join(
        str(field.get(key) or "")
        for key in ("id", "name", "field_id", "field_name", "description")
    ).lower()
    numeric_hints = (
        "score",
        "rank",
        "return",
        "float",
        "value",
        "momentum",
        "estimate",
        "revision",
        "component",
        "ratio",
    )
    if any(token in text for token in numeric_hints):
        return False
    group_tokens = ("group", "bucket", "cluster", "classification", "category")
    return any(token in text for token in group_tokens)


def _prefer_numeric_fields(fields: List[Dict]) -> List[Dict]:
    """Return alpha-usable numeric fields for generation prompts."""
    numeric = [
        field for field in fields
        if not _is_group_like_field(field)
        and str(field.get("type") or field.get("field_type") or "MATRIX").upper() in {"MATRIX", "VECTOR"}
    ]
    return sorted(numeric, key=_field_priority_key)


def _field_priority_key(field: Dict) -> Tuple:
    """Prefer fields that have enough platform evidence to support mining.

    Large datasets can contain hundreds of sparse variants. Without a quality
    sort, deterministic templates spend scarce simulations on whichever fields
    the API returned first instead of the durable revision/momentum/residual
    fields that historical IND evidence points to.
    """
    field_id = _field_key(field).lower()
    description = str(field.get("description") or "").lower()
    text = f"{field_id} {description}"
    durable_tokens = (
        "revision_component",
        "estimate_revision",
        "value_momentum",
        "momentum_rank",
        "residual_return",
        "eps_mean",
        "smartestimate",
        "weighted_estimate",
        "revenue_revision",
        "earnings_revision",
    )
    durable_score = sum(1 for token in durable_tokens if token in text)
    alpha_count = int(field.get("alpha_count") or field.get("alphaCount") or 0)
    user_count = int(field.get("user_count") or field.get("userCount") or 0)
    coverage = float(field.get("coverage") or 0)
    pyramid = float(field.get("pyramid_multiplier") or field.get("pyramidMultiplier") or 0)
    field_type = str(field.get("type") or field.get("field_type") or "MATRIX").upper()
    vector_penalty = 1 if field_type == "VECTOR" else 0
    float_penalty = 0 if field_id.endswith("_float") else 1
    return (
        -durable_score,
        vector_penalty,
        float_penalty,
        -alpha_count,
        -user_count,
        -coverage,
        -pyramid,
        len(field_id),
        field_id,
    )


def _operator_names(operators: List[Dict]) -> set:
    return {
        str(op.get("name") if isinstance(op, dict) else op).lower()
        for op in operators
        if (op.get("name") if isinstance(op, dict) else op)
    }


def _regular_operator_names(operators: List[Dict]) -> List[str]:
    """Return active operators that are allowed in REGULAR expressions."""
    names = []
    seen = set()
    for op in operators:
        name = str(op.get("name") if isinstance(op, dict) else op or "").strip()
        if not name:
            continue
        scope = op.get("scope") if isinstance(op, dict) else None
        if scope and "REGULAR" not in {str(item).upper() for item in scope}:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(key)

    priority = [
        "rank",
        "group_rank",
        "ts_delta",
        "ts_zscore",
        "ts_rank",
        "ts_mean",
        "group_neutralize",
        "signed_power",
        "winsorize",
        "zscore",
        "normalize",
        "scale",
        "subtract",
        "divide",
        "reverse",
        "ts_av_diff",
        "ts_std_dev",
        "ts_sum",
        "ts_decay_linear",
        "ts_returns",
        "bucket",
        "group_zscore",
        "group_scale",
        "group_sum",
        "group_mean",
        "group_count",
    ]
    priority_index = {name: idx for idx, name in enumerate(priority)}
    return sorted(names, key=lambda name: (priority_index.get(name, len(priority)), name))


def _field_signal(field: Dict, vector_agg: str = "vec_sum") -> Tuple[str, int]:
    """Return a MATRIX-like signal expression and the aggregation op count."""
    field_id = _field_key(field)
    field_type = str(field.get("type") or field.get("field_type") or "MATRIX").upper()
    if field_type == "VECTOR":
        return f"{vector_agg}({field_id})", 1
    return field_id, 0


def _op_count(expression: str) -> int:
    return len(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", expression or ""))


def _operators_in_expression(expression: str) -> set:
    return {
        match.group(1).lower()
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression or "")
    }


def _first_order_expression_for_operator(
    op: str,
    x: str,
    y: str,
    z: str,
    vector_x: Optional[str],
    vector_y: Optional[str],
) -> Optional[str]:
    """Build a minimal probe that isolates one REGULAR operator where possible."""
    unary = {
        "abs", "arc_tan", "hump", "inverse", "is_nan",
        "is_not_finite", "log", "normalize", "not", "pasteurize",
        "purify", "quantile", "rank", "reverse", "round", "scale",
        "sign", "sqrt", "winsorize", "zscore",
        "days_from_last_change",
    }
    binary = {
        "add", "and", "divide", "equal", "greater", "greater_equal",
        "less", "less_equal", "max", "min", "multiply", "nan_mask",
        "not_equal", "or", "subtract", "vector_neut",
    }
    group_x = {
        "group_count", "group_max", "group_min", "group_neutralize",
        "group_rank", "group_scale", "group_std_dev", "group_sum",
        "group_zscore",
    }
    ts_x_d = {
        "jump_decay", "kth_element", "last_diff_value", "ts_arg_max",
        "ts_arg_min", "ts_av_diff", "ts_backfill", "ts_count_nans",
        "ts_decay_linear", "ts_delay", "ts_delta", "ts_ir",
        "ts_kurtosis", "ts_max", "ts_max_diff", "ts_mean", "ts_min",
        "ts_product", "ts_quantile", "ts_rank", "ts_returns",
        "ts_scale", "ts_std_dev", "ts_sum", "ts_zscore",
    }
    vec_ops = {
        "vec_avg", "vec_count", "vec_max", "vec_min", "vec_norm",
        "vec_range", "vec_stddev", "vec_sum",
    }

    if op in unary:
        return f"{op}({x})"
    if op in binary:
        return f"{op}({x}, {y})"
    if op in group_x:
        return f"{op}({x}, industry)"
    if op in ts_x_d:
        if op == "kth_element":
            return f"{op}({x}, 20, k=1)"
        if op == "ts_backfill":
            return f"{op}({x}, 20)"
        if op == "jump_decay":
            return f"{op}({x}, 20, sensitivity=0.5, force=0.1)"
        return f"{op}({x}, 20)"
    if op in vec_ops:
        return f"{op}({vector_x})" if vector_x else None

    special = {
        "bucket": f"group_count({x}, bucket(rank({x}), range=\"0,1,0.1\"))",
        "clamp": f"clamp({x}, lower=-1, upper=1)",
        "densify": f"group_count({x}, densify(industry))",
        "filter": f"filter({x}, h=\"1,2,3,4\", t=\"0.5\")",
        "group_backfill": f"group_backfill({x}, industry, 20)",
        "group_cartesian_product": f"group_count({x}, group_cartesian_product(industry, subindustry))",
        "group_mean": f"group_mean({x}, 1, industry)",
        "group_vector_neut": f"group_vector_neut({x}, {y}, industry)",
        "if_else": f"if_else({x} > 0, {y}, 0)",
        "keep": f"keep({x}, {y}, period=5)",
        "power": f"power({x}, 2)",
        "round_down": f"round_down({x})",
        "scale_down": f"scale_down({x})",
        "signed_power": f"signed_power({x}, 2)",
        "tail": f"tail({x}, lower=-0.5, upper=0.5, newval=0)",
        "to_nan": f"to_nan({x})",
        "trade_when": f"trade_when({x}, {y}, -1)",
        "ts_co_skewness": f"ts_co_skewness({y}, {x}, 60)",
        "ts_corr": f"ts_corr({x}, {y}, 60)",
        "ts_covariance": f"ts_covariance({y}, {x}, 60)",
        "ts_rank_gmean_amean_diff": f"ts_rank_gmean_amean_diff({x}, {y}, {x}, {y}, 60)",
        "ts_regression": f"ts_regression({y}, {x}, 60)",
        "ts_step": "ts_step(1)",
        "ts_target_tvr_decay": f"ts_target_tvr_decay({x}, lambda_min=0, lambda_max=1, target_tvr=0.1)",
        "ts_target_tvr_delta_limit": f"ts_target_tvr_delta_limit({x}, {y}, lambda_min=0, lambda_max=1, target_tvr=0.1)",
        "ts_target_tvr_hump": f"ts_target_tvr_hump({x}, lambda_min=0, lambda_max=1, target_tvr=0.1)",
        "ts_triple_corr": f"ts_triple_corr({x}, {y}, {x}, 60)",
        "ts_vector_neut": f"ts_vector_neut({x}, {y}, 60)",
        "ts_vector_proj": f"ts_vector_proj({x}, {y}, 60)",
    }
    return special.get(op)


def _first_order_operator_probe_candidates(
    fields: List[Dict],
    operators: List[Dict],
    max_operator_count: int = 5,
    preferred_fields: Optional[List[str]] = None,
) -> List[AlphaCandidate]:
    """Generate one-main-operator probes for every REGULAR operator.

    Vector aggregation used to coerce VECTOR fields is treated as data
    preparation by the research design, but the expression still respects the
    platform hard operator limit.
    """
    numeric = _prefer_numeric_fields(fields)
    if preferred_fields:
        preferred_order = {str(field): idx for idx, field in enumerate(preferred_fields)}
        numeric = sorted(
            numeric,
            key=lambda field: (
                preferred_order.get(_field_key(field), len(preferred_order)),
                _field_priority_key(field),
            ),
        )
    matrix_fields = [
        field for field in numeric
        if str(field.get("type") or field.get("field_type") or "MATRIX").upper() == "MATRIX"
    ]
    vector_fields = [
        field for field in numeric
        if str(field.get("type") or field.get("field_type") or "MATRIX").upper() == "VECTOR"
    ]
    base_fields = matrix_fields or numeric
    if not base_fields:
        return []

    x = _field_key(base_fields[0])
    y = _field_key(base_fields[1] if len(base_fields) > 1 else base_fields[0])
    z = _field_key(base_fields[2] if len(base_fields) > 2 else base_fields[-1])
    vector_x = _field_key(vector_fields[0]) if vector_fields else None
    vector_y = _field_key(vector_fields[1] if len(vector_fields) > 1 else vector_fields[0]) if vector_fields else None

    candidates: List[AlphaCandidate] = []
    seen = set()
    for op in _regular_operator_names(operators):
        expression = _first_order_expression_for_operator(op, x, y, z, vector_x, vector_y)
        if not expression or expression in seen:
            continue
        seen.add(expression)
        if _op_count(expression) > max_operator_count:
            continue
        candidates.append(AlphaCandidate(
            expression=expression,
            hypothesis=f"First-order probe for REGULAR operator `{op}`.",
            explanation=(
                "One-main-operator signal probe used to map which operators "
                "carry standalone predictive signal before adding another operator."
            ),
            metadata={
                "source": "first_order_operator_probe",
                "probe_operator": op,
                "operator_order": 1,
                "fields_used": [field for field in (x, y, z) if field],
                "operator_skeleton": op,
                "strategy_style": "operator_coverage_probe",
                "complexity": "first_order",
                "novelty_level": "coverage",
            },
        ))
    return candidates


def _template_candidates(
    fields: List[Dict],
    operators: List[Dict],
    max_operator_count: int = 5,
) -> List[AlphaCandidate]:
    """Generate compact forum-style template candidates from available fields.

    These candidates force coverage of durable construction families that the
    LLM often under-samples: short-horizon change, center deviation, spread,
    ratio, group rank, and group-neutral residual.
    """
    available_ops = _operator_names(operators)
    numeric = _prefer_numeric_fields(fields)
    if not numeric:
        return []

    templates: List[Tuple[str, str, str, List[str]]] = []
    primary = numeric[:4]
    by_id = {_field_key(field): field for field in numeric}

    def add_known_blend(left_id: str, right_id: str, expression: str, skeleton: str, explanation: str) -> None:
        if left_id in by_id and right_id in by_id:
            templates.append((expression, explanation, skeleton, [left_id, right_id]))

    add_known_blend(
        "sector_value_momentum_rank_float",
        "industry_value_momentum_rank_float",
        "group_rank(add(ts_delta(sector_value_momentum_rank_float, 21), ts_zscore(industry_value_momentum_rank_float, 44)), industry)",
        "template_priority_low_prod_sector_industry_blend",
        "Industry-relative blend that has shown lower production correlation than pure value-momentum persistence.",
    )
    add_known_blend(
        "sector_value_momentum_rank_float",
        "industry_value_momentum_rank_float",
        "group_rank(add(ts_delta(sector_value_momentum_rank_float, 21), ts_zscore(industry_value_momentum_rank_float, 44)), subindustry)",
        "template_priority_low_prod_sector_industry_blend_subindustry",
        "Narrow-peer version of the lower-correlation sector/industry momentum blend.",
    )
    add_known_blend(
        "sector_value_momentum_rank_float",
        "short_term_price_momentum_score_2",
        "group_rank(add(sector_value_momentum_rank_float, ts_delta(short_term_price_momentum_score_2, 10)), industry)",
        "template_priority_low_prod_value_short_momentum_blend",
        "Lower-correlation blend of stable value momentum and short-term price impulse.",
    )
    add_known_blend(
        "sector_value_momentum_rank_float",
        "short_term_price_momentum_score_2",
        "group_rank(add(sector_value_momentum_rank_float, ts_delta(short_term_price_momentum_score_2, 10)), subindustry)",
        "template_priority_low_prod_value_short_momentum_blend_subindustry",
        "Narrow-peer blend of stable value momentum and short-term price impulse.",
    )

    # Priority pass: spread the highest-evidence durable skeletons across
    # multiple fields before spending budget on lower-priority variants. This
    # avoids exhausting the candidate pool on many variants of only the first
    # sorted field after historical DB de-duplication.
    priority_fields = numeric[:8]
    for field in priority_fields:
        field_id = _field_key(field)
        signal, _ = _field_signal(field)
        if {"signed_power", "group_rank", "ts_rank"}.issubset(available_ops):
            templates.append(
                (
                    f"signed_power(group_rank(ts_rank({signal}, 500), industry), 3.0)",
                    "Long-window rank persistence amplified after industry ranking.",
                    "template_priority_group_power_rank_industry",
                    [field_id],
                )
            )
        if {"signed_power", "group_rank", "ts_zscore"}.issubset(available_ops):
            templates.append(
                (
                    f"signed_power(group_rank(ts_zscore({signal}, 500), industry), 3.0)",
                    "Long-window zscore persistence amplified after industry ranking.",
                    "template_priority_group_power_zscore_industry",
                    [field_id],
                )
            )
            templates.append(
                (
                    f"signed_power(group_rank(ts_zscore({signal}, 500), subindustry), 2.0)",
                    "Long-window zscore persistence ranked within narrower peer groups.",
                    "template_priority_group_power_zscore_subindustry",
                    [field_id],
                )
            )
        if {"rank", "signed_power", "group_rank", "ts_zscore"}.issubset(available_ops):
            templates.append(
                (
                    f"rank(signed_power(group_rank(ts_zscore({signal}, 500), industry), 3.0))",
                    "Cross-sectionally normalized durable industry-relative signal.",
                    "template_priority_ranked_group_power_zscore",
                    [field_id],
                )
            )
        if {"group_neutralize", "ts_av_diff"}.issubset(available_ops):
            templates.append(
                (
                    f"group_neutralize(ts_av_diff({signal}, 44), industry)",
                    "Industry-neutral average-difference persistence.",
                    "template_priority_group_neutral_av_diff",
                    [field_id],
                )
            )
            if "ts_zscore" in available_ops:
                neutral_signal = f"group_neutralize(ts_av_diff({signal}, 44), industry)"
                templates.append(
                    (
                        f"ts_zscore({neutral_signal}, 60)",
                        "Time-standardized industry-neutral persistence for lower production correlation.",
                        "template_priority_low_prod_neutral_zscore",
                        [field_id],
                    )
                )
                if "rank" in available_ops:
                    templates.append(
                        (
                            f"rank(ts_zscore({neutral_signal}, 60))",
                            "Cross-sectionally normalized low-correlation neutral persistence.",
                            "template_priority_low_prod_ranked_neutral_zscore",
                            [field_id],
                        )
                    )
                if "signed_power" in available_ops:
                    templates.append(
                        (
                            f"signed_power(ts_zscore({neutral_signal}, 60), 2.0)",
                            "Amplified low-correlation neutral persistence.",
                            "template_priority_low_prod_power_neutral_zscore",
                            [field_id],
                        )
                    )
                if {"signed_power", "group_rank"}.issubset(available_ops):
                    templates.append(
                        (
                            f"signed_power(group_rank(ts_zscore({neutral_signal}, 60), industry), 2.0)",
                            "Industry-ranked low-correlation neutral persistence.",
                            "template_priority_low_prod_group_power_neutral_zscore",
                            [field_id],
                        )
                    )

    for field in primary:
        field_id = _field_key(field)
        signal, _ = _field_signal(field)
        if {"signed_power", "group_rank", "ts_zscore"}.issubset(available_ops):
            for window, power in ((500, 3.0), (252, 2.0)):
                templates.append(
                    (
                        f"signed_power(group_rank(ts_zscore({signal}, {window}), industry), {power})",
                        "Durable long-window revision/value signal amplified after industry ranking.",
                        "template_durable_group_power_zscore",
                        [field_id],
                    )
                )
        if {"signed_power", "group_rank", "ts_rank"}.issubset(available_ops):
            templates.append(
                (
                    f"signed_power(group_rank(ts_rank({signal}, 500), industry), 3.0)",
                    "Long-window rank persistence amplified after industry ranking.",
                    "template_durable_group_power_rank",
                    [field_id],
                )
            )
        if {"group_neutralize", "ts_av_diff"}.issubset(available_ops):
            for window in (44, 66, 126):
                templates.append(
                    (
                        f"group_neutralize(ts_av_diff({signal}, {window}), industry)",
                        "Industry-neutral average-difference persistence.",
                        "template_durable_group_neutral_av_diff",
                        [field_id],
                    )
                )
        if {"group_rank", "ts_av_diff"}.issubset(available_ops):
            templates.append(
                (
                    f"group_rank(ts_av_diff({signal}, 66), subindustry)",
                    "Subindustry-relative average-difference persistence.",
                    "template_durable_group_rank_av_diff",
                    [field_id],
                )
            )
        templates.append(
            (
                f"group_rank(add(ts_mean({signal}, 20), ts_delta({signal}, 5)), subindustry)",
                "Subindustry-relative blend of stable level and recent impulse.",
                "template_group_level_impulse",
                [field_id],
            )
        )
        if {"divide", "ts_std_dev"}.issubset(available_ops):
            templates.extend([
                (
                    f"rank(divide(ts_delta({signal}, 5), ts_std_dev({signal}, 20)))",
                    "Short change scaled by recent signal volatility.",
                    "template_vol_scaled_change",
                    [field_id],
                ),
                (
                    f"group_rank(divide(ts_delta({signal}, 5), ts_std_dev({signal}, 20)), subindustry)",
                    "Subindustry-relative volatility-scaled change.",
                    "template_group_vol_scaled_change",
                    [field_id],
                ),
            ])
        templates.extend([
            (
                f"rank(ts_delta({signal}, 5))",
                "Short-horizon change in the primary signal.",
                "template_short_change",
                [field_id],
            ),
            (
                f"group_rank(ts_delta({signal}, 5), subindustry)",
                "Subindustry-relative short-horizon change.",
                "template_group_short_change",
                [field_id],
            ),
            (
                f"rank(subtract({signal}, ts_mean({signal}, 20)))",
                "Deviation from a rolling center.",
                "template_center_deviation",
                [field_id],
            ),
            (
                f"ts_zscore(ts_delta({signal}, 5), 20)",
                "Standardized recent acceleration.",
                "template_standardized_change",
                [field_id],
            ),
            (
                f"group_neutralize(rank(ts_delta({signal}, 5)), subindustry)",
                "Group-neutral residual of a short-horizon change.",
                "template_group_neutral_change",
                [field_id],
            ),
        ])

    for left, right in zip(numeric[:4], numeric[1:5]):
        left_id = _field_key(left)
        right_id = _field_key(right)
        left_signal, _ = _field_signal(left, vector_agg="vec_avg")
        right_signal, _ = _field_signal(right, vector_agg="vec_avg")
        left_is_vector = str(left.get("type") or left.get("field_type") or "MATRIX").upper() == "VECTOR"
        right_is_vector = str(right.get("type") or right.get("field_type") or "MATRIX").upper() == "VECTOR"

        if left_is_vector or right_is_vector:
            spread = f"subtract({left_signal}, {right_signal})"
        else:
            spread = f"subtract(ts_zscore({left_signal}, 20), ts_zscore({right_signal}, 20))"

        templates.extend([
            (
                f"rank({spread})",
                "Comparable-scale spread between two related fields.",
                "template_spread",
                [left_id, right_id],
            ),
            (
                f"group_rank({spread}, industry)",
                "Industry-relative spread between two related fields.",
                "template_group_spread",
                [left_id, right_id],
            ),
            (
                f"rank(divide({left_signal}, {right_signal}))",
                "Ratio of two comparable fields.",
                "template_ratio",
                [left_id, right_id],
            ),
        ])

    candidates: List[AlphaCandidate] = []
    seen = set()
    for expression, explanation, skeleton, used_fields in templates:
        if expression in seen:
            continue
        seen.add(expression)
        ops = _operators_in_expression(expression)
        if available_ops and not ops.issubset(available_ops):
            continue
        if _op_count(expression) > max_operator_count:
            continue
        candidates.append(AlphaCandidate(
            expression=expression,
            hypothesis=explanation,
            explanation=(
                "Deterministic forum-style template: comparable scale, "
                "center deviation, short change, or group-relative residual."
            ),
            metadata={
                "fields_used": used_fields,
                "operator_skeleton": skeleton,
                "strategy_style": "template_diversification",
                "complexity": "simple",
                "novelty_level": "variation",
                "source": "deterministic_template_library",
            },
        ))
    return candidates


def _augment_with_template_candidates(
    pending_alphas: List[AlphaCandidate],
    fields: List[Dict],
    operators: List[Dict],
    task_config: Dict,
    target_count: int,
) -> List[AlphaCandidate]:
    pool_size = int(task_config.get("generation_candidate_pool", max(target_count + 4, 8)))
    deterministic_first = bool(task_config.get("deterministic_templates_first", False))
    if pool_size <= len(pending_alphas) and not deterministic_first:
        return pending_alphas
    max_operator_count = int(task_config.get("max_operator_count", 5))
    attempted_expressions = {
        str(expression).strip()
        for expression in (task_config.get("attempted_expressions") or [])
        if str(expression).strip()
    }
    preferred_fields = [
        str(field).strip()
        for field in (task_config.get("preferred_fields") or [])
        if str(field).strip()
    ]
    if preferred_fields:
        preferred_order = {field: idx for idx, field in enumerate(preferred_fields)}
        fields = sorted(
            fields,
            key=lambda field: (
                preferred_order.get(_field_key(field), len(preferred_order)),
                _field_priority_key(field),
            ),
        )
    template_candidates = _template_candidates(
        fields,
        operators,
        max_operator_count=max_operator_count,
    )
    strengthening_candidates = _second_order_strengthening_candidates(
        task_config.get("first_order_strengthening_seeds") or [],
        max_operator_count=max_operator_count,
    )
    if task_config.get("first_order_operator_probe", False):
        first_order_candidates = _first_order_operator_probe_candidates(
            fields,
            operators,
            max_operator_count=max_operator_count,
            preferred_fields=preferred_fields,
        )
        target_probe_ops = [
            str(op).strip().lower()
            for op in (task_config.get("first_order_operator_probe_target_operators") or [])
            if str(op).strip()
        ]
        excluded_probe_ops = {
            str(op).strip().lower()
            for op in (task_config.get("first_order_operator_probe_exclude_operators") or [])
            if str(op).strip()
        }
        if target_probe_ops:
            target_order = {op: idx for idx, op in enumerate(target_probe_ops)}
            first_order_candidates = [
                candidate for candidate in first_order_candidates
                if str(candidate.metadata.get("probe_operator") or "").lower() in target_order
            ]
            first_order_candidates.sort(
                key=lambda candidate: target_order[
                    str(candidate.metadata.get("probe_operator") or "").lower()
                ]
            )
        elif excluded_probe_ops:
            first_order_candidates = [
                candidate for candidate in first_order_candidates
                if str(candidate.metadata.get("probe_operator") or "").lower() not in excluded_probe_ops
            ]
        first_order_batch_size = int(task_config.get("first_order_operator_probe_batch_size", 24) or 0)
        if (
            not target_probe_ops
            and first_order_batch_size > 0
            and len(first_order_candidates) > first_order_batch_size
        ):
            iteration = max(1, int(task_config.get("_iteration", 1) or 1))
            start_offset = max(0, int(task_config.get("first_order_operator_probe_start_index", 0) or 0))
            start = (start_offset + (iteration - 1) * first_order_batch_size) % len(first_order_candidates)
            first_order_candidates = (
                first_order_candidates[start: start + first_order_batch_size]
                or first_order_candidates[:first_order_batch_size]
            )
        template_candidates = first_order_candidates + template_candidates
    if strengthening_candidates:
        template_candidates = strengthening_candidates + template_candidates
    template_quota = int(task_config.get("template_candidate_quota", pool_size))

    if deterministic_first:
        augmented = []
        existing = set(attempted_expressions)
        for candidate in template_candidates:
            if len(augmented) >= min(pool_size, template_quota):
                break
            if candidate.expression in existing:
                continue
            augmented.append(candidate)
            existing.add(candidate.expression)
        for alpha in pending_alphas:
            if len(augmented) >= pool_size:
                break
            if not alpha.expression or alpha.expression in existing:
                continue
            augmented.append(alpha)
            existing.add(alpha.expression)
        return augmented

    existing = {
        alpha.expression for alpha in pending_alphas
        if alpha.expression
    } | attempted_expressions
    augmented = list(pending_alphas)
    for candidate in template_candidates:
        if len(augmented) >= pool_size:
            break
        if candidate.expression in existing:
            continue
        augmented.append(candidate)
        existing.add(candidate.expression)
    return augmented


def _second_order_strengthening_candidates(
    seeds: List[Dict],
    max_operator_count: int = 5,
) -> List[AlphaCandidate]:
    """Generate one-added-operator variants from known first-order signals."""
    if not seeds:
        return []

    try:
        from backend.optimization_chain import generate_local_rewrites
    except Exception as exc:
        logger.warning(f"Second-order strengthening unavailable: {exc}")
        return []

    seed_candidate_groups: List[List[AlphaCandidate]] = []
    seen = set()
    for seed in seeds:
        if isinstance(seed, str):
            seed = {"expression": seed}
        if not isinstance(seed, dict):
            continue
        expression = str(seed.get("expression") or "").strip()
        if not expression:
            continue
        probe_operator = str(seed.get("probe_operator") or seed.get("operator") or "seed").lower()
        metrics = {
            "sharpe": seed.get("sharpe", 0),
            "fitness": seed.get("fitness", 0),
            "turnover": seed.get("turnover", 0),
            "_candidate_metadata": {
                "source": "first_order_operator_probe",
                "probe_operator": probe_operator,
            },
        }
        seed_candidates: List[AlphaCandidate] = []
        variants = generate_local_rewrites(
            expression=expression,
            sim_result=metrics,
            feedback=seed.get("reason"),
            max_variants=int(seed.get("max_variants", 12) or 12),
        )
        for variant in variants:
            candidate_expression = str(variant.get("expression") or "").strip()
            if not candidate_expression or candidate_expression in seen:
                continue
            if _op_count(candidate_expression) > max_operator_count:
                continue
            seen.add(candidate_expression)
            seed_candidates.append(AlphaCandidate(
                expression=candidate_expression,
                hypothesis=(
                    f"Second-order strengthening of first-order `{probe_operator}` "
                    "signal by adding one operator."
                ),
                explanation=variant.get("rationale") or variant.get("description") or "",
                metadata={
                    "source": "second_order_strengthening",
                    "parent_expression": expression,
                    "parent_probe_operator": probe_operator,
                    "operator_order": 2,
                    "change_type": variant.get("change_type"),
                    "operator_skeleton": variant.get("description"),
                    "strategy_style": "first_order_signal_strengthening",
                    "complexity": "second_order",
                    "novelty_level": "targeted",
                },
            ))
        if seed_candidates:
            seed_candidate_groups.append(seed_candidates)

    candidates: List[AlphaCandidate] = []
    max_group_size = max((len(group) for group in seed_candidate_groups), default=0)
    for variant_index in range(max_group_size):
        for group in seed_candidate_groups:
            if variant_index < len(group):
                candidates.append(group[variant_index])
    return candidates


# =============================================================================
# NODE: Code Generation
# =============================================================================

async def node_code_gen(
    state: MiningState,
    llm_service: LLMService,
    config: RunnableConfig = None
) -> Dict:
    """
    Generate Alpha expressions using hypothesis-driven approach.
    
    Redesigned based on RD-Agent principles:
    - Each expression tests a specific hypothesis
    - Learns from previous experiment feedback
    - No preconceived biases about operators or patterns
    - Balanced approach between convention and exploration
    
    Input State:
        - dataset_id, fields, operators, patterns, pitfalls
        - hypotheses: Generated hypotheses to implement
        
    Config:
        - strategy: Evolution strategy dict with exploration parameters
        - experiment_feedback: Previous experiment results for learning
        - target_hypothesis: Optional specific hypothesis to implement
    
    Output Updates:
        - pending_alphas
        - trace_steps
    """
    start_time = time.time()
    node_name = "CODE_GEN"

    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    strategy_dict = config.get("configurable", {}).get("strategy", {}) if config else {}
    task_config = config.get("configurable", {}).get("task_config", {}) if config else {}
    
    # Extract strategy parameters
    temperature = strategy_dict.get("temperature", 0.7)
    exploration_weight = strategy_dict.get("exploration_weight", 0.5)
    preferred_fields = strategy_dict.get("preferred_fields", [])
    avoid_fields = strategy_dict.get("avoid_fields", [])
    focus_hypotheses = strategy_dict.get("focus_hypotheses", [])
    avoid_patterns = strategy_dict.get("avoid_patterns", [])
    preferred_operators = strategy_dict.get("preferred_operators", [])
    avoid_operators = strategy_dict.get("avoid_operators", [])
    strategy_action = strategy_dict.get("action_summary", "")
    strategy_reasoning = strategy_dict.get("reasoning", "")
    
    # New: Get experiment feedback for learning
    experiment_feedback = strategy_dict.get("experiment_feedback", [])
    target_hypothesis = strategy_dict.get("target_hypothesis")
    
    logger.info(
        f"[{node_name}] Starting | task={state.task_id} "
        f"temp={temperature:.2f} explore={exploration_weight:.2f} "
        f"feedback_len={len(experiment_feedback)}"
    )
    
    # Build structured prompt context
    prompt_context = PromptContext(
        dataset_id=state.dataset_id,
        dataset_description=state.dataset_description or "",
        dataset_category=state.dataset_category or "",
        region=state.region,
        universe=state.universe,
        fields=(_prefer_numeric_fields(state.focused_fields) if state.focused_fields else _prefer_numeric_fields(state.fields))[:30]
        or (state.focused_fields if state.focused_fields else state.fields[:30]),
        operators=state.operators[:50],
        success_patterns=state.patterns[:5],
        failure_pitfalls=state.pitfalls[:5],
        preferred_fields=preferred_fields,
        avoid_fields=avoid_fields,
        focus_hypotheses=[
            item for item in (strategy_action, strategy_reasoning) if item
        ] + focus_hypotheses + [
            h.get("statement", h.get("idea", str(h))) if isinstance(h, dict) else str(h)
            for h in state.hypotheses[:3]
        ],
        avoid_patterns=avoid_patterns,
        preferred_operators=preferred_operators,
        avoid_operators=avoid_operators,
        num_alphas=state.num_alphas_target,
        exploration_weight=exploration_weight,
        max_operator_count=int(task_config.get("max_operator_count", 5)),
    )
    
    # Use enhanced prompt builder with hypothesis and feedback context
    prompt = build_alpha_generation_prompt(
        prompt_context,
        target_hypothesis=target_hypothesis,
        experiment_feedback=experiment_feedback
    )
    
    try:
        response = await llm_service.call(
            system_prompt=ALPHA_GENERATION_SYSTEM,
            user_prompt=prompt,
            temperature=temperature,
            json_mode=True,
            max_tokens=2048
        )
    except Exception as llm_err:
        logger.error(f"[{node_name}] LLM call exception: {llm_err}")
        response = type('obj', (object,), {'success': False, 'parsed': None, 'error': str(llm_err)})()
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    # Parse alphas into candidates
    pending_alphas = []
    implementation_notes = ""
    alternatives_considered = []
    
    if response.success and response.parsed and isinstance(response.parsed, dict):
        parsed = response.parsed
        raw_alphas = parsed.get("alphas", []) or []
        implementation_notes = parsed.get("implementation_notes", "")
        alternatives_considered = parsed.get("alternatives_considered", [])
        
        for alpha_data in raw_alphas:
            # Handle both old format (hypothesis) and new format (hypothesis_tested)
            hypothesis_text = alpha_data.get("hypothesis_tested", alpha_data.get("hypothesis", ""))
            
            # Handle both old format (string) and new format (dict) for explanation
            explanation_raw = alpha_data.get("explanation", "")
            if isinstance(explanation_raw, dict):
                explanation = f"{explanation_raw.get('approach', '')} - {explanation_raw.get('market_logic', '')}"
            else:
                explanation = explanation_raw
            
            candidate = AlphaCandidate(
                expression=alpha_data.get("expression", ""),
                hypothesis=hypothesis_text,
                explanation=explanation,
                expected_sharpe=alpha_data.get("expected_sharpe")
            )
            
            # Attach additional metadata for tracking
            candidate.metadata = {
                "fields_used": alpha_data.get("fields_used", []),
                "operator_skeleton": alpha_data.get("operator_skeleton", ""),
                "strategy_style": alpha_data.get("strategy_style", ""),
                "complexity": alpha_data.get("complexity", "unknown"),
                "novelty_level": alpha_data.get("novelty_level", "unknown"),
            }
            
            if candidate.expression and candidate.expression.strip():
                pending_alphas.append(candidate)

    template_source_fields = (
        _prefer_numeric_fields(state.focused_fields)
        if state.focused_fields
        else _prefer_numeric_fields(state.fields)
    )
    before_template_count = len(pending_alphas)
    pending_alphas = _augment_with_template_candidates(
        pending_alphas=pending_alphas,
        fields=template_source_fields or state.fields,
        operators=state.operators,
        task_config=task_config,
        target_count=state.num_alphas_target,
    )
    template_added = max(0, len(pending_alphas) - before_template_count)
    if template_added:
        logger.info(f"[{node_name}] Added deterministic template candidates | count={template_added}")
    
    _debug_log("A", "nodes.py:code_gen:result", "Alpha code generation complete", {
        "alphas_generated": len(pending_alphas),
        "target": state.num_alphas_target,
        "duration_ms": duration_ms,
        "llm_success": response.success,
        "temperature": temperature,
        "implementation_notes": implementation_notes[:100] if implementation_notes else ""
    })
    
    logger.info(f"[{node_name}] Complete | alphas={len(pending_alphas)}")
    
    trace_update = await record_trace(
        state, trace_service, node_name,
        {
            "num_alphas_target": state.num_alphas_target,
            "strategy": {
                "temperature": temperature,
                "exploration_weight": exploration_weight,
                "preferred_fields_count": len(preferred_fields),
                "avoid_fields_count": len(avoid_fields),
                "preferred_operators_count": len(preferred_operators),
                "avoid_operators_count": len(avoid_operators),
                "has_target_hypothesis": target_hypothesis is not None,
                "feedback_length": len(experiment_feedback),
            }
        },
        {
            "alphas_generated": len(pending_alphas),
            "expressions": [a.expression[:200] for a in pending_alphas],
            "implementation_notes": implementation_notes,
            "alternatives_count": len(alternatives_considered),
            "template_candidates_added": template_added,
        },
        duration_ms,
        "SUCCESS" if response.success else "FAILED",
        response.error if hasattr(response, 'error') else None
    )
    
    return {
        "pending_alphas": pending_alphas,
        "current_alpha_index": 0,
        **trace_update
    }
