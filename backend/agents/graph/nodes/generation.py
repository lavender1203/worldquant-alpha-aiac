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


def _build_mechanism_groups(
    fields: List[Dict],
    dataset_id: str,
    dataset_category: str,
) -> Dict[str, List[Dict]]:
    if not _is_news_dataset(dataset_id, dataset_category, fields):
        return {}

    groups: Dict[str, List[Dict]] = {}
    for field in _prefer_numeric_fields(fields):
        mechanism = _classify_news_mechanism(field)
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
        max_tokens=1024
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
    text = " ".join(
        str(field.get(key) or "")
        for key in ("id", "name", "field_id", "field_name", "description")
    ).lower()
    group_tokens = ("group", "bucket", "cluster", "classification", "category")
    return any(token in text for token in group_tokens)


def _prefer_numeric_fields(fields: List[Dict]) -> List[Dict]:
    """Return scalar-looking MATRIX fields first for alpha generation prompts."""
    return [
        field for field in fields
        if not _is_group_like_field(field)
        and str(field.get("type") or field.get("field_type") or "MATRIX").upper() == "MATRIX"
    ]


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
    
    # Extract strategy parameters
    temperature = strategy_dict.get("temperature", 0.7)
    exploration_weight = strategy_dict.get("exploration_weight", 0.5)
    preferred_fields = strategy_dict.get("preferred_fields", [])
    avoid_fields = strategy_dict.get("avoid_fields", [])
    focus_hypotheses = strategy_dict.get("focus_hypotheses", [])
    avoid_patterns = strategy_dict.get("avoid_patterns", [])
    preferred_operators = strategy_dict.get("preferred_operators", [])
    avoid_operators = strategy_dict.get("avoid_operators", [])
    
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
        focus_hypotheses=focus_hypotheses + [
            h.get("statement", h.get("idea", str(h))) if isinstance(h, dict) else str(h)
            for h in state.hypotheses[:3]
        ],
        avoid_patterns=avoid_patterns,
        preferred_operators=preferred_operators,
        avoid_operators=avoid_operators,
        num_alphas=state.num_alphas_target,
        exploration_weight=exploration_weight,
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
            "alternatives_count": len(alternatives_considered)
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
