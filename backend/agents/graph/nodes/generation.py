"""
Generation nodes for LangGraph workflow.

Redesigned based on RD-Agent's hypothesis-driven approach:
- Each experiment tests a specific hypothesis
- Knowledge transfer from previous experiments
- Balanced exploration and exploitation
- No preconceived biases

P0 Fix: Added diversity constraints to prevent template traps

ENHANCED (v2.1):
- P0-1: Field availability pre-check before generation
- P0-2: Hypothesis-implementation alignment enforcement
- P1-1: CoSTEER hard constraint integration
- P2-2: Multi-fidelity pre-screening

Contains:
- node_rag_query: Retrieve patterns from knowledge base
- node_distill_context: Distill concepts from fields
- node_hypothesis: Generate investment hypotheses
- node_code_gen: Generate alpha expressions (ENHANCED)
"""

import time
import random
from typing import Dict, List, Optional
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
from backend.diversity_tracker import get_operator_diversity_manager

# P0/P1/P2 Enhancements
from backend.agents.optimization_integration import (
    enhanced_pre_generation_check,
    enhanced_post_generation_validate,
    get_costeer_hard_constraints,
    pre_screen_expressions,
    get_smart_exploration_params,
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
    
    ENHANCED: Uses field-aware retrieval when fields are available.
    This adapts patterns to the actual field types in the dataset,
    fixing the knowledge-reality mismatch issue.
    
    Input State:
        - dataset_id, region, fields (optional)
    
    Output Updates:
        - patterns, pitfalls
        - trace_steps
    """
    start_time = time.time()
    node_name = "RAG_QUERY"
    
    logger.info(f"[{node_name}] Starting | task={state.task_id} dataset={state.dataset_id}")
    
    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    
    try:
        # Use field-aware retrieval if fields are available
        # This generates patterns adapted to actual available fields
        if state.fields and len(state.fields) > 0:
            logger.debug(f"[{node_name}] Using field-aware retrieval with {len(state.fields)} fields")
            result = await rag_service.query_with_field_adaptation(
                dataset_id=state.dataset_id,
                fields=state.fields,
                region=state.region,
                max_patterns=8,  # Get more patterns since some are adaptive
                max_pitfalls=10
            )
        else:
            # Fallback to standard query
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
        
        # Safe access to patterns and pitfalls
        safe_patterns_list = list(result.patterns)[:3] if result.patterns else []
        safe_pitfalls_list = list(result.pitfalls)[:3] if result.pitfalls else []
        
        trace_update = await record_trace(
            state, trace_service, step_type=node_name,
            input_data={"dataset_id": state.dataset_id, "region": state.region},
            output_data={
                "patterns_count": len(result.patterns) if result.patterns else 0,
                "pitfalls_count": len(result.pitfalls) if result.pitfalls else 0,
                "top_patterns": [p.get('pattern', '') for p in safe_patterns_list],
                "top_pitfalls": [p.get('pattern', '') for p in safe_pitfalls_list]
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
    
    safe_patterns = list(state.patterns)[:3] if state.patterns else []
    success_patterns_text = "\n".join([
        f"- {p.get('pattern', '')}" for p in safe_patterns
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
            json_mode=True
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
    
    if not focused_fields:
        logger.warning(f"[{node_name}] Distillation yielded 0 fields. Falling back to top 30.")
        focused_fields = state.fields[:30]
    
    logger.info(f"[{node_name}] Complete | concepts={selected_concepts} focused={len(focused_fields)}")
    
    trace_update = await record_trace(
        state, trace_service, node_name,
        {"field_count": len(state.fields), "categories": list(categories.keys())},
        {
            "selected_concepts": selected_concepts,
            "focused_count": len(focused_fields),
            "reasoning": reasoning
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
    
    # Safe list operations for hypothesis generation
    if state.focused_fields:
        target_fields = list(state.focused_fields)
    elif state.fields:
        target_fields = list(state.fields)[:20]
    else:
        target_fields = []
    
    safe_operators = list(state.operators)[:30] if state.operators else []
    safe_patterns = list(state.patterns)[:5] if state.patterns else []
    safe_pitfalls = list(state.pitfalls)[:5] if state.pitfalls else []
    
    # Build prompt context
    prompt_context = PromptContext(
        dataset_id=state.dataset_id,
        dataset_description=state.dataset_description or "",
        dataset_category=state.dataset_category or "",
        region=state.region,
        universe=state.universe,
        fields=target_fields,
        operators=safe_operators,
        success_patterns=safe_patterns,
        failure_pitfalls=safe_pitfalls,
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
        json_mode=True
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
    
    # New: Get experiment feedback for learning
    experiment_feedback = strategy_dict.get("experiment_feedback", [])
    target_hypothesis = strategy_dict.get("target_hypothesis")
    
    # P1 Enhancement: Get SOTA info and CoSTEER constraints
    sota_info = strategy_dict.get("sota_info")
    costeer_feedback = strategy_dict.get("costeer_feedback", {})
    
    logger.info(
        f"[{node_name}] Starting | task={state.task_id} "
        f"temp={temperature:.2f} explore={exploration_weight:.2f} "
        f"feedback_len={len(experiment_feedback)} "
        f"has_sota={sota_info is not None} has_costeer={bool(costeer_feedback)}"
    )
    
    # =========================================================================
    # P0-1 ENHANCEMENT: Pre-generation field availability check (SAFE)
    # =========================================================================
    # Get hypothesis key_fields for mandatory field validation
    hypothesis_key_fields = []
    try:
        for h in (state.hypotheses or []):
            if isinstance(h, dict):
                kf = h.get("key_fields", [])
                if isinstance(kf, list):
                    hypothesis_key_fields.extend(kf)
    except Exception as e:
        logger.warning(f"[{node_name}] Failed to extract hypothesis key_fields: {e}")
    
    # Pre-check field availability (with safe list handling)
    try:
        base_fields = state.focused_fields if state.focused_fields else state.fields
        # Ensure we have a list
        if not isinstance(base_fields, list):
            base_fields = list(base_fields) if base_fields else []
        target_fields = base_fields[:30] if len(base_fields) > 30 else base_fields
        
        filtered_fields, field_check_result = enhanced_pre_generation_check(
            fields=target_fields,
            region=state.region,
            universe=state.universe,
            hypothesis_key_fields=hypothesis_key_fields,
            dataset_id=state.dataset_id
        )
        
        if field_check_result.warning_message:
            logger.warning(f"[{node_name}] Field check warning: {field_check_result.warning_message}")
        
        # Safe access to list attributes
        recommended = field_check_result.recommended_fields
        if not isinstance(recommended, list):
            recommended = []
        
        _debug_log("A", "nodes.py:code_gen:field_check", "Field availability pre-check", {
            "total_fields": len(target_fields) if target_fields else 0,
            "available_fields": len(filtered_fields) if filtered_fields else 0,
            "blocked_fields": len(field_check_result.blocked_fields) if field_check_result.blocked_fields else 0,
            "recommended_fields": recommended[:5] if recommended else [],
        })
        
        # Use filtered fields for generation
        target_fields = filtered_fields if filtered_fields else target_fields
    except Exception as e:
        logger.warning(f"[{node_name}] P0-1 field check failed, using original fields: {e}")
        base_fields = state.focused_fields if state.focused_fields else state.fields
        target_fields = list(base_fields)[:30] if base_fields else []
        field_check_result = None
    
    # =========================================================================
    # P1-1 ENHANCEMENT: Get CoSTEER hard constraints (SAFE)
    # =========================================================================
    try:
        costeer_hard_constraints = get_costeer_hard_constraints()
    except Exception as e:
        logger.warning(f"[{node_name}] P1-1 CoSTEER constraints failed: {e}")
        costeer_hard_constraints = {"forbidden_fields": [], "forbidden_patterns": []}
    
    # Merge with any existing costeer_feedback
    # NOTE: costeer_feedback expects a specific format for build_alpha_generation_prompt:
    # - 'hard_constraints': list of dicts with keys: expression, issue, lesson
    # - 'forbidden_patterns': list of pattern strings
    # We convert our CoSTEER constraints to this format
    if not costeer_feedback:
        costeer_feedback = {}
    
    # Convert forbidden_patterns to hard_constraints format if not already present
    if "hard_constraints" not in costeer_feedback or not costeer_feedback["hard_constraints"]:
        costeer_feedback["hard_constraints"] = []  # Keep empty, patterns go elsewhere
    
    # Add forbidden patterns separately (the prompt builder expects this)
    existing_forbidden = costeer_feedback.get("forbidden_patterns", [])
    if isinstance(existing_forbidden, list):
        new_patterns = costeer_hard_constraints.get("forbidden_patterns", [])
        if isinstance(new_patterns, list):
            costeer_feedback["forbidden_patterns"] = list(existing_forbidden) + list(new_patterns)[:5]
        else:
            costeer_feedback["forbidden_patterns"] = list(existing_forbidden)
    else:
        costeer_feedback["forbidden_patterns"] = costeer_hard_constraints.get("forbidden_patterns", [])[:5]
    
    # =========================================================================
    # P1-3 ENHANCEMENT: Smart exploration parameters (SAFE)
    # =========================================================================
    try:
        # Get adaptive exploration parameters based on current state
        smart_params = get_smart_exploration_params(
            current_progress=0.5,  # Default, can be computed from state
            max_iterations=10,
            iteration=getattr(state, 'iteration', 0)
        )
        
        # Override temperature if smart strategy suggests it
        if smart_params and smart_params.get("temperature"):
            temperature = smart_params["temperature"]
            logger.debug(f"[{node_name}] Using smart temperature: {temperature:.2f}")
    except Exception as e:
        logger.warning(f"[{node_name}] P1-3 smart params failed: {e}")
        smart_params = {}
    
    # P0 FIX: Get diversity constraints to prevent template traps (SAFE)
    try:
        diversity_manager = get_operator_diversity_manager()
        diversity_constraints = diversity_manager.get_diversity_constraints(state.num_alphas_target)
        
        # Safe access to diversity constraints
        if not isinstance(diversity_constraints, dict):
            diversity_constraints = {}
        
        required_archetypes = diversity_constraints.get('required_archetypes', [])
        if not isinstance(required_archetypes, list):
            required_archetypes = []
        
        logger.debug(
            f"[{node_name}] Diversity constraints | "
            f"required_archetypes={required_archetypes[:4]} "
            f"repetition_warning={diversity_constraints.get('repetition_warning', False)}"
        )
    except Exception as div_e:
        logger.warning(f"[{node_name}] Diversity constraints failed: {div_e}")
        diversity_constraints = {}
    
    # Build structured prompt context (using filtered fields from P0-1)
    # Safe list operations
    safe_operators = list(state.operators)[:50] if state.operators else []
    safe_patterns = list(state.patterns)[:5] if state.patterns else []
    safe_pitfalls = list(state.pitfalls)[:5] if state.pitfalls else []
    safe_hypotheses = list(state.hypotheses)[:3] if state.hypotheses else []
    
    # P0-1: Safe access to field check results
    recommended_fields = []
    blocked_fields = []
    if field_check_result:
        recommended_fields = list(field_check_result.recommended_fields or [])
        blocked_fields = list(field_check_result.blocked_fields or [])[:10]
    
    # P1-1: Safe access to CoSTEER constraints
    forbidden_patterns = []
    if isinstance(costeer_hard_constraints, dict):
        forbidden_patterns = list(costeer_hard_constraints.get("forbidden_patterns", []) or [])
    
    prompt_context = PromptContext(
        dataset_id=state.dataset_id,
        dataset_description=state.dataset_description or "",
        dataset_category=state.dataset_category or "",
        region=state.region,
        universe=state.universe,
        fields=target_fields,  # P0-1: Use pre-checked fields
        operators=safe_operators,
        success_patterns=safe_patterns,
        failure_pitfalls=safe_pitfalls,
        preferred_fields=list(preferred_fields or []) + recommended_fields,  # P0-1: Add recommended
        avoid_fields=list(avoid_fields or []) + blocked_fields,  # P0-1: Add blocked
        focus_hypotheses=list(focus_hypotheses or []) + [
            h.get("statement", h.get("idea", str(h))) if isinstance(h, dict) else str(h)
            for h in safe_hypotheses
        ],
        avoid_patterns=list(avoid_patterns or []) + forbidden_patterns,  # P1-1
        num_alphas=state.num_alphas_target,
        exploration_weight=exploration_weight,
    )
    
    # Use enhanced prompt builder with hypothesis, feedback, SOTA, CoSTEER, and diversity context
    prompt = build_alpha_generation_prompt(
        prompt_context,
        target_hypothesis=target_hypothesis,
        experiment_feedback=experiment_feedback,
        diversity_constraints=diversity_constraints,  # P0 FIX: Enforce diversity
        sota_info=sota_info,  # P1: SOTA comparison target
        costeer_feedback=costeer_feedback,  # P1: CoSTEER hard/soft constraints
    )
    
    try:
        response = await llm_service.call(
            system_prompt=ALPHA_GENERATION_SYSTEM,
            user_prompt=prompt,
            temperature=temperature,
            json_mode=True
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
                "complexity": alpha_data.get("complexity", "unknown"),
                "novelty_level": alpha_data.get("novelty_level", "unknown"),
            }
            
            if candidate.expression and candidate.expression.strip():
                # =========================================================
                # P0-2 + P2-2 ENHANCEMENT: Post-generation validation (SAFE)
                # =========================================================
                # This combines hypothesis alignment check (P0-2) with
                # multi-fidelity pre-screening (P2-2)
                
                try:
                    # Find matching hypothesis for this alpha
                    matching_hypo = None
                    if hypothesis_text and state.hypotheses:
                        for h in (state.hypotheses or []):
                            try:
                                hypo_text = h.get("statement", h.get("idea", str(h))) if isinstance(h, dict) else str(h)
                                if hypo_text and (hypo_text in hypothesis_text or hypothesis_text in hypo_text):
                                    matching_hypo = h if isinstance(h, dict) else {"statement": str(h)}
                                    break
                            except Exception:
                                continue
                    
                    hypothesis_dict = matching_hypo or {"statement": hypothesis_text, "key_fields": []}
                    
                    # Convert state.fields to list safely
                    all_fields = list(state.fields) if state.fields else []
                    
                    # Enhanced validation (P0-2 + P1-1 + P2-2)
                    is_valid, corrected_expr, validation_issues = enhanced_post_generation_validate(
                        expression=candidate.expression,
                        hypothesis=hypothesis_dict,
                        all_fields=all_fields
                    )
                    
                    if not is_valid:
                        logger.warning(
                            f"[{node_name}] Alpha rejected by P0-2/P1-1/P2-2 validation: "
                            f"{validation_issues[:2] if validation_issues else ['unknown']}"
                        )
                        # Mark as invalid but still track for learning
                        candidate.metadata["_validation_rejected"] = True
                        candidate.metadata["_rejection_reasons"] = validation_issues or []
                        candidate.is_valid = False
                        continue  # Skip this alpha
                    
                    # Use corrected expression if different
                    if corrected_expr and corrected_expr != candidate.expression:
                        logger.info(
                            f"[{node_name}] Auto-corrected expression: "
                            f"{candidate.expression[:40]}... -> {corrected_expr[:40]}..."
                        )
                        candidate.metadata["_original_expression"] = candidate.expression
                        candidate.expression = corrected_expr
                        
                except Exception as val_err:
                    logger.warning(f"[{node_name}] P0-2/P2-2 validation failed, allowing alpha: {val_err}")
                    # On validation failure, allow the alpha to proceed
                
                # === Legacy P0 ENHANCEMENT: HYPOTHESIS-FIELD BINDING VALIDATION ===
                # (Kept for backward compatibility, now redundant with P0-2)
                from backend.field_availability_checker import get_field_availability_checker
                
                field_checker = get_field_availability_checker()
                binding_valid = True
                binding_warning = ""
                
                if hypothesis_text and state.hypotheses:
                    # Find matching hypothesis for this alpha
                    matching_hypo = None
                    for h in state.hypotheses:
                        hypo_text = h.get("statement", h.get("idea", str(h))) if isinstance(h, dict) else str(h)
                        if hypo_text and (hypo_text in hypothesis_text or hypothesis_text in hypo_text):
                            matching_hypo = h if isinstance(h, dict) else {"statement": str(h)}
                            break
                    
                    if matching_hypo:
                        # Get key_fields from hypothesis
                        key_fields = matching_hypo.get("key_fields", [])
                        
                        if key_fields:
                            # Validate field binding
                            binding_result = field_checker.validate_hypothesis_field_binding(
                                expression=candidate.expression,
                                hypothesis_key_fields=key_fields,
                                all_available_fields=[f.get("id", f.get("name", "")) for f in state.fields]
                            )
                            
                            binding_valid = binding_result.is_valid
                            binding_warning = binding_result.warning_message
                            
                            # Attach binding info to metadata
                            candidate.metadata["hypothesis_key_fields"] = key_fields
                            candidate.metadata["mandatory_fields_used"] = binding_result.mandatory_fields_used
                            candidate.metadata["binding_valid"] = binding_valid
                            
                            if not binding_valid:
                                logger.warning(
                                    f"[{node_name}] Field binding violation: {binding_warning[:100]}"
                                )
                                # P2: Consider rejecting or flagging for regeneration
                                candidate.metadata["needs_regeneration"] = True
                                candidate.metadata["substitution_suggestions"] = binding_result.substitution_suggestions
                        
                        # Also do the alignment check
                        from backend.agents.prompts.alignment import quick_alignment_check
                        is_aligned, alignment_issues = quick_alignment_check(
                            matching_hypo, candidate.expression, state.fields
                        )
                        
                        # Attach alignment info to metadata
                        candidate.metadata["is_aligned"] = is_aligned
                        candidate.metadata["alignment_issues"] = alignment_issues
                        
                        if not is_aligned:
                            logger.warning(
                                f"[{node_name}] Alignment warning for alpha: {alignment_issues[:2]}"
                            )
                
                # P2: Track field effectiveness
                field_checker.record_field_effectiveness(
                    expression=candidate.expression,
                    region=state.region,
                    success=binding_valid,  # Preliminary - will be updated after simulation
                    sharpe=0.0
                )
                
                pending_alphas.append(candidate)
    
    # P0 FIX: Record usage in diversity manager
    for candidate in pending_alphas:
        diversity_manager.record_usage(candidate.expression)
    
    # Log diversity stats
    pattern_stats = diversity_manager.get_pattern_statistics()
    logger.debug(
        f"[{node_name}] Diversity stats | "
        f"unique_ratio={pattern_stats.get('unique_ratio', 0):.2f} "
        f"consecutive_same={pattern_stats.get('consecutive_same', 0)}"
    )
    
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
