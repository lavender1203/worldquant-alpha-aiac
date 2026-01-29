"""
Validation nodes for LangGraph workflow.

Redesigned based on RD-Agent principles:
- Learn from similar errors and their fixes
- Extract transferable knowledge from corrections
- No preconceived biases about error handling

Contains:
- node_validate: Batch validate alpha expressions
- node_self_correct: Attempt to fix invalid alphas with error pattern learning
"""

import time
from typing import Dict, List, Optional
from loguru import logger
from langchain_core.runnables import RunnableConfig

from backend.agents.graph.state import MiningState
from backend.agents.graph.nodes.base import record_trace, _debug_log
from backend.agents.services import LLMService
from backend.agents.prompts import SELF_CORRECT_SYSTEM, SELF_CORRECT_USER, build_self_correct_prompt

from validator import ExpressionValidator
from backend.alpha_semantic_validator import (
    AlphaSemanticValidator,
    ExpressionDeduplicator,
    load_operators_from_db,
)

# Initialize Validators (Singleton-ish)
_VALIDATOR = ExpressionValidator()


# =============================================================================
# NODE: Validate
# =============================================================================

async def node_validate(state: MiningState, config: RunnableConfig = None) -> Dict:
    """
    Batch validate ALL pending alpha expressions.
    
    Enhanced with:
    - Semantic type validation (MATRIX/VECTOR constraints)
    - Deduplication gate (skip already-seen expressions)
    
    Input State:
        - pending_alphas
        - fields (with type info for semantic validation)
    
    Output Updates:
        - pending_alphas (with validation result)
        - trace_steps
    """
    start_time = time.time()
    node_name = "VALIDATE"
    
    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    
    # Reset deduplicator for this batch
    batch_dedup = ExpressionDeduplicator(similarity_threshold=0.90)
    
    updated_alphas = []
    valid_count = 0
    syntax_errors = []
    semantic_errors = []
    duplicate_count = 0
    type_warnings = []
    
    logger.info(f"[{node_name}] Starting batch validation | count={len(state.pending_alphas)}")
    
    # Ensure operators are loaded from DB for semantic validation
    await load_operators_from_db()
    
    # Build field list for validators
    allowed_fields = []
    for f in state.fields:
        if isinstance(f, dict):
            allowed_fields.append(f.get("id", f.get("name")))
        else:
            allowed_fields.append(str(f))
    
    _debug_log("D", "nodes.py:validate:fields", "Allowed fields for validation", {
        "allowed_fields": allowed_fields,
        "fields_count": len(allowed_fields),
        "expressions": [a.expression[:100] for a in state.pending_alphas]
    })
    
    # Initialize semantic validator with field type info
    semantic_validator = AlphaSemanticValidator(
        fields=state.fields,
        operators=None,
        strict_field_check=False,
        strict_type_check=True
    )
    
    for alpha in state.pending_alphas:
        expression = alpha.expression
        is_valid = True
        error = None
        warnings = []
        
        if not expression or not expression.strip():
            is_valid = False
            error = "Empty expression"
        else:
            try:
                # Step 1: Deduplication check
                is_dup, dup_reason = batch_dedup.is_duplicate(expression)
                if is_dup:
                    is_valid = False
                    error = f"Duplicate: {dup_reason}"
                    duplicate_count += 1
                else:
                    batch_dedup.add(expression)
                    
                    # Step 2: Syntax validation
                    syntax_result = _VALIDATOR.check_expression(
                        expression, allowed_fields=allowed_fields
                    )
                    if not syntax_result.get("valid", False):
                        is_valid = False
                        err_list = syntax_result.get("errors", [])
                        error = "; ".join(err_list) if err_list else "Syntax error"
                        syntax_errors.append(error)
                    else:
                        # Step 3: Semantic validation (type constraints)
                        sem_result = semantic_validator.validate(expression)
                        
                        if sem_result.warnings:
                            warnings.extend(sem_result.warnings)
                            type_warnings.extend(sem_result.warnings[:2])
                        
                        if sem_result.errors:
                            warnings.extend(sem_result.errors)
                            semantic_errors.extend(sem_result.errors[:2])
                            
            except Exception as e:
                is_valid = False
                error = f"Validation Exception: {str(e)}"
        
        updated_alpha = alpha.model_copy()
        updated_alpha.is_valid = is_valid
        updated_alpha.validation_error = error
        
        if warnings and not error:
            updated_alpha.validation_error = f"[WARNINGS] {'; '.join(warnings[:3])}"
        
        if is_valid:
            valid_count += 1
        else:
            if error and "Duplicate" not in error:
                syntax_errors.append(f"{expression[:50]}... -> {error}")
        
        updated_alphas.append(updated_alpha)
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    _debug_log("D", "nodes.py:validate:result", "Validation complete", {
        "total": len(updated_alphas),
        "valid": valid_count,
        "invalid": len(updated_alphas) - valid_count,
        "duplicates": duplicate_count,
        "syntax_error_count": len(syntax_errors),
        "duration_ms": duration_ms,
        "pass_rate": round(valid_count / max(1, len(updated_alphas)) * 100, 1)
    })
    
    logger.info(
        f"[{node_name}] Complete | valid={valid_count}/{len(updated_alphas)} "
        f"duplicates={duplicate_count} type_warnings={len(type_warnings)}"
    )
    
    if syntax_errors:
        logger.warning(f"[{node_name}] Syntax Errors: {syntax_errors[:3]}")
    if semantic_errors:
        logger.warning(f"[{node_name}] Semantic Warnings: {semantic_errors[:3]}")
    
    trace_update = await record_trace(
        state, trace_service, node_name,
        {"count": len(updated_alphas)},
        {
            "valid_count": valid_count,
            "invalid_count": len(updated_alphas) - valid_count,
            "duplicate_count": duplicate_count,
            "type_warnings": type_warnings[:5],
            "failures": [
                {"expression": a.expression[:100], "error": a.validation_error}
                for a in updated_alphas if not a.is_valid
            ][:10]
        },
        duration_ms,
        "SUCCESS"
    )
    
    return {
        "pending_alphas": updated_alphas,
        **trace_update
    }


# =============================================================================
# NODE: Self-Correct
# =============================================================================

# Error knowledge base for learning from corrections
_ERROR_KNOWLEDGE_BASE: List[Dict] = []


def _categorize_error(error_message: str) -> str:
    """Categorize error type for knowledge matching."""
    error_lower = error_message.lower() if error_message else ""
    
    if "field" in error_lower or "unknown" in error_lower:
        return "field_name"
    elif "syntax" in error_lower or "parse" in error_lower:
        return "syntax"
    elif "operator" in error_lower or "function" in error_lower:
        return "operator_usage"
    elif "type" in error_lower or "matrix" in error_lower or "vector" in error_lower:
        return "type_error"
    elif "duplicate" in error_lower:
        return "duplicate"
    else:
        return "other"


def _find_similar_errors(
    error_message: str,
    error_type: str,
    knowledge_base: List[Dict],
    max_results: int = 3
) -> List[Dict]:
    """Find similar errors from knowledge base for learning."""
    similar = []
    error_category = _categorize_error(error_message)
    
    for entry in knowledge_base:
        if entry.get("error_category") == error_category:
            similar.append(entry)
            if len(similar) >= max_results:
                break
    
    return similar


def _record_correction(
    original_expression: str,
    fixed_expression: str,
    error_message: str,
    error_type: str,
    fix_description: str
) -> None:
    """Record a successful correction for future learning."""
    global _ERROR_KNOWLEDGE_BASE
    
    _ERROR_KNOWLEDGE_BASE.append({
        "failed_expression": original_expression,
        "fixed_expression": fixed_expression,
        "error": error_message,
        "error_category": _categorize_error(error_message),
        "fix_description": fix_description
    })
    
    # Keep knowledge base manageable
    if len(_ERROR_KNOWLEDGE_BASE) > 100:
        _ERROR_KNOWLEDGE_BASE = _ERROR_KNOWLEDGE_BASE[-50:]


async def node_self_correct(
    state: MiningState,
    llm_service: LLMService,
    config: RunnableConfig = None
) -> Dict:
    """
    Batch attempt to fix ALL invalid alphas with error pattern learning.
    
    Redesigned based on RD-Agent principles:
    - Learn from similar errors and their successful fixes
    - Extract transferable knowledge ("If this error, then this fix")
    - Multiple fix approaches without prescriptive bias
    
    Input State:
        - pending_alphas
        - retry_count
    
    Output Updates:
        - pending_alphas (updated)
        - retry_count
        - knowledge_extracted (new corrections for future learning)
    """
    start_time = time.time()
    node_name = "SELF_CORRECT"
    
    trace_service = config.get("configurable", {}).get("trace_service") if config else None
    
    # Identify invalid alphas
    invalid_indices = [
        i for i, a in enumerate(state.pending_alphas)
        if not a.is_valid
    ]
    
    if not invalid_indices:
        logger.info(f"[{node_name}] No alphas need correction")
        return {"retry_count": state.retry_count + 1}
    
    logger.info(f"[{node_name}] Starting batch fix | count={len(invalid_indices)} pass={state.retry_count + 1}")
    
    # Build allowed fields list
    allowed_fields = []
    for f in state.fields[:50]:
        fid = f.get('id', f.get('name', ''))
        if fid:
            allowed_fields.append(fid)
    
    updated_alphas = list(state.pending_alphas)
    fixed_count = 0
    corrections_made = []
    knowledge_extracted = []
    
    for idx in invalid_indices:
        current = state.pending_alphas[idx]
        error_message = current.validation_error or "Unknown error"
        error_type = _categorize_error(error_message)
        
        # Find similar errors for learning
        similar_errors = _find_similar_errors(
            error_message, error_type, _ERROR_KNOWLEDGE_BASE
        )
        
        if similar_errors:
            logger.debug(f"[{node_name}] Found {len(similar_errors)} similar errors for learning")
        
        # Use enhanced prompt builder with error learning
        prompt = build_self_correct_prompt(
            expression=current.expression,
            error_message=error_message,
            error_type=error_type,
            available_fields=allowed_fields,
            similar_errors=similar_errors if similar_errors else None
        )
        
        try:
            response = await llm_service.call(
                system_prompt=SELF_CORRECT_SYSTEM,
                user_prompt=prompt,
                temperature=0.3,
                json_mode=True
            )
            
            updated_alpha = current.model_copy()
            updated_alpha.correction_attempts += 1
            if not updated_alpha.original_expression:
                updated_alpha.original_expression = current.expression
            
            if response.success and response.parsed:
                parsed = response.parsed
                
                # Handle both old format (fixed_expression) and new format (fix.fixed_expression)
                fix_data = parsed.get("fix", {})
                fixed = fix_data.get("fixed_expression") if isinstance(fix_data, dict) else None
                if not fixed:
                    fixed = parsed.get("fixed_expression")
                
                if fixed:
                    # Get fix description
                    changes_made = fix_data.get("changes_made", "") if isinstance(fix_data, dict) else ""
                    if not changes_made:
                        changes_made = parsed.get("changes_made", "")
                    
                    corrections_made.append({
                        "original": current.expression,
                        "fixed": fixed,
                        "error": error_message,
                        "changes": changes_made
                    })
                    
                    updated_alpha.expression = fixed
                    updated_alpha.is_valid = None
                    updated_alpha.validation_error = None
                    fixed_count += 1
                    
                    # Record for future learning
                    _record_correction(
                        original_expression=current.expression,
                        fixed_expression=fixed,
                        error_message=error_message,
                        error_type=error_type,
                        fix_description=changes_made
                    )
                    
                    # Extract transferable knowledge
                    knowledge = parsed.get("knowledge_extracted")
                    if knowledge:
                        knowledge_extracted.append(knowledge)
            
            updated_alphas[idx] = updated_alpha
            
        except Exception as e:
            logger.error(f"[{node_name}] Fix failed for index {idx}: {e}")
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    logger.info(f"[{node_name}] Complete | fixed_attempts={fixed_count}/{len(invalid_indices)}")
    
    if knowledge_extracted:
        logger.info(f"[{node_name}] Extracted {len(knowledge_extracted)} knowledge rules")
        for rule in knowledge_extracted[:3]:
            logger.debug(f"[{node_name}] Rule: {rule}")
    
    trace_update = await record_trace(
        state, trace_service, node_name,
        {
            "fix_targets": len(invalid_indices),
            "similar_errors_found": sum(1 for _ in _ERROR_KNOWLEDGE_BASE)
        },
        {
            "fixed_count": fixed_count,
            "corrections": corrections_made,
            "knowledge_extracted": knowledge_extracted
        },
        duration_ms,
        "SUCCESS"
    )
    
    return {
        "pending_alphas": updated_alphas,
        "retry_count": state.retry_count + 1,
        **trace_update
    }
