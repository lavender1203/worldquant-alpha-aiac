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

import re
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
)

# Initialize Validators (Singleton-ish)
_VALIDATOR = ExpressionValidator()


_OPERATOR_ALIASES = {
    "ts_stddev": "ts_std_dev",
    "ts_std": "ts_std_dev",
    "group_stddev": "group_std_dev",
}


def _canonicalize_operator_aliases(expression: str) -> str:
    """Normalize common LLM operator aliases to platform operator names."""
    if not expression:
        return expression

    normalized = expression
    for alias, canonical in _OPERATOR_ALIASES.items():
        normalized = re.sub(
            rf"\b{re.escape(alias)}\s*\(",
            f"{canonical}(",
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized


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
    task_config = config.get("configurable", {}).get("task_config", {}) if config else {}
    
    # Reset deduplicator for this batch
    batch_dedup = ExpressionDeduplicator(similarity_threshold=0.90)
    
    updated_alphas = []
    valid_count = 0
    syntax_errors = []
    semantic_errors = []
    duplicate_count = 0
    type_warnings = []
    
    logger.info(f"[{node_name}] Starting batch validation | count={len(state.pending_alphas)}")
    
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
    
    operator_names = [
        str(op.get("name") if isinstance(op, dict) else op)
        for op in state.operators
        if (op.get("name") if isinstance(op, dict) else op)
    ]

    # Initialize semantic validator with field type/operator info.
    semantic_validator = AlphaSemanticValidator(
        fields=state.fields,
        operators=operator_names or None,
        strict_field_check=False,
        strict_type_check=True
    )
    
    for alpha in state.pending_alphas:
        expression = _canonicalize_operator_aliases(alpha.expression)
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
                            is_valid = False
                            error = "; ".join(sem_result.errors[:3])
                            semantic_errors.extend(sem_result.errors[:2])

                    # Step 4: Task-level hard constraints. These are enforced
                    # after syntax/semantic checks so valid-looking expressions
                    # cannot bypass explicit mining instructions.
                    if is_valid:
                        constraint_errors = _validate_task_constraints(
                            expression=expression,
                            allowed_fields=allowed_fields,
                            fields=state.fields,
                            task_config=task_config,
                            candidate_metadata=alpha.metadata,
                        )
                        if constraint_errors:
                            is_valid = False
                            error = "; ".join(constraint_errors[:3])
                            semantic_errors.extend(constraint_errors[:2])
                            
            except Exception as e:
                is_valid = False
                error = f"Validation Exception: {str(e)}"
        
        updated_alpha = alpha.model_copy()
        if expression != alpha.expression:
            updated_alpha.original_expression = updated_alpha.original_expression or alpha.expression
            updated_alpha.expression = expression
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


def _validate_task_constraints(
    expression: str,
    allowed_fields: List[str],
    task_config: Dict,
    fields: Optional[List[Dict]] = None,
    candidate_metadata: Optional[Dict] = None,
) -> List[str]:
    """Enforce MiningTask.config constraints before simulation."""
    if not task_config:
        return []

    errors = []
    used_fields = _extract_used_fields(expression, allowed_fields)
    used_operator_calls = _extract_used_operators(expression)
    used_operators = set(used_operator_calls)
    candidate_metadata = candidate_metadata or {}
    is_trade_when_probe = (
        candidate_metadata.get("source") == "first_order_operator_probe"
        and candidate_metadata.get("probe_operator") == "trade_when"
    )

    max_fields = task_config.get("max_fields")
    if max_fields is not None and len(used_fields) > int(max_fields):
        errors.append(
            f"Too many fields: {len(used_fields)} > {int(max_fields)} "
            f"({', '.join(sorted(used_fields))})"
        )

    min_fields = task_config.get("min_fields")
    if min_fields is not None and len(used_fields) < int(min_fields):
        errors.append(
            f"Too few fields: {len(used_fields)} < {int(min_fields)} "
            f"({', '.join(sorted(used_fields)) or 'none'})"
        )

    exact_fields = task_config.get("exact_fields")
    if exact_fields is not None and len(used_fields) != int(exact_fields):
        errors.append(
            f"Wrong field count: {len(used_fields)} != {int(exact_fields)} "
            f"({', '.join(sorted(used_fields)) or 'none'})"
        )

    max_operator_count = task_config.get("max_operator_count")
    if max_operator_count is not None and len(used_operator_calls) > int(max_operator_count):
        errors.append(
            f"Too many operators: {len(used_operator_calls)} > {int(max_operator_count)}"
        )

    if (
        _config_bool(task_config.get("no_trade_when"), default=False)
        and "trade_when" in used_operators
        and not is_trade_when_probe
    ):
        errors.append("Forbidden operator: trade_when")

    diagnostic_operators = {"self_corr", "inst_pnl", "generate_stats"}
    if not _config_bool(task_config.get("allow_diagnostic_operators"), default=False):
        blocked_diagnostic = used_operators & diagnostic_operators
        if blocked_diagnostic:
            errors.append(
                "Diagnostic/meta operators are not valid alpha signal operators: "
                f"{', '.join(sorted(blocked_diagnostic))}"
            )

    avoid_fields = set(_config_list(task_config.get("avoid_fields")))
    blocked_fields = used_fields & avoid_fields
    if blocked_fields:
        errors.append(f"Forbidden fields used: {', '.join(sorted(blocked_fields))}")

    avoid_operators = set(_config_list(task_config.get("avoid_operators")))
    blocked_operators = used_operators & avoid_operators
    if blocked_operators:
        errors.append(f"Forbidden operators used: {', '.join(sorted(blocked_operators))}")

    avoid_prefixes = tuple(_config_list(task_config.get("avoid_operator_prefixes")))
    if avoid_prefixes:
        prefixed = sorted(
            op for op in used_operators
            if any(op.startswith(prefix) for prefix in avoid_prefixes)
        )
        if prefixed:
            errors.append(f"Forbidden operator prefixes used: {', '.join(prefixed)}")

    min_ts_delta_window = task_config.get("min_ts_delta_window")
    if min_ts_delta_window is not None:
        errors.extend(
            _minimum_window_errors(
                expression=expression,
                operator="ts_delta",
                min_window=int(min_ts_delta_window),
            )
        )

    min_ts_corr_window = task_config.get("min_ts_corr_window")
    if min_ts_corr_window is not None:
        errors.extend(
            _minimum_window_errors(
                expression=expression,
                operator="ts_corr",
                min_window=int(min_ts_corr_window),
            )
        )

    if _config_bool(task_config.get("avoid_raw_field_multiply"), default=False):
        raw_pairs = _raw_field_multiply_pairs(expression, allowed_fields)
        if raw_pairs:
            formatted = ", ".join(f"{left}*{right}" for left, right in raw_pairs[:3])
            errors.append(
                "Raw field multiplication is blocked by the current low-turnover strategy; "
                f"smooth or normalize fields before multiplying ({formatted})"
            )

    vector_fields = _extract_vector_fields(fields or [])
    used_vector_fields = used_fields & vector_fields
    if "reverse" in used_operators and (used_vector_fields or _uses_vec_operator(expression)):
        detail = ", ".join(sorted(used_vector_fields)) or "vec_* expression"
        errors.append(
            "reverse() is not allowed on VECTOR/event-derived signals; "
            f"use a sign-safe non-reverse skeleton instead ({detail})"
        )

    event_arithmetic_errors = _event_vector_arithmetic_errors(
        expression,
        used_vector_fields,
    )
    errors.extend(event_arithmetic_errors)

    errors.extend(_group_argument_errors(expression, fields or []))

    return errors


def _extract_used_fields(expression: str, allowed_fields: List[str]) -> set:
    used = set()
    for field in allowed_fields:
        if not field:
            continue
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(str(field))}(?![A-Za-z0-9_])"
        if re.search(pattern, expression):
            used.add(str(field))
    return used


def _extract_used_operators(expression: str) -> List[str]:
    return [
        match.group(1).lower()
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression)
    ]


def _minimum_window_errors(expression: str, operator: str, min_window: int) -> List[str]:
    if min_window <= 1:
        return []

    errors = []
    pattern = rf"\b{re.escape(operator)}\s*\(([^()]*(?:\([^()]*\)[^()]*)*),\s*(-?\d+)\s*\)"
    for match in re.finditer(pattern, expression, flags=re.IGNORECASE):
        window = int(match.group(2))
        if window < min_window:
            errors.append(
                f"{operator} window too short for current strategy: {window} < {min_window}"
            )
    return errors


def _raw_field_multiply_pairs(expression: str, allowed_fields: List[str]) -> List[tuple]:
    if not allowed_fields:
        return []

    field_pattern = "|".join(re.escape(str(field)) for field in allowed_fields if field)
    if not field_pattern:
        return []

    pattern = rf"\bmultiply\s*\(\s*({field_pattern})\s*,\s*({field_pattern})\s*\)"
    return [
        (match.group(1), match.group(2))
        for match in re.finditer(pattern, expression, flags=re.IGNORECASE)
    ]


def _extract_vector_fields(fields: List[Dict]) -> set:
    vector_fields = set()
    for field in fields:
        if not isinstance(field, dict):
            continue
        field_type = str(field.get("type") or field.get("field_type") or "").upper()
        if field_type != "VECTOR":
            continue
        field_id = field.get("id") or field.get("name") or field.get("field_id")
        if field_id:
            vector_fields.add(str(field_id))
    return vector_fields


def _extract_group_fields(fields: List[Dict]) -> set:
    group_fields = set()
    for field in fields:
        if not isinstance(field, dict):
            continue
        field_type = str(field.get("type") or field.get("field_type") or "").upper()
        if field_type != "GROUP":
            continue
        field_id = field.get("id") or field.get("name") or field.get("field_id")
        if field_id:
            group_fields.add(str(field_id).lower())
    return group_fields


def _uses_vec_operator(expression: str) -> bool:
    return bool(re.search(r"\bvec_[A-Za-z0-9_]*\s*\(", expression or ""))


def _group_argument_errors(expression: str, fields: List[Dict]) -> List[str]:
    """Ensure group operators receive real group categories, not numeric fields."""
    group_ops = {
        "group_rank",
        "group_neutralize",
        "group_zscore",
        "group_scale",
        "group_mean",
        "group_median",
        "group_min",
        "group_max",
    }
    builtins = {"sector", "industry", "subindustry", "exchange", "country", "market"}
    group_fields = _extract_group_fields(fields)
    errors = []

    for func_name, args in _iter_function_calls(expression):
        if func_name.lower() not in group_ops or len(args) < 2:
            continue
        group_arg_index = 2 if func_name.lower() == "group_mean" and len(args) >= 3 else 1
        group_arg = args[group_arg_index].strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", group_arg):
            errors.append(f"Invalid group argument for {func_name}: {group_arg[:60]}")
            continue
        group_lower = group_arg.lower()
        if group_lower not in builtins and group_lower not in group_fields:
            errors.append(
                f"Invalid group argument for {func_name}: {group_arg}. "
                "Use sector, industry, subindustry, exchange, country, market, or a GROUP field."
            )

    return errors[:3]


def _iter_function_calls(expression: str) -> List[tuple]:
    calls = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression or ""):
        func_name = match.group(1)
        open_idx = expression.find("(", match.start())
        depth = 0
        close_idx = None
        for idx in range(open_idx, len(expression)):
            char = expression[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close_idx = idx
                    break
        if close_idx is None:
            continue
        calls.append((func_name, _split_args(expression[open_idx + 1:close_idx])))
    return calls


def _split_args(args_text: str) -> List[str]:
    args = []
    depth = 0
    start = 0
    for idx, char in enumerate(args_text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            arg = args_text[start:idx].strip()
            if arg:
                args.append(arg)
            start = idx + 1
    tail = args_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def _event_vector_arithmetic_errors(expression: str, vector_fields: set) -> List[str]:
    """Block arithmetic before VECTOR/event aggregation.

    BRAIN event inputs must be aggregated with vec_* first. Expressions such as
    vec_avg(subtract(event_a, event_b)) fail on platform; the valid skeleton is
    subtract(vec_avg(event_a), vec_avg(event_b)).
    """
    if not vector_fields:
        return []

    errors = []
    arithmetic_ops = ("add", "subtract", "multiply", "divide")

    if re.search(
        rf"\bvec_[A-Za-z0-9_]*\s*\(\s*({'|'.join(arithmetic_ops)})\s*\(",
        expression or "",
        flags=re.IGNORECASE,
    ):
        errors.append(
            "VECTOR/event fields must be aggregated before arithmetic: use "
            "subtract(vec_avg(event_a), vec_avg(event_b)), not "
            "vec_avg(subtract(event_a, event_b))"
        )

    for field in vector_fields:
        field_pattern = re.escape(str(field))
        for op in arithmetic_ops:
            raw_arg_pattern = (
                rf"\b{op}\s*\(\s*(?<![A-Za-z0-9_]){field_pattern}(?![A-Za-z0-9_])"
                rf"|,\s*(?<![A-Za-z0-9_]){field_pattern}(?![A-Za-z0-9_])"
            )
            if re.search(raw_arg_pattern, expression or "", flags=re.IGNORECASE):
                errors.append(
                    f"Operator {op} cannot consume raw VECTOR/event field {field}; "
                    "wrap each event field with vec_avg/vec_sum before arithmetic"
                )
                break

    return errors[:3]


def _config_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [str(value).strip().lower()] if str(value).strip() else []


def _config_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
    duplicate_indices = [
        i for i in invalid_indices
        if "duplicate" in (state.pending_alphas[i].validation_error or "").lower()
    ]
    invalid_indices = [i for i in invalid_indices if i not in set(duplicate_indices)]
    
    if duplicate_indices:
        logger.info(f"[{node_name}] Skipping duplicate corrections | count={len(duplicate_indices)}")

    if not invalid_indices:
        logger.info(f"[{node_name}] No non-duplicate alphas need correction")
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
                    fixed = _canonicalize_operator_aliases(fixed)
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
                    metadata = dict(updated_alpha.metadata or {})
                    if metadata.get("source") == "first_order_operator_probe":
                        probe_operator = str(metadata.get("probe_operator") or "").lower()
                        fixed_operators = _extract_used_operators(fixed)
                        if probe_operator and probe_operator not in fixed_operators:
                            metadata["source"] = "self_corrected_from_first_order_operator_probe"
                            metadata["original_probe_operator"] = probe_operator
                            metadata.pop("probe_operator", None)
                            metadata["operator_skeleton"] = ",".join(sorted(fixed_operators)) or "field_only"
                            updated_alpha.metadata = metadata
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
