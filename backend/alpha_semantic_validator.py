"""
Alpha Semantic Validator - Enhanced validation with MATRIX/VECTOR type constraints.

This module provides semantic validation beyond syntax checking:
1. Field existence validation
2. Operator existence validation  
3. MATRIX/VECTOR type constraint enforcement
4. Expression deduplication
5. Diversity scoring

P0-1: Core type/signature validation
"""

import re
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class FieldType(Enum):
    """BRAIN platform field types"""
    MATRIX = "MATRIX"  # Time-series data, supports ts_* operators
    VECTOR = "VECTOR"  # Cross-sectional/static data, supports vec_* operators
    GROUP = "GROUP"    # Grouping fields (sector, industry, etc.)
    UNKNOWN = "UNKNOWN"


@dataclass
class FieldInfo:
    """Field metadata for validation"""
    field_id: str
    field_type: FieldType = FieldType.UNKNOWN
    coverage: float = 1.0
    alpha_count: int = 0
    pyramid_multiplier: float = 1.0
    description: str = ""
    field_name: str = ""
    
    @classmethod
    def from_dict(cls, d: Dict) -> "FieldInfo":
        field_type_str = d.get("type") or d.get("field_type") or "MATRIX"
        try:
            field_type = FieldType(field_type_str.upper()) if field_type_str else FieldType.UNKNOWN
        except ValueError:
            field_type = FieldType.UNKNOWN
            
        return cls(
            field_id=d.get("id") or d.get("name", ""),
            field_type=field_type,
            coverage=d.get("coverage", 1.0) or 1.0,
            alpha_count=d.get("alpha_count", 0) or 0,
            pyramid_multiplier=d.get("pyramid_multiplier", 1.0) or 1.0,
            description=d.get("description", ""),
            field_name=d.get("field_name") or d.get("name", "")
        )

    @property
    def is_group_like(self) -> bool:
        """BRAIN may expose group-valued fields as MATRIX in metadata."""
        if self.field_type == FieldType.GROUP:
            return True
        text = f"{self.field_id} {self.field_name} {self.description}".lower()
        return bool(
            re.search(r"(^|[_\W])group\d*($|[_\W])", text)
            or re.search(r"(^|[_\W])(bucket|cluster|classification|category|sector|industry|subindustry)($|[_\W])", text)
        )


# =============================================================================
# Operator Registry - Dynamic loading from database
# =============================================================================

class OperatorRegistry:
    """
    Global registry for operators loaded from database.
    
    Provides:
    - Async loading from database
    - In-memory caching
    - Category-based operator sets
    
    Note: No hardcoded fallback - operators must be synced from BRAIN platform.
    """
    
    _instance: Optional["OperatorRegistry"] = None
    
    def __init__(self):
        self._operators: Set[str] = set()
        self._operators_by_category: Dict[str, Set[str]] = {}
        self._loaded = False
        self._warned = False  # Only warn once
        self._load_lock = asyncio.Lock()
    
    @classmethod
    def get_instance(cls) -> "OperatorRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def operators(self) -> Set[str]:
        """Get all known operators."""
        if not self._operators and not self._warned:
            logger.warning("[OperatorRegistry] No operators loaded. Run 'POST /api/v1/operators/sync' first.")
            self._warned = True
        return self._operators
    
    @property
    def ts_operators(self) -> Set[str]:
        """Get time-series operators."""
        return self._operators_by_category.get("Time Series", set())
    
    @property
    def vec_operators(self) -> Set[str]:
        """Get vector operators."""
        return self._operators_by_category.get("Vector", set())
    
    @property
    def group_operators(self) -> Set[str]:
        """Get group operators."""
        return self._operators_by_category.get("Group", set())
    
    async def load_from_db(self, db=None) -> bool:
        """
        Load operators from database.
        
        Args:
            db: AsyncSession instance (optional, will create if not provided)
            
        Returns:
            True if loaded successfully
        """
        async with self._load_lock:
            if self._loaded and self._operators:
                return True
            
            try:
                if db is None:
                    from backend.database import AsyncSessionLocal
                    async with AsyncSessionLocal() as session:
                        return await self._load_operators(session)
                else:
                    return await self._load_operators(db)
            except Exception as e:
                logger.error(f"[OperatorRegistry] Failed to load from DB: {e}. Sync operators first.")
                return False
    
    async def _load_operators(self, db) -> bool:
        """Internal load implementation."""
        from sqlalchemy import select, func
        from backend.models import Operator
        
        # First check total count
        count_result = await db.execute(select(func.count()).select_from(Operator))
        total_count = count_result.scalar()
        logger.debug(f"[OperatorRegistry] Total operators in DB: {total_count}")
        
        # Load all operators (don't filter by is_active, some may be NULL)
        result = await db.execute(
            select(Operator.name, Operator.category)
        )
        rows = result.all()
        
        if not rows:
            logger.warning("[OperatorRegistry] No operators in database. Run 'POST /api/v1/operators/sync' first.")
            return False
        
        self._operators = set()
        self._operators_by_category = {}
        
        for name, category in rows:
            if not name:
                continue
            name_lower = name.lower()
            self._operators.add(name_lower)
            
            if category:
                if category not in self._operators_by_category:
                    self._operators_by_category[category] = set()
                self._operators_by_category[category].add(name_lower)
        
        self._loaded = True
        self._warned = False  # Reset warning flag after successful load
        logger.info(f"[OperatorRegistry] Loaded {len(self._operators)} operators from database")
        return True
    
    def reload(self):
        """Force reload on next access."""
        self._loaded = False
        self._warned = False
        self._operators = set()
        self._operators_by_category = {}


# Global registry instance
_operator_registry = OperatorRegistry.get_instance()


async def load_operators_from_db(db=None) -> Set[str]:
    """
    Load operators from database.
    
    Convenience function for async contexts.
    """
    await _operator_registry.load_from_db(db)
    return _operator_registry.operators


def get_known_operators() -> Set[str]:
    """
    Get known operators (sync).
    
    Returns cached operators or fallback if not loaded.
    """
    return _operator_registry.operators


# Built-in group fields (these are not operators, kept hardcoded)
BUILTIN_GROUPS = {"sector", "subindustry", "industry", "exchange", "country", "market"}


@dataclass
class SemanticValidationResult:
    """Result of semantic validation"""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Extracted info
    used_fields: Set[str] = field(default_factory=set)
    used_operators: Set[str] = field(default_factory=set)
    field_types_used: Set[str] = field(default_factory=set)
    
    # Metrics
    complexity_score: float = 0.0
    diversity_score: float = 0.0
    
    def add_error(self, msg: str):
        self.errors.append(msg)
        self.valid = False
        
    def add_warning(self, msg: str):
        self.warnings.append(msg)


class AlphaSemanticValidator:
    """
    Enhanced semantic validator for alpha expressions.
    
    Validates:
    - Field existence in dataset
    - Operator existence in platform
    - Type constraints (MATRIX vs VECTOR)
    - Coverage warnings
    """
    
    def __init__(
        self,
        fields: Optional[List[Dict]] = None,
        operators: Optional[List[str]] = None,
        strict_field_check: bool = True,
        strict_type_check: bool = True
    ):
        """
        Initialize validator with dataset context.
        
        Args:
            fields: List of field dicts with id, type, coverage, etc.
            operators: List of allowed operator names
            strict_field_check: If True, unknown fields are errors; if False, warnings
            strict_type_check: If True, type mismatches are errors; if False, warnings
        """
        self.strict_field_check = strict_field_check
        self.strict_type_check = strict_type_check
        
        # Build field lookup
        self.field_map: Dict[str, FieldInfo] = {}
        if fields:
            for f in fields:
                info = FieldInfo.from_dict(f)
                if info.field_id:
                    self.field_map[info.field_id.lower()] = info
        
        # Build operator set
        self.allowed_operators: Set[str] = set()
        if operators:
            self.allowed_operators = {op.lower() for op in operators}
        else:
            # Default: allow all operators from registry (loaded from DB or fallback)
            self.allowed_operators = get_known_operators()
            
        # Regex patterns for parsing
        self._field_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
        self._func_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        
    def validate(self, expression: str) -> SemanticValidationResult:
        """
        Perform semantic validation on an expression.
        
        Args:
            expression: Alpha expression string
            
        Returns:
            SemanticValidationResult with errors, warnings, and extracted info
        """
        result = SemanticValidationResult()
        
        if not expression or not expression.strip():
            result.add_error("Empty expression")
            return result
            
        expression = expression.strip()
        
        # 1. Extract operators used
        operators_used = self._extract_operators(expression)
        result.used_operators = operators_used
        
        # 2. Extract fields used (identifiers not matching operators)
        fields_used = self._extract_fields(expression, operators_used)
        result.used_fields = fields_used
        
        # 3. Validate operators exist
        for op in operators_used:
            op_lower = op.lower()
            if self.allowed_operators and op_lower not in self.allowed_operators:
                # Check against all known operators from registry
                all_known = get_known_operators()
                if op_lower not in all_known:
                    result.add_warning(f"Unknown operator: {op}")
                    
        # 4. Validate fields exist and collect type info
        matrix_fields = set()
        vector_fields = set()
        group_like_fields = set()
        unknown_fields = set()
        
        for field_id in fields_used:
            field_lower = field_id.lower()
            
            # Skip built-in groups
            if field_lower in BUILTIN_GROUPS:
                continue
                
            # Skip numeric literals and keywords
            if field_lower in {"true", "false", "nan", "inf"}:
                continue
                
            if field_lower in self.field_map:
                info = self.field_map[field_lower]
                result.field_types_used.add(info.field_type.value)
                
                if info.is_group_like:
                    group_like_fields.add(field_id)
                elif info.field_type == FieldType.MATRIX:
                    matrix_fields.add(field_id)
                elif info.field_type == FieldType.VECTOR:
                    vector_fields.add(field_id)
                    
                # Coverage warning
                if info.coverage < 0.5:
                    result.add_warning(f"Low coverage field: {field_id} ({info.coverage:.1%})")
            else:
                unknown_fields.add(field_id)
                
        # Handle unknown fields
        for field_id in unknown_fields:
            msg = f"Field not found in dataset: {field_id}"
            if self.strict_field_check:
                result.add_error(msg)
            else:
                result.add_warning(msg)
                
        # 5. Type constraint validation
        type_errors = self._validate_type_constraints(
            expression, operators_used, matrix_fields, vector_fields, group_like_fields
        )
        for err in type_errors:
            if self.strict_type_check:
                result.add_error(err)
            else:
                result.add_warning(err)
                
        # 6. Calculate complexity score
        result.complexity_score = len(operators_used) + len(fields_used) * 0.5
        
        return result
    
    def _extract_operators(self, expression: str) -> Set[str]:
        """Extract function/operator names from expression"""
        operators = set()
        for match in self._func_pattern.finditer(expression):
            operators.add(match.group(1))
        return operators
    
    def _extract_fields(self, expression: str, operators: Set[str]) -> Set[str]:
        """Extract field identifiers (non-operator identifiers)"""
        fields = set()
        op_lower = {op.lower() for op in operators}
        
        # Keywords and built-ins to skip
        skip = {
            "true", "false", "nan", "inf",
            "sector", "subindustry", "industry", "exchange", "country", "market",
            "std", "k", "mode", "lag", "rettype", "filter", "scale", "rate",
            "constant", "percentage", "driver", "sigma", "lower", "upper",
            "target", "dest", "event", "sensitivity", "force", "h", "t", "period",
            "stddev", "factor", "usetd", "limit", "gaussian", "uniform", "cauchy",
            "buckets", "range", "nth", "precise", "longscale", "shortscale"
        }
        
        for match in self._field_pattern.finditer(expression):
            ident = match.group(1)
            ident_lower = ident.lower()
            
            # Skip if it's an operator
            if ident_lower in op_lower:
                continue
                
            # Skip keywords/params
            if ident_lower in skip:
                continue
                
            # Skip pure numbers (shouldn't match pattern but just in case)
            if ident.isdigit():
                continue
                
            fields.add(ident)
            
        return fields
    
    def _validate_type_constraints(
        self,
        expression: str,
        operators: Set[str],
        matrix_fields: Set[str],
        vector_fields: Set[str],
        group_like_fields: Set[str]
    ) -> List[str]:
        """
        Validate that field types match operator requirements.
        
        Key rules:
        - ts_* operators work best with MATRIX fields (time-series)
        - vec_* operators require VECTOR fields
        - Using VECTOR fields with ts_* may cause issues
        """
        errors = []
        
        expr_lower = expression.lower()
        group_like_lower = {f.lower() for f in group_like_fields}
        numeric_first_arg_ops = {
            "rank", "zscore", "normalize", "scale", "winsorize", "log", "sqrt",
            "signed_power", "densify", "ts_backfill", "ts_delta", "ts_delay",
            "ts_rank", "ts_zscore", "ts_std_dev", "ts_mean", "ts_sum",
            "ts_returns", "ts_av_diff", "ts_scale", "ts_product", "ts_arg_max",
            "ts_arg_min", "ts_min", "ts_max",
        }
        numeric_all_arg_ops = {
            "add", "subtract", "multiply", "divide", "min", "max",
            "ts_corr", "ts_covariance", "correlation", "covariance", "corr",
        }
        identity_pair_ops = {"ts_corr", "ts_covariance", "correlation", "covariance", "corr"}

        for func_name, args in self._iter_function_calls(expression):
            op_lower = func_name.lower()
            normalized_args = [self._normalize_arg(arg) for arg in args]

            if op_lower in identity_pair_ops and len(normalized_args) >= 2:
                if normalized_args[0] == normalized_args[1]:
                    errors.append(
                        f"Degenerate expression: '{func_name}' uses the same input for both operands."
                    )

            args_to_check = []
            if op_lower.startswith("ts_") or op_lower in numeric_first_arg_ops:
                args_to_check = args[:1]
            elif op_lower in numeric_all_arg_ops:
                args_to_check = args

            for arg in args_to_check:
                root_field = self._first_field_identifier(arg, operators)
                if root_field and root_field.lower() in group_like_lower:
                    errors.append(
                        f"Type mismatch: group-like field '{root_field}' used as numeric input of operator '{func_name}'. "
                        "Use scalar MATRIX fields for numeric/time-series operators."
                    )
        
        for op in operators:
            op_lower = op.lower()
            
            # Check ts_* operators with VECTOR fields (use naming convention)
            if op_lower.startswith("ts_"):
                # Look for vec_ prefix fields being passed to ts_ functions
                # This is a heuristic - we look for vector field names near ts_ calls
                for vf in vector_fields:
                    # Simple heuristic: if vector field appears right after ts_xxx(
                    pattern = rf'{op_lower}\s*\(\s*{re.escape(vf.lower())}'
                    if re.search(pattern, expr_lower):
                        errors.append(
                            f"Type mismatch: VECTOR field '{vf}' used as first arg of time-series operator '{op}'. "
                            f"Consider using vec_* wrapper or MATRIX equivalent."
                        )
                        
            # Check vec_* operators - they expect aggregation over vector dimensions
            # (vec_* operators on MATRIX fields is actually fine - aggregates across vector dim)
                
        return errors

    def _iter_function_calls(self, expression: str) -> List[Tuple[str, List[str]]]:
        """Return best-effort top-level argument lists for function calls."""
        calls = []
        for match in self._func_pattern.finditer(expression):
            name = match.group(1)
            open_idx = match.end() - 1
            depth = 0
            close_idx = None
            for idx in range(open_idx, len(expression)):
                ch = expression[idx]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close_idx = idx
                        break
            if close_idx is None:
                continue
            args = self._split_args(expression[open_idx + 1:close_idx])
            calls.append((name, args))
        return calls

    def _split_args(self, args_text: str) -> List[str]:
        args = []
        start = 0
        depth = 0
        for idx, ch in enumerate(args_text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                args.append(args_text[start:idx].strip())
                start = idx + 1
        tail = args_text[start:].strip()
        if tail:
            args.append(tail)
        return args

    def _normalize_arg(self, arg: str) -> str:
        return re.sub(r"\s+", "", arg).lower()

    def _first_field_identifier(self, arg: str, operators: Set[str]) -> Optional[str]:
        op_lower = {op.lower() for op in operators}
        for match in self._field_pattern.finditer(arg):
            ident = match.group(1)
            ident_lower = ident.lower()
            if ident_lower in op_lower or ident_lower in {"true", "false", "nan", "inf"}:
                continue
            if ident_lower in BUILTIN_GROUPS:
                continue
            if ident_lower in self.field_map:
                return ident
        return None


def compute_expression_hash(expression: str) -> str:
    """
    Compute a normalized hash for expression deduplication.
    
    Normalizes:
    - Whitespace
    - Case (for operators)
    - Numeric precision
    """
    # Normalize whitespace
    normalized = " ".join(expression.split())
    
    # Normalize operator case using registry
    for op in get_known_operators():
        pattern = re.compile(re.escape(op), re.IGNORECASE)
        normalized = pattern.sub(op.lower(), normalized)
        
    # Hash
    return hashlib.md5(normalized.encode()).hexdigest()


def compute_structural_similarity(expr1: str, expr2: str) -> float:
    """
    Compute structural similarity between two expressions.
    
    Based on:
    - Operator n-gram overlap
    - Field Jaccard similarity
    
    Returns: Similarity score 0.0 to 1.0
    """
    validator = AlphaSemanticValidator()
    
    # Extract operators
    ops1 = validator._extract_operators(expr1)
    ops2 = validator._extract_operators(expr2)
    
    # Extract fields
    fields1 = validator._extract_fields(expr1, ops1)
    fields2 = validator._extract_fields(expr2, ops2)
    
    # Operator overlap (Jaccard)
    if ops1 or ops2:
        op_jaccard = len(ops1 & ops2) / len(ops1 | ops2) if (ops1 | ops2) else 0
    else:
        op_jaccard = 1.0
        
    # Field overlap (Jaccard)
    if fields1 or fields2:
        field_jaccard = len(fields1 & fields2) / len(fields1 | fields2) if (fields1 | fields2) else 0
    else:
        field_jaccard = 1.0
        
    # Weighted combination
    return 0.4 * op_jaccard + 0.6 * field_jaccard


class ExpressionDeduplicator:
    """
    Track seen expressions and detect duplicates.
    
    P0-2: Deduplication gate before simulation
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.seen_hashes: Set[str] = set()
        self.seen_expressions: List[str] = []
        self.similarity_threshold = similarity_threshold
        
    def is_duplicate(self, expression: str) -> Tuple[bool, Optional[str]]:
        """
        Check if expression is a duplicate.
        
        Returns:
            (is_duplicate, reason)
        """
        expr_hash = compute_expression_hash(expression)
        
        # Exact hash match
        if expr_hash in self.seen_hashes:
            return True, "Exact duplicate (hash match)"
            
        # Structural similarity check (expensive, limit to recent)
        recent = self.seen_expressions[-100:]  # Only check last 100
        for seen in recent:
            sim = compute_structural_similarity(expression, seen)
            if sim >= self.similarity_threshold:
                return True, f"Structurally similar ({sim:.1%}) to: {seen[:50]}..."
                
        return False, None
        
    def add(self, expression: str):
        """Add expression to seen set"""
        expr_hash = compute_expression_hash(expression)
        self.seen_hashes.add(expr_hash)
        self.seen_expressions.append(expression)
        
    def clear(self):
        """Clear all seen expressions"""
        self.seen_hashes.clear()
        self.seen_expressions.clear()


# =============================================================================
# P1-4: Diversity scoring for batch evaluation
# =============================================================================

def compute_batch_diversity(expressions: List[str]) -> float:
    """
    Compute diversity score for a batch of expressions.
    
    Higher is better (more diverse).
    
    Returns: 0.0 to 1.0
    """
    if len(expressions) <= 1:
        return 1.0
        
    # Pairwise similarity
    similarities = []
    for i in range(len(expressions)):
        for j in range(i + 1, len(expressions)):
            sim = compute_structural_similarity(expressions[i], expressions[j])
            similarities.append(sim)
            
    if not similarities:
        return 1.0
        
    # Diversity = 1 - average similarity
    avg_sim = sum(similarities) / len(similarities)
    return 1.0 - avg_sim


# =============================================================================
# Integration helper for node_validate
# =============================================================================

def validate_alpha_semantically(
    expression: str,
    fields: List[Dict],
    operators: Optional[List[str]] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Convenience function for semantic validation.
    
    Args:
        expression: Alpha expression
        fields: List of field dicts from state
        operators: Optional list of allowed operators
        strict: If True, use strict checking
        
    Returns:
        Dict with 'valid', 'errors', 'warnings', 'used_fields', 'used_operators'
    """
    validator = AlphaSemanticValidator(
        fields=fields,
        operators=operators,
        strict_field_check=strict,
        strict_type_check=strict
    )
    
    result = validator.validate(expression)
    
    return {
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "used_fields": list(result.used_fields),
        "used_operators": list(result.used_operators),
        "field_types": list(result.field_types_used),
        "complexity_score": result.complexity_score
    }
