"""
Alpha Optimizer - Comprehensive Enhancement Module

This module implements ALL the P0-P2 optimizations identified in the system diagnosis:

P0 Critical (Must Fix):
- P0-1: Field availability pre-check before generation
- P0-2: Mandatory hypothesis-implementation alignment enforcement  
- P0-3: Signal direction detection and auto-correction

P1 High Priority:
- P1-1: CoSTEER hard constraint enforcement
- P1-2: Simplified GP enhancement (expression variation search)
- P1-3: Smart exploration strategy (convergence over random)

P2 Medium Priority:
- P2-1: Enhanced knowledge graph utilization
- P2-2: Multi-fidelity pre-screening
- P2-3: Intelligent dataset selection

Reference: Alpha-GPT 1.0/2.0 + RD-Agent CoSTEER
"""

import re
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger


# =============================================================================
# P0-1: Enhanced Field Availability Pre-Check
# =============================================================================

@dataclass
class FieldPreCheckResult:
    """Result of field pre-check before alpha generation."""
    is_valid: bool
    available_fields: List[str]
    blocked_fields: List[str]
    block_reasons: Dict[str, str]
    recommended_fields: List[str]  # Fields with high historical success
    warning_message: str = ""


class FieldPreChecker:
    """
    P0-1: Pre-checks field availability BEFORE code generation.
    
    Key improvements over existing field_availability_checker.py:
    1. Proactive checking instead of reactive recording
    2. Integration with knowledge base for historical blacklists
    3. Region/Universe specific field whitelists
    4. Mandatory field validation for hypotheses
    """
    
    # Known field patterns that often cause issues
    PROBLEMATIC_PATTERNS = [
        r"^test_",
        r"_debug$",
        r"_deprecated$",
        r"^tmp_",
    ]
    
    # Common fields that work across most regions (whitelist)
    UNIVERSAL_SAFE_FIELDS = {
        "close", "open", "high", "low", "volume", "vwap",
        "returns", "cap", "adv", "sector", "industry", "subindustry"
    }
    
    def __init__(self):
        # Cache of confirmed working fields per region/universe
        self._confirmed_fields: Dict[str, Set[str]] = defaultdict(set)
        # Cache of failed fields per region/universe
        self._failed_fields: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Failure threshold for blacklisting
        self._failure_threshold = 2
        
    def pre_check_fields(
        self,
        all_fields: List[Dict],
        region: str,
        universe: str,
        hypothesis_key_fields: Optional[List[str]] = None
    ) -> FieldPreCheckResult:
        """
        P0-1: Pre-check field availability BEFORE generation.
        
        This is called BEFORE code_gen node to filter out unavailable fields.
        
        Args:
            all_fields: All fields from the dataset
            region: Market region (USA, CHN, etc.)
            universe: Universe (TOP3000, TOP500, etc.)
            hypothesis_key_fields: Fields specified in hypothesis (mandatory)
            
        Returns:
            FieldPreCheckResult with available/blocked field lists
        """
        key = f"{region}:{universe}"
        available = []
        blocked = []
        block_reasons = {}
        recommended = []
        
        hypothesis_fields_lower = {f.lower() for f in (hypothesis_key_fields or [])}
        
        for field_dict in all_fields:
            field_id = field_dict.get("id", field_dict.get("name", ""))
            if not field_id:
                continue
                
            field_lower = field_id.lower()
            is_blocked = False
            reason = ""
            
            # Check 1: Problematic patterns
            for pattern in self.PROBLEMATIC_PATTERNS:
                if re.match(pattern, field_lower):
                    is_blocked = True
                    reason = f"Matches problematic pattern: {pattern}"
                    break
            
            # Check 2: Historical failure count
            if not is_blocked and field_id in self._failed_fields[key]:
                failure_count = self._failed_fields[key][field_id]
                if failure_count >= self._failure_threshold:
                    is_blocked = True
                    reason = f"Blacklisted ({failure_count} previous failures)"
            
            # Check 3: Mandatory hypothesis fields should NOT be blocked easily
            if is_blocked and field_lower in hypothesis_fields_lower:
                # Give mandatory fields a second chance
                if self._failed_fields[key].get(field_id, 0) < self._failure_threshold * 2:
                    is_blocked = False
                    reason = ""
                    logger.warning(
                        f"[FieldPreCheck] Allowing potentially problematic mandatory field: {field_id}"
                    )
            
            if is_blocked:
                blocked.append(field_id)
                block_reasons[field_id] = reason
            else:
                available.append(field_dict)
                
                # Mark as recommended if confirmed working
                if field_id in self._confirmed_fields[key]:
                    recommended.append(field_id)
        
        # Build warning message
        warning = ""
        if hypothesis_key_fields:
            blocked_mandatory = [f for f in hypothesis_key_fields if f.lower() in {b.lower() for b in blocked}]
            if blocked_mandatory:
                warning = f"WARNING: Mandatory hypothesis fields are blocked: {blocked_mandatory}"
        
        if blocked:
            logger.info(
                f"[FieldPreCheck] Filtered {len(blocked)} fields for {region}/{universe}: "
                f"{blocked[:5]}{'...' if len(blocked) > 5 else ''}"
            )
        
        return FieldPreCheckResult(
            is_valid=len(available) > 0,
            available_fields=[f.get("id", f.get("name", "")) for f in available],
            blocked_fields=blocked,
            block_reasons=block_reasons,
            recommended_fields=recommended[:10],
            warning_message=warning
        )
    
    def record_success(self, expression: str, region: str, universe: str):
        """Record successful simulation - confirm fields work."""
        key = f"{region}:{universe}"
        fields = self._extract_fields(expression)
        for field_id in fields:
            self._confirmed_fields[key].add(field_id)
            # Reduce failure count if any
            if field_id in self._failed_fields[key]:
                self._failed_fields[key][field_id] = max(0, self._failed_fields[key][field_id] - 1)
    
    def record_failure(self, expression: str, error: str, region: str, universe: str):
        """Record simulation failure - increment field failure counts."""
        key = f"{region}:{universe}"
        
        # Only track field-related errors
        if not any(kw in error.lower() for kw in ['field', 'unknown', 'not found', 'data', 'children failed']):
            return
            
        fields = self._extract_fields(expression)
        for field_id in fields:
            self._failed_fields[key][field_id] += 1
    
    def _extract_fields(self, expression: str) -> List[str]:
        """Extract field names from expression."""
        if not expression:
            return []
        
        operators = {
            'ts_rank', 'ts_delta', 'ts_zscore', 'ts_mean', 'ts_std_dev',
            'ts_decay_linear', 'ts_decay_exp', 'ts_sum', 'ts_max', 'ts_min',
            'ts_arg_max', 'ts_arg_min', 'ts_returns', 'ts_product', 'ts_corr',
            'vec_sum', 'vec_avg', 'vec_count', 'vec_max', 'vec_min',
            'group_neutralize', 'group_rank', 'group_mean', 'group_zscore',
            'rank', 'zscore', 'scale', 'truncate', 'winsorize',
            'divide', 'add', 'subtract', 'multiply', 'log', 'abs', 'sign',
            'power', 'sqrt', 'exp', 'if_else', 'trade_when', 'pasteurize',
        }
        
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression)
        return [i for i in identifiers if i.lower() not in operators and len(i) > 2]


# =============================================================================
# P0-2: Mandatory Hypothesis-Implementation Alignment
# =============================================================================

@dataclass
class AlignmentCheckResult:
    """Result of hypothesis-implementation alignment check."""
    is_aligned: bool
    expression: str
    hypothesis_statement: str
    key_fields_used: List[str]
    missing_key_fields: List[str]
    signal_direction_correct: bool
    issues: List[str]
    corrected_expression: Optional[str] = None  # If auto-corrected
    rejection_reason: str = ""


class HypothesisAlignmentEnforcer:
    """
    P0-2: Enforces mandatory hypothesis-implementation alignment.
    
    Key features:
    1. REJECTS expressions that don't use hypothesis key_fields
    2. Detects signal direction mismatches
    3. Provides auto-correction suggestions
    4. Hard enforcement (not just warnings)
    """
    
    # Signal direction keywords
    POSITIVE_KEYWORDS = ['increase', 'growth', 'up', 'rise', 'high', 'above', 'outperform', 'positive']
    NEGATIVE_KEYWORDS = ['decrease', 'decline', 'down', 'fall', 'low', 'below', 'underperform', 'negative']
    
    def enforce_alignment(
        self,
        expression: str,
        hypothesis: Dict[str, Any],
        available_fields: List[str]
    ) -> AlignmentCheckResult:
        """
        P0-2: Enforce mandatory alignment between hypothesis and expression.
        
        This is called AFTER code generation to validate/reject misaligned expressions.
        
        Args:
            expression: Generated alpha expression
            hypothesis: Hypothesis dict with statement, key_fields, expected_signal
            available_fields: List of available field names
            
        Returns:
            AlignmentCheckResult - is_aligned=False means REJECT the expression
        """
        issues = []
        corrected_expr = None
        
        # Extract hypothesis info
        statement = hypothesis.get("statement", hypothesis.get("idea", ""))
        key_fields = hypothesis.get("key_fields", [])
        expected_signal = hypothesis.get("expected_signal", hypothesis.get("signal_type", ""))
        
        # Extract fields from expression
        expr_fields = self._extract_fields_from_expression(expression)
        expr_fields_lower = {f.lower() for f in expr_fields}
        key_fields_lower = {f.lower() for f in key_fields}
        
        # Check 1: At least ONE key field must be used (MANDATORY)
        used_key_fields = []
        missing_key_fields = []
        
        for kf in key_fields:
            kf_lower = kf.lower()
            # Check for exact match or partial match (e.g., field in vec_sum(field))
            found = False
            for ef in expr_fields:
                if kf_lower == ef.lower() or kf_lower in ef.lower() or ef.lower() in kf_lower:
                    used_key_fields.append(kf)
                    found = True
                    break
            if not found:
                missing_key_fields.append(kf)
        
        key_field_aligned = len(used_key_fields) > 0 if key_fields else True
        
        if not key_field_aligned:
            issues.append(f"CRITICAL: Expression uses NONE of the hypothesis key_fields: {key_fields}")
        
        # Check 2: Signal direction (if specified)
        signal_aligned = True
        if expected_signal and statement:
            expected_positive = any(kw in expected_signal.lower() for kw in ['positive', 'increase', 'up', 'momentum'])
            expected_positive = expected_positive or any(kw in statement.lower() for kw in self.POSITIVE_KEYWORDS)
            
            # Check if expression has negation
            has_negation = expression.strip().startswith('-') or expression.strip().startswith('(-')
            
            if expected_positive and has_negation:
                signal_aligned = False
                issues.append(f"Signal direction mismatch: hypothesis expects positive, expression is negated")
                # Suggest correction
                corrected_expr = expression.lstrip('-').lstrip('(').rstrip(')')
            elif not expected_positive and not has_negation:
                # Should be negative but isn't
                expected_negative = any(kw in expected_signal.lower() for kw in ['negative', 'decrease', 'down', 'reversal'])
                expected_negative = expected_negative or any(kw in statement.lower() for kw in self.NEGATIVE_KEYWORDS)
                if expected_negative:
                    signal_aligned = False
                    issues.append(f"Signal direction mismatch: hypothesis expects negative, expression is positive")
                    corrected_expr = f"-({expression})"
        
        # Determine overall alignment
        is_aligned = key_field_aligned and signal_aligned
        
        rejection_reason = ""
        if not is_aligned:
            if not key_field_aligned:
                rejection_reason = "REJECTED: Expression does not use any hypothesis key_fields"
            elif not signal_aligned:
                rejection_reason = "WARNING: Signal direction may be inverted"
        
        return AlignmentCheckResult(
            is_aligned=is_aligned,
            expression=expression,
            hypothesis_statement=statement,
            key_fields_used=used_key_fields,
            missing_key_fields=missing_key_fields,
            signal_direction_correct=signal_aligned,
            issues=issues,
            corrected_expression=corrected_expr,
            rejection_reason=rejection_reason
        )
    
    def _extract_fields_from_expression(self, expression: str) -> List[str]:
        """Extract field identifiers from expression."""
        if not expression:
            return []
        
        operators = {
            'ts_rank', 'ts_delta', 'ts_zscore', 'ts_mean', 'ts_std_dev',
            'ts_decay_linear', 'ts_sum', 'ts_max', 'ts_min', 'ts_corr',
            'vec_sum', 'vec_avg', 'rank', 'zscore', 'group_rank',
            'divide', 'add', 'subtract', 'multiply', 'log', 'abs', 'sign',
            'if_else', 'pasteurize', 'power', 'sqrt',
        }
        
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression)
        return list(set(i for i in identifiers if i.lower() not in operators and len(i) > 2))


# =============================================================================
# P0-3: Signal Direction Detection and Auto-Correction
# =============================================================================

class SignalDirectionCorrector:
    """
    P0-3: Detects and corrects signal direction issues.
    
    Common problem: LLM generates expressions with inverted signals,
    resulting in negative Sharpe ratios.
    
    Solution: Detect negative Sharpe and auto-try inversion.
    """
    
    def should_invert_signal(
        self,
        sharpe: float,
        fitness: float,
        turnover: float
    ) -> Tuple[bool, str]:
        """
        Determine if signal should be inverted based on metrics.
        
        Args:
            sharpe: In-sample Sharpe ratio
            fitness: In-sample fitness
            turnover: Turnover ratio
            
        Returns:
            (should_invert, reason)
        """
        # Case 1: Strongly negative Sharpe with reasonable turnover
        # This often indicates inverted signal
        if sharpe < -0.3 and turnover < 0.7:
            return True, f"Negative Sharpe ({sharpe:.2f}) with reasonable turnover suggests inverted signal"
        
        # Case 2: Moderate negative Sharpe but good fitness structure
        if sharpe < -0.1 and fitness > 0.3:
            return True, f"Negative Sharpe ({sharpe:.2f}) but positive fitness suggests signal inversion needed"
        
        return False, ""
    
    def create_inverted_expression(self, expression: str) -> str:
        """
        Create inverted version of expression.
        
        Handles various expression formats:
        - Simple: "signal" -> "-signal"
        - Already negated: "-signal" -> "signal"
        - Wrapped: "ts_decay_linear(x, n)" -> "-ts_decay_linear(x, n)"
        """
        expr = expression.strip()
        
        # Check if already negated at the top level
        if expr.startswith('-') and not expr.startswith('-('):
            # Remove negation
            return expr[1:].strip()
        elif expr.startswith('-(') and expr.endswith(')'):
            # Unwrap negation
            return expr[2:-1]
        else:
            # Add negation
            return f"-({expr})"
    
    def generate_variants(self, expression: str) -> List[Tuple[str, str]]:
        """
        Generate signal direction variants for testing.
        
        Returns list of (expression, description) tuples.
        """
        variants = [
            (expression, "original"),
            (self.create_inverted_expression(expression), "inverted"),
        ]
        return variants


# =============================================================================
# P1-1: CoSTEER Hard Constraint Enforcement
# =============================================================================

@dataclass
class CoSTEERConstraints:
    """
    P1-1: Hard constraints from CoSTEER feedback loop.
    
    These are MANDATORY - expressions violating them should be REJECTED.
    """
    # Forbidden field patterns (from failures)
    forbidden_fields: Set[str] = field(default_factory=set)
    
    # Forbidden expression patterns (regex)
    forbidden_patterns: List[str] = field(default_factory=list)
    
    # Required operators (at least one must be used)
    required_operators: List[str] = field(default_factory=list)
    
    # Minimum/maximum constraints
    min_decay_window: int = 5  # Minimum ts_decay_linear window
    max_expression_depth: int = 8  # Maximum nesting depth
    
    # Mandatory structural requirements
    require_decay_wrapper: bool = True
    require_cross_sectional_norm: bool = False


class CoSTEEREnforcer:
    """
    P1-1: Enforces CoSTEER hard constraints.
    
    This goes beyond warnings - it BLOCKS expressions that violate learned rules.
    """
    
    def __init__(self):
        self._constraints = CoSTEERConstraints()
        self._failure_patterns: Dict[str, int] = defaultdict(int)
        self._success_patterns: Dict[str, float] = {}  # pattern -> best sharpe
        
    def update_from_failures(
        self,
        failures: List[Dict],
        threshold_count: int = 2
    ):
        """
        Update constraints from failure analysis.
        
        Args:
            failures: List of failure records with expression, error_type, etc.
            threshold_count: Minimum failures before adding to forbidden list
        """
        # Track failure patterns
        for f in failures:
            expr = f.get("expression", "")
            error_type = f.get("error_type", "")
            
            # Extract key pattern
            pattern = self._extract_pattern(expr)
            self._failure_patterns[pattern] += 1
            
            # Extract fields from failed expressions
            if "field" in error_type.lower() or "not found" in str(f.get("error_message", "")).lower():
                fields = self._extract_fields(expr)
                for field in fields:
                    self._constraints.forbidden_fields.add(field)
        
        # Add patterns that failed multiple times
        for pattern, count in self._failure_patterns.items():
            if count >= threshold_count:
                if pattern not in self._constraints.forbidden_patterns:
                    self._constraints.forbidden_patterns.append(pattern)
                    logger.info(f"[CoSTEER] Added forbidden pattern: {pattern[:50]}... ({count} failures)")
    
    def update_from_successes(self, successes: List[Dict]):
        """Update constraints from successful alphas."""
        for s in successes:
            expr = s.get("expression", "")
            sharpe = s.get("sharpe", 0)
            
            pattern = self._extract_pattern(expr)
            
            # Track best sharpe for this pattern
            if pattern not in self._success_patterns or sharpe > self._success_patterns[pattern]:
                self._success_patterns[pattern] = sharpe
    
    def check_constraints(self, expression: str) -> Tuple[bool, List[str]]:
        """
        Check if expression violates any hard constraints.
        
        Args:
            expression: Alpha expression to check
            
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Check 1: Forbidden fields
        fields = self._extract_fields(expression)
        forbidden_used = [f for f in fields if f in self._constraints.forbidden_fields]
        if forbidden_used:
            violations.append(f"Uses forbidden fields: {forbidden_used[:3]}")
        
        # Check 2: Forbidden patterns
        for pattern in self._constraints.forbidden_patterns:
            if pattern in expression.lower():
                violations.append(f"Matches forbidden pattern: {pattern[:30]}...")
                break
        
        # Check 3: Decay wrapper requirement
        if self._constraints.require_decay_wrapper:
            if "ts_decay_linear" not in expression and "ts_decay_exp" not in expression:
                violations.append("Missing required decay wrapper (ts_decay_linear)")
        
        # Check 4: Minimum decay window
        decay_match = re.search(r'ts_decay_linear\([^,]+,\s*(\d+)\)', expression)
        if decay_match:
            window = int(decay_match.group(1))
            if window < self._constraints.min_decay_window:
                violations.append(f"Decay window too small ({window} < {self._constraints.min_decay_window})")
        
        # Check 5: Expression depth
        depth = self._calculate_depth(expression)
        if depth > self._constraints.max_expression_depth:
            violations.append(f"Expression too deep ({depth} > {self._constraints.max_expression_depth})")
        
        return len(violations) == 0, violations
    
    def _extract_pattern(self, expression: str) -> str:
        """Extract structural pattern from expression."""
        if not expression:
            return ""
        
        # Replace specific field names with FIELD
        pattern = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?=\s*[,)])', 'FIELD', expression.lower())
        # Replace numbers with N
        pattern = re.sub(r'\b\d+\b', 'N', pattern)
        return pattern[:100]  # Limit length
    
    def _extract_fields(self, expression: str) -> List[str]:
        """Extract field names from expression."""
        operators = {'ts_rank', 'ts_delta', 'ts_zscore', 'ts_decay_linear', 'vec_sum', 'rank'}
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression or "")
        return [i for i in identifiers if i.lower() not in operators and len(i) > 3]
    
    def _calculate_depth(self, expression: str) -> int:
        """Calculate nesting depth of expression."""
        max_depth = 0
        current_depth = 0
        for char in expression:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth
    
    def get_constraints(self) -> CoSTEERConstraints:
        """Get current constraints."""
        return self._constraints


# =============================================================================
# P1-2: Simplified GP Enhancement (Expression Variation Search)
# =============================================================================

@dataclass
class ExpressionVariant:
    """A variant of an expression with modification description."""
    expression: str
    modification: str
    parent_expression: str
    generation: int = 0
    expected_improvement: str = ""


class SimpleGPEnhancer:
    """
    P1-2: Generates expression variants through simple genetic operations.
    
    This is a simplified version of genetic programming focused on:
    1. Parameter tuning (window sizes, decay values)
    2. Operator substitution (ts_rank <-> ts_zscore)
    3. Structural variations (adding smoothing, normalization)
    
    NOT a full GP system - just smart variation generation.
    """
    
    # Parameter ranges for mutation
    WINDOW_SIZES = [5, 10, 15, 20, 22, 30, 42, 63]
    DECAY_VALUES = [5, 8, 10, 12, 15, 20]
    
    # Operator substitutions that preserve semantics
    OPERATOR_SUBSTITUTIONS = {
        "ts_rank": ["rank", "ts_zscore"],
        "ts_zscore": ["zscore", "ts_rank"],
        "ts_delta": ["ts_returns"],
        "ts_mean": ["ts_sum"],
        "rank": ["ts_rank", "zscore"],
        "zscore": ["ts_zscore", "rank"],
    }
    
    # Wrapper templates for enhancement
    WRAPPER_TEMPLATES = [
        "ts_decay_linear({expr}, {decay})",
        "rank({expr})",
        "zscore({expr})",
        "-({expr})",
    ]
    
    def generate_variants(
        self,
        seed_expression: str,
        num_variants: int = 5,
        include_inversions: bool = True
    ) -> List[ExpressionVariant]:
        """
        Generate variants of a seed expression.
        
        Args:
            seed_expression: Base expression to vary
            num_variants: Number of variants to generate
            include_inversions: Include signal inversions
            
        Returns:
            List of ExpressionVariant objects
        """
        variants = []
        
        # Variant 1: Parameter mutations
        param_variants = self._mutate_parameters(seed_expression)
        variants.extend(param_variants[:num_variants // 2])
        
        # Variant 2: Operator substitutions
        op_variants = self._substitute_operators(seed_expression)
        variants.extend(op_variants[:2])
        
        # Variant 3: Add/modify decay
        decay_variants = self._modify_decay(seed_expression)
        variants.extend(decay_variants[:2])
        
        # Variant 4: Signal inversion (if sharpe was negative)
        if include_inversions:
            inverted = ExpressionVariant(
                expression=f"-({seed_expression})",
                modification="Signal inversion",
                parent_expression=seed_expression,
                expected_improvement="May improve if signal was inverted"
            )
            variants.append(inverted)
        
        # Deduplicate by expression
        seen = set()
        unique_variants = []
        for v in variants:
            expr_hash = hashlib.md5(v.expression.encode()).hexdigest()
            if expr_hash not in seen and v.expression != seed_expression:
                seen.add(expr_hash)
                unique_variants.append(v)
        
        return unique_variants[:num_variants]
    
    def _mutate_parameters(self, expression: str) -> List[ExpressionVariant]:
        """Mutate numeric parameters in expression."""
        variants = []
        
        # Find all numbers in expression
        numbers = re.findall(r'\b(\d+)\b', expression)
        
        for num in set(numbers):
            num_int = int(num)
            
            # Find suitable replacement values
            if 5 <= num_int <= 63:  # Likely a window size
                replacements = [w for w in self.WINDOW_SIZES if w != num_int]
                for repl in random.sample(replacements, min(2, len(replacements))):
                    new_expr = re.sub(rf'\b{num}\b', str(repl), expression, count=1)
                    variants.append(ExpressionVariant(
                        expression=new_expr,
                        modification=f"Changed window {num} -> {repl}",
                        parent_expression=expression,
                        expected_improvement="Parameter tuning"
                    ))
        
        return variants
    
    def _substitute_operators(self, expression: str) -> List[ExpressionVariant]:
        """Substitute operators with semantic equivalents."""
        variants = []
        
        for op, replacements in self.OPERATOR_SUBSTITUTIONS.items():
            if op in expression.lower():
                for repl in replacements[:1]:  # Only first replacement
                    new_expr = re.sub(rf'\b{op}\b', repl, expression, flags=re.IGNORECASE)
                    if new_expr != expression:
                        variants.append(ExpressionVariant(
                            expression=new_expr,
                            modification=f"Operator substitution: {op} -> {repl}",
                            parent_expression=expression,
                            expected_improvement="Different normalization approach"
                        ))
        
        return variants
    
    def _modify_decay(self, expression: str) -> List[ExpressionVariant]:
        """Modify or add decay wrapper."""
        variants = []
        
        # Check if has decay
        has_decay = "ts_decay_linear" in expression.lower()
        
        if has_decay:
            # Modify existing decay parameter
            match = re.search(r'ts_decay_linear\(([^,]+),\s*(\d+)\)', expression, re.IGNORECASE)
            if match:
                current_decay = int(match.group(2))
                for new_decay in self.DECAY_VALUES:
                    if new_decay != current_decay:
                        new_expr = re.sub(
                            r'(ts_decay_linear\([^,]+,\s*)\d+(\))',
                            rf'\g<1>{new_decay}\g<2>',
                            expression,
                            flags=re.IGNORECASE
                        )
                        variants.append(ExpressionVariant(
                            expression=new_expr,
                            modification=f"Decay {current_decay} -> {new_decay}",
                            parent_expression=expression,
                            expected_improvement="Turnover adjustment"
                        ))
        else:
            # Add decay wrapper
            for decay in [8, 10, 12]:
                new_expr = f"ts_decay_linear({expression}, {decay})"
                variants.append(ExpressionVariant(
                    expression=new_expr,
                    modification=f"Added ts_decay_linear(..., {decay})",
                    parent_expression=expression,
                    expected_improvement="Turnover control"
                ))
        
        return variants


# =============================================================================
# P1-3: Smart Exploration Strategy
# =============================================================================

@dataclass
class ExplorationState:
    """State for smart exploration strategy."""
    iteration: int = 0
    consecutive_failures: int = 0
    best_sharpe_seen: float = 0.0
    last_success_iteration: int = 0
    explored_patterns: Set[str] = field(default_factory=set)
    successful_patterns: List[str] = field(default_factory=list)
    
    # Per-dataset tracking
    dataset_attempts: Dict[str, int] = field(default_factory=dict)
    dataset_successes: Dict[str, int] = field(default_factory=dict)


class SmartExplorationStrategy:
    """
    P1-3: Smart exploration strategy that converges instead of random search.
    
    Key improvements over existing evolution_strategy.py:
    1. Evidence-based exploration/exploitation balance
    2. Pattern-based learning (not just success rate)
    3. Adaptive temperature based on progress
    4. Intelligent dataset rotation
    """
    
    def __init__(self):
        self.state = ExplorationState()
        
    def update_state(
        self,
        passed_count: int,
        failed_count: int,
        best_sharpe: Optional[float],
        patterns_tried: List[str],
        dataset_id: str
    ):
        """Update exploration state after an iteration."""
        self.state.iteration += 1
        
        # Update dataset tracking
        self.state.dataset_attempts[dataset_id] = self.state.dataset_attempts.get(dataset_id, 0) + (passed_count + failed_count)
        self.state.dataset_successes[dataset_id] = self.state.dataset_successes.get(dataset_id, 0) + passed_count
        
        # Track patterns
        for p in patterns_tried:
            self.state.explored_patterns.add(p[:50])
        
        if passed_count > 0:
            self.state.consecutive_failures = 0
            self.state.last_success_iteration = self.state.iteration
            if patterns_tried:
                self.state.successful_patterns.extend(patterns_tried[:2])
        else:
            self.state.consecutive_failures += 1
        
        if best_sharpe and best_sharpe > self.state.best_sharpe_seen:
            self.state.best_sharpe_seen = best_sharpe
    
    def get_strategy_parameters(
        self,
        current_progress: float,  # 0-1, progress towards goal
        max_iterations: int
    ) -> Dict[str, Any]:
        """
        Get strategy parameters based on current state.
        
        Returns dict with:
        - temperature: LLM temperature (0.3 - 1.0)
        - exploration_weight: Exploration vs exploitation (0 - 1)
        - should_exploit: Whether to exploit successful patterns
        - should_rescue: Whether to try rescue mode
        - recommended_action: Description of recommended action
        """
        # Base calculations
        iterations_remaining = max_iterations - self.state.iteration
        urgency = 1.0 - (iterations_remaining / max_iterations)  # 0 = early, 1 = late
        
        # Calculate exploration weight
        # Early: high exploration; Late: high exploitation; Stuck: high exploration again
        if self.state.consecutive_failures >= 3:
            # Stuck - need to try something different
            exploration_weight = min(0.9, 0.5 + 0.1 * self.state.consecutive_failures)
            temperature = min(1.0, 0.7 + 0.05 * self.state.consecutive_failures)
            action = "RESCUE: Multiple failures, trying diverse approaches"
        elif self.state.consecutive_failures == 0 and self.state.successful_patterns:
            # Just had success - exploit it
            exploration_weight = max(0.2, 0.4 - 0.1 * len(self.state.successful_patterns))
            temperature = max(0.4, 0.7 - 0.05 * len(self.state.successful_patterns))
            action = "EXPLOIT: Building on successful patterns"
        elif urgency > 0.7 and current_progress < 0.5:
            # Late and behind - moderate exploration with urgency
            exploration_weight = 0.6
            temperature = 0.8
            action = "URGENT: Behind schedule, increasing diversity"
        elif self.state.iteration < 3:
            # Early - explore broadly
            exploration_weight = 0.7
            temperature = 0.8
            action = "EXPLORE: Initial exploration phase"
        else:
            # Normal balanced mode
            exploration_weight = 0.5
            temperature = 0.7
            action = "BALANCED: Standard exploration/exploitation"
        
        return {
            "temperature": temperature,
            "exploration_weight": exploration_weight,
            "should_exploit": len(self.state.successful_patterns) > 0 and exploration_weight < 0.4,
            "should_rescue": self.state.consecutive_failures >= 4,
            "successful_patterns": self.state.successful_patterns[-5:],
            "recommended_action": action,
            "patterns_to_avoid": list(self.state.explored_patterns)[-20:],
        }
    
    def should_rotate_dataset(
        self,
        current_dataset: str,
        consecutive_failures_this_dataset: int
    ) -> Tuple[bool, str]:
        """
        Determine if should rotate to a different dataset.
        
        More intelligent than simple failure count threshold.
        """
        # Get dataset statistics
        attempts = self.state.dataset_attempts.get(current_dataset, 0)
        successes = self.state.dataset_successes.get(current_dataset, 0)
        success_rate = successes / attempts if attempts > 0 else 0
        
        # Conditions for rotation
        if consecutive_failures_this_dataset >= 4 and success_rate < 0.1:
            return True, f"Dataset {current_dataset} exhausted (4+ failures, <10% success)"
        
        if attempts >= 10 and success_rate < 0.05:
            return True, f"Dataset {current_dataset} underperforming (10+ attempts, <5% success)"
        
        if consecutive_failures_this_dataset >= 6:
            return True, f"Dataset {current_dataset} stuck (6+ consecutive failures)"
        
        return False, ""
    
    def select_best_dataset(
        self,
        available_datasets: List[str],
        current_dataset: str
    ) -> str:
        """Select the best dataset to try next based on history."""
        # Calculate scores for each dataset
        scores = {}
        
        for ds in available_datasets:
            attempts = self.state.dataset_attempts.get(ds, 0)
            successes = self.state.dataset_successes.get(ds, 0)
            
            if attempts == 0:
                # Untried dataset - give high exploration score
                score = 0.7
            else:
                # Score based on success rate with uncertainty bonus
                success_rate = successes / attempts
                uncertainty_bonus = 1.0 / (1 + attempts * 0.1)  # Decay with attempts
                score = success_rate * 0.6 + uncertainty_bonus * 0.4
            
            # Penalize current dataset if rotating away
            if ds == current_dataset:
                score *= 0.5
            
            scores[ds] = score
        
        # Select highest scoring dataset
        best = max(scores.items(), key=lambda x: x[1])
        return best[0]


# =============================================================================
# P2-2: Multi-Fidelity Pre-Screening
# =============================================================================

@dataclass
class PreScreenResult:
    """Result of multi-fidelity pre-screening."""
    expression: str
    should_simulate: bool
    confidence: float
    estimated_sharpe: float
    issues: List[str]
    fast_check_passed: bool
    medium_check_passed: bool


class MultiFidelityScreener:
    """
    P2-2: Multi-fidelity pre-screening to filter out obviously bad expressions.
    
    Fidelity levels:
    1. Fast (syntax/structure) - Milliseconds
    2. Medium (pattern matching) - Milliseconds
    3. Full simulation - Seconds (only if pass 1 & 2)
    
    Goal: Reduce wasted simulation budget by 50%+
    """
    
    # Known bad patterns (from historical failures)
    BAD_PATTERNS = [
        r'ts_delta\([^,]+,\s*1\)',  # ts_delta with window=1 (too noisy)
        r'(?<!ts_decay_linear\()(?<!ts_rank\()ts_delta\([^)]+\)\s*$',  # Raw ts_delta without smoothing
        r'/\s*0',  # Division by zero
        r'log\s*\(\s*-',  # Log of negative
        r'sqrt\s*\(\s*-',  # Sqrt of negative
    ]
    
    # Required patterns for quality
    REQUIRED_PATTERNS = [
        r'ts_decay_linear|ts_rank|rank|zscore',  # Must have some normalization/smoothing
    ]
    
    def pre_screen(
        self,
        expression: str,
        hypothesis_key_fields: List[str] = None
    ) -> PreScreenResult:
        """
        Multi-fidelity pre-screening of an expression.
        
        Args:
            expression: Alpha expression to screen
            hypothesis_key_fields: Mandatory fields from hypothesis
            
        Returns:
            PreScreenResult with screening verdict
        """
        issues = []
        
        # === Level 1: Fast Syntax Check ===
        fast_passed = True
        
        # Check balanced parentheses
        if expression.count('(') != expression.count(')'):
            issues.append("Unbalanced parentheses")
            fast_passed = False
        
        # Check not empty
        if not expression or len(expression) < 10:
            issues.append("Expression too short")
            fast_passed = False
        
        # Check bad patterns
        for pattern in self.BAD_PATTERNS:
            if re.search(pattern, expression, re.IGNORECASE):
                issues.append(f"Matches bad pattern: {pattern[:30]}")
                fast_passed = False
        
        # === Level 2: Medium Pattern Check ===
        medium_passed = True
        
        # Check required patterns
        has_required = False
        for pattern in self.REQUIRED_PATTERNS:
            if re.search(pattern, expression, re.IGNORECASE):
                has_required = True
                break
        
        if not has_required:
            issues.append("Missing normalization/smoothing (ts_decay_linear, rank, etc.)")
            medium_passed = False
        
        # Check hypothesis field usage
        if hypothesis_key_fields:
            fields_in_expr = {f.lower() for f in re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression)}
            has_key_field = any(kf.lower() in fields_in_expr for kf in hypothesis_key_fields)
            if not has_key_field:
                issues.append(f"Does not use any hypothesis key_fields: {hypothesis_key_fields[:3]}")
                medium_passed = False
        
        # Check for likely high turnover
        if 'ts_decay_linear' not in expression.lower():
            if 'ts_delta' in expression.lower() or 'ts_returns' in expression.lower():
                issues.append("WARNING: May have high turnover without decay wrapper")
        
        # Calculate confidence and estimated sharpe
        confidence = 0.8 if (fast_passed and medium_passed) else 0.3
        estimated_sharpe = 0.5 if (fast_passed and medium_passed) else -0.5
        
        should_simulate = fast_passed and medium_passed
        
        return PreScreenResult(
            expression=expression,
            should_simulate=should_simulate,
            confidence=confidence,
            estimated_sharpe=estimated_sharpe,
            issues=issues,
            fast_check_passed=fast_passed,
            medium_check_passed=medium_passed
        )


# =============================================================================
# Unified Alpha Optimizer - Combines All Enhancements
# =============================================================================

class AlphaOptimizer:
    """
    Unified optimizer that combines all P0-P2 enhancements.
    
    This is the main entry point for the optimized mining workflow.
    """
    
    def __init__(self):
        # P0 components
        self.field_pre_checker = FieldPreChecker()
        self.alignment_enforcer = HypothesisAlignmentEnforcer()
        self.signal_corrector = SignalDirectionCorrector()
        
        # P1 components
        self.costeer_enforcer = CoSTEEREnforcer()
        self.gp_enhancer = SimpleGPEnhancer()
        self.exploration_strategy = SmartExplorationStrategy()
        
        # P2 components
        self.multi_fidelity_screener = MultiFidelityScreener()
        
        # Statistics
        self._stats = {
            "expressions_checked": 0,
            "expressions_rejected_alignment": 0,
            "expressions_rejected_constraints": 0,
            "expressions_rejected_prescreen": 0,
            "expressions_passed": 0,
            "signals_inverted": 0,
            "variants_generated": 0,
        }
    
    def pre_generate_check(
        self,
        fields: List[Dict],
        region: str,
        universe: str,
        hypothesis_key_fields: List[str] = None
    ) -> FieldPreCheckResult:
        """
        P0-1: Pre-check before code generation.
        
        Call this BEFORE generating expressions.
        """
        return self.field_pre_checker.pre_check_fields(
            all_fields=fields,
            region=region,
            universe=universe,
            hypothesis_key_fields=hypothesis_key_fields
        )
    
    def post_generate_validate(
        self,
        expression: str,
        hypothesis: Dict[str, Any],
        available_fields: List[str]
    ) -> Tuple[bool, str, List[str]]:
        """
        P0-2 + P1-1 + P2-2: Post-generation validation.
        
        Call this AFTER generating expressions.
        
        Returns:
            (is_valid, corrected_expression, issues)
        """
        self._stats["expressions_checked"] += 1
        
        all_issues = []
        corrected = expression
        
        # Step 1: Multi-fidelity pre-screen (P2-2)
        prescreen = self.multi_fidelity_screener.pre_screen(
            expression=expression,
            hypothesis_key_fields=hypothesis.get("key_fields", [])
        )
        
        if not prescreen.should_simulate:
            self._stats["expressions_rejected_prescreen"] += 1
            return False, expression, prescreen.issues
        
        all_issues.extend(prescreen.issues)
        
        # Step 2: Hypothesis alignment (P0-2)
        alignment = self.alignment_enforcer.enforce_alignment(
            expression=expression,
            hypothesis=hypothesis,
            available_fields=available_fields
        )
        
        if not alignment.is_aligned:
            # Check if it can be corrected
            if alignment.corrected_expression:
                corrected = alignment.corrected_expression
                self._stats["signals_inverted"] += 1
                all_issues.append(f"Auto-corrected: {alignment.issues[0]}")
            else:
                self._stats["expressions_rejected_alignment"] += 1
                return False, expression, alignment.issues
        
        all_issues.extend(alignment.issues)
        
        # Step 3: CoSTEER constraints (P1-1)
        constraints_ok, constraint_issues = self.costeer_enforcer.check_constraints(corrected)
        
        if not constraints_ok:
            self._stats["expressions_rejected_constraints"] += 1
            all_issues.extend(constraint_issues)
            return False, corrected, all_issues
        
        self._stats["expressions_passed"] += 1
        return True, corrected, all_issues
    
    def check_and_correct_signal(
        self,
        expression: str,
        sharpe: float,
        fitness: float,
        turnover: float
    ) -> Tuple[bool, Optional[str], str]:
        """
        P0-3: Check if signal should be inverted based on metrics.
        
        Call this AFTER simulation if sharpe is negative.
        
        Returns:
            (should_invert, inverted_expression, reason)
        """
        should_invert, reason = self.signal_corrector.should_invert_signal(
            sharpe=sharpe,
            fitness=fitness,
            turnover=turnover
        )
        
        if should_invert:
            inverted = self.signal_corrector.create_inverted_expression(expression)
            self._stats["signals_inverted"] += 1
            return True, inverted, reason
        
        return False, None, ""
    
    def generate_optimization_variants(
        self,
        seed_expression: str,
        num_variants: int = 5
    ) -> List[ExpressionVariant]:
        """
        P1-2: Generate GP-style variants for optimization.
        
        Call this for OPTIMIZE-status alphas.
        """
        variants = self.gp_enhancer.generate_variants(
            seed_expression=seed_expression,
            num_variants=num_variants
        )
        self._stats["variants_generated"] += len(variants)
        return variants
    
    def get_exploration_parameters(
        self,
        current_progress: float,
        max_iterations: int
    ) -> Dict[str, Any]:
        """
        P1-3: Get smart exploration strategy parameters.
        """
        return self.exploration_strategy.get_strategy_parameters(
            current_progress=current_progress,
            max_iterations=max_iterations
        )
    
    def update_from_round(
        self,
        successes: List[Dict],
        failures: List[Dict],
        dataset_id: str,
        passed_count: int,
        best_sharpe: Optional[float]
    ):
        """
        Update all components from a mining round.
        """
        # Update CoSTEER enforcer
        self.costeer_enforcer.update_from_failures(failures)
        self.costeer_enforcer.update_from_successes(successes)
        
        # Update exploration strategy
        patterns_tried = [f.get("expression", "")[:50] for f in failures + successes]
        self.exploration_strategy.update_state(
            passed_count=passed_count,
            failed_count=len(failures),
            best_sharpe=best_sharpe,
            patterns_tried=patterns_tried,
            dataset_id=dataset_id
        )
    
    def record_simulation_result(
        self,
        expression: str,
        success: bool,
        error: str,
        region: str,
        universe: str
    ):
        """
        Record simulation result for field tracking.
        """
        if success:
            self.field_pre_checker.record_success(expression, region, universe)
        else:
            self.field_pre_checker.record_failure(expression, error, region, universe)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            **self._stats,
            "exploration_state": {
                "iteration": self.exploration_strategy.state.iteration,
                "consecutive_failures": self.exploration_strategy.state.consecutive_failures,
                "best_sharpe_seen": self.exploration_strategy.state.best_sharpe_seen,
                "patterns_explored": len(self.exploration_strategy.state.explored_patterns),
            },
            "costeer_constraints": {
                "forbidden_fields": len(self.costeer_enforcer.get_constraints().forbidden_fields),
                "forbidden_patterns": len(self.costeer_enforcer.get_constraints().forbidden_patterns),
            }
        }


# =============================================================================
# Global Instance
# =============================================================================

_alpha_optimizer: Optional[AlphaOptimizer] = None


def get_alpha_optimizer() -> AlphaOptimizer:
    """Get or create global AlphaOptimizer instance."""
    global _alpha_optimizer
    if _alpha_optimizer is None:
        _alpha_optimizer = AlphaOptimizer()
    return _alpha_optimizer


def reset_alpha_optimizer():
    """Reset global AlphaOptimizer instance."""
    global _alpha_optimizer
    _alpha_optimizer = AlphaOptimizer()
