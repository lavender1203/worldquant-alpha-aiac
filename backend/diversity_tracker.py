"""
Diversity Tracker - Encourage exploration of diverse alpha combinations

Features:
1. Track tried combinations (dataset, fields, operators, settings)
2. Calculate diversity scores
3. Suggest underexplored directions
4. Prevent repetitive exploration

This module ensures the mining system explores diverse alphas rather than
getting stuck in local optima or repeating similar patterns.
"""

import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from loguru import logger

from backend.models import Alpha, KnowledgeEntry


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ExplorationRecord:
    """Record of a single exploration attempt."""
    dataset_id: str
    region: str
    universe: str
    
    # Expression components
    fields_used: List[str] = field(default_factory=list)
    operators_used: List[str] = field(default_factory=list)
    operator_skeleton: str = ""
    
    # Settings
    delay: int = 1
    decay: int = 0
    neutralization: str = "NONE"
    
    # Results
    was_successful: bool = False
    sharpe: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def fingerprint(self) -> str:
        """Generate unique fingerprint for this combination."""
        components = [
            self.dataset_id,
            self.region,
            self.operator_skeleton,
            str(sorted(self.fields_used)[:5]),  # Top 5 fields
            str(self.delay),
            self.neutralization
        ]
        return hashlib.md5("|".join(components).encode()).hexdigest()[:12]


@dataclass
class DiversityScore:
    """Diversity assessment for a proposed alpha."""
    # Component scores (0-1, higher = more diverse/novel)
    dataset_diversity: float = 0.0
    field_diversity: float = 0.0
    operator_diversity: float = 0.0
    settings_diversity: float = 0.0
    
    # Combined score
    overall_score: float = 0.0
    
    # Suggestions
    suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    similar_attempts: int = 0
    last_similar_attempt: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_diversity": round(self.dataset_diversity, 3),
            "field_diversity": round(self.field_diversity, 3),
            "operator_diversity": round(self.operator_diversity, 3),
            "settings_diversity": round(self.settings_diversity, 3),
            "overall_score": round(self.overall_score, 3),
            "similar_attempts": self.similar_attempts,
            "suggestions": self.suggestions,
        }


@dataclass
class ExplorationSuggestion:
    """Suggested direction for exploration."""
    dimension: str  # "dataset", "field", "operator", "setting"
    suggestion: str  # Human-readable suggestion
    priority: float  # 0-1, higher = more important
    underexplored_items: List[str] = field(default_factory=list)


# =============================================================================
# Diversity Tracker
# =============================================================================

class DiversityTracker:
    """
    Tracks exploration diversity and suggests new directions.
    
    Maintains:
    - History of tried combinations
    - Usage counts by dimension
    - Diversity metrics
    
    Usage:
        tracker = DiversityTracker(db)
        await tracker.initialize(region="USA")
        
        # Before generating alpha, check diversity
        score = tracker.evaluate_diversity(
            dataset_id="analyst15",
            fields=["eps_est", "revenue_est"],
            operators=["ts_rank", "ts_delta"]
        )
        
        # Get suggestions for underexplored areas
        suggestions = tracker.get_exploration_suggestions()
        
        # After mining, record attempt
        tracker.record_attempt(record)
    """
    
    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db
        
        # In-memory tracking (session-level)
        self.attempts: List[ExplorationRecord] = []
        self.fingerprints: Set[str] = set()
        
        # Usage counts
        self.dataset_usage: Dict[str, int] = defaultdict(int)
        self.field_usage: Dict[str, int] = defaultdict(int)
        self.operator_usage: Dict[str, int] = defaultdict(int)
        self.setting_usage: Dict[str, int] = defaultdict(int)
        
        # Success tracking
        self.dataset_success: Dict[str, int] = defaultdict(int)
        self.operator_success: Dict[str, int] = defaultdict(int)
        
        # Configuration
        self.region: str = "USA"
        self.max_attempts_memory: int = 1000
        
        # All available items (populated from DB or config)
        self.available_datasets: Set[str] = set()
        self.available_operators: Set[str] = set()
        
        self._initialized = False
    
    async def initialize(
        self,
        region: str = "USA",
        load_history: bool = True,
        history_days: int = 30
    ):
        """
        Initialize tracker with historical data.
        
        Args:
            region: Market region
            load_history: Whether to load past attempts from DB
            history_days: How many days of history to load
        """
        self.region = region
        
        if self.db and load_history:
            await self._load_historical_attempts(history_days)
        
        # Load available operators from database
        await self._load_available_operators()
        
        self._initialized = True
        logger.info(
            f"[DiversityTracker] Initialized | region={region} "
            f"history_attempts={len(self.attempts)} datasets={len(self.available_datasets)}"
        )
    
    async def _load_historical_attempts(self, days: int):
        """Load historical attempts from database."""
        if not self.db:
            return
        
        cutoff = datetime.now() - timedelta(days=days)
        
        try:
            query = select(Alpha).where(
                Alpha.region == self.region,
                Alpha.created_at >= cutoff
            ).limit(self.max_attempts_memory)
            
            result = await self.db.execute(query)
            alphas = result.scalars().all()
            
            for alpha in alphas:
                # Reconstruct exploration record
                record = ExplorationRecord(
                    dataset_id=alpha.dataset_id or "unknown",
                    region=alpha.region,
                    universe=alpha.universe,
                    fields_used=alpha.fields_used or [],
                    operators_used=alpha.operators_used or [],
                    operator_skeleton=self._extract_skeleton(alpha.expression),
                    delay=alpha.delay,
                    decay=alpha.decay,
                    neutralization=alpha.neutralization,
                    was_successful=alpha.quality_status == "PASS",
                    sharpe=alpha.is_sharpe or 0,
                    timestamp=alpha.created_at
                )
                
                self._update_usage_counts(record)
                self.fingerprints.add(record.fingerprint)
                self.available_datasets.add(record.dataset_id)
            
            logger.debug(f"[DiversityTracker] Loaded {len(alphas)} historical attempts")
            
        except Exception as e:
            logger.warning(f"[DiversityTracker] Failed to load history: {e}")
    
    async def _load_available_operators(self):
        """Load list of available operators from database."""
        # Try to load from database
        if self.db:
            try:
                from backend.models import Operator
                result = await self.db.execute(
                    select(Operator.name).where(Operator.is_active == True)
                )
                operators = result.scalars().all()
                if operators:
                    self.available_operators = {op.lower() for op in operators}
                    logger.debug(f"[DiversityTracker] Loaded {len(self.available_operators)} operators from DB")
                    return
            except Exception as e:
                logger.warning(f"[DiversityTracker] Failed to load operators from DB: {e}")
        
        # Fallback to semantic validator's registry
        from backend.alpha_semantic_validator import get_known_operators
        self.available_operators = get_known_operators()
        logger.debug(f"[DiversityTracker] Using {len(self.available_operators)} operators from registry")
    
    def _extract_skeleton(self, expression: str) -> str:
        """Extract operator skeleton from expression."""
        if not expression:
            return ""
        
        # Extract function names
        func_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        operators = []
        
        for match in func_pattern.finditer(expression):
            op = match.group(1).lower()
            if op in self.available_operators:
                operators.append(op)
        
        return "->".join(operators[:5])  # Top 5 operators
    
    def _update_usage_counts(self, record: ExplorationRecord):
        """Update usage counters from a record."""
        self.dataset_usage[record.dataset_id] += 1
        
        for field in record.fields_used:
            self.field_usage[field] += 1
        
        for op in record.operators_used:
            self.operator_usage[op] += 1
        
        setting_key = f"{record.delay}-{record.decay}-{record.neutralization}"
        self.setting_usage[setting_key] += 1
        
        if record.was_successful:
            self.dataset_success[record.dataset_id] += 1
            for op in record.operators_used:
                self.operator_success[op] += 1
    
    def record_attempt(self, record: ExplorationRecord):
        """
        Record a new exploration attempt.
        
        Args:
            record: ExplorationRecord with attempt details
        """
        # Add to memory
        self.attempts.append(record)
        self.fingerprints.add(record.fingerprint)
        
        # Update counts
        self._update_usage_counts(record)
        
        # Prune if too many
        if len(self.attempts) > self.max_attempts_memory:
            self.attempts = self.attempts[-self.max_attempts_memory:]
        
        logger.debug(
            f"[DiversityTracker] Recorded attempt | dataset={record.dataset_id} "
            f"success={record.was_successful} fingerprint={record.fingerprint[:8]}"
        )
    
    def evaluate_diversity(
        self,
        dataset_id: str,
        fields: List[str],
        operators: List[str],
        delay: int = 1,
        decay: int = 0,
        neutralization: str = "NONE"
    ) -> DiversityScore:
        """
        Evaluate diversity of a proposed alpha combination.
        
        Args:
            dataset_id: Proposed dataset
            fields: Proposed fields
            operators: Proposed operators
            delay: Trading delay
            decay: Decay setting
            neutralization: Neutralization setting
        
        Returns:
            DiversityScore with assessment and suggestions
        """
        score = DiversityScore()
        suggestions = []
        
        # 1. Dataset diversity
        ds_count = self.dataset_usage.get(dataset_id, 0)
        total_ds_attempts = sum(self.dataset_usage.values()) or 1
        ds_freq = ds_count / total_ds_attempts
        
        # Higher score for less-explored datasets
        score.dataset_diversity = max(0, 1.0 - ds_freq * 5)
        
        if ds_count > 10:
            suggestions.append(f"Dataset '{dataset_id}' heavily explored ({ds_count} attempts)")
        
        # 2. Field diversity
        if fields:
            field_counts = [self.field_usage.get(f, 0) for f in fields]
            avg_field_usage = sum(field_counts) / len(field_counts) if field_counts else 0
            max_field_count = max(self.field_usage.values()) if self.field_usage else 1
            score.field_diversity = max(0, 1.0 - avg_field_usage / (max_field_count + 1))
            
            overused_fields = [f for f, c in zip(fields, field_counts) if c > 5]
            if overused_fields:
                suggestions.append(f"Fields heavily used: {overused_fields[:3]}")
        else:
            score.field_diversity = 1.0
        
        # 3. Operator diversity
        if operators:
            op_counts = [self.operator_usage.get(op.lower(), 0) for op in operators]
            avg_op_usage = sum(op_counts) / len(op_counts) if op_counts else 0
            max_op_count = max(self.operator_usage.values()) if self.operator_usage else 1
            score.operator_diversity = max(0, 1.0 - avg_op_usage / (max_op_count + 1))
            
            # Suggest underused operators
            underused = self._get_underused_operators(operators)
            if underused:
                suggestions.append(f"Consider operators: {underused[:5]}")
        else:
            score.operator_diversity = 1.0
        
        # 4. Settings diversity
        setting_key = f"{delay}-{decay}-{neutralization}"
        setting_count = self.setting_usage.get(setting_key, 0)
        max_setting_count = max(self.setting_usage.values()) if self.setting_usage else 1
        score.settings_diversity = max(0, 1.0 - setting_count / (max_setting_count + 1))
        
        if setting_count > 20:
            suggestions.append(f"Setting combo '{setting_key}' overused ({setting_count} times)")
        
        # Check fingerprint collision
        temp_record = ExplorationRecord(
            dataset_id=dataset_id,
            region=self.region,
            universe="",
            fields_used=fields,
            operators_used=operators,
            operator_skeleton="->".join(operators[:5]),
            delay=delay,
            decay=decay,
            neutralization=neutralization
        )
        
        if temp_record.fingerprint in self.fingerprints:
            score.similar_attempts += 1
            suggestions.append("Nearly identical combination was tried before")
        
        # Calculate overall score
        score.overall_score = (
            0.30 * score.dataset_diversity +
            0.30 * score.field_diversity +
            0.25 * score.operator_diversity +
            0.15 * score.settings_diversity
        )
        
        score.suggestions = suggestions
        
        return score
    
    def _get_underused_operators(self, current_ops: List[str]) -> List[str]:
        """Get operators that are underused relative to others."""
        current_set = set(op.lower() for op in current_ops)
        
        # Calculate average usage
        avg_usage = sum(self.operator_usage.values()) / len(self.operator_usage) if self.operator_usage else 0
        
        underused = []
        for op in self.available_operators:
            if op not in current_set:
                usage = self.operator_usage.get(op, 0)
                if usage < avg_usage * 0.5:  # Less than half average
                    underused.append(op)
        
        # Sort by usage (least used first)
        underused.sort(key=lambda x: self.operator_usage.get(x, 0))
        
        return underused
    
    def get_exploration_suggestions(self, n: int = 5) -> List[ExplorationSuggestion]:
        """
        Get suggestions for underexplored directions.
        
        Returns:
            List of exploration suggestions prioritized by novelty
        """
        suggestions = []
        
        # 1. Underexplored datasets
        if self.available_datasets:
            ds_by_usage = sorted(
                self.available_datasets,
                key=lambda x: self.dataset_usage.get(x, 0)
            )
            underexplored_ds = ds_by_usage[:5]
            
            if underexplored_ds:
                suggestions.append(ExplorationSuggestion(
                    dimension="dataset",
                    suggestion="Try underexplored datasets",
                    priority=0.9,
                    underexplored_items=underexplored_ds
                ))
        
        # 2. Underused operators
        underused_ops = self._get_underused_operators([])[:10]
        if underused_ops:
            suggestions.append(ExplorationSuggestion(
                dimension="operator",
                suggestion="Incorporate underused operators",
                priority=0.8,
                underexplored_items=underused_ops
            ))
        
        # 3. Unexplored settings combinations
        common_settings = [
            ("delay", [0, 1]),
            ("decay", [0, 2, 4, 6, 8]),
            ("neutralization", ["NONE", "MARKET", "INDUSTRY", "SUBINDUSTRY"]),
        ]
        
        underused_settings = []
        for setting_name, values in common_settings:
            for val in values:
                setting_key_prefix = str(val)
                usage = sum(
                    count for key, count in self.setting_usage.items()
                    if setting_key_prefix in key.split("-")
                )
                if usage < 5:
                    underused_settings.append(f"{setting_name}={val}")
        
        if underused_settings:
            suggestions.append(ExplorationSuggestion(
                dimension="setting",
                suggestion="Try different settings combinations",
                priority=0.7,
                underexplored_items=underused_settings[:5]
            ))
        
        # 4. High-success but underexplored combinations
        if self.dataset_success:
            # Find datasets with good success rate but low attempt count
            promising = []
            for ds, success_count in self.dataset_success.items():
                total = self.dataset_usage.get(ds, 0)
                if total > 0 and total < 10:  # Low attempts
                    success_rate = success_count / total
                    if success_rate > 0.2:  # > 20% success
                        promising.append(ds)
            
            if promising:
                suggestions.append(ExplorationSuggestion(
                    dimension="dataset",
                    suggestion="Promising datasets with low exploration",
                    priority=0.95,
                    underexplored_items=promising[:5]
                ))
        
        # Sort by priority
        suggestions.sort(key=lambda x: x.priority, reverse=True)
        
        return suggestions[:n]
    
    def get_diversity_stats(self) -> Dict[str, Any]:
        """Get statistics about exploration diversity."""
        return {
            "total_attempts": len(self.attempts),
            "unique_fingerprints": len(self.fingerprints),
            "datasets_explored": len(self.dataset_usage),
            "operators_used": len(self.operator_usage),
            "fields_used": len(self.field_usage),
            "settings_combos": len(self.setting_usage),
            "top_datasets": sorted(
                self.dataset_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "top_operators": sorted(
                self.operator_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "success_rate_by_dataset": {
                ds: self.dataset_success.get(ds, 0) / count if count > 0 else 0
                for ds, count in self.dataset_usage.items()
            },
        }
    
    def should_force_diversity(self) -> Tuple[bool, str]:
        """
        Check if system should be forced to explore new directions.
        
        Returns:
            (should_force, reason)
        """
        recent_attempts = [a for a in self.attempts if 
                          (datetime.now() - a.timestamp).total_seconds() < 3600]  # Last hour
        
        if not recent_attempts:
            return False, "No recent attempts"
        
        # Check for repetition
        recent_fingerprints = [a.fingerprint for a in recent_attempts]
        unique_ratio = len(set(recent_fingerprints)) / len(recent_fingerprints)
        
        if unique_ratio < 0.5:
            return True, f"High repetition detected ({unique_ratio:.1%} unique)"
        
        # Check for dataset concentration
        recent_datasets = [a.dataset_id for a in recent_attempts]
        if len(set(recent_datasets)) < 3 and len(recent_datasets) > 10:
            return True, f"Dataset concentration: only {len(set(recent_datasets))} datasets in last {len(recent_datasets)} attempts"
        
        # Check for success rate
        recent_success_rate = sum(1 for a in recent_attempts if a.was_successful) / len(recent_attempts)
        if recent_success_rate < 0.05 and len(recent_attempts) > 20:
            return True, f"Very low success rate ({recent_success_rate:.1%}), need new direction"
        
        return False, "Diversity OK"


# =============================================================================
# OPERATOR PATTERN CONSTRAINTS (P0 Fix for Expression Diversity)
# =============================================================================

# DYNAMIC ARCHETYPE DETECTION
# Instead of hardcoding operators, we detect archetypes by name patterns
# This allows new operators to be automatically categorized

ARCHETYPE_PATTERNS = {
    "momentum": {
        "patterns": [r"_delta", r"_returns", r"_momentum", r"_diff"],
        "description": "Captures changes over time",
    },
    "ranking": {
        "patterns": [r"^rank$", r"_rank", r"percentile"],
        "description": "Cross-sectional or time-series ranking",
    },
    "normalization": {
        "patterns": [r"zscore", r"_scale$", r"normalize"],
        "description": "Statistical normalization",
    },
    "correlation": {
        "patterns": [r"_corr", r"_cov"],
        "description": "Relationship between fields",
    },
    "volatility": {
        "patterns": [r"std_dev", r"variance", r"_atr", r"_range$"],
        "description": "Dispersion measures",
    },
    "extremes": {
        "patterns": [r"_max", r"_min", r"argmax", r"argmin"],
        "description": "Extreme values",
    },
    "central": {
        "patterns": [r"_mean", r"_median", r"_sum", r"_avg", r"_product"],
        "description": "Central tendency",
    },
    "group": {
        "patterns": [r"^group_"],
        "description": "Group/sector relative",
    },
    "smoothing": {
        "patterns": [r"decay", r"^hump$", r"pasteurize"],
        "description": "Signal smoothing",
    },
    "conditional": {
        "patterns": [r"^if_", r"trade_when", r"filter"],
        "description": "Conditional logic",
    },
}


def classify_operator(operator_name: str) -> str:
    """
    Classify an operator into an archetype using pattern matching.
    
    This is DYNAMIC - works with any operator, not just hardcoded ones.
    """
    import re
    op_lower = operator_name.lower()
    
    for archetype, info in ARCHETYPE_PATTERNS.items():
        for pattern in info["patterns"]:
            if re.search(pattern, op_lower):
                return archetype
    
    return "other"


def get_archetype_description(archetype: str) -> str:
    """Get description for an archetype."""
    info = ARCHETYPE_PATTERNS.get(archetype)
    return info["description"] if info else "Other operations"


# For backward compatibility - dynamically build OPERATOR_ARCHETYPES
# But this is now auto-populated, not hardcoded
def _build_operator_archetypes_from_db():
    """Build archetype mapping from database operators."""
    try:
        from backend.alpha_semantic_validator import get_known_operators
        operators = get_known_operators()
        
        archetypes = {arch: {"core_ops": [], "description": info["description"]}
                      for arch, info in ARCHETYPE_PATTERNS.items()}
        archetypes["other"] = {"core_ops": [], "description": "Other operations"}
        
        for op in operators:
            archetype = classify_operator(op)
            archetypes[archetype]["core_ops"].append(op)
        
        return archetypes
    except Exception:
        # Fallback - return pattern-based classification
        return {arch: {"core_ops": [], "description": info["description"]}
                for arch, info in ARCHETYPE_PATTERNS.items()}


# Lazy-loaded archetypes
_operator_archetypes_cache = None

def get_operator_archetypes() -> Dict:
    """Get operator archetypes (lazy-loaded from DB)."""
    global _operator_archetypes_cache
    if _operator_archetypes_cache is None:
        _operator_archetypes_cache = _build_operator_archetypes_from_db()
    return _operator_archetypes_cache


# For backward compatibility
OPERATOR_ARCHETYPES = ARCHETYPE_PATTERNS  # Use patterns instead of hardcoded ops


@dataclass 
class OperatorQuota:
    """Tracks operator archetype usage to enforce diversity."""
    archetype: str
    max_per_batch: int = 2  # Max alphas using this archetype per batch
    current_count: int = 0
    cooldown_rounds: int = 0  # Rounds to wait before reusing


class OperatorDiversityManager:
    """
    Manages operator diversity constraints to prevent template traps.
    
    The main issue identified is that LLMs tend to generate expressions like:
    `ts_decay_linear(ts_rank(...), N)` repeatedly.
    
    This manager:
    1. Tracks which operator archetypes have been used (via pattern matching)
    2. Enforces quotas per batch
    3. Suggests alternative archetypes for diversity
    4. Prevents over-reliance on any single pattern
    
    NOTE: Uses dynamic pattern matching, NOT hardcoded operator lists.
    """
    
    def __init__(self):
        # Use pattern-based archetypes
        self.quotas: Dict[str, OperatorQuota] = {
            archetype: OperatorQuota(archetype=archetype)
            for archetype in ARCHETYPE_PATTERNS
        }
        self.batch_usage: Dict[str, List[str]] = defaultdict(list)
        self.historical_patterns: List[str] = []
        self.consecutive_same_pattern: int = 0
        self.last_dominant_pattern: str = ""
        
    def reset_batch(self):
        """Reset counters for a new batch."""
        for quota in self.quotas.values():
            quota.current_count = 0
        self.batch_usage.clear()
    
    def classify_expression(self, expression: str) -> List[str]:
        """
        Classify an expression into operator archetypes.
        
        Uses DYNAMIC pattern matching - works with any operator.
        """
        if not expression:
            return []
        
        # Extract operators from expression
        import re
        func_pattern = re.compile(r'([a-z_][a-z0-9_]*)\s*\(', re.IGNORECASE)
        operators = func_pattern.findall(expression.lower())
        
        # Classify each operator and collect unique archetypes
        archetypes_found = set()
        for op in operators:
            archetype = classify_operator(op)
            if archetype != "other":
                archetypes_found.add(archetype)
        
        return list(archetypes_found)
    
    def can_use_archetype(self, archetype: str) -> bool:
        """Check if an archetype can still be used in this batch."""
        if archetype not in self.quotas:
            return True
        quota = self.quotas[archetype]
        return quota.current_count < quota.max_per_batch and quota.cooldown_rounds == 0
    
    def record_usage(self, expression: str):
        """Record usage of an expression and update quotas."""
        archetypes = self.classify_expression(expression)
        self.batch_usage[expression] = archetypes
        
        for archetype in archetypes:
            if archetype in self.quotas:
                self.quotas[archetype].current_count += 1
        
        # Track pattern for repetition detection
        pattern_key = "->".join(sorted(archetypes))
        if pattern_key == self.last_dominant_pattern:
            self.consecutive_same_pattern += 1
        else:
            self.consecutive_same_pattern = 0
            self.last_dominant_pattern = pattern_key
        
        self.historical_patterns.append(pattern_key)
        
    def get_available_archetypes(self) -> List[str]:
        """Get list of archetypes that can still be used."""
        return [
            archetype for archetype, quota in self.quotas.items()
            if quota.current_count < quota.max_per_batch and quota.cooldown_rounds == 0
        ]
    
    def get_required_archetypes(self, num_alphas: int) -> List[str]:
        """
        Get list of archetypes that MUST be used to ensure diversity.
        
        If generating N alphas, require N different primary archetypes.
        """
        available = self.get_available_archetypes()
        
        # Prioritize underused archetypes
        archetype_usage = defaultdict(int)
        for pattern in self.historical_patterns[-100:]:  # Last 100 patterns
            for archetype in pattern.split("->"):
                if archetype:
                    archetype_usage[archetype] += 1
        
        # Sort by usage (ascending - least used first)
        available_sorted = sorted(
            available, 
            key=lambda x: archetype_usage.get(x, 0)
        )
        
        return available_sorted[:num_alphas]
    
    def get_diversity_constraints(self, num_alphas: int) -> Dict[str, Any]:
        """
        Get diversity constraints to include in generation prompt.
        
        Returns a dictionary with:
        - required_archetypes: Archetypes that must be used
        - forbidden_archetypes: Archetypes to avoid this batch
        - suggestions: Specific operator suggestions
        """
        required = self.get_required_archetypes(num_alphas)
        
        # Find overused archetypes
        forbidden = []
        for archetype, quota in self.quotas.items():
            if quota.cooldown_rounds > 0:
                forbidden.append(archetype)
        
        # Build specific suggestions using dynamic patterns
        suggestions = []
        for archetype in required:
            info = ARCHETYPE_PATTERNS.get(archetype, {})
            description = info.get("description", "Various operations")
            # Don't provide hardcoded example ops - let LLM explore
            suggestions.append({
                "archetype": archetype,
                "description": description,
                "example_ops": [],  # No hardcoded ops - encourage exploration
                "example": f"Try operators matching: {archetype}"
            })
        
        return {
            "required_archetypes": required,
            "forbidden_archetypes": forbidden,
            "suggestions": suggestions,
            "must_be_different": True,  # Flag that each alpha must use different archetype
            "repetition_warning": self.consecutive_same_pattern > 2
        }
    
    def advance_round(self):
        """Advance to next round - update cooldowns."""
        for quota in self.quotas.values():
            if quota.cooldown_rounds > 0:
                quota.cooldown_rounds -= 1
            # If heavily used last batch, add cooldown
            if quota.current_count >= quota.max_per_batch:
                quota.cooldown_rounds = 1
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern usage."""
        if not self.historical_patterns:
            return {"total_patterns": 0}
        
        pattern_counts = defaultdict(int)
        for p in self.historical_patterns:
            pattern_counts[p] += 1
        
        most_common = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        unique_ratio = len(set(self.historical_patterns)) / len(self.historical_patterns)
        
        return {
            "total_patterns": len(self.historical_patterns),
            "unique_patterns": len(set(self.historical_patterns)),
            "unique_ratio": round(unique_ratio, 3),
            "most_common": most_common,
            "consecutive_same": self.consecutive_same_pattern
        }


# Global instance for session-level tracking
_operator_diversity_manager: Optional[OperatorDiversityManager] = None


def get_operator_diversity_manager() -> OperatorDiversityManager:
    """Get or create the global operator diversity manager."""
    global _operator_diversity_manager
    if _operator_diversity_manager is None:
        _operator_diversity_manager = OperatorDiversityManager()
    return _operator_diversity_manager


def reset_operator_diversity_manager():
    """Reset the global operator diversity manager (for new task)."""
    global _operator_diversity_manager
    _operator_diversity_manager = OperatorDiversityManager()


async def persist_diversity_stats(db) -> bool:
    """
    P1 FIX: Persist operator diversity statistics to the database.
    
    This ensures diversity learning persists across restarts.
    Stores stats as a KnowledgeEntry for easy retrieval.
    
    Args:
        db: Async database session
    
    Returns:
        True if successful
    """
    global _operator_diversity_manager
    if _operator_diversity_manager is None:
        return False
    
    try:
        from backend.models import KnowledgeEntry
        from sqlalchemy import select
        from datetime import datetime
        import json
        
        stats = _operator_diversity_manager.get_pattern_statistics()
        
        # Find existing stats entry
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.entry_type == 'OPERATOR_DIVERSITY_STATS',
            KnowledgeEntry.is_active == True
        )
        result = await db.execute(query)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing
            existing.meta_data = {
                **(existing.meta_data or {}),
                "total_patterns": stats.get("total_patterns", 0),
                "unique_patterns": stats.get("unique_patterns", 0),
                "unique_ratio": stats.get("unique_ratio", 0),
                "most_common": stats.get("most_common", []),
                "consecutive_same": stats.get("consecutive_same", 0),
                "historical_patterns": _operator_diversity_manager.historical_patterns[-100:],
                "last_updated": datetime.now().isoformat(),
            }
        else:
            # Create new
            entry = KnowledgeEntry(
                entry_type='OPERATOR_DIVERSITY_STATS',
                pattern='OPERATOR_DIVERSITY_TRACKING',
                description='Tracks operator pattern usage for diversity enforcement',
                is_active=True,
                meta_data={
                    "total_patterns": stats.get("total_patterns", 0),
                    "unique_patterns": stats.get("unique_patterns", 0),
                    "unique_ratio": stats.get("unique_ratio", 0),
                    "most_common": stats.get("most_common", []),
                    "consecutive_same": stats.get("consecutive_same", 0),
                    "historical_patterns": _operator_diversity_manager.historical_patterns[-100:],
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                },
            )
            db.add(entry)
        
        await db.commit()
        logger.info("[DiversityTracker] Persisted operator diversity stats to database")
        return True
        
    except Exception as e:
        logger.error(f"[DiversityTracker] Failed to persist stats: {e}")
        return False


async def load_diversity_stats(db) -> bool:
    """
    P1 FIX: Load operator diversity statistics from the database.
    
    Restores diversity learning from previous sessions.
    
    Args:
        db: Async database session
    
    Returns:
        True if successful
    """
    global _operator_diversity_manager
    
    try:
        from backend.models import KnowledgeEntry
        from sqlalchemy import select
        
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.entry_type == 'OPERATOR_DIVERSITY_STATS',
            KnowledgeEntry.is_active == True
        )
        result = await db.execute(query)
        entry = result.scalar_one_or_none()
        
        if not entry or not entry.meta_data:
            logger.info("[DiversityTracker] No existing diversity stats found in database")
            return False
        
        # Initialize manager if needed
        if _operator_diversity_manager is None:
            _operator_diversity_manager = OperatorDiversityManager()
        
        # Restore historical patterns
        historical = entry.meta_data.get("historical_patterns", [])
        if historical:
            _operator_diversity_manager.historical_patterns = list(historical)
            _operator_diversity_manager.last_dominant_pattern = historical[-1] if historical else ""
            logger.info(
                f"[DiversityTracker] Loaded {len(historical)} historical patterns from database"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"[DiversityTracker] Failed to load stats: {e}")
        return False


# =============================================================================
# Helper Functions
# =============================================================================

def create_exploration_record(
    expression: str,
    dataset_id: str,
    region: str,
    universe: str,
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "NONE",
    was_successful: bool = False,
    sharpe: float = 0.0
) -> ExplorationRecord:
    """
    Create an exploration record from an alpha expression.
    
    Extracts fields and operators automatically.
    """
    # Extract operators
    func_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    operators = []
    for match in func_pattern.finditer(expression):
        op = match.group(1).lower()
        operators.append(op)
    
    # Extract potential field names (words that aren't operators)
    # Use dynamic operator detection instead of hardcoded list
    try:
        from backend.alpha_semantic_validator import get_known_operators
        known_operators = get_known_operators()
    except Exception:
        # Fallback: detect by naming patterns (prefixes common to operators)
        known_operators = set()
    
    # Also treat function calls as operators
    called_functions = set(operators)
    
    word_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
    fields = []
    for match in word_pattern.finditer(expression):
        word = match.group(1).lower()
        # Skip if it's a known operator, a called function, or a number
        if word not in known_operators and word not in called_functions and not word.isdigit():
            # Skip common operator prefixes
            if not any(word.startswith(p) for p in ['ts_', 'group_', 'vec_', 'log', 'sqrt', 'abs', 'sign']):
                fields.append(word)
    
    # Deduplicate
    fields = list(dict.fromkeys(fields))[:10]
    operators = list(dict.fromkeys(operators))[:10]
    
    return ExplorationRecord(
        dataset_id=dataset_id,
        region=region,
        universe=universe,
        fields_used=fields,
        operators_used=operators,
        operator_skeleton="->".join(operators[:5]),
        delay=delay,
        decay=decay,
        neutralization=neutralization,
        was_successful=was_successful,
        sharpe=sharpe
    )
