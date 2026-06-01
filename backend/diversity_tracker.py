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
        
        await self._load_available_datasets()

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

    async def _load_available_datasets(self):
        """Load available datasets for underexplored-dataset suggestions."""
        if not self.db:
            return

        try:
            from backend.models import DatasetMetadata

            result = await self.db.execute(
                select(DatasetMetadata.dataset_id).where(
                    DatasetMetadata.region == self.region,
                    DatasetMetadata.is_active == True,
                )
            )
            datasets = {dataset_id for dataset_id in result.scalars().all() if dataset_id}
            if datasets:
                self.available_datasets.update(datasets)
                logger.debug(f"[DiversityTracker] Loaded {len(datasets)} datasets from DB")
        except Exception as e:
            logger.warning(f"[DiversityTracker] Failed to load datasets from DB: {e}")
    
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
        neutralization_options = ["NONE", "MARKET", "INDUSTRY", "SUBINDUSTRY"]
        if str(self.region).upper() == "IND":
            neutralization_options = [
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

        common_settings = [
            ("delay", [0, 1]),
            ("decay", [0, 2, 4, 6, 8]),
            ("neutralization", neutralization_options),
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
    known_operators = {
        "ts_rank", "ts_zscore", "ts_mean", "ts_sum", "ts_delta", "ts_std_dev",
        "ts_decay_linear", "rank", "zscore", "group_rank", "group_neutralize",
        "vec_sum", "vec_avg", "log", "sqrt", "abs", "sign", "add", "subtract",
        "multiply", "divide", "if_else", "trade_when", "pasteurize"
    }
    
    word_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
    fields = []
    for match in word_pattern.finditer(expression):
        word = match.group(1).lower()
        if word not in known_operators and not word.isdigit():
            # Likely a field name
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
