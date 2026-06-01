"""
Selection Strategy Module - Bandit algorithms and multi-objective scoring for dataset/field selection.

P1-1: Dataset selection with UCB/Thompson Sampling
P1-2: Field selection with multi-objective scoring
P1-4: Diversity constraints

This module implements "exploration-exploitation" strategies to improve alpha mining efficiency.
"""

import math
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger


# =============================================================================
# P1-1: Dataset Bandit Selection
# =============================================================================

@dataclass
class DatasetArm:
    """Bandit arm for dataset selection"""
    dataset_id: str
    region: str = "USA"
    universe: str = "TOP3000"
    
    # Reward statistics
    total_pulls: int = 0
    total_reward: float = 0.0
    success_count: int = 0
    fail_count: int = 0
    
    # Prior info (from platform metadata)
    pyramid_multiplier: float = 1.0
    alpha_count: int = 0  # Platform saturation indicator
    field_count: int = 0
    
    # Time decay
    last_pulled: Optional[datetime] = None
    
    @property
    def mean_reward(self) -> float:
        if self.total_pulls == 0:
            return 0.0
        return self.total_reward / self.total_pulls
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5  # Prior
        return self.success_count / total


class DatasetBandit:
    """
    Multi-Armed Bandit for dataset selection.
    
    Uses Upper Confidence Bound (UCB1) with prior bonuses for:
    - Pyramid multiplier (encourage high-value datasets)
    - Inverse alpha count (penalize saturated datasets)
    - Field count (more fields = more exploration space)
    """
    
    def __init__(
        self,
        exploration_weight: float = 2.0,
        pyramid_bonus_weight: float = 0.3,
        saturation_penalty_weight: float = 0.2,
        time_decay_days: int = 7
    ):
        self.arms: Dict[str, DatasetArm] = {}
        self.exploration_weight = exploration_weight
        self.pyramid_bonus_weight = pyramid_bonus_weight
        self.saturation_penalty_weight = saturation_penalty_weight
        self.time_decay_days = time_decay_days
        self.total_pulls = 0
        
    def add_arm(self, arm: DatasetArm):
        """Add or update a dataset arm"""
        key = f"{arm.dataset_id}:{arm.region}:{arm.universe}"
        self.arms[key] = arm
        
    def add_from_metadata(self, datasets: List[Dict], region: str = "USA", universe: str = "TOP3000"):
        """Initialize arms from dataset metadata"""
        for ds in datasets:
            arm = DatasetArm(
                dataset_id=ds.get("id") or ds.get("dataset_id", ""),
                region=region,
                universe=universe,
                pyramid_multiplier=ds.get("pyramid_multiplier", 1.0) or 1.0,
                alpha_count=ds.get("alpha_count", 0) or 0,
                field_count=ds.get("field_count", 0) or 0,
            )
            if arm.dataset_id:
                self.add_arm(arm)
                
    def select(self, n: int = 1, exclude: Optional[Set[str]] = None) -> List[DatasetArm]:
        """
        Select top N datasets using UCB1 with priors.
        
        UCB1 score = mean_reward + exploration_bonus + prior_bonus
        """
        exclude = exclude or set()
        candidates = [a for k, a in self.arms.items() if a.dataset_id not in exclude]
        
        if not candidates:
            return []
            
        # Calculate UCB scores
        scores = []
        for arm in candidates:
            ucb_score = self._calculate_ucb(arm)
            scores.append((ucb_score, arm))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N
        return [arm for _, arm in scores[:n]]
    
    def _calculate_ucb(self, arm: DatasetArm) -> float:
        """Calculate UCB1 score with priors"""
        # Base mean reward
        mean_reward = arm.mean_reward
        
        # Exploration bonus (UCB1)
        if arm.total_pulls == 0:
            exploration_bonus = float('inf')  # Unexplored arms get priority
        else:
            exploration_bonus = self.exploration_weight * math.sqrt(
                2 * math.log(max(self.total_pulls, 1)) / arm.total_pulls
            )
        
        # Prior bonuses
        pyramid_bonus = self.pyramid_bonus_weight * (arm.pyramid_multiplier - 1.0)
        
        # Saturation penalty (more alphas = more saturated = lower score)
        # Normalize: assume 10000 alphas is "fully saturated"
        saturation = min(arm.alpha_count / 10000.0, 1.0)
        saturation_penalty = self.saturation_penalty_weight * saturation
        
        # Time decay bonus (encourage re-visiting stale datasets)
        time_bonus = 0.0
        if arm.last_pulled:
            days_since = (datetime.now() - arm.last_pulled).days
            if days_since > self.time_decay_days:
                time_bonus = 0.1 * min(days_since / self.time_decay_days, 3.0)
        else:
            time_bonus = 0.2  # Never pulled bonus
        
        return mean_reward + exploration_bonus + pyramid_bonus - saturation_penalty + time_bonus
    
    def update(self, dataset_id: str, reward: float, success: bool = True, region: str = "USA", universe: str = "TOP3000"):
        """Update arm statistics after a pull"""
        key = f"{dataset_id}:{region}:{universe}"
        
        if key not in self.arms:
            # Create arm if not exists
            self.arms[key] = DatasetArm(dataset_id=dataset_id, region=region, universe=universe)
            
        arm = self.arms[key]
        arm.total_pulls += 1
        arm.total_reward += reward
        arm.last_pulled = datetime.now()
        
        if success:
            arm.success_count += 1
        else:
            arm.fail_count += 1
            
        self.total_pulls += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics"""
        return {
            "total_arms": len(self.arms),
            "total_pulls": self.total_pulls,
            "top_5_by_reward": [
                {"id": a.dataset_id, "mean_reward": a.mean_reward, "pulls": a.total_pulls}
                for a in sorted(self.arms.values(), key=lambda x: x.mean_reward, reverse=True)[:5]
            ],
            "least_explored": [
                {"id": a.dataset_id, "pulls": a.total_pulls}
                for a in sorted(self.arms.values(), key=lambda x: x.total_pulls)[:5]
            ]
        }


# =============================================================================
# P1-2: Field Selection with Multi-Objective Scoring
# =============================================================================

@dataclass
class FieldScore:
    """Scored field for selection"""
    field_id: str
    field_type: str = "MATRIX"
    score: float = 0.0
    
    # Component scores
    coverage_score: float = 0.0
    novelty_score: float = 0.0  # Inverse of alpha_count (crowding penalty)
    pyramid_score: float = 0.0
    
    # Metadata
    coverage: float = 1.0
    alpha_count: int = 0
    pyramid_multiplier: float = 1.0
    description: str = ""


class FieldSelector:
    """
    Multi-objective field selection for alpha generation.
    
    Scores fields based on:
    - Coverage (prefer high coverage for stability)
    - Novelty (penalize crowded fields with many alphas)
    - Pyramid multiplier (encourage high-value fields)
    - Type diversity (balance MATRIX vs VECTOR)
    """
    
    def __init__(
        self,
        coverage_weight: float = 0.3,
        novelty_weight: float = 0.4,
        pyramid_weight: float = 0.3,
        min_coverage: float = 0.3
    ):
        self.coverage_weight = coverage_weight
        self.novelty_weight = novelty_weight
        self.pyramid_weight = pyramid_weight
        self.min_coverage = min_coverage
        
    def score_fields(self, fields: List[Dict]) -> List[FieldScore]:
        """Score all fields and return sorted list"""
        if not fields:
            return []
            
        # Calculate normalization factors
        max_alpha_count = max((f.get("alpha_count", 0) or 0) for f in fields) or 1
        max_pyramid = max((f.get("pyramid_multiplier", 1.0) or 1.0) for f in fields) or 1.0
        
        scored = []
        for f in fields:
            coverage = f.get("coverage", 1.0) or 1.0
            alpha_count = f.get("alpha_count", 0) or 0
            pyramid = f.get("pyramid_multiplier", 1.0) or 1.0
            
            # Skip low coverage fields
            if coverage < self.min_coverage:
                continue
                
            # Coverage score (0-1)
            coverage_score = coverage
            
            # Novelty score (inverse of crowding, 0-1)
            # Fields with fewer alphas get higher novelty
            novelty_score = 1.0 - (alpha_count / max_alpha_count)
            
            # Pyramid score (normalized, 0-1)
            pyramid_score = (pyramid - 1.0) / max(max_pyramid - 1.0, 0.1)
            
            # Weighted total
            total_score = (
                self.coverage_weight * coverage_score +
                self.novelty_weight * novelty_score +
                self.pyramid_weight * pyramid_score
            )
            
            scored.append(FieldScore(
                field_id=f.get("id") or f.get("name", ""),
                field_type=f.get("type", "MATRIX"),
                score=total_score,
                coverage_score=coverage_score,
                novelty_score=novelty_score,
                pyramid_score=pyramid_score,
                coverage=coverage,
                alpha_count=alpha_count,
                pyramid_multiplier=pyramid,
                description=f.get("description", "")
            ))
            
        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored
    
    def select_diverse(
        self,
        fields: List[Dict],
        n: int = 20,
        matrix_ratio: float = 0.7
    ) -> List[Dict]:
        """
        Select top N fields with type diversity constraint.
        
        Args:
            fields: List of field dicts
            n: Number of fields to select
            matrix_ratio: Target ratio of MATRIX fields (default 70%)
        """
        scored = self.score_fields(fields)
        
        if not scored:
            return fields[:n] if fields else []
            
        # Separate by type
        matrix_fields = [f for f in scored if f.field_type == "MATRIX"]
        vector_fields = [f for f in scored if f.field_type == "VECTOR"]
        
        # Calculate target counts
        target_matrix = int(n * matrix_ratio)
        target_vector = n - target_matrix
        
        # Select with fallback
        selected_matrix = matrix_fields[:target_matrix]
        selected_vector = vector_fields[:target_vector]
        
        # Fill remaining slots
        remaining = n - len(selected_matrix) - len(selected_vector)
        if remaining > 0:
            all_remaining = [f for f in scored if f not in selected_matrix and f not in selected_vector]
            selected_extra = all_remaining[:remaining]
        else:
            selected_extra = []
            
        # Combine and return as dicts
        selected = selected_matrix + selected_vector + selected_extra
        
        # Map back to original dicts
        selected_ids = {f.field_id for f in selected}
        return [f for f in fields if (f.get("id") or f.get("name", "")) in selected_ids][:n]


# =============================================================================
# P1-4: Diversity Constraints for Batch Evaluation
# =============================================================================

def extract_operator_ngrams(expression: str, n: int = 2) -> Set[Tuple[str, ...]]:
    """Extract operator n-grams from expression for similarity"""
    import re
    
    # Extract function calls
    func_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    operators = [m.group(1).lower() for m in func_pattern.finditer(expression)]
    
    if len(operators) < n:
        return {tuple(operators)} if operators else set()
        
    ngrams = set()
    for i in range(len(operators) - n + 1):
        ngrams.add(tuple(operators[i:i+n]))
        
    return ngrams


def extract_field_set(expression: str, known_operators: Optional[Set[str]] = None) -> Set[str]:
    """Extract field identifiers from expression"""
    import re
    
    if known_operators is None:
        known_operators = {
            "ts_mean", "ts_rank", "ts_delta", "ts_std_dev", "ts_corr", "ts_zscore",
            "rank", "group_rank", "group_neutralize", "group_mean", "group_zscore",
            "vec_sum", "vec_avg", "vec_max", "vec_min",
            "abs", "log", "sign", "sqrt", "sigmoid", "tanh",
            "add", "subtract", "multiply", "divide", "power",
            "if_else", "max", "min", "winsorize", "scale", "normalize"
        }
    
    # Skip keywords
    skip = {
        "true", "false", "nan", "inf",
        "sector", "subindustry", "industry", "exchange", "country", "market"
    }
    
    pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
    fields = set()
    
    for match in pattern.finditer(expression):
        ident = match.group(1).lower()
        if ident not in known_operators and ident not in skip:
            fields.add(ident)
            
    return fields


def calculate_diversity_penalty(
    new_expression: str,
    existing_expressions: List[str],
    operator_weight: float = 0.4,
    field_weight: float = 0.6
) -> float:
    """
    Calculate diversity penalty for a new expression.
    
    Returns: 0.0 (highly diverse) to 1.0 (highly similar)
    """
    if not existing_expressions:
        return 0.0
        
    new_ngrams = extract_operator_ngrams(new_expression)
    new_fields = extract_field_set(new_expression)
    
    max_similarity = 0.0
    
    for existing in existing_expressions:
        exist_ngrams = extract_operator_ngrams(existing)
        exist_fields = extract_field_set(existing)
        
        # Operator n-gram Jaccard
        if new_ngrams or exist_ngrams:
            op_sim = len(new_ngrams & exist_ngrams) / len(new_ngrams | exist_ngrams) if (new_ngrams | exist_ngrams) else 0
        else:
            op_sim = 0.0
            
        # Field Jaccard
        if new_fields or exist_fields:
            field_sim = len(new_fields & exist_fields) / len(new_fields | exist_fields) if (new_fields | exist_fields) else 0
        else:
            field_sim = 0.0
            
        # Weighted similarity
        similarity = operator_weight * op_sim + field_weight * field_sim
        max_similarity = max(max_similarity, similarity)
        
    return max_similarity


class DiversityFilter:
    """
    Filter candidates to maintain diversity in generated alphas.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.accepted_expressions: List[str] = []
        
    def should_accept(self, expression: str) -> Tuple[bool, float]:
        """
        Check if expression is diverse enough to accept.
        
        Returns: (should_accept, similarity_score)
        """
        if not self.accepted_expressions:
            return True, 0.0
            
        similarity = calculate_diversity_penalty(expression, self.accepted_expressions[-50:])
        
        if similarity >= self.similarity_threshold:
            return False, similarity
        return True, similarity
        
    def accept(self, expression: str):
        """Add expression to accepted set"""
        self.accepted_expressions.append(expression)
        
    def reset(self):
        """Clear accepted expressions"""
        self.accepted_expressions.clear()


# =============================================================================
# P0-2: DB-Level Deduplication Helper
# =============================================================================

async def check_expression_exists_in_db(
    db_session,
    expression: str,
    region: str = "USA",
    universe: str = "TOP3000"
) -> Tuple[bool, Optional[str]]:
    """
    Check if expression already exists in database.
    
    Args:
        db_session: AsyncSession
        expression: Alpha expression to check
        region: Region filter
        universe: Universe filter
        
    Returns:
        (exists, existing_alpha_id)
    """
    from sqlalchemy import select
    from backend.models import Alpha
    from backend.alpha_semantic_validator import compute_expression_hash
    
    expr_hash = compute_expression_hash(expression)
    
    # Check by hash first (fast)
    stmt = select(Alpha.alpha_id).where(
        Alpha.expression_hash == expr_hash,
        Alpha.region == region,
        Alpha.universe == universe
    ).limit(1)
    
    result = await db_session.execute(stmt)
    existing = result.scalar_one_or_none()
    
    if existing:
        return True, existing
        
    # Fallback: exact expression match (slower but catches hash collisions)
    stmt2 = select(Alpha.alpha_id).where(
        Alpha.expression == expression.strip(),
        Alpha.region == region,
        Alpha.universe == universe
    ).limit(1)
    
    result2 = await db_session.execute(stmt2)
    existing2 = result2.scalar_one_or_none()
    
    if existing2:
        return True, existing2
        
    return False, None


async def filter_unsimulated_expressions(
    db_session,
    expressions: List[str],
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int | None = None,
    decay: int | None = None,
    neutralization: str | None = None,
    truncation: float | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Filter expressions to only those not already in database for the same settings.

    Settings matter: an expression that failed under one neutralization or decay
    may be worth retesting under another. Treating expression-only matches as
    duplicates makes the optimization chain skip exactly the sweeps it creates.
    
    Returns:
        (new_expressions, duplicate_expressions)
    """
    from sqlalchemy import select
    from backend.models import Alpha
    from backend.alpha_semantic_validator import compute_expression_hash
    
    # Compute hashes
    expr_hashes = {compute_expression_hash(e): e for e in expressions}
    
    # Query existing hashes
    stmt = select(Alpha.expression_hash).where(
        Alpha.expression_hash.in_(list(expr_hashes.keys())),
        Alpha.region == region,
        Alpha.universe == universe
    )
    if delay is not None:
        stmt = stmt.where(Alpha.delay == int(delay))
    if decay is not None:
        stmt = stmt.where(Alpha.decay == int(decay))
    if neutralization is not None:
        stmt = stmt.where(Alpha.neutralization == str(neutralization))
    if truncation is not None:
        stmt = stmt.where(Alpha.truncation == float(truncation))
    
    result = await db_session.execute(stmt)
    existing_hashes = {row[0] for row in result.fetchall()}
    
    new_expressions = []
    duplicate_expressions = []
    
    for h, expr in expr_hashes.items():
        if h in existing_hashes:
            duplicate_expressions.append(expr)
        else:
            new_expressions.append(expr)
            
    return new_expressions, duplicate_expressions
