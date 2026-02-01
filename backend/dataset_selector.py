"""
Dataset Selector Service - Enhanced with Pre-Mining Quality Evaluation

Features:
1. DatasetEvaluator: Pre-mining quality assessment
2. DatasetSelector: Bandit-based intelligent selection
3. Category-aware quality scoring
4. Historical success rate tracking

P1-fix-1: Enable adaptive dataset selection to escape ineffective datasets.

ENHANCED (v2.1) - P2-3:
- Cross-category exploration with failure-triggered rotation
- Smart dataset rotation based on historical performance
- Adaptive exploration weight based on session progress
- Integration with SmartExplorationStrategy
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, and_
from loguru import logger

from backend.selection_strategy import DatasetBandit, DatasetArm
from backend.models import DatasetMetadata, Alpha, DataField, KnowledgeEntry
from backend.config import settings


# =============================================================================
# Dataset Quality Evaluation
# =============================================================================

@dataclass
class DatasetQualityScore:
    """Comprehensive quality assessment for a dataset."""
    dataset_id: str
    region: str
    universe: str
    
    # Raw metrics
    coverage: float = 0.0          # Data coverage (0-1)
    field_count: int = 0           # Number of fields
    alpha_count: int = 0           # Historical alphas produced
    success_rate: float = 0.0      # Historical success rate
    avg_sharpe: float = 0.0        # Average Sharpe of successful alphas
    pyramid_multiplier: float = 1.0  # Platform pyramid multiplier
    
    # Derived scores (0-1)
    coverage_score: float = 0.0
    richness_score: float = 0.0    # Based on field count
    track_record_score: float = 0.0  # Based on historical success
    potential_score: float = 0.0    # Estimated potential
    
    # Final combined score
    overall_score: float = 0.0
    
    # Metadata
    category: str = "other"
    last_success_date: Optional[datetime] = None
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "region": self.region,
            "universe": self.universe,
            "coverage": self.coverage,
            "field_count": self.field_count,
            "alpha_count": self.alpha_count,
            "success_rate": self.success_rate,
            "avg_sharpe": self.avg_sharpe,
            "pyramid_multiplier": self.pyramid_multiplier,
            "coverage_score": round(self.coverage_score, 3),
            "richness_score": round(self.richness_score, 3),
            "track_record_score": round(self.track_record_score, 3),
            "potential_score": round(self.potential_score, 3),
            "overall_score": round(self.overall_score, 3),
            "category": self.category,
            "recommendation": self.recommendation,
        }


class DatasetEvaluator:
    """
    Pre-mining dataset quality evaluation.
    
    Evaluates datasets based on:
    1. Data coverage and field richness
    2. Historical alpha production success
    3. Category-based potential estimation
    4. Platform pyramid multipliers
    
    Usage:
        evaluator = DatasetEvaluator(db)
        scores = await evaluator.evaluate_datasets(region="USA", universe="TOP3000")
        top_datasets = evaluator.rank_datasets(scores, top_n=10)
    """
    
    # Category-based potential multipliers (from empirical observations)
    CATEGORY_POTENTIAL = {
        "pv": 1.0,           # Price-volume: standard, competitive
        "analyst": 1.3,      # Analyst: high alpha potential
        "fundamental": 1.2,  # Fundamental: solid, less competitive
        "news": 0.9,         # News/sentiment: noisy, harder
        "other": 0.8,        # Other: unknown, lower priority
    }
    
    # Minimum thresholds for consideration
    MIN_COVERAGE = 0.3       # 30% coverage minimum
    MIN_FIELDS = 5           # At least 5 fields
    
    def __init__(self, db: AsyncSession):
        self.db = db
        
    async def evaluate_datasets(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
        dataset_ids: Optional[List[str]] = None,
        include_zero_alpha: bool = True
    ) -> List[DatasetQualityScore]:
        """
        Evaluate all available datasets for a region/universe.
        
        Args:
            region: Market region
            universe: Universe
            dataset_ids: Optional specific datasets to evaluate
            include_zero_alpha: Include datasets with no historical alphas
            
        Returns:
            List of DatasetQualityScore objects
        """
        logger.info(f"[DatasetEvaluator] Evaluating datasets | region={region} universe={universe}")
        
        # 1. Get dataset metadata
        query = select(DatasetMetadata).where(
            DatasetMetadata.region == region,
            DatasetMetadata.universe == universe,
            DatasetMetadata.is_active == True
        )
        
        if dataset_ids:
            query = query.where(DatasetMetadata.dataset_id.in_(dataset_ids))
            
        result = await self.db.execute(query)
        datasets = result.scalars().all()
        
        if not datasets:
            logger.warning(f"[DatasetEvaluator] No datasets found for region={region}")
            return []
        
        # 2. Get historical alpha statistics
        alpha_stats = await self._get_alpha_statistics(region, universe)
        
        # 3. Evaluate each dataset
        scores = []
        for ds in datasets:
            score = await self._evaluate_single_dataset(
                dataset=ds,
                region=region,
                universe=universe,
                alpha_stats=alpha_stats
            )
            
            # Filter out low-quality datasets
            if score.coverage >= self.MIN_COVERAGE or include_zero_alpha:
                if score.field_count >= self.MIN_FIELDS or include_zero_alpha:
                    scores.append(score)
        
        logger.info(f"[DatasetEvaluator] Evaluated {len(scores)} datasets")
        return scores
    
    async def _get_alpha_statistics(
        self,
        region: str,
        universe: str
    ) -> Dict[str, Dict]:
        """Get historical alpha statistics by dataset."""
        # Query alphas in the last 90 days
        cutoff = datetime.now() - timedelta(days=90)
        
        query = select(
            Alpha.dataset_id,
            func.count(Alpha.id).label('total'),
            func.count(Alpha.id).filter(Alpha.quality_status == 'PASS').label('passed'),
            func.avg(Alpha.is_sharpe).label('avg_sharpe'),
            func.max(Alpha.created_at).label('last_alpha')
        ).where(
            Alpha.region == region,
            Alpha.created_at >= cutoff
        ).group_by(Alpha.dataset_id)
        
        result = await self.db.execute(query)
        rows = result.fetchall()
        
        stats = {}
        for row in rows:
            if row.dataset_id:
                stats[row.dataset_id] = {
                    'total': row.total or 0,
                    'passed': row.passed or 0,
                    'avg_sharpe': float(row.avg_sharpe or 0),
                    'last_alpha': row.last_alpha
                }
        
        return stats
    
    async def _evaluate_single_dataset(
        self,
        dataset: DatasetMetadata,
        region: str,
        universe: str,
        alpha_stats: Dict[str, Dict]
    ) -> DatasetQualityScore:
        """Evaluate a single dataset."""
        ds_id = dataset.dataset_id
        
        # Get historical stats
        hist = alpha_stats.get(ds_id, {'total': 0, 'passed': 0, 'avg_sharpe': 0, 'last_alpha': None})
        
        # Calculate success rate
        success_rate = hist['passed'] / hist['total'] if hist['total'] > 0 else 0.5  # Default 50%
        
        # Infer category
        category = self._infer_category(ds_id, dataset.category)
        
        # Create score object
        score = DatasetQualityScore(
            dataset_id=ds_id,
            region=region,
            universe=universe,
            coverage=dataset.coverage or dataset.date_coverage or 0.5,
            field_count=dataset.field_count or 0,
            alpha_count=hist['total'],
            success_rate=success_rate,
            avg_sharpe=hist['avg_sharpe'],
            pyramid_multiplier=dataset.pyramid_multiplier or dataset.mining_weight or 1.0,
            category=category,
            last_success_date=hist['last_alpha']
        )
        
        # Calculate component scores
        score.coverage_score = self._score_coverage(score.coverage)
        score.richness_score = self._score_richness(score.field_count)
        score.track_record_score = self._score_track_record(
            success_rate=score.success_rate,
            total_alphas=score.alpha_count,
            avg_sharpe=score.avg_sharpe
        )
        score.potential_score = self._score_potential(
            category=category,
            pyramid_multiplier=score.pyramid_multiplier,
            coverage=score.coverage
        )
        
        # Combined overall score (weighted average)
        weights = {
            'coverage': 0.15,
            'richness': 0.10,
            'track_record': 0.35,
            'potential': 0.40
        }
        
        score.overall_score = (
            weights['coverage'] * score.coverage_score +
            weights['richness'] * score.richness_score +
            weights['track_record'] * score.track_record_score +
            weights['potential'] * score.potential_score
        )
        
        # Generate recommendation
        score.recommendation = self._generate_recommendation(score)
        
        return score
    
    def _infer_category(self, dataset_id: str, db_category: Optional[str]) -> str:
        """Infer dataset category from ID or database category."""
        if db_category:
            cat_lower = db_category.lower()
            for cat in ['pv', 'analyst', 'fundamental', 'news']:
                if cat in cat_lower:
                    return cat
        
        # Infer from dataset_id
        ds_lower = dataset_id.lower()
        category_keywords = {
            'pv': ['pv', 'price', 'volume', 'trade'],
            'analyst': ['anl', 'analyst', 'estimate'],
            'fundamental': ['fnd', 'fundamental', 'fin'],
            'news': ['news', 'sentiment', 'oth', 'social']
        }
        
        for cat, keywords in category_keywords.items():
            for kw in keywords:
                if kw in ds_lower:
                    return cat
        
        return 'other'
    
    def _score_coverage(self, coverage: float) -> float:
        """Score data coverage (0-1)."""
        if coverage >= 0.9:
            return 1.0
        elif coverage >= 0.7:
            return 0.8 + (coverage - 0.7) * 1.0  # 0.8 to 1.0
        elif coverage >= 0.5:
            return 0.5 + (coverage - 0.5) * 1.5  # 0.5 to 0.8
        else:
            return max(0, coverage * 1.0)
    
    def _score_richness(self, field_count: int) -> float:
        """Score field richness (0-1)."""
        if field_count >= 100:
            return 1.0
        elif field_count >= 50:
            return 0.8 + (field_count - 50) * 0.004
        elif field_count >= 20:
            return 0.5 + (field_count - 20) * 0.01
        elif field_count >= 5:
            return 0.2 + (field_count - 5) * 0.02
        else:
            return max(0, field_count * 0.04)
    
    def _score_track_record(
        self,
        success_rate: float,
        total_alphas: int,
        avg_sharpe: float
    ) -> float:
        """Score historical track record."""
        if total_alphas == 0:
            return 0.5  # Unknown - neutral score
        
        # Base score from success rate
        rate_score = min(success_rate * 2, 1.0)  # Double the rate, cap at 1
        
        # Confidence adjustment based on sample size
        confidence = min(total_alphas / 20, 1.0)  # Full confidence at 20+ alphas
        
        # Sharpe bonus
        sharpe_bonus = min(avg_sharpe / 2.0, 0.3) if avg_sharpe > 0 else 0
        
        # Combined score
        return rate_score * 0.6 + confidence * 0.2 + (0.5 + sharpe_bonus) * 0.2
    
    def _score_potential(
        self,
        category: str,
        pyramid_multiplier: float,
        coverage: float
    ) -> float:
        """Score estimated potential based on category and platform metrics."""
        # Category multiplier
        cat_mult = self.CATEGORY_POTENTIAL.get(category, 0.8)
        
        # Pyramid multiplier bonus (higher pyramid = higher payout potential)
        pyramid_score = min(pyramid_multiplier / 2.0, 1.0)
        
        # Coverage factor
        coverage_factor = 0.5 + coverage * 0.5  # 50% base + coverage bonus
        
        return cat_mult * 0.5 + pyramid_score * 0.3 + coverage_factor * 0.2
    
    def _generate_recommendation(self, score: DatasetQualityScore) -> str:
        """Generate human-readable recommendation."""
        if score.overall_score >= 0.8:
            return "HIGHLY_RECOMMENDED"
        elif score.overall_score >= 0.6:
            return "RECOMMENDED"
        elif score.overall_score >= 0.4:
            return "WORTH_TRYING"
        elif score.overall_score >= 0.2:
            return "LOW_PRIORITY"
        else:
            return "NOT_RECOMMENDED"
    
    def rank_datasets(
        self,
        scores: List[DatasetQualityScore],
        top_n: int = 10,
        min_score: float = 0.2
    ) -> List[DatasetQualityScore]:
        """
        Rank datasets by overall score.
        
        Args:
            scores: List of quality scores
            top_n: Number of top datasets to return
            min_score: Minimum overall score threshold
            
        Returns:
            Top N datasets sorted by overall score
        """
        # Filter by minimum score
        filtered = [s for s in scores if s.overall_score >= min_score]
        
        # Sort by overall score
        filtered.sort(key=lambda x: x.overall_score, reverse=True)
        
        return filtered[:top_n]
    
    async def get_recommended_datasets(
        self,
        region: str,
        universe: str,
        top_n: int = 5
    ) -> List[str]:
        """
        Get top recommended dataset IDs for mining.
        
        Convenience method that returns just the dataset IDs.
        """
        scores = await self.evaluate_datasets(region, universe)
        ranked = self.rank_datasets(scores, top_n=top_n)
        return [s.dataset_id for s in ranked]


class DatasetSelector:
    """
    Intelligent dataset selection using Multi-Armed Bandit.
    
    Usage:
        selector = DatasetSelector(db)
        await selector.initialize(region="KOR", universe="TOP600")
        
        # Select dataset for mining
        dataset_id = await selector.select_dataset()
        
        # After mining iteration, update rewards
        await selector.update_reward(dataset_id, pass_count=2, total_count=4)
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.bandit: Optional[DatasetBandit] = None
        self.region: str = "USA"
        self.universe: str = "TOP3000"
        self._initialized = False
        
    async def initialize(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
        dataset_ids: Optional[List[str]] = None
    ):
        """
        Initialize the Bandit with available datasets.
        
        Args:
            region: Market region
            universe: Universe of stocks
            dataset_ids: Optional list of specific dataset IDs to consider
        """
        self.region = region
        self.universe = universe
        
        # Get bandit parameters from settings
        exploration_weight = getattr(settings, 'BANDIT_EXPLORATION_WEIGHT', 2.0)
        pyramid_bonus = getattr(settings, 'BANDIT_PYRAMID_BONUS_WEIGHT', 0.3)
        saturation_penalty = getattr(settings, 'BANDIT_SATURATION_PENALTY_WEIGHT', 0.2)
        
        self.bandit = DatasetBandit(
            exploration_weight=exploration_weight,
            pyramid_bonus_weight=pyramid_bonus,
            saturation_penalty_weight=saturation_penalty
        )
        
        # Load datasets from database
        await self._load_datasets(dataset_ids)
        
        # Load persisted bandit state if exists
        await self._load_bandit_state()
        
        self._initialized = True
        logger.info(f"[DatasetSelector] Initialized | region={region} arms={len(self.bandit.arms)}")
        
    async def _load_datasets(self, dataset_ids: Optional[List[str]] = None):
        """Load datasets from DB and create Bandit arms"""
        # P0 OPTIMIZATION: Import region availability filter
        from backend.agents.mining_agent import is_dataset_available_for_region
        
        query = select(DatasetMetadata).where(
            DatasetMetadata.region == self.region,
            DatasetMetadata.universe == self.universe
        )
        
        if dataset_ids:
            query = query.where(DatasetMetadata.dataset_id.in_(dataset_ids))
            
        result = await self.db.execute(query)
        datasets = result.scalars().all()
        
        filtered_count = 0
        for ds in datasets:
            # P0 OPTIMIZATION: Skip datasets known to fail in this region
            if not is_dataset_available_for_region(ds.dataset_id, self.region):
                logger.debug(f"[DatasetSelector] Skipping {ds.dataset_id} - unavailable for {self.region}")
                filtered_count += 1
                continue
                
            arm = DatasetArm(
                dataset_id=ds.dataset_id,
                region=ds.region,
                universe=self.universe,
                pyramid_multiplier=ds.mining_weight or 1.0,
                alpha_count=ds.alpha_count or 0,
                field_count=ds.field_count or 0
            )
            self.bandit.add_arm(arm)
            
        logger.info(f"[DatasetSelector] Loaded {len(datasets) - filtered_count} datasets as Bandit arms (filtered {filtered_count} unavailable)")
        
    async def _load_bandit_state(self):
        """Load persisted Bandit state from database"""
        # Try to load from KnowledgeEntry or a dedicated table
        # For now, we start fresh each session but could persist to JSON
        pass
        
    async def _save_bandit_state(self):
        """Persist Bandit state to database"""
        # Could save to a dedicated table or KnowledgeEntry
        pass
        
    async def select_dataset(self, n: int = 1) -> List[str]:
        """
        Select dataset(s) using UCB algorithm.
        
        Args:
            n: Number of datasets to select
            
        Returns:
            List of selected dataset IDs
        """
        if not self._initialized:
            raise RuntimeError("DatasetSelector not initialized. Call initialize() first.")
            
        if not self.bandit.arms:
            logger.warning("[DatasetSelector] No datasets available for selection")
            return []
            
        selected_arms = self.bandit.select(n=n)
        dataset_ids = [arm.dataset_id for arm in selected_arms]
        
        logger.info(f"[DatasetSelector] Selected datasets: {dataset_ids}")
        return dataset_ids
        
    async def update_reward(
        self,
        dataset_id: str,
        pass_count: int,
        total_count: int,
        avg_sharpe: float = 0.0
    ):
        """
        Update Bandit reward after mining iteration.
        
        Args:
            dataset_id: Dataset that was mined
            pass_count: Number of PASS alphas
            total_count: Total alphas evaluated
            avg_sharpe: Average Sharpe ratio of passes
        """
        if not self._initialized:
            return
            
        # Calculate reward: pass rate with Sharpe bonus
        pass_rate = pass_count / total_count if total_count > 0 else 0
        sharpe_bonus = min(avg_sharpe / 3.0, 0.5) if avg_sharpe > 0 else 0  # Cap at 0.5
        reward = pass_rate + sharpe_bonus * 0.3
        
        success = pass_count > 0
        
        self.bandit.update(
            dataset_id=dataset_id,
            reward=reward,
            success=success,
            region=self.region,
            universe=self.universe
        )
        
        logger.info(
            f"[DatasetSelector] Updated reward | dataset={dataset_id} "
            f"pass={pass_count}/{total_count} reward={reward:.3f}"
        )
        
        # Persist state
        await self._save_bandit_state()
        
    def get_stats(self) -> Dict:
        """Get Bandit statistics for observability"""
        if not self.bandit:
            return {}
        return self.bandit.get_stats()


# =============================================================================
# Helper function for MiningAgent integration
# =============================================================================

async def select_next_dataset(
    db: AsyncSession,
    region: str,
    universe: str,
    available_datasets: List[str],
    fallback_dataset: str
) -> str:
    """
    Helper function to select the next dataset for mining.
    
    This can be called by MiningAgent to get intelligent dataset selection.
    
    Args:
        db: Database session
        region: Market region
        universe: Universe
        available_datasets: List of available dataset IDs
        fallback_dataset: Dataset to use if selection fails
        
    Returns:
        Selected dataset ID
    """
    # Check if Bandit selection is enabled
    bandit_enabled = getattr(settings, 'BANDIT_SELECTION_ENABLED', False)
    
    if not bandit_enabled:
        logger.debug("[DatasetSelector] Bandit disabled, using fallback")
        return fallback_dataset
        
    try:
        selector = DatasetSelector(db)
        await selector.initialize(
            region=region,
            universe=universe,
            dataset_ids=available_datasets
        )
        
        selected = await selector.select_dataset(n=1)
        
        if selected:
            return selected[0]
        else:
            logger.warning("[DatasetSelector] No dataset selected, using fallback")
            return fallback_dataset
            
    except Exception as e:
        logger.error(f"[DatasetSelector] Selection failed: {e}")
        return fallback_dataset


async def update_dataset_reward(
    db: AsyncSession,
    dataset_id: str,
    region: str,
    universe: str,
    pass_count: int,
    total_count: int,
    avg_sharpe: float = 0.0
):
    """
    Helper function to update dataset reward after mining.
    
    This can be called by MiningAgent after each iteration.
    """
    bandit_enabled = getattr(settings, 'BANDIT_SELECTION_ENABLED', False)
    
    if not bandit_enabled:
        return
        
    try:
        selector = DatasetSelector(db)
        await selector.initialize(region=region, universe=universe)
        await selector.update_reward(
            dataset_id=dataset_id,
            pass_count=pass_count,
            total_count=total_count,
            avg_sharpe=avg_sharpe
        )
    except Exception as e:
        logger.warning(f"[DatasetSelector] Reward update failed: {e}")


# =============================================================================
# Helper functions for Dataset Evaluation
# =============================================================================

async def evaluate_and_select_datasets(
    db: AsyncSession,
    region: str,
    universe: str,
    top_n: int = 5,
    min_score: float = 0.3
) -> List[str]:
    """
    Evaluate datasets and return top N by quality score.
    
    This is the recommended entry point for mining agents.
    
    Args:
        db: Database session
        region: Market region
        universe: Universe
        top_n: Number of datasets to select
        min_score: Minimum quality score threshold
        
    Returns:
        List of recommended dataset IDs
    """
    try:
        evaluator = DatasetEvaluator(db)
        scores = await evaluator.evaluate_datasets(region, universe)
        ranked = evaluator.rank_datasets(scores, top_n=top_n, min_score=min_score)
        
        logger.info(
            f"[DatasetEvaluator] Top {len(ranked)} datasets for {region}/{universe}: "
            f"{[s.dataset_id for s in ranked]}"
        )
        
        return [s.dataset_id for s in ranked]
        
    except Exception as e:
        logger.error(f"[DatasetEvaluator] Evaluation failed: {e}")
        return []


async def get_dataset_quality_report(
    db: AsyncSession,
    region: str,
    universe: str
) -> Dict[str, Any]:
    """
    Generate a comprehensive dataset quality report.
    
    Useful for debugging and monitoring.
    
    Returns:
        Dictionary with evaluation results and statistics
    """
    try:
        evaluator = DatasetEvaluator(db)
        scores = await evaluator.evaluate_datasets(region, universe)
        
        # Group by recommendation
        by_recommendation = {}
        for score in scores:
            rec = score.recommendation
            if rec not in by_recommendation:
                by_recommendation[rec] = []
            by_recommendation[rec].append(score.dataset_id)
        
        # Top 10 by overall score
        ranked = evaluator.rank_datasets(scores, top_n=10)
        
        # Category distribution
        category_counts = {}
        for score in scores:
            cat = score.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "region": region,
            "universe": universe,
            "total_datasets": len(scores),
            "by_recommendation": by_recommendation,
            "top_10": [s.to_dict() for s in ranked],
            "category_distribution": category_counts,
            "avg_overall_score": sum(s.overall_score for s in scores) / len(scores) if scores else 0,
        }
        
    except Exception as e:
        logger.error(f"[DatasetEvaluator] Report generation failed: {e}")
        return {"error": str(e)}


# =============================================================================
# P2-3 ENHANCEMENT: Smart Cross-Category Dataset Selection
# =============================================================================

# Dataset category definitions
DATASET_CATEGORIES = {
    "pv": ["pv", "price", "volume", "trade"],
    "analyst": ["analyst", "anl", "estimate"],
    "fundamental": ["fundamental", "fnd", "fin"],
    "news": ["news", "sentiment", "social"],
    "other": ["oth", "other", "alt"]
}


def infer_dataset_category(dataset_id: str) -> str:
    """Infer dataset category from its ID."""
    if not dataset_id:
        return "other"
    ds_lower = dataset_id.lower()
    for category, keywords in DATASET_CATEGORIES.items():
        for kw in keywords:
            if kw in ds_lower:
                return category
    return "other"


@dataclass
class SmartDatasetSelection:
    """Result of smart dataset selection."""
    selected_dataset: str
    category: str
    reason: str
    alternatives: List[str]
    confidence: float
    should_rotate: bool
    rotation_reason: str = ""


class SmartDatasetSelector:
    """
    P2-3: Smart dataset selection with cross-category exploration.
    
    Key features:
    1. Category-aware selection to ensure diversity
    2. Failure-triggered rotation to escape local minima
    3. Historical success rate tracking per category
    4. Adaptive exploration based on session progress
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._category_attempts: Dict[str, int] = {}
        self._category_successes: Dict[str, int] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._last_category: Optional[str] = None
        
    async def select_smart(
        self,
        region: str,
        universe: str,
        available_datasets: List[str],
        current_dataset: Optional[str] = None,
        consecutive_failures: int = 0,
        force_cross_category: bool = False
    ) -> SmartDatasetSelection:
        """
        Smart dataset selection with cross-category exploration.
        
        Args:
            region: Market region
            universe: Universe
            available_datasets: List of available dataset IDs
            current_dataset: Currently used dataset (if any)
            consecutive_failures: Number of consecutive failures on current dataset
            force_cross_category: Force selection from different category
            
        Returns:
            SmartDatasetSelection with selected dataset and metadata
        """
        if not available_datasets:
            return SmartDatasetSelection(
                selected_dataset="",
                category="",
                reason="No datasets available",
                alternatives=[],
                confidence=0.0,
                should_rotate=False
            )
        
        # Group datasets by category
        category_datasets: Dict[str, List[str]] = {}
        for ds in available_datasets:
            cat = infer_dataset_category(ds)
            if cat not in category_datasets:
                category_datasets[cat] = []
            category_datasets[cat].append(ds)
        
        # Determine if rotation is needed
        should_rotate = False
        rotation_reason = ""
        
        current_category = infer_dataset_category(current_dataset) if current_dataset else None
        
        if force_cross_category:
            should_rotate = True
            rotation_reason = "Forced cross-category exploration"
        elif consecutive_failures >= 3:
            should_rotate = True
            rotation_reason = f"High failure count ({consecutive_failures})"
        elif current_category and self._category_successes.get(current_category, 0) == 0:
            attempts = self._category_attempts.get(current_category, 0)
            if attempts >= 5:
                should_rotate = True
                rotation_reason = f"Category {current_category} has 0 successes in {attempts} attempts"
        
        # Select category
        if should_rotate and current_category:
            # Exclude current category
            available_categories = [c for c in category_datasets.keys() if c != current_category]
            if not available_categories:
                available_categories = list(category_datasets.keys())
        else:
            available_categories = list(category_datasets.keys())
        
        # Score categories
        category_scores = {}
        for cat in available_categories:
            attempts = self._category_attempts.get(cat, 0)
            successes = self._category_successes.get(cat, 0)
            
            if attempts == 0:
                # Unexplored - give exploration bonus
                score = 0.7
            else:
                success_rate = successes / attempts
                # UCB-style scoring
                exploration_bonus = 1.0 / (1 + attempts * 0.1)
                score = success_rate * 0.6 + exploration_bonus * 0.4
            
            # Category priority bonus (analyst > fundamental > pv > news > other)
            priority_bonus = {
                "analyst": 0.15,
                "fundamental": 0.10,
                "pv": 0.05,
                "news": 0.0,
                "other": -0.05
            }.get(cat, 0.0)
            
            category_scores[cat] = score + priority_bonus
        
        # Select best category
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            best_category = available_categories[0] if available_categories else "other"
        
        # Select dataset from best category
        category_ds = category_datasets.get(best_category, [])
        
        if not category_ds:
            # Fallback
            selected = available_datasets[0]
            best_category = infer_dataset_category(selected)
        else:
            # Score datasets within category using evaluator if possible
            try:
                evaluator = DatasetEvaluator(self.db)
                scores = await evaluator.evaluate_datasets(
                    region=region,
                    universe=universe,
                    dataset_ids=category_ds
                )
                
                if scores:
                    ranked = evaluator.rank_datasets(scores, top_n=1)
                    selected = ranked[0].dataset_id if ranked else category_ds[0]
                else:
                    selected = category_ds[0]
            except Exception:
                selected = category_ds[0]
        
        # Get alternatives from other categories
        alternatives = []
        for cat, ds_list in category_datasets.items():
            if cat != best_category and ds_list:
                alternatives.append(ds_list[0])
        
        # Calculate confidence
        confidence = category_scores.get(best_category, 0.5)
        
        return SmartDatasetSelection(
            selected_dataset=selected,
            category=best_category,
            reason=f"Selected from {best_category} category (score={confidence:.2f})",
            alternatives=alternatives[:3],
            confidence=confidence,
            should_rotate=should_rotate,
            rotation_reason=rotation_reason
        )
    
    def update_from_result(
        self,
        dataset_id: str,
        passed_count: int,
        failed_count: int
    ):
        """Update category statistics from mining result."""
        category = infer_dataset_category(dataset_id)
        
        self._category_attempts[category] = self._category_attempts.get(category, 0) + 1
        
        if passed_count > 0:
            self._category_successes[category] = self._category_successes.get(category, 0) + 1
            self._consecutive_failures[category] = 0
        else:
            self._consecutive_failures[category] = self._consecutive_failures.get(category, 0) + 1
        
        self._last_category = category


async def select_dataset_smart(
    db: AsyncSession,
    region: str,
    universe: str,
    available_datasets: List[str],
    current_dataset: Optional[str] = None,
    consecutive_failures: int = 0,
    force_cross_category: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    P2-3: Smart dataset selection with cross-category exploration.
    
    Entry point for MiningAgent integration.
    
    Returns:
        (selected_dataset_id, selection_metadata)
    """
    try:
        selector = SmartDatasetSelector(db)
        result = await selector.select_smart(
            region=region,
            universe=universe,
            available_datasets=available_datasets,
            current_dataset=current_dataset,
            consecutive_failures=consecutive_failures,
            force_cross_category=force_cross_category
        )
        
        logger.info(
            f"[SmartSelector] Selected: {result.selected_dataset} "
            f"(category={result.category}, confidence={result.confidence:.2f}) "
            f"| rotation={result.should_rotate}: {result.rotation_reason}"
        )
        
        return result.selected_dataset, {
            "category": result.category,
            "reason": result.reason,
            "alternatives": result.alternatives,
            "confidence": result.confidence,
            "should_rotate": result.should_rotate,
            "rotation_reason": result.rotation_reason,
        }
        
    except Exception as e:
        logger.error(f"[SmartSelector] Selection failed: {e}")
        fallback = available_datasets[0] if available_datasets else ""
        return fallback, {"error": str(e)}
