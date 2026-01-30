"""
Feedback Agent - Enhanced with Structured Analysis and Pattern Promotion

Features:
1. Structured failure analysis with root cause identification
2. Success pattern automatic promotion and boosting
3. Knowledge base quality scoring and pruning
4. Evolutionary strategy recommendations

Implements the CoSTEER (Collaborative Evolving Strategy) feedback loop
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, update

from backend.models import AlphaFailure, KnowledgeEntry, Alpha
from backend.agents.prompts import FAILURE_ANALYSIS_SYSTEM, FAILURE_ANALYSIS_USER
from backend.config import settings
from backend.agents.services.llm_service import LLMService, get_llm_service
from backend.protocols import LLMProtocol

from loguru import logger


# =============================================================================
# Structured Failure Analysis
# =============================================================================

@dataclass
class FailureAnalysis:
    """Structured analysis of alpha failures."""
    # Root cause categories
    category: str  # SYNTAX, SEMANTIC, LOW_SHARPE, HIGH_TURNOVER, OVERFITTING, etc.
    severity: str  # high, medium, low
    
    # Details
    error_pattern: str
    root_cause: str
    recommendation: str
    
    # Metrics if available
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    
    # Context
    dataset_id: str = ""
    region: str = ""
    operators_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "error_pattern": self.error_pattern,
            "root_cause": self.root_cause,
            "recommendation": self.recommendation,
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover": self.turnover,
            "dataset_id": self.dataset_id,
            "region": self.region,
            "operators_used": self.operators_used,
        }


@dataclass 
class PatternScore:
    """Score for a knowledge base pattern."""
    pattern_id: int
    pattern: str
    entry_type: str
    
    # Usage metrics
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    # Derived scores
    success_rate: float = 0.0
    recency_score: float = 0.0  # Based on last use
    confidence_score: float = 0.0  # Based on sample size
    overall_score: float = 0.0
    
    # Timestamps
    created_at: datetime = None
    last_used: datetime = None
    
    def calculate_scores(self):
        """Calculate derived scores."""
        total = self.success_count + self.failure_count
        self.success_rate = self.success_count / total if total > 0 else 0.5
        
        # Confidence: higher with more samples
        self.confidence_score = min(1.0, total / 10.0)
        
        # Recency: decay over time
        if self.last_used:
            days_since = (datetime.now() - self.last_used).days
            self.recency_score = max(0, 1.0 - days_since / 60)  # 60-day decay
        else:
            self.recency_score = 0.5
        
        # Overall score
        self.overall_score = (
            0.50 * self.success_rate +
            0.30 * self.confidence_score +
            0.20 * self.recency_score
        )


# Failure category classification rules
FAILURE_CATEGORIES = {
    "SYNTAX_ERROR": {
        "keywords": ["syntax", "parse", "invalid", "unexpected token"],
        "severity": "high",
        "recommendation": "Check expression syntax and operator usage"
    },
    "SEMANTIC_ERROR": {
        "keywords": ["type error", "vector", "matrix", "incompatible"],
        "severity": "high", 
        "recommendation": "Verify field types (VECTOR vs MATRIX) and operator compatibility"
    },
    "FIELD_NOT_FOUND": {
        "keywords": ["field not found", "unknown field", "does not exist"],
        "severity": "high",
        "recommendation": "Verify field exists in dataset for the specified region/universe"
    },
    "LOW_SHARPE": {
        "keywords": ["low sharpe", "sharpe below", "insufficient sharpe"],
        "severity": "medium",
        "recommendation": "Add smoothing (ts_decay_linear), try different fields, or adjust windows"
    },
    "HIGH_TURNOVER": {
        "keywords": ["turnover", "high turnover"],
        "severity": "medium",
        "recommendation": "Add decay, increase lookback windows, or use ts_rank instead of rank"
    },
    "LOW_FITNESS": {
        "keywords": ["fitness", "low fitness"],
        "severity": "medium",
        "recommendation": "Try different neutralization or add risk controls"
    },
    "OVERFITTING": {
        "keywords": ["overfit", "is/os gap", "oos"],
        "severity": "medium",
        "recommendation": "Simplify expression, add smoothing, or reduce complexity"
    },
    "HIGH_CORRELATION": {
        "keywords": ["correlation", "similar alpha", "duplicate"],
        "severity": "low",
        "recommendation": "Try different operators or fields for more novelty"
    },
    "TIMEOUT": {
        "keywords": ["timeout", "time out", "too slow"],
        "severity": "medium",
        "recommendation": "Simplify expression or reduce computational complexity"
    },
    "UNKNOWN": {
        "keywords": [],
        "severity": "medium",
        "recommendation": "Review error message and expression carefully"
    }
}


def classify_failure(error_type: str, error_message: str, metrics: Dict = None) -> FailureAnalysis:
    """
    Classify a failure into structured categories.
    
    Args:
        error_type: Original error type string
        error_message: Error message text
        metrics: Optional metrics dict
    
    Returns:
        FailureAnalysis with classification
    """
    error_lower = (error_type or "").lower() + " " + (error_message or "").lower()
    
    # Try to match category
    matched_category = "UNKNOWN"
    for cat, config in FAILURE_CATEGORIES.items():
        for keyword in config["keywords"]:
            if keyword in error_lower:
                matched_category = cat
                break
        if matched_category != "UNKNOWN":
            break
    
    # Check metrics for additional classification
    if metrics:
        sharpe = metrics.get("sharpe", 0)
        turnover = metrics.get("turnover", 0)
        fitness = metrics.get("fitness", 0)
        
        if sharpe < 0.3 and matched_category == "UNKNOWN":
            matched_category = "LOW_SHARPE"
        elif turnover > 0.7 and matched_category == "UNKNOWN":
            matched_category = "HIGH_TURNOVER"
        elif fitness < 0.5 and matched_category == "UNKNOWN":
            matched_category = "LOW_FITNESS"
    
    config = FAILURE_CATEGORIES[matched_category]
    
    return FailureAnalysis(
        category=matched_category,
        severity=config["severity"],
        error_pattern=error_type or "unknown",
        root_cause=error_message[:200] if error_message else "No message",
        recommendation=config["recommendation"],
        sharpe=metrics.get("sharpe", 0) if metrics else 0,
        fitness=metrics.get("fitness", 0) if metrics else 0,
        turnover=metrics.get("turnover", 0) if metrics else 0,
    )


class FeedbackAgent:
    """
    Feedback Agent - Responsible for:
    1. Analyzing failure patterns
    2. Updating the knowledge base
    3. Generating prompt improvements
    
    Uses dependency injection for the LLM service to enable testing.
    """
    
    def __init__(self, db: AsyncSession, llm_service: LLMProtocol = None):
        """
        Initialize FeedbackAgent.
        
        Args:
            db: Async database session
            llm_service: Optional LLM service (uses singleton if not provided)
        """
        self.db = db
        self._llm_service = llm_service
    
    @property
    def llm_service(self) -> LLMService:
        """Get LLM service (lazy initialization)."""
        if self._llm_service is None:
            self._llm_service = get_llm_service()
        return self._llm_service
    
    async def run_daily_feedback(self) -> Dict:
        """
        Run the daily feedback loop:
        1. Collect today's failures
        2. Analyze patterns
        3. Update knowledge base
        """
        logger.info("Starting daily feedback analysis...")
        
        # Get today's failures
        today = datetime.now().date()
        start_of_day = datetime.combine(today, datetime.min.time())
        
        failures_query = select(AlphaFailure).where(
            AlphaFailure.created_at >= start_of_day,
            AlphaFailure.is_analyzed == False
        )
        result = await self.db.execute(failures_query)
        failures = result.scalars().all()
        
        if not failures:
            logger.info("No new failures to analyze")
            return {"status": "no_failures", "analyzed": 0}
        
        logger.info(f"Analyzing {len(failures)} failures...")
        
        # Group by error type
        error_distribution = Counter(f.error_type for f in failures)
        
        # Get sample failures for each type
        sample_failures = []
        for error_type in error_distribution.keys():
            samples = [f for f in failures if f.error_type == error_type][:3]
            for s in samples:
                sample_failures.append({
                    'expression': s.expression[:200] if s.expression else '',
                    'error_type': s.error_type,
                    'error_message': s.error_message[:200] if s.error_message else ''
                })
        
        # Use LLM to analyze patterns
        analysis = await self._analyze_with_llm(
            count=len(failures),
            error_distribution=dict(error_distribution),
            sample_failures=sample_failures
        )
        
        # Update knowledge base with new pitfalls
        new_entries = 0
        if analysis.get('patterns'):
            for pattern in analysis['patterns']:
                # Check if similar pattern already exists
                exists = await self._pattern_exists(pattern['pattern'])
                if not exists:
                    entry = KnowledgeEntry(
                        entry_type='FAILURE_PITFALL',
                        pattern=pattern['pattern'],
                        description=pattern.get('recommendation', ''),
                        meta_data={
                            'frequency': pattern.get('frequency', 0),
                            'source': 'feedback_agent',
                            'date': today.isoformat()
                        },
                        created_by='SYSTEM'
                    )
                    self.db.add(entry)
                    new_entries += 1
        
        # Mark failures as analyzed
        for failure in failures:
            failure.is_analyzed = True
        
        await self.db.commit()
        
        logger.info(f"Feedback analysis complete: {new_entries} new knowledge entries")
        
        return {
            "status": "success",
            "analyzed": len(failures),
            "new_entries": new_entries,
            "patterns": analysis.get('patterns', []),
            "improvements": analysis.get('prompt_improvements', [])
        }
    
    async def learn_from_success(self, alpha: Alpha) -> Optional[Dict]:
        """
        Learn from a successful alpha (especially if liked by human).
        Extract patterns for knowledge base.
        """
        if not alpha.expression:
            return None
        
        # Extract pattern from the alpha
        operators = alpha.operators_used or []
        pattern_parts = []
        
        # Build pattern description
        for op in operators[:3]:  # Top 3 operators
            pattern_parts.append(op)
        
        if not pattern_parts:
            return None
        
        pattern = " + ".join(pattern_parts)
        
        # Check if similar pattern exists
        exists = await self._pattern_exists(pattern)
        if exists:
            # Update usage count
            await self._increment_pattern_usage(pattern)
            return {"action": "incremented", "pattern": pattern}
        
        # Create new success pattern
        entry = KnowledgeEntry(
            entry_type='SUCCESS_PATTERN',
            pattern=pattern,
            description=alpha.hypothesis or alpha.logic_explanation or '',
            meta_data={
                'sharpe': alpha.metrics.get('sharpe') if alpha.metrics else None,
                'dataset': alpha.dataset_id,
                'region': alpha.region,
                'human_feedback': alpha.human_feedback
            },
            created_by='SYSTEM'
        )
        self.db.add(entry)
        await self.db.commit()
        
        logger.info(f"New success pattern learned: {pattern}")
        return {"action": "created", "pattern": pattern}
    
    async def update_operator_stats(self) -> Dict:
        """
        Update operator usage and failure statistics.
        """
        # Get all alphas from last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        alphas_query = select(Alpha).where(
            Alpha.created_at >= thirty_days_ago
        )
        result = await self.db.execute(alphas_query)
        alphas = result.scalars().all()
        
        # Count operator usage and success
        operator_stats = {}
        for alpha in alphas:
            operators = alpha.operators_used or []
            is_success = alpha.quality_status == 'PASS'
            
            for op in operators:
                if op not in operator_stats:
                    operator_stats[op] = {'usage': 0, 'success': 0}
                operator_stats[op]['usage'] += 1
                if is_success:
                    operator_stats[op]['success'] += 1
        
        # Update operator_prefs table
        from backend.models import OperatorPreference
        
        for op_name, stats in operator_stats.items():
            failure_rate = 1 - (stats['success'] / stats['usage']) if stats['usage'] > 0 else 0
            
            # Check if exists
            query = select(OperatorPreference).where(
                OperatorPreference.operator_name == op_name
            )
            result = await self.db.execute(query)
            existing = result.scalar_one_or_none()
            
            if existing:
                existing.usage_count = stats['usage']
                existing.success_count = stats['success']
                existing.failure_rate = failure_rate
                # Auto-ban if failure rate > 80%
                if failure_rate > 0.8:
                    existing.status = 'BANNED'
            else:
                pref = OperatorPreference(
                    operator_name=op_name,
                    usage_count=stats['usage'],
                    success_count=stats['success'],
                    failure_rate=failure_rate,
                    status='ACTIVE' if failure_rate <= 0.8 else 'BANNED'
                )
                self.db.add(pref)
        
        await self.db.commit()
        return operator_stats
    
    async def _analyze_with_llm(
        self, count: int, error_distribution: Dict, sample_failures: List[Dict]
    ) -> Dict:
        """Use LLM to analyze failure patterns."""
        try:
            prompt = FAILURE_ANALYSIS_USER.format(
                count=count,
                error_distribution=json.dumps(error_distribution, indent=2),
                sample_failures=json.dumps(sample_failures, indent=2, ensure_ascii=False)
            )
            
            response = await self.llm_service.call(
                system_prompt=FAILURE_ANALYSIS_SYSTEM,
                user_prompt=prompt,
                temperature=0.5,
                json_mode=True
            )
            
            if response.success and response.parsed:
                return response.parsed
            elif response.success:
                return json.loads(self._clean_json(response.content))
            else:
                logger.error(f"LLM analysis failed: {response.error}")
                return {"patterns": [], "prompt_improvements": []}
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"patterns": [], "prompt_improvements": []}
    
    async def _pattern_exists(self, pattern: str) -> bool:
        """Check if a pattern already exists in knowledge base."""
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.pattern == pattern,
            KnowledgeEntry.is_active == True
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none() is not None
    
    async def _increment_pattern_usage(self, pattern: str):
        """Increment usage count for existing pattern."""
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.pattern == pattern
        )
        result = await self.db.execute(query)
        entry = result.scalar_one_or_none()
        if entry:
            entry.usage_count += 1
    
    # =========================================================================
    # Enhanced Pattern Promotion and Quality Management
    # =========================================================================
    
    async def boost_success_pattern(
        self,
        pattern: str,
        sharpe: float,
        region: str = None,
        dataset_id: str = None
    ) -> bool:
        """
        Boost a success pattern's score and visibility.
        
        Called when a pattern leads to successful alpha.
        """
        try:
            query = select(KnowledgeEntry).where(
                KnowledgeEntry.pattern == pattern,
                KnowledgeEntry.entry_type == 'SUCCESS_PATTERN',
                KnowledgeEntry.is_active == True
            )
            result = await self.db.execute(query)
            entry = result.scalar_one_or_none()
            
            if entry:
                # Update success metrics
                entry.usage_count = (entry.usage_count or 0) + 1
                
                meta = entry.meta_data or {}
                meta['success_count'] = meta.get('success_count', 0) + 1
                meta['last_success'] = datetime.now().isoformat()
                
                # Update running average Sharpe
                n = meta.get('success_count', 1)
                old_sharpe = meta.get('avg_sharpe', 0)
                meta['avg_sharpe'] = (old_sharpe * (n - 1) + sharpe) / n
                
                # Boost score
                current_score = meta.get('score', 0.5)
                boost = min(0.1, sharpe / 20)  # Cap boost at 0.1
                meta['score'] = min(1.0, current_score + boost)
                
                entry.meta_data = meta
                await self.db.commit()
                
                logger.info(f"[Feedback] Boosted pattern score | pattern={pattern[:50]} new_score={meta['score']:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[Feedback] Boost pattern failed: {e}")
            return False
    
    async def penalize_failure_pattern(
        self,
        pattern: str,
        error_type: str,
        severity: str = "medium"
    ) -> bool:
        """
        Penalize a pattern that led to failure.
        
        Multiple failures may lead to deactivation.
        """
        try:
            query = select(KnowledgeEntry).where(
                KnowledgeEntry.pattern == pattern,
                KnowledgeEntry.is_active == True
            )
            result = await self.db.execute(query)
            entry = result.scalar_one_or_none()
            
            if entry:
                meta = entry.meta_data or {}
                meta['failure_count'] = meta.get('failure_count', 0) + 1
                meta['last_failure'] = datetime.now().isoformat()
                meta['last_error_type'] = error_type
                
                # Reduce score
                current_score = meta.get('score', 0.5)
                penalty = 0.05 if severity == "low" else 0.1 if severity == "medium" else 0.15
                meta['score'] = max(0.1, current_score - penalty)
                
                # Deactivate if too many failures
                if meta.get('failure_count', 0) > 5 and meta.get('success_count', 0) < 2:
                    entry.is_active = False
                    logger.warning(f"[Feedback] Deactivated pattern due to repeated failures | pattern={pattern[:50]}")
                
                entry.meta_data = meta
                await self.db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[Feedback] Penalize pattern failed: {e}")
            return False
    
    async def prune_low_quality_patterns(
        self,
        min_score: float = 0.2,
        max_age_days: int = 90,
        min_usage: int = 0
    ) -> int:
        """
        Prune low-quality patterns from knowledge base.
        
        Deactivates patterns that are:
        - Below minimum score threshold
        - Older than max age with no usage
        - Have high failure rates
        
        Returns:
            Number of patterns deactivated
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        deactivated = 0
        
        try:
            # Get all active patterns
            query = select(KnowledgeEntry).where(
                KnowledgeEntry.is_active == True
            )
            result = await self.db.execute(query)
            entries = result.scalars().all()
            
            for entry in entries:
                should_deactivate = False
                reason = ""
                
                meta = entry.meta_data or {}
                score = meta.get('score', 0.5)
                success_count = meta.get('success_count', 0)
                failure_count = meta.get('failure_count', 0)
                total = success_count + failure_count
                
                # Check score
                if score < min_score and total > 3:
                    should_deactivate = True
                    reason = f"low_score ({score:.2f})"
                
                # Check age and usage
                if entry.created_at and entry.created_at < cutoff_date:
                    if entry.usage_count < min_usage:
                        should_deactivate = True
                        reason = f"old_unused (age>{max_age_days}d, usage={entry.usage_count})"
                
                # Check failure rate
                if total > 5:
                    success_rate = success_count / total
                    if success_rate < 0.1:  # < 10% success
                        should_deactivate = True
                        reason = f"high_failure_rate ({success_rate:.1%})"
                
                if should_deactivate:
                    entry.is_active = False
                    deactivated += 1
                    logger.debug(f"[Feedback] Pruned pattern | reason={reason} pattern={entry.pattern[:40]}")
            
            await self.db.commit()
            
            if deactivated > 0:
                logger.info(f"[Feedback] Pruned {deactivated} low-quality patterns")
            
            return deactivated
            
        except Exception as e:
            logger.error(f"[Feedback] Prune patterns failed: {e}")
            return 0
    
    async def get_pattern_quality_report(self) -> Dict[str, Any]:
        """
        Generate a quality report for the knowledge base.
        
        Returns stats on pattern quality, usage, and recommendations.
        """
        try:
            query = select(KnowledgeEntry).where(KnowledgeEntry.is_active == True)
            result = await self.db.execute(query)
            entries = result.scalars().all()
            
            # Categorize patterns
            success_patterns = []
            failure_patterns = []
            
            for entry in entries:
                meta = entry.meta_data or {}
                
                ps = PatternScore(
                    pattern_id=entry.id,
                    pattern=entry.pattern,
                    entry_type=entry.entry_type,
                    usage_count=entry.usage_count or 0,
                    success_count=meta.get('success_count', 0),
                    failure_count=meta.get('failure_count', 0),
                    created_at=entry.created_at,
                )
                ps.calculate_scores()
                
                if entry.entry_type == 'SUCCESS_PATTERN':
                    success_patterns.append(ps)
                elif entry.entry_type == 'FAILURE_PITFALL':
                    failure_patterns.append(ps)
            
            # Sort by overall score
            success_patterns.sort(key=lambda x: x.overall_score, reverse=True)
            failure_patterns.sort(key=lambda x: x.usage_count, reverse=True)
            
            # Calculate stats
            avg_success_score = sum(p.overall_score for p in success_patterns) / len(success_patterns) if success_patterns else 0
            avg_success_rate = sum(p.success_rate for p in success_patterns) / len(success_patterns) if success_patterns else 0
            
            return {
                "total_patterns": len(entries),
                "success_patterns": len(success_patterns),
                "failure_patterns": len(failure_patterns),
                "avg_success_score": round(avg_success_score, 3),
                "avg_success_rate": round(avg_success_rate, 3),
                "top_success_patterns": [
                    {"pattern": p.pattern[:60], "score": round(p.overall_score, 3)}
                    for p in success_patterns[:10]
                ],
                "most_used_pitfalls": [
                    {"pattern": p.pattern[:60], "usage": p.usage_count}
                    for p in failure_patterns[:10]
                ],
            }
            
        except Exception as e:
            logger.error(f"[Feedback] Quality report failed: {e}")
            return {"error": str(e)}
    
    async def analyze_failure_structured(
        self,
        expression: str,
        error_type: str,
        error_message: str,
        metrics: Dict = None,
        dataset_id: str = None,
        region: str = None
    ) -> FailureAnalysis:
        """
        Perform structured failure analysis.
        
        Returns classified failure with recommendations.
        """
        analysis = classify_failure(error_type, error_message, metrics)
        analysis.dataset_id = dataset_id or ""
        analysis.region = region or ""
        
        # Extract operators from expression
        import re
        func_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        operators = [m.group(1).lower() for m in func_pattern.finditer(expression or "")]
        analysis.operators_used = operators[:10]
        
        logger.debug(
            f"[Feedback] Failure analysis | category={analysis.category} "
            f"severity={analysis.severity} operators={operators[:5]}"
        )
        
        return analysis
    
    def _clean_json(self, content: str) -> str:
        """Clean JSON response from LLM."""
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        return content.strip()
        
    async def learn_from_round(
        self,
        successes: List[Alpha],
        failures: List[Dict],
        iteration: int,
        dataset_id: str,
        region: str,
        cumulative_success: int = 0,
        target_goal: int = 4,
        max_iterations: int = 10,
    ) -> Dict:
        """
        Learn from a complete mining round (Successes & Failures).
        This enables evolutionary improvement between iterations.
        
        Args:
            successes: List of passed Alpha objects
            failures: List of failure dicts (from Workflow result)
            iteration: Current iteration index
            dataset_id: Context
            region: Context
            cumulative_success: Total successful alphas so far
            target_goal: Target number of alphas
            max_iterations: Max iterations allowed
            
        Returns:
            Dict with learned stats
        """
        from backend.agents.prompts import ROUND_ANALYSIS_SYSTEM, ROUND_ANALYSIS_USER
        
        # Skip if too little data to learn
        if not successes and not failures:
            return {"status": "skipped", "reason": "no_data"}
            
        logger.info(f"[Feedback] Learning from Round {iteration} (Success={len(successes)}, Fail={len(failures)})")
        
        # Prepare examples for LLM
        success_examples = "\n".join([
            f"- Expr: {a.expression}\n  Logic: {a.logic_explanation}\n  Sharpe: {a.metrics.get('sharpe', 'N/A')}"
            for a in successes[:5]
        ]) or "None"
        
        failure_examples = "\n".join([
            f"- Expr: {f.get('expression', 'N/A')[:100]}...\n  Error: {f.get('error_message', 'N/A')[:150]}"
            for f in failures[:5]
        ]) or "None"
        
        # Build metrics summary
        metrics_summary = f"- Pass: {len(successes)}, Fail: {len(failures)}, Total: {len(successes) + len(failures)}"
        if successes:
            avg_sharpe = sum(a.metrics.get('sharpe', 0) or 0 for a in successes) / len(successes)
            metrics_summary += f"\n- Avg Sharpe: {avg_sharpe:.3f}"
        
        remaining_rounds = max_iterations - iteration
        
        try:
            # Call LLM for analysis
            prompt = ROUND_ANALYSIS_USER.format(
                iteration=iteration,
                cumulative_success=cumulative_success,
                target_goal=target_goal,
                remaining_rounds=remaining_rounds,
                metrics_summary=metrics_summary,
                success_count=len(successes),
                success_examples=success_examples,
                failure_count=len(failures),
                failure_examples=failure_examples,
                dataset_id=dataset_id,
                region=region
            )
            
            response = await self.llm_service.call(
                system_prompt=ROUND_ANALYSIS_SYSTEM,
                user_prompt=prompt,
                temperature=0.7,
                json_mode=True
            )
            
            if response.success and response.parsed:
                analysis = response.parsed
            elif response.success:
                analysis = json.loads(self._clean_json(response.content))
            else:
                logger.error(f"Round analysis LLM call failed: {response.error}")
                return {"success": False, "error": response.error}
            
            # Store learned knowledge with enhanced structure
            new_entries = 0
            
            # 1. Store New Patterns (with templates and variants)
            for p in analysis.get("new_patterns", []):
                pattern_str = p.get("pattern", "")
                if pattern_str and not await self._pattern_exists(pattern_str):
                    entry = KnowledgeEntry(
                        entry_type='SUCCESS_PATTERN',
                        pattern=pattern_str,
                        description=p.get("description"),
                        meta_data={
                            'round': iteration,
                            'dataset_id': dataset_id,
                            'region': region,
                            'score': p.get("score"),
                            'template': p.get("template"),  # Generalized template
                            'economic_logic': p.get("economic_logic"),
                            'variants': p.get("variants", []),
                            'source': 'evolution_loop'
                        }
                    )
                    self.db.add(entry)
                    new_entries += 1

            # 2. Store New Pitfalls (with error type and severity)
            for p in analysis.get("new_pitfalls", []):
                pattern_str = p.get("pattern", "")
                if pattern_str and not await self._pattern_exists(pattern_str):
                    entry = KnowledgeEntry(
                        entry_type='FAILURE_PITFALL',
                        pattern=pattern_str,
                        description=p.get("description"),
                        meta_data={
                            'round': iteration,
                            'dataset_id': dataset_id,
                            'region': region,
                            'error_type': p.get("error_type"),
                            'recommendation': p.get("recommendation"),
                            'severity': p.get("severity", "medium"),
                            'source': 'evolution_loop'
                        }
                    )
                    self.db.add(entry)
                    new_entries += 1
            
            # 3. Store Field Insights (for RAG retrieval)
            field_insights = analysis.get("field_insights", {})
            if field_insights:
                # Store effective fields
                for field in field_insights.get("effective_fields", []):
                    field_pattern = f"FIELD_EFFECTIVE:{field}"
                    if not await self._pattern_exists(field_pattern):
                        entry = KnowledgeEntry(
                            entry_type='FIELD_INSIGHT',
                            pattern=field_pattern,
                            description=f"字段 {field} 在 {dataset_id} 数据集中表现有效",
                            meta_data={
                                'round': iteration,
                                'dataset_id': dataset_id,
                                'region': region,
                                'insight_type': 'effective',
                                'source': 'evolution_loop'
                            }
                        )
                        self.db.add(entry)
                        new_entries += 1
                
                # Store problematic fields
                for field in field_insights.get("problematic_fields", []):
                    field_pattern = f"FIELD_PROBLEMATIC:{field}"
                    if not await self._pattern_exists(field_pattern):
                        entry = KnowledgeEntry(
                            entry_type='FIELD_INSIGHT',
                            pattern=field_pattern,
                            description=f"字段 {field} 在 {dataset_id} 数据集中存在问题，建议避免",
                            meta_data={
                                'round': iteration,
                                'dataset_id': dataset_id,
                                'region': region,
                                'insight_type': 'problematic',
                                'source': 'evolution_loop'
                            }
                        )
                        self.db.add(entry)
                        new_entries += 1
            
            # 4. Store Hypothesis Evolution Insights
            hypothesis_evo = analysis.get("hypothesis_evolution", {})
            if hypothesis_evo:
                # Store promising directions
                for direction in hypothesis_evo.get("promising_directions", []):
                    hypo_pattern = f"HYPOTHESIS_PROMISING:{direction[:50]}"
                    if not await self._pattern_exists(hypo_pattern):
                        entry = KnowledgeEntry(
                            entry_type='HYPOTHESIS_INSIGHT',
                            pattern=hypo_pattern,
                            description=f"投资假设方向值得继续探索: {direction}",
                            meta_data={
                                'round': iteration,
                                'dataset_id': dataset_id,
                                'region': region,
                                'direction_type': 'promising',
                                'source': 'evolution_loop'
                            }
                        )
                        self.db.add(entry)
                        new_entries += 1
            
            await self.db.commit()
            logger.info(f"[Feedback] Round learning complete. Added {new_entries} knowledge entries (patterns, pitfalls, field/hypothesis insights).")
            
            return {
                "new_entries": new_entries,
                "patterns_added": len(analysis.get("new_patterns", [])),
                "pitfalls_added": len(analysis.get("new_pitfalls", [])),
                "field_insights": field_insights,
                "hypothesis_evolution": hypothesis_evo,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"[Feedback] Learn from round failed: {e}")
            return {"error": str(e)}
    
    def extract_injectable_feedback(
        self,
        alphas: List[Alpha],
        failures: List[Dict],
        max_items: int = 5
    ) -> List[Dict]:
        """
        Extract structured feedback that can be injected into the next generation prompt.
        
        This is the KEY mechanism for CoSTEER-style feedback injection:
        - Previous attempts are formatted for direct inclusion in the prompt
        - Includes both successes (for reinforcement) and failures (for learning)
        - Provides actionable insights not just raw data
        
        Args:
            alphas: Alpha objects from this round
            failures: Failure records from this round
            max_items: Maximum number of feedback items to return
        
        Returns:
            List of feedback dicts for prompt injection
        """
        feedback_items = []
        
        # 1. Process successes (if any)
        for alpha in alphas[:max_items // 2]:
            metrics = getattr(alpha, 'metrics', {}) or {}
            status = getattr(alpha, 'quality_status', 'UNKNOWN')
            
            if status == 'PASS':
                feedback_items.append({
                    'expression': alpha.expression[:150],
                    'result': 'SUCCESS',
                    'sharpe': metrics.get('sharpe', 'N/A'),
                    'fitness': metrics.get('fitness', 'N/A'),
                    'turnover': metrics.get('turnover', 'N/A'),
                    'issue': None,
                    'lesson': f"This pattern worked: {alpha.hypothesis or 'hypothesis not recorded'}"
                })
            elif status == 'OPTIMIZE':
                # Promising but not passing
                sharpe = metrics.get('sharpe', 0)
                turnover = metrics.get('turnover', 0)
                
                issues = []
                if isinstance(sharpe, (int, float)) and sharpe < 1.25:
                    issues.append(f"Sharpe too low ({sharpe:.2f} < 1.25)")
                if isinstance(turnover, (int, float)) and turnover > 0.7:
                    issues.append(f"Turnover too high ({turnover:.2f} > 0.7)")
                
                feedback_items.append({
                    'expression': alpha.expression[:150],
                    'result': 'NEEDS_OPTIMIZATION',
                    'sharpe': sharpe,
                    'fitness': metrics.get('fitness', 'N/A'),
                    'turnover': turnover,
                    'issue': '; '.join(issues) if issues else 'Below thresholds',
                    'lesson': 'Consider adding ts_decay_linear for turnover control or longer windows for stability'
                })
            else:
                # Failed
                sharpe = metrics.get('sharpe', 0)
                turnover = metrics.get('turnover', 0)
                
                issues = []
                recommendations = []
                
                if isinstance(sharpe, (int, float)):
                    if sharpe < 0:
                        issues.append(f"Negative Sharpe ({sharpe:.2f})")
                        recommendations.append("Try inverting the signal with negative sign")
                    elif sharpe < 0.5:
                        issues.append(f"Very low Sharpe ({sharpe:.2f})")
                        recommendations.append("Signal may be too weak or noisy")
                
                if isinstance(turnover, (int, float)) and turnover > 0.7:
                    issues.append(f"Excessive turnover ({turnover:.2f})")
                    recommendations.append("Wrap with ts_decay_linear(signal, 10+)")
                
                feedback_items.append({
                    'expression': alpha.expression[:150],
                    'result': 'FAILED',
                    'sharpe': sharpe,
                    'fitness': metrics.get('fitness', 'N/A'),
                    'turnover': turnover,
                    'issue': '; '.join(issues) if issues else 'Failed quality gates',
                    'lesson': '; '.join(recommendations) if recommendations else 'Try different approach'
                })
        
        # 2. Process explicit failures (syntax errors, simulation errors, etc.)
        for failure in failures[:max_items - len(feedback_items)]:
            error_type = failure.get('error_type', 'UNKNOWN')
            error_msg = failure.get('error_message', '')[:200]
            expression = failure.get('expression', '')[:150]
            
            lesson = ""
            if 'syntax' in error_type.lower():
                lesson = "Check expression syntax - ensure valid operator calls and field names"
            elif 'field' in error_type.lower() or 'field' in error_msg.lower():
                lesson = "Field not found - only use fields that exist in the dataset"
            elif 'vector' in error_msg.lower() or 'matrix' in error_msg.lower():
                lesson = "Type mismatch - use vec_avg/vec_sum for VECTOR fields"
            elif 'timeout' in error_type.lower():
                lesson = "Expression too complex - simplify or reduce nesting"
            else:
                lesson = f"Error: {error_msg[:100]}"
            
            feedback_items.append({
                'expression': expression,
                'result': 'ERROR',
                'sharpe': 'N/A',
                'fitness': 'N/A',
                'turnover': 'N/A',
                'issue': f"{error_type}: {error_msg[:100]}",
                'lesson': lesson
            })
        
        logger.debug(f"[Feedback] Extracted {len(feedback_items)} injectable feedback items")
        
        return feedback_items[:max_items]
    
    def generate_feedback_summary(
        self,
        alphas: List[Alpha],
        failures: List[Dict]
    ) -> str:
        """
        Generate a human-readable summary of round feedback.
        
        This can be included in prompts or logs to provide context.
        """
        passed = [a for a in alphas if getattr(a, 'quality_status', None) == 'PASS']
        optimize = [a for a in alphas if getattr(a, 'quality_status', None) == 'OPTIMIZE']
        failed = [a for a in alphas if getattr(a, 'quality_status', None) not in ['PASS', 'OPTIMIZE']]
        
        lines = [
            f"=== Round Feedback Summary ===",
            f"Passed: {len(passed)}, Optimize: {len(optimize)}, Failed: {len(failed)}, Errors: {len(failures)}",
        ]
        
        if passed:
            best = max(passed, key=lambda a: (a.metrics or {}).get('sharpe', 0))
            lines.append(f"Best Alpha: Sharpe={best.metrics.get('sharpe', 'N/A'):.2f}")
        
        # Common failure reasons
        failure_reasons = {}
        for a in failed:
            m = getattr(a, 'metrics', {}) or {}
            sharpe = m.get('sharpe', 0)
            turnover = m.get('turnover', 0)
            
            if isinstance(sharpe, (int, float)) and sharpe < 0:
                failure_reasons['negative_sharpe'] = failure_reasons.get('negative_sharpe', 0) + 1
            elif isinstance(turnover, (int, float)) and turnover > 0.7:
                failure_reasons['high_turnover'] = failure_reasons.get('high_turnover', 0) + 1
            else:
                failure_reasons['low_sharpe'] = failure_reasons.get('low_sharpe', 0) + 1
        
        for f in failures:
            error_type = f.get('error_type', 'unknown').lower()
            if 'syntax' in error_type:
                failure_reasons['syntax_error'] = failure_reasons.get('syntax_error', 0) + 1
            else:
                failure_reasons['other_error'] = failure_reasons.get('other_error', 0) + 1
        
        if failure_reasons:
            lines.append("Common Issues: " + ", ".join([f"{k}={v}" for k, v in failure_reasons.items()]))
        
        return "\n".join(lines)
