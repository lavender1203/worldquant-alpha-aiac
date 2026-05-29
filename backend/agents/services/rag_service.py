"""
RAG Service - Enhanced Knowledge Base Retrieval for Mining Patterns

Features:
1. Dataset category-aware pattern retrieval
2. Region-specific pattern filtering
3. Intelligent fallback to generic patterns
4. Success/failure pattern recording with proper categorization
5. Pattern usage tracking and scoring
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.dialects.postgresql import JSONB
from loguru import logger

from backend.models import KnowledgeEntry, DatasetMetadata


# Dataset category mapping for intelligent pattern matching
DATASET_CATEGORY_MAPPING = {
    "pv": ["pv", "price", "volume", "trade", "ohlc", "vwap"],
    "analyst": ["analyst", "anl", "estimate", "forecast", "recommendation", "eps", "target"],
    "fundamental": ["fundamental", "fnd", "fin", "balance", "income", "cash", "ratio", "margin"],
    "macro": ["macro", "mcr"],
    "socialmedia": ["socialmedia", "social", "scl", "snt"],
    "news": ["news", "headline", "article", "media", "oth635"],
    "sentiment": ["sentiment", "sent"],
    "earnings": ["earnings", "earning", "eps"],
    "imbalance": ["imbalance", "imb"],
    "insiders": ["insider", "insiders"],
    "institutions": ["institution", "institutions", "inst"],
    "risk": ["risk"],
    "other": ["other", "oth", "misc", "alternative"],
}


def infer_dataset_category(dataset_id: str) -> str:
    """
    Infer the category of a dataset from its ID.
    
    Args:
        dataset_id: Dataset identifier (e.g., "analyst15", "pv6", "other635")
    
    Returns:
        Category string (pv, analyst, fundamental, news, other)
    """
    if not dataset_id:
        return "other"
    
    dataset_lower = dataset_id.lower()
    
    for category, keywords in DATASET_CATEGORY_MAPPING.items():
        for keyword in keywords:
            if keyword in dataset_lower:
                return category
    
    return "other"


class RAGResult:
    """RAG query result container with enhanced metadata."""
    
    def __init__(
        self,
        patterns: List[Dict] = None,
        pitfalls: List[Dict] = None,
        dataset_info: Optional[Dict] = None,
        category: str = "other",
        region: str = None
    ):
        self.patterns = patterns or []
        self.pitfalls = pitfalls or []
        self.dataset_info = dataset_info
        self.category = category
        self.region = region
    
    def to_dict(self) -> Dict:
        return {
            "patterns": self.patterns,
            "pitfalls": self.pitfalls,
            "dataset_info": self.dataset_info,
            "category": self.category,
            "region": self.region
        }
    
    def get_few_shot_text(self) -> str:
        """Format patterns as few-shot examples for prompts."""
        if not self.patterns:
            return "暂无成功模式参考"
        
        lines = []
        for p in self.patterns:
            pattern = p.get('pattern', '')
            desc = p.get('description', '')
            sharpe = p.get('metadata', {}).get('expected_sharpe', '')
            sharpe_str = f" [Expected Sharpe: {sharpe}]" if sharpe else ""
            lines.append(f"- {pattern}: {desc}{sharpe_str}")
        
        return "\n".join(lines)
    
    def get_constraints_text(self) -> str:
        """Format pitfalls as negative constraints for prompts."""
        if not self.pitfalls:
            return "暂无特殊限制"
        
        lines = []
        for p in self.pitfalls:
            pattern = p.get('pattern', '')
            desc = p.get('description', '')
            err_type = p.get('error_type', '')
            err_str = f" [{err_type}]" if err_type else ""
            lines.append(f"- 避免: {pattern}{err_str} (原因: {desc})")
        
        return "\n".join(lines)


class RAGService:
    """
    Enhanced Knowledge Base Retrieval Service.
    
    Features:
    - Category-aware success pattern retrieval
    - Region-specific pattern filtering
    - Intelligent fallback to generic patterns
    - Failure pitfall retrieval with severity ranking
    - Pattern usage tracking
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def query(
        self,
        dataset_id: str = None,
        region: str = None,
        max_patterns: int = 5,
        max_pitfalls: int = 10
    ) -> RAGResult:
        """
        Query knowledge base for relevant patterns and pitfalls.
        
        Enhanced with category-aware retrieval:
        1. First try dataset-specific patterns
        2. Then category-specific patterns  
        3. Finally fall back to generic patterns
        
        Args:
            dataset_id: Optional dataset to filter by
            region: Optional region to filter by
            max_patterns: Maximum success patterns to return
            max_pitfalls: Maximum failure pitfalls to return
            
        Returns:
            RAGResult with patterns, pitfalls, and dataset info
        """
        # Prefer authoritative dataset metadata over ID-prefix inference.
        dataset_info = await self._get_dataset_info(dataset_id) if dataset_id else None
        category = (dataset_info or {}).get("category") or (
            infer_dataset_category(dataset_id) if dataset_id else "other"
        )
        
        logger.debug(
            f"[RAGService] Query | dataset={dataset_id} region={region} category={category}"
        )
        
        # Get success patterns with category awareness
        patterns = await self._get_success_patterns_enhanced(
            dataset_id=dataset_id,
            category=category,
            region=region,
            limit=max_patterns
        )
        
        # Get failure pitfalls
        pitfalls = await self._get_failure_pitfalls_enhanced(
            dataset_id=dataset_id,
            category=category,
            region=region,
            limit=max_pitfalls
        )
        
        logger.info(
            f"[RAGService] Query complete | "
            f"category={category} patterns={len(patterns)} pitfalls={len(pitfalls)}"
        )
        
        return RAGResult(
            patterns=patterns,
            pitfalls=pitfalls,
            dataset_info=dataset_info,
            category=category,
            region=region
        )
    
    async def _get_success_patterns_enhanced(
        self,
        dataset_id: str = None,
        category: str = "other",
        region: str = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Get success patterns with intelligent category matching.
        
        Priority order:
        1. Exact dataset match
        2. Category match
        3. Region match
        4. Generic patterns (sorted by usage/score)
        """
        patterns = []
        
        # Query all active success patterns
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.entry_type == 'SUCCESS_PATTERN',
            KnowledgeEntry.is_active == True
        )
        
        result = await self.db.execute(query)
        entries = result.scalars().all()
        
        # Score and sort patterns
        scored_patterns = []
        for entry in entries:
            metadata = entry.meta_data or {}
            
            # Skip region config entries (they're metadata, not patterns)
            if metadata.get('pattern_type') == 'region_config':
                continue
            
            score = 0.0
            
            # 1. Dataset match (highest priority)
            entry_dataset = metadata.get('dataset', metadata.get('dataset_id', ''))
            if dataset_id and entry_dataset:
                if entry_dataset.lower() == dataset_id.lower():
                    score += 100.0
            
            # 2. Category match
            entry_categories = metadata.get('dataset_categories', [])
            entry_category = metadata.get('dataset_category', '')
            
            if category:
                if category in entry_categories:
                    score += 50.0
                elif entry_category == category:
                    score += 50.0
                elif category in str(entry_category).lower():
                    score += 30.0
            
            # 3. Region match
            entry_regions = metadata.get('regions', [])
            if region:
                if region in entry_regions:
                    score += 20.0
                elif not entry_regions:  # Generic pattern
                    score += 5.0
            
            # 4. Base score from metadata
            base_score = metadata.get('score', 0.5)
            expected_sharpe = metadata.get('expected_sharpe', 1.0)
            score += base_score * 10.0
            score += min(expected_sharpe, 2.0) * 5.0
            
            # 5. Usage count bonus (popular patterns)
            score += min(entry.usage_count or 0, 10) * 0.5
            
            scored_patterns.append({
                'entry': entry,
                'metadata': metadata,
                'score': score
            })
        
        # Sort by score descending
        scored_patterns.sort(key=lambda x: x['score'], reverse=True)
        
        # Build result list
        for sp in scored_patterns[:limit]:
            entry = sp['entry']
            metadata = sp['metadata']
            patterns.append({
                'pattern': entry.pattern,
                'description': entry.description,
                'usage_count': entry.usage_count,
                'metadata': metadata,
                'match_score': sp['score']
            })
        
        # Log pattern sources for debugging
        if patterns:
            sources = [p.get('metadata', {}).get('source', 'unknown') for p in patterns]
            logger.debug(f"[RAGService] Pattern sources: {sources}")
        
        return patterns
    
    async def _get_failure_pitfalls_enhanced(
        self,
        dataset_id: str = None,
        category: str = "other",
        region: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get failure pitfalls with severity-based ranking.
        
        Priority:
        1. High severity errors first
        2. Category-relevant pitfalls
        3. Recent pitfalls
        """
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.entry_type == 'FAILURE_PITFALL',
            KnowledgeEntry.is_active == True
        )
        
        result = await self.db.execute(query)
        entries = result.scalars().all()
        
        # Score pitfalls
        scored_pitfalls = []
        severity_weights = {'high': 30, 'medium': 20, 'low': 10}
        
        for entry in entries:
            metadata = entry.meta_data or {}
            score = 0.0
            
            # Severity weight
            severity = metadata.get('severity', 'medium')
            score += severity_weights.get(severity, 15)
            
            # Category relevance
            pitfall_category = metadata.get('dataset_category', '')
            if category and pitfall_category:
                if category == pitfall_category:
                    score += 20.0
            
            # Error type relevance
            error_type = metadata.get('error_type', '')
            # Prioritize type errors and syntax errors
            if error_type in ['TYPE_ERROR', 'SYNTAX_ERROR', 'SEMANTIC_ERROR']:
                score += 15.0
            
            scored_pitfalls.append({
                'entry': entry,
                'metadata': metadata,
                'score': score
            })
        
        # Sort by score
        scored_pitfalls.sort(key=lambda x: x['score'], reverse=True)
        
        # Build result
        pitfalls = []
        for sp in scored_pitfalls[:limit]:
            entry = sp['entry']
            metadata = sp['metadata']
            pitfalls.append({
                'pattern': entry.pattern,
                'description': entry.description,
                'error_type': metadata.get('error_type'),
                'severity': metadata.get('severity'),
                'metadata': metadata
            })
        
        return pitfalls
    
    # Legacy method for backward compatibility
    async def _get_success_patterns(
        self,
        dataset_id: str = None,
        region: str = None,
        limit: int = 5
    ) -> List[Dict]:
        """Legacy method - redirects to enhanced version."""
        category = infer_dataset_category(dataset_id) if dataset_id else "other"
        return await self._get_success_patterns_enhanced(
            dataset_id=dataset_id,
            category=category,
            region=region,
            limit=limit
        )
    
    # Legacy method for backward compatibility
    async def _get_failure_pitfalls(
        self,
        dataset_id: str = None,
        region: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """Legacy method - redirects to enhanced version."""
        category = infer_dataset_category(dataset_id) if dataset_id else "other"
        return await self._get_failure_pitfalls_enhanced(
            dataset_id=dataset_id,
            category=category,
            region=region,
            limit=limit
        )
    
    async def get_field_blacklist(self, region: str = None) -> List[str]:
        """Get list of blacklisted fields."""
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.entry_type == 'FIELD_BLACKLIST',
            KnowledgeEntry.is_active == True
        )
        
        result = await self.db.execute(query)
        entries = result.scalars().all()
        
        blacklist = []
        for entry in entries:
            metadata = entry.meta_data or {}
            if region and metadata.get('region') and metadata['region'] != region:
                continue
            
            field_name = metadata.get('field') or entry.pattern
            if field_name:
                blacklist.append(field_name)
        
        return blacklist
    
    async def _get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset metadata."""
        query = select(DatasetMetadata).where(
            DatasetMetadata.dataset_id == dataset_id
        ).limit(1)
        result = await self.db.execute(query)
        dataset = result.scalars().first()
        
        if not dataset:
            return None
        
        return {
            'dataset_id': dataset.dataset_id,
            'region': dataset.region,
            'category': dataset.category,
            'subcategory': dataset.subcategory,
            'description': dataset.description,
            'field_count': dataset.field_count,
            'mining_weight': dataset.mining_weight
        }
    
    async def increment_pattern_usage(self, pattern: str) -> bool:
        """Increment usage count for a pattern (called on successful use)."""
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.pattern == pattern,
            KnowledgeEntry.is_active == True
        )
        result = await self.db.execute(query)
        entry = result.scalar_one_or_none()
        
        if entry:
            entry.usage_count += 1
            logger.debug(f"[RAGService] Incremented usage | pattern={pattern}")
            return True
        
        return False
    
    # =========================================================================
    # P0-fix-1: Knowledge Feedback Loop - Write patterns back to KB
    # =========================================================================
    
    async def record_failure_pattern(
        self,
        expression: str,
        error_type: str,
        metrics: Dict = None,
        region: str = None,
        dataset_id: str = None
    ) -> bool:
        """
        Record a failure pattern to the knowledge base.
        
        This is the KEY feedback loop that enables learning from failures.
        Called after evaluation identifies a failed alpha.
        """
        from backend.knowledge_extraction import expression_to_skeleton, extract_operator_chain
        
        try:
            # Extract pattern skeleton (structural, not specific)
            skeleton = expression_to_skeleton(expression)
            op_chain = extract_operator_chain(expression)
            
            # Infer category from dataset_id
            category = infer_dataset_category(dataset_id) if dataset_id else "other"
            
            # Check if similar pattern already exists
            existing = await self._find_similar_pitfall(skeleton, region)
            
            if existing:
                # Update existing pattern's failure count
                existing.meta_data = existing.meta_data or {}
                existing.meta_data['failure_count'] = existing.meta_data.get('failure_count', 0) + 1
                existing.meta_data['last_failure'] = datetime.now().isoformat()
                if metrics:
                    existing.meta_data['avg_sharpe'] = metrics.get('sharpe', 0)
                logger.debug(f"[RAGService] Updated existing pitfall | skeleton={skeleton[:50]}")
            else:
                # Create new pitfall entry
                description = self._generate_pitfall_description(error_type, metrics, op_chain)
                
                # Determine severity based on error type
                severity = 'medium'
                if error_type in ['TYPE_ERROR', 'SYNTAX_ERROR', 'SEMANTIC_ERROR']:
                    severity = 'high'
                elif error_type in ['LOW_SHARPE', 'HIGH_TURNOVER']:
                    severity = 'medium'
                elif error_type == 'NEGATIVE_SIGNAL':
                    severity = 'low'  # Can be fixed by sign flip
                
                new_entry = KnowledgeEntry(
                    pattern=skeleton,
                    description=description,
                    entry_type='FAILURE_PITFALL',
                    is_active=True,
                    usage_count=0,
                    meta_data={
                        'source': 'feedback_loop',
                        'region': region,
                        'dataset': dataset_id,
                        'dataset_category': category,
                        'error_type': error_type,
                        'severity': severity,
                        'operator_chain': op_chain[:5] if op_chain else [],
                        'example_expression': expression[:200],
                        'failure_count': 1,
                        'sharpe': metrics.get('sharpe', 0) if metrics else 0,
                        'fitness': metrics.get('fitness', 0) if metrics else 0,
                        'turnover': metrics.get('turnover', 0) if metrics else 0,
                        'created_at': datetime.now().isoformat()
                    }
                )
                self.db.add(new_entry)
                logger.info(f"[RAGService] Created new pitfall | skeleton={skeleton[:50]} error={error_type} category={category}")
            
            await self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"[RAGService] Failed to record pitfall | error={e}")
            await self.db.rollback()
            return False
    
    async def record_success_pattern(
        self,
        expression: str,
        metrics: Dict,
        region: str = None,
        dataset_id: str = None,
        alpha_id: str = None
    ) -> bool:
        """
        Record a success pattern to the knowledge base.
        
        Called when an alpha passes all quality thresholds.
        """
        from backend.knowledge_extraction import expression_to_skeleton, extract_operator_chain
        
        try:
            skeleton = expression_to_skeleton(expression)
            op_chain = extract_operator_chain(expression)
            
            # Infer category from dataset_id
            category = infer_dataset_category(dataset_id) if dataset_id else "other"
            
            # Check if similar pattern exists
            existing = await self._find_similar_success(skeleton, region)
            
            if existing:
                # Update existing pattern
                existing.usage_count += 1
                existing.meta_data = existing.meta_data or {}
                existing.meta_data['success_count'] = existing.meta_data.get('success_count', 0) + 1
                existing.meta_data['last_success'] = datetime.now().isoformat()
                # Update running average metrics
                n = existing.meta_data.get('success_count', 1)
                old_sharpe = existing.meta_data.get('avg_sharpe', 0)
                existing.meta_data['avg_sharpe'] = (old_sharpe * (n-1) + metrics.get('sharpe', 0)) / n
                logger.info(f"[RAGService] Updated success pattern | skeleton={skeleton[:50]}")
            else:
                # Create new success pattern with full category info
                sharpe = metrics.get('sharpe', 0)
                fitness = metrics.get('fitness', 0)
                turnover = metrics.get('turnover', 0)
                
                description = f"Sharpe: {sharpe:.2f}, Fitness: {fitness:.2f}, Turnover: {turnover:.2f}"
                
                # Calculate a quality score
                score = min(1.0, (sharpe / 2.0) * 0.6 + (fitness / 1.5) * 0.3 + max(0, (0.7 - turnover)) * 0.1)
                
                new_entry = KnowledgeEntry(
                    pattern=skeleton,
                    description=description,
                    entry_type='SUCCESS_PATTERN',
                    is_active=True,
                    usage_count=1,
                    meta_data={
                        'source': 'feedback_loop',
                        'region': region,
                        'regions': [region] if region else [],
                        'dataset': dataset_id,
                        'dataset_category': category,
                        'dataset_categories': [category],
                        'operator_chain': op_chain[:5] if op_chain else [],
                        'example_expression': expression[:200],
                        'alpha_id': alpha_id,
                        'success_count': 1,
                        'avg_sharpe': sharpe,
                        'avg_fitness': fitness,
                        'avg_turnover': turnover,
                        'expected_sharpe': sharpe,
                        'score': score,
                        'created_at': datetime.now().isoformat()
                    }
                )
                self.db.add(new_entry)
                logger.info(f"[RAGService] Created new success pattern | skeleton={skeleton[:50]} sharpe={sharpe:.2f} category={category}")
            
            await self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"[RAGService] Failed to record success | error={e}")
            await self.db.rollback()
            return False
    
    async def _find_similar_pitfall(self, skeleton: str, region: str = None) -> Optional[KnowledgeEntry]:
        """Find existing pitfall with similar skeleton"""
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.entry_type == 'FAILURE_PITFALL',
            KnowledgeEntry.pattern == skeleton,
            KnowledgeEntry.is_active == True
        )
        if region:
            # Also match patterns without region (global)
            pass  # We'll match exact skeleton first
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def _find_similar_success(self, skeleton: str, region: str = None) -> Optional[KnowledgeEntry]:
        """Find existing success pattern with similar skeleton"""
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.entry_type == 'SUCCESS_PATTERN',
            KnowledgeEntry.pattern == skeleton,
            KnowledgeEntry.is_active == True
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    def _generate_pitfall_description(self, error_type: str, metrics: Dict, op_chain: List) -> str:
        """Generate human-readable pitfall description"""
        parts = []
        
        if error_type == 'LOW_SHARPE':
            sharpe = metrics.get('sharpe', 0) if metrics else 0
            parts.append(f"低Sharpe ({sharpe:.2f})")
        elif error_type == 'LOW_FITNESS':
            fitness = metrics.get('fitness', 0) if metrics else 0
            parts.append(f"低Fitness ({fitness:.2f})")
        elif error_type == 'HIGH_TURNOVER':
            turnover = metrics.get('turnover', 0) if metrics else 0
            parts.append(f"高Turnover ({turnover:.2f})")
        elif error_type == 'HIGH_CORRELATION':
            parts.append("高相关性 - 与现有alpha重复")
        elif error_type == 'NEGATIVE_SIGNAL':
            parts.append("负信号 - 方向相反")
        else:
            parts.append(f"失败类型: {error_type}")
        
        if op_chain:
            parts.append(f"算子链: {' → '.join(op_chain[:3])}")
        
        return "; ".join(parts)
    
    async def get_region_config(self, region: str) -> Optional[Dict]:
        """
        Get recommended configuration for a region from knowledge base.
        
        Args:
            region: Region code (USA, KOR, ASI, etc.)
        
        Returns:
            Dict with recommended settings or None if not found
        """
        query = select(KnowledgeEntry).where(
            KnowledgeEntry.entry_type == 'SUCCESS_PATTERN',
            KnowledgeEntry.pattern == f"REGION_CONFIG:{region.upper()}",
            KnowledgeEntry.is_active == True
        )
        
        result = await self.db.execute(query)
        entry = result.scalar_one_or_none()
        
        if entry and entry.meta_data:
            return {
                'region': region.upper(),
                'recommended_universe': entry.meta_data.get('recommended_universe'),
                'recommended_decay': entry.meta_data.get('recommended_decay'),
                'recommended_neutralization': entry.meta_data.get('recommended_neutralization'),
                'sharpe_adjustment': entry.meta_data.get('sharpe_adjustment', 1.0),
                'notes': entry.description
            }
        
        # Fallback to default USA settings if not found
        return {
            'region': region.upper(),
            'recommended_universe': 'TOP3000',
            'recommended_decay': 4,
            'recommended_neutralization': 'SUBINDUSTRY',
            'sharpe_adjustment': 1.0,
            'notes': 'Default settings'
        }
    
    async def get_patterns_by_category(
        self,
        category: str,
        region: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get success patterns for a specific dataset category.
        
        Args:
            category: Dataset category (pv, analyst, fundamental, news, other)
            region: Optional region filter
            limit: Maximum patterns to return
        
        Returns:
            List of pattern dictionaries
        """
        return await self._get_success_patterns_enhanced(
            dataset_id=None,
            category=category,
            region=region,
            limit=limit
        )
