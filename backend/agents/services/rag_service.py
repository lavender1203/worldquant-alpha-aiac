"""
RAG Service - Enhanced Knowledge Base Retrieval for Mining Patterns

Features:
1. Dataset category-aware pattern retrieval
2. Region-specific pattern filtering
3. Intelligent fallback to generic patterns
4. Success/failure pattern recording with proper categorization
5. Pattern usage tracking and scoring
"""

from typing import Dict, List, Optional, Tuple, Any
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
    "news": ["news", "sentiment", "headline", "article", "media", "social", "oth635"],
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
        # Infer category from dataset_id
        category = infer_dataset_category(dataset_id) if dataset_id else "other"
        
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
        
        # Get dataset info
        dataset_info = None
        if dataset_id:
            dataset_info = await self._get_dataset_info(dataset_id)
        
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
    
    # =========================================================================
    # Dynamic Field-Aware Pattern Adaptation
    # =========================================================================
    
    def analyze_field_types(self, fields: List[Dict]) -> Dict[str, List[str]]:
        """
        Analyze available fields and categorize them by semantic type.
        
        This helps generate contextually relevant patterns based on actual fields.
        
        Args:
            fields: List of field dictionaries with 'id', 'name', 'description', etc.
        
        Returns:
            Dict mapping field types to field IDs
        """
        field_types = {
            'sentiment': [],      # Sentiment scores, sentiment indicators
            'count': [],          # Count-based fields (article counts, etc.)
            'ratio': [],          # Ratio/percentage fields
            'timestamp': [],      # Time-related fields
            'text_derived': [],   # Fields derived from text (char count, word count)
            'numeric': [],        # Generic numeric fields
            'binary': [],         # Binary/flag fields
            'categorical': [],    # Category/classification fields
        }
        
        # Keywords for field type inference
        type_keywords = {
            'sentiment': ['sentiment', 'score', 'polarity', 'positive', 'negative', 'mood'],
            'count': ['count', 'num', 'total', 'frequency'],
            'ratio': ['ratio', 'percent', 'pct', 'rate', 'proportion'],
            'timestamp': ['date', 'time', 'timestamp', 'day', 'hour'],
            'text_derived': ['char', 'word', 'length', 'headline', 'title', 'character'],
            'binary': ['flag', 'indicator', 'is_', 'has_', 'bool'],
            'categorical': ['category', 'type', 'class', 'sector', 'industry'],
        }
        
        for f in fields:
            field_id = f.get('id', f.get('name', ''))
            field_name = f.get('name', field_id)
            field_desc = f.get('description', '')
            
            combined_text = f"{field_id} {field_name} {field_desc}".lower()
            
            categorized = False
            for ftype, keywords in type_keywords.items():
                if any(kw in combined_text for kw in keywords):
                    field_types[ftype].append(field_id)
                    categorized = True
                    break
            
            if not categorized:
                field_types['numeric'].append(field_id)
        
        return field_types
    
    def generate_adaptive_patterns(
        self,
        field_types: Dict[str, List[str]],
        dataset_id: str = None,
        category: str = "other"
    ) -> List[Dict]:
        """
        Generate contextually relevant pattern templates based on actual field types.
        
        This is the KEY fix for the knowledge-reality mismatch:
        Instead of using generic patterns like 'ts_rank(close, 10)',
        we generate patterns using actual available fields.
        
        Args:
            field_types: Output from analyze_field_types()
            dataset_id: Dataset identifier
            category: Dataset category
        
        Returns:
            List of adaptive pattern dictionaries
        """
        patterns = []
        
        # Pattern templates by field type combination
        # Each template uses placeholders that get filled with actual fields
        
        # 1. Sentiment-based patterns (news/alternative data)
        if field_types.get('sentiment'):
            sent_field = field_types['sentiment'][0]
            patterns.extend([
                {
                    'pattern': f"ts_decay_linear(ts_rank(vec_avg({sent_field}), 10), 8)",
                    'description': f"Smoothed sentiment momentum using {sent_field}. Decay reduces turnover.",
                    'metadata': {'field_type': 'sentiment', 'expected_sharpe': 1.0, 'source': 'adaptive'}
                },
                {
                    'pattern': f"ts_decay_linear(ts_zscore(vec_avg({sent_field}), 20), 10)",
                    'description': f"Z-scored sentiment deviation with decay for stability.",
                    'metadata': {'field_type': 'sentiment', 'expected_sharpe': 1.1, 'source': 'adaptive'}
                },
            ])
        
        # 2. Count-based patterns (article counts, etc.)
        if field_types.get('count'):
            count_field = field_types['count'][0]
            patterns.extend([
                {
                    'pattern': f"ts_decay_linear(ts_rank(ts_delta(vec_sum({count_field}), 5), 15), 10)",
                    'description': f"Change in {count_field} ranked and smoothed. Captures attention shifts.",
                    'metadata': {'field_type': 'count', 'expected_sharpe': 0.9, 'source': 'adaptive'}
                },
                {
                    'pattern': f"ts_decay_linear(rank(vec_sum({count_field})), 8)",
                    'description': f"Cross-sectional rank of {count_field} with decay.",
                    'metadata': {'field_type': 'count', 'expected_sharpe': 0.8, 'source': 'adaptive'}
                },
            ])
        
        # 3. Text-derived patterns (character count, word count)
        if field_types.get('text_derived'):
            text_field = field_types['text_derived'][0]
            patterns.extend([
                {
                    'pattern': f"ts_decay_linear(ts_rank(ts_std_dev(vec_sum({text_field}), 10), 15), 10)",
                    'description': f"Volatility of {text_field} ranked. High volatility may indicate uncertainty.",
                    'metadata': {'field_type': 'text_derived', 'expected_sharpe': 0.8, 'source': 'adaptive'}
                },
            ])
        
        # 4. Cross-field patterns (combine multiple field types)
        if field_types.get('sentiment') and field_types.get('count'):
            sent_field = field_types['sentiment'][0]
            count_field = field_types['count'][0]
            patterns.append({
                'pattern': f"ts_decay_linear(ts_rank(divide(vec_avg({sent_field}), add(vec_sum({count_field}), 1)), 15), 10)",
                'description': f"Sentiment per article count - quality over quantity signal.",
                'metadata': {'field_type': 'combined', 'expected_sharpe': 1.2, 'source': 'adaptive'}
            })
        
        # 5. Generic patterns for any numeric fields
        if field_types.get('numeric'):
            num_field = field_types['numeric'][0]
            patterns.extend([
                {
                    'pattern': f"ts_decay_linear(ts_rank(vec_avg({num_field}), 10), 8)",
                    'description': f"Standard time-series ranking with decay for {num_field}.",
                    'metadata': {'field_type': 'numeric', 'expected_sharpe': 0.7, 'source': 'adaptive'}
                },
            ])
        
        # 6. Universal turnover-safe patterns (always include)
        patterns.append({
            'pattern': "ALWAYS use ts_decay_linear(signal, N) where N >= 5 as the outer wrapper",
            'description': "CRITICAL: High turnover is the #1 failure. Always apply decay.",
            'metadata': {'field_type': 'universal', 'severity': 'high', 'source': 'adaptive'}
        })
        
        logger.debug(f"[RAGService] Generated {len(patterns)} adaptive patterns for {dataset_id or category}")
        
        return patterns
    
    # =========================================================================
    # Hierarchical RAG (Alpha-GPT Style)
    # =========================================================================
    
    async def hierarchical_rag_query(
        self,
        region: str,
        universe: str,
        exploration_level: str = "category",  # "category" | "subcategory" | "field"
        target_category: str = None,
        target_subcategory: str = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Alpha-GPT style Hierarchical RAG for systematic dataset exploration.
        
        This implements the three-level exploration strategy from Alpha-GPT 1.0:
        1. Level 1 (Category): Explore high-level categories (PV, Analyst, Fundamental, News)
        2. Level 2 (Subcategory): Explore subcategories within a chosen category
        3. Level 3 (Field): Explore specific fields within a subcategory
        
        Args:
            region: Market region
            universe: Universe (TOP3000, etc.)
            exploration_level: Which level to explore
            target_category: For Level 2/3, the category to focus on
            target_subcategory: For Level 3, the subcategory to focus on
            max_results: Maximum results per level
        
        Returns:
            Dict with exploration results and suggestions
        """
        logger.info(
            f"[HierarchicalRAG] Query | level={exploration_level} "
            f"category={target_category} subcategory={target_subcategory}"
        )
        
        if exploration_level == "category":
            return await self._explore_categories(region, universe, max_results)
        elif exploration_level == "subcategory":
            return await self._explore_subcategories(region, universe, target_category, max_results)
        elif exploration_level == "field":
            return await self._explore_fields(region, universe, target_category, target_subcategory, max_results)
        else:
            return await self._explore_categories(region, universe, max_results)
    
    async def _explore_categories(
        self,
        region: str,
        universe: str,
        max_results: int
    ) -> Dict[str, Any]:
        """
        Level 1: Explore high-level dataset categories.
        
        Returns categories with their statistics and recommendations.
        """
        # Query datasets grouped by category
        query = select(DatasetMetadata).where(
            DatasetMetadata.region == region,
            DatasetMetadata.is_active == True
        )
        
        result = await self.db.execute(query)
        datasets = result.scalars().all()
        
        # Group by category
        category_stats = {}
        for ds in datasets:
            cat = ds.category or infer_dataset_category(ds.dataset_id)
            
            if cat not in category_stats:
                category_stats[cat] = {
                    "category": cat,
                    "datasets": [],
                    "total_fields": 0,
                    "total_alphas": 0,
                    "avg_success_rate": 0.0,
                    "subcategories": set()
                }
            
            category_stats[cat]["datasets"].append(ds.dataset_id)
            category_stats[cat]["total_fields"] += ds.field_count or 0
            category_stats[cat]["total_alphas"] += ds.alpha_success_count or 0
            
            if ds.subcategory:
                category_stats[cat]["subcategories"].add(ds.subcategory)
        
        # Calculate success rates and sort
        categories = []
        for cat, stats in category_stats.items():
            # Get historical success info from knowledge base
            success_patterns = await self._get_success_patterns_enhanced(
                category=cat, region=region, limit=3
            )
            
            stats["subcategories"] = list(stats["subcategories"])[:5]
            stats["pattern_count"] = len(success_patterns)
            stats["exploration_priority"] = self._calculate_exploration_priority(stats)
            
            categories.append(stats)
        
        # Sort by exploration priority
        categories.sort(key=lambda x: x["exploration_priority"], reverse=True)
        
        return {
            "level": "category",
            "region": region,
            "universe": universe,
            "results": categories[:max_results],
            "recommendation": self._generate_category_recommendation(categories),
            "next_level": "subcategory"
        }
    
    async def _explore_subcategories(
        self,
        region: str,
        universe: str,
        target_category: str,
        max_results: int
    ) -> Dict[str, Any]:
        """
        Level 2: Explore subcategories within a category.
        """
        # Query datasets in this category
        query = select(DatasetMetadata).where(
            DatasetMetadata.region == region,
            DatasetMetadata.is_active == True
        )
        
        result = await self.db.execute(query)
        all_datasets = result.scalars().all()
        
        # Filter to target category
        datasets = [
            ds for ds in all_datasets
            if (ds.category or infer_dataset_category(ds.dataset_id)) == target_category
        ]
        
        # Group by subcategory
        subcategory_stats = {}
        for ds in datasets:
            subcat = ds.subcategory or "general"
            
            if subcat not in subcategory_stats:
                subcategory_stats[subcat] = {
                    "subcategory": subcat,
                    "datasets": [],
                    "total_fields": 0,
                    "sample_fields": []
                }
            
            subcategory_stats[subcat]["datasets"].append(ds.dataset_id)
            subcategory_stats[subcat]["total_fields"] += ds.field_count or 0
        
        # Add sample fields for each subcategory
        subcategories = []
        for subcat, stats in subcategory_stats.items():
            if stats["datasets"]:
                # Get sample fields from first dataset
                # First, look up the integer ID for the dataset_id string
                sample_ds_str = stats["datasets"][0]
                from backend.models import DataField
                
                # Find the integer ID for this dataset_id string
                ds_id_query = select(DatasetMetadata.id).where(
                    DatasetMetadata.dataset_id == sample_ds_str,
                    DatasetMetadata.region == region
                ).limit(1)
                ds_id_result = await self.db.execute(ds_id_query)
                sample_ds_int = ds_id_result.scalar_one_or_none()
                
                if sample_ds_int:
                    field_query = select(DataField).where(
                        DataField.dataset_id == sample_ds_int
                    ).limit(5)
                    field_result = await self.db.execute(field_query)
                    fields = field_result.scalars().all()
                    stats["sample_fields"] = [f.field_id for f in fields]
                else:
                    stats["sample_fields"] = []
            
            subcategories.append(stats)
        
        # Sort by field count (more fields = more opportunity)
        subcategories.sort(key=lambda x: x["total_fields"], reverse=True)
        
        return {
            "level": "subcategory",
            "category": target_category,
            "region": region,
            "results": subcategories[:max_results],
            "recommendation": f"Explore {subcategories[0]['subcategory'] if subcategories else 'general'} subcategory",
            "next_level": "field"
        }
    
    async def _explore_fields(
        self,
        region: str,
        universe: str,
        target_category: str,
        target_subcategory: str,
        max_results: int
    ) -> Dict[str, Any]:
        """
        Level 3: Explore specific fields within a subcategory.
        """
        from backend.models import DataField
        
        # Get datasets in this category/subcategory
        ds_query = select(DatasetMetadata).where(
            DatasetMetadata.region == region,
            DatasetMetadata.is_active == True
        )
        ds_result = await self.db.execute(ds_query)
        all_datasets = ds_result.scalars().all()
        
        # Filter by category and subcategory
        target_datasets = []
        for ds in all_datasets:
            ds_cat = ds.category or infer_dataset_category(ds.dataset_id)
            ds_subcat = ds.subcategory or "general"
            
            if ds_cat == target_category:
                if not target_subcategory or ds_subcat == target_subcategory:
                    target_datasets.append(ds.dataset_id)
        
        if not target_datasets:
            return {
                "level": "field",
                "category": target_category,
                "subcategory": target_subcategory,
                "results": [],
                "recommendation": "No datasets found in this category/subcategory"
            }
        
        # Look up integer IDs for the string dataset_ids
        ds_id_query = select(DatasetMetadata.id).where(
            DatasetMetadata.dataset_id.in_(target_datasets[:3]),
            DatasetMetadata.region == region
        )
        ds_id_result = await self.db.execute(ds_id_query)
        target_dataset_ids = [row for row in ds_id_result.scalars().all()]
        
        if not target_dataset_ids:
            return {
                "level": "field",
                "category": target_category,
                "subcategory": target_subcategory,
                "results": [],
                "recommendation": "No datasets found with valid IDs"
            }
        
        # Get fields from these datasets using integer IDs
        field_query = select(DataField).where(
            DataField.dataset_id.in_(target_dataset_ids)  # Use integer IDs
        ).limit(max_results * 3)
        
        field_result = await self.db.execute(field_query)
        fields = field_result.scalars().all()
        
        # Analyze and categorize fields
        field_info = []
        for f in fields:
            field_info.append({
                "field_id": f.field_id,
                "name": f.field_name,  # DataField uses field_name, not name
                "description": f.description,
                "dataset_id": f.dataset_id,
                "type": f.field_type,  # DataField uses field_type, not type
                "suggested_usage": self._suggest_field_usage(f)
            })
        
        # Deduplicate by field_id
        seen = set()
        unique_fields = []
        for f in field_info:
            if f["field_id"] not in seen:
                seen.add(f["field_id"])
                unique_fields.append(f)
        
        return {
            "level": "field",
            "category": target_category,
            "subcategory": target_subcategory,
            "datasets": target_datasets[:3],
            "results": unique_fields[:max_results],
            "recommendation": self._generate_field_recommendation(unique_fields)
        }
    
    def _calculate_exploration_priority(self, stats: Dict) -> float:
        """Calculate exploration priority for a category."""
        # Factors: field count, pattern success, unexplored potential
        field_score = min(stats["total_fields"] / 100, 1.0) * 0.3
        pattern_score = min(stats.get("pattern_count", 0) / 5, 1.0) * 0.3
        dataset_score = min(len(stats["datasets"]) / 10, 1.0) * 0.2
        unexplored_bonus = 0.2 if stats["total_alphas"] < 10 else 0.0
        
        return field_score + pattern_score + dataset_score + unexplored_bonus
    
    def _generate_category_recommendation(self, categories: List[Dict]) -> str:
        """Generate recommendation for which category to explore."""
        if not categories:
            return "No categories available"
        
        top = categories[0]
        return (
            f"Recommend exploring '{top['category']}' category with "
            f"{top['total_fields']} fields across {len(top['datasets'])} datasets"
        )
    
    def _suggest_field_usage(self, field) -> str:
        """Suggest how to use a field based on its characteristics."""
        name = (field.name or "").lower()
        desc = (field.description or "").lower()
        
        if "sentiment" in name or "sentiment" in desc:
            return "ts_decay_linear(ts_rank(vec_avg(FIELD), 10), 8)"
        elif "count" in name:
            return "ts_decay_linear(rank(vec_sum(FIELD)), 10)"
        elif "ratio" in name or "percent" in name:
            return "ts_decay_linear(ts_zscore(FIELD, 20), 10)"
        else:
            return "ts_decay_linear(ts_rank(FIELD, 15), 10)"
    
    def _generate_field_recommendation(self, fields: List[Dict]) -> str:
        """Generate recommendation for which fields to use."""
        if not fields:
            return "No fields available"
        
        sentiment_fields = [f for f in fields if "sentiment" in f.get("name", "").lower()]
        count_fields = [f for f in fields if "count" in f.get("name", "").lower()]
        
        if sentiment_fields:
            return f"Try sentiment field: {sentiment_fields[0]['field_id']}"
        elif count_fields:
            return f"Try count-based field: {count_fields[0]['field_id']}"
        else:
            return f"Start with field: {fields[0]['field_id']}"
    
    async def autonomous_hierarchical_exploration(
        self,
        region: str,
        universe: str,
        exploration_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Fully autonomous hierarchical exploration (Alpha-GPT autonomous mode).
        
        Automatically navigates all three levels and returns a complete
        exploration result with recommended fields and patterns.
        
        Args:
            region: Market region
            universe: Universe
            exploration_depth: How many categories/subcategories to explore
        
        Returns:
            Complete exploration result with recommended fields and patterns
        """
        logger.info(f"[HierarchicalRAG] Autonomous exploration | region={region}")
        
        # Level 1: Get top categories
        cat_result = await self.hierarchical_rag_query(
            region=region,
            universe=universe,
            exploration_level="category",
            max_results=exploration_depth
        )
        
        exploration_results = {
            "region": region,
            "universe": universe,
            "categories_explored": [],
            "recommended_fields": [],
            "recommended_patterns": []
        }
        
        # Level 2 & 3: Explore each top category
        for cat in cat_result.get("results", [])[:exploration_depth]:
            cat_name = cat["category"]
            
            # Get subcategories
            subcat_result = await self.hierarchical_rag_query(
                region=region,
                universe=universe,
                exploration_level="subcategory",
                target_category=cat_name,
                max_results=2
            )
            
            cat_exploration = {
                "category": cat_name,
                "subcategories": []
            }
            
            # Level 3: Get fields for each subcategory
            for subcat in subcat_result.get("results", [])[:2]:
                subcat_name = subcat["subcategory"]
                
                field_result = await self.hierarchical_rag_query(
                    region=region,
                    universe=universe,
                    exploration_level="field",
                    target_category=cat_name,
                    target_subcategory=subcat_name,
                    max_results=5
                )
                
                cat_exploration["subcategories"].append({
                    "subcategory": subcat_name,
                    "fields": field_result.get("results", [])
                })
                
                # Collect recommended fields
                for f in field_result.get("results", [])[:2]:
                    exploration_results["recommended_fields"].append({
                        "field_id": f["field_id"],
                        "category": cat_name,
                        "subcategory": subcat_name,
                        "usage": f.get("suggested_usage", "")
                    })
            
            exploration_results["categories_explored"].append(cat_exploration)
        
        # Generate patterns from recommended fields
        for f in exploration_results["recommended_fields"][:5]:
            exploration_results["recommended_patterns"].append({
                "pattern": f["usage"].replace("FIELD", f["field_id"]),
                "source_field": f["field_id"],
                "category": f["category"]
            })
        
        logger.info(
            f"[HierarchicalRAG] Exploration complete | "
            f"categories={len(exploration_results['categories_explored'])} "
            f"fields={len(exploration_results['recommended_fields'])}"
        )
        
        return exploration_results
    
    async def query_with_field_adaptation(
        self,
        dataset_id: str,
        fields: List[Dict],
        region: str = None,
        max_patterns: int = 8,
        max_pitfalls: int = 10
    ) -> RAGResult:
        """
        Enhanced RAG query that adapts patterns to actual available fields.
        
        This is the main entry point for field-aware RAG retrieval.
        Combines:
        1. Stored patterns from knowledge base
        2. Dynamically generated patterns based on actual fields
        
        Args:
            dataset_id: Dataset identifier
            fields: List of available field dictionaries
            region: Optional region filter
            max_patterns: Maximum patterns to return
            max_pitfalls: Maximum pitfalls to return
        
        Returns:
            RAGResult with both stored and adaptive patterns
        """
        # 1. Get standard RAG results
        base_result = await self.query(
            dataset_id=dataset_id,
            region=region,
            max_patterns=max_patterns // 2,  # Reserve space for adaptive patterns
            max_pitfalls=max_pitfalls
        )
        
        # 2. Analyze actual fields
        field_types = self.analyze_field_types(fields)
        
        logger.debug(
            f"[RAGService] Field types detected: "
            f"sentiment={len(field_types['sentiment'])} "
            f"count={len(field_types['count'])} "
            f"text={len(field_types['text_derived'])} "
            f"numeric={len(field_types['numeric'])}"
        )
        
        # 3. Generate adaptive patterns
        adaptive_patterns = self.generate_adaptive_patterns(
            field_types=field_types,
            dataset_id=dataset_id,
            category=base_result.category
        )
        
        # 4. Merge patterns (adaptive patterns first, they're more relevant)
        merged_patterns = adaptive_patterns[:max_patterns // 2] + base_result.patterns[:max_patterns // 2]
        
        # 5. Return enhanced result
        return RAGResult(
            patterns=merged_patterns,
            pitfalls=base_result.pitfalls,
            dataset_info=base_result.dataset_info,
            category=base_result.category,
            region=region
        )
