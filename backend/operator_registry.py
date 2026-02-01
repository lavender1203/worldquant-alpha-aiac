"""
Dynamic Operator Registry

Provides dynamic operator classification without hardcoding.

Key principles:
1. Load operators from database (not hardcoded)
2. Classify operators by name patterns (auto-detect category)
3. Fallback to minimal assumptions
4. Learn from usage patterns over time
"""

import re
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class OperatorInfo:
    """Information about an operator."""
    name: str
    category: str = "unknown"
    description: str = ""
    usage_count: int = 0
    success_count: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.5
        return self.success_count / self.usage_count


class DynamicOperatorRegistry:
    """
    Dynamic operator registry that learns from database and usage.
    
    Instead of hardcoding operator lists, we:
    1. Load from database
    2. Auto-classify by name patterns
    3. Learn from actual usage
    """
    
    # Name patterns for auto-classification (minimal assumptions)
    # These are patterns, not hardcoded lists
    CATEGORY_PATTERNS = {
        "time_series": [
            r"^ts_",           # ts_* operators
            r"_delay$",        # *_delay operators
            r"_past$",         # *_past operators
        ],
        "cross_sectional": [
            r"^rank$",
            r"^zscore$", 
            r"^scale$",
            r"^percentile",
        ],
        "group": [
            r"^group_",        # group_* operators
        ],
        "vector": [
            r"^vec_",          # vec_* operators
        ],
        "math": [
            r"^log",
            r"^sqrt",
            r"^abs$",
            r"^sign$",
            r"^power$",
            r"^exp$",
        ],
        "arithmetic": [
            r"^add$",
            r"^subtract$",
            r"^multiply$",
            r"^divide$",
        ],
        "control": [
            r"^if_",
            r"^trade_when",
            r"^filter",
            r"pasteurize",
            r"^clamp",
            r"^tail$",
        ],
    }
    
    # Functional categories (for layering suggestions)
    # NOT prescriptive - just for organization
    FUNCTIONAL_HINTS = {
        "signal_extraction": [r"^ts_delta", r"^ts_returns", r"^ts_diff", r"^ts_corr", r"^ts_std"],
        "standardization": [r"^rank", r"^zscore", r"^scale", r"^ts_rank", r"^ts_zscore", r"^group_rank", r"^group_zscore"],
        "smoothing": [r"^ts_decay", r"^ts_mean", r"^hump", r"^ts_av"],
    }
    
    def __init__(self):
        self._operators: Dict[str, OperatorInfo] = {}
        self._by_category: Dict[str, Set[str]] = defaultdict(set)
        self._by_function: Dict[str, Set[str]] = defaultdict(set)
        self._loaded = False
    
    async def load_from_db(self, db=None) -> bool:
        """
        Load operators from database.
        
        This is the ONLY source of truth for operators.
        """
        try:
            if db is None:
                from backend.database import AsyncSessionLocal
                async with AsyncSessionLocal() as db:
                    return await self._load_impl(db)
            else:
                return await self._load_impl(db)
        except Exception as e:
            logger.error(f"[OperatorRegistry] Failed to load from DB: {e}")
            return False
    
    async def _load_impl(self, db) -> bool:
        """Internal load implementation."""
        from sqlalchemy import select
        from backend.models import Operator
        
        result = await db.execute(select(Operator.name, Operator.category, Operator.description))
        rows = result.all()
        
        if not rows:
            logger.warning("[OperatorRegistry] No operators in database")
            return False
        
        self._operators.clear()
        self._by_category.clear()
        self._by_function.clear()
        
        for name, db_category, description in rows:
            name_lower = name.lower()
            
            # Use DB category if available, else auto-classify
            category = (db_category or "").lower() or self._auto_classify(name_lower)
            
            info = OperatorInfo(
                name=name_lower,
                category=category,
                description=description or ""
            )
            self._operators[name_lower] = info
            self._by_category[category].add(name_lower)
            
            # Also classify by function
            for func, patterns in self.FUNCTIONAL_HINTS.items():
                for pattern in patterns:
                    if re.search(pattern, name_lower):
                        self._by_function[func].add(name_lower)
                        break
        
        self._loaded = True
        logger.info(f"[OperatorRegistry] Loaded {len(self._operators)} operators from DB")
        return True
    
    def _auto_classify(self, name: str) -> str:
        """Auto-classify operator by name pattern."""
        name_lower = name.lower()
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    return category
        
        return "other"
    
    @property
    def all_operators(self) -> Set[str]:
        """Get all known operators."""
        return set(self._operators.keys())
    
    def get_by_category(self, category: str) -> Set[str]:
        """Get operators by category."""
        return self._by_category.get(category.lower(), set())
    
    def get_by_function(self, function: str) -> Set[str]:
        """Get operators by functional role (for layering hints)."""
        return self._by_function.get(function.lower(), set())
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self._by_category.keys())
    
    def is_valid_operator(self, name: str) -> bool:
        """Check if an operator exists."""
        return name.lower() in self._operators
    
    def get_info(self, name: str) -> Optional[OperatorInfo]:
        """Get operator info."""
        return self._operators.get(name.lower())
    
    def record_usage(self, name: str, success: bool = False):
        """Record operator usage for learning."""
        name_lower = name.lower()
        if name_lower in self._operators:
            self._operators[name_lower].usage_count += 1
            if success:
                self._operators[name_lower].success_count += 1
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for analysis."""
        if not self._operators:
            return {}
        
        # Sort by success rate
        sorted_ops = sorted(
            self._operators.values(),
            key=lambda x: (x.success_rate, x.usage_count),
            reverse=True
        )
        
        return {
            "total_operators": len(self._operators),
            "by_category": {cat: len(ops) for cat, ops in self._by_category.items()},
            "top_success": [
                {"name": op.name, "success_rate": op.success_rate, "usage": op.usage_count}
                for op in sorted_ops[:10] if op.usage_count > 0
            ],
            "underused": [op.name for op in sorted_ops if op.usage_count < 3][:20]
        }
    
    def suggest_operators(self, context: str = "", num: int = 5) -> List[str]:
        """
        Suggest operators based on context.
        
        NOT prescriptive - just surfaces options for the LLM to consider.
        """
        if not self._operators:
            return []
        
        # If context mentions certain patterns, suggest relevant operators
        context_lower = context.lower()
        suggestions = set()
        
        if "group" in context_lower or "sector" in context_lower:
            suggestions.update(list(self._by_category.get("group", set()))[:3])
        
        if "time" in context_lower or "momentum" in context_lower:
            ts_ops = self._by_category.get("time_series", set())
            suggestions.update(list(ts_ops)[:3])
        
        if "rank" in context_lower or "compare" in context_lower:
            suggestions.update(list(self._by_category.get("cross_sectional", set()))[:3])
        
        # Fill with random underused operators to encourage exploration
        if len(suggestions) < num:
            underused = [
                op.name for op in self._operators.values()
                if op.usage_count < 5
            ]
            import random
            random.shuffle(underused)
            suggestions.update(underused[:num - len(suggestions)])
        
        return list(suggestions)[:num]


# Global instance
_registry = DynamicOperatorRegistry()


def get_operator_registry() -> DynamicOperatorRegistry:
    """Get the global operator registry."""
    return _registry


async def ensure_operators_loaded(db=None) -> bool:
    """Ensure operators are loaded from DB."""
    if not _registry._loaded:
        return await _registry.load_from_db(db)
    return True


def get_operators_by_function(function: str) -> Set[str]:
    """Get operators by functional role (convenience function)."""
    return _registry.get_by_function(function)


def get_all_operators() -> Set[str]:
    """Get all known operators (convenience function)."""
    return _registry.all_operators


def is_valid_operator(name: str) -> bool:
    """Check if operator exists (convenience function)."""
    return _registry.is_valid_operator(name)
