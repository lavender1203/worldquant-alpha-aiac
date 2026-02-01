"""
Adaptive Strategy Evolver

IMPORTANT DESIGN PRINCIPLES:
1. NO hardcoded operator lists - learn from actual successes
2. Strategies are DISCOVERED from data, not prescribed
3. Exploration encouraged, not constrained
4. Minimal assumptions about what works

This module:
1. Tracks which operator patterns succeed/fail
2. Uses bandit algorithms to balance exploration/exploitation
3. Discovers new strategies from successful alphas
4. Provides soft suggestions without constraining creativity
"""

import re
import math
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from loguru import logger


@dataclass
class DiscoveredPattern:
    """A pattern discovered from successful alphas."""
    pattern_id: str
    operator_sequence: List[str]  # Extracted from successful alphas
    example_expression: str
    
    # Performance stats
    times_seen: int = 0
    successes: int = 0
    total_sharpe: float = 0.0
    
    # Metadata
    discovered_at: str = ""
    source: str = "discovered"  # "discovered" or "user_provided"
    
    @property
    def success_rate(self) -> float:
        if self.times_seen == 0:
            return 0.5
        return self.successes / self.times_seen
    
    @property
    def avg_sharpe(self) -> float:
        if self.successes == 0:
            return 0.0
        return self.total_sharpe / self.successes
    
    def ucb1_score(self, total: int, c: float = 2.0) -> float:
        """UCB1 score for exploration."""
        if self.times_seen == 0:
            return float('inf')
        return self.success_rate + c * math.sqrt(math.log(total + 1) / self.times_seen)


@dataclass
class DatasetLearning:
    """Learning state for a dataset type."""
    dataset_type: str
    patterns: Dict[str, DiscoveredPattern] = field(default_factory=dict)
    total_trials: int = 0
    
    # Raw stats
    all_operators_seen: Set[str] = field(default_factory=set)
    operator_success_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    operator_usage_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class StrategyEvolver:
    """
    Learns strategies from actual alpha performance.
    
    Key difference from previous version:
    - NO hardcoded base strategies
    - Everything is learned from data
    - Minimal assumptions
    """
    
    def __init__(self, state_path: Optional[str] = None):
        self.state_path = Path(state_path) if state_path else None
        self.dataset_learning: Dict[str, DatasetLearning] = {}
        
        # Exploration parameters
        self.exploration_weight = 2.0
        self.min_exploration_rate = 0.15  # 15% random exploration
        
        # Pattern extraction
        self.min_successes_to_suggest = 2  # Need at least 2 successes
        self.max_patterns_to_show = 3
        
        self._load_state()
    
    def _load_state(self):
        """Load saved state if available."""
        if self.state_path and self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                    self._deserialize(data)
                logger.info(f"[StrategyEvolver] Loaded state from {self.state_path}")
            except Exception as e:
                logger.warning(f"[StrategyEvolver] Failed to load state: {e}")
    
    def _save_state(self):
        """Save state to disk."""
        if not self.state_path:
            return
        
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, 'w') as f:
                json.dump(self._serialize(), f, indent=2)
        except Exception as e:
            logger.error(f"[StrategyEvolver] Failed to save: {e}")
    
    def _serialize(self) -> Dict:
        """Serialize state."""
        return {
            "version": "2.0",  # New version - no hardcoded strategies
            "saved_at": datetime.now().isoformat(),
            "dataset_learning": {
                dtype: {
                    "dataset_type": learning.dataset_type,
                    "total_trials": learning.total_trials,
                    "all_operators_seen": list(learning.all_operators_seen),
                    "operator_success_count": dict(learning.operator_success_count),
                    "operator_usage_count": dict(learning.operator_usage_count),
                    "patterns": {
                        pid: {
                            "pattern_id": p.pattern_id,
                            "operator_sequence": p.operator_sequence,
                            "example_expression": p.example_expression,
                            "times_seen": p.times_seen,
                            "successes": p.successes,
                            "total_sharpe": p.total_sharpe,
                            "discovered_at": p.discovered_at,
                            "source": p.source,
                        }
                        for pid, p in learning.patterns.items()
                    }
                }
                for dtype, learning in self.dataset_learning.items()
            }
        }
    
    def _deserialize(self, data: Dict):
        """Deserialize state."""
        for dtype, ldata in data.get("dataset_learning", {}).items():
            learning = DatasetLearning(
                dataset_type=dtype,
                total_trials=ldata.get("total_trials", 0),
                all_operators_seen=set(ldata.get("all_operators_seen", [])),
                operator_success_count=defaultdict(int, ldata.get("operator_success_count", {})),
                operator_usage_count=defaultdict(int, ldata.get("operator_usage_count", {})),
            )
            
            for pid, pdata in ldata.get("patterns", {}).items():
                learning.patterns[pid] = DiscoveredPattern(
                    pattern_id=pdata["pattern_id"],
                    operator_sequence=pdata["operator_sequence"],
                    example_expression=pdata["example_expression"],
                    times_seen=pdata.get("times_seen", 0),
                    successes=pdata.get("successes", 0),
                    total_sharpe=pdata.get("total_sharpe", 0.0),
                    discovered_at=pdata.get("discovered_at", ""),
                    source=pdata.get("source", "discovered"),
                )
            
            self.dataset_learning[dtype] = learning
    
    # =========================================================================
    # LEARNING: Record feedback from simulations
    # =========================================================================
    
    def record_feedback(self, dataset_type: str, expression: str,
                        fitness: float, sharpe: float, turnover: float,
                        success: bool, metadata: Optional[Dict] = None):
        """
        Record feedback from a simulation.
        
        This is where we LEARN - extracting patterns from successes.
        """
        # Get or create learning state
        if dataset_type not in self.dataset_learning:
            self.dataset_learning[dataset_type] = DatasetLearning(dataset_type=dataset_type)
        
        learning = self.dataset_learning[dataset_type]
        learning.total_trials += 1
        
        # Extract operators from expression
        operators = self._extract_operators(expression)
        
        # Update operator stats
        for op in operators:
            learning.all_operators_seen.add(op)
            learning.operator_usage_count[op] += 1
            if success:
                learning.operator_success_count[op] += 1
        
        # If successful, create/update pattern
        if success and len(operators) >= 2:
            pattern_key = "->".join(operators[:5])  # Use first 5 ops as key
            
            if pattern_key in learning.patterns:
                pattern = learning.patterns[pattern_key]
                pattern.times_seen += 1
                pattern.successes += 1
                pattern.total_sharpe += sharpe
            else:
                # Discover new pattern!
                learning.patterns[pattern_key] = DiscoveredPattern(
                    pattern_id=pattern_key,
                    operator_sequence=operators[:5],
                    example_expression=expression[:200],  # Truncate
                    times_seen=1,
                    successes=1,
                    total_sharpe=sharpe,
                    discovered_at=datetime.now().isoformat(),
                    source="discovered"
                )
                logger.info(f"[StrategyEvolver] Discovered new pattern: {pattern_key}")
        
        elif not success and len(operators) >= 2:
            # Track failed patterns too
            pattern_key = "->".join(operators[:5])
            if pattern_key in learning.patterns:
                learning.patterns[pattern_key].times_seen += 1
        
        # Save periodically
        if learning.total_trials % 10 == 0:
            self._save_state()
    
    def _extract_operators(self, expression: str) -> List[str]:
        """Extract operator names from expression."""
        if not expression:
            return []
        
        # Find function calls
        pattern = re.compile(r'([a-z_][a-z0-9_]*)\s*\(', re.IGNORECASE)
        matches = pattern.findall(expression.lower())
        
        # Filter to likely operators (not field names)
        # We don't hardcode - just use heuristics
        operators = []
        for m in matches:
            # Operators typically start with common prefixes
            if any(m.startswith(p) for p in ['ts_', 'group_', 'vec_', 'rank', 'zscore', 'scale']):
                operators.append(m)
            elif m in {'log', 'sqrt', 'abs', 'sign', 'add', 'subtract', 'multiply', 'divide', 
                       'if_else', 'clamp', 'tail', 'bucket', 'hump', 'pasteurize'}:
                operators.append(m)
        
        return operators
    
    # =========================================================================
    # SUGGESTION: Provide soft guidance based on learnings
    # =========================================================================
    
    def get_strategy_prompt(self, dataset_type: str, num_strategies: int = 3,
                            exploration_boost: float = 0.0) -> str:
        """
        Generate suggestions based on learned patterns.
        
        IMPORTANT: These are suggestions for inspiration, NOT rules.
        """
        learning = self.dataset_learning.get(dataset_type)
        
        # No data yet - encourage exploration
        if not learning or learning.total_trials < 5:
            return f"""
## Learning Mode

No performance data yet for this data type. 
This is an opportunity to explore freely and help the system learn!

Try diverse operator combinations. Every attempt teaches us something.
"""
        
        # High exploration = minimal suggestions
        if exploration_boost > 0.5:
            return """
## Exploration Mode

Explore freely! Try novel operator combinations.
The system is in exploration mode - creativity is encouraged.
"""
        
        # Get top patterns by UCB1 score
        patterns = list(learning.patterns.values())
        
        # Filter to patterns with enough data
        qualified = [p for p in patterns if p.successes >= self.min_successes_to_suggest]
        
        if not qualified:
            # Not enough successes yet
            return f"""
## Early Learning Phase

{learning.total_trials} trials recorded, still gathering data.
Continue experimenting - patterns will emerge from successes.
"""
        
        # Sort by UCB1 for exploration-exploitation balance
        qualified.sort(key=lambda p: p.ucb1_score(learning.total_trials), reverse=True)
        top_patterns = qualified[:self.max_patterns_to_show]
        
        # Build suggestion text
        lines = [
            "\n## Patterns That Have Worked (for inspiration)",
            "",
            f"Based on {learning.total_trials} trials, these patterns showed promise:",
            ""
        ]
        
        for i, p in enumerate(top_patterns, 1):
            rate = int(p.success_rate * 100)
            lines.append(f"**{i}. {' -> '.join(p.operator_sequence[:3])}**")
            lines.append(f"   Success: {rate}% ({p.successes}/{p.times_seen}), Avg Sharpe: {p.avg_sharpe:.2f}")
            lines.append("")
        
        # Strong emphasis on freedom
        lines.extend([
            "---",
            "",
            "**These are observations, NOT prescriptions.**",
            "",
            "You should:",
            "- Invent entirely new approaches",
            "- Challenge whether past patterns will work in the future",
            "- Trust your economic reasoning about the data",
            "",
            "The best discoveries often come from trying something unexpected.",
        ])
        
        return "\n".join(lines)
    
    def get_exploration_guidance(self, dataset_type: str) -> str:
        """Generate exploration suggestions."""
        learning = self.dataset_learning.get(dataset_type)
        
        if not learning:
            return ""
        
        # Find underexplored operators
        all_ops = list(learning.all_operators_seen)
        underused = [
            op for op in all_ops
            if learning.operator_usage_count.get(op, 0) < 3
        ]
        
        if not underused:
            return ""
        
        # Suggest a few underused operators
        random.shuffle(underused)
        suggestions = underused[:3]
        
        return f"""
## Exploration Opportunity

These operators have been seen but rarely used:
{', '.join(suggestions)}

Consider experimenting with them (or any other operators you find interesting).
"""
    
    def get_statistics(self, dataset_type: Optional[str] = None) -> Dict:
        """Get learning statistics."""
        if dataset_type:
            learning = self.dataset_learning.get(dataset_type)
            if not learning:
                return {"message": "No data for this dataset type"}
            
            return {
                "total_trials": learning.total_trials,
                "patterns_discovered": len(learning.patterns),
                "operators_seen": len(learning.all_operators_seen),
                "top_patterns": [
                    {
                        "pattern": "->".join(p.operator_sequence[:3]),
                        "success_rate": f"{p.success_rate*100:.0f}%",
                        "trials": p.times_seen,
                    }
                    for p in sorted(learning.patterns.values(), 
                                    key=lambda x: x.success_rate, reverse=True)[:5]
                    if p.times_seen > 0
                ]
            }
        
        # All dataset types
        return {
            dtype: self.get_statistics(dtype)
            for dtype in self.dataset_learning.keys()
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_default_state_path = Path(__file__).parent / "data" / "strategy_evolver_state.json"
_evolver = StrategyEvolver(state_path=str(_default_state_path))


def get_strategy_evolver() -> StrategyEvolver:
    """Get the global strategy evolver instance."""
    return _evolver
