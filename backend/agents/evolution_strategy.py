"""
Evolution Strategy Module

Provides a unified abstraction for mining strategy management across the evolution loop.
This module bridges StrategyAgent output with the actual mining execution.

Design Principles:
1. Immutable Strategy Objects: Strategies are value objects, not modified in place
2. Clear State Machine: Strategy transitions are explicit and traceable
3. Separation of Concerns: Strategy generation vs Strategy application
4. Testability: Pure functions where possible
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Dict, Any, Optional, Protocol
from enum import Enum
import json


class StrategyMode(Enum):
    """Strategy modes that determine exploration/exploitation balance."""
    EXPLORE = "explore"       # High diversity, novel approaches
    EXPLOIT = "exploit"       # Refine successful patterns
    BALANCED = "balanced"     # Mix of both
    OPTIMIZE = "optimize"     # Focus on improving existing alphas
    RESCUE = "rescue"         # Emergency mode after repeated failures


@dataclass(frozen=True)
class EvolutionStrategy:
    """
    Immutable strategy object that guides a mining iteration.
    
    This is the single source of truth for all strategy parameters.
    Created by StrategyAgent, consumed by MiningWorkflow.
    """
    # Core parameters
    mode: StrategyMode = StrategyMode.BALANCED
    temperature: float = 0.7
    exploration_weight: float = 0.5
    
    # Field guidance
    preferred_fields: tuple = field(default_factory=tuple)
    avoid_fields: tuple = field(default_factory=tuple)
    screened_fields: tuple = field(default_factory=tuple)  # From FieldScreener
    
    # Hypothesis guidance
    focus_hypotheses: tuple = field(default_factory=tuple)
    avoid_patterns: tuple = field(default_factory=tuple)
    amplify_patterns: tuple = field(default_factory=tuple)
    
    # Operator guidance
    preferred_operators: tuple = field(default_factory=tuple)
    avoid_operators: tuple = field(default_factory=tuple)
    
    # Optimization targets (Chain-of-Alpha style)
    optimization_targets: tuple = field(default_factory=tuple)
    
    # Metadata
    action_summary: str = ""
    reasoning: str = ""
    iteration: int = 0
    
    def with_updates(self, **kwargs) -> EvolutionStrategy:
        """Create new strategy with specified updates (immutable pattern)."""
        return replace(self, **kwargs)
    
    def to_prompt_context(self) -> Dict[str, Any]:
        """Convert strategy to prompt context dictionary."""
        return {
            "exploration_weight": self.exploration_weight,
            "preferred_fields": list(self.preferred_fields),
            "avoid_fields": list(self.avoid_fields),
            "focus_hypotheses": list(self.focus_hypotheses),
            "avoid_patterns": list(self.avoid_patterns),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/persistence."""
        return {
            "mode": self.mode.value,
            "temperature": self.temperature,
            "exploration_weight": self.exploration_weight,
            "preferred_fields": list(self.preferred_fields),
            "avoid_fields": list(self.avoid_fields),
            "screened_fields": list(self.screened_fields),
            "focus_hypotheses": list(self.focus_hypotheses),
            "avoid_patterns": list(self.avoid_patterns),
            "amplify_patterns": list(self.amplify_patterns),
            "preferred_operators": list(self.preferred_operators),
            "avoid_operators": list(self.avoid_operators),
            "optimization_targets": list(self.optimization_targets),
            "action_summary": self.action_summary,
            "reasoning": self.reasoning,
            "iteration": self.iteration,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvolutionStrategy:
        """Deserialize from dictionary."""
        mode_str = data.get("mode", "balanced")
        try:
            mode = StrategyMode(mode_str)
        except ValueError:
            mode = StrategyMode.BALANCED
        
        return cls(
            mode=mode,
            temperature=data.get("temperature", 0.7),
            exploration_weight=data.get("exploration_weight", 0.5),
            preferred_fields=tuple(data.get("preferred_fields", [])),
            avoid_fields=tuple(data.get("avoid_fields", [])),
            screened_fields=tuple(data.get("screened_fields", [])),
            focus_hypotheses=tuple(data.get("focus_hypotheses", [])),
            avoid_patterns=tuple(data.get("avoid_patterns", [])),
            amplify_patterns=tuple(data.get("amplify_patterns", [])),
            preferred_operators=tuple(data.get("preferred_operators", [])),
            avoid_operators=tuple(data.get("avoid_operators", [])),
            optimization_targets=tuple(data.get("optimization_targets", [])),
            action_summary=data.get("action_summary", ""),
            reasoning=data.get("reasoning", ""),
            iteration=data.get("iteration", 0),
        )
    
    @classmethod
    def default(cls) -> EvolutionStrategy:
        """Create default balanced strategy."""
        return cls(
            mode=StrategyMode.BALANCED,
            action_summary="Default balanced strategy",
            reasoning="Initial iteration with no prior data"
        )
    
    @classmethod
    def explore_mode(cls, iteration: int = 0) -> EvolutionStrategy:
        """Create high-exploration strategy."""
        return cls(
            mode=StrategyMode.EXPLORE,
            temperature=0.9,
            exploration_weight=0.8,
            action_summary="Exploration mode: seeking novel approaches",
            reasoning="Prioritizing diversity over refinement",
            iteration=iteration,
        )
    
    @classmethod
    def exploit_mode(cls, successful_patterns: List[str], iteration: int = 0) -> EvolutionStrategy:
        """Create exploitation strategy based on successful patterns."""
        return cls(
            mode=StrategyMode.EXPLOIT,
            temperature=0.5,
            exploration_weight=0.2,
            amplify_patterns=tuple(successful_patterns[:5]),
            action_summary="Exploitation mode: refining successful patterns",
            reasoning="Building on patterns that have shown promise",
            iteration=iteration,
        )
    
    @classmethod
    def rescue_mode(cls, problematic_fields: List[str], iteration: int = 0) -> EvolutionStrategy:
        """Create rescue strategy after repeated failures."""
        return cls(
            mode=StrategyMode.RESCUE,
            temperature=1.0,
            exploration_weight=0.95,
            avoid_fields=tuple(problematic_fields[:10]),
            action_summary="Rescue mode: breaking out of failure pattern",
            reasoning="Multiple failures detected, drastically changing approach",
            iteration=iteration,
        )


@dataclass
class RoundResult:
    """Results from a single mining round, used to generate next strategy."""
    iteration: int
    total_generated: int = 0
    total_simulated: int = 0
    passed_count: int = 0
    failed_count: int = 0
    
    # Quality metrics (from passed alphas)
    best_sharpe: Optional[float] = None
    best_abs_sharpe: Optional[float] = None
    avg_sharpe: Optional[float] = None
    best_fitness: Optional[float] = None
    best_abs_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    best_rn_sharpe: Optional[float] = None
    best_rn_fitness: Optional[float] = None
    avg_turnover: Optional[float] = None
    
    # Failure analysis
    syntax_errors: int = 0
    simulation_errors: int = 0
    quality_failures: int = 0
    
    # Identified patterns
    successful_patterns: List[str] = field(default_factory=list)
    problematic_fields: List[str] = field(default_factory=list)
    problematic_operators: List[str] = field(default_factory=list)
    
    # Optimization candidates
    optimization_candidates: List[Dict] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.total_generated
        return self.passed_count / max(total, 1)
    
    @property
    def simulation_rate(self) -> float:
        """Calculate simulation success rate."""
        return self.total_simulated / max(self.total_generated, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "iteration": self.iteration,
            "total_generated": self.total_generated,
            "total_simulated": self.total_simulated,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "success_rate": round(self.success_rate, 3),
            "best_sharpe": self.best_sharpe,
            "best_abs_sharpe": self.best_abs_sharpe,
            "avg_sharpe": round(self.avg_sharpe, 3) if self.avg_sharpe else None,
            "best_fitness": self.best_fitness,
            "best_abs_fitness": self.best_abs_fitness,
            "best_rn_sharpe": self.best_rn_sharpe,
            "best_rn_fitness": self.best_rn_fitness,
            "syntax_errors": self.syntax_errors,
            "simulation_errors": self.simulation_errors,
            "quality_failures": self.quality_failures,
        }


class StrategyTransitionProtocol(Protocol):
    """Protocol for strategy transition logic (dependency injection)."""
    
    def compute_next_strategy(
        self,
        current_strategy: EvolutionStrategy,
        round_result: RoundResult,
        cumulative_success: int,
        target_goal: int,
        max_iterations: int
    ) -> EvolutionStrategy:
        """Compute next strategy based on round results."""
        ...


class RuleBasedTransition:
    """
    Rule-based strategy transition (fallback when LLM unavailable).
    
    Implements clear, deterministic rules for strategy evolution.
    """
    
    # Thresholds for strategy transitions
    EXPLORE_THRESHOLD = 0.1   # Below this success rate -> explore more
    EXPLOIT_THRESHOLD = 0.5   # Above this success rate -> exploit more
    RESCUE_THRESHOLD = 0      # Zero success -> rescue mode
    
    def compute_next_strategy(
        self,
        current_strategy: EvolutionStrategy,
        round_result: RoundResult,
        cumulative_success: int,
        target_goal: int,
        max_iterations: int
    ) -> EvolutionStrategy:
        """Compute next strategy using deterministic rules."""
        
        success_rate = round_result.success_rate
        progress = cumulative_success / max(target_goal, 1)
        remaining = max_iterations - round_result.iteration
        
        # Determine mode
        if success_rate == self.RESCUE_THRESHOLD and round_result.total_generated > 0:
            mode = StrategyMode.RESCUE
            temperature = 1.0
            exploration_weight = 0.95
            action = "Rescue: zero success, drastically changing approach"
        elif success_rate < self.EXPLORE_THRESHOLD:
            mode = StrategyMode.EXPLORE
            temperature = min(1.0, 0.7 + 0.1 * round_result.iteration)
            exploration_weight = min(0.9, 0.5 + 0.1 * round_result.iteration)
            action = f"Explore: low success rate ({success_rate:.1%})"
        elif success_rate > self.EXPLOIT_THRESHOLD:
            mode = StrategyMode.EXPLOIT
            temperature = max(0.3, 0.7 - 0.1 * round_result.passed_count)
            exploration_weight = 0.3
            action = f"Exploit: good success rate ({success_rate:.1%})"
        else:
            mode = StrategyMode.BALANCED
            temperature = 0.7
            exploration_weight = 0.5
            action = "Balanced: moderate success rate"
        
        # Urgency adjustment: if behind schedule, increase exploration
        expected_progress = round_result.iteration / max_iterations
        if progress < expected_progress * 0.7 and remaining > 1:
            exploration_weight = min(1.0, exploration_weight + 0.2)
            action += " [urgency boost]"
        
        # Build avoidance lists
        avoid_fields = tuple(round_result.problematic_fields[:5])
        avoid_operators = tuple(round_result.problematic_operators[:3])
        
        # Build amplification list from successful patterns
        amplify = tuple(round_result.successful_patterns[:5])
        
        # Identify optimization candidates
        opt_targets = tuple(
            c.get("expression", "") 
            for c in round_result.optimization_candidates[:3]
        )
        
        return EvolutionStrategy(
            mode=mode,
            temperature=temperature,
            exploration_weight=exploration_weight,
            avoid_fields=avoid_fields,
            avoid_operators=avoid_operators,
            amplify_patterns=amplify,
            optimization_targets=opt_targets,
            action_summary=action,
            reasoning=f"Success: {success_rate:.1%}, Progress: {progress:.1%}",
            iteration=round_result.iteration + 1,
        )


def merge_strategies(
    base: EvolutionStrategy,
    llm_strategy: Optional[Dict],
    rule_strategy: EvolutionStrategy
) -> EvolutionStrategy:
    """
    Merge LLM-generated strategy with rule-based fallback.
    
    LLM provides creativity, rules provide guardrails.
    """
    if not llm_strategy:
        return rule_strategy
    
    # Extract LLM suggestions with validation
    next_strat = llm_strategy.get("strategy", llm_strategy.get("next_strategy", {}))
    
    # Use LLM values where provided and valid, else fall back to rules
    temperature = next_strat.get("temperature")
    if temperature is None or not (0.0 <= temperature <= 1.0):
        temperature = rule_strategy.temperature
    
    exploration_weight = next_strat.get("exploration_weight")
    if exploration_weight is None or not (0.0 <= exploration_weight <= 1.0):
        exploration_weight = rule_strategy.exploration_weight
    
    # Merge lists (combine LLM suggestions with rule-based avoidances)
    preferred_fields = tuple(
        next_strat.get("preferred_fields", [])
    )
    avoid_fields = tuple(set(
        list(next_strat.get("avoid_fields", [])) +
        list(rule_strategy.avoid_fields)
    ))
    
    focus_hypotheses = tuple(
        next_strat.get("focus_hypotheses", [])
    )
    avoid_patterns = tuple(
        next_strat.get("avoid_patterns", [])
    )
    amplify_patterns = tuple(
        next_strat.get("amplify_patterns", []) or
        list(rule_strategy.amplify_patterns)
    )
    preferred_operators = tuple(
        next_strat.get("preferred_operators", []) or
        list(rule_strategy.preferred_operators)
    )
    avoid_operators = tuple(set(
        list(next_strat.get("avoid_operators", [])) +
        list(rule_strategy.avoid_operators)
    ))
    
    # Extract optimization targets
    opt_targets_raw = llm_strategy.get("optimization_targets", [])
    opt_targets = tuple(
        t.get("expression", t) if isinstance(t, dict) else t
        for t in opt_targets_raw[:5]
    )
    
    return EvolutionStrategy(
        mode=rule_strategy.mode,  # Mode from rules (more reliable)
        temperature=temperature,
        exploration_weight=exploration_weight,
        preferred_fields=preferred_fields,
        avoid_fields=avoid_fields,
        focus_hypotheses=focus_hypotheses,
        avoid_patterns=avoid_patterns,
        amplify_patterns=amplify_patterns,
        preferred_operators=preferred_operators,
        avoid_operators=avoid_operators,
        optimization_targets=opt_targets or rule_strategy.optimization_targets,
        action_summary=next_strat.get("action_summary", rule_strategy.action_summary),
        reasoning=next_strat.get("reasoning", rule_strategy.reasoning),
        iteration=rule_strategy.iteration,
    )
