"""
Genetic Programming Optimizer - Systematic Mutation Search for Alpha Improvement

Features:
1. Population-based optimization with selection, mutation, crossover
2. Multi-objective fitness (Sharpe, Fitness, Turnover, Novelty)
3. Adaptive mutation rates based on improvement trajectory
4. Diversity maintenance through niching
5. Efficient batch simulation

This module performs systematic exploration of the alpha expression space
to find high-quality variants of promising alphas.
"""

import re
import random
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from loguru import logger


# =============================================================================
# Configuration
# =============================================================================

# Operator substitution groups (semantically similar)
OPERATOR_GROUPS = {
    "rank_normalize": ["rank", "ts_rank", "ts_zscore", "zscore", "quantile"],
    "aggregation": ["ts_mean", "ts_median", "ts_sum", "ts_decay_linear"],
    "volatility": ["ts_std_dev", "ts_kurtosis", "ts_skewness"],
    "change": ["ts_delta", "ts_returns", "ts_av_diff", "ts_max_diff"],
    "extrema": ["ts_max", "ts_min", "ts_argmax", "ts_argmin"],
    "correlation": ["ts_corr", "ts_cov", "ts_covariance"],
    "group_ops": ["group_rank", "group_zscore", "group_mean", "group_neutralize"],
    "math": ["log", "sqrt", "abs", "sign", "sigmoid", "tanh"],
    "vector": ["vec_sum", "vec_avg", "vec_max", "vec_min", "vec_count"],
}

# Window values for mutation
WINDOW_VALUES = [5, 10, 20, 22, 40, 44, 60, 66, 120, 126, 252]

# Decay values
DECAY_VALUES = [0, 2, 4, 6, 8, 12, 16]

# Common wrapper patterns
WRAPPER_PATTERNS = [
    ("rank", "rank({})"),
    ("ts_rank", "ts_rank({}, 20)"),
    ("ts_zscore", "ts_zscore({}, 60)"),
    ("ts_decay_linear", "ts_decay_linear({}, 10)"),
    ("group_neutralize", "group_neutralize({}, sector)"),
    ("abs", "abs({})"),
    ("sign", "sign({})"),
]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Individual:
    """Represents an alpha expression individual in the population."""
    expression: str
    generation: int = 0
    parent_expression: str = ""
    
    # Fitness metrics
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    os_sharpe: float = 0.0
    
    # Derived scores
    overall_fitness: float = 0.0
    novelty_score: float = 0.0
    
    # Metadata
    alpha_id: str = ""
    raw_result: Dict[str, Any] = field(default_factory=dict)
    mutation_type: str = ""
    mutation_description: str = ""
    simulated: bool = False
    passed: bool = False
    
    @property
    def fingerprint(self) -> str:
        """Unique fingerprint for deduplication."""
        return hashlib.md5(self.expression.encode()).hexdigest()[:12]
    
    def calculate_fitness(self, weights: Dict[str, float] = None):
        """Calculate overall fitness from component metrics."""
        w = weights or {
            "sharpe": 0.50,
            "fitness": 0.20,
            "turnover": 0.15,  # Negative weight (lower is better)
            "os_sharpe": 0.15,
        }
        
        # Normalize components
        sharpe_score = min(1.0, self.sharpe / 2.0) if self.sharpe > 0 else 0
        fitness_score = min(1.0, self.fitness / 1.5) if self.fitness > 0 else 0
        turnover_score = max(0, 1.0 - self.turnover) if self.turnover < 1.0 else 0
        os_score = min(1.0, self.os_sharpe / 1.5) if self.os_sharpe > 0 else 0
        
        self.overall_fitness = (
            w["sharpe"] * sharpe_score +
            w["fitness"] * fitness_score +
            w["turnover"] * turnover_score +
            w["os_sharpe"] * os_score
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.expression,
            "generation": self.generation,
            "sharpe": round(self.sharpe, 4),
            "fitness": round(self.fitness, 4),
            "turnover": round(self.turnover, 4),
            "os_sharpe": round(self.os_sharpe, 4),
            "overall_fitness": round(self.overall_fitness, 4),
            "alpha_id": self.alpha_id,
            "mutation_type": self.mutation_type,
            "mutation_description": self.mutation_description,
            "passed": self.passed,
        }


@dataclass
class Population:
    """Collection of individuals with diversity management."""
    individuals: List[Individual] = field(default_factory=list)
    generation: int = 0
    
    # Tracking
    fingerprints: Set[str] = field(default_factory=set)
    best_fitness_history: List[float] = field(default_factory=list)
    
    def add(self, individual: Individual) -> bool:
        """Add individual if not duplicate. Returns True if added."""
        if individual.fingerprint in self.fingerprints:
            return False
        
        self.individuals.append(individual)
        self.fingerprints.add(individual.fingerprint)
        return True
    
    def get_best(self, n: int = 1) -> List[Individual]:
        """Get top N individuals by overall fitness."""
        sorted_pop = sorted(
            self.individuals,
            key=lambda x: x.overall_fitness,
            reverse=True
        )
        return sorted_pop[:n]
    
    def get_passed(self) -> List[Individual]:
        """Get individuals that passed quality threshold."""
        return [i for i in self.individuals if i.passed]
    
    def stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        if not self.individuals:
            return {"size": 0}
        
        fitness_values = [i.overall_fitness for i in self.individuals if i.simulated]
        
        return {
            "size": len(self.individuals),
            "simulated": sum(1 for i in self.individuals if i.simulated),
            "passed": len(self.get_passed()),
            "avg_fitness": sum(fitness_values) / len(fitness_values) if fitness_values else 0,
            "max_fitness": max(fitness_values) if fitness_values else 0,
            "generation": self.generation,
        }


@dataclass
class OptimizationConfig:
    """Configuration for genetic optimization."""
    population_size: int = 50
    generations: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    elite_ratio: float = 0.1
    tournament_size: int = 3
    
    # Thresholds for passing
    sharpe_threshold: float = 1.25
    fitness_threshold: float = 1.0
    turnover_threshold: float = 0.7
    
    # Simulation budget
    max_simulations: int = 100


# =============================================================================
# Mutation Operators
# =============================================================================

def mutate_operator_substitution(expression: str) -> Tuple[str, str]:
    """
    Substitute an operator with a semantically similar one.
    
    Returns:
        (mutated_expression, description)
    """
    # Find all function calls
    func_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    matches = list(func_pattern.finditer(expression))
    
    if not matches:
        return expression, "no_change"
    
    # Pick random function to mutate
    match = random.choice(matches)
    func_name = match.group(1).lower()
    
    # Find operator group
    for group_name, operators in OPERATOR_GROUPS.items():
        if func_name in operators:
            # Pick different operator from same group
            alternatives = [op for op in operators if op != func_name]
            if alternatives:
                new_op = random.choice(alternatives)
                mutated = expression[:match.start(1)] + new_op + expression[match.end(1):]
                return mutated, f"operator_sub: {func_name} -> {new_op}"
    
    return expression, "no_substitution_found"


def mutate_window_parameter(expression: str) -> Tuple[str, str]:
    """
    Mutate window parameter values.
    
    Returns:
        (mutated_expression, description)
    """
    # Pattern: function(field, NUMBER)
    window_pattern = re.compile(r'(ts_\w+|group_\w+)\s*\(\s*([^,]+)\s*,\s*(\d+)')
    matches = list(window_pattern.finditer(expression))
    
    if not matches:
        return expression, "no_window_params"
    
    # Pick random window to mutate
    match = random.choice(matches)
    func_name = match.group(1)
    original_window = int(match.group(3))
    
    # Pick new window value
    new_window = random.choice([w for w in WINDOW_VALUES if w != original_window])
    
    mutated = expression[:match.start(3)] + str(new_window) + expression[match.end(3):]
    return mutated, f"window: {func_name} {original_window} -> {new_window}"


def mutate_add_wrapper(expression: str) -> Tuple[str, str]:
    """
    Add a wrapper function around the expression.
    
    Returns:
        (mutated_expression, description)
    """
    wrapper_name, pattern = random.choice(WRAPPER_PATTERNS)
    
    # Don't double-wrap with same function
    if expression.startswith(f"{wrapper_name}("):
        return expression, "already_wrapped"
    
    mutated = pattern.format(expression)
    return mutated, f"add_wrapper: {wrapper_name}"


def mutate_remove_wrapper(expression: str) -> Tuple[str, str]:
    """
    Remove outermost wrapper function.
    
    Returns:
        (mutated_expression, description)
    """
    # Check for function wrapper pattern
    wrapper_pattern = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(.+)\s*\)$', expression.strip())
    
    if wrapper_pattern:
        wrapper = wrapper_pattern.group(1)
        inner = wrapper_pattern.group(2)
        
        # Don't remove if inner has unbalanced parens
        if inner.count('(') == inner.count(')'):
            return inner, f"remove_wrapper: {wrapper}"
    
    return expression, "no_wrapper_to_remove"


def mutate_sign_flip(expression: str) -> Tuple[str, str]:
    """
    Flip the sign of the expression.
    
    Returns:
        (mutated_expression, description)
    """
    if expression.startswith("-1 * ") or expression.startswith("-1*"):
        # Remove negative
        mutated = expression.replace("-1 * ", "", 1).replace("-1*", "", 1)
        return mutated, "remove_negative"
    elif expression.startswith("-(") and expression.endswith(")"):
        # Remove negation wrapper
        return expression[2:-1], "remove_negation"
    else:
        # Add negative
        return f"-1 * ({expression})", "add_negative"


def mutate_structure_modification(expression: str) -> Tuple[str, str]:
    """
    Modify expression structure (e.g., add neutralization).
    
    Returns:
        (mutated_expression, description)
    """
    modifications = [
        (f"group_neutralize({expression}, sector)", "add sector neutralization"),
        (f"group_neutralize({expression}, industry)", "add industry neutralization"),
        (f"ts_decay_linear({expression}, 5)", "add short decay"),
        (f"ts_decay_linear({expression}, 10)", "add medium decay"),
        (f"pasteurize({expression})", "add pasteurize"),
    ]
    
    # Filter out already present modifications
    filtered = []
    for mod_expr, desc in modifications:
        key = desc.split()[1] if len(desc.split()) > 1 else desc
        if key not in expression.lower():
            filtered.append((mod_expr, desc))
    
    if filtered:
        mutated, desc = random.choice(filtered)
        return mutated, f"structure: {desc}"
    
    return expression, "no_structure_change"


# =============================================================================
# Crossover Operators
# =============================================================================

def crossover_swap_inner(expr1: str, expr2: str) -> Tuple[str, str]:
    """
    Swap inner expressions between two alphas.
    
    Returns:
        (child1, child2)
    """
    # Extract outer function and inner expression
    pattern = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(.+)\s*\)$', expr1.strip())
    if not pattern:
        return expr1, expr2
    
    outer1, inner1 = pattern.groups()
    
    pattern2 = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(.+)\s*\)$', expr2.strip())
    if not pattern2:
        return expr1, expr2
    
    outer2, inner2 = pattern2.groups()
    
    # Swap inners
    child1 = f"{outer1}({inner2})"
    child2 = f"{outer2}({inner1})"
    
    return child1, child2


def crossover_combine(expr1: str, expr2: str) -> str:
    """
    Combine two expressions with an arithmetic operator.
    
    Returns:
        Combined expression
    """
    operators = [
        ("add", f"add({expr1}, {expr2})"),
        ("multiply", f"multiply({expr1}, {expr2})"),
        ("average", f"divide(add({expr1}, {expr2}), 2)"),
    ]
    
    _, combined = random.choice(operators)
    return combined


# =============================================================================
# Genetic Optimizer
# =============================================================================

class GeneticOptimizer:
    """
    Genetic programming optimizer for alpha expressions.
    
    Usage:
        optimizer = GeneticOptimizer(config)
        
        # Initialize with seed expression
        optimizer.initialize(seed_expression, seed_metrics)
        
        # Evolve population
        for gen in range(config.generations):
            # Get individuals to simulate
            candidates = optimizer.get_simulation_candidates(batch_size=10)
            
            # Simulate and update
            for ind, result in zip(candidates, simulation_results):
                optimizer.update_individual(ind, result)
            
            # Evolve to next generation
            optimizer.evolve()
        
        # Get best results
        best = optimizer.get_best_individuals(n=5)
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.population = Population()
        self.all_fingerprints: Set[str] = set()  # Global dedup
        self.simulations_used = 0
        
        # Adaptive mutation rates
        self.mutation_rates = {
            "operator_sub": 0.25,
            "window": 0.25,
            "add_wrapper": 0.15,
            "remove_wrapper": 0.10,
            "sign_flip": 0.10,
            "structure": 0.15,
        }
        
        # Tracking
        self.generation_stats: List[Dict] = []
    
    def initialize(
        self,
        seed_expression: str,
        seed_metrics: Dict[str, float] = None
    ):
        """
        Initialize population with seed expression and its mutations.
        
        Args:
            seed_expression: Starting alpha expression
            seed_metrics: Optional metrics from seed simulation
        """
        self.population = Population()
        self.all_fingerprints.clear()
        
        # Add seed as first individual
        seed = Individual(
            expression=seed_expression,
            generation=0,
            mutation_type="seed",
            mutation_description="original",
        )
        
        if seed_metrics:
            seed.sharpe = seed_metrics.get("sharpe", 0)
            seed.fitness = seed_metrics.get("fitness", 0)
            seed.turnover = seed_metrics.get("turnover", 0)
            seed.os_sharpe = seed_metrics.get("os_sharpe", 0)
            seed.calculate_fitness()
            seed.simulated = True
        
        self.population.add(seed)
        self.all_fingerprints.add(seed.fingerprint)
        
        # Generate initial mutations
        self._generate_initial_mutations(seed_expression)
        
        logger.info(
            f"[GeneticOpt] Initialized | population={len(self.population.individuals)} "
            f"seed_fitness={seed.overall_fitness:.3f}"
        )
    
    def _generate_initial_mutations(self, seed: str, count: int = None):
        """Generate initial population through mutations."""
        target_count = count or self.config.population_size
        
        mutation_funcs = [
            mutate_operator_substitution,
            mutate_window_parameter,
            mutate_add_wrapper,
            mutate_remove_wrapper,
            mutate_sign_flip,
            mutate_structure_modification,
        ]
        
        attempts = 0
        max_attempts = target_count * 3
        
        while len(self.population.individuals) < target_count and attempts < max_attempts:
            # Apply random mutation
            mutation_func = random.choice(mutation_funcs)
            mutated, description = mutation_func(seed)
            
            if mutated != seed and "no_" not in description:
                ind = Individual(
                    expression=mutated,
                    generation=0,
                    parent_expression=seed,
                    mutation_type=mutation_func.__name__.replace("mutate_", ""),
                    mutation_description=description,
                )
                
                if ind.fingerprint not in self.all_fingerprints:
                    self.population.add(ind)
                    self.all_fingerprints.add(ind.fingerprint)
            
            attempts += 1
    
    def get_simulation_candidates(self, batch_size: int = 10) -> List[Individual]:
        """
        Get unsimulated individuals for batch simulation.
        
        Returns individuals prioritized by expected quality.
        """
        unsimulated = [i for i in self.population.individuals if not i.simulated]
        
        # Prioritize by mutation type (some mutations are more likely to succeed)
        priority_order = ["window", "operator_sub", "sign_flip", "add_wrapper", "structure"]
        
        def priority_key(ind: Individual) -> int:
            try:
                return priority_order.index(ind.mutation_type)
            except ValueError:
                return len(priority_order)
        
        unsimulated.sort(key=priority_key)
        
        return unsimulated[:batch_size]
    
    def update_individual(
        self,
        individual: Individual,
        sim_result: Dict[str, Any]
    ):
        """
        Update individual with simulation results.
        
        Args:
            individual: Individual to update
            sim_result: Simulation result dict
        """
        # Extract metrics
        is_stats = sim_result.get("is", sim_result.get("train", {})) or {}
        os_stats = sim_result.get("os", sim_result.get("test", {})) or {}
        
        individual.alpha_id = sim_result.get("alpha_id") or ""
        individual.raw_result = sim_result
        individual.sharpe = float(is_stats.get("sharpe", is_stats.get("Sharpe", 0)) or 0)
        individual.fitness = float(is_stats.get("fitness", is_stats.get("Fitness", 0)) or 0)
        individual.turnover = float(is_stats.get("turnover", is_stats.get("Turnover", 0)) or 0)
        individual.os_sharpe = float(os_stats.get("sharpe", os_stats.get("Sharpe", 0)) or 0)
        
        individual.calculate_fitness()
        individual.simulated = True
        
        # Check if passed thresholds
        individual.passed = (
            individual.sharpe >= self.config.sharpe_threshold and
            individual.fitness >= self.config.fitness_threshold and
            individual.turnover <= self.config.turnover_threshold
        )
        
        self.simulations_used += 1
    
    def evolve(self):
        """
        Evolve population to next generation.
        
        1. Select parents (tournament selection)
        2. Create offspring through mutation and crossover
        3. Add elite individuals
        4. Maintain diversity
        """
        self.population.generation += 1
        gen = self.population.generation
        
        # Record stats before evolution
        stats = self.population.stats()
        self.generation_stats.append(stats)
        
        # Get simulated individuals
        simulated = [i for i in self.population.individuals if i.simulated]
        
        if not simulated:
            logger.warning("[GeneticOpt] No simulated individuals to evolve from")
            return
        
        # Sort by fitness
        simulated.sort(key=lambda x: x.overall_fitness, reverse=True)
        
        # New population
        new_individuals: List[Individual] = []
        
        # Elite preservation
        elite_count = max(1, int(len(simulated) * self.config.elite_ratio))
        elites = simulated[:elite_count]
        for elite in elites:
            new_individuals.append(elite)
        
        # Generate offspring
        target_size = self.config.population_size - len(new_individuals)
        
        while len(new_individuals) < self.config.population_size:
            # Tournament selection
            parent = self._tournament_select(simulated)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                offspring = self._mutate(parent.expression, gen)
                if offspring and offspring.fingerprint not in self.all_fingerprints:
                    new_individuals.append(offspring)
                    self.all_fingerprints.add(offspring.fingerprint)
            
            # Crossover
            if random.random() < self.config.crossover_rate and len(simulated) > 1:
                parent2 = self._tournament_select(simulated)
                offspring = self._crossover(parent, parent2, gen)
                if offspring and offspring.fingerprint not in self.all_fingerprints:
                    new_individuals.append(offspring)
                    self.all_fingerprints.add(offspring.fingerprint)
        
        # Update population
        self.population.individuals = new_individuals
        self.population.fingerprints = {i.fingerprint for i in new_individuals}
        
        logger.info(
            f"[GeneticOpt] Generation {gen} | population={len(new_individuals)} "
            f"elite={elite_count} best_fitness={stats['max_fitness']:.3f}"
        )
    
    def _tournament_select(self, candidates: List[Individual]) -> Individual:
        """Select individual through tournament selection."""
        tournament = random.sample(
            candidates,
            min(self.config.tournament_size, len(candidates))
        )
        return max(tournament, key=lambda x: x.overall_fitness)
    
    def _mutate(self, expression: str, generation: int) -> Optional[Individual]:
        """Apply random mutation to expression."""
        mutation_funcs = [
            (mutate_operator_substitution, self.mutation_rates["operator_sub"]),
            (mutate_window_parameter, self.mutation_rates["window"]),
            (mutate_add_wrapper, self.mutation_rates["add_wrapper"]),
            (mutate_remove_wrapper, self.mutation_rates["remove_wrapper"]),
            (mutate_sign_flip, self.mutation_rates["sign_flip"]),
            (mutate_structure_modification, self.mutation_rates["structure"]),
        ]
        
        # Weighted random selection
        total_weight = sum(w for _, w in mutation_funcs)
        r = random.random() * total_weight
        
        cumulative = 0
        selected_func = mutation_funcs[0][0]
        for func, weight in mutation_funcs:
            cumulative += weight
            if r <= cumulative:
                selected_func = func
                break
        
        mutated, description = selected_func(expression)
        
        if mutated == expression or "no_" in description:
            return None
        
        return Individual(
            expression=mutated,
            generation=generation,
            parent_expression=expression,
            mutation_type=selected_func.__name__.replace("mutate_", ""),
            mutation_description=description,
        )
    
    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int
    ) -> Optional[Individual]:
        """Create offspring through crossover."""
        child1, child2 = crossover_swap_inner(
            parent1.expression,
            parent2.expression
        )
        
        # Pick the child that's more different from parents
        if child1 != parent1.expression and child1 != parent2.expression:
            return Individual(
                expression=child1,
                generation=generation,
                parent_expression=parent1.expression,
                mutation_type="crossover",
                mutation_description=f"swap_inner with {parent2.fingerprint[:6]}",
            )
        
        return None
    
    def get_best_individuals(self, n: int = 5) -> List[Individual]:
        """Get top N individuals by fitness."""
        return self.population.get_best(n)
    
    def get_passed_individuals(self) -> List[Individual]:
        """Get all individuals that passed quality thresholds."""
        return self.population.get_passed()
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        return {
            "generations": self.population.generation,
            "simulations_used": self.simulations_used,
            "population_stats": self.population.stats(),
            "generation_history": self.generation_stats,
            "best_individuals": [i.to_dict() for i in self.get_best_individuals(5)],
            "passed_count": len(self.get_passed_individuals()),
            "mutation_rates": self.mutation_rates,
        }
    
    def adapt_mutation_rates(self):
        """
        Adapt mutation rates based on success history.
        
        Increase rates for mutations that led to improvements.
        """
        if len(self.generation_stats) < 2:
            return
        
        # Check which mutation types led to improvements
        passed = self.get_passed_individuals()
        
        mutation_success = defaultdict(int)
        mutation_total = defaultdict(int)
        
        for ind in self.population.individuals:
            if ind.simulated:
                mutation_total[ind.mutation_type] += 1
                if ind.passed or ind.overall_fitness > 0.5:
                    mutation_success[ind.mutation_type] += 1
        
        # Update rates
        base_rate = 1.0 / len(self.mutation_rates)
        
        for mut_type in self.mutation_rates:
            total = mutation_total.get(mut_type, 0)
            success = mutation_success.get(mut_type, 0)
            
            if total > 3:
                success_rate = success / total
                # Adjust rate towards success rate
                current = self.mutation_rates[mut_type]
                self.mutation_rates[mut_type] = 0.7 * current + 0.3 * max(0.05, success_rate)


# =============================================================================
# Helper Functions
# =============================================================================

async def run_genetic_optimization(
    seed_expression: str,
    seed_metrics: Dict[str, float],
    simulate_func,  # async function(expression) -> Dict
    config: OptimizationConfig = None,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "INDUSTRY",
) -> Dict[str, Any]:
    """
    Run complete genetic optimization on a seed expression.
    
    Args:
        seed_expression: Starting alpha expression
        seed_metrics: Metrics from seed simulation
        simulate_func: Async function to simulate an expression
        config: Optimization configuration
        region, universe, delay, decay, neutralization: Simulation settings
    
    Returns:
        Optimization result dictionary
    """
    config = config or OptimizationConfig()
    optimizer = GeneticOptimizer(config)
    
    # Initialize
    optimizer.initialize(seed_expression, seed_metrics)
    
    # Evolution loop
    for gen in range(config.generations):
        # Get candidates to simulate
        candidates = optimizer.get_simulation_candidates(batch_size=10)
        
        if not candidates:
            logger.info(f"[GeneticOpt] No more candidates at generation {gen}")
            break
        
        # Check budget
        if optimizer.simulations_used >= config.max_simulations:
            logger.info(f"[GeneticOpt] Simulation budget exhausted at {optimizer.simulations_used}")
            break
        
        # Simulate candidates
        for ind in candidates:
            try:
                result = await simulate_func(
                    expression=ind.expression,
                    region=region,
                    universe=universe,
                    delay=delay,
                    decay=decay,
                    neutralization=neutralization,
                )
                
                if result.get("success"):
                    optimizer.update_individual(ind, result)
                else:
                    ind.simulated = True  # Mark as tried
                
            except Exception as e:
                logger.warning(f"[GeneticOpt] Simulation failed: {e}")
                ind.simulated = True
        
        # Evolve
        optimizer.evolve()
        
        # Adapt mutation rates
        optimizer.adapt_mutation_rates()
    
    # Generate report
    report = optimizer.get_optimization_report()
    
    # Add best variants for downstream use
    best = optimizer.get_best_individuals(10)
    report["best_expressions"] = [i.expression for i in best]
    
    passed = optimizer.get_passed_individuals()
    report["passed_expressions"] = [i.expression for i in passed]
    report["passed_individuals"] = [i.to_dict() for i in passed]
    
    logger.info(
        f"[GeneticOpt] Complete | generations={report['generations']} "
        f"simulations={report['simulations_used']} passed={report['passed_count']}"
    )
    
    return report
