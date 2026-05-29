"""
Mining Agent - High-level Entry Point for Alpha Mining

This module provides:
1. Backward-compatible interface (run_mining_iteration)
2. Evolution loop with actual strategy application
3. Integration with LangGraph workflow and optimization chain

Design Principles:
1. Strategy flows through the entire pipeline (not just recorded)
2. Clear separation between orchestration and execution
3. Explicit state transitions with full traceability
4. Graceful degradation (rule-based fallback when LLM fails)
"""

from typing import List, Dict, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
from datetime import datetime, timedelta
import json, time, os  # #region agent log

def _debug_log(hypo_id, location, message, data=None):
    try:
        log_path = r"e:\AIACV2_v1.2\worldquant-alpha-aiac\.cursor\debug.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        entry = {"hypothesisId": hypo_id, "location": location, "message": message, "data": data or {}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}
        with open(log_path, "a", encoding="utf-8") as f: f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except: pass
# #endregion

from backend.models import MiningTask, Alpha, AlphaFailure
from backend.agents.graph import MiningWorkflow, create_mining_graph
from backend.agents.services import LLMService, get_llm_service
from backend.agents.services.trace_service import TraceService
from backend.agents.strategy_agent import StrategyAgent, create_strategy_agent
from backend.agents.evolution_strategy import (
    EvolutionStrategy, StrategyMode, RoundResult, 
    RuleBasedTransition, merge_strategies
)
from backend.agents.feedback_agent import FeedbackAgent
from backend.adapters.brain_adapter import BrainAdapter
from backend.adapters.mcp_brain_adapter import MCPBrainAdapter


class MiningAgent:
    """
    Mining Agent - Orchestrates the alpha mining process.
    
    Key Responsibilities:
    1. Manage evolution loop across multiple rounds
    2. Ensure strategy is propagated to all pipeline stages
    3. Coordinate feedback learning and knowledge accumulation
    4. Handle failures gracefully with automatic recovery
    
    Usage:
        agent = MiningAgent(db, brain)
        result = await agent.run_evolution_loop(task, dataset_id, fields, operators)
    """
    
    def __init__(
        self,
        db: AsyncSession,
        brain_adapter: BrainAdapter = None,
        llm_service: LLMService = None
    ):
        """
        Initialize MiningAgent with dependencies.
        
        Args:
            db: Async SQLAlchemy session for persistence
            brain_adapter: BRAIN platform adapter for simulation
            llm_service: LLM service for generation and analysis
        """
        self.db = db
        self.brain = brain_adapter or MCPBrainAdapter()
        self.llm_service = llm_service or get_llm_service()
        
        # Create LangGraph workflow
        self._workflow = create_mining_graph(
            db=db,
            brain=self.brain,
            llm_service=self.llm_service
        )
        
        # Create Strategy Agent for intelligent planning
        self._strategy_agent = create_strategy_agent(llm_service=self.llm_service)
        
        # Rule-based transition for fallback
        self._rule_transition = RuleBasedTransition()
        
        # Feedback Agent for knowledge accumulation
        self._feedback_agent = FeedbackAgent(db)
        
        logger.info("[MiningAgent] Initialized with strategy-aware pipeline")
    
    async def run_mining_iteration(
        self,
        task: MiningTask,
        dataset_id: str,
        fields: List[Dict],
        operators: List[Dict],
        num_alphas: int = 3,
        iteration: int = 1,
        strategy: Optional[EvolutionStrategy] = None,
        run_id: Optional[int] = None,
    ) -> List[Alpha]:
        """
        Run a single mining iteration with strategy application.
        
        Args:
            task: Mining task instance
            dataset_id: Dataset to mine
            fields: Available data fields
            operators: Available operators
            num_alphas: Target number of alphas
            iteration: Current iteration number
            strategy: Evolution strategy to apply (uses default if None)
            
        Returns:
            List of generated Alpha models (both passed and failed)
        """
        # Use default strategy if none provided
        if strategy is None:
            strategy = EvolutionStrategy.default()
        
        logger.info(
            f"[MiningAgent] Starting iteration {iteration} | "
            f"mode={strategy.mode.value} temp={strategy.temperature:.2f} "
            f"explore={strategy.exploration_weight:.2f}"
        )
        
        # Initialize TraceService
        trace_service = TraceService(self.db, task.id, iteration=iteration, run_id=run_id)
        
        try:
            # Run workflow with strategy context
            result = await self._workflow.run_with_persistence(
                task=task,
                dataset_id=dataset_id,
                fields=self._apply_field_filters(fields, strategy),
                operators=self._apply_operator_filters(operators, strategy),
                num_alphas=num_alphas,
                config={
                    "configurable": {
                        "trace_service": trace_service,
                        "rag_service": self._workflow.rag_service,
                        "strategy": strategy.to_dict(),  # Pass strategy to all nodes
                        "task_config": task.config or {},
                        "run_id": run_id,
                    }
                }
            )
            
            # Collect generated alphas from database
            generated_alphas = await self._collect_iteration_alphas(
                task.id, result.get("generated_alphas", [])
            )
            
            logger.info(
                f"[MiningAgent] Iteration {iteration} complete | "
                f"alphas={len(generated_alphas)} "
                f"failures={len(result.get('failures', []))}"
            )
            
            return generated_alphas
            
        except Exception as e:
            logger.error(f"[MiningAgent] Iteration {iteration} failed: {e}")
            raise
    
    def _apply_field_filters(
        self, 
        fields: List[Dict], 
        strategy: EvolutionStrategy
    ) -> List[Dict]:
        """
        Apply strategy-based field filtering.
        
        Prioritizes preferred fields, demotes avoided fields.
        """
        avoid_set = set(strategy.avoid_fields)
        preferred_set = set(strategy.preferred_fields)
        screened_set = set(strategy.screened_fields)
        
        # If we have screened fields, prioritize them
        if screened_set:
            # Put screened fields first, filter out avoided
            screened = [f for f in fields if f.get("id", f.get("name")) in screened_set]
            others = [
                f for f in fields 
                if f.get("id", f.get("name")) not in screened_set
                and f.get("id", f.get("name")) not in avoid_set
            ]
            return self._limit_fields_for_prompt(screened + self._rank_fields_for_mining(others), 30)
        
        # Otherwise, use preferred/avoid logic
        preferred = []
        neutral = []
        avoided = []
        
        for f in fields:
            field_id = f.get("id", f.get("name"))
            if field_id in avoid_set:
                avoided.append(f)
            elif field_id in preferred_set:
                preferred.append(f)
            else:
                neutral.append(f)
        
        # Preferred first, then ranked neutral fields. Avoided fields are excluded from prompts.
        return self._limit_fields_for_prompt(preferred + self._rank_fields_for_mining(neutral), 30)

    def _rank_fields_for_mining(self, fields: List[Dict]) -> List[Dict]:
        """Rank fields so prompts are not dominated by the dataset's storage order."""
        positive_keywords = (
            "return", "momentum", "volume", "liquidity", "volatility", "price",
            "earnings", "eps", "cash", "flow", "sales", "revenue", "profit",
            "margin", "growth", "value", "yield", "leverage", "debt", "asset",
            "estimate", "revision", "surprise", "short", "buyback", "dividend",
        )
        negative_keywords = (
            "governance", "board", "environment", "emission", "social",
            "responsibility", "sustainability", "policy", "compensation",
            "relations", "qsg", "asset4",
        )

        def score(field: Dict) -> int:
            text = " ".join(
                str(field.get(key) or "")
                for key in ("id", "name", "field_id", "field_name", "description")
            ).lower()
            value = 0
            value += sum(3 for keyword in positive_keywords if keyword in text)
            value -= sum(4 for keyword in negative_keywords if keyword in text)
            if "group" in text or "cluster" in text or "bucket" in text:
                value -= 10
            return value

        return sorted(fields, key=lambda f: score(f), reverse=True)

    def _limit_fields_for_prompt(self, fields: List[Dict], limit: int) -> List[Dict]:
        """Keep strong fields while preserving coverage across the full ranked list."""
        deduped = []
        seen = set()
        for field in fields:
            field_id = field.get("id", field.get("name"))
            if not field_id or field_id in seen:
                continue
            seen.add(field_id)
            deduped.append(field)

        if len(deduped) <= limit:
            return deduped

        top_count = max(0, limit - 8)
        selected = deduped[:top_count]
        tail = deduped[top_count:]
        if tail:
            step = max(1, len(tail) // 8)
            selected.extend(tail[i] for i in range(0, len(tail), step)[: limit - len(selected)])
        return selected[:limit]

    def _apply_diversity_guidance(
        self,
        strategy: EvolutionStrategy,
        diversity_tracker,
    ) -> EvolutionStrategy:
        """Fold DiversityTracker suggestions into the next prompt strategy."""
        if not diversity_tracker:
            return strategy

        try:
            suggestions = diversity_tracker.get_exploration_suggestions(n=3)
        except Exception as e:
            logger.debug(f"[MiningAgent] Diversity guidance skipped: {e}")
            return strategy

        if not suggestions:
            return strategy

        preferred_operators = list(strategy.preferred_operators)
        focus_hypotheses = list(strategy.focus_hypotheses)

        for suggestion in suggestions:
            items = list(suggestion.underexplored_items or [])[:5]
            if not items:
                continue
            if suggestion.dimension == "operator":
                preferred_operators.extend(items)
            else:
                focus_hypotheses.append(f"Diversity: {suggestion.suggestion} ({', '.join(items)})")

        if not preferred_operators and not focus_hypotheses:
            return strategy

        return strategy.with_updates(
            preferred_operators=tuple(dict.fromkeys(preferred_operators)),
            focus_hypotheses=tuple(dict.fromkeys(focus_hypotheses)),
            action_summary=f"{strategy.action_summary}; diversity-guided exploration",
            reasoning=f"{strategy.reasoning}\nDiversityTracker suggested underexplored directions.",
        )

    def _apply_operator_filters(
        self,
        operators: List[Dict],
        strategy: EvolutionStrategy,
    ) -> List[Dict]:
        """Prioritize preferred operators and exclude avoided operators."""
        preferred = {op.lower() for op in strategy.preferred_operators}
        avoided = {op.lower() for op in strategy.avoid_operators}

        if not preferred and not avoided:
            return operators

        preferred_ops = []
        neutral_ops = []
        for op in operators:
            name = str(op.get("name") or "").lower()
            if name in avoided:
                continue
            if name in preferred:
                preferred_ops.append(op)
            else:
                neutral_ops.append(op)

        return preferred_ops + neutral_ops

    def _extract_expression_components(
        self,
        expression: str,
        fields: List[Dict],
    ) -> tuple[List[str], List[str]]:
        """Extract field/operator usage for persistence and diversity tracking."""
        if not expression:
            return [], []

        try:
            from backend.alpha_semantic_validator import AlphaSemanticValidator

            validator = AlphaSemanticValidator(fields=fields, strict_field_check=False)
            validation = validator.validate(expression)
            return sorted(validation.used_fields), sorted(op.lower() for op in validation.used_operators)
        except Exception:
            import re

            ops = sorted({m.group(1).lower() for m in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expression)})
            field_ids = {str(f.get("id") or f.get("name") or "").lower() for f in fields}
            used_fields = sorted(
                {
                    token
                    for token in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", expression)
                    if token.lower() in field_ids
                }
            )
            return used_fields, ops

    async def _record_observability(
        self,
        task: MiningTask,
        dataset_id: str,
        fields: List[Dict],
        alphas: List[Alpha],
        round_result: RoundResult,
        strategy: EvolutionStrategy,
        iteration: int,
        diversity_tracker,
        metrics_tracker,
    ) -> None:
        """Record round-level metrics and exploration diversity."""
        if not diversity_tracker and not metrics_tracker:
            return

        try:
            round_metrics = None
            if metrics_tracker:
                round_metrics = metrics_tracker.create_round_metrics(
                    round_id=iteration,
                    dataset_id=dataset_id,
                    region=task.region,
                    strategy_mode=strategy.mode.value,
                )
                round_metrics.alphas_generated = round_result.total_generated
                round_metrics.alphas_passed = round_result.passed_count
                round_metrics.alphas_failed = round_result.failed_count
                round_metrics.alphas_optimized = len(round_result.optimization_candidates)
                round_metrics.simulation_count = round_result.total_simulated
                round_metrics.avg_sharpe = round_result.avg_sharpe or 0.0
                round_metrics.max_sharpe = round_result.best_sharpe or 0.0
                round_metrics.avg_fitness = round_result.avg_fitness or 0.0
                round_metrics.avg_turnover = round_result.avg_turnover or 0.0

            diversity_scores = []
            all_fields = set()
            all_operators = set()
            changed = False

            for alpha in alphas:
                metrics = alpha.metrics or alpha.is_metrics or {}
                fields_used, operators_used = self._extract_expression_components(alpha.expression, fields)

                if fields_used and not alpha.fields_used:
                    alpha.fields_used = fields_used
                    changed = True
                if operators_used and not alpha.operators_used:
                    alpha.operators_used = operators_used
                    changed = True

                all_fields.update(fields_used)
                all_operators.update(operators_used)

                if diversity_tracker:
                    score = diversity_tracker.evaluate_diversity(
                        dataset_id=dataset_id,
                        fields=fields_used,
                        operators=operators_used,
                        delay=alpha.delay or 1,
                        decay=alpha.decay or 0,
                        neutralization=alpha.neutralization or "NONE",
                    )
                    diversity_scores.append(score.overall_score)

                    from backend.diversity_tracker import ExplorationRecord

                    diversity_tracker.record_attempt(ExplorationRecord(
                        dataset_id=dataset_id,
                        region=task.region,
                        universe=task.universe,
                        fields_used=fields_used,
                        operators_used=operators_used,
                        operator_skeleton="->".join(operators_used[:5]),
                        delay=alpha.delay or 1,
                        decay=alpha.decay or 0,
                        neutralization=alpha.neutralization or "NONE",
                        was_successful=alpha.quality_status == "PASS",
                        sharpe=float(metrics.get("sharpe") or alpha.is_sharpe or 0),
                        timestamp=alpha.created_at or datetime.utcnow(),
                    ))

            if round_metrics:
                round_metrics.unique_fields = len(all_fields)
                round_metrics.unique_operators = len(all_operators)
                round_metrics.unique_datasets = 1
                if diversity_scores:
                    round_metrics.diversity_score = sum(diversity_scores) / len(diversity_scores)
                metrics_tracker.complete_round(round_metrics)

            if changed:
                await self.db.commit()
        except Exception as e:
            logger.warning(f"[MiningAgent] Observability recording failed: {e}")
            try:
                await self.db.rollback()
            except Exception:
                pass
    
    async def _collect_iteration_alphas(
        self, 
        task_id: str, 
        alpha_results: List[Any]
    ) -> List[Alpha]:
        """Collect persisted Alpha models for this iteration."""
        alphas = []
        
        for alpha_result in alpha_results:
            query = select(Alpha).where(
                Alpha.task_id == task_id,
                Alpha.expression == alpha_result.expression
            ).order_by(Alpha.id.desc()).limit(1)
            
            db_result = await self.db.execute(query)
            alpha = db_result.scalar_one_or_none()
            
            if alpha:
                alphas.append(alpha)
        
        return alphas
    
    async def run_evolution_loop(
        self,
        task: MiningTask,
        dataset_id: str,
        fields: List[Dict],
        operators: List[Dict],
        max_iterations: int = 10,
        target_alphas: int = 4,
        num_alphas_per_round: int = 4,
        initial_strategy: Optional[EvolutionStrategy] = None,
        run_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run multi-round evolution loop for alpha mining.
        
        This is the main entry point for production mining. It:
        1. Iterates through mining rounds until goal or max iterations
        2. Applies and evolves strategy based on results
        3. Triggers optimization chain for promising weak alphas
        4. Accumulates knowledge through feedback agent
        
        Args:
            task: Mining task instance
            dataset_id: Dataset to mine
            fields: Available data fields
            operators: Available operators
            max_iterations: Maximum mining rounds
            target_alphas: Target number of successful alphas
            num_alphas_per_round: Alphas to generate per round
            initial_strategy: Optional starting strategy
            
        Returns:
            Dict with complete evolution results
        """
        logger.info(
            f"[MiningAgent] Starting Evolution Loop | "
            f"task={task.id} dataset={dataset_id} "
            f"max_iter={max_iterations} target={target_alphas}"
        )
        # #region agent log
        _debug_log("B", "mining_agent.py:run_evolution_loop:start", "Evolution loop start", {"dataset_id": dataset_id, "fields_count": len(fields), "operators_count": len(operators), "target": target_alphas})
        loop_start_time = time.time()
        # #endregion
        
        # Initialize state
        iteration = 0
        total_success = 0
        all_alphas: List[Alpha] = []
        all_failures: List[Dict] = []
        strategy_history: List[EvolutionStrategy] = []
        
        # Start with provided, task-config-derived, or default strategy
        current_strategy = initial_strategy or self._initial_strategy_from_task_config(task)
        diversity_tracker = None
        metrics_tracker = None
        metrics_report = None

        try:
            from backend.diversity_tracker import DiversityTracker
            from backend.metrics_tracker import MetricsTracker

            diversity_tracker = DiversityTracker(self.db)
            await diversity_tracker.initialize(region=task.region)

            metrics_tracker = MetricsTracker(task_id=task.id, db=self.db)
            metrics_tracker.start_session(
                session_id=f"task_{task.id}_run_{run_id or datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            )
        except Exception as e:
            logger.warning(f"[MiningAgent] Observability modules unavailable: {e}")
        
        # Ensure Brain session is active and authenticated
        async with self.brain:
            while iteration < max_iterations:
                iteration += 1
                round_strategy = self._apply_diversity_guidance(current_strategy, diversity_tracker)
            
                logger.info(
                    f"[MiningAgent] === Round {iteration}/{max_iterations} === "
                    f"Strategy: {round_strategy.action_summary}"
                )
                # #region agent log
                round_start = time.time()
                _debug_log("A", f"mining_agent.py:round_{iteration}:start", f"Round {iteration} start", {"strategy_mode": current_strategy.mode.value, "temperature": current_strategy.temperature})
                # #endregion
                
                try:
                    # Execute mining iteration with current strategy
                    alphas = await self.run_mining_iteration(
                        task=task,
                        dataset_id=dataset_id,
                        fields=fields,
                        operators=operators,
                        num_alphas=num_alphas_per_round,
                        iteration=iteration,
                        strategy=round_strategy,
                        run_id=run_id,
                    )
                    
                    # Analyze round results
                    round_result = await self._analyze_round_results(
                        task_id=task.id,
                        alphas=alphas,
                        iteration=iteration
                    )
                    
                    # Update counters
                    total_success += round_result.passed_count
                    all_alphas.extend(alphas)
                    strategy_history.append(round_strategy)

                    await self._record_observability(
                        task=task,
                        dataset_id=dataset_id,
                        fields=fields,
                        alphas=alphas,
                        round_result=round_result,
                        strategy=round_strategy,
                        iteration=iteration,
                        diversity_tracker=diversity_tracker,
                        metrics_tracker=metrics_tracker,
                    )
                    # #region agent log
                    round_elapsed = time.time() - round_start
                    _debug_log("A", f"mining_agent.py:round_{iteration}:end", f"Round {iteration} complete", {
                        "elapsed_sec": round(round_elapsed, 2),
                        "generated": round_result.total_generated,
                        "simulated": round_result.total_simulated,
                        "passed": round_result.passed_count,
                        "failed": round_result.failed_count,
                        "syntax_errors": round_result.syntax_errors,
                        "simulation_errors": round_result.simulation_errors,
                        "quality_failures": round_result.quality_failures,
                        "best_sharpe": round_result.best_sharpe,
                        "cumulative_success": total_success
                    })
                    # #endregion
                    
                    logger.info(
                        f"[MiningAgent] Round {iteration} | "
                        f"passed={round_result.passed_count} "
                        f"total={total_success}/{target_alphas}"
                    )
                    
                    # Check termination: goal reached
                    if total_success >= target_alphas:
                        logger.info(
                            f"[MiningAgent] Goal reached! "
                            f"{total_success}/{target_alphas} in {iteration} rounds"
                        )
                        break
                    
                    # Check termination: task stopped externally
                    await self.db.refresh(task)
                    if task.status in ["STOPPED", "PAUSED"]:
                        logger.info(f"[MiningAgent] Task {task.status}, stopping")
                        break
                    
                    # === STRATEGY EVOLUTION ===
                    current_strategy = await self._evolve_strategy(
                        task_id=task.id,
                        current_strategy=round_strategy,
                        round_result=round_result,
                        cumulative_success=total_success,
                        target_goal=target_alphas,
                        max_iterations=max_iterations,
                        dataset_id=dataset_id,
                        region=task.region
                    )
                    
                    # === RECORD ROUND SUMMARY ===
                    await self._record_round_summary(
                        task=task,
                        iteration=iteration,
                        round_result=round_result,
                        strategy=current_strategy,
                        cumulative_success=total_success,
                        target_alphas=target_alphas,
                        run_id=run_id,
                    )
                    
                    # === FEEDBACK LEARNING ===
                    await self._run_feedback_learning(
                        task=task,
                        alphas=alphas,
                        round_result=round_result,
                        iteration=iteration,
                        dataset_id=dataset_id,
                        cumulative_success=total_success,
                        target_alphas=target_alphas,
                        max_iterations=max_iterations,
                    )
                    
                    # === OPTIMIZATION CHAIN (if applicable) ===
                    if round_result.optimization_candidates:
                        await self._run_optimization_chain(
                            task=task,
                            candidates=round_result.optimization_candidates,
                            strategy=current_strategy,
                            iteration=iteration,
                            dataset_id=dataset_id,
                            fields=fields,
                            run_id=run_id,
                        )
                    
                except Exception as e:
                    logger.error(f"[MiningAgent] Round {iteration} error: {e}")
                    # Rollback any failed transaction
                    try:
                        await self.db.rollback()
                    except Exception:
                        pass
                    # Create rescue strategy and continue
                    current_strategy = EvolutionStrategy.rescue_mode(
                        problematic_fields=list(current_strategy.avoid_fields),
                        iteration=iteration
                    )
                    continue

        if metrics_tracker:
            try:
                await metrics_tracker.snapshot_knowledge_metrics()
                metrics_tracker.end_session()
                metrics_report = metrics_tracker.generate_report()
            except Exception as e:
                logger.warning(f"[MiningAgent] Metrics finalization failed: {e}")
        
        # Final summary
        logger.info(
            f"[MiningAgent] Evolution Complete | "
            f"iterations={iteration} success={total_success}"
        )
        
        return {
            "iterations_completed": iteration,
            "total_success": total_success,
            "target_reached": total_success >= target_alphas,
            "all_alphas": all_alphas,
            "all_failures": all_failures,
            "strategy_history": [s.to_dict() for s in strategy_history],
            "final_strategy": current_strategy.to_dict(),
            "metrics_report": metrics_report,
        }

    def _initial_strategy_from_task_config(self, task: MiningTask) -> EvolutionStrategy:
        """Build initial strategy hints from task.config."""
        config = task.config or {}
        base = EvolutionStrategy.default()

        focus = []
        if config.get("target_hypothesis"):
            focus.append(str(config["target_hypothesis"]))
        if config.get("idea_style"):
            focus.append(f"Idea style: {config['idea_style']}")
        if config.get("constraints"):
            focus.extend(str(item) for item in config.get("constraints", [])[:8])

        preferred_fields = tuple(config.get("preferred_fields", []) or [])
        avoid_fields = tuple(config.get("avoid_fields", []) or [])
        preferred_operators = tuple(config.get("preferred_operators", []) or [])
        avoid_operators = tuple(config.get("avoid_operators", []) or [])

        if not focus and not preferred_fields and not avoid_fields and not preferred_operators and not avoid_operators:
            return base

        return base.with_updates(
            focus_hypotheses=tuple(focus),
            preferred_fields=preferred_fields,
            avoid_fields=avoid_fields,
            preferred_operators=preferred_operators,
            avoid_operators=avoid_operators,
            action_summary="Task-config guided strategy",
            reasoning="Initial strategy was built from MiningTask.config constraints.",
        )
    
    async def _analyze_round_results(
        self,
        task_id: str,
        alphas: List[Alpha],
        iteration: int
    ) -> RoundResult:
        """
        Analyze results from a mining round to inform next strategy.
        
        Extracts metrics, identifies patterns, and flags optimization candidates.
        """
        result = RoundResult(iteration=iteration)
        result.total_generated = len(alphas)
        
        # Separate passed and failed
        passed = [a for a in alphas if getattr(a, "quality_status", None) == "PASS"]
        failed = [a for a in alphas if getattr(a, "quality_status", None) != "PASS"]
        
        result.passed_count = len(passed)
        result.failed_count = len(failed)
        
        # Count simulated (passed + quality failures)
        result.total_simulated = len(passed) + len([
            a for a in failed 
            if self._alpha_was_simulated(a)
        ])
        
        # Extract metrics from passed alphas
        if passed:
            sharpes = []
            fitnesses = []
            turnovers = []
            
            for a in passed:
                metrics = getattr(a, "metrics", {}) or {}
                if isinstance(metrics, dict):
                    if metrics.get("sharpe") is not None:
                        sharpes.append(metrics["sharpe"])
                    if metrics.get("fitness") is not None:
                        fitnesses.append(metrics["fitness"])
                    if metrics.get("turnover") is not None:
                        turnovers.append(metrics["turnover"])
            
            if sharpes:
                result.best_sharpe = max(sharpes)
                result.avg_sharpe = sum(sharpes) / len(sharpes)
            if fitnesses:
                result.best_fitness = max(fitnesses)
                result.avg_fitness = sum(fitnesses) / len(fitnesses)
            if turnovers:
                result.avg_turnover = sum(turnovers) / len(turnovers)
        
        # Query recent failures for analysis
        failures = await self._query_recent_failures(task_id)
        
        # Analyze failure patterns
        problematic_fields = {}
        for f in failures:
            err_msg = f.get("error_message", "") or ""
            err_type = f.get("error_type", "")
            
            # Count error types
            if "syntax" in err_msg.lower() or err_type == "SYNTAX_ERROR":
                result.syntax_errors += 1
            elif "simulation" in err_msg.lower() or err_type == "SIMULATION_ERROR":
                result.simulation_errors += 1
            elif err_type == "QUALITY_CHECK_FAILED":
                result.quality_failures += 1
            
            # Extract problematic fields
            import re
            field_match = re.search(r"field[:\s]+['\"]?(\w+)['\"]?", err_msg.lower())
            if field_match:
                fname = field_match.group(1)
                problematic_fields[fname] = problematic_fields.get(fname, 0) + 1
        
        result.problematic_fields = sorted(
            problematic_fields.keys(),
            key=lambda x: problematic_fields[x],
            reverse=True
        )[:5]
        
        # Identify optimization candidates (weak but promising)
        result.optimization_candidates = await self._identify_optimization_candidates(
            alphas=failed,
            task_id=task_id
        )
        
        return result
    
    async def _query_recent_failures(self, task_id: str) -> List[Dict]:
        """Query recent failure records for analysis."""
        query = select(AlphaFailure).where(
            AlphaFailure.task_id == task_id,
            AlphaFailure.created_at >= datetime.utcnow() - timedelta(minutes=10),
            AlphaFailure.is_analyzed == False
        )
        res = await self.db.execute(query)
        failures = res.scalars().all()
        
        return [
            {
                "expression": f.expression,
                "error_message": f.error_message,
                "error_type": f.error_type
            }
            for f in failures
        ]
    
    async def _identify_optimization_candidates(
        self,
        alphas: List[Alpha],
        task_id: str
    ) -> List[Dict]:
        """
        Identify weak alphas that are worth optimizing.
        
        Criteria (from alpha_scoring.should_optimize):
        - Positive but below threshold
        - Risk-neutralized significantly better than raw
        - IS/OS gap suggests overfitting (fixable with decay/window)
        """
        from backend.alpha_scoring import should_optimize
        
        candidates = []
        
        for a in alphas:
            # Consider alphas that were optimized or simulated but failed quality
            status = getattr(a, "quality_status", None)
            is_sim = self._alpha_was_simulated(a)
            
            if not is_sim:
                continue
                
            metrics = getattr(a, "metrics", {}) or {}
            try:
                sharpe = float(metrics.get("sharpe") or 0)
            except (TypeError, ValueError):
                sharpe = 0.0
            try:
                fitness = float(metrics.get("fitness") or 0)
            except (TypeError, ValueError):
                fitness = 0.0

            sign_reversal_candidate = bool(
                metrics.get("_sign_reversal_candidate")
                or (sharpe < 0 and abs(sharpe) >= 0.30)
            )

            # If explicit optimize status, always include
            if status == "OPTIMIZE":
                candidates.append({
                    "expression": a.expression,
                    "hypothesis": getattr(a, "hypothesis", ""),
                    "metrics": metrics,
                    "reason": metrics.get("_optimize_reason", "Marked for optimization"),
                    "priority": abs(sharpe) + max(abs(fitness), 0) * 0.25,
                })
                continue

            if sign_reversal_candidate:
                candidates.append({
                    "expression": a.expression,
                    "hypothesis": getattr(a, "hypothesis", ""),
                    "metrics": metrics,
                    "reason": (
                        f"NEGATIVE_SIGNAL_REVERSAL: Sharpe {sharpe:.2f}; "
                        "test reverse() before discarding factor"
                    ),
                    "priority": abs(sharpe) + max(abs(fitness), 0) * 0.25 + 1.0,
                })
                continue
            
            # Wrap metrics in structure alpha_scoring expects if needed
            sim_result = {
                "train": metrics,
                "is_stats": [metrics],
                "riskNeutralized": metrics.get("riskNeutralized", {}),
                "investabilityConstrained": metrics.get("investabilityConstrained", {})
            }
            
            should_opt, reason = should_optimize(sim_result)
            
            if should_opt:
                candidates.append({
                    "expression": a.expression,
                    "hypothesis": getattr(a, "hypothesis", ""),
                    "metrics": metrics,
                    "reason": reason,
                    "priority": max(sharpe, 0) + max(fitness, 0) * 0.25,
                })
        
        candidates.sort(key=lambda c: c.get("priority", 0), reverse=True)
        return candidates[:5]  # Limit to top 5

    def _alpha_was_simulated(self, alpha: Alpha) -> bool:
        """Infer simulation state from the persisted Alpha schema."""
        if getattr(alpha, "alpha_id", None):
            return True
        if getattr(alpha, "is_metrics", None) or getattr(alpha, "metrics", None):
            return True
        return getattr(alpha, "status", None) in {"simulated", "UNSUBMITTED", "SUBMITTED"}
    
    async def _evolve_strategy(
        self,
        task_id: str,
        current_strategy: EvolutionStrategy,
        round_result: RoundResult,
        cumulative_success: int,
        target_goal: int,
        max_iterations: int,
        dataset_id: str,
        region: str
    ) -> EvolutionStrategy:
        """
        Evolve strategy based on round results.
        
        Uses LLM analysis when available, falls back to rules.
        """
        # Compute rule-based strategy (always available)
        rule_strategy = self._rule_transition.compute_next_strategy(
            current_strategy=current_strategy,
            round_result=round_result,
            cumulative_success=cumulative_success,
            target_goal=target_goal,
            max_iterations=max_iterations
        )

        # CRITICAL FIX: If we have optimization candidates, FORCE exploit/optimize mode
        # to ensure we don't skip the opportunity to refine them.
        if round_result.optimization_candidates:
            logger.info(
                f"[Strategy] Found {len(round_result.optimization_candidates)} optimization candidates. "
                "Using diversified optimization mode."
            )
            smooth_ops = (
                "rank",
                "zscore",
                "winsorize",
                "ts_mean",
                "ts_rank",
                "ts_zscore",
                "group_neutralize",
                "scale",
                "reverse",
            )
            focus = [
                "Refine promising signals, but keep mechanism, field pair, skeleton, window, and direction diversity.",
                "If turnover is high or margin is low, prefer smoother lower-turnover skeletons; do not collapse into single-field ts_returns parameter sweeps.",
                "Test sign reversal with reverse() when the observed signal direction is adverse.",
            ]
            focus.extend(
                f"Optimize: {c['reason']} | expr={c.get('expression', '')[:120]}"
                for c in round_result.optimization_candidates[:4]
            )
            return rule_strategy.with_updates(
                mode=StrategyMode.OPTIMIZE,
                exploration_weight=max(min(rule_strategy.exploration_weight, 0.65), 0.45),
                temperature=max(min(rule_strategy.temperature, 0.75), 0.55),
                focus_hypotheses=tuple(dict.fromkeys((*current_strategy.focus_hypotheses, *focus))),
                avoid_patterns=tuple(dict.fromkeys((
                    *current_strategy.avoid_patterns,
                    "single-field ts_returns/news_eod_close parameter sweep",
                    "commutative duplicate rewrite or +0 no-op rewrite",
                    "forcing every idea into ratio/spread when factor style does not call for it",
                ))),
                preferred_fields=current_strategy.preferred_fields,
                avoid_fields=tuple(dict.fromkeys((
                    *current_strategy.avoid_fields,
                    "news_eod_close",
                    "news_eod_high",
                    "news_eod_low",
                    "news_eod_open",
                ))),
                preferred_operators=tuple(dict.fromkeys((*current_strategy.preferred_operators, *smooth_ops))),
                avoid_operators=current_strategy.avoid_operators,
                optimization_targets=tuple(
                    c.get("expression", "")
                    for c in round_result.optimization_candidates[:5]
                    if c.get("expression")
                ),
                action_summary="Diversified optimization of promising weak or reversed signals",
                reasoning=(
                    "Optimization candidates were identified from simulated metrics; guardrails preserve "
                    "strategy diversity and directly address high-turnover/low-margin failures."
                ),
            )
        
        
        # Try LLM-based strategy enhancement
        try:
            # Get recent alphas for this task (for LLM analysis)
            query = select(Alpha).where(
                Alpha.task_id == task_id
            ).order_by(Alpha.created_at.desc()).limit(10)
            
            res = await self.db.execute(query)
            recent_alphas = res.scalars().all()
            
            llm_response = await self._strategy_agent.generate_strategy(
                iteration=round_result.iteration,
                max_iterations=max_iterations,
                alphas=recent_alphas,
                failures=await self._query_recent_failures(task_id),
                dataset_id=dataset_id,
                region=region,
                cumulative_success=cumulative_success,
                target_goal=target_goal,
                previous_strategy=current_strategy
            )
            
            # Convert to dict for merging
            llm_dict = {
                "strategy": {
                    "temperature": llm_response.temperature,
                    "exploration_weight": llm_response.exploration_weight,
                    "focus_hypotheses": llm_response.focus_hypotheses,
                    "avoid_patterns": llm_response.avoid_patterns,
                    "preferred_fields": llm_response.preferred_fields,
                    "avoid_fields": llm_response.avoid_fields,
                    "action_summary": llm_response.action_summary,
                    "reasoning": llm_response.reasoning,
                },
                "optimization_targets": llm_response.optimization_suggestions
            }
            
            # Merge LLM suggestions with rule guardrails
            return merge_strategies(current_strategy, llm_dict, rule_strategy)
            
        except Exception as e:
            logger.warning(f"[MiningAgent] LLM strategy failed, using rules: {e}")
            return rule_strategy
    
    async def _record_round_summary(
        self,
        task: MiningTask,
        iteration: int,
        round_result: RoundResult,
        strategy: EvolutionStrategy,
        cumulative_success: int,
        target_alphas: int,
        run_id: Optional[int] = None,
    ):
        """Record comprehensive round summary for tracing."""
        try:
            trace_service = TraceService(
                self.db, task.id, 
                initial_step_order=99, 
                iteration=iteration,
                run_id=run_id,
            )
            
            record = trace_service.create_record(
                step_type="ROUND_SUMMARY",
                status="SUCCESS",
                input_data={
                    "round": iteration,
                    "target_alphas": target_alphas,
                    "strategy_mode": strategy.mode.value,
                    "strategy_params": {
                        "temperature": strategy.temperature,
                        "exploration": strategy.exploration_weight,
                        "focus_hypos": len(strategy.focus_hypotheses),
                        "avoid_patterns": len(strategy.avoid_patterns)
                    }
                },
                output_data={
                    "cumulative_success": cumulative_success,
                    "round_metrics": round_result.to_dict(),
                    "next_action": strategy.action_summary,
                    "next_reasoning": strategy.reasoning,
                    "optimization_candidates": len(round_result.optimization_candidates)
                }
            )
            
            await trace_service.persist_record(record)
            
        except Exception as e:
            logger.error(f"Failed to record round summary: {e}")
    
    async def _run_feedback_learning(
        self,
        task: MiningTask,
        alphas: List[Alpha],
        round_result: RoundResult,
        iteration: int,
        dataset_id: str,
        cumulative_success: int = 0,
        target_alphas: int = 4,
        max_iterations: int = 10,
    ):
        """Run feedback learning to accumulate knowledge."""
        try:
            failures = await self._query_recent_failures(task.id)
            
            await self._feedback_agent.learn_from_round(
                successes=alphas,
                failures=failures,
                iteration=iteration,
                dataset_id=dataset_id,
                region=task.region,
                cumulative_success=cumulative_success,
                target_goal=target_alphas,
                max_iterations=max_iterations,
            )
            
            # Mark failures as analyzed
            query = select(AlphaFailure).where(
                AlphaFailure.task_id == task.id,
                AlphaFailure.is_analyzed == False
            )
            res = await self.db.execute(query)
            for f in res.scalars().all():
                f.is_analyzed = True
            
            await self.db.commit()
            
        except Exception as e:
            logger.warning(f"[MiningAgent] Feedback learning failed: {e}")
            try:
                await self.db.rollback()
            except Exception:
                pass
    
    async def _run_optimization_chain(
        self,
        task: MiningTask,
        candidates: List[Dict],
        strategy: EvolutionStrategy,
        iteration: int,
        dataset_id: Optional[str] = None,
        fields: Optional[List[Dict]] = None,
        run_id: Optional[int] = None,
    ):
        """
        Run optimization chain on promising weak alphas.
        
        This is the Chain-of-Alpha style optimization loop.
        """
        from backend.optimization_chain import generate_local_rewrites, generate_settings_variants
        
        logger.info(f"[MiningAgent] Running optimization chain on {len(candidates)} candidates")
        
        for candidate in candidates[:3]:  # Limit to top 3
            expression = candidate.get("expression", "")
            metrics = candidate.get("metrics", {})
            reason = candidate.get("reason", "")
            
            if not expression:
                continue
            
            try:
                # Generate expression variants
                expr_variants = generate_local_rewrites(
                    expression=expression,
                    sim_result=metrics,
                    feedback=reason,
                    max_variants=10
                )
                
                # Generate settings variants
                settings_variants = generate_settings_variants({
                    "neutralization": "INDUSTRY",
                    "decay": 4,
                    "truncation": 0.02
                })
                
                # Simulate top variants (budget-limited)
                await self._simulate_optimization_variants(
                    task=task,
                    original_expression=expression,
                    expr_variants=expr_variants[:5],
                    settings_variants=settings_variants[:3],
                    iteration=iteration,
                    dataset_id=dataset_id,
                    fields=fields or [],
                    run_id=run_id,
                )

                if self._task_config_bool(task, "enable_genetic_optimization"):
                    await self._run_genetic_optimization(
                        task=task,
                        seed_expression=expression,
                        seed_metrics=metrics,
                        dataset_id=dataset_id,
                        run_id=run_id,
                    )
                
            except Exception as e:
                logger.warning(f"Optimization failed for {expression[:50]}: {e}")

    def _task_config_bool(self, task: MiningTask, key: str, default: bool = False) -> bool:
        value = (task.config or {}).get(key, default)
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    async def _run_genetic_optimization(
        self,
        task: MiningTask,
        seed_expression: str,
        seed_metrics: Dict,
        dataset_id: Optional[str] = None,
        run_id: Optional[int] = None,
    ) -> None:
        """Run optional budget-limited genetic optimization and persist winners."""
        try:
            from backend.genetic_optimizer import OptimizationConfig, run_genetic_optimization
            from backend.alpha_semantic_validator import compute_expression_hash

            task_config = task.config or {}
            config = OptimizationConfig(
                population_size=int(task_config.get("genetic_population_size", 12)),
                generations=int(task_config.get("genetic_generations", 2)),
                mutation_rate=float(task_config.get("genetic_mutation_rate", 0.35)),
                crossover_rate=float(task_config.get("genetic_crossover_rate", 0.15)),
                max_simulations=int(task_config.get("genetic_max_simulations", 20)),
                sharpe_threshold=float(task_config.get("genetic_sharpe_threshold", 1.5)),
                fitness_threshold=float(task_config.get("genetic_fitness_threshold", 1.0)),
                turnover_threshold=float(task_config.get("genetic_turnover_threshold", 0.7)),
            )

            async def simulate_for_genetic(**kwargs):
                result = await self.brain.simulate_alpha(**kwargs)
                metrics = result.get("metrics", {}) or {}
                return {
                    **result,
                    "is": metrics,
                    "train": metrics,
                    "os": result.get("os_metrics", {}) or {},
                    "test": result.get("os_metrics", {}) or {},
                }

            report = await run_genetic_optimization(
                seed_expression=seed_expression,
                seed_metrics=seed_metrics,
                simulate_func=simulate_for_genetic,
                config=config,
                region=task.region,
                universe=task.universe,
                delay=1,
                decay=4,
                neutralization="INDUSTRY",
            )

            saved = 0
            for individual in report.get("passed_individuals", [])[:5]:
                expression = individual.get("expression")
                if not expression:
                    continue

                metrics = {
                    "sharpe": individual.get("sharpe", 0),
                    "fitness": individual.get("fitness", 0),
                    "turnover": individual.get("turnover", 0),
                    "_genetic_overall_fitness": individual.get("overall_fitness", 0),
                    "_genetic_mutation_type": individual.get("mutation_type"),
                    "_genetic_mutation_description": individual.get("mutation_description"),
                }

                self.db.add(Alpha(
                    task_id=task.id,
                    run_id=run_id,
                    alpha_id=individual.get("alpha_id") or None,
                    expression=expression,
                    expression_hash=compute_expression_hash(expression),
                    hypothesis=f"Genetic optimization of {seed_expression[:20]}...",
                    logic_explanation=individual.get("mutation_description"),
                    region=task.region,
                    universe=task.universe,
                    dataset_id=dataset_id,
                    decay=4,
                    neutralization="INDUSTRY",
                    quality_status="OPTIMIZE",
                    status="simulated",
                    stage="IS",
                    is_sharpe=metrics.get("sharpe"),
                    is_turnover=metrics.get("turnover"),
                    is_fitness=metrics.get("fitness"),
                    is_metrics=metrics,
                    metrics=metrics,
                ))
                saved += 1

            if saved:
                await self.db.commit()
            logger.info(
                f"[MiningAgent] Genetic optimization complete | "
                f"simulations={report.get('simulations_used')} saved={saved}"
            )
        except Exception as e:
            logger.warning(f"[MiningAgent] Genetic optimization skipped/failed: {e}")
            try:
                await self.db.rollback()
            except Exception:
                pass
    
    async def _simulate_optimization_variants(
        self,
        task: MiningTask,
        original_expression: str,
        expr_variants: List[Dict],
        settings_variants: List[Dict],
        iteration: int,
        dataset_id: Optional[str] = None,
        fields: Optional[List[Dict]] = None,
        run_id: Optional[int] = None,
    ):
        """Simulate optimization variants and save improvements."""
        logger.info(
            f"[MiningAgent] Simulating {len(expr_variants)} variants for optimization"
        )
        
        settings_to_try = settings_variants or [{
            "neutralization": "INDUSTRY",
            "decay": 4,
            "truncation": 0.02,
            "description": "Base settings",
        }]

        # Process expression variants with a small settings sweep.
        simulation_jobs = []
        allowed_fields = [
            str(f.get("id") or f.get("name") or f.get("field_id") or "")
            for f in (fields or [])
            if f.get("id") or f.get("name") or f.get("field_id")
        ]
        try:
            from backend.agents.graph.nodes.validation import _validate_task_constraints
        except Exception:
            _validate_task_constraints = None

        for variant in expr_variants:
            expression = variant.get("expression")
            if not expression:
                continue
            if _validate_task_constraints:
                constraint_errors = _validate_task_constraints(
                    expression=expression,
                    allowed_fields=allowed_fields,
                    task_config=task.config or {},
                )
                if constraint_errors:
                    logger.info(
                        f"[MiningAgent] Skipping optimization variant due to constraints: "
                        f"{constraint_errors[:2]} expr={expression[:80]}"
                    )
                    continue
            for settings_variant in settings_to_try:
                simulation_jobs.append((variant, expression, settings_variant))

        from backend.alpha_semantic_validator import compute_expression_hash

        grouped_jobs: Dict[tuple, List[tuple]] = {}
        for job in simulation_jobs:
            settings_variant = job[2]
            settings_key = (
                settings_variant.get("decay", 4),
                settings_variant.get("neutralization", "INDUSTRY"),
                settings_variant.get("truncation", 0.02),
            )
            grouped_jobs.setdefault(settings_key, []).append(job)

        for (decay, neutralization, truncation), jobs in grouped_jobs.items():
            for offset in range(0, len(jobs), 4):
                batch = jobs[offset: offset + 4]
                if len(batch) < 2:
                    logger.info(
                        "[MiningAgent] Skipping single optimization variant to preserve "
                        f"multi-simulation-only constraint | expr={batch[0][1][:80] if batch else ''}"
                    )
                    continue
                expressions = [job[1] for job in batch]
                try:
                    results = await self.brain.simulate_batch(
                        expressions=expressions,
                        region=task.region,
                        universe=task.universe,
                        delay=1,
                        decay=decay,
                        neutralization=neutralization,
                        truncation=truncation,
                    )
                except Exception as e:
                    logger.warning(f"Optimization batch simulation failed: {e}")
                    continue

                for (variant, expression, settings_variant), result in zip(batch, results):
                    await self._save_optimization_result(
                        task=task,
                        original_expression=original_expression,
                        variant=variant,
                        expression=expression,
                        settings_variant=settings_variant,
                        result=result,
                        dataset_id=dataset_id,
                        run_id=run_id,
                        compute_expression_hash=compute_expression_hash,
                    )
                
        await self.db.commit()

    async def _save_optimization_result(
        self,
        task: MiningTask,
        original_expression: str,
        variant: Dict,
        expression: str,
        settings_variant: Dict,
        result: Dict,
        dataset_id: Optional[str],
        run_id: Optional[int],
        compute_expression_hash,
    ) -> None:
        """Persist one optimization result if it clears the lightweight improvement screen."""
        if not result.get("success"):
            return

        metrics = result.get("metrics", {})
        try:
            sharpe = float(metrics.get("sharpe") or 0)
        except (TypeError, ValueError):
            sharpe = 0

        if sharpe <= 1.2:
            return

        alpha = Alpha(
            task_id=task.id,
            run_id=run_id,
            alpha_id=result.get("alpha_id"),
            expression=expression,
            expression_hash=compute_expression_hash(expression),
            hypothesis=f"Optimization of {original_expression[:20]}...",
            logic_explanation=(
                f"Variant: {variant.get('description')}; "
                f"Settings: {settings_variant.get('description')}"
            ),
            region=task.region,
            universe=task.universe,
            dataset_id=dataset_id,
            decay=settings_variant.get("decay", 4),
            neutralization=settings_variant.get("neutralization", "INDUSTRY"),
            truncation=settings_variant.get("truncation", 0.02),
            status=result.get("status") or "simulated",
            stage=metrics.get("stage") or "IS",
            quality_status="OPTIMIZE",
            is_sharpe=metrics.get("sharpe"),
            is_turnover=metrics.get("turnover"),
            is_fitness=metrics.get("fitness"),
            is_returns=metrics.get("returns"),
            is_drawdown=metrics.get("drawdown"),
            is_margin=metrics.get("margin"),
            is_long_count=metrics.get("longCount"),
            is_short_count=metrics.get("shortCount"),
            checks=metrics.get("checks"),
            is_metrics=metrics,
            metrics=metrics,
        )
        self.db.add(alpha)
        logger.info(f"[MiningAgent] Optimization success: {expression[:30]} (Sharpe: {sharpe})")
    
    @property
    def workflow(self) -> MiningWorkflow:
        """Access the underlying LangGraph workflow."""
        return self._workflow


# =============================================================================
# Factory Function
# =============================================================================

def create_mining_agent(
    db: AsyncSession,
    brain: BrainAdapter = None
) -> MiningAgent:
    """
    Factory function to create MiningAgent.
    
    Usage:
        agent = create_mining_agent(db)
        result = await agent.run_evolution_loop(task, ...)
    """
    return MiningAgent(db=db, brain_adapter=brain)
