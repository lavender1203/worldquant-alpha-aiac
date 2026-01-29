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
from pathlib import Path

def _debug_log(hypo_id, location, message, data=None):
    try:
        repo_root = Path(__file__).resolve().parents[2]
        log_path = repo_root / ".cursor" / "debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
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
        self.brain = brain_adapter or BrainAdapter()
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
                operators=operators,
                num_alphas=num_alphas,
                config={
                    "configurable": {
                        "trace_service": trace_service,
                        "strategy": strategy.to_dict(),  # Pass strategy to all nodes
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
            candidate_fields = screened + others
        else:
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
            
            # Preferred first, then neutral, avoided last (or excluded)
            candidate_fields = preferred + neutral
        
        # Optional: metadata-only field screening (no extra Brain sims)
        try:
            from backend.config import settings
            if getattr(settings, "FIELD_SCREENING_ENABLED", False):
                from backend.selection_strategy import FieldSelector
                selector = FieldSelector(
                    coverage_weight=getattr(settings, "FIELD_COVERAGE_WEIGHT", 0.3),
                    novelty_weight=getattr(settings, "FIELD_NOVELTY_WEIGHT", 0.4),
                    pyramid_weight=getattr(settings, "FIELD_PYRAMID_WEIGHT", 0.3),
                    min_coverage=getattr(settings, "FIELD_MIN_COVERAGE", 0.3),
                )
                top_k = int(getattr(settings, "FIELD_SCREENING_TOP_K", 20) or 20)
                screened_fields = selector.select_diverse(candidate_fields, n=top_k)
                return screened_fields if screened_fields else candidate_fields[:top_k]
        except Exception:
            pass

        # Fallback: keep a manageable slice
        return candidate_fields[:30]
    
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
        
        # Start with provided or default strategy
        current_strategy = initial_strategy or EvolutionStrategy.default()
        
        # Ensure Brain session is active and authenticated
        async with self.brain:
            while iteration < max_iterations:
                iteration += 1
            
                logger.info(
                    f"[MiningAgent] === Round {iteration}/{max_iterations} === "
                    f"Strategy: {current_strategy.action_summary}"
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
                        strategy=current_strategy,
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
                    strategy_history.append(current_strategy)
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
                        current_strategy=current_strategy,
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
        }
    
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
        
        # Count simulated (Alpha rows persisted after simulation).
        # NOTE: SQLAlchemy Alpha model does not have `is_simulated`; infer from metrics/alpha_id.
        def _is_simulated_alpha(a: Alpha) -> bool:
            if getattr(a, "alpha_id", None):
                return True
            m = getattr(a, "metrics", None) or {}
            return isinstance(m, dict) and (
                m.get("sharpe") is not None or m.get("_score") is not None or bool(m.get("checks"))
            )

        result.total_simulated = len([a for a in alphas if _is_simulated_alpha(a)])
        
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
            elif err_type in {
                "QUALITY_CHECK_FAILED",
                "LOW_SHARPE",
                "LOW_FITNESS",
                "HIGH_TURNOVER",
                "NEGATIVE_SIGNAL",
                "PENDING_CHECKS",
            }:
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
            metrics = getattr(a, "metrics", {}) or {}

            # Infer simulation completion (Alpha rows are persisted after simulation).
            is_simulated = bool(getattr(a, "alpha_id", None)) or (
                isinstance(metrics, dict) and (metrics.get("sharpe") is not None or metrics.get("_score") is not None)
            )
            if not is_simulated:
                continue
                
            # If explicit optimize status, always include
            if status == "OPTIMIZE":
                candidates.append({
                    "expression": a.expression,
                    "hypothesis": getattr(a, "hypothesis", ""),
                    "metrics": metrics,
                    "reason": metrics.get("_optimize_reason", "Marked for optimization")
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
                    "reason": reason
                })
        
        return candidates[:5]  # Limit to top 5
    
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
            logger.info(f"[Strategy] Found {len(round_result.optimization_candidates)} optimization candidates. Forcing EXPLOIT mode.")
            rule_strategy.mode = StrategyMode.EXPLOIT
            rule_strategy.focus_hypotheses = [
                f"Optimize: {c['reason']}" for c in round_result.optimization_candidates
            ]
            rule_strategy.reasoning = "Focusing on optimizing identified promising alphas."
            return rule_strategy
        
        
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
                metrics = metrics if isinstance(metrics, dict) else {}
                # Build a sim_result-shaped dict for optimization heuristics
                sim_result = {
                    "train": {
                        "sharpe": metrics.get("train_sharpe", metrics.get("sharpe", 0)),
                        "fitness": metrics.get("train_fitness", metrics.get("fitness", 0)),
                        "turnover": metrics.get("train_turnover", metrics.get("turnover", 0)),
                        "returns": metrics.get("train_returns", metrics.get("returns", 0)),
                    },
                    "test": {
                        "sharpe": metrics.get("test_sharpe", (metrics.get("sharpe", 0) or 0) * 0.8),
                        "fitness": metrics.get("test_fitness", metrics.get("fitness", 0)),
                    },
                    "is": {
                        "sharpe": metrics.get("sharpe", 0),
                        "fitness": metrics.get("fitness", 0),
                        "turnover": metrics.get("turnover", 0),
                        "drawdown": metrics.get("drawdown", 0),
                        "checks": metrics.get("checks", []),
                    },
                    "riskNeutralized": metrics.get("riskNeutralized", {}),
                    "investabilityConstrained": metrics.get("investabilityConstrained", {}),
                    "checks": metrics.get("checks", []),
                    "can_submit": metrics.get("can_submit", False),
                }

                # Generate expression variants
                expr_variants = generate_local_rewrites(
                    expression=expression,
                    sim_result=sim_result,
                    feedback=reason,
                    max_variants=10
                )
                
                # Generate settings variants (based on the original simulation settings if available)
                base_brain_settings = {}
                if isinstance(metrics, dict):
                    base_brain_settings = metrics.get("_brain_settings") or {}
                base_settings = {
                    "neutralization": base_brain_settings.get("neutralization", "SUBINDUSTRY"),
                    "decay": base_brain_settings.get("decay", 4),
                    "truncation": base_brain_settings.get("truncation", 0.08),
                    "delay": base_brain_settings.get("delay", 1),
                    "testPeriod": base_brain_settings.get("testPeriod", "P2Y0M"),
                }
                settings_variants = generate_settings_variants(
                    {
                        "neutralization": base_settings["neutralization"],
                        "decay": base_settings["decay"],
                        "truncation": base_settings["truncation"],
                    }
                )
                
                # Simulate top variants (budget-limited)
                await self._simulate_optimization_variants(
                    task=task,
                    original_expression=expression,
                    expr_variants=expr_variants[:5],
                    settings_variants=settings_variants[:6],
                    iteration=iteration,
                    dataset_id=dataset_id,
                    run_id=run_id,
                    base_settings=base_settings,
                    baseline_metrics=metrics if isinstance(metrics, dict) else {},
                )
                
            except Exception as e:
                logger.warning(f"Optimization failed for {expression[:50]}: {e}")
    
    async def _simulate_optimization_variants(
        self,
        task: MiningTask,
        original_expression: str,
        expr_variants: List[Dict],
        settings_variants: List[Dict],
        iteration: int,
        dataset_id: Optional[str] = None,
        run_id: Optional[int] = None,
        base_settings: Optional[Dict] = None,
        baseline_metrics: Optional[Dict] = None,
    ):
        """Simulate optimization variants (expression + settings) and persist improvements."""
        from backend.config import settings as app_settings
        from backend.alpha_semantic_validator import compute_expression_hash
        from backend.alpha_scoring import calculate_alpha_score, should_optimize, evaluate_with_brain_checks

        base_settings = base_settings or {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08, "delay": 1, "testPeriod": "P2Y0M"}
        baseline_metrics = baseline_metrics or {}

        budget = int(getattr(app_settings, "OPTIMIZATION_BUDGET_PER_ALPHA", 20) or 20)
        budget = max(5, min(50, budget))
        expr_budget = min(len(expr_variants), max(1, budget // 2))
        settings_budget = min(len(settings_variants), max(0, budget - expr_budget))

        logger.info(
            f"[MiningAgent] Optimization sims | expr_variants={len(expr_variants)} settings_variants={len(settings_variants)} "
            f"budget={budget} (expr={expr_budget}, settings={settings_budget})"
        )

        async def _simulate_and_persist(expr: str, note: str, sim_settings: Dict):
            """Run a single simulation and persist if valuable."""
            result = await self.brain.simulate_alpha(
                expression=expr,
                region=task.region,
                universe=task.universe,
                delay=int(sim_settings.get("delay", 1)),
                decay=int(sim_settings.get("decay", 4)),
                neutralization=str(sim_settings.get("neutralization", "SUBINDUSTRY")),
                truncation=float(sim_settings.get("truncation", 0.08)),
                test_period=str(sim_settings.get("testPeriod", "P2Y0M")),
            )

            if not result.get("success"):
                return None

            # Merge top-level checks/can_submit into metrics for consistent scoring
            m = result.get("metrics", {}) or {}
            merged = dict(m) if isinstance(m, dict) else {}
            if result.get("checks") is not None:
                merged["checks"] = result.get("checks")
            if result.get("can_submit") is not None:
                merged["can_submit"] = result.get("can_submit")

            # Build sim_result for scoring/optimization checks
            sim_result = {
                "train": {
                    "sharpe": merged.get("train_sharpe", merged.get("sharpe", 0)),
                    "fitness": merged.get("train_fitness", merged.get("fitness", 0)),
                    "turnover": merged.get("train_turnover", merged.get("turnover", 0)),
                    "returns": merged.get("train_returns", merged.get("returns", 0)),
                },
                "test": {
                    "sharpe": merged.get("test_sharpe", merged.get("sharpe", 0) * 0.8),
                    "fitness": merged.get("test_fitness", merged.get("fitness", 0)),
                },
                "is": {
                    "sharpe": merged.get("sharpe", 0),
                    "fitness": merged.get("fitness", 0),
                    "turnover": merged.get("turnover", 0),
                    "drawdown": merged.get("drawdown", 0),
                    "checks": merged.get("checks", []),
                },
                "riskNeutralized": merged.get("riskNeutralized", {}),
                "investabilityConstrained": merged.get("investabilityConstrained", {}),
                "checks": merged.get("checks", []),
                "can_submit": bool(merged.get("can_submit", False)),
            }

            brain_eval = evaluate_with_brain_checks(sim_result)
            score = calculate_alpha_score(sim_result=sim_result, prod_corr=0.0, self_corr=0.0)
            opt_ok, opt_reason = should_optimize(sim_result)

            # Two-tier status (correlation check omitted here for budget reasons)
            if brain_eval.get("can_submit", False):
                quality_status = "PASS"
            elif opt_ok and score >= getattr(app_settings, "SCORE_OPTIMIZE_THRESHOLD", 0.3):
                quality_status = "OPTIMIZE"
            elif score >= getattr(app_settings, "SCORE_PASS_THRESHOLD", 0.8):
                quality_status = "PROMISING"
            else:
                quality_status = "FAIL"

            # Persist only valuable outcomes (submit-ready, or better-than-baseline, or explicit OPTIMIZE)
            baseline_sharpe = float(baseline_metrics.get("sharpe", 0) or 0)
            new_sharpe = float(merged.get("sharpe", 0) or 0)
            improved = (new_sharpe - baseline_sharpe) >= 0.25 or quality_status in {"PASS", "OPTIMIZE", "PROMISING"}

            if not improved:
                return None

            expr_hash = compute_expression_hash(expr) if expr else None
            merged["_score"] = round(score, 4)
            merged["_optimize_reason"] = opt_reason
            merged["_brain_can_submit"] = brain_eval.get("can_submit", False)
            merged["_brain_failed_checks"] = brain_eval.get("failed_checks", [])
            merged["_brain_pending_checks"] = brain_eval.get("pending_checks", [])
            merged["_optimization_note"] = note

            alpha = Alpha(
                task_id=task.id,
                run_id=run_id,
                alpha_id=result.get("alpha_id"),
                expression=expr,
                expression_hash=expr_hash,
                hypothesis=f"Optimization: {note}",
                logic_explanation=note,
                region=task.region,
                universe=task.universe,
                dataset_id=dataset_id,
                status="simulated",
                stage=result.get("stage") or "IS",
                quality_status=quality_status,
                settings=result.get("settings") or sim_settings,
                checks=result.get("checks"),
                metrics=merged,
            )
            self.db.add(alpha)
            return alpha
        
        created = 0
        used = 0

        # 1) Expression-level variants under base settings
        for variant in expr_variants[:expr_budget]:
            if used >= budget:
                break
            try:
                expr = variant.get("expression")
                if not expr:
                    continue
                note = f"expr_variant: {variant.get('description')}"
                alpha = await _simulate_and_persist(expr, note, base_settings)
                used += 1
                if alpha is not None:
                    created += 1
            except Exception as e:
                logger.warning(f"Optimization (expr) simulation failed: {e}")
                used += 1

        # 2) Settings-level sweep on the original expression
        for s in settings_variants[:settings_budget]:
            if used >= budget:
                break
            try:
                sim_settings = dict(base_settings)
                sim_settings.update({
                    "neutralization": s.get("neutralization", sim_settings.get("neutralization")),
                    "decay": s.get("decay", sim_settings.get("decay")),
                    "truncation": s.get("truncation", sim_settings.get("truncation")),
                })

                # Light testPeriod sweep: keep base period, and occasionally try a quicker one.
                # This keeps cost bounded while still exploring the lever.
                candidates_settings = [sim_settings]
                quick_period = getattr(app_settings, "QUICK_TEST_PERIOD", None)
                if quick_period and quick_period != sim_settings.get("testPeriod"):
                    candidates_settings.append({**sim_settings, "testPeriod": quick_period})

                for j, ss in enumerate(candidates_settings[:2]):
                    if used >= budget:
                        break
                    note = f"settings_variant: {s.get('description')} | testPeriod={ss.get('testPeriod')}"
                    alpha = await _simulate_and_persist(original_expression, note, ss)
                    used += 1
                    if alpha is not None:
                        created += 1
            except Exception as e:
                logger.warning(f"Optimization (settings) simulation failed: {e}")
                used += 1

        await self.db.commit()
        logger.info(f"[MiningAgent] Optimization persistence complete | sims_used={used}/{budget} created={created}")
    
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
