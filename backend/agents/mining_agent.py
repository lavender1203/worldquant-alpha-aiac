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

from typing import List, Dict, Optional, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from loguru import logger
from datetime import datetime, timedelta
import re
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
        task_config = {**(task.config or {}), "_iteration": iteration}
        task_config = await self._with_first_order_probe_plan(
            task=task,
            dataset_id=dataset_id,
            fields=fields,
            operators=operators,
            task_config=task_config,
        )
        task_config = await self._with_attempted_expression_filter(
            task=task,
            dataset_id=dataset_id,
            task_config=task_config,
        )
        
        try:
            workflow_operators = operators if task_config.get("first_order_operator_probe") else self._apply_operator_filters(operators, strategy)
            # Run workflow with strategy context
            result = await self._workflow.run_with_persistence(
                task=task,
                dataset_id=dataset_id,
                fields=self._apply_field_filters(fields, strategy),
                operators=workflow_operators,
                num_alphas=num_alphas,
                config={
                    "configurable": {
                        "trace_service": trace_service,
                        "rag_service": self._workflow.rag_service,
                        "strategy": strategy.to_dict(),  # Pass strategy to all nodes
                        "task_config": task_config,
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

    async def _with_first_order_probe_plan(
        self,
        task: MiningTask,
        dataset_id: str,
        fields: List[Dict],
        operators: List[Dict],
        task_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inject a cumulative next-operator plan for first-order probing."""
        if not task_config.get("first_order_operator_probe"):
            return task_config
        if not task_config.get("first_order_operator_probe_skip_covered", True):
            return task_config
        if task_config.get("first_order_operator_probe_target_operators"):
            return task_config

        try:
            from backend.agents.graph.nodes.generation import _regular_operator_names

            regular_ops = _regular_operator_names(operators)
        except Exception:
            regular_ops = sorted(self._regular_operator_names(operators))
        if not regular_ops:
            return task_config

        max_operator_count = int(task_config.get("max_operator_count", 5) or 5)
        probeable_ops = self._first_order_probeable_operator_names(
            fields=fields,
            operators=operators,
            max_operator_count=max_operator_count,
            regular_order=regular_ops,
        )
        unavailable_ops = [op for op in regular_ops if op not in set(probeable_ops)]
        if not probeable_ops:
            logger.warning(
                "[MiningAgent] First-order probe has no probeable operators | "
                f"dataset={dataset_id} unavailable={unavailable_ops}"
            )
            return task_config

        completed_ops = await self._query_first_order_probe_operators(
            task=task,
            dataset_id=dataset_id,
            completed_only=True,
        )
        completed_ops = completed_ops & set(probeable_ops)
        deferred_counts = await self._query_deferred_first_order_probe_operator_counts(
            task=task,
            dataset_id=dataset_id,
            completed_ops=completed_ops,
        )
        deferred_ops = set(deferred_counts) & set(probeable_ops)
        start_index = max(0, int(task_config.get("first_order_operator_probe_start_index", 0) or 0))
        ordered_ops = probeable_ops[start_index:] + probeable_ops[:start_index]
        active_remaining = [
            op for op in ordered_ops
            if op not in completed_ops and op not in deferred_ops
        ]
        deferred_remaining = [
            op for op in ordered_ops
            if op not in completed_ops and op in deferred_ops
        ]

        if (
            not active_remaining
            and task_config.get("first_order_auto_strengthen_after_active_coverage", True)
        ):
            weak_sharpe_floor = float(task_config.get("dataset_weak_signal_sharpe_floor", 0.5))
            weak_fitness_floor = float(task_config.get("dataset_weak_signal_fitness_floor", 0.2))
            reversal_min = float(task_config.get("optimization_reversal_abs_sharpe_min", 0.8))
            signal_ops, signal_details = await self._query_first_order_signal_operators(
                task=task,
                dataset_id=dataset_id,
                weak_sharpe_floor=weak_sharpe_floor,
                weak_fitness_floor=weak_fitness_floor,
                reversal_min=reversal_min,
                margin_min=float(task_config.get("margin_min", 0.001)),
            )
            strengthening_seeds = self._merge_first_order_strengthening_seeds(
                existing_seeds=task_config.get("first_order_strengthening_seeds") or [],
                signal_details=signal_details,
                max_auto_seeds=int(task_config.get("first_order_auto_strengthen_max_seeds", 6) or 6),
                max_variants_per_seed=int(task_config.get("first_order_auto_strengthen_variants_per_seed", 4) or 4),
            )
            if strengthening_seeds:
                planned = dict(task_config)
                planned["first_order_operator_probe"] = False
                planned["first_order_operator_probe_completed_operators"] = sorted(completed_ops)
                planned["first_order_operator_probe_deferred_operators"] = sorted(deferred_ops)
                planned["first_order_operator_probe_unavailable_operators"] = unavailable_ops
                planned["first_order_operator_probe_deferred_retry_operators"] = []
                planned["first_order_operator_probe_active_target_operators"] = []
                planned["first_order_strengthening_seeds"] = strengthening_seeds
                planned["first_order_strengthening_auto_seeded"] = True
                planned["first_order_strengthening_auto_signal_operators"] = sorted(signal_ops)
                logger.info(
                    "[MiningAgent] First-order active coverage complete; switching to "
                    "second-order strengthening | "
                    f"dataset={dataset_id} completed={len(completed_ops)}/{len(probeable_ops)} "
                    f"deferred={len(deferred_ops)} seeds={len(strengthening_seeds)}"
                )
                return planned

        pool_size = int(task_config.get("generation_candidate_pool", len(regular_ops)) or len(regular_ops))
        quota = int(task_config.get("template_candidate_quota", pool_size) or pool_size)
        batch_size = int(task_config.get("first_order_operator_probe_batch_size", pool_size) or pool_size)
        deferred_retry_slots = max(
            0,
            int(task_config.get("first_order_probe_deferred_retry_slots", 0) or 0),
        )
        available_count = len(active_remaining)
        if deferred_retry_slots or not active_remaining:
            available_count += len(deferred_remaining)
        target_size = max(1, min(pool_size, quota, batch_size, available_count or 1))
        target_ops = self._compose_first_order_probe_targets(
            ordered_ops=ordered_ops,
            active_remaining=active_remaining,
            deferred_remaining=deferred_remaining,
            deferred_counts=deferred_counts,
            target_size=target_size,
            deferred_retry_slots=deferred_retry_slots,
        )
        if not target_ops:
            return task_config

        planned = dict(task_config)
        planned["first_order_operator_probe_target_operators"] = target_ops
        planned["first_order_operator_probe_completed_operators"] = sorted(completed_ops)
        planned["first_order_operator_probe_deferred_operators"] = sorted(deferred_ops)
        planned["first_order_operator_probe_unavailable_operators"] = unavailable_ops
        planned["first_order_operator_probe_deferred_retry_operators"] = [
            op for op in target_ops if op in deferred_ops
        ]
        planned["first_order_operator_probe_active_target_operators"] = [
            op for op in target_ops if op not in deferred_ops
        ]
        if planned.get("avoid_operators"):
            target_set = set(target_ops)
            planned["avoid_operators"] = [
                op for op in planned.get("avoid_operators", [])
                if str(op).lower() not in target_set
            ]
        logger.info(
            "[MiningAgent] First-order probe plan | "
            f"dataset={dataset_id} target={target_ops} "
            f"completed={len(completed_ops)}/{len(probeable_ops)} "
            f"deferred={len(deferred_ops)} "
            f"unavailable={len(unavailable_ops)} "
            f"deferred_retry={planned['first_order_operator_probe_deferred_retry_operators']}"
        )
        return planned

    async def _with_attempted_expression_filter(
        self,
        task: MiningTask,
        dataset_id: str,
        task_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inject recent exact expressions so generation avoids known DB skips."""
        if not task_config.get("avoid_attempted_expressions", True):
            return task_config
        if task_config.get("attempted_expressions"):
            return task_config

        limit = max(0, int(task_config.get("attempted_expression_limit", 500) or 0))
        if limit <= 0:
            return task_config

        attempted = await self._query_attempted_expressions(
            task=task,
            dataset_id=dataset_id,
            limit=limit,
        )
        if not attempted:
            return task_config

        planned = dict(task_config)
        planned["attempted_expressions"] = attempted
        logger.info(
            "[MiningAgent] Injected attempted expression filter | "
            f"dataset={dataset_id} count={len(attempted)}"
        )
        return planned

    async def _query_attempted_expressions(
        self,
        task: MiningTask,
        dataset_id: str,
        limit: int,
    ) -> List[str]:
        """Return recent expressions already attempted for this dataset context."""
        attempted: List[str] = []
        seen = set()

        alpha_rows = (
            await self.db.execute(
                select(Alpha.expression)
                .where(
                    Alpha.region == task.region,
                    Alpha.universe == task.universe,
                    Alpha.dataset_id == dataset_id,
                )
                .order_by(Alpha.id.desc())
                .limit(limit)
            )
        ).scalars().all()
        for expression in alpha_rows:
            expression = str(expression or "").strip()
            if expression and expression not in seen:
                attempted.append(expression)
                seen.add(expression)

        remaining = max(0, limit - len(attempted))
        if remaining <= 0:
            return attempted

        failure_rows = (
            await self.db.execute(
                select(AlphaFailure.expression, MiningTask)
                .join(MiningTask, AlphaFailure.task_id == MiningTask.id)
                .where(
                    MiningTask.region == task.region,
                    MiningTask.universe == task.universe,
                )
                .order_by(AlphaFailure.id.desc())
                .limit(limit)
            )
        ).all()
        for expression, failure_task in failure_rows:
            if dataset_id not in (failure_task.target_datasets or []):
                continue
            expression = str(expression or "").strip()
            if expression and expression not in seen:
                attempted.append(expression)
                seen.add(expression)
            if len(attempted) >= limit:
                break

        return attempted

    @staticmethod
    def _first_order_probeable_operator_names(
        fields: List[Dict],
        operators: List[Dict],
        max_operator_count: int,
        regular_order: Optional[List[str]] = None,
    ) -> List[str]:
        """Operators that can actually produce a first-order candidate for this field set."""
        try:
            from backend.agents.graph.nodes.generation import (
                _first_order_operator_probe_candidates,
                _regular_operator_names,
            )

            candidates = _first_order_operator_probe_candidates(
                fields=fields,
                operators=operators,
                max_operator_count=max_operator_count,
            )
            candidate_ops = {
                str(candidate.metadata.get("probe_operator") or "").lower()
                for candidate in candidates
                if candidate.metadata.get("probe_operator")
            }
            ordered = regular_order or _regular_operator_names(operators)
            return [op for op in ordered if op in candidate_ops]
        except Exception as exc:
            logger.warning(f"[MiningAgent] Could not derive probeable first-order operators: {exc}")
            return list(regular_order or sorted(MiningAgent._regular_operator_names_static(operators)))

    @staticmethod
    def _regular_operator_names_static(operators: List[Dict]) -> set:
        names = set()
        for op in operators or []:
            name = str(op.get("name") if isinstance(op, dict) else op or "").lower()
            if not name:
                continue
            scope = op.get("scope") if isinstance(op, dict) else None
            if scope and "REGULAR" not in {str(item).upper() for item in scope}:
                continue
            names.add(name)
        return names

    @staticmethod
    def _compose_first_order_probe_targets(
        ordered_ops: List[str],
        active_remaining: List[str],
        deferred_remaining: List[str],
        deferred_counts: Dict[str, int],
        target_size: int,
        deferred_retry_slots: int,
    ) -> List[str]:
        """Build the next first-order probe batch without permanently skipping timeouts."""
        target_size = max(0, int(target_size or 0))
        if target_size <= 0:
            return []

        order = {op: idx for idx, op in enumerate(ordered_ops)}
        deferred_ordered = sorted(
            dict.fromkeys(deferred_remaining),
            key=lambda op: (deferred_counts.get(op, 0), order.get(op, len(order))),
        )

        if not active_remaining:
            return deferred_ordered[:target_size]

        retry_slots = min(max(0, int(deferred_retry_slots or 0)), target_size)
        retry_ops = deferred_ordered[:retry_slots]
        active_slots = target_size - len(retry_ops)
        selected = list(dict.fromkeys(active_remaining[:active_slots] + retry_ops))

        if len(selected) < target_size:
            for op in active_remaining[active_slots:] + deferred_ordered[retry_slots:]:
                if op not in selected:
                    selected.append(op)
                if len(selected) >= target_size:
                    break

        return selected

    @staticmethod
    def _merge_first_order_strengthening_seeds(
        existing_seeds: List[Any],
        signal_details: List[Dict[str, Any]],
        max_auto_seeds: int,
        max_variants_per_seed: int,
    ) -> List[Dict[str, Any]]:
        """Merge manual seeds with top first-order signal details."""
        merged: List[Dict[str, Any]] = []
        seen_expressions = set()

        for seed in existing_seeds or []:
            if isinstance(seed, str):
                seed = {"expression": seed}
            if not isinstance(seed, dict):
                continue
            expression = str(seed.get("expression") or "").strip()
            if not expression or expression in seen_expressions:
                continue
            merged.append(dict(seed))
            seen_expressions.add(expression)

        remaining_slots = max(0, int(max_auto_seeds or 0))
        for detail in signal_details or []:
            if remaining_slots <= 0:
                break
            expression = str(detail.get("expression") or "").strip()
            if not expression or expression in seen_expressions:
                continue
            operator = str(detail.get("operator") or detail.get("probe_operator") or "seed").lower()
            merged.append({
                "expression": expression,
                "probe_operator": operator,
                "sharpe": detail.get("sharpe", 0),
                "fitness": detail.get("fitness", 0),
                "turnover": detail.get("turnover", 0),
                "margin": detail.get("margin", 0),
                "max_variants": max(1, int(max_variants_per_seed or 1)),
                "reason": (
                    "Auto-seeded after first-order active operator coverage completed; "
                    f"source alpha={detail.get('alpha_id') or 'unknown'}."
                ),
            })
            seen_expressions.add(expression)
            remaining_slots -= 1

        return merged

    async def _query_first_order_probe_operators(
        self,
        task: MiningTask,
        dataset_id: str,
        completed_only: bool,
    ) -> set:
        """Return probe operators observed for this region/universe/dataset.

        completed_only=True uses persisted Alpha rows with metrics. Simulation
        timeouts are deliberately not treated as completed because they do not
        provide signal evidence.
        """
        ops = set()
        query = select(Alpha).where(
            Alpha.region == task.region,
            Alpha.universe == task.universe,
            Alpha.dataset_id == dataset_id,
        )
        rows = (await self.db.execute(query)).scalars().all()
        for alpha in rows:
            metrics = alpha.metrics or alpha.is_metrics or {}
            if not isinstance(metrics, dict):
                continue
            candidate_meta = metrics.get("_candidate_metadata") or {}
            if candidate_meta.get("source") != "first_order_operator_probe":
                continue
            if completed_only and alpha.is_sharpe is None and alpha.is_fitness is None:
                continue
            probe_op = candidate_meta.get("probe_operator")
            if probe_op and self._expression_contains_operator(alpha.expression, str(probe_op)):
                ops.add(str(probe_op).lower())

        if completed_only:
            return ops

        failure_query = select(AlphaFailure).where(AlphaFailure.task_id == task.id)
        failure_rows = (await self.db.execute(failure_query)).scalars().all()
        for failure in failure_rows:
            try:
                details = json.loads(failure.raw_response or "{}")
            except Exception:
                details = {}
            candidate_meta = details.get("candidate_metadata") if isinstance(details, dict) else {}
            if (candidate_meta or {}).get("source") != "first_order_operator_probe":
                continue
            probe_op = (candidate_meta or {}).get("probe_operator")
            if probe_op:
                ops.add(str(probe_op).lower())
        return ops

    async def _query_deferred_first_order_probe_operators(
        self,
        task: MiningTask,
        dataset_id: str,
        completed_ops: set,
    ) -> set:
        """Return first-order operators with unrecovered timeout parents.

        These are not counted as completed coverage, but temporarily skipping
        them prevents a stuck BRAIN parent from blocking the whole ordered
        operator sweep. Project backfill can recover them later.
        """
        counts = await self._query_deferred_first_order_probe_operator_counts(
            task=task,
            dataset_id=dataset_id,
            completed_ops=completed_ops,
        )
        return set(counts)

    async def _query_deferred_first_order_probe_operator_counts(
        self,
        task: MiningTask,
        dataset_id: str,
        completed_ops: set,
    ) -> Dict[str, int]:
        """Return timeout counts for first-order operators that should be deferred."""
        if not (task.config or {}).get("first_order_probe_defer_timeouts", True):
            return {}

        defer_after = max(1, int((task.config or {}).get("first_order_probe_defer_after_timeouts", 1) or 1))
        rows = (
            await self.db.execute(
                select(AlphaFailure, MiningTask)
                .join(MiningTask, AlphaFailure.task_id == MiningTask.id)
                .where(
                    MiningTask.region == task.region,
                    MiningTask.universe == task.universe,
                    AlphaFailure.error_type == "SIMULATION_ERROR",
                )
                .order_by(AlphaFailure.id.desc())
                .limit(500)
            )
        ).all()

        counts: Dict[str, int] = {}
        for failure, failure_task in rows:
            if dataset_id not in (failure_task.target_datasets or []):
                continue
            try:
                details = json.loads(failure.raw_response or "{}")
            except Exception:
                details = {}
            if not isinstance(details, dict):
                continue
            meta = details.get("candidate_metadata") if isinstance(details.get("candidate_metadata"), dict) else {}
            if meta.get("source") != "first_order_operator_probe":
                continue
            metrics = details.get("metrics") if isinstance(details.get("metrics"), dict) else {}
            location = (
                metrics.get("_simulation_location")
                or details.get("_simulation_location")
                or details.get("simulation_location")
                or details.get("location")
            )
            if not location:
                continue
            message = f"{failure.error_message or ''} {details.get('error') or ''}".lower()
            if "timed out" not in message and "timeout" not in message:
                continue
            if "multi-simulation" not in message and "simulation timed out" not in message:
                continue
            probe_op = str(meta.get("probe_operator") or "").lower()
            if not probe_op or probe_op in completed_ops:
                continue
            counts[probe_op] = counts.get(probe_op, 0) + 1

        return {op: count for op, count in counts.items() if count >= defer_after}

    @staticmethod
    def _expression_contains_operator(expression: str, operator_name: str) -> bool:
        operator = str(operator_name or "").lower()
        if not expression or not operator:
            return False
        return any(
            match.group(1).lower() == operator
            for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression)
        )
    
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
        bad_round_streak = 0
        weak_round_streak = 0
        dataset_stop_reason = None
        
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
                await self.db.execute(
                    update(MiningTask)
                    .where(MiningTask.id == task.id)
                    .values(current_iteration=iteration, updated_at=datetime.utcnow())
                )
                await self.db.commit()
            
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
                        iteration=iteration,
                        task_config=task.config or {},
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
                    await self._maybe_record_operator_exploration_snapshot(
                        task=task,
                        dataset_id=dataset_id,
                        fields=fields,
                        operators=operators,
                        iteration=iteration,
                        run_id=run_id,
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

                    round_health = self._classify_dataset_round_health(task, round_result)
                    if round_health["kind"] == "blocking_error":
                        bad_round_streak += 1
                        weak_round_streak = 0
                    elif round_health["kind"] == "weak_signal":
                        weak_round_streak += 1
                        bad_round_streak = 0
                    else:
                        bad_round_streak = 0
                        weak_round_streak = 0

                    error_limit = int((task.config or {}).get("dataset_error_round_limit", 2))
                    weak_limit = int((task.config or {}).get("dataset_weak_signal_round_limit", 4))
                    if bad_round_streak >= error_limit or weak_round_streak >= weak_limit:
                        dataset_stop_reason = (
                            f"{round_health['reason']} | "
                            f"bad_round_streak={bad_round_streak}/{error_limit} "
                            f"weak_round_streak={weak_round_streak}/{weak_limit}"
                        )
                        logger.warning(
                            "[MiningAgent] Dataset circuit breaker triggered | "
                            f"dataset={dataset_id} reason={dataset_stop_reason}"
                        )
                        await self._record_dataset_stop(
                            task=task,
                            dataset_id=dataset_id,
                            iteration=iteration,
                            reason=dataset_stop_reason,
                            round_result=round_result,
                            run_id=run_id,
                        )
                        break
                    
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
                        elapsed = time.time() - loop_start_time
                        max_loop_seconds = float((task.config or {}).get("max_evolution_loop_seconds", 3300))
                        optimization_min_remaining = float((task.config or {}).get("optimization_min_remaining_seconds", 420))
                        if elapsed + optimization_min_remaining >= max_loop_seconds:
                            logger.info(
                                "[MiningAgent] Skipping optimization chain to preserve iteration budget | "
                                f"elapsed={elapsed:.0f}s max_loop_seconds={max_loop_seconds:.0f}s "
                                f"iteration={iteration}/{max_iterations}"
                            )
                            optimized_passes = 0
                        else:
                            optimized_passes = await self._run_optimization_chain(
                                task=task,
                                candidates=round_result.optimization_candidates,
                                strategy=current_strategy,
                                iteration=iteration,
                                dataset_id=dataset_id,
                                fields=fields,
                                run_id=run_id,
                            )
                        if optimized_passes:
                            total_success += optimized_passes
                            logger.info(
                                "[MiningAgent] Optimization added strict passes | "
                                f"optimized_passes={optimized_passes} "
                                f"total={total_success}/{target_alphas}"
                            )

                    if total_success >= target_alphas:
                        logger.info(
                            f"[MiningAgent] Goal reached after optimization! "
                            f"{total_success}/{target_alphas} in {iteration} rounds"
                        )
                        break
                    
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
            "dataset_stop_reason": dataset_stop_reason,
            "strategy_history": [s.to_dict() for s in strategy_history],
            "final_strategy": current_strategy.to_dict(),
            "metrics_report": metrics_report,
        }

    def _classify_dataset_round_health(
        self,
        task: MiningTask,
        round_result: RoundResult,
    ) -> Dict[str, str]:
        """Classify whether the current dataset is still worth another round."""
        config = task.config or {}
        min_errors = int(config.get("dataset_error_min_count", 2))
        weak_sharpe_floor = float(config.get("dataset_weak_signal_sharpe_floor", 0.40))
        weak_fitness_floor = float(config.get("dataset_weak_signal_fitness_floor", 0.15))
        reversal_min = float(config.get("optimization_reversal_abs_sharpe_min", 0.8))
        rn_sharpe_min = float(config.get("rn_sharpe_min", 1.58))
        rn_fitness_min = float(config.get("rn_fitness_min", 1.0))
        rn_stop_ratio = float(config.get("dataset_weak_rn_stop_ratio", 0.60))
        has_reversal_strength = (round_result.best_abs_sharpe or 0.0) >= reversal_min

        if (
            round_result.total_simulated == 0
            and round_result.simulation_errors >= min_errors
        ):
            return {
                "kind": "blocking_error",
                "reason": (
                    "No alpha reached usable simulation output and platform simulation "
                    f"errors={round_result.simulation_errors}"
                ),
            }

        if (
            round_result.total_generated == 0
            and (round_result.syntax_errors + round_result.simulation_errors) >= min_errors
        ):
            return {
                "kind": "blocking_error",
                "reason": (
                    "Generation/correction produced no persisted alphas and recent "
                    f"errors={round_result.syntax_errors + round_result.simulation_errors}"
                ),
            }

        if (
            round_result.total_simulated > 0
            and round_result.passed_count == 0
            and (round_result.best_sharpe or 0.0) < weak_sharpe_floor
            and (round_result.best_fitness or 0.0) < weak_fitness_floor
            and not has_reversal_strength
        ):
            return {
                "kind": "weak_signal",
                "reason": (
                    "Simulated signals remain weak: "
                    f"best_sharpe={round_result.best_sharpe} "
                    f"best_fitness={round_result.best_fitness}"
                ),
            }

        if (
            round_result.total_simulated > 0
            and round_result.passed_count == 0
            and round_result.best_rn_sharpe is not None
            and round_result.best_rn_fitness is not None
            and round_result.best_rn_sharpe < rn_sharpe_min * rn_stop_ratio
            and round_result.best_rn_fitness < rn_fitness_min * rn_stop_ratio
            and (round_result.best_sharpe or 0.0) < max(weak_sharpe_floor, rn_sharpe_min * rn_stop_ratio)
            and not has_reversal_strength
        ):
            return {
                "kind": "weak_signal",
                "reason": (
                    "Risk-neutralized signal remains far from target: "
                    f"best_sharpe={round_result.best_sharpe} "
                    f"best_fitness={round_result.best_fitness} "
                    f"best_rn_sharpe={round_result.best_rn_sharpe} "
                    f"best_rn_fitness={round_result.best_rn_fitness}"
                ),
            }

        return {"kind": "healthy", "reason": "round produced usable feedback"}

    async def _record_dataset_stop(
        self,
        task: MiningTask,
        dataset_id: str,
        iteration: int,
        reason: str,
        round_result: RoundResult,
        run_id: Optional[int] = None,
    ) -> None:
        """Persist dataset-level stop reason so later diagnosis is grounded."""
        try:
            trace_service = TraceService(
                self.db,
                task.id,
                initial_step_order=150,
                iteration=iteration,
                run_id=run_id,
            )
            record = trace_service.create_record(
                step_type="DATASET_STOP",
                status="SUCCESS",
                input_data={"dataset_id": dataset_id, "round": iteration},
                output_data={
                    "reason": reason,
                    "round_metrics": round_result.to_dict(),
                },
            )
            await trace_service.persist_record(record)
        except Exception as e:
            logger.warning(f"[MiningAgent] Failed to record dataset stop: {e}")

    async def _maybe_record_operator_exploration_snapshot(
        self,
        task: MiningTask,
        dataset_id: str,
        fields: List[Dict],
        operators: List[Dict],
        iteration: int,
        run_id: Optional[int] = None,
    ) -> None:
        """Persist operator coverage snapshots for ordered exploration audits."""
        config = task.config or {}
        interval = int(config.get("operator_coverage_interval", 10) or 0)
        should_record = interval > 0 and iteration % interval == 0
        if config.get("first_order_operator_probe") and iteration == 1:
            should_record = True
        if not should_record:
            return

        regular_ops = self._regular_operator_names(operators)
        if not regular_ops:
            return
        probeable_ops = set(self._first_order_probeable_operator_names(
            fields=fields,
            operators=operators,
            max_operator_count=int(config.get("max_operator_count", 5) or 5),
            regular_order=sorted(regular_ops),
        ))
        unavailable_probe_ops = sorted(set(regular_ops) - probeable_ops)
        coverage_denominator_ops = probeable_ops or set(regular_ops)

        try:
            query = select(Alpha).where(Alpha.task_id == task.id)
            if run_id is not None:
                query = query.where(Alpha.run_id == run_id)
            rows = (await self.db.execute(query)).scalars().all()
            failure_query = select(AlphaFailure).where(AlphaFailure.task_id == task.id)
            if run_id is not None:
                failure_query = failure_query.where(AlphaFailure.run_id == run_id)
            failure_rows = (await self.db.execute(failure_query)).scalars().all()

            used_counts: Dict[str, int] = {}
            attempted_counts: Dict[str, int] = {}
            probe_counts: Dict[str, int] = {}
            weak_signal_ops = set()
            pass_ops = set()
            first_order_signal_ops = set()
            weak_sharpe_floor = float(config.get("dataset_weak_signal_sharpe_floor", 0.5))
            weak_fitness_floor = float(config.get("dataset_weak_signal_fitness_floor", 0.2))
            reversal_min = float(config.get("optimization_reversal_abs_sharpe_min", 0.8))

            for alpha in rows:
                metrics = alpha.metrics or alpha.is_metrics or {}
                candidate_meta = metrics.get("_candidate_metadata") if isinstance(metrics, dict) else {}
                probe_op = (candidate_meta or {}).get("probe_operator")

                operators_used = alpha.operators_used or []
                if not operators_used:
                    _, operators_used = self._extract_expression_components(alpha.expression, fields)
                operator_set = {str(op).lower() for op in operators_used}
                probe_op_valid = bool(probe_op and str(probe_op).lower() in operator_set)
                if probe_op_valid:
                    probe_key = str(probe_op).lower()
                    probe_counts[probe_key] = probe_counts.get(probe_key, 0) + 1

                for op in operators_used:
                    key = str(op).lower()
                    if key in regular_ops:
                        used_counts[key] = used_counts.get(key, 0) + 1
                        attempted_counts[key] = attempted_counts.get(key, 0) + 1

                sharpe = self._safe_metric_float(metrics.get("sharpe") if isinstance(metrics, dict) else None)
                fitness = self._safe_metric_float(metrics.get("fitness") if isinstance(metrics, dict) else None)
                sign_reversal_candidate = bool(metrics.get("_sign_reversal_candidate")) or (
                    sharpe < 0 and abs(sharpe) >= reversal_min
                )
                if (
                    sharpe >= weak_sharpe_floor
                    or fitness >= weak_fitness_floor
                    or sign_reversal_candidate
                ):
                    weak_signal_ops.update(str(op).lower() for op in operators_used if str(op).lower() in regular_ops)
                    if probe_op_valid:
                        weak_signal_ops.add(str(probe_op).lower())
                        first_order_signal_ops.add(str(probe_op).lower())
                if alpha.quality_status == "PASS":
                    pass_ops.update(str(op).lower() for op in operators_used if str(op).lower() in regular_ops)
                    if probe_op_valid:
                        pass_ops.add(str(probe_op).lower())

            for failure in failure_rows:
                expression = failure.expression or ""
                _, operators_used = self._extract_expression_components(expression, fields)
                for op in operators_used:
                    key = str(op).lower()
                    if key in regular_ops:
                        attempted_counts[key] = attempted_counts.get(key, 0) + 1

                try:
                    details = json.loads(failure.raw_response or "{}")
                except Exception:
                    details = {}
                candidate_meta = details.get("candidate_metadata") if isinstance(details, dict) else {}
                probe_op = (candidate_meta or {}).get("probe_operator")
                if probe_op:
                    probe_counts[str(probe_op).lower()] = probe_counts.get(str(probe_op).lower(), 0) + 1
                    attempted_counts[str(probe_op).lower()] = attempted_counts.get(str(probe_op).lower(), 0) + 1

            explored_ops = sorted(set(attempted_counts) | set(probe_counts))
            missing_ops = sorted(set(regular_ops) - set(explored_ops))
            first_order_explored_ops = sorted(set(probe_counts))
            first_order_missing_ops = sorted(coverage_denominator_ops - set(first_order_explored_ops))
            global_completed_probe_ops = await self._query_first_order_probe_operators(
                task=task,
                dataset_id=dataset_id,
                completed_only=True,
            )
            global_completed_probe_ops = global_completed_probe_ops & coverage_denominator_ops
            global_deferred_probe_ops = await self._query_deferred_first_order_probe_operators(
                task=task,
                dataset_id=dataset_id,
                completed_ops=global_completed_probe_ops,
            )
            global_deferred_probe_ops = global_deferred_probe_ops & coverage_denominator_ops
            global_first_order_signal_ops, global_first_order_signal_details = (
                await self._query_first_order_signal_operators(
                    task=task,
                    dataset_id=dataset_id,
                    weak_sharpe_floor=weak_sharpe_floor,
                    weak_fitness_floor=weak_fitness_floor,
                    reversal_min=reversal_min,
                    margin_min=float(config.get("margin_min", 0.001)),
                )
            )
            global_first_order_missing_ops = sorted(coverage_denominator_ops - global_completed_probe_ops)
            global_first_order_active_missing_ops = sorted(
                coverage_denominator_ops - global_completed_probe_ops - global_deferred_probe_ops
            )
            snapshot = {
                "dataset_id": dataset_id,
                "round": iteration,
                "regular_operator_count": len(regular_ops),
                "first_order_probeable_operator_count": len(coverage_denominator_ops),
                "first_order_unavailable_operator_count": len(unavailable_probe_ops),
                "first_order_unavailable_operators": unavailable_probe_ops,
                "explored_operator_count": len(explored_ops),
                "coverage_ratio": round(len(explored_ops) / max(1, len(regular_ops)), 4),
                "explored_operators": explored_ops,
                "missing_operators": missing_ops[:50],
                "missing_operator_count": len(missing_ops),
                "first_order_probe_count": sum(probe_counts.values()),
                "first_order_probe_operators": sorted(probe_counts),
                "first_order_explored_operator_count": len(first_order_explored_ops),
                "first_order_coverage_ratio": round(
                    len(first_order_explored_ops) / max(1, len(coverage_denominator_ops)), 4
                ),
                "first_order_missing_operators": first_order_missing_ops[:50],
                "first_order_missing_operator_count": len(first_order_missing_ops),
                "global_completed_first_order_probe_operators": sorted(global_completed_probe_ops),
                "global_completed_first_order_probe_count": len(global_completed_probe_ops),
                "global_deferred_first_order_probe_operators": sorted(global_deferred_probe_ops),
                "global_deferred_first_order_probe_count": len(global_deferred_probe_ops),
                "global_first_order_coverage_ratio": round(
                    len(global_completed_probe_ops) / max(1, len(coverage_denominator_ops)), 4
                ),
                "global_first_order_missing_operators": global_first_order_missing_ops[:50],
                "global_first_order_missing_operator_count": len(global_first_order_missing_ops),
                "global_first_order_active_missing_operators": global_first_order_active_missing_ops[:50],
                "global_first_order_active_missing_operator_count": len(global_first_order_active_missing_ops),
                "weak_signal_operators": sorted(weak_signal_ops),
                "first_order_signal_operators": sorted(first_order_signal_ops),
                "global_first_order_signal_operators": sorted(global_first_order_signal_ops),
                "global_first_order_signal_details": global_first_order_signal_details[:30],
                "pass_operators": sorted(pass_ops),
                "top_attempted_operator_counts": sorted(
                    attempted_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:20],
                "top_operator_counts": sorted(
                    used_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:20],
            }

            trace_service = TraceService(
                self.db,
                task.id,
                initial_step_order=160,
                iteration=iteration,
                run_id=run_id,
            )
            record = trace_service.create_record(
                step_type="OPERATOR_EXPLORATION",
                status="SUCCESS",
                input_data={"dataset_id": dataset_id, "round": iteration},
                output_data=snapshot,
            )
            await trace_service.persist_record(record)
            logger.info(
                "[MiningAgent] Operator exploration snapshot | "
                f"round={iteration} coverage={snapshot['explored_operator_count']}/"
                f"{snapshot['regular_operator_count']} weak={len(weak_signal_ops)} pass={len(pass_ops)}"
            )
        except Exception as e:
            logger.warning(f"[MiningAgent] Operator exploration snapshot failed: {e}")

    async def _query_first_order_signal_operators(
        self,
        task: MiningTask,
        dataset_id: str,
        weak_sharpe_floor: float,
        weak_fitness_floor: float,
        reversal_min: float,
        margin_min: float,
    ) -> Tuple[set, List[Dict[str, Any]]]:
        """Return globally observed first-order operators with usable signal."""
        ops = set()
        details: List[Dict[str, Any]] = []
        rows = (
            await self.db.execute(
                select(Alpha)
                .where(
                    Alpha.region == task.region,
                    Alpha.universe == task.universe,
                    Alpha.dataset_id == dataset_id,
                )
                .order_by(Alpha.id.desc())
            )
        ).scalars().all()

        for alpha in rows:
            metrics = alpha.metrics or alpha.is_metrics or {}
            if not isinstance(metrics, dict):
                continue
            candidate_meta = metrics.get("_candidate_metadata") or {}
            if candidate_meta.get("source") != "first_order_operator_probe":
                continue
            probe_op = candidate_meta.get("probe_operator")
            if not probe_op or not self._expression_contains_operator(alpha.expression, str(probe_op)):
                continue

            sharpe = self._safe_metric_float(metrics.get("sharpe"))
            fitness = self._safe_metric_float(metrics.get("fitness"))
            margin = self._safe_metric_float(metrics.get("margin"))
            sign_reversal_candidate = bool(metrics.get("_sign_reversal_candidate")) or (
                sharpe < 0 and abs(sharpe) >= reversal_min
            )
            has_signal = (
                sharpe >= weak_sharpe_floor
                or fitness >= weak_fitness_floor
                or margin >= margin_min
                or sign_reversal_candidate
            )
            if not has_signal:
                continue

            op = str(probe_op).lower()
            ops.add(op)
            details.append({
                "operator": op,
                "alpha_id": alpha.alpha_id,
                "expression": alpha.expression,
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover": self._safe_metric_float(metrics.get("turnover")),
                "margin": margin,
                "quality_status": alpha.quality_status,
                "sign_reversal_candidate": sign_reversal_candidate,
            })

        details.sort(
            key=lambda item: (
                abs(float(item.get("sharpe") or 0.0)),
                float(item.get("fitness") or 0.0),
                float(item.get("margin") or 0.0),
            ),
            reverse=True,
        )
        return ops, details

    def _regular_operator_names(self, operators: List[Dict]) -> set:
        names = set()
        for op in operators or []:
            name = str(op.get("name") if isinstance(op, dict) else op or "").lower()
            if not name:
                continue
            scope = op.get("scope") if isinstance(op, dict) else None
            if scope and "REGULAR" not in {str(item).upper() for item in scope}:
                continue
            names.add(name)
        return names

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
        if str(task.region).upper() == "IND":
            focus.extend([
                "IND/D1: prioritize compact risk-aware expressions with <=5 operator calls.",
                "Use comparable-scale construction or residualize one primary signal against risk, returns, volatility, liquidity, or crowding.",
                "Prefer group_rank/group_neutralize within industry/subindustry and let settings sweeps test CROWDING/FAST/SLOW/SLOW_AND_FAST.",
            ])

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
        iteration: int,
        task_config: Optional[Dict[str, Any]] = None,
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
        
        # Extract metrics from all simulated alphas. Failed quality alphas are
        # still the main signal for weak-signal and sign-reversal diagnosis.
        simulated = [a for a in alphas if self._alpha_was_simulated(a)]
        if simulated:
            sharpes = []
            fitnesses = []
            rn_sharpes = []
            rn_fitnesses = []
            turnovers = []
            
            for a in simulated:
                metrics = getattr(a, "metrics", {}) or {}
                if isinstance(metrics, dict):
                    if metrics.get("sharpe") is not None:
                        sharpes.append(metrics["sharpe"])
                    if metrics.get("fitness") is not None:
                        fitnesses.append(metrics["fitness"])
                    rn_metrics = metrics.get("riskNeutralized") or {}
                    if isinstance(rn_metrics, dict):
                        rn_sharpe = rn_metrics.get("sharpe")
                        rn_fitness = rn_metrics.get("fitness")
                    else:
                        rn_sharpe = None
                        rn_fitness = None
                    if rn_sharpe is None:
                        rn_sharpe = metrics.get("risk_neutralized_sharpe", metrics.get("rn_sharpe"))
                    if rn_fitness is None:
                        rn_fitness = metrics.get("risk_neutralized_fitness", metrics.get("rn_fitness"))
                    if rn_sharpe is not None:
                        rn_sharpes.append(self._safe_metric_float(rn_sharpe))
                    if rn_fitness is not None:
                        rn_fitnesses.append(self._safe_metric_float(rn_fitness))
                    if metrics.get("turnover") is not None:
                        turnovers.append(metrics["turnover"])
            
            if sharpes:
                result.best_sharpe = max(sharpes)
                result.best_abs_sharpe = max(abs(self._safe_metric_float(value)) for value in sharpes)
                result.avg_sharpe = sum(sharpes) / len(sharpes)
            if fitnesses:
                result.best_fitness = max(fitnesses)
                result.best_abs_fitness = max(abs(self._safe_metric_float(value)) for value in fitnesses)
                result.avg_fitness = sum(fitnesses) / len(fitnesses)
            if rn_sharpes:
                result.best_rn_sharpe = max(rn_sharpes)
            if rn_fitnesses:
                result.best_rn_fitness = max(rn_fitnesses)
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
            task_id=task_id,
            task_config=task_config or {},
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
        task_id: str,
        task_config: Optional[Dict[str, Any]] = None,
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
        task_config = task_config or {}
        min_sharpe = float(task_config.get("optimization_min_sharpe", 0.6))
        min_fitness = float(task_config.get("optimization_min_fitness", 0.25))
        reversal_min = float(task_config.get("optimization_reversal_abs_sharpe_min", 0.8))
        
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

            sign_reversal_candidate = (
                sharpe < 0
                and abs(sharpe) >= reversal_min
            )

            actionable = (
                sign_reversal_candidate
                or (sharpe >= min_sharpe and fitness >= min_fitness)
            )

            # If explicit optimize status, always include
            if status == "OPTIMIZE":
                if not actionable:
                    logger.info(
                        "[MiningAgent] Ignoring low-ROI OPTIMIZE candidate before strategy switch | "
                        f"sharpe={sharpe:.2f} fitness={fitness:.2f} "
                        f"min_sharpe={min_sharpe:.2f} min_fitness={min_fitness:.2f} "
                        f"expr={a.expression[:80]}"
                    )
                    continue
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
            
            if should_opt and actionable:
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
    ) -> int:
        """
        Run optimization chain on promising weak alphas.
        
        This is the Chain-of-Alpha style optimization loop.
        """
        from backend.optimization_chain import (
            _build_optimization_context,
            generate_local_rewrites,
            generate_settings_variants,
        )
        
        logger.info(f"[MiningAgent] Running optimization chain on {len(candidates)} candidates")
        
        task_config = task.config or {}
        max_candidates = int(task_config.get("optimization_max_candidates", 1))
        max_rewrites = int(task_config.get("optimization_max_rewrites", 10))
        pass_count = 0
        round_batch_budget = max(0, int(task_config.get("optimization_max_batches_per_round", 1)))

        for candidate in candidates[:max_candidates]:
            expression = candidate.get("expression", "")
            metrics = candidate.get("metrics", {})
            reason = candidate.get("reason", "")
            
            if not expression:
                continue

            sharpe = self._safe_metric_float(metrics.get("sharpe"))
            fitness = self._safe_metric_float(metrics.get("fitness"))
            min_sharpe = float((task.config or {}).get("optimization_min_sharpe", 0.6))
            min_fitness = float((task.config or {}).get("optimization_min_fitness", 0.25))
            reversal_min = float((task.config or {}).get("optimization_reversal_abs_sharpe_min", 0.8))
            is_reversal = bool(metrics.get("_sign_reversal_candidate")) and abs(sharpe) >= reversal_min

            if not is_reversal and (sharpe < min_sharpe or fitness < min_fitness):
                logger.info(
                    "[MiningAgent] Skipping low-ROI optimization candidate | "
                    f"sharpe={sharpe:.2f} fitness={fitness:.2f} "
                    f"min_sharpe={min_sharpe:.2f} min_fitness={min_fitness:.2f} "
                    f"expr={expression[:80]}"
                )
                continue
            
            try:
                # Generate expression variants
                expr_variants = generate_local_rewrites(
                    expression=expression,
                    sim_result=metrics,
                    feedback=reason,
                    max_variants=max_rewrites,
                )
                
                base_settings = self._optimization_base_settings(task)
                context = _build_optimization_context(expression, metrics)
                settings_variants = [
                    {
                        **base_settings,
                        "description": "Task base settings",
                        "change_type": "base",
                    },
                    *generate_settings_variants(base_settings, context=context),
                ]
                settings_limit = int(task_config.get("optimization_settings_limit", 2))
                settings_variants = settings_variants[:max(1, settings_limit)]
                
                # Simulate top variants (budget-limited)
                if round_batch_budget <= 0:
                    logger.info(
                        "[MiningAgent] Optimization round batch budget exhausted | "
                        f"iteration={iteration} dataset={dataset_id}"
                    )
                    break

                result = await self._simulate_optimization_variants(
                    task=task,
                    original_expression=expression,
                    expr_variants=expr_variants[:5],
                    settings_variants=settings_variants[:3],
                    iteration=iteration,
                    dataset_id=dataset_id,
                    fields=fields or [],
                    run_id=run_id,
                    max_batches_override=round_batch_budget,
                )
                pass_count += result.get("pass_total", 0)
                round_batch_budget -= result.get("batches_run", 0)

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

        return pass_count

    def _safe_metric_float(self, value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _optimization_base_settings(self, task: MiningTask) -> Dict[str, Any]:
        """Use the task's real simulation settings for optimization sweeps."""
        config = task.config or {}
        return {
            "delay": int(config.get("delay", 1)),
            "neutralization": config.get("neutralization", "SUBINDUSTRY"),
            "decay": int(config.get("decay", 4)),
            "truncation": float(config.get("truncation", 0.08)),
            "test_period": config.get("test_period", "P2Y0M"),
        }

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

            base_settings = self._optimization_base_settings(task)
            report = await run_genetic_optimization(
                seed_expression=seed_expression,
                seed_metrics=seed_metrics,
                simulate_func=simulate_for_genetic,
                config=config,
                region=task.region,
                universe=task.universe,
                delay=base_settings["delay"],
                decay=base_settings["decay"],
                neutralization=base_settings["neutralization"],
                truncation=base_settings["truncation"],
                test_period=base_settings["test_period"],
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
                    delay=base_settings["delay"],
                    decay=base_settings["decay"],
                    neutralization=base_settings["neutralization"],
                    truncation=base_settings["truncation"],
                    settings=base_settings,
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
        max_batches_override: Optional[int] = None,
    ) -> Dict[str, int]:
        """Simulate optimization variants and save improvements."""
        logger.info(
            f"[MiningAgent] Simulating {len(expr_variants)} variants for optimization"
        )
        
        base_settings = self._optimization_base_settings(task)
        raw_settings_to_try = settings_variants or [{
            **base_settings,
            "description": "Base settings",
        }]
        settings_to_try = [{**base_settings, **variant} for variant in raw_settings_to_try]

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
        try:
            from backend.alpha_semantic_validator import AlphaSemanticValidator

            semantic_validator = AlphaSemanticValidator(
                fields=fields or [],
                strict_field_check=True,
            )
        except Exception as exc:
            logger.warning(f"[MiningAgent] Optimization semantic validator unavailable: {exc}")
            semantic_validator = None

        for variant in expr_variants:
            expression = variant.get("expression")
            if not expression:
                continue
            if semantic_validator:
                semantic_result = semantic_validator.validate(expression)
                if not semantic_result.valid:
                    logger.info(
                        "[MiningAgent] Skipping optimization variant due to semantic errors: "
                        f"{semantic_result.errors[:2]} expr={expression[:80]}"
                    )
                    continue
            if _validate_task_constraints:
                constraint_errors = _validate_task_constraints(
                    expression=expression,
                    allowed_fields=allowed_fields,
                    task_config=task.config or {},
                    fields=fields or [],
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

        existing_signatures = set()
        try:
            existing_res = await self.db.execute(
                select(
                    Alpha.expression_hash,
                    Alpha.delay,
                    Alpha.decay,
                    Alpha.neutralization,
                    Alpha.truncation,
                ).where(Alpha.task_id == task.id)
            )
            for expr_hash, delay, decay, neutralization, truncation in existing_res.all():
                existing_signatures.add((
                    expr_hash,
                    int(delay or 0),
                    int(decay or 0),
                    str(neutralization or "NONE"),
                    round(float(truncation or 0), 6),
                ))
        except Exception as e:
            logger.warning(f"[MiningAgent] Could not load optimization dedup signatures: {e}")

        grouped_jobs: Dict[tuple, List[tuple]] = {}
        seen_signatures = set(existing_signatures)
        for job in simulation_jobs:
            settings_variant = job[2]
            expression = job[1]
            settings_key = (
                settings_variant.get("delay", (task.config or {}).get("delay", 1)),
                settings_variant.get("decay", 4),
                settings_variant.get("neutralization", "SUBINDUSTRY"),
                settings_variant.get("truncation", 0.08),
            )
            signature = (
                compute_expression_hash(expression),
                int(settings_key[0]),
                int(settings_key[1]),
                str(settings_key[2]),
                round(float(settings_key[3]), 6),
            )
            if signature in seen_signatures:
                logger.info(
                    "[MiningAgent] Skipping duplicate optimization variant | "
                    f"settings={settings_key} expr={expression[:80]}"
                )
                continue
            seen_signatures.add(signature)
            grouped_jobs.setdefault(settings_key, []).append(job)

        configured_max_batches = int((task.config or {}).get("optimization_max_batches_per_candidate", 1))
        max_batches = configured_max_batches if max_batches_override is None else max_batches_override
        batches_run = 0
        saved_total = 0
        pass_total = 0
        for (delay, decay, neutralization, truncation), jobs in grouped_jobs.items():
            for offset in range(0, len(jobs), 4):
                if batches_run >= max_batches:
                    logger.info(
                        "[MiningAgent] Optimization batch budget reached | "
                        f"max_batches={max_batches} original={original_expression[:80]}"
                    )
                    break
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
                        delay=int(delay),
                        decay=decay,
                        neutralization=neutralization,
                        truncation=truncation,
                        test_period=(task.config or {}).get("test_period", "P2Y0M"),
                        max_wait=int((task.config or {}).get("optimization_simulation_max_wait", 300)),
                        timeout_grace_seconds=int((task.config or {}).get("optimization_timeout_grace_seconds", 60)),
                    )
                    batches_run += 1
                except Exception as e:
                    logger.warning(f"Optimization batch simulation failed: {e}")
                    continue

                for (variant, expression, settings_variant), result in zip(batch, results):
                    saved_status = await self._save_optimization_result(
                        task=task,
                        original_expression=original_expression,
                        variant=variant,
                        expression=expression,
                        settings_variant=settings_variant,
                        result=result,
                        dataset_id=dataset_id,
                        run_id=run_id,
                        fields=fields or [],
                        compute_expression_hash=compute_expression_hash,
                    )
                    saved_total += int(bool(saved_status))
                    pass_total += int(saved_status == "PASS")

                # Persist after every multi-sim batch. A later slow batch or
                # timeout should not erase useful observations from earlier
                # batches in the optimization chain.
                await self.db.commit()
            if batches_run >= max_batches:
                break

        logger.info(
            "[MiningAgent] Optimization variants complete | "
            f"batches={batches_run} saved={saved_total} strict_pass={pass_total} "
            f"original={original_expression[:80]}"
        )
        return {
            "batches_run": batches_run,
            "saved_total": saved_total,
            "pass_total": pass_total,
        }

    async def run_expression_settings_sweep(
        self,
        task: MiningTask,
        dataset_id: str,
        seed_expressions: List[Dict[str, Any]],
        settings_variants: List[Dict[str, Any]],
        fields: Optional[List[Dict]] = None,
        run_id: Optional[int] = None,
    ) -> Dict[str, int]:
        """Retest promising seed expressions across settings through the project pipeline."""
        if not seed_expressions or not settings_variants:
            return {"jobs": 0, "batches": 0, "saved": 0, "strict_pass": 0, "skipped": 0}

        base_settings = self._optimization_base_settings(task)
        normalized_settings = [
            {**base_settings, **variant}
            for variant in settings_variants
        ]
        batch_size = max(2, int((task.config or {}).get("settings_sweep_batch_size", 4)))
        max_batches = int((task.config or {}).get("settings_sweep_max_batches", 6))

        from backend.alpha_semantic_validator import compute_expression_hash

        grouped_jobs: Dict[tuple, List[tuple]] = {}
        skipped = 0
        seed_hashes = [
            compute_expression_hash(str(seed.get("expression") or ""))
            for seed in seed_expressions
            if seed.get("expression")
        ]
        existing_signatures = set()
        if seed_hashes:
            try:
                existing_res = await self.db.execute(
                    select(
                        Alpha.expression_hash,
                        Alpha.delay,
                        Alpha.decay,
                        Alpha.neutralization,
                        Alpha.truncation,
                    ).where(
                        Alpha.region == task.region,
                        Alpha.universe == task.universe,
                        Alpha.expression_hash.in_(seed_hashes),
                    )
                )
                for expr_hash, delay, decay, neutralization, truncation in existing_res.all():
                    existing_signatures.add((
                        expr_hash,
                        int(delay or 0),
                        int(decay or 0),
                        str(neutralization or "NONE"),
                        round(float(truncation or 0), 6),
                    ))
            except Exception as exc:
                logger.warning(f"[MiningAgent] Could not load sweep dedup signatures: {exc}")

        seen_signatures = set(existing_signatures)
        for setting in normalized_settings:
            settings_key = (
                int(setting.get("delay", (task.config or {}).get("delay", 1))),
                int(setting.get("decay", 4)),
                str(setting.get("neutralization", "SUBINDUSTRY")),
                round(float(setting.get("truncation", 0.08)), 6),
            )
            for seed in seed_expressions:
                expression = str(seed.get("expression") or "").strip()
                if not expression:
                    skipped += 1
                    continue
                signature = (compute_expression_hash(expression), *settings_key)
                if signature in seen_signatures:
                    skipped += 1
                    continue
                seen_signatures.add(signature)
                grouped_jobs.setdefault(settings_key, []).append((seed, expression, setting))

        jobs_total = sum(len(jobs) for jobs in grouped_jobs.values())
        batches_run = 0
        saved_total = 0
        pass_total = 0

        for (delay, decay, neutralization, truncation), jobs in grouped_jobs.items():
            batches = [
                jobs[offset: offset + batch_size]
                for offset in range(0, len(jobs), batch_size)
            ]
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2].extend(batches[-1])
                batches.pop()

            for batch in batches:
                if batches_run >= max_batches:
                    logger.info(
                        "[MiningAgent] Settings sweep batch budget reached | "
                        f"max_batches={max_batches} dataset={dataset_id}"
                    )
                    break
                if len(batch) < 2:
                    skipped += len(batch)
                    logger.info(
                        "[MiningAgent] Skipping single settings-sweep job to preserve "
                        f"multi-simulation-only constraint | setting={(delay, decay, neutralization, truncation)}"
                    )
                    continue

                expressions = [job[1] for job in batch]
                try:
                    results = await self.brain.simulate_batch(
                        expressions=expressions,
                        region=task.region,
                        universe=task.universe,
                        delay=delay,
                        decay=decay,
                        neutralization=neutralization,
                        truncation=truncation,
                        test_period=(task.config or {}).get("test_period", "P2Y0M"),
                        max_wait=int((task.config or {}).get("simulation_max_wait", 900)),
                        timeout_grace_seconds=int((task.config or {}).get("simulation_timeout_grace_seconds", 180)),
                        no_child_timeout_seconds=int((task.config or {}).get("simulation_no_child_timeout_seconds", 0)),
                    )
                    batches_run += 1
                except Exception as exc:
                    logger.warning(f"[MiningAgent] Settings sweep simulation failed: {exc}")
                    continue

                settings_variant = {
                    **base_settings,
                    "delay": delay,
                    "decay": decay,
                    "neutralization": neutralization,
                    "truncation": truncation,
                    "description": f"Settings sweep d{delay} decay{decay} {neutralization} trunc{truncation:g}",
                }
                for (seed, expression, _setting), result in zip(batch, results):
                    if not result.get("success"):
                        failure_details = {
                            "alpha_id": None,
                            "quality_status": "FAIL",
                            "metrics": {
                                "checks": [],
                                "can_submit": False,
                                "failed_checks": [],
                                "pending_checks": [],
                                "passed_checks": [],
                                "stage": None,
                                "status": None,
                                "_settings": settings_variant,
                                "_raw_response": result.get("raw_response") or result.get("raw"),
                                "_simulation_location": result.get("location"),
                            },
                            "candidate_metadata": {
                                "source": "settings_sweep",
                                "dataset_id": dataset_id,
                                "settings_variant": settings_variant.get("description"),
                                "seed_description": seed.get("description"),
                            },
                            "is_simulated": True,
                            "simulation_success": False,
                        }
                        self.db.add(AlphaFailure(
                            task_id=task.id,
                            run_id=run_id,
                            expression=expression[:2000],
                            error_type="SIMULATION_ERROR",
                            error_message=str(result.get("error") or "Settings sweep simulation failed")[:500],
                            raw_response=json.dumps(failure_details, ensure_ascii=False, default=str),
                        ))
                        skipped += 1
                        continue

                    variant = {
                        "expression": expression,
                        "description": seed.get("description") or "Settings sweep seed",
                    }
                    saved_status = await self._save_optimization_result(
                        task=task,
                        original_expression=expression,
                        variant=variant,
                        expression=expression,
                        settings_variant=settings_variant,
                        result=result,
                        dataset_id=dataset_id,
                        run_id=run_id,
                        fields=fields or [],
                        compute_expression_hash=compute_expression_hash,
                    )
                    saved_total += int(bool(saved_status))
                    pass_total += int(saved_status == "PASS")

                await self.db.commit()

            if batches_run >= max_batches:
                break

        logger.info(
            "[MiningAgent] Settings sweep complete | "
            f"jobs={jobs_total} batches={batches_run} saved={saved_total} "
            f"strict_pass={pass_total} skipped={skipped} dataset={dataset_id}"
        )
        return {
            "jobs": jobs_total,
            "batches": batches_run,
            "saved": saved_total,
            "strict_pass": pass_total,
            "skipped": skipped,
        }

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
        fields: List[Dict],
        compute_expression_hash,
    ) -> Optional[str]:
        """Persist one optimization result if it clears the lightweight improvement screen."""
        if not result.get("success"):
            return None

        metrics = result.get("metrics", {})
        try:
            sharpe = float(metrics.get("sharpe") or 0)
        except (TypeError, ValueError):
            sharpe = 0

        metrics = {
            **metrics,
            "checks": result.get("checks", metrics.get("checks", [])),
            "can_submit": result.get("can_submit", metrics.get("can_submit", False)),
            "failed_checks": result.get("failed_checks", metrics.get("failed_checks", [])),
            "pending_checks": result.get("pending_checks", metrics.get("pending_checks", [])),
            "passed_checks": result.get("passed_checks", metrics.get("passed_checks", [])),
            "stage": result.get("stage", metrics.get("stage")),
            "status": result.get("status", metrics.get("status")),
            "_settings": result.get("settings", {}),
            "_optimization_of": original_expression,
            "_optimization_variant": variant.get("description"),
            "_optimization_settings": settings_variant.get("description"),
        }
        quality_status = await self._quality_status_for_optimization_result(
            task=task,
            expression=expression,
            metrics=metrics,
            fields=fields,
            alpha_id=result.get("alpha_id"),
        )

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
            delay=int(settings_variant.get("delay", (task.config or {}).get("delay", 1))),
            decay=settings_variant.get("decay", 4),
            neutralization=settings_variant.get("neutralization", "SUBINDUSTRY"),
            truncation=settings_variant.get("truncation", 0.08),
            settings=settings_variant,
            status=result.get("status") or "simulated",
            stage=metrics.get("stage") or "IS",
            quality_status=quality_status,
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
        logger.info(
            "[MiningAgent] Optimization saved: "
            f"status={quality_status} expr={expression[:30]} sharpe={sharpe}"
        )
        return quality_status

    async def _quality_status_for_optimization_result(
        self,
        task: MiningTask,
        expression: str,
        metrics: Dict[str, Any],
        fields: List[Dict],
        alpha_id: Optional[str],
    ) -> str:
        """Apply the same strict deliverable gates to optimization-chain output."""
        from backend.alpha_scoring import evaluate_with_brain_checks
        from backend.agents.graph.nodes.evaluation import _strict_gate_failures

        config = task.config or {}
        thresholds = {
            "sharpe_min": float(config.get("sharpe_min", 1.58)),
            "two_year_sharpe_min": float(config.get("two_year_sharpe_min", 1.6)),
            "fitness_min": float(config.get("fitness_min", 1.0)),
            "rn_sharpe_min": float(config.get("rn_sharpe_min", 1.58)),
            "rn_fitness_min": float(config.get("rn_fitness_min", 1.0)),
            "margin_min": float(config.get("margin_min", 0.001)),
            "turnover_min": float(config.get("turnover_min", 0.05)),
            "turnover_max": float(config.get("turnover_max", 0.30)),
            "prod_corr_max": float(config.get("prod_corr_max", 0.7)),
            "self_corr_max": float(config.get("self_corr_max", 0.5)),
            "ra_fails_max": int(config.get("ra_fails_max", 0)),
            "max_operator_count": int(config.get("max_operator_count", 5)),
        }
        sim_result = {
            "is": {
                "sharpe": metrics.get("sharpe", 0),
                "fitness": metrics.get("fitness", 0),
                "turnover": metrics.get("turnover", 0),
                "drawdown": metrics.get("drawdown", 0),
                "margin": metrics.get("margin", 0),
                "checks": metrics.get("checks", []),
            },
            "riskNeutralized": metrics.get("riskNeutralized", {}),
            "investabilityConstrained": metrics.get("investabilityConstrained", {}),
            "checks": metrics.get("checks", []),
            "can_submit": metrics.get("can_submit", False),
        }
        brain_eval = evaluate_with_brain_checks(sim_result)
        prod_corr = None
        self_corr = None

        local_failures = _strict_gate_failures(
            metrics=metrics,
            brain_failed_checks=brain_eval.get("failed_checks", []),
            prod_corr=0.0,
            self_corr=0.0,
            thresholds=thresholds,
            expression=expression,
            fields=fields,
        )
        local_failures = [
            failure
            for failure in local_failures
            if failure not in {"PROD_CORR_MISSING", "SELF_CORR_MISSING"}
            and not failure.startswith("HIGH_PROD_CORR")
            and not failure.startswith("HIGH_SELF_CORR")
        ]
        if local_failures or not alpha_id:
            metrics.update({
                "_strict_gate_failures": local_failures,
                "_hard_pass": False,
                "_corr_checked": False,
            })
            return "OPTIMIZE"

        try:
            self_corr_result = await self.brain.check_correlation(alpha_id, check_type="SELF")
            if isinstance(self_corr_result, dict) and self_corr_result.get("max") is not None:
                self_corr = float(self_corr_result["max"])
        except Exception as exc:
            logger.warning(f"[MiningAgent] Optimization SELF correlation failed for {alpha_id}: {exc}")

        if self_corr is not None and self_corr < thresholds["self_corr_max"]:
            try:
                prod_corr_result = await self.brain.check_correlation(alpha_id, check_type="PROD")
                if isinstance(prod_corr_result, dict) and prod_corr_result.get("max") is not None:
                    prod_corr = float(prod_corr_result["max"])
            except Exception as exc:
                logger.warning(f"[MiningAgent] Optimization PROD correlation failed for {alpha_id}: {exc}")

        strict_failures = _strict_gate_failures(
            metrics=metrics,
            brain_failed_checks=brain_eval.get("failed_checks", []),
            prod_corr=prod_corr,
            self_corr=self_corr,
            thresholds=thresholds,
            expression=expression,
            fields=fields,
        )
        metrics.update({
            "_prod_corr": round(prod_corr, 4) if prod_corr is not None else None,
            "_self_corr": round(self_corr, 4) if self_corr is not None else None,
            "_corr_checked": True,
            "_strict_gate_failures": strict_failures,
            "_hard_pass": not strict_failures,
            "_brain_failed_checks": brain_eval.get("failed_checks", []),
            "_brain_pending_checks": brain_eval.get("pending_checks", []),
        })
        return "PASS" if not strict_failures else "OPTIMIZE"
    
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
