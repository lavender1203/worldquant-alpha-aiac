"""
LangGraph Mining Workflow
Orchestrates the complete alpha mining state graph
"""

from typing import Dict, List, Optional, Any, Annotated
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from backend.agents.graph.state import MiningState, AlphaResult, FailureRecord
from backend.agents.graph.nodes import (
    node_rag_query,
    node_distill_context,
    node_hypothesis,
    node_code_gen,
    node_validate,
    node_self_correct,
    node_simulate,
    node_evaluate,
    node_save_results
)
from backend.agents.graph.edges import (
    route_after_validate,
    route_check_error
)
from backend.agents.services import LLMService, RAGService, get_llm_service
from backend.adapters.brain_adapter import BrainAdapter
from backend.models import MiningTask


class MiningWorkflow:
    """
    LangGraph-based mining workflow.
    
    Features:
    - Strongly typed state (Pydantic)
    - Conditional edges for self-correction loops
    - Full trace recording
    - Configurable checkpointing
    """
    
    def __init__(
        self,
        db: AsyncSession,
        brain: BrainAdapter = None,
        llm_service: LLMService = None,
        checkpointer: Optional[BaseCheckpointSaver] = None
    ):
        self.db = db
        self.brain = brain or BrainAdapter()
        self.llm_service = llm_service or get_llm_service()
        self.rag_service = RAGService(db)
        self.checkpointer = checkpointer
        
        self._graph = self._build_graph()
        
        logger.info("[MiningWorkflow] Initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the mining state graph.
        
        Graph structure (Batch):
        START -> rag_query -> distill_context -> hypothesis -> code_gen 
                 -> validate <--> self_correct
                    | (All processed)
                    v
                 simulate -> evaluate -> save_results -> END
        """
        # Create graph with state type
        workflow = StateGraph(MiningState)
        
        # =====================================================================
        # Add Nodes
        # =====================================================================
        
        # RAG query node (bind dependencies)
        workflow.add_node(
            "rag_query",
            partial(node_rag_query, rag_service=self.rag_service)
        )

        # Distill Context node
        workflow.add_node(
            "distill_context",
            partial(node_distill_context, llm_service=self.llm_service)
        )
        
        # Hypothesis node
        workflow.add_node(
            "hypothesis",
            partial(node_hypothesis, llm_service=self.llm_service)
        )
        
        # Code Generation node
        workflow.add_node(
            "code_gen",
            partial(node_code_gen, llm_service=self.llm_service)
        )
        
        # Validation node (no external deps)
        workflow.add_node("validate", node_validate)
        
        # Self-correction node
        workflow.add_node(
            "self_correct",
            partial(node_self_correct, llm_service=self.llm_service)
        )
        
        # Simulation node
        workflow.add_node(
            "simulate",
            partial(node_simulate, brain=self.brain)
        )
        
        # Evaluation node
        workflow.add_node(
            "evaluate",
            partial(node_evaluate, brain=self.brain, rag_service=self.rag_service)
        )
        
        # Save results node (handles both success and failure saving)
        workflow.add_node("save_results", node_save_results)
        
        
        # =====================================================================
        # Add Edges
        # =====================================================================
        
        # Linear flow: START → rag_query → distill_context → hypothesis → code_gen → validate
        workflow.set_entry_point("rag_query")
        workflow.add_edge("rag_query", "distill_context")
        workflow.add_edge("distill_context", "hypothesis")
        workflow.add_edge("hypothesis", "code_gen")
        workflow.add_edge("code_gen", "validate")
        
        # After validate: conditional routing (Batch)
        workflow.add_conditional_edges(
            "validate",
            route_after_validate,
            {
                "simulate": "simulate",
                "self_correct": "self_correct"
            }
        )
        
        # After self-correct: Always back to validate (to re-check fixes)
        workflow.add_edge("self_correct", "validate")
        
        # After simulate: Unconditional -> evaluate
        workflow.add_edge("simulate", "evaluate")
        
        # After evaluate: Unconditional -> save_results
        workflow.add_edge("evaluate", "save_results")
        
        # After save_results: END
        workflow.add_edge("save_results", END)
        
        return workflow
    
    def compile(self):
        """Compile the graph with optional checkpointer."""
        return self._graph.compile(checkpointer=self.checkpointer)
    
    async def run(
        self,
        task: MiningTask,
        dataset_id: str,
        fields: List[Dict],
        operators: List[str],
        num_alphas: int = 3,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute the mining workflow.
        
        Args:
            task: Mining task instance
            dataset_id: Dataset to mine
            fields: Available data fields
            operators: Available operators
            num_alphas: Target number of alphas
            
        Returns:
            Dictionary with generated_alphas and failures
        """
        logger.info(
            f"[MiningWorkflow] 开始执行 | "
            f"task={task.id} dataset={dataset_id} target={num_alphas}"
        )
        
        # Initialize state
        initial_state = MiningState(
            task_id=task.id,
            region=task.region,
            universe=task.universe,
            dataset_id=dataset_id,
            fields=fields,
            operators=operators,
            num_alphas_target=num_alphas
        )
        
        # Compile and run
        app = self.compile()
        
        # Execute graph (Synchronous-style for full state)
        # We use invoke to ensure we get the accumulated final state, NOT just partial updates
        final_state = await app.ainvoke(initial_state, config=config)
        
        # Log completion
        logger.info("[MiningWorkflow] Worklfow execution finished")
        
        # Get results
        generated_alphas = []
        failures = []
        
        if hasattr(final_state, 'generated_alphas'):
            generated_alphas = final_state.generated_alphas
        elif isinstance(final_state, dict):
            generated_alphas = final_state.get('generated_alphas', [])
        
        if hasattr(final_state, 'failures'):
            failures = final_state.failures
        elif isinstance(final_state, dict):
            failures = final_state.get('failures', [])
        
        logger.info(
            f"[MiningWorkflow] 执行完成 | "
            f"success={len(generated_alphas)} failed={len(failures)}"
        )
        
        return {
            "generated_alphas": generated_alphas,
            "failures": failures,
            "trace_steps": final_state.trace_steps if hasattr(final_state, 'trace_steps') else []
        }
    
    async def run_with_persistence(
        self,
        task: MiningTask,
        dataset_id: str,
        fields: List[Dict],
        operators: List[str],
        num_alphas: int = 3,
        config: Dict[str, Any] = None
    ):
        """
        Execute workflow and persist results to database.
        """
        from backend.models import Alpha, AlphaFailure, TraceStep
        
        result = await self.run(task, dataset_id, fields, operators, num_alphas, config)

        configurable = (config or {}).get("configurable", {})
        run_id = configurable.get("run_id")
        
        try:
            import json
            # Persist alphas
            # P0-fix-2: Import hash function for deduplication
            from backend.alpha_semantic_validator import compute_expression_hash
            
            for alpha_result in result.get("generated_alphas", []):
                try:
                    # P0-fix-2: Compute expression hash for DB-level deduplication
                    expr_hash = compute_expression_hash(alpha_result.expression) if alpha_result.expression else None
                    
                    alpha = Alpha(
                        task_id=task.id,
                        run_id=run_id,
                        alpha_id=alpha_result.alpha_id,
                        expression=alpha_result.expression,
                        expression_hash=expr_hash,  # P0-fix-2: Enable DB deduplication
                        hypothesis=alpha_result.hypothesis,
                        logic_explanation=alpha_result.explanation,
                        region=task.region,
                        universe=task.universe,
                        dataset_id=dataset_id,
                        quality_status=alpha_result.quality_status,
                        metrics=alpha_result.metrics
                    )
                    self.db.add(alpha)
                except Exception as e:
                    logger.warning(f"[MiningWorkflow] Failed to add alpha: {e}")
            
            # Persist failures
            for failure in result.get("failures", []):
                try:
                    raw_response = None
                    if hasattr(failure, "details") and failure.details is not None:
                        try:
                            raw_response = json.dumps(failure.details, ensure_ascii=False, default=str)
                        except Exception:
                            raw_response = str(failure.details)

                    fail_record = AlphaFailure(
                        task_id=task.id,
                        run_id=run_id,
                        expression=failure.expression[:2000] if failure.expression else None,  # Limit length
                        error_type=failure.error_type,
                        error_message=failure.error_message[:500] if failure.error_message else None,  # Limit length
                        raw_response=raw_response[:20000] if raw_response else None,  # Keep compact evidence for feedback
                    )
                    self.db.add(fail_record)
                except Exception as e:
                    logger.warning(f"[MiningWorkflow] Failed to add failure record: {e}")
            
            # Persist trace steps (ONLY if TraceService was NOT used)
            # If TraceService is in config, we assume it handled real-time persistence
            has_realtime_trace = config and config.get("configurable", {}).get("trace_service")
            
            if not has_realtime_trace:
                for trace in result.get("trace_steps", []):
                    try:
                        step = TraceStep(
                            task_id=task.id,
                            run_id=run_id,
                            step_type=trace.step_type,
                            step_order=trace.step_order,
                            input_data=trace.input_data,
                            output_data=trace.output_data,
                            duration_ms=trace.duration_ms,
                            status=trace.status,
                            error_message=trace.error_message
                        )
                        self.db.add(step)
                    except Exception as e:
                        logger.warning(f"[MiningWorkflow] Failed to add trace step: {e}")
            
            await self.db.commit()
            logger.info(f"[MiningWorkflow] 持久化完成 | task={task.id}")
            
        except Exception as e:
            logger.error(f"[MiningWorkflow] Persistence failed: {e}")
            # Rollback failed transaction to allow subsequent operations
            try:
                await self.db.rollback()
            except Exception:
                pass
            # Don't raise - return result anyway so mining continues
        
        return result


def create_mining_graph(
    db: AsyncSession,
    brain: BrainAdapter = None,
    llm_service: LLMService = None
) -> MiningWorkflow:
    """
    Factory function to create mining workflow.
    
    Usage:
        workflow = create_mining_graph(db, brain)
        result = await workflow.run(task, dataset_id, fields, operators)
    """
    return MiningWorkflow(
        db=db,
        brain=brain,
        llm_service=llm_service
    )
