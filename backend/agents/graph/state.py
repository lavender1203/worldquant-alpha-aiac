"""
LangGraph State Definitions
Strongly typed state using Pydantic for the mining workflow
"""

from typing import List, Dict, Optional, Any, Annotated
from pydantic import BaseModel, Field
from datetime import datetime
from operator import add


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AlphaCandidate(BaseModel):
    """A candidate alpha expression to be validated and simulated."""
    expression: str
    hypothesis: Optional[str] = None
    explanation: Optional[str] = None
    expected_sharpe: Optional[float] = None
    
    # Validation state
    is_valid: Optional[bool] = None
    validation_error: Optional[str] = None
    
    # Simulation state
    is_simulated: bool = False
    simulation_success: Optional[bool] = None
    alpha_id: Optional[str] = None
    metrics: Dict = Field(default_factory=dict)
    simulation_error: Optional[str] = None
    
    # Correction state
    correction_attempts: int = 0
    original_expression: Optional[str] = None  # If corrected
    
    # Evaluation state
    quality_status: str = "PENDING"  # PASS, FAIL, PENDING
    
    # Additional metadata for tracking
    metadata: Dict = Field(default_factory=dict)


class AlphaResult(BaseModel):
    """Final result for a processed alpha."""
    expression: str
    hypothesis: Optional[str] = None
    explanation: Optional[str] = None
    alpha_id: Optional[str] = None
    metrics: Dict = Field(default_factory=dict)
    quality_status: str = "PENDING"  # PASS, OPTIMIZE, FAIL, PENDING
    trace_step_id: Optional[int] = None


class FailureRecord(BaseModel):
    """Record of a failed alpha attempt."""
    expression: str
    error_type: str
    error_message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    trace_step_id: Optional[int] = None


class TraceStepData(BaseModel):
    """Trace step data for state accumulation."""
    step_type: str
    step_order: int
    input_data: Dict = Field(default_factory=dict)
    output_data: Dict = Field(default_factory=dict)
    duration_ms: int = 0
    status: str = "SUCCESS"
    error_message: Optional[str] = None


# =============================================================================
# MAIN STATE
# =============================================================================

class MiningState(BaseModel):
    """
    Main state for the mining workflow graph.
    
    Designed for:
    - Strong typing with Pydantic
    - Immutable updates (return new state)
    - Full traceability
    """
    
    # -------------------------------------------------------------------------
    # Task Context (immutable after init)
    # -------------------------------------------------------------------------
    task_id: int
    region: str = "USA"
    universe: str = "TOP3000"
    dataset_id: str = ""

    # MCP runtime tools loaded from the web-managed registry for this graph run.
    mcp_execution_enabled: bool = False
    mcp_tool_names: List[str] = Field(default_factory=list)
    mcp_tools: List[Dict] = Field(default_factory=list)
    
    # Context data
    fields: List[Dict] = Field(default_factory=list)
    operators: List[Dict] = Field(default_factory=list)
    num_alphas_target: int = 3
    
    # -------------------------------------------------------------------------
    # RAG Results
    # -------------------------------------------------------------------------
    patterns: List[Dict] = Field(default_factory=list)
    pitfalls: List[Dict] = Field(default_factory=list)
    dataset_description: str = ""
    dataset_category: str = ""
    
    # Distillation Results
    distilled_concepts: List[str] = Field(default_factory=list)
    focused_fields: List[Dict] = Field(default_factory=list)

    hypotheses: List[Dict] = Field(default_factory=list)
    
    # -------------------------------------------------------------------------
    # Alpha Processing Queue
    # -------------------------------------------------------------------------
    pending_alphas: List[AlphaCandidate] = Field(default_factory=list)
    current_alpha: Optional[AlphaCandidate] = None
    current_alpha_index: int = 0
    
    # -------------------------------------------------------------------------
    # Self-Correction Loop Control
    # -------------------------------------------------------------------------
    retry_count: int = 0
    max_retries: int = 3
    
    # -------------------------------------------------------------------------
    # Outputs (accumulated)
    # -------------------------------------------------------------------------
    generated_alphas: List[AlphaResult] = Field(default_factory=list)
    failures: List[FailureRecord] = Field(default_factory=list)
    
    # -------------------------------------------------------------------------
    # Trace (accumulated)
    # -------------------------------------------------------------------------
    step_order: int = 0
    trace_steps: List[TraceStepData] = Field(default_factory=list)
    
    # -------------------------------------------------------------------------
    # Control Flags
    # -------------------------------------------------------------------------
    should_stop: bool = False
    error: Optional[str] = None
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def increment_step(self) -> int:
        """Get next step order (for external use, state is immutable)."""
        return self.step_order + 1
    
    def has_more_alphas(self) -> bool:
        """Check if there are more alphas to process."""
        return self.current_alpha_index < len(self.pending_alphas)
    
    def get_current_alpha(self) -> Optional[AlphaCandidate]:
        """Get current alpha from queue."""
        if self.current_alpha_index < len(self.pending_alphas):
            return self.pending_alphas[self.current_alpha_index]
        return None
    
    class Config:
        """Pydantic config."""
        validate_assignment = True


# =============================================================================
# STATE UPDATE HELPERS
# =============================================================================

def merge_state(state: MiningState, updates: Dict) -> Dict:
    """
    Create a partial state update dict.
    Used in node functions to return updates.
    """
    return updates


def add_trace_step(
    state: MiningState,
    step_type: str,
    input_data: Dict = None,
    output_data: Dict = None,
    duration_ms: int = 0,
    status: str = "SUCCESS",
    error_message: str = None
) -> Dict:
    """
    Create a trace step and return state update.
    """
    new_step = TraceStepData(
        step_type=step_type,
        step_order=state.step_order + 1,
        input_data=input_data or {},
        output_data=output_data or {},
        duration_ms=duration_ms,
        status=status,
        error_message=error_message
    )
    
    return {
        "step_order": state.step_order + 1,
        "trace_steps": state.trace_steps + [new_step]
    }
