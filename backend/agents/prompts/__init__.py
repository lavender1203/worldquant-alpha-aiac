"""
Prompt Templates Package for Alpha Mining

Redesigned based on RD-Agent's principles:
- Hypothesis-driven experiment cycle
- No preconceived biases
- Knowledge transfer from experiments
- Structured feedback loop

This package organizes prompt templates by responsibility:
- base: Data classes and helper functions
- generation: Alpha generation prompts
- hypothesis: Hypothesis generation prompts  
- validation: Self-correction and validation prompts
- analysis: Round analysis, failure analysis, and feedback prompts
- legacy: Backward-compatible legacy templates
- registry: Dynamic prompt selection

For backward compatibility, all exports are re-exported here.
"""

# Base components
from backend.agents.prompts.base import (
    PromptContext,
    build_fields_context,
    build_operators_context,
    build_patterns_context,
    build_factor_construction_context,
    build_strategy_constraints,
)

# Generation prompts
from backend.agents.prompts.generation import (
    ALPHA_GENERATION_SYSTEM,
    build_alpha_generation_prompt,
)

# Hypothesis prompts
from backend.agents.prompts.hypothesis import (
    HYPOTHESIS_SYSTEM,
    DISTILL_SYSTEM,
    build_hypothesis_prompt,
    build_distill_prompt,
)

# Validation prompts
from backend.agents.prompts.validation import (
    SELF_CORRECT_SYSTEM,
    OPTIMIZATION_SYSTEM,
    build_self_correct_prompt,
    build_optimization_prompt,
)

# Analysis prompts
from backend.agents.prompts.analysis import (
    ROUND_ANALYSIS_SYSTEM,
    FAILURE_ANALYSIS_SYSTEM,
    FEEDBACK_GENERATION_SYSTEM,
    build_round_analysis_prompt,
    build_feedback_prompt,
    build_enhanced_feedback_prompt,
    FAILURE_ANALYSIS_USER,
)

# Alignment prompts (hypothesis-implementation gap handling)
from backend.agents.prompts.alignment import (
    AlignmentResult,
    ExperimentAttribution,
    ALIGNMENT_CHECK_SYSTEM,
    ATTRIBUTION_SYSTEM,
    build_alignment_check_prompt,
    build_attribution_prompt,
    quick_alignment_check,
    determine_attribution_heuristic,
    filter_knowledge_by_attribution,
)

# Legacy templates (backward compatibility)
from backend.agents.prompts.legacy import (
    DISTILL_USER,
    HYPOTHESIS_USER,
    ALPHA_GENERATION_USER,
    SELF_CORRECT_USER,
    ROUND_ANALYSIS_USER,
)

# Registry
from backend.agents.prompts.registry import PromptRegistry

# Loader (YAML-based prompts)
from backend.agents.prompts.loader import (
    PromptLoader,
    get_prompt_loader,
    get_prompt,
    get_rendered_prompt,
)

__all__ = [
    # Base
    "PromptContext",
    "build_fields_context",
    "build_operators_context",
    "build_patterns_context",
    "build_factor_construction_context",
    "build_strategy_constraints",
    # Generation
    "ALPHA_GENERATION_SYSTEM",
    "build_alpha_generation_prompt",
    # Hypothesis
    "HYPOTHESIS_SYSTEM",
    "DISTILL_SYSTEM",
    "build_hypothesis_prompt",
    "build_distill_prompt",
    # Validation
    "SELF_CORRECT_SYSTEM",
    "OPTIMIZATION_SYSTEM",
    "build_self_correct_prompt",
    "build_optimization_prompt",
    # Analysis
    "ROUND_ANALYSIS_SYSTEM",
    "FAILURE_ANALYSIS_SYSTEM",
    "FEEDBACK_GENERATION_SYSTEM",
    "build_round_analysis_prompt",
    "build_feedback_prompt",
    "build_enhanced_feedback_prompt",
    "FAILURE_ANALYSIS_USER",
    # Alignment (hypothesis-implementation gap)
    "AlignmentResult",
    "ExperimentAttribution",
    "ALIGNMENT_CHECK_SYSTEM",
    "ATTRIBUTION_SYSTEM",
    "build_alignment_check_prompt",
    "build_attribution_prompt",
    "quick_alignment_check",
    "determine_attribution_heuristic",
    "filter_knowledge_by_attribution",
    # Legacy
    "DISTILL_USER",
    "HYPOTHESIS_USER",
    "ALPHA_GENERATION_USER",
    "SELF_CORRECT_USER",
    "ROUND_ANALYSIS_USER",
    # Registry
    "PromptRegistry",
    # Loader
    "PromptLoader",
    "get_prompt_loader",
    "get_prompt",
    "get_rendered_prompt",
]
