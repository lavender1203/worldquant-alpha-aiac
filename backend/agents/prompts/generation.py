"""
Alpha generation prompts.

Redesigned based on RD-Agent's principles:
- Hypothesis-driven generation
- No preconceived biases about what works
- Emphasis on testability and precision
- Learning from experiment feedback

Contains:
- ALPHA_GENERATION_SYSTEM: System prompt for alpha generation
- build_alpha_generation_prompt: Builder function for user prompt
"""

from typing import List, Dict, Optional

from backend.agents.prompts.base import (
    PromptContext,
    build_fields_context,
    build_operators_context,
    build_patterns_context,
    build_strategy_constraints,
)


ALPHA_GENERATION_SYSTEM = """You are a quantitative researcher implementing alpha expressions to test investment hypotheses.

Your role is to translate hypotheses into mathematical expressions that can be backtested.

**Core Principles**:
1. **Precision**: Each expression should test exactly one hypothesis
2. **Simplicity First**: Start with simple implementations; complexity can be added if needed
3. **Objectivity**: Do not assume certain operators or patterns are inherently better
4. **Testability**: Every expression should produce measurable results

**Implementation Guidelines**:
- Use only the provided fields and operators
- Ensure syntactic correctness
- Document the reasoning clearly
- Consider multiple ways to implement the same hypothesis

**CRITICAL - Turnover Control**:
Turnover must be between 1% and 70% for alpha submission. High turnover is the most common failure mode.
To control turnover, you MUST:
1. Always wrap final signals with `ts_decay_linear(signal, N)` where N >= 5 (typically 5-15)
2. Use longer lookback windows (10+ days instead of 1-5 days)
3. Prefer `ts_rank` over raw values to reduce noise
4. Apply multiple smoothing layers if needed: ts_decay_linear(ts_rank(...), N)

Example pattern for turnover control:
- BAD: `ts_delta(field, 1)` (high turnover, noisy)
- GOOD: `ts_decay_linear(ts_rank(ts_delta(field, 5), 10), 10)` (smoothed, stable)

Output must be valid JSON matching the specified schema."""


def build_alpha_generation_prompt(
    ctx: PromptContext,
    target_hypothesis: Optional[Dict] = None,
    experiment_feedback: Optional[List[Dict]] = None
) -> str:
    """
    Build user prompt for alpha generation.
    
    Redesigned to be hypothesis-driven and feedback-aware.
    
    Args:
        ctx: Prompt context with fields, operators, etc.
        target_hypothesis: Optional specific hypothesis to implement
        experiment_feedback: Optional list of previous experiment results
    """
    
    # Build hypothesis section
    hypothesis_section = ""
    if target_hypothesis:
        hypothesis_section = f"""
## Target Hypothesis

You are implementing this specific hypothesis:

**Statement**: {target_hypothesis.get('statement', 'Not specified')}
**Rationale**: {target_hypothesis.get('rationale', 'Not specified')}
**Expected Signal**: {target_hypothesis.get('expected_signal', 'Not specified')}
**Suggested Fields**: {', '.join(target_hypothesis.get('key_fields', []))}
"""
    
    # Build feedback section
    feedback_section = ""
    if experiment_feedback:
        recent_feedback = experiment_feedback[-5:]  # Last 5 experiments
        feedback_entries = []
        
        for fb in recent_feedback:
            expr = fb.get('expression', 'N/A')
            if len(expr) > 80:
                expr = expr[:80] + "..."
            
            feedback_entries.append(f"""
- **Expression**: `{expr}`
  - Result: {fb.get('result', 'N/A')}
  - Sharpe: {fb.get('sharpe', 'N/A')}, Fitness: {fb.get('fitness', 'N/A')}
  - Issue: {fb.get('issue', 'None identified')}
""")
        
        feedback_section = f"""
## Recent Experiment Feedback

Learn from these recent attempts:
{''.join(feedback_entries)}

Consider:
- What worked partially that could be refined?
- What approaches haven't been tried yet?
- Are there common failure patterns to avoid?
"""
    
    # Build implementation guidance (non-prescriptive)
    implementation_guidance = """
## Implementation Approach

Consider multiple ways to implement the hypothesis:
1. **Direct implementation**: Straightforward translation of the hypothesis
2. **Normalized version**: Apply cross-sectional normalization (rank, zscore)
3. **Smoothed version**: Add time-series smoothing if appropriate
4. **Inverted version**: Test if the opposite relationship holds

Start with simpler implementations. Complexity can be added in subsequent iterations if needed.

## CRITICAL: Turnover Control (Required)

**High turnover is the #1 failure mode.** Every expression MUST include turnover control:

1. **Always apply decay**: Wrap your signal with `ts_decay_linear(signal, N)` where N >= 5
2. **Use longer windows**: Prefer 10-20 day windows over 1-5 day windows
3. **Smooth with rank**: Use `ts_rank(signal, N)` before decay for stability

**Required Pattern**:
```
ts_decay_linear(ts_rank(YOUR_CORE_SIGNAL, lookback), decay_days)
```

Example transformations:
- Raw signal: `ts_delta(field, 5)` → With turnover control: `ts_decay_linear(ts_rank(ts_delta(field, 5), 10), 10)`
- Correlation: `ts_corr(a, b, 10)` → Smoothed: `ts_decay_linear(ts_rank(ts_corr(a, b, 10), 15), 8)`
"""
    
    # Build field reminder (critical constraint, but framed as a resource)
    field_section = f"""
## Available Resources

**Data Fields** (use only these):
{build_fields_context(ctx.fields)}

**Operators** (grouped by function):
{build_operators_context(ctx.operators)}

Note: Only fields listed above exist. Do not assume standard fields like 'close', 'volume', 
'returns', or 'cap' exist unless explicitly listed.
"""
    
    # Build patterns section (framed as observations, not rules)
    patterns_section = ""
    if ctx.success_patterns or ctx.failure_pitfalls:
        patterns_section = f"""
## Historical Observations

**Patterns that have worked** (for reference, not prescription):
{build_patterns_context(ctx.success_patterns, "success patterns")}

**Approaches that have struggled** (considerations, not prohibitions):
{build_patterns_context(ctx.failure_pitfalls, "challenges")}

These are historical observations. Context matters - what failed in one setting may work in another.
"""
    
    return f"""## Context

**Dataset**: {ctx.dataset_id}
**Description**: {ctx.dataset_description or 'Not provided'}
**Category**: {ctx.dataset_category or 'General'}
**Region**: {ctx.region} | **Universe**: {ctx.universe}
{hypothesis_section}
{field_section}
{patterns_section}
{feedback_section}
{implementation_guidance}

## Constraints

{build_strategy_constraints(ctx)}

## Task

Generate {ctx.num_alphas} distinct alpha expressions.

**CRITICAL - Hypothesis-Implementation Alignment**:
Each alpha MUST:
1. Clearly reference which hypothesis it tests (from the hypotheses list above if provided)
2. Use the fields suggested by that hypothesis (key_fields)
3. Implement the signal type specified (momentum, mean_reversion, etc.)
4. Have a clear logical connection between the hypothesis rationale and the expression logic

If a hypothesis says "stocks with increasing sentiment outperform", the expression MUST:
- Use a sentiment-related field
- Capture the "increasing" aspect (e.g., ts_delta, ts_returns)
- Have a positive relationship (or explicitly negative for short signals)

Misaligned implementations (e.g., using random fields not related to the hypothesis) will be REJECTED.

For each expression:
1. State the specific hypothesis being tested (MUST match one from the hypothesis list)
2. Explain how the implementation DIRECTLY tests that hypothesis
3. Describe what market behavior this captures and why it connects to the hypothesis
4. Note any assumptions or limitations

**Output Schema** (JSON):
```json
{{
  "implementation_notes": "Brief notes on the overall approach taken",
  "alphas": [
    {{
      "hypothesis_tested": "The specific hypothesis this expression tests",
      "expression": "Valid expression using only provided fields and operators",
      "explanation": {{
        "approach": "How the hypothesis is translated into code",
        "market_logic": "What market inefficiency or behavior this captures",
        "assumptions": "Key assumptions this relies on"
      }},
      "fields_used": ["field1", "field2"],
      "complexity": "simple | moderate | complex",
      "novelty_level": "established | variation | experimental"
    }}
  ],
  "alternatives_considered": [
    {{
      "expression": "Alternative implementation not used",
      "reason_not_chosen": "Why this wasn't the primary choice"
    }}
  ]
}}
```"""
