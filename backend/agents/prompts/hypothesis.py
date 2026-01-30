"""
Hypothesis and distillation prompts.

Redesigned based on RD-Agent's hypothesis-driven approach:
- Precise, testable, actionable hypotheses
- Balanced exploration and exploitation
- Knowledge transfer from experiments
- No preconceived biases about what works

Contains:
- HYPOTHESIS_SYSTEM: System prompt for hypothesis generation
- DISTILL_SYSTEM: System prompt for concept distillation
- build_hypothesis_prompt: Builder for hypothesis prompt
- build_distill_prompt: Builder for distillation prompt
"""

from typing import Dict, List, Optional

from backend.agents.prompts.base import (
    PromptContext,
    build_fields_context,
    build_patterns_context,
)


HYPOTHESIS_SYSTEM = """You are a quantitative research scientist conducting data-driven research for WorldQuant BRAIN alpha mining.

Your role is to generate investment hypotheses for testing. The approach is empirical:
1. Observe the data characteristics and historical experiment results
2. Form a precise, testable hypothesis about potential market relationships
3. Design an experiment to validate or refute the hypothesis

**Core Principles**:
- Be objective: Do not assume any particular approach is better a priori
- Be precise: Each hypothesis should focus on a single testable idea
- Be exploratory: Consider unconventional relationships the data might reveal
- Learn from feedback: Analyze why previous experiments succeeded or failed

**Hypothesis Quality Standards**:
1. Testable: Can be validated with a concrete experiment
2. Specific: Avoids vague statements like "improve performance"
3. Actionable: Clear enough to implement directly
4. Focused: One direction per hypothesis, not "A or B might work"

**CRITICAL Implementation Constraints**:
When suggesting approaches, always consider turnover control:
- HIGH TURNOVER IS THE #1 FAILURE MODE - hypotheses requiring rapid signal changes will fail
- Prefer hypotheses that can be tested with stable, slow-moving signals
- Suggest longer lookback windows (10-20 days) over short ones (1-5 days)
- Always mention if signal smoothing (ts_decay_linear) should be applied

**Economic Intuition by Dataset Category**:
When generating hypotheses, consider these domain-specific insights:

ANALYST DATA (analyst*, ana*):
- Analyst estimate revisions → Information asymmetry (upward revisions precede positive returns)
- Consensus vs. dispersion → Uncertainty premium (high dispersion = higher volatility)
- Estimate changes momentum → Analysts herd behavior creates predictable patterns
- Distance from consensus → Contrarian opportunities
- Target price ratios → Value signal with analyst validation
- Coverage changes → Attention signals (new coverage = institutional interest)

FUNDAMENTAL DATA (fnd*, fundamental*):
- Quality metrics → Quality factor (ROE, margins, asset turnover)
- Accruals → Earnings quality signal (low accruals = sustainable earnings)
- Leverage changes → Financial distress risk or growth indication
- Working capital efficiency → Operational excellence signal
- Payout ratios → Capital discipline signal

SENTIMENT/NEWS DATA (oth*, news*, sentiment*):
- Sentiment momentum → Trend continuation
- Sentiment extremes → Mean reversion opportunities
- Volume of coverage → Attention/liquidity signal
- Sentiment divergence → Information asymmetry

PRICE-VOLUME DATA (pv*, price*, volume*):
- Price momentum → Trend-following signals
- Relative volume → Institutional activity
- Volatility patterns → Risk-adjusted opportunities
- Liquidity measures → Transaction cost factors

Output must be valid JSON."""


DISTILL_SYSTEM = """You are a research assistant helping to identify promising research directions.

Your role is to analyze dataset characteristics and suggest field categories
that may contain useful signals. Be objective in your analysis:
- Do not assume certain field types are inherently better
- Consider the specific context and data characteristics
- Balance well-known approaches with unexplored categories

Selection should be based on evidence, not assumptions."""


def _get_dataset_specific_guidance(dataset_id: str, category: str) -> str:
    """Generate dataset-specific hypothesis guidance based on category."""
    
    category_lower = (category or "").lower()
    dataset_lower = (dataset_id or "").lower()
    
    guidance_map = {
        "analyst": """
**Dataset-Specific Guidance (Analyst Data)**:
This dataset contains analyst estimates and forecasts. Consider these high-value signals:
1. **Estimate Revisions**: Changes in EPS/revenue estimates signal information flow. Upward revisions often precede positive returns.
2. **Consensus Changes**: When consensus estimates shift, stocks tend to move in that direction.
3. **Analyst Dispersion**: High disagreement among analysts indicates uncertainty premium.
4. **Coverage Breadth**: Changes in analyst coverage indicate institutional attention.
5. **Target Price Ratios**: Current price vs. target price indicates upside potential.

Recommended Approach: Focus on CHANGES and MOMENTUM in analyst estimates rather than levels.
""",
        "fundamental": """
**Dataset-Specific Guidance (Fundamental Data)**:
This dataset contains financial statement data. Consider these signals:
1. **Quality Metrics**: ROE, asset turnover, and margins indicate business quality.
2. **Accruals**: Low accruals indicate sustainable earnings (accrual anomaly).
3. **Growth Consistency**: Stable growth is more valuable than volatile high growth.
4. **Capital Efficiency**: Working capital and asset efficiency signal management quality.
5. **Financial Health**: Leverage and interest coverage indicate risk.

Recommended Approach: Focus on RELATIVE rankings within sectors using group_neutralize.
""",
        "sentiment": """
**Dataset-Specific Guidance (Sentiment/News Data)**:
This dataset contains sentiment or news-derived signals. Consider:
1. **Sentiment Momentum**: Improving sentiment tends to persist in the short term.
2. **Extreme Values**: Very high/low sentiment often mean-reverts.
3. **Divergence**: Sentiment vs. price divergence can signal turning points.
4. **Volume Effects**: High news volume with positive sentiment is bullish.

Recommended Approach: Use smoothing (ts_decay_linear) to reduce noise in sentiment signals.
""",
        "price": """
**Dataset-Specific Guidance (Price-Volume Data)**:
This dataset contains market data. Consider:
1. **Momentum**: Past returns predict future returns (trend following).
2. **Mean Reversion**: Extreme short-term moves tend to reverse.
3. **Liquidity**: High relative volume indicates institutional activity.
4. **Volatility**: Normalize signals by volatility for risk-adjusted measures.

Recommended Approach: Always use ts_rank or zscore for cross-sectional standardization.
"""
    }
    
    # Match category or dataset name to guidance
    for key, guidance in guidance_map.items():
        if key in category_lower or key in dataset_lower:
            return guidance
    
    # Default guidance for unknown categories
    return """
**Dataset-Specific Guidance**:
Explore this dataset methodically:
1. Start with simple field ratios and rankings
2. Apply time-series operators (ts_delta, ts_rank) to capture momentum
3. Use cross-sectional operators (rank, zscore) for comparability
4. Always include turnover control (ts_decay_linear wrapper)
"""


def build_hypothesis_prompt(
    ctx: PromptContext,
    experiment_trace: Optional[List[Dict]] = None
) -> str:
    """
    Build prompt for hypothesis generation.
    
    Redesigned based on RD-Agent's hypothesis-driven approach:
    - Includes experiment history with feedback
    - Emphasizes learning from failures
    - Encourages both exploration and exploitation
    - Provides dataset-specific economic guidance
    """
    
    # Get dataset-specific guidance
    dataset_guidance = _get_dataset_specific_guidance(ctx.dataset_id, ctx.dataset_category)
    
    # Build experiment trace section if available
    trace_section = ""
    if experiment_trace:
        trace_entries = []
        for i, entry in enumerate(experiment_trace[-10:], 1):  # Last 10 experiments
            exp = entry.get('experiment', {})
            feedback = entry.get('feedback', {})
            
            trace_entries.append(f"""
### Experiment {i}
**Hypothesis**: {exp.get('hypothesis', 'Not recorded')}
**Expression Tested**: `{exp.get('expression', 'N/A')[:100]}`
**Results**:
- Sharpe: {exp.get('sharpe', 'N/A')}, Fitness: {exp.get('fitness', 'N/A')}, Turnover: {exp.get('turnover', 'N/A')}
**Observation**: {feedback.get('observation', 'No observation recorded')}
**Evaluation**: {feedback.get('evaluation', 'Not evaluated')}
**Outcome**: {'SUCCESS' if feedback.get('success') else 'FAILED'} - {feedback.get('reason', '')}
""")
        
        trace_section = f"""
## Experiment History

The following experiments have been conducted. Analyze them to understand:
- What worked and why
- What failed and why
- What directions remain unexplored

{''.join(trace_entries)}
"""
    
    # Build strategy guidance (non-prescriptive)
    strategy_section = """
## Research Strategy

Consider both:
1. **Exploitation**: Refine approaches that showed promise in previous experiments
2. **Exploration**: Test new directions that haven't been tried yet

The balance depends on current progress:
- If recent experiments are failing: Consider new directions
- If recent experiments show partial success: Consider refinements
- If no clear pattern: Prioritize diverse exploration
"""
    
    # Build field categories overview
    field_overview = build_fields_context(ctx.fields, max_fields=20)
    
    return f"""## Research Context

**Dataset**: {ctx.dataset_id}
**Category**: {ctx.dataset_category or 'General'}
**Description**: {ctx.dataset_description or 'Not provided'}
**Region**: {ctx.region} | **Universe**: {ctx.universe}
{dataset_guidance}

## Available Data Fields (Sample)

{field_overview}

## Historical Patterns (For Reference Only)

**Approaches that have worked in similar contexts**:
{build_patterns_context(ctx.success_patterns, "patterns")}

**Approaches that have not worked**:
{build_patterns_context(ctx.failure_pitfalls, "pitfalls")}

Note: These are observations, not rules. What failed before may work in different contexts.
{trace_section}
{strategy_section}

## Task

Generate 3-5 investment hypotheses for this dataset.

**Requirements**:
1. Each hypothesis should be specific and testable
2. Include both conventional and unconventional ideas
3. Explain the reasoning behind each hypothesis
4. Consider what market behavior or inefficiency the data might capture

**Output Schema** (JSON):
```json
{{
  "analysis": {{
    "data_observations": "Key observations about the dataset characteristics",
    "unexplored_directions": "Promising directions not yet tested",
    "refinement_opportunities": "Ways to improve on partial successes"
  }},
  "hypotheses": [
    {{
      "id": "H1",
      "statement": "Clear, testable hypothesis in one sentence",
      "rationale": "Economic or behavioral reasoning behind this hypothesis",
      "expected_signal": "momentum | mean_reversion | value | other",
      "key_fields": ["field1", "field2"],
      "suggested_approach": "Brief description of how to test this",
      "confidence": "high | medium | low",
      "novelty": "established | emerging | experimental"
    }}
  ],
  "knowledge_transfer": {{
    "if_then_rules": [
      "If [condition observed in experiments], then [conclusion]"
    ],
    "patterns_discovered": "Any new patterns discovered from experiment analysis"
  }}
}}
```"""


def build_distill_prompt(ctx: PromptContext, field_categories: Dict[str, List[str]]) -> str:
    """
    Build prompt for concept distillation.
    
    Redesigned to be more objective and less prescriptive.
    """
    
    categories_text = []
    for cat, fields in sorted(field_categories.items()):
        sample = ", ".join(fields[:5])
        if len(fields) > 5:
            sample += f" ... (+{len(fields) - 5} more)"
        categories_text.append(f"- **{cat}** ({len(fields)} fields): {sample}")
    
    return f"""## Analysis Task

**Dataset**: {ctx.dataset_id}
**Description**: {ctx.dataset_description or 'Not provided'}
**Category**: {ctx.dataset_category or 'General'}

## Available Field Categories

{chr(10).join(categories_text)}

## Historical Context (For Reference)

Previous successful patterns have used these types of data:
{build_patterns_context(ctx.success_patterns, "patterns")}

Note: This is historical observation, not a prescription. New opportunities may exist elsewhere.

## Task

Identify 3-5 field categories that warrant investigation.

**Selection Approach**:
- Consider both high-probability and high-potential categories
- Include at least one less-explored category
- Balance between exploitation (known useful) and exploration (potentially useful)

**Output Schema** (JSON):
```json
{{
  "analysis": {{
    "dataset_characteristics": "Key features of this dataset",
    "category_assessment": "Brief assessment of each category's potential"
  }},
  "selected_categories": [
    {{
      "category": "Exact category name",
      "rationale": "Why this category may contain useful signals",
      "exploration_type": "exploitation | exploration | balanced"
    }}
  ],
  "reasoning": "Overall selection strategy explanation"
}}
```

**Important**: Use exact category names from the list above."""
