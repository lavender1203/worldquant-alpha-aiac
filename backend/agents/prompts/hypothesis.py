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


def _build_field_details(fields: List[Dict], max_fields: int = 15) -> str:
    """
    Build detailed field information with descriptions for hypothesis generation.
    
    This is crucial - hypotheses should be based on understanding what fields actually represent.
    """
    if not fields:
        return "No fields available."
    
    # Prioritize fields with descriptions
    fields_with_desc = [f for f in fields if f.get("description")]
    fields_without_desc = [f for f in fields if not f.get("description")]
    
    # Take prioritized fields
    selected = fields_with_desc[:max_fields]
    if len(selected) < max_fields:
        selected.extend(fields_without_desc[:max_fields - len(selected)])
    
    lines = []
    for f in selected[:max_fields]:
        field_id = f.get("id", f.get("name", f.get("field_id", "unknown")))
        field_type = f.get("type", f.get("field_type", "MATRIX")).upper()
        description = f.get("description", "No description available")
        
        # Truncate long descriptions
        if len(description) > 150:
            description = description[:150] + "..."
        
        lines.append(f"- **{field_id}** ({field_type}): {description}")
    
    if len(fields) > max_fields:
        lines.append(f"\n*... and {len(fields) - max_fields} more fields available*")
    
    return "\n".join(lines)


def _build_question_driven_analysis(fields: List[Dict]) -> str:
    """
    Generate question-driven feature ideas based on the give_me_idea methodology.
    
    Asks 8 fundamental questions about the data to generate feature ideas:
    1. What is stable? - Invariants
    2. What is changing? - Change patterns
    3. What is anomalous? - Deviations
    4. What is combined? - Interactions
    5. What is structural? - Compositions
    6. What is cumulative? - Accumulation effects
    7. What is relative? - Comparisons
    8. What is essential? - Core meaning
    """
    if not fields:
        return "No fields available for analysis."
    
    # Extract field names for examples
    field_names = [f.get("id", f.get("name", "field")) for f in fields[:10]]
    f1 = field_names[0] if field_names else "field1"
    f2 = field_names[1] if len(field_names) > 1 else "field2"
    f3 = field_names[2] if len(field_names) > 2 else "field3"
    
    questions = f"""
### Question 1: "What is STABLE?"
*Look for invariants - fields or combinations that remain relatively constant.*
- Which fields show low volatility over time?
- What stability measures make sense?
- **Feature Concept**: `ts_std_dev({f1}, 60)` → Lower stability = higher uncertainty
- **Application**: Stable fundamental metrics may indicate quality; instability may indicate risk

### Question 2: "What is CHANGING?"
*Analyze change patterns - rate, acceleration, volatility.*
- How fast are values changing? (Rate)
- Is the change accelerating? (Second derivative)
- **Feature Concept**: `ts_delta(ts_delta({f1}, 10), 10)` → Acceleration of change
- **Application**: Momentum in changes often precedes price moves

### Question 3: "What is ANOMALOUS?"
*Identify deviations - outliers, unusual patterns, breaks from normal.*
- What constitutes "normal" for each field?
- How do we measure deviation significance?
- **Feature Concept**: `abs({f1} - ts_mean({f1}, 60)) / ts_std_dev({f1}, 60)` → Z-score deviation
- **Application**: Extreme deviations may mean-revert or signal regime change

### Question 4: "What is COMBINED?"
*Examine interactions - how fields amplify or offset each other.*
- Which fields interact meaningfully?
- Do combinations create new meaning?
- **Feature Concept**: `{f1} * sign({f2})` → One field weighted by direction of another
- **Application**: Interaction effects often capture non-linear relationships

### Question 5: "What is STRUCTURAL?"
*Study compositions - constituent parts, proportional relationships.*
- What are the parts vs. wholes?
- How do proportions change over time?
- **Feature Concept**: `{f1} / ({f1} + {f2})` → Proportional share
- **Application**: Structural shifts in composition indicate business model changes

### Question 6: "What is CUMULATIVE?"
*Explore accumulation effects - building up, decay, memory.*
- What builds up over time?
- What has persistence vs. mean-reversion?
- **Feature Concept**: `ts_sum({f1}, 60) / ts_sum({f1}, 252)` → Short vs long term accumulation
- **Application**: Cumulative patterns reveal persistent trends vs. noise

### Question 7: "What is RELATIVE?"
*Make comparisons - ranking, normalization, context.*
- How does each stock compare to peers?
- What's the relevant comparison universe?
- **Feature Concept**: `group_rank({f1}, sector) - rank({f1})` → Sector vs universe ranking gap
- **Application**: Relative positioning removes market-wide effects

### Question 8: "What is ESSENTIAL?"
*Distill to core meaning - first principles, essence.*
- What does this field REALLY measure?
- What's the fundamental economic concept?
- **Feature Concept**: Strip to core → `sign({f1}) * log1p(abs({f1}))` → Preserve direction, compress magnitude
- **Application**: Reducing to essence removes noise and captures fundamental signal
"""
    return questions


def _analyze_field_combinations(fields: List[Dict]) -> str:
    """
    Analyze potential field combinations and suggest meaningful pairs/groups.
    
    This is key to generating good hypotheses - most valuable alphas come from
    field COMBINATIONS, not individual fields.
    """
    if not fields or len(fields) < 2:
        return "Not enough fields for combination analysis."
    
    # Categorize fields by semantic type based on name/description patterns
    estimates = []      # Analyst estimates, forecasts
    actuals = []        # Actual reported values
    changes = []        # Delta, change, revision fields
    counts = []         # Count, number fields
    ratios = []         # Ratio, percent, rate fields
    flags = []          # Flag, indicator fields
    prices = []         # Price-related fields
    other = []          # Everything else
    
    for f in fields:
        field_id = f.get("id", f.get("name", "")).lower()
        desc = (f.get("description") or "").lower()
        combined = field_id + " " + desc
        
        if any(x in combined for x in ["estimate", "forecast", "expect", "predict", "target"]):
            estimates.append(f)
        elif any(x in combined for x in ["actual", "reported", "realized"]):
            actuals.append(f)
        elif any(x in combined for x in ["change", "delta", "revision", "diff", "growth"]):
            changes.append(f)
        elif any(x in combined for x in ["count", "number", "num_", "_cnt", "coverage"]):
            counts.append(f)
        elif any(x in combined for x in ["ratio", "percent", "pct", "rate", "margin", "yield"]):
            ratios.append(f)
        elif any(x in combined for x in ["flag", "indicator", "is_", "has_"]):
            flags.append(f)
        elif any(x in combined for x in ["price", "value", "amount", "cap"]):
            prices.append(f)
        else:
            other.append(f)
    
    combinations = []
    
    # 1. Estimate vs Actual (surprise/beat)
    if estimates and actuals:
        est_name = estimates[0].get("id", estimates[0].get("name", "estimate"))
        act_name = actuals[0].get("id", actuals[0].get("name", "actual"))
        combinations.append(f"""
**Surprise/Beat Pattern** (Estimate vs Actual):
- Fields: `{est_name}` + `{act_name}`
- Idea: `({act_name} - {est_name}) / abs({est_name})` → Earnings surprise signal
- Logic: Positive surprise indicates underestimation, often followed by price drift""")
    
    # 2. Estimate Revisions (change in estimates)
    if estimates and len(estimates) >= 1:
        est_name = estimates[0].get("id", estimates[0].get("name", "estimate"))
        combinations.append(f"""
**Revision Momentum Pattern**:
- Field: `{est_name}`
- Idea: `ts_delta({est_name}, 20)` or `ts_rank(ts_delta({est_name}, 10), 30)`
- Logic: Upward revisions tend to cluster; momentum in analyst expectations""")
    
    # 3. Ratio analysis
    if ratios and len(ratios) >= 2:
        r1 = ratios[0].get("id", ratios[0].get("name", "ratio1"))
        r2 = ratios[1].get("id", ratios[1].get("name", "ratio2"))
        combinations.append(f"""
**Ratio Comparison Pattern**:
- Fields: `{r1}` + `{r2}`
- Idea: `ts_rank({r1}, 20) - ts_rank({r2}, 20)` → Relative ranking divergence
- Logic: Divergence between related ratios may signal mispricing""")
    
    # 4. Count as weight/filter
    if counts and (estimates or ratios or other):
        cnt_name = counts[0].get("id", counts[0].get("name", "count"))
        signal_source = estimates or ratios or other
        sig_name = signal_source[0].get("id", signal_source[0].get("name", "signal"))
        combinations.append(f"""
**Coverage-Weighted Pattern**:
- Fields: `{cnt_name}` + `{sig_name}`
- Idea: `if_else({cnt_name} > threshold, ts_rank({sig_name}, 20), 0)` → Filter by coverage
- Logic: Higher coverage = more reliable signal, filter out low-coverage stocks""")
    
    # 5. Cross-sectional comparison
    if len(fields) >= 2:
        f1 = fields[0].get("id", fields[0].get("name", "field1"))
        f2 = fields[1].get("id", fields[1].get("name", "field2"))
        combinations.append(f"""
**Cross-Sectional Correlation Pattern**:
- Fields: `{f1}` + `{f2}`
- Idea: `ts_corr({f1}, {f2}, 60)` → Rolling correlation between fields
- Logic: Changes in correlation structure can signal regime shifts""")
    
    # 6. Group-relative pattern
    if ratios or estimates or other:
        source = ratios or estimates or other
        src_name = source[0].get("id", source[0].get("name", "field"))
        combinations.append(f"""
**Sector-Relative Pattern** (using GROUP operators):
- Field: `{src_name}`
- Idea: `group_zscore({src_name}, subindustry)` or `group_rank(ts_delta({src_name}, 10), sector)`
- Logic: Compare within peer group to remove sector bias; identifies relative winners""")
    
    if not combinations:
        return "No obvious field combination patterns detected. Consider exploring field relationships manually."
    
    return "\n".join(combinations)


def build_hypothesis_prompt(
    ctx: PromptContext,
    experiment_trace: Optional[List[Dict]] = None
) -> str:
    """
    Build prompt for hypothesis generation.
    
    Redesigned based on RD-Agent's hypothesis-driven approach:
    - Includes DETAILED FIELD DESCRIPTIONS for informed hypothesis generation
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
    
    # Build DETAILED field information with descriptions
    # This is crucial for hypothesis generation - LLM needs to understand what fields represent
    field_details = _build_field_details(ctx.fields, max_fields=15)
    
    # Build field combination analysis - most valuable alphas come from field COMBINATIONS
    field_combinations = _analyze_field_combinations(ctx.fields)
    
    # Build question-driven analysis (from give_me_idea methodology)
    question_analysis = _build_question_driven_analysis(ctx.fields)
    
    # Also build simple overview for reference
    field_overview = build_fields_context(ctx.fields, max_fields=20)
    
    return f"""## Research Context

**Dataset**: {ctx.dataset_id}
**Category**: {ctx.dataset_category or 'General'}
**Description**: {ctx.dataset_description or 'Not provided'}
**Region**: {ctx.region} | **Universe**: {ctx.universe}
{dataset_guidance}

## CRITICAL: Available Data Fields with Descriptions

**READ THESE FIELD DESCRIPTIONS CAREFULLY** - Your hypotheses MUST be based on what these fields actually represent:

{field_details}

## Field Type Summary

{field_overview}

## [KEY] FIELD COMBINATION IDEAS (Most Valuable!)

**The best alphas often come from COMBINING multiple fields.** Here are potential combination patterns based on the available fields:

{field_combinations}

## QUESTION-DRIVEN FEATURE GENERATION (Deep Analysis Framework)

Ask these 8 fundamental questions about your data to generate novel ideas:

{question_analysis}

**Key Combination Strategies**:
1. **Surprise/Beat**: Compare estimates vs actuals → `(actual - estimate) / abs(estimate)`
2. **Revision Momentum**: Track changes in estimates → `ts_delta(estimate_field, N)`
3. **Cross-field Ratio**: Create ratios between related fields → `field_a / field_b`
4. **Conditional Signal**: Use one field to filter/weight another → `if_else(filter_field > X, signal_field, 0)`
5. **Correlation Structure**: Detect relationship changes → `ts_corr(field_a, field_b, 60)`
6. **Group Relative**: Compare within sector → `group_rank(field, subindustry)`

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

**CRITICAL**: Your hypotheses should leverage FIELD COMBINATIONS, not just individual fields!

**IMPORTANT**: Each hypothesis MUST:
1. Combine 2+ fields in a meaningful way (ratio, difference, correlation, conditional)
2. Reference SPECIFIC fields from the list above by name
3. Be grounded in what those fields actually measure (per their descriptions)
4. Explain the economic logic of WHY this combination should predict returns
5. Be testable with a concrete expression using the available fields

**Requirements**:
1. Each hypothesis should be specific and testable
2. At least 2 hypotheses should combine multiple fields
3. Include both conventional and unconventional combination ideas
4. Explain the reasoning behind each hypothesis
5. Consider what market behavior or inefficiency the field combination might capture
6. **USE THE FIELD COMBINATION IDEAS** above as inspiration

**Output Schema** (JSON):
```json
{{
  "analysis": {{
    "data_observations": "Key observations about the dataset characteristics",
    "field_relationships": "Notable relationships/combinations between fields",
    "unexplored_directions": "Promising directions not yet tested",
    "refinement_opportunities": "Ways to improve on partial successes"
  }},
  "hypotheses": [
    {{
      "id": "H1",
      "statement": "Clear, testable hypothesis in one sentence",
      "rationale": "Economic or behavioral reasoning behind this hypothesis",
      "expected_signal": "momentum | mean_reversion | value | relative | surprise",
      "key_fields": ["field1", "field2"],
      "combination_type": "ratio | difference | correlation | conditional | group_relative | single",
      "suggested_expression": "ts_decay_linear(ts_rank(field_a / field_b, 20), 10)",
      "why_this_combination": "Explanation of why combining these fields makes economic sense",
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
