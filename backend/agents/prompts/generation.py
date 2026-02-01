"""
Alpha generation prompts.

Redesigned based on RD-Agent's principles:
- Hypothesis-driven generation
- No preconceived biases about what works
- Emphasis on testability and precision
- Learning from experiment feedback

P0 Fix: Added diversity constraints to prevent template traps
(e.g., over-reliance on ts_decay_linear(ts_rank(...)))

P0 Enhancement: Strong Hypothesis-Expression alignment enforcement
- MANDATORY field binding from hypothesis key_fields
- Signal direction consistency check
- Rejection of misaligned implementations

P1 Enhancement: CoSTEER-style feedback injection
- Hard constraints from failures (DO NOT patterns)
- Soft preferences from successes (PREFER patterns)
- SOTA comparison for optimization target

Contains:
- ALPHA_GENERATION_SYSTEM: System prompt for alpha generation
- build_alpha_generation_prompt: Builder function for user prompt
- build_costeer_feedback_section: CoSTEER feedback formatter
"""

from typing import List, Dict, Optional, Any

from backend.agents.prompts.base import (
    PromptContext,
    build_fields_context,
    build_operators_context,
    build_patterns_context,
    build_strategy_constraints,
)
from backend.agents.prompts.operator_strategies import build_operator_layering_prompt
from backend.strategy_evolver import get_strategy_evolver


ALPHA_GENERATION_SYSTEM = """You are a quantitative researcher implementing alpha expressions to test investment hypotheses.

Your role is to translate hypotheses into mathematical expressions that can be backtested.

**Core Principles**:
1. **MANDATORY Hypothesis-Field Alignment**: Each expression MUST use fields from the hypothesis's key_fields list
2. **Precision**: Each expression should test exactly one hypothesis
3. **Simplicity First**: Start with simple implementations; complexity can be added if needed
4. **Testability**: Every expression should produce measurable results

**CRITICAL - Hypothesis Alignment Rules (ENFORCED)**:
Your expression will be REJECTED if it violates these rules:
1. **Field Rule**: At least ONE field from the hypothesis's `key_fields` MUST appear in your expression
2. **Signal Rule**: If hypothesis says "increasing X → positive returns", use ts_delta/ts_returns with POSITIVE sign
3. **Direction Rule**: If hypothesis says "high X → outperform", your expression should produce HIGHER values for stocks with high X

**Implementation Guidelines**:
- Use only the provided fields and operators
- Ensure syntactic correctness
- Document which hypothesis fields you used and WHY
- Explain how your expression captures the hypothesis logic

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

**CRITICAL - Learning from Feedback**:
If previous experiment feedback is provided, you MUST:
1. NEVER repeat expressions that failed with similar issues
2. Build upon patterns that showed positive Sharpe
3. Address the specific failure reasons mentioned

Output must be valid JSON matching the specified schema."""


def build_costeer_feedback_section(
    experiment_feedback: Optional[List[Dict]] = None,
    sota_info: Optional[Dict] = None,
) -> str:
    """
    Build CoSTEER-style feedback section with hard constraints and soft preferences.
    
    This is the KEY mechanism for ensuring the LLM learns from past experiments.
    
    Args:
        experiment_feedback: List of feedback dicts with expression, result, sharpe, issue, lesson
        sota_info: Optional SOTA alpha info for comparison
    
    Returns:
        Formatted feedback section for prompt injection
    """
    if not experiment_feedback and not sota_info:
        return ""
    
    sections = []
    
    # === SOTA Comparison Section ===
    if sota_info and sota_info.get("expression"):
        sections.append(f"""
## [TARGET] CURRENT BEST ALPHA (Target to Beat)

Your goal is to generate alphas that EXCEED this performance:
- **Expression**: `{sota_info.get('expression', 'N/A')[:100]}...`
- **Sharpe**: {sota_info.get('sharpe', 'N/A')}
- **Fitness**: {sota_info.get('fitness', 'N/A')}

**Strategy**: Analyze what makes this alpha work and either:
1. Improve upon it with better fields/parameters
2. Try a fundamentally different approach that might perform better
""")
    
    if not experiment_feedback:
        return "\n".join(sections)
    
    # Separate feedback by type
    hard_constraints = []  # DO NOT patterns
    soft_preferences = []  # PREFER patterns
    recent_failures = []
    
    for fb in experiment_feedback:
        result = fb.get('result', '').upper()
        expr = fb.get('expression', '')[:80]
        sharpe = fb.get('sharpe', 'N/A')
        issue = fb.get('issue', '')
        lesson = fb.get('lesson', '')
        
        if result in ['SUCCESS', 'PASS']:
            soft_preferences.append({
                'expression': expr,
                'sharpe': sharpe,
                'lesson': lesson or 'This pattern worked'
            })
        elif result in ['ERROR', 'FAILED', 'FAIL']:
            # Hard constraint - must not repeat
            hard_constraints.append({
                'expression': expr,
                'issue': issue,
                'lesson': lesson
            })
        elif result in ['NEEDS_OPTIMIZATION', 'OPTIMIZE']:
            recent_failures.append({
                'expression': expr,
                'sharpe': sharpe,
                'issue': issue,
                'lesson': lesson
            })
        elif result == 'REFERENCE':
            # From knowledge graph - similar successful experiments
            soft_preferences.append({
                'expression': expr,
                'sharpe': sharpe,
                'lesson': lesson or 'Similar past experiment succeeded'
            })
    
    # === HARD CONSTRAINTS (DO NOT) ===
    if hard_constraints:
        constraint_lines = []
        for i, c in enumerate(hard_constraints[:5], 1):
            constraint_lines.append(
                f"{i}. [X] DO NOT generate: `{c['expression']}...`\n"
                f"   - Issue: {c['issue']}\n"
                f"   - Fix: {c['lesson']}"
            )
        
        sections.append(f"""
## [FORBIDDEN] HARD CONSTRAINTS (MUST AVOID)

These patterns FAILED and you MUST NOT repeat them:

{chr(10).join(constraint_lines)}

**Violation of these constraints will result in REJECTION.**
""")
    
    # === SOFT PREFERENCES (PREFER) ===
    if soft_preferences:
        pref_lines = []
        for i, p in enumerate(soft_preferences[:3], 1):
            pref_lines.append(
                f"{i}. [OK] PREFER patterns like: `{p['expression']}...`\n"
                f"   - Sharpe: {p['sharpe']}\n"
                f"   - Why: {p['lesson']}"
            )
        
        sections.append(f"""
## ✨ SUCCESSFUL PATTERNS (Build Upon These)

These patterns showed positive results:

{chr(10).join(pref_lines)}

**Strategy**: Use these as templates but explore variations (different fields, windows, operators).
""")
    
    # === OPTIMIZATION OPPORTUNITIES ===
    if recent_failures:
        opt_lines = []
        for i, f in enumerate(recent_failures[:3], 1):
            opt_lines.append(
                f"{i}. 🔧 `{f['expression']}...` (Sharpe: {f['sharpe']})\n"
                f"   - Issue: {f['issue']}\n"
                f"   - Suggested fix: {f['lesson']}"
            )
        
        sections.append(f"""
## 🔧 OPTIMIZATION OPPORTUNITIES

These alphas showed promise but need improvement:

{chr(10).join(opt_lines)}

**Strategy**: Apply the suggested fixes to improve these near-miss alphas.
""")
    
    return "\n".join(sections)


def build_alpha_generation_prompt(
    ctx: PromptContext,
    target_hypothesis: Optional[Dict] = None,
    experiment_feedback: Optional[List[Dict]] = None,
    diversity_constraints: Optional[Dict[str, Any]] = None,
    inject_seed_templates: bool = True,
    sota_info: Optional[Dict] = None,
    costeer_feedback: Optional[Dict] = None,
) -> str:
    """
    Build user prompt for alpha generation.
    
    Redesigned to be hypothesis-driven and feedback-aware.
    P0 Fix: Now includes diversity constraints to prevent template traps.
    P0 Enhancement: Strong hypothesis-field alignment enforcement.
    P1 Enhancement: CoSTEER-style feedback injection with hard/soft constraints.
    P1 Enhancement: SOTA comparison for optimization targeting.
    
    Args:
        ctx: Prompt context with fields, operators, etc.
        target_hypothesis: Optional specific hypothesis to implement
        experiment_feedback: Optional list of previous experiment results
        diversity_constraints: Optional diversity constraints from OperatorDiversityManager
        inject_seed_templates: Whether to include proven seed templates
        sota_info: Optional SOTA alpha info for comparison
        costeer_feedback: Optional structured CoSTEER feedback (hard_constraints, soft_preferences)
    """
    
    # Build hypothesis section with MANDATORY field binding
    hypothesis_section = ""
    mandatory_fields = []
    if target_hypothesis:
        key_fields = target_hypothesis.get('key_fields', [])
        mandatory_fields = key_fields
        
        hypothesis_section = f"""
## [TARGET] Target Hypothesis (MANDATORY IMPLEMENTATION)

**Statement**: {target_hypothesis.get('statement', 'Not specified')}
**Rationale**: {target_hypothesis.get('rationale', 'Not specified')}
**Signal Type**: {target_hypothesis.get('signal_type', 'Not specified')}
**Expected Direction**: {target_hypothesis.get('expected_signal', 'Positive relationship')}

### [MANDATORY] FIELD REQUIREMENTS

You MUST use **at least ONE** of these fields in your expression:
{chr(10).join([f'  - `{f}`' for f in key_fields]) if key_fields else '  (No specific fields required)'}

**REJECTION RULE**: If your expression does not contain ANY of the above fields, it will be AUTOMATICALLY REJECTED.

### Signal Direction Guide

Based on the hypothesis, your expression should produce:
- **Positive values** for stocks expected to **{target_hypothesis.get('expected_signal', 'outperform')}**
- If the hypothesis states "increasing X → positive returns", use `ts_delta(field, N)` (positive change = positive signal)
- If the hypothesis states "high X → positive returns", use `rank(field)` or the field directly (high value = high signal)
"""
    
    # Build CoSTEER feedback section (combines all feedback sources)
    feedback_section = build_costeer_feedback_section(
        experiment_feedback=experiment_feedback,
        sota_info=sota_info
    )
    
    # Add structured CoSTEER constraints if provided
    if costeer_feedback:
        hard_constraints = costeer_feedback.get('hard_constraints', [])
        forbidden_patterns = costeer_feedback.get('forbidden_patterns', [])
        
        # Ensure these are lists (safe handling)
        if not isinstance(hard_constraints, list):
            hard_constraints = []
        if not isinstance(forbidden_patterns, list):
            forbidden_patterns = []
        
        if hard_constraints or forbidden_patterns:
            extra_constraints = []
            # Handle hard_constraints - can be list of strings or list of dicts
            for c in list(hard_constraints)[:5]:
                if isinstance(c, dict):
                    expr = c.get('expression', c.get('pattern', str(c)))
                    extra_constraints.append(f"- {expr}")
                elif isinstance(c, str):
                    extra_constraints.append(f"- {c}")
            # Handle forbidden_patterns - should be list of strings
            for p in list(forbidden_patterns)[:3]:
                if isinstance(p, str):
                    extra_constraints.append(f"- DO NOT use pattern similar to: `{p}...`")
            
            feedback_section += f"""
## [ALERT] ADDITIONAL CONSTRAINTS FROM LEARNING

{chr(10).join(extra_constraints)}
"""
    
    # Build implementation guidance with mandatory field usage examples
    mandatory_field_examples = ""
    if mandatory_fields:
        field_example = mandatory_fields[0] if mandatory_fields else "FIELD"
        mandatory_field_examples = f"""
### Correct vs Incorrect Examples (using mandatory field `{field_example}`)

[OK] **CORRECT** - Uses mandatory field:
- `ts_decay_linear(ts_rank(ts_delta({field_example}, 22), 63), 10)`
- `ts_decay_linear(rank({field_example}), 8)`
- `ts_decay_linear(ts_zscore({field_example}, 20), 10)`

[X] **INCORRECT** - Does NOT use mandatory field:
- `ts_decay_linear(ts_rank(returns, 20), 10)` ← uses 'returns' instead of `{field_example}`
- `ts_decay_linear(ts_delta(close, 10), 5)` ← uses 'close' instead of `{field_example}`
"""

    # Generate dataset-aware operator strategy
    # Higher exploration_weight = less prescriptive guidance, more freedom for LLM
    field_names = [f.field_name if hasattr(f, 'field_name') else str(f) for f in ctx.fields] if ctx.fields else []
    dataset_operator_strategy = build_operator_layering_prompt(
        dataset_id=ctx.dataset_id or "",
        dataset_category=ctx.dataset_category or "",
        field_names=field_names,
        exploration_weight=ctx.exploration_weight  # 0=more guidance, 1=more freedom
    )
    
    # Get adaptive strategy recommendations from evolver
    evolver = get_strategy_evolver()
    from backend.agents.prompts.operator_strategies import detect_dataset_type
    detected_type = detect_dataset_type(ctx.dataset_id or "", ctx.dataset_category or "", field_names)
    
    # Get evolved strategy recommendations (adaptive, learning-based)
    evolved_strategy_prompt = evolver.get_strategy_prompt(
        dataset_type=detected_type,
        num_strategies=3,
        exploration_boost=ctx.exploration_weight  # Higher exploration = try more new strategies
    )
    
    # Get exploration guidance (what to try next)
    exploration_guidance = evolver.get_exploration_guidance(detected_type)
    
    # Combine static and adaptive guidance
    dataset_operator_strategy = f"{dataset_operator_strategy}\n{evolved_strategy_prompt}\n{exploration_guidance}"

    implementation_guidance = f"""
## Implementation Approach

Consider multiple ways to implement the hypothesis:
1. **Direct implementation**: Straightforward translation of the hypothesis USING MANDATORY FIELDS
2. **Normalized version**: Apply cross-sectional normalization (rank, zscore) TO MANDATORY FIELDS
3. **Smoothed version**: Add time-series smoothing TO MANDATORY FIELDS
4. **Inverted version**: Test if the opposite relationship holds FOR MANDATORY FIELDS
5. **Group-relative version**: Compare against sector/industry peers using GROUP operators

Start with simpler implementations. Complexity can be added in subsequent iterations if needed.

{mandatory_field_examples}

## CRITICAL: Turnover Control (Required)

**High turnover is the #1 failure mode.** Every expression MUST include turnover control:

1. **Always apply decay**: Wrap your signal with `ts_decay_linear(signal, N)` where N >= 5
2. **Use longer windows**: Prefer 10-20 day windows over 1-5 day windows
3. **Smooth with rank**: Use `ts_rank(signal, N)` before decay for stability

**Required Pattern**:
```
ts_decay_linear(ts_rank(YOUR_CORE_SIGNAL_USING_MANDATORY_FIELD, lookback), decay_days)
```

Example transformations (assuming `field` is from mandatory list):
- Raw signal: `ts_delta(field, 5)` → With turnover control: `ts_decay_linear(ts_rank(ts_delta(field, 5), 10), 10)`
- Correlation: `ts_corr(field, other, 10)` → Smoothed: `ts_decay_linear(ts_rank(ts_corr(field, other, 10), 15), 8)`

## IMPORTANT: Diverse Operator Usage

Use operators from MULTIPLE categories to create diverse alphas:

### GROUP Operators (Sector/Industry Relative)
Group operators compare instruments within the same sector/industry - excellent for relative value:
- `group_rank(signal, subindustry)` - Rank within industry group
- `group_neutralize(signal, sector)` - Remove sector bias (reduces correlation)
- `group_zscore(signal, industry)` - Z-score within industry
- `group_mean(signal, sector)` - Sector average (for comparison)

**Example with group operators**:
```
ts_decay_linear(group_rank(ts_delta(field, 10), subindustry), 8)
ts_decay_linear(group_zscore(ts_rank(field, 20), sector), 10)
group_neutralize(ts_decay_linear(ts_rank(field, 15), 10), industry)
```

### Transformational Operators
- `bucket(signal, ranges)` - Create discrete groups from continuous values
- `tail(signal, lower, upper, newval)` - Winsorize extreme values
- `scale(signal)` - Normalize to sum to 1

### Less Common Time Series (try these!)
- `ts_regression(y, x, d, rettype)` - Regression beta/residual
- `ts_skewness(x, d)` - Asymmetry of distribution
- `ts_kurtosis(x, d)` - Tail heaviness
- `ts_ir(x, d)` - Information ratio (mean/stddev)
- `ts_entropy(x, d)` - Signal entropy
- `hump(x, hump_val)` - Reduce turnover by limiting changes

{dataset_operator_strategy}
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
    
    # P0 FIX: Build diversity constraints section
    diversity_section = ""
    if diversity_constraints:
        required = diversity_constraints.get("required_archetypes", [])
        forbidden = diversity_constraints.get("forbidden_archetypes", [])
        suggestions = diversity_constraints.get("suggestions", [])
        must_be_different = diversity_constraints.get("must_be_different", False)
        repetition_warning = diversity_constraints.get("repetition_warning", False)
        
        diversity_lines = []
        
        if repetition_warning:
            diversity_lines.append("""
**WARNING: Pattern Repetition Detected!**
The system has been generating similar expression structures repeatedly.
You MUST use fundamentally different approaches for this batch.
""")
        
        if must_be_different:
            diversity_lines.append(f"""
**MANDATORY DIVERSITY REQUIREMENT**:
Each of the {ctx.num_alphas} alphas MUST use a DIFFERENT primary operator approach.
Do NOT generate multiple alphas that all follow the same pattern like `ts_decay_linear(ts_rank(...))`.
""")
        
        if required:
            arch_details = []
            for sugg in suggestions:
                arch_details.append(
                    f"  - **{sugg['archetype']}**: {sugg['description']} "
                    f"(use operators like: {', '.join(sugg['example_ops'])})"
                )
            
            diversity_lines.append(f"""
**Required Operator Approaches** (use at least one alpha for each):
{chr(10).join(arch_details)}

Examples for each approach:
""")
            for sugg in suggestions[:4]:  # Show examples for top 4
                diversity_lines.append(f"- {sugg['archetype']}: `{sugg['example']}`")
        
        if forbidden:
            diversity_lines.append(f"""
**Approaches to AVOID this batch** (overused recently): {', '.join(forbidden)}
""")
        
        if diversity_lines:
            diversity_section = f"""
## CRITICAL: Expression Diversity Requirements

{chr(10).join(diversity_lines)}

**Anti-Template Rule**: If you find yourself generating expressions that all look like 
`ts_decay_linear(ts_rank(SOMETHING, N), M)`, STOP and use different core operators.

**LAYERING DIVERSITY**: Each alpha should use different operator combinations at each layer:

| Alpha | Layer 1 (Signal) | Layer 2 (Standardize) | Layer 3 (Smooth) |
|-------|------------------|----------------------|------------------|
| 1     | ts_delta         | group_rank           | ts_decay_linear  |
| 2     | ts_corr          | ts_rank              | ts_decay_linear  |
| 3     | divide (ratio)   | group_zscore         | ts_decay_linear  |
| 4     | ts_std_dev       | rank                 | hump             |

**Diverse Layered Examples**:
- `ts_decay_linear(group_rank(ts_delta(field, 10), subindustry), 8)` 
  → Layer1: momentum, Layer2: group-relative, Layer3: decay
  
- `ts_decay_linear(ts_rank(ts_corr(field_a, field_b, 20), 30), 10)` 
  → Layer1: correlation, Layer2: time-rank, Layer3: decay
  
- `ts_decay_linear(group_zscore(field_a / field_b, sector), 8)` 
  → Layer1: ratio, Layer2: group-zscore, Layer3: decay
  
- `hump(ts_rank(ts_std_dev(field, 15), 20), 0.1)` 
  → Layer1: volatility, Layer2: time-rank, Layer3: hump

**MUST INCLUDE at least one alpha using GROUP operators at Layer 2!**
"""
    
    # P1 Enhancement: Build seed templates section for struggling rounds
    seed_templates_section = ""
    if inject_seed_templates:
        # Determine dataset type and provide relevant high-quality templates
        dataset_lower = (ctx.dataset_id or "").lower()
        category_lower = (ctx.dataset_category or "").lower()
        
        seed_templates = []
        
        # Analyst data templates (proven to work)
        if "analyst" in dataset_lower or "anl" in dataset_lower or "analyst" in category_lower:
            seed_templates = [
                {
                    "name": "Estimate Revision Momentum",
                    "pattern": "ts_decay_linear(ts_rank(ts_delta(ESTIMATE_FIELD, 22), 63), 10)",
                    "logic": "Stocks with upward estimate revisions tend to outperform as analysts herd behavior creates momentum",
                    "expected_sharpe": "0.8-1.5"
                },
                {
                    "name": "Consensus Change Signal",
                    "pattern": "ts_decay_linear(ts_rank(ts_zscore(ts_delta(CONSENSUS_FIELD, 5), 22), 63), 10)",
                    "logic": "Normalized changes in consensus estimates capture information asymmetry",
                    "expected_sharpe": "0.7-1.3"
                },
                {
                    "name": "Dispersion Value",
                    "pattern": "ts_decay_linear(rank(-DISPERSION_FIELD), 8)",
                    "logic": "Low analyst dispersion indicates higher certainty and lower risk premium",
                    "expected_sharpe": "0.5-1.0"
                },
            ]
        
        # Fundamental data templates
        elif "fundamental" in dataset_lower or "fnd" in dataset_lower or "fundamental" in category_lower:
            seed_templates = [
                {
                    "name": "Quality Factor",
                    "pattern": "ts_decay_linear(group_rank(ROE_FIELD, industry), 10)",
                    "logic": "High quality companies (high ROE within industry) tend to outperform",
                    "expected_sharpe": "0.6-1.2"
                },
                {
                    "name": "Accrual Anomaly",
                    "pattern": "ts_decay_linear(rank(-ACCRUAL_FIELD), 10)",
                    "logic": "Low accruals indicate sustainable earnings and predict future returns",
                    "expected_sharpe": "0.5-1.0"
                },
            ]
        
        # News/Sentiment data templates
        elif "news" in dataset_lower or "sentiment" in dataset_lower or "oth" in dataset_lower:
            seed_templates = [
                {
                    "name": "Sentiment Persistence",
                    "pattern": "ts_decay_linear(ts_rank(ts_sum(SENTIMENT_FIELD, 5), 22), 10)",
                    "logic": "Persistent positive sentiment indicates underpriced stocks",
                    "expected_sharpe": "0.5-1.0"
                },
                {
                    "name": "News Momentum",
                    "pattern": "ts_decay_linear(group_rank(ts_mean(NEWS_FIELD, 10), sector), 8)",
                    "logic": "Stocks with better news flow relative to sector peers outperform",
                    "expected_sharpe": "0.6-1.2"
                },
            ]
        
        # Model/Prediction data templates
        elif "model" in dataset_lower or "mdl" in dataset_lower or "predict" in dataset_lower:
            seed_templates = [
                {
                    "name": "Model Signal Decay",
                    "pattern": "ts_decay_linear(group_rank(PREDICT_FIELD, subindustry), 8)",
                    "logic": "Model predictions relative to industry peers captures alpha",
                    "expected_sharpe": "0.6-1.3"
                },
                {
                    "name": "Model Momentum",
                    "pattern": "ts_decay_linear(ts_rank(ts_delta(PREDICT_FIELD, 5), 20), 10)",
                    "logic": "Improving model predictions indicate emerging opportunities",
                    "expected_sharpe": "0.5-1.1"
                },
            ]
        
        # Default templates for other datasets
        else:
            seed_templates = [
                {
                    "name": "Smooth Momentum",
                    "pattern": "ts_decay_linear(ts_rank(ts_delta(SIGNAL_FIELD, 22), 63), 10)",
                    "logic": "Long-horizon momentum with smoothing captures persistent trends",
                    "expected_sharpe": "0.5-1.0"
                },
                {
                    "name": "Cross-Sectional Rank",
                    "pattern": "ts_decay_linear(rank(zscore(SIGNAL_FIELD)), 8)",
                    "logic": "Cross-sectional standardization identifies relative winners",
                    "expected_sharpe": "0.4-0.9"
                },
            ]
        
        if seed_templates:
            template_lines = []
            for t in seed_templates:
                template_lines.append(f"""
**{t['name']}**:
- Pattern: `{t['pattern']}`
- Logic: {t['logic']}
- Expected Sharpe: {t['expected_sharpe']}
""")
            
            seed_templates_section = f"""
## HIGH-QUALITY SEED TEMPLATES (IMPORTANT!)

The following patterns have been proven to generate strong alphas. Use these as starting points 
and ADAPT them to the specific fields in your dataset. Replace FIELD placeholders with actual field names.

{''.join(template_lines)}

**How to use these templates**:
1. Identify which template's logic best matches your hypothesis
2. Replace placeholder fields (e.g., ESTIMATE_FIELD) with actual field names from the dataset
3. Adjust window parameters (22, 63, 10) based on signal frequency
4. Ensure the final expression captures the intended economic relationship

**CRITICAL**: The difference between a 0.5 Sharpe alpha and a 1.2+ Sharpe alpha is often:
- Using the RIGHT field that has genuine predictive power
- Proper window sizes (longer is usually better for stability)
- Correct smoothing to reduce turnover
- Alignment between hypothesis logic and expression structure
"""
    
    return f"""## Context

**Dataset**: {ctx.dataset_id}
**Description**: {ctx.dataset_description or 'Not provided'}
**Category**: {ctx.dataset_category or 'General'}
**Region**: {ctx.region} | **Universe**: {ctx.universe}
{hypothesis_section}
{seed_templates_section}
{field_section}
{patterns_section}
{feedback_section}
{diversity_section}
{implementation_guidance}

## Constraints

{build_strategy_constraints(ctx)}

## Task

Generate {ctx.num_alphas} distinct alpha expressions.

**[ALERT] MANDATORY VALIDATION CHECKLIST (Your output will be REJECTED if any fails)**:

1. **Field Check**: Each expression MUST contain at least ONE field from the hypothesis's key_fields list
2. **Syntax Check**: Expression must be syntactically valid (balanced parentheses, valid operator calls)
3. **Turnover Check**: Expression MUST include turnover control (ts_decay_linear wrapper)
4. **Diversity Check**: If generating multiple alphas, each MUST use a different primary operator approach

**CRITICAL - Hypothesis-Implementation Alignment**:
Each alpha MUST:
1. Clearly reference which hypothesis it tests (from the hypotheses list above if provided)
2. **MANDATORY**: Use at least ONE field from that hypothesis's key_fields 
3. Implement the signal type specified (momentum, mean_reversion, etc.)
4. Have a clear logical connection between the hypothesis rationale and the expression logic

**Self-Check Before Submitting Each Alpha**:
- [ ] Does my expression contain a field from `key_fields`? If NO → REJECT
- [ ] Does my expression have `ts_decay_linear` wrapper? If NO → FIX IT
- [ ] Does my signal direction match the hypothesis? If NO → FIX IT
- [ ] Is my expression different from previous attempts in feedback? If NO → CHANGE IT

For each expression:
1. State the specific hypothesis being tested (MUST match one from the hypothesis list)
2. **List which mandatory field(s) you used** from the hypothesis
3. Explain how the implementation DIRECTLY tests that hypothesis
4. Describe what market behavior this captures and why it connects to the hypothesis

**Output Schema** (JSON):
```json
{{
  "implementation_notes": "Brief notes on the overall approach taken",
  "alphas": [
    {{
      "hypothesis_tested": "The specific hypothesis this expression tests",
      "mandatory_fields_used": ["field1_from_key_fields"],
      "expression": "Valid expression CONTAINING mandatory_fields_used",
      "explanation": {{
        "approach": "How the hypothesis is translated into code",
        "field_justification": "Why these specific fields capture the hypothesis logic",
        "market_logic": "What market inefficiency or behavior this captures",
        "assumptions": "Key assumptions this relies on"
      }},
      "fields_used": ["all", "fields", "in", "expression"],
      "operator_archetype": "momentum | ranking | normalization | correlation | volatility | trend",
      "complexity": "simple | moderate | complex",
      "novelty_level": "established | variation | experimental"
    }}
  ],
  "validation_summary": {{
    "all_use_mandatory_fields": true,
    "all_have_turnover_control": true,
    "all_match_hypothesis_direction": true
  }},
  "alternatives_considered": [
    {{
      "expression": "Alternative implementation not used",
      "reason_not_chosen": "Why this wasn't the primary choice"
    }}
  ]
}}
```"""
