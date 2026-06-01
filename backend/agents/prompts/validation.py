"""
Validation and optimization prompts.

Redesigned based on RD-Agent's feedback-driven approach:
- Emphasize learning from errors
- Multiple solution paths without prescriptive bias
- Knowledge transfer from similar problems

Contains:
- SELF_CORRECT_SYSTEM: System prompt for self-correction
- OPTIMIZATION_SYSTEM: System prompt for alpha optimization
- build_self_correct_prompt: Builder for self-correction prompt
- build_optimization_prompt: Builder for optimization prompt
"""

from typing import Dict, List, Optional


SELF_CORRECT_SYSTEM = """You are a code debugger helping to fix alpha expressions.

Your role is to:
1. Diagnose why an expression failed
2. Understand the root cause
3. Propose a minimal fix that addresses the issue

**Approach**:
- Focus on fixing the specific error, not rewriting everything
- Consider multiple possible fixes and choose the most appropriate
- Learn from the error pattern for future reference

Be precise and targeted in your corrections."""


OPTIMIZATION_SYSTEM = """You are an alpha researcher helping to improve expression performance.

Your role is to analyze backtest results and suggest modifications that might improve metrics.

**Core Principles**:
1. **Evidence-based**: Base suggestions on the specific feedback, not generic advice
2. **Targeted**: Address the identified issues, don't change things randomly
3. **Multiple paths**: Consider different approaches without assuming one is best
4. **Incremental**: Prefer small changes that test specific hypotheses

**Optimization is hypothesis testing**: Each modification is an experiment to test 
whether a specific change improves performance."""


def build_self_correct_prompt(
    expression: str,
    error_message: str,
    error_type: str,
    available_fields: List[str],
    similar_errors: Optional[List[Dict]] = None
) -> str:
    """
    Build prompt for self-correction.
    
    Redesigned to include learning from similar errors (RD-Agent pattern).
    
    Args:
        expression: The failed expression
        error_message: Error message from the failure
        error_type: Categorized error type
        available_fields: List of valid field names
        similar_errors: Optional list of similar errors and their fixes
    """
    
    # Build similar errors section if available
    similar_section = ""
    if similar_errors:
        examples = []
        for i, err in enumerate(similar_errors[:3], 1):
            examples.append(f"""
**Example {i}**:
- Failed: `{err.get('failed_expression', 'N/A')[:80]}`
- Error: {err.get('error', 'N/A')}
- Fixed: `{err.get('fixed_expression', 'N/A')[:80]}`
- Fix approach: {err.get('fix_description', 'N/A')}
""")
        
        similar_section = f"""
## Similar Errors and Fixes

These similar errors were resolved before. Learn from these patterns:
{''.join(examples)}
"""
    
    return f"""## Failed Expression

```
{expression}
```

## Error Information

**Error Type**: {error_type}
**Error Message**: 
```
{error_message}
```

## Available Fields

The following fields are valid in this context:
```
{', '.join(sorted(available_fields)[:50])}
```

{f"(... and {len(available_fields) - 50} more)" if len(available_fields) > 50 else ""}
{similar_section}

## Task

1. **Diagnose**: What specifically caused this error?
2. **Fix**: What is the minimal change needed to resolve it?
3. **Verify**: Why will the fix work?

Consider multiple possible fixes and choose the most appropriate one.

Use canonical BRAIN operator names when correcting syntax. For standard deviation,
use `ts_std_dev(x, d)` for time-series fields, `group_std_dev(x, group)` for
group operations, and `vec_stddev(x)` for VECTOR aggregations. Do not output
`ts_stddev` or `ts_std`.

**Output Schema** (JSON):
```json
{{
  "diagnosis": {{
    "root_cause": "The specific reason for the error",
    "error_location": "Where in the expression the error occurs",
    "error_category": "syntax | field_name | operator_usage | parameter | other"
  }},
  "fix": {{
    "approach": "Description of the fix approach",
    "fixed_expression": "The corrected expression",
    "changes_made": "Specific changes applied",
    "confidence": "high | medium | low"
  }},
  "alternatives": [
    {{
      "expression": "Alternative fix if applicable",
      "trade_off": "Why this wasn't chosen as primary"
    }}
  ],
  "knowledge_extracted": "If [this error pattern], then [this fix approach]"
}}
```"""


def build_optimization_prompt(
    expression: str,
    metrics: Dict,
    failed_checks: List[str],
    optimization_reason: str,
    brain_checks: Optional[List[Dict]] = None,
    previous_attempts: Optional[List[Dict]] = None
) -> str:
    """
    Build prompt for alpha optimization.
    
    Redesigned to be more evidence-based and include experiment history.
    
    Args:
        expression: The alpha expression to optimize
        metrics: Backtest metrics
        failed_checks: List of failed submission checks
        optimization_reason: Why optimization is suggested
        brain_checks: Optional BRAIN platform official checks
        previous_attempts: Optional previous optimization attempts
    """
    
    # Build metrics section
    metrics_text = f"""
**Core Metrics**:
- IS Sharpe: {metrics.get('sharpe', 'N/A')}
- Fitness: {metrics.get('fitness', 'N/A')}
- Turnover: {metrics.get('turnover', 'N/A')}
- Drawdown: {metrics.get('drawdown', 'N/A')}

**Train/Test Split**:
- Train Sharpe: {metrics.get('train_sharpe', 'N/A')}
- Test Sharpe: {metrics.get('test_sharpe', 'N/A')}
- Train/Test Ratio: {_safe_ratio(metrics.get('test_sharpe'), metrics.get('train_sharpe'))}

**Constraint Metrics**:
- Risk-Neutralized Sharpe: {metrics.get('rn_sharpe', metrics.get('riskNeutralized', {}).get('sharpe', 'N/A'))}
- Investability-Constrained Sharpe: {metrics.get('invest_sharpe', metrics.get('investabilityConstrained', {}).get('sharpe', 'N/A'))}
"""
    
    # Build BRAIN checks section if available
    checks_section = ""
    if brain_checks:
        check_items = []
        for check in brain_checks[:10]:
            name = check.get('name', 'Unknown')
            result = check.get('result', 'N/A')
            limit = check.get('limit')
            value = check.get('value')
            
            if limit is not None and value is not None:
                check_items.append(f"- {name}: {result} (value={value:.3f}, limit={limit:.3f})")
            else:
                check_items.append(f"- {name}: {result}")
        
        checks_section = f"""
## BRAIN Platform Checks (Official)

These are the actual platform checks and their results:
{chr(10).join(check_items)}

Focus on addressing FAIL results to enable submission.
"""
    
    # Build previous attempts section if available
    attempts_section = ""
    if previous_attempts:
        attempt_items = []
        for i, attempt in enumerate(previous_attempts[-5:], 1):
            attempt_items.append(f"""
**Attempt {i}**:
- Modification: {attempt.get('modification_type', 'N/A')}
- Expression: `{attempt.get('expression', 'N/A')[:60]}...`
- Result: Sharpe {attempt.get('sharpe', 'N/A')}, Fitness {attempt.get('fitness', 'N/A')}
- Outcome: {attempt.get('outcome', 'N/A')}
""")
        
        attempts_section = f"""
## Previous Optimization Attempts

Learn from these previous attempts on this alpha:
{''.join(attempt_items)}

Avoid repeating unsuccessful approaches. Build on partial successes.
"""
    
    return f"""## Alpha Under Optimization

```
{expression}
```

## Backtest Results

{metrics_text}
{checks_section}

## Issues Identified

**Failed Checks**: {', '.join(failed_checks) if failed_checks else 'None identified'}
**Optimization Trigger**: {optimization_reason}
{attempts_section}

## Task

Generate targeted modifications to improve this alpha.

**Approach**:
1. Analyze the specific issues identified (not generic problems)
2. Propose modifications that address these issues
3. Each modification should test a specific hypothesis
4. Consider both conventional and unconventional approaches

**Modification Types to Consider** (as appropriate):
- Window adjustment: Different lookback periods
- Normalization: rank(), zscore(), scale()
- Smoothing: ts_decay_linear(), ts_mean()
- Structure changes: Operator substitution, nesting
- Sign exploration: If relationship might be inverted
- Neutralization adjustments: Different risk factor handling

**Output Schema** (JSON):
```json
{{
  "analysis": {{
    "primary_issues": ["List of main issues to address"],
    "likely_causes": ["Potential underlying causes"],
    "optimization_strategy": "Overall approach to improvement"
  }},
  "modifications": [
    {{
      "id": "M1",
      "type": "window | normalization | smoothing | structure | sign | other",
      "expression": "Modified expression",
      "hypothesis": "What this modification tests",
      "expected_impact": "How this might improve metrics",
      "addresses_issue": "Which identified issue this targets",
      "confidence": "high | medium | low"
    }}
  ],
  "priority_order": ["M1", "M2", ...],
  "knowledge_gained": "If [this pattern of metrics], then [these modifications] may help"
}}
```"""


def _safe_ratio(a, b):
    """Safely compute ratio."""
    try:
        if a is None or b is None or b == 0:
            return 'N/A'
        return f"{float(a) / float(b):.2f}"
    except:
        return 'N/A'
