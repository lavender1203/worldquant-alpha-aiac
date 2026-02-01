"""
Dataset-aware operator guidance.

IMPORTANT DESIGN PRINCIPLES:
1. NO hardcoded operator lists - operators come from the database
2. Minimal assumptions - let the LLM decide what's best
3. Soft guidance only - observations, not rules
4. Dataset context - help LLM understand the data type

This module provides:
1. Dataset type detection (by name patterns)
2. Contextual hints about data characteristics
3. Soft guidance that doesn't constrain creativity
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DatasetContext:
    """Context about a dataset type - observations, not rules."""
    dataset_type: str
    description: str
    data_characteristics: List[str]  # What's unique about this data
    common_pitfalls: List[str]       # Things that sometimes cause issues
    exploration_hints: List[str]      # Ideas to consider (not prescriptions)


# =============================================================================
# DATASET CONTEXTS (Observations, not rules)
# =============================================================================

DATASET_CONTEXTS: Dict[str, DatasetContext] = {
    
    "news_event": DatasetContext(
        dataset_type="news_event",
        description="Discrete event signals (news, sentiment, filings)",
        data_characteristics=[
            "Events occur sporadically, not every day",
            "Many stocks may have no events on a given day (NaN handling important)",
            "Event recency often matters (recent events may be more relevant)",
        ],
        common_pitfalls=[
            "Calculating changes on sparse data can produce noisy signals",
        ],
        exploration_hints=[
            "Consider how to handle days with no events",
            "Event accumulation over windows may be useful",
        ]
    ),
    
    "analyst_estimates": DatasetContext(
        dataset_type="analyst_estimates",
        description="Analyst forecasts, estimates, revisions",
        data_characteristics=[
            "Updates happen at analyst publication, not daily",
            "Estimate changes may be more informative than levels",
            "Multiple analysts contribute to consensus",
        ],
        common_pitfalls=[
            "Raw estimate levels vary widely across stocks",
        ],
        exploration_hints=[
            "Revisions in estimates may contain information",
            "Comparing to peers might be valuable",
        ]
    ),
    
    "fundamental": DatasetContext(
        dataset_type="fundamental",
        description="Financial statement data (quarterly/annual)",
        data_characteristics=[
            "Updates quarterly or annually, not daily",
            "Comparing across industries may need normalization",
            "Accounting metrics have different meanings by sector",
        ],
        common_pitfalls=[
            "Short lookback windows may not capture enough updates",
        ],
        exploration_hints=[
            "Changes in fundamentals over time may be informative",
            "Industry-relative comparison might be useful",
        ]
    ),
    
    "price_technical": DatasetContext(
        dataset_type="price_technical",
        description="Price, volume, and derived technical data",
        data_characteristics=[
            "Updates daily (high frequency)",
            "Highly correlated with market movements",
            "Transaction costs matter for high-turnover signals",
        ],
        common_pitfalls=[
            "High turnover can erode returns through transaction costs",
        ],
        exploration_hints=[
            "Smoothing may help reduce turnover",
            "Consider both momentum and mean-reversion patterns",
        ]
    ),
    
    "alternative": DatasetContext(
        dataset_type="alternative",
        description="Non-traditional data (web, satellite, social, etc.)",
        data_characteristics=[
            "Coverage may vary across stocks",
            "Data quality and update frequency vary",
            "May have outliers or unusual distributions",
        ],
        common_pitfalls=[
            "Not all stocks may have data (check coverage)",
            "Outliers can dominate signals",
        ],
        exploration_hints=[
            "Robust transformations (rank, log) may help",
            "Consider data coverage when building signals",
        ]
    ),
    
    "ownership": DatasetContext(
        dataset_type="ownership",
        description="Institutional holdings, 13F filings",
        data_characteristics=[
            "13F data updates quarterly with ~45 day delay",
            "Ownership changes may be more informative than levels",
        ],
        common_pitfalls=[
            "Data staleness - filings are delayed",
        ],
        exploration_hints=[
            "Changes in ownership over time may be informative",
        ]
    ),
    
    "model_derived": DatasetContext(
        dataset_type="model_derived",
        description="Pre-computed model scores or signals",
        data_characteristics=[
            "Already processed - may not need additional transformation",
            "Model assumptions are embedded in the data",
        ],
        common_pitfalls=[
            "Over-processing may hurt signal quality",
        ],
        exploration_hints=[
            "Simple transformations may work best",
            "Understand what the model predicts before using",
        ]
    ),
}


# =============================================================================
# DATASET TYPE DETECTION
# =============================================================================

def detect_dataset_type(dataset_id: str, dataset_category: str = "", 
                        field_names: List[str] = None) -> str:
    """
    Detect the dataset type based on ID and category.
    
    Uses simple pattern matching - no assumptions about operators.
    """
    dataset_lower = (dataset_id or "").lower()
    category_lower = (dataset_category or "").lower()
    
    # News/Event data
    if any(kw in dataset_lower or kw in category_lower for kw in 
           ["news", "sentiment", "event", "social", "filing", "press"]):
        return "news_event"
    
    # Analyst/Estimates
    if any(kw in dataset_lower or kw in category_lower for kw in 
           ["analyst", "estimate", "consensus", "forecast", "anl"]):
        return "analyst_estimates"
    
    # Ownership
    if any(kw in dataset_lower or kw in category_lower for kw in 
           ["ownership", "13f", "institutional", "holder", "short"]):
        return "ownership"
    
    # Price/Technical
    if any(kw in dataset_lower or kw in category_lower for kw in 
           ["price", "volume", "return", "technical", "momentum"]):
        return "price_technical"
    
    # Fundamental
    if any(kw in dataset_lower or kw in category_lower for kw in 
           ["fundamental", "financial", "balance", "income", "fnd"]):
        return "fundamental"
    
    # Model/Derived
    if any(kw in dataset_lower or kw in category_lower for kw in 
           ["model", "score", "signal", "factor", "derived"]):
        return "model_derived"
    
    # Alternative
    if any(kw in dataset_lower or kw in category_lower for kw in 
           ["alternative", "web", "satellite", "credit", "app"]):
        return "alternative"
    
    # Default - no specific guidance needed
    return "fundamental"


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def build_operator_layering_prompt(dataset_id: str, dataset_category: str = "",
                                   field_names: List[str] = None,
                                   exploration_weight: float = 0.5) -> str:
    """
    Build dataset context for the prompt.
    
    IMPORTANT: This provides CONTEXT, not rules. The LLM should feel free
    to use any operators in any combination.
    
    Args:
        dataset_id: The dataset identifier
        dataset_category: Optional category hint
        field_names: Optional list of field names
        exploration_weight: 0-1, higher = less guidance
        
    Returns:
        Formatted prompt section with context (not rules)
    """
    dataset_type = detect_dataset_type(dataset_id, dataset_category, field_names)
    context = DATASET_CONTEXTS.get(dataset_type)
    
    if not context:
        return ""
    
    # High exploration = minimal context
    if exploration_weight > 0.7:
        return f"""
## Dataset Context

This is **{context.description}** data.

No specific operator guidance provided - explore freely!
"""
    
    # Normal mode - provide context but emphasize freedom
    characteristics = "\n".join(f"- {c}" for c in context.data_characteristics[:3])
    
    # Only mention ONE pitfall to avoid over-constraining
    pitfall = context.common_pitfalls[0] if context.common_pitfalls else ""
    
    # Only mention ONE hint to spark ideas without prescribing
    hint = context.exploration_hints[0] if context.exploration_hints else ""
    
    prompt = f"""
## Dataset Context: {context.description}

**Data Characteristics** (for your awareness):
{characteristics}

**Something to consider**: {pitfall}

**One idea to explore** (or ignore entirely): {hint}

---

**FREEDOM TO INNOVATE**: The above is just context about the data type.
You are NOT constrained to any particular operators or patterns.
Use whatever combination of operators makes sense for the economic logic
of your hypothesis. The only hard requirement is turnover control for tradability.
"""
    return prompt


def get_dataset_context(dataset_id: str, dataset_category: str = "") -> Optional[DatasetContext]:
    """Get the context object for a dataset (for programmatic access)."""
    dataset_type = detect_dataset_type(dataset_id, dataset_category)
    return DATASET_CONTEXTS.get(dataset_type)
