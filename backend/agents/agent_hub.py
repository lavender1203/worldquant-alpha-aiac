from typing import List, Dict, Optional
import os
import json
import logging
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from backend.config import settings

logger = logging.getLogger("agent_hub")

# --- Schemas ---
class AlphaExpression(BaseModel):
    alpha_expression: str 
    economic_rationale: str
    data_fields_used: List[str]
    operators_used: List[str]

class AlphaExpressions(BaseModel):
    alphas: List[AlphaExpression]

# --- Prompts ---
SYSTEM_PROMPT = """You are an expert quantitative researcher with deep knowledge of WorldQuant Brain syntax. 
Your goal is to generate high-quality, diverse alpha expressions based on given datasets and hypotheses.
You must strictly follow the syntax rules and output JSON only."""

class AgentHub:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        self.model = settings.OPENAI_MODEL

    async def generate_alphas(self, 
                              hypothesis: str, 
                              dataset_context: str, 
                              operator_context: str,
                              n: int = 5) -> List[Dict]:
        
        user_prompt = f"""
        Hypothesis: "{hypothesis}"
        
        Target: Generate {n} diverse alpha expressions.
        
        Dataset Info:
        {dataset_context}
        
        Operators:
        {operator_context}
        
        Requirements:
        1. Valid WorldQuant syntax.
        2. Use specified fields/operators.
        3. Explain logic clearly.
        4. Output JSON format matching schema:
        {{
            "alphas": [
                {{
                    "alpha_expression": "...", 
                    "economic_rationale": "...",
                    "data_fields_used": ["..."],
                    "operators_used": ["..."]
                }}
            ]
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            # Basic cleaning
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
                
            data = json.loads(content)
            return data.get("alphas", [])
            
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return []

agent_hub = AgentHub()
