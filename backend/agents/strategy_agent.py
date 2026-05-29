"""
Strategy Agent - Intelligent Next-Round Strategy Generation

Inspired by:
- RD-Agent: Research-driven hypothesis refinement with feedback loops
- Alpha-GPT: Human-AI interactive alpha mining with iterative improvement
- Chain-of-Alpha: Dual-chain (Generation + Optimization) iterative refinement

Core Principles:
1. Multi-dimensional metrics analysis (not just Sharpe)
2. Failure pattern recognition and avoidance
3. Success pattern amplification
4. Exploration-Exploitation balance with context awareness
5. Hypothesis evolution based on feedback
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from backend.agents.services import LLMService, get_llm_service


@dataclass
class RoundMetrics:
    """Comprehensive metrics for a mining round."""
    # Success counts
    total_alphas: int = 0
    simulated_alphas: int = 0
    passed_alphas: int = 0
    failed_alphas: int = 0
    success_rate: float = 0.0
    
    # Quality metrics (from passed alphas)
    best_sharpe: Optional[float] = None
    avg_sharpe: Optional[float] = None
    best_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    avg_turnover: Optional[float] = None
    avg_returns: Optional[float] = None
    
    # Failure analysis
    syntax_errors: int = 0
    simulation_errors: int = 0
    quality_failures: int = 0  # Passed sim but failed quality thresholds
    
    # Error patterns
    common_error_types: List[str] = field(default_factory=list)
    problematic_fields: List[str] = field(default_factory=list)
    problematic_operators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "total_alphas": self.total_alphas,
            "simulated_alphas": self.simulated_alphas,
            "passed_alphas": self.passed_alphas,
            "failed_alphas": self.failed_alphas,
            "success_rate": round(self.success_rate, 3),
            "best_sharpe": self.best_sharpe,
            "avg_sharpe": round(self.avg_sharpe, 3) if self.avg_sharpe else None,
            "best_fitness": self.best_fitness,
            "avg_fitness": round(self.avg_fitness, 3) if self.avg_fitness else None,
            "avg_turnover": round(self.avg_turnover, 3) if self.avg_turnover else None,
            "avg_returns": round(self.avg_returns, 4) if self.avg_returns else None,
            "syntax_errors": self.syntax_errors,
            "simulation_errors": self.simulation_errors,
            "quality_failures": self.quality_failures,
            "common_error_types": self.common_error_types[:5],
            "problematic_fields": self.problematic_fields[:5],
            "problematic_operators": self.problematic_operators[:3],
        }


@dataclass
class NextRoundStrategy:
    """Strategic decisions for the next mining round."""
    # Temperature / Exploration
    temperature: float = 0.7
    exploration_weight: float = 0.5
    
    # Focus adjustments
    focus_hypotheses: List[str] = field(default_factory=list)  # Hypotheses to explore more
    avoid_patterns: List[str] = field(default_factory=list)     # Patterns to avoid
    amplify_patterns: List[str] = field(default_factory=list)   # Successful patterns to amplify
    
    # Field/Operator guidance
    preferred_fields: List[str] = field(default_factory=list)
    avoid_fields: List[str] = field(default_factory=list)
    preferred_operators: List[str] = field(default_factory=list)
    
    # Strategy description
    action_summary: str = ""
    reasoning: str = ""
    
    # Optimization suggestions (Chain-of-Alpha style)
    optimization_suggestions: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "temperature": self.temperature,
            "exploration_weight": self.exploration_weight,
            "focus_hypotheses": self.focus_hypotheses,
            "avoid_patterns": self.avoid_patterns,
            "amplify_patterns": self.amplify_patterns,
            "preferred_fields": self.preferred_fields,
            "avoid_fields": self.avoid_fields,
            "preferred_operators": self.preferred_operators,
            "action": self.action_summary,
            "reasoning": self.reasoning,
            "optimization_suggestions": self.optimization_suggestions[:3],
        }


STRATEGY_SYSTEM_PROMPT = """你是一位专业的Alpha挖掘策略优化专家，擅长分析挖掘结果并制定下一轮的优化策略。

你的核心能力：
1. **失败模式识别**: 从失败案例中提取共性问题（语法错误、字段误用、算子组合问题）
2. **成功模式放大**: 识别成功Alpha的关键因素并建议强化
3. **假设演进**: 基于反馈调整投资假设的方向
4. **探索-利用平衡**: 根据当前进展动态调整探索与利用的权重

参考框架（来自学术论文）：
- RD-Agent: 研究驱动的假设细化，通过反馈循环持续优化
- Alpha-GPT: 人机交互式Alpha挖掘，迭代改进
- Chain-of-Alpha: 双链迭代（生成链+优化链），用回测反馈做局部改写"""


STRATEGY_USER_PROMPT = """## 第 {iteration} 轮挖掘结果分析

### 本轮指标汇总
{metrics_summary}

### 成功案例 ({success_count} 个)
{success_examples}

### 失败案例 ({failure_count} 个)
{failure_examples}

### 历史上下文
- 数据集: {dataset_id}
- 区域: {region}
- 累计成功: {cumulative_success}/{target_goal}
- 已完成轮次: {iteration}/{max_iterations}

### 任务
请分析以上结果，生成下一轮的优化策略。要求：
- 不要把策略收缩成单字段、单骨架、单参数扫参；必须保留机制、字段组合、算子骨架、窗口参数和信号方向多样性。
- OPTIMIZE 案例是“有信息量的弱信号”，需要解释其失败门槛并提出降换手、提 margin、改善 risk neutralization 的变体。
- 字段建议必须来自当前数据集已出现字段或上一轮已有字段，避免虚构字段。

输出JSON格式：

```json
{{
  "metrics_analysis": {{
    "key_findings": ["发现1", "发现2"],
    "bottleneck": "主要瓶颈描述",
    "opportunity": "主要机会描述"
  }},
  "next_strategy": {{
    "temperature": 0.7,
    "exploration_weight": 0.5,
    "focus_hypotheses": ["建议深入探索的假设方向"],
    "avoid_patterns": ["应该避免的模式"],
    "amplify_patterns": ["应该强化的成功模式"],
    "preferred_fields": ["推荐使用的字段"],
    "avoid_fields": ["应该避免的字段"],
    "preferred_operators": ["推荐的算子组合"],
    "action_summary": "简洁的策略描述",
    "reasoning": "策略制定的逻辑推理"
  }},
  "optimization_suggestions": [
    {{
      "type": "parameter_tuning|structure_variation|hypothesis_pivot",
      "target": "具体优化目标",
      "suggestion": "具体建议",
      "expected_impact": "预期影响"
    }}
  ]
}}
```"""


class StrategyAgent:
    """
    Intelligent strategy agent for next-round optimization.
    
    Implements RD-Agent/Alpha-GPT style feedback-driven strategy evolution.
    """
    
    def __init__(self, llm_service: LLMService = None):
        self.llm_service = llm_service or get_llm_service()
        
    def compute_round_metrics(
        self,
        alphas: List[Any],
        failures: List[Dict]
    ) -> RoundMetrics:
        """
        Compute comprehensive metrics from a mining round.
        
        Args:
            alphas: List of Alpha objects (from DB)
            failures: List of failure dicts
            
        Returns:
            RoundMetrics with full analysis
        """
        metrics = RoundMetrics()
        
        # Count totals
        metrics.total_alphas = len(alphas) + len(failures)
        
        # Simulated alphas: All passed alphas + failures that actually ran (QUALITY_CHECK_FAILED)
        quality_failures_count = sum(1 for f in failures if f.get('error_type') == 'QUALITY_CHECK_FAILED')
        metrics.simulated_alphas = len(alphas) + quality_failures_count
        
        # Alphas list contains only successful ones in this context
        metrics.passed_alphas = len(alphas)
        metrics.failed_alphas = len(failures)
        metrics.success_rate = metrics.passed_alphas / max(metrics.total_alphas, 1)
        
        # Compute quality metrics from passed alphas
        passed = [a for a in alphas if getattr(a, 'quality_status', None) == "PASS"]
        if passed:
            sharpes = []
            fitnesses = []
            turnovers = []
            returns_list = []
            
            for a in passed:
                m = getattr(a, 'metrics', {}) or {}
                if isinstance(m, dict):
                    if m.get('sharpe') is not None:
                        sharpes.append(m['sharpe'])
                    if m.get('fitness') is not None:
                        fitnesses.append(m['fitness'])
                    if m.get('turnover') is not None:
                        turnovers.append(m['turnover'])
                    if m.get('returns') is not None:
                        returns_list.append(m['returns'])
            
            if sharpes:
                metrics.best_sharpe = max(sharpes)
                metrics.avg_sharpe = sum(sharpes) / len(sharpes)
            if fitnesses:
                metrics.best_fitness = max(fitnesses)
                metrics.avg_fitness = sum(fitnesses) / len(fitnesses)
            if turnovers:
                metrics.avg_turnover = sum(turnovers) / len(turnovers)
            if returns_list:
                metrics.avg_returns = sum(returns_list) / len(returns_list)
        
        # Analyze failures
        error_types = {}
        problem_fields = {}
        problem_ops = {}
        
        for f in failures:
            err = f.get('error_message', '') or ''
            err_lower = err.lower()
            
            # Categorize error
            if 'syntax' in err_lower or 'parse' in err_lower:
                metrics.syntax_errors += 1
                error_types['SYNTAX_ERROR'] = error_types.get('SYNTAX_ERROR', 0) + 1
            elif 'simulation' in err_lower or 'timeout' in err_lower:
                metrics.simulation_errors += 1
                error_types['SIMULATION_ERROR'] = error_types.get('SIMULATION_ERROR', 0) + 1
            elif 'field' in err_lower or 'not found' in err_lower:
                error_types['FIELD_ERROR'] = error_types.get('FIELD_ERROR', 0) + 1
                # Try to extract field name
                import re
                field_match = re.search(r"field[:\s]+['\"]?(\w+)['\"]?", err_lower)
                if field_match:
                    fname = field_match.group(1)
                    problem_fields[fname] = problem_fields.get(fname, 0) + 1
            else:
                error_types['OTHER'] = error_types.get('OTHER', 0) + 1
        
        # Also count quality failures (simulated but didn't pass thresholds)
        # We already calculated this as part of simulated_alphas logic
        metrics.quality_failures = quality_failures_count
        
        # Sort and extract top patterns
        metrics.common_error_types = sorted(error_types.keys(), key=lambda x: error_types[x], reverse=True)
        metrics.problematic_fields = sorted(problem_fields.keys(), key=lambda x: problem_fields[x], reverse=True)
        
        return metrics
    
    async def generate_strategy(
        self,
        iteration: int,
        max_iterations: int,
        alphas: List[Any],
        failures: List[Dict],
        dataset_id: str,
        region: str,
        cumulative_success: int,
        target_goal: int,
        previous_strategy: Optional[NextRoundStrategy] = None
    ) -> NextRoundStrategy:
        """
        Generate intelligent next-round strategy using LLM analysis.
        
        Args:
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            alphas: Alpha objects from this round
            failures: Failure records from this round
            dataset_id: Current dataset
            region: Market region
            cumulative_success: Total successes so far
            target_goal: Target number of successful alphas
            previous_strategy: Strategy from previous round (for continuity)
            
        Returns:
            NextRoundStrategy with comprehensive guidance
        """
        # Compute metrics
        metrics = self.compute_round_metrics(alphas, failures)
        
        # Prepare success/promising examples. OPTIMIZE candidates are not final
        # successes, but they are valuable feedback for the next round.
        passed = [a for a in alphas if getattr(a, 'quality_status', None) == "PASS"]
        promising = [
            a for a in alphas
            if getattr(a, 'quality_status', None) == "OPTIMIZE"
        ]
        success_examples = []
        for a in (passed + promising)[:6]:
            m = getattr(a, 'metrics', {}) or {}
            label = getattr(a, 'quality_status', None) or "UNKNOWN"
            failed_gates = m.get("_strict_gate_failures") or m.get("_failed_tests") or []
            success_examples.append(
                f"- [{label}] Expr: {getattr(a, 'expression', 'N/A')[:100]}...\n"
                f"  Sharpe: {m.get('sharpe', 'N/A')}, Fitness: {m.get('fitness', 'N/A')}, "
                f"Turnover: {m.get('turnover', 'N/A')}, Margin: {m.get('margin', 'N/A')}\n"
                f"  Failed gates: {failed_gates[:4] if isinstance(failed_gates, list) else failed_gates}"
            )
        success_text = "\n".join(success_examples) if success_examples else "无成功案例"
        
        # Prepare failure examples
        failure_examples = []
        for f in failures[:5]:
            failure_examples.append(
                f"- Expr: {f.get('expression', 'N/A')[:80]}...\n"
                f"  Error: {f.get('error_message', 'N/A')[:100]}"
            )
        failure_text = "\n".join(failure_examples) if failure_examples else "无失败记录"
        
        # Metrics summary
        metrics_summary = f"""
- 总Alpha数: {metrics.total_alphas}
- 通过数: {metrics.passed_alphas} (成功率: {metrics.success_rate:.1%})
- 最佳Sharpe: {metrics.best_sharpe or 'N/A'}
- 平均Sharpe: {metrics.avg_sharpe or 'N/A'}
- 最佳Fitness: {metrics.best_fitness or 'N/A'}
- 平均Fitness: {metrics.avg_fitness or 'N/A'}
- 平均Turnover: {metrics.avg_turnover or 'N/A'}
- 语法错误: {metrics.syntax_errors}
- 模拟错误: {metrics.simulation_errors}
- 质量未达标: {metrics.quality_failures}
- 常见错误类型: {', '.join(metrics.common_error_types) or 'N/A'}
- 问题字段: {', '.join(metrics.problematic_fields) or 'N/A'}
"""
        
        # Build prompt
        prompt = STRATEGY_USER_PROMPT.format(
            iteration=iteration,
            metrics_summary=metrics_summary,
            success_count=len(passed),
            success_examples=success_text,
            failure_count=len(failures),
            failure_examples=failure_text,
            dataset_id=dataset_id,
            region=region,
            cumulative_success=cumulative_success,
            target_goal=target_goal,
            max_iterations=max_iterations
        )
        
        try:
            response = await self.llm_service.call(
                system_prompt=STRATEGY_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.5,
                json_mode=True
            )
            
            if response.success and response.parsed:
                data = response.parsed
                next_strat = data.get("next_strategy", {})
                
                strategy = NextRoundStrategy(
                    temperature=next_strat.get("temperature", 0.7),
                    exploration_weight=next_strat.get("exploration_weight", 0.5),
                    focus_hypotheses=next_strat.get("focus_hypotheses", []),
                    avoid_patterns=next_strat.get("avoid_patterns", []),
                    amplify_patterns=next_strat.get("amplify_patterns", []),
                    preferred_fields=next_strat.get("preferred_fields", []),
                    avoid_fields=next_strat.get("avoid_fields", []),
                    preferred_operators=next_strat.get("preferred_operators", []),
                    action_summary=next_strat.get("action_summary", ""),
                    reasoning=next_strat.get("reasoning", ""),
                    optimization_suggestions=data.get("optimization_suggestions", [])
                )
                
                logger.info(f"[StrategyAgent] Generated strategy: {strategy.action_summary}")
                return strategy
                
        except Exception as e:
            logger.error(f"[StrategyAgent] LLM strategy generation failed: {e}")
        
        # Fallback: Rule-based strategy
        return self._fallback_strategy(metrics, iteration, cumulative_success, target_goal)
    
    def _fallback_strategy(
        self,
        metrics: RoundMetrics,
        iteration: int,
        cumulative_success: int,
        target_goal: int
    ) -> NextRoundStrategy:
        """Rule-based fallback strategy when LLM fails."""
        strategy = NextRoundStrategy()
        
        progress = cumulative_success / max(target_goal, 1)
        
        # Temperature adjustment based on success rate
        if metrics.success_rate == 0:
            # Full failure - increase exploration significantly
            strategy.temperature = min(1.0, 0.7 + 0.1 * iteration)
            strategy.exploration_weight = min(0.9, 0.5 + 0.1 * iteration)
            strategy.action_summary = "探索模式: 全轮失败，大幅提升多样性"
            strategy.reasoning = f"本轮成功率为0，需要跳出当前搜索空间，增加Temperature至{strategy.temperature}，探索权重至{strategy.exploration_weight}"
        elif metrics.success_rate < 0.3:
            # Low success - moderate exploration increase
            strategy.temperature = 0.8
            strategy.exploration_weight = 0.6
            strategy.action_summary = "调整模式: 成功率偏低，适度增加探索"
            strategy.reasoning = f"成功率{metrics.success_rate:.1%}偏低，适度提升探索以发现新方向"
        elif metrics.success_rate > 0.5:
            # Good success - shift to exploitation
            strategy.temperature = max(0.3, 0.7 - 0.1 * metrics.passed_alphas)
            strategy.exploration_weight = 0.3
            strategy.action_summary = "收敛模式: 成功率良好，强化成功模式"
            strategy.reasoning = f"成功率{metrics.success_rate:.1%}良好，降低Temperature以稳定产出"
        else:
            # Moderate success - balanced
            strategy.temperature = 0.7
            strategy.exploration_weight = 0.5
            strategy.action_summary = "平衡模式: 维持当前策略"
            strategy.reasoning = "成功率适中，保持探索与利用的平衡"
        
        # Add problematic fields to avoid list
        strategy.avoid_fields = metrics.problematic_fields[:3]
        
        return strategy


def create_strategy_agent(llm_service: LLMService = None) -> StrategyAgent:
    """Factory function to create StrategyAgent."""
    return StrategyAgent(llm_service=llm_service)
