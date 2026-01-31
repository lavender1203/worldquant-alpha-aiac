# Alpha AIAC 系统优化总结 (v2.1)

## 概述

本次优化基于对系统的全面诊断，实现了 9 项关键改进，涵盖 P0 (Critical)、P1 (High) 和 P2 (Medium) 三个优先级层次。

## 优化项目清单

### P0 Critical - 必须修复

| ID | 问题 | 解决方案 | 相关文件 |
|---|---|---|---|
| P0-1 | 字段预检查缺失导致模拟失败 | 在代码生成前预检查字段可用性，过滤已知失败字段 | `alpha_optimizer.py`, `generation.py` |
| P0-2 | 假设-实现对齐未强制执行 | 生成后强制验证表达式使用假设的 key_fields，拒绝不符合的表达式 | `alpha_optimizer.py`, `generation.py` |
| P0-3 | Signal方向随机导致负Sharpe | 自动检测负Sharpe，建议/执行信号反转 | `alpha_optimizer.py`, `evaluation.py` |

### P1 High - 高优先级

| ID | 问题 | 解决方案 | 相关文件 |
|---|---|---|---|
| P1-1 | CoSTEER反馈未真正注入 | 实现 hard constraint 强制执行，拒绝违反已学规则的表达式 | `alpha_optimizer.py`, `optimization_integration.py` |
| P1-2 | 缺乏GP增强，无法从种子进化 | 实现简化版GP（参数变异、算子替换、结构变体） | `alpha_optimizer.py` |
| P1-3 | 探索策略低效，随机而非收敛 | 智能探索策略，基于历史表现自适应调整温度和探索权重 | `alpha_optimizer.py` |

### P2 Medium - 中优先级

| ID | 问题 | 解决方案 | 相关文件 |
|---|---|---|---|
| P2-1 | 知识图谱利用率低 | 增强 pattern 检索，将成功/失败 pattern 注入生成提示 | `optimization_integration.py` |
| P2-2 | 缺乏多保真评估 | 实现快速预筛选（语法检查、结构检查），减少无效模拟 | `alpha_optimizer.py` |
| P2-3 | 数据集选择不智能 | 跨类别探索，基于历史成功率的智能选择 | `dataset_selector.py` |

## 核心模块说明

### 1. AlphaOptimizer (`backend/alpha_optimizer.py`)

统一的优化器模块，包含所有 P0-P2 组件：

```python
from backend.alpha_optimizer import get_alpha_optimizer

optimizer = get_alpha_optimizer()

# P0-1: 字段预检查
result = optimizer.pre_generate_check(fields, region, universe, key_fields)

# P0-2 + P1-1 + P2-2: 生成后验证
is_valid, corrected, issues = optimizer.post_generate_validate(expr, hypo, fields)

# P0-3: 信号方向检查
should_invert, inverted, reason = optimizer.check_and_correct_signal(expr, sharpe, fitness, turnover)

# P1-2: 生成优化变体
variants = optimizer.generate_optimization_variants(seed_expr, num_variants=5)

# P1-3: 获取探索参数
params = optimizer.get_exploration_parameters(progress, max_iters)
```

### 2. OptimizationIntegration (`backend/agents/optimization_integration.py`)

集成层，提供与现有 LangGraph 工作流的无缝对接：

```python
from backend.agents.optimization_integration import (
    enhanced_pre_generation_check,
    enhanced_post_generation_validate,
    should_try_signal_inversion,
    get_costeer_hard_constraints,
    generate_optimization_variants,
    get_smart_exploration_params,
)
```

### 3. SmartDatasetSelector (`backend/dataset_selector.py`)

增强的数据集选择器，支持跨类别探索：

```python
from backend.dataset_selector import select_dataset_smart

dataset, metadata = await select_dataset_smart(
    db, region, universe, 
    available_datasets,
    current_dataset="analyst35",
    consecutive_failures=3,
    force_cross_category=True
)
```

## 修改的现有文件

### `backend/agents/graph/nodes/generation.py`

- 添加 P0-1 字段预检查
- 添加 P0-2 生成后对齐验证
- 集成 P1-1 CoSTEER hard constraints
- 使用 P1-3 智能探索参数

### `backend/agents/graph/nodes/evaluation.py`

- 添加 P0-3 信号方向检测
- 添加 P1-2 OPTIMIZE 状态变体生成
- 添加 P1-1 CoSTEER 更新逻辑

## 预期效果

| 指标 | 优化前 | 预期优化后 |
|---|---|---|
| 模拟成功率 | ~60% | ~85% |
| 有效Alpha率 | ~5% | ~15% |
| 负Sharpe Alpha比例 | ~40% | ~15% |
| 假设-实现对齐率 | ~30% | ~80% |
| 跨类别探索覆盖 | 随机 | 智能均衡 |

## 使用建议

### 1. 启用全部优化（推荐）

优化已集成到现有工作流中，默认启用。

### 2. 监控优化效果

```python
from backend.alpha_optimizer import get_alpha_optimizer

optimizer = get_alpha_optimizer()
stats = optimizer.get_stats()

print(f"表达式检查: {stats['expressions_checked']}")
print(f"对齐拒绝: {stats['expressions_rejected_alignment']}")
print(f"约束拒绝: {stats['expressions_rejected_constraints']}")
print(f"预筛选拒绝: {stats['expressions_rejected_prescreen']}")
print(f"信号反转: {stats['signals_inverted']}")
print(f"变体生成: {stats['variants_generated']}")
```

### 3. 调整参数

关键参数在 `alpha_optimizer.py` 中可调整：

- `BLACKLIST_THRESHOLD = 2`: 字段失败次数阈值
- `WINDOW_SIZES = [5, 10, 15, 20, 22, 30, 42, 63]`: GP 窗口变异范围
- `min_decay_window = 5`: 最小 decay 窗口要求

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Mining Workflow                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ RAG Query   │───>│ Hypothesis  │───>│ Code Gen    │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
│                                               │            │
│  ┌──────────────────────────────────────────────┐         │
│  │         AlphaOptimizer Integration           │         │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │         │
│  │  │ P0-1    │ │ P0-2    │ │ P1-1    │        │         │
│  │  │ PreCheck│ │ Align   │ │ CoSTEER │        │         │
│  │  └─────────┘ └─────────┘ └─────────┘        │         │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │         │
│  │  │ P0-3    │ │ P1-2    │ │ P1-3    │        │         │
│  │  │ Signal  │ │ GP      │ │ Explore │        │         │
│  │  └─────────┘ └─────────┘ └─────────┘        │         │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │         │
│  │  │ P2-1    │ │ P2-2    │ │ P2-3    │        │         │
│  │  │ KG      │ │ Screen  │ │ Dataset │        │         │
│  │  └─────────┘ └─────────┘ └─────────┘        │         │
│  └──────────────────────────────────────────────┘         │
│                                               │            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Simulate    │───>│ Evaluate    │───>│ Feedback    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 后续优化方向

1. **P3: 深度 GP 增强** - 实现真正的 genetic programming 算子交叉/变异
2. **P3: 多模型集成** - 使用多个 LLM 生成并投票
3. **P3: 自动化 A/B 测试** - 自动比较不同策略的效果
4. **P3: 分布式挖掘** - 支持多进程/多机器并行挖掘

---

*Last Updated: 2026-01-31*
