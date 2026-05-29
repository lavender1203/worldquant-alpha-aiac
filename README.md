# AIAC 2.0 - Alpha-GPT Mining System

<div align="center">

![Version](https://img.shields.io/badge/version-2.1.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![React](https://img.shields.io/badge/react-18.x-61DAFB)
![License](https://img.shields.io/badge/license-MIT-yellow)

**Human-AI Collaborative Alpha Mining Platform**  
*基于 Alpha-GPT 范式 + RD-Agent CoSTEER 反馈闘环*

[English](#features) | [中文](#功能特性)

</div>

---

## 🌟 Overview

AIAC 2.0 是一个基于 **Alpha-GPT** 理念的智能 Alpha 挖掘系统，融合了 **RD-Agent** 的 CoSTEER 反馈闘环机制，实现：

- 🎯 **每日稳定产出 3-4 个合格 Alpha**
- 🔄 **持续多样性探索**（跨区域、跨数据集）
- 👁️ **全链路 Trace 可视化**（RD-Agent 风格）
- 🧠 **知识库自演进**（成功模式 + 失败教训）
- 🤝 **人机协作**（Human-in-the-Loop 干预）

---

## 📋 Features

### 核心功能

| 模块 | 功能 | 状态 |
|------|------|------|
| **Dashboard** | 每日目标进度、KPI 卡片、实时活动流 | ✅ |
| **Task Management** | 任务创建、启动/暂停、Trace 时间线 | ✅ |
| **Alpha Lab** | Alpha 列表、详情、人工反馈、**Brain 同步** | ✅ |
| **Config Center** | 质量门槛、算子偏好、知识库管理 | ✅ |
| **Mining Agent** | Hierarchical RAG + LLM 生成 | ✅ |
| **Feedback Loop** | CoSTEER 双循环（自修正 + 知识演进） | ✅ |
| **Genetic Optimizer** | Alpha 表达式变异优化 (6 种变异类型) | ✅ 可选接入 |
| **Diversity Tracker** | 探索多样性追踪 & 建议 | ✅ 主链路接入 |
| **External Knowledge** | 论坛 & 学术论文模式导入 | ✅ 定时导入 |
| **Metrics Tracker** | Session/Round/Alpha 指标追踪 | ✅ 主链路接入 |
| **Benchmark Test** | 系统效果基准测试 | ✅ 🆕 |
| **Test Suite** | 综合测试 + 回归检测 | ⚠️ 需安装测试依赖 |

### 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)              │
│  Dashboard | Tasks | Alpha Lab | Config | Data Mgmt     │
└─────────────────────────────────────────────────────────┘
                            │ REST API + SSE
┌─────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Agent Hub (LangGraph)               │    │
│  │  Mining | Strategy | Feedback | Field Screener  │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │            Core Services                         │    │
│  │  Knowledge Base | Prompt Engine | BRAIN Adapter │    │
│  │  Credentials | Mining Service | Evolution        │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │         Advanced Optimization (NEW)              │    │
│  │  Genetic Optimizer | Diversity Tracker          │    │
│  │  External Knowledge | Metrics Tracker            │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                            │
         ┌─────────────────┴─────────────────┐
         │                                   │
┌───────────────────┐       ┌───────────────────┐
│   PostgreSQL     │       │   Redis + Celery  │
│   (SQLAlchemy)   │       │   (异步任务)       │
└───────────────────┘       └───────────────────┘
```

### 新增高级功能 (v2.1)

| 模块 | 功能 | 描述 |
|------|------|------|
| **Genetic Optimizer** | Alpha 遗传优化 | 可通过 `task.config.enable_genetic_optimization=true` 在优化候选上启用预算型遗传搜索 |
| **Diversity Tracker** | 多样性追踪 | 主挖掘循环会记录探索组合，并把未充分探索方向注入下一轮策略 |
| **External Knowledge** | 外部知识集成 | Celery Beat 会定时导入内置学术模式；论坛同步需要 MCP/forum client |
| **Metrics Tracker** | 指标追踪器 | 主挖掘循环会记录 session/round 指标、pass rate、diversity score 和知识库快照 |
| **Benchmark Test** | 效果基准测试 | 系统组件测试、模拟评估、改进建议生成 |
| **Test Suite** | 综合测试套件 | 18 个测试 (单元/集成/回归/E2E)，基准对比，回归检测 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+ (用于 Celery 异步任务)
- WorldQuant BRAIN 账号

### 方法一：一键启动 (推荐)

**Windows:**
```bash
# 启动服务（默认：重启模式，会先清理残留进程）
run.bat

# 仅启动（不清理）
run.bat --start

# 停止所有服务
run.bat --stop

# 指定端口
run.bat --port 8002
```

**Linux/macOS:**
```bash
chmod +x run.sh

# 启动服务（默认：重启模式）
./run.sh

# 仅启动
./run.sh --start

# 停止所有服务
./run.sh --stop

# 指定端口
./run.sh --port 8002
```

**支持的参数:**
| 参数 | 说明 |
|------|------|
| `--start` | 启动服务（跳过已运行的） |
| `--restart` | 停止并重新启动（默认） |
| `--stop` / `--end` | 停止所有服务 |
| `--port NUM` | 指定后端端口（默认 8001） |
| `-h` / `--help` | 显示帮助信息 |

启动脚本会**自动检测**环境状态：
1. 清理残留进程（restart 模式）
2. 检查 `.env` 配置文件（不存在则创建并打开编辑）
3. 检测 Python 虚拟环境和依赖（缺失自动安装）
4. 检测 Node.js 依赖（缺失自动安装）
5. 检测数据库连接（失败自动创建）
6. 运行数据库迁移 (Alembic)
7. 启动 Backend、Frontend 和 Celery Worker

### 方法二：手动启动

#### 1. Clone & Setup

```bash
git clone https://github.com/your-repo/worldquant-alpha-aiac.git
cd worldquant-alpha-aiac

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

#### 2. Configure Environment

复制并编辑 `.env` 文件:

```bash
cp .env.example .env
```

```env
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here
POSTGRES_SERVER=localhost
POSTGRES_PORT=5433
POSTGRES_DB=alpha_gpt

# Redis (for Celery and SSE)
REDIS_URL=redis://localhost:6579/0

# WorldQuant BRAIN Platform
BRAIN_EMAIL=your_email@example.com
BRAIN_PASSWORD=your_brain_password

# LLM Configuration (OpenAI Compatible)
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

# Mining Defaults (optional)
DEFAULT_REGION=USA
DEFAULT_UNIVERSE=TOP3000
DEFAULT_DAILY_GOAL=4

# Optional: comma-separated pyramid categories that AUTO mining must avoid.
# Useful when some pyramid categories are already lit for the current quarter.
MINING_EXCLUDE_CATEGORIES=pv,analyst,model,option,shortinterest,other

# Quality Thresholds (optional)
SHARPE_MIN=1.58
TURNOVER_MAX=0.30
FITNESS_MIN=1.0
MAX_CORRELATION=0.7
```

#### 3. Initialize Database

数据库和表结构会自动创建，无需手动操作：

- **数据库**: 由 `init_database.py` 自动创建（首次运行时）
- **表结构**: 当前由 `backend/database.py` 的 `metadata.create_all` 在启动时创建

只需确保 PostgreSQL 服务正在运行且 `.env` 中的凭证配置正确即可。

```bash
# 可选：手动创建数据库（如果自动创建失败）
python backend/migrations/init_database.py

cd backend && alembic upgrade head
```

#### 数据库迁移 (Alembic)

项目已包含初始迁移 `backend/alembic/versions/0001_initial_schema.py`。本地开发启动仍会通过 `metadata.create_all` 兜底创建表；生产环境建议使用 Alembic 管理 schema。

```bash
cd backend

# 查看当前迁移版本
alembic current

# 应用所有迁移
alembic upgrade head

# 生成新迁移（修改模型后）
alembic revision --autogenerate -m "描述变更"

# 回滚一个版本
alembic downgrade -1

# 查看迁移历史
alembic history
```

#### 4. Start Services

**Terminal 1 - Backend:**
```bash
uvicorn backend.main:app --reload --port 8001
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal 3 - Celery Worker (可选，用于后台任务):**
```bash
celery -A backend.celery_app worker --loglevel=info --pool=solo
```

**Access:**
- Frontend: http://localhost:5174
- API Docs: http://localhost:8001/docs
- API: http://localhost:8001/api/v1

---

## 📁 Project Structure

```
worldquant-alpha-aiac/
├── backend/
│   ├── main.py                 # FastAPI 应用入口
│   ├── config.py               # 配置管理
│   ├── database.py             # 数据库连接
│   ├── models.py               # SQLAlchemy 模型
│   ├── celery_app.py           # Celery 配置
│   ├── tasks.py                # Celery 任务定义
│   ├── alpha_scoring.py        # Alpha 综合评估 & 自适应阈值
│   ├── alpha_semantic_validator.py # Alpha 语义验证器
│   ├── dataset_selector.py     # 数据集选择器
│   ├── benchmark_test.py       # 🆕 效果基准测试工具
│   ├── diversity_tracker.py    # 🆕 多样性追踪器
│   ├── external_knowledge.py   # 🆕 外部知识集成
│   ├── genetic_optimizer.py    # 🆕 遗传优化器
│   ├── metrics_tracker.py      # 🆕 指标追踪器
│   ├── routers/
│   │   ├── dashboard.py        # 统计 & Live Feed
│   │   ├── tasks.py            # 任务管理 & Trace
│   │   ├── alphas.py           # Alpha 管理 & 反馈
│   │   ├── knowledge.py        # 知识库管理
│   │   ├── mining.py           # Legacy API（默认不注册，主入口为 tasks.py）
│   │   ├── config.py           # 配置 API
│   │   ├── datasets.py         # 数据集 API
│   │   ├── operators.py        # 算子 API
│   │   └── analysis.py         # 分析 API
│   ├── services/
│   │   ├── mining_service.py   # Legacy 挖掘服务（主链路为 agents/mining_agent.py）
│   │   ├── analysis_service.py # 分析服务
│   │   └── credentials_service.py # 凭证管理服务
│   ├── agents/
│   │   ├── agent_hub.py        # Agent 统一入口
│   │   ├── mining_agent.py     # 挖掘 Agent
│   │   ├── feedback_agent.py   # 反馈 Agent (增强失败分类)
│   │   ├── strategy_agent.py   # 策略 Agent
│   │   ├── field_screener.py   # 字段筛选器
│   │   ├── evolution_strategy.py # 进化策略
│   │   ├── knowledge_seed.py   # 知识库种子 (101-Alpha 模式)
│   │   ├── prompts.py          # Prompt 模板
│   │   ├── graph/              # LangGraph 工作流
│   │   └── services/           # Agent 内部服务 (含 RAG)
│   ├── adapters/
│   │   ├── brain_adapter.py    # BRAIN API 封装
│   │   └── brain.py            # BRAIN 底层接口
│   ├── tests/
│   │   ├── test_suite.py       # 🆕 综合测试套件 (推荐)
│   │   ├── test_integration.py # 集成测试
│   │   └── baseline.json       # 回归测试基准
│   ├── migrations/
│   │   └── init_database.py    # 数据库初始化辅助脚本
│   └── alembic/                # 🆕 数据库迁移 (Alembic)
│       ├── env.py              # Alembic 环境配置
│       └── versions/           # 迁移脚本
├── frontend/
│   ├── Dockerfile              # 前端 Docker 镜像
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css           # 暗色主题
│   │   ├── components/
│   │   │   ├── AppSidebar.jsx
│   │   │   └── AppHeader.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── TaskManagement.jsx
│   │   │   ├── TaskDetail.jsx
│   │   │   ├── AlphaLab.jsx
│   │   │   ├── AlphaDetail.jsx
│   │   │   ├── ConfigCenter.jsx
│   │   │   └── DataManagement.jsx  # 数据管理
│   │   └── services/
│   │       └── api.js
│   ├── package.json
│   └── vite.config.js
├── data/                       # 设计文档
│   ├── 需求说明文档.md
│   ├── 详细设计说明文档.md
│   └── ui_design_spec.md
├── .env.example                # 环境变量模板
├── requirements.txt            # Python 依赖
├── docker-compose.yml          # Docker 编排配置
├── Dockerfile.backend          # 后端 Docker 镜像
├── run.bat                     # Windows 启动/停止脚本 (支持 --start/--stop/--restart)
├── run.sh                      # Linux/macOS 启动/停止脚本
└── README.md
```

---

## 🔧 API Reference

### Dashboard

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/stats/daily` | GET | 今日挖掘统计 |
| `/api/v1/stats/kpi` | GET | KPI 指标 |
| `/api/v1/stats/live-feed` | GET | SSE 实时活动流 |

### Tasks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tasks` | GET | 任务列表 |
| `/api/v1/tasks` | POST | 创建任务 |
| `/api/v1/tasks/{id}` | GET | 任务详情 (含 Trace) |
| `/api/v1/tasks/{id}/trace` | GET | 完整 Trace 时间线 |
| `/api/v1/tasks/{id}/start` | POST | 启动任务 |
| `/api/v1/tasks/{id}/intervene` | POST | 人工干预 (暂停/跳过/调整) |

### Alphas

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/alphas` | GET | Alpha 列表 |
| `/api/v1/alphas/sync` | POST | 同步 Brain 平台 Alpha |
| `/api/v1/alphas/{id}` | GET | Alpha 详情 |
| `/api/v1/alphas/{id}/feedback` | POST | 提交人工反馈 |

### Knowledge

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/knowledge` | GET | 知识库条目 |
| `/api/v1/knowledge/success-patterns` | GET | 成功模式 |
| `/api/v1/knowledge/failure-pitfalls` | GET | 失败教训 |

---

## 🐳 Docker Deployment

Docker 方式适合生产环境部署，包含完整的服务编排。

### 服务组件

| 服务 | 端口 | 说明 |
|------|------|------|
| `db` | 5433:5432 | PostgreSQL 数据库 |
| `redis` | 6579:6379 | Redis (Celery & SSE) |
| `backend` | 8000:8000 | FastAPI 后端 |
| `frontend` | 3000:3000 | React 前端 |
| `celery-worker` | - | Celery 工作进程 |
| `celery-beat` | - | Celery 定时任务 |

### 使用方法

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的配置

# 2. 构建并启动所有服务
docker-compose up -d

# 3. 查看日志
docker-compose logs -f

# 4. 查看特定服务日志
docker-compose logs -f backend

# 5. 停止所有服务
docker-compose down

# 6. 停止并删除数据卷（清空数据）
docker-compose down -v
```

### Docker 访问地址

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📊 Key Concepts

### Trace Visualization (RD-Agent Style)

每个挖掘任务的步骤完全透明，对应 `TraceStepType` 枚举：

```
Step 1: RAG_QUERY     → 检索知识库成功模式
Step 2: HYPOTHESIS    → 生成投资假设
Step 3: CODE_GEN      → 生成 Alpha 表达式
Step 4: VALIDATE      → 语法校验
Step 5: SIMULATE      → BRAIN 平台模拟
Step 6: SELF_CORRECT  → 失败时自我修正（可选）
Step 7: EVALUATE      → 质量评估（Sharpe/Turnover/Fitness）
```

### Evolution Loop (进化循环)

Mining Agent 支持多轮进化挖掘：

```
Round 1: 初始策略 → 生成 Alpha → 分析结果 → 策略演进
Round 2: 新策略 → 字段过滤 → 生成 Alpha → 分析结果 → 策略演进
Round N: 累积学习 → 达到目标或最大迭代
```

关键组件：
- **EvolutionStrategy**: 控制字段偏好、算子选择
- **RoundResult**: 记录每轮成功/失败统计
- **FeedbackAgent**: 从失败中学习优化策略

### BRAIN Multi-Simulation Guardrails

当前主挖掘链路只应通过批量 multi-simulation 路径回测候选表达式：

- `node_simulate` 每轮默认提交 4 个表达式。
- `BrainAdapter.simulate_batch()` 会拒绝单表达式批量请求，避免 BRAIN 返回 `400 Multi-simulations require multiple simulations...`。
- 优化链如果某个设置分组只剩 1 个表达式，会跳过该单例，不会退化成 single simulation。
- direct BRAIN multi-simulation 等待上限为 1200 秒，并每约 60 秒输出进度日志。
- MCP `create_multi_simulation` 可在 Config Center 启用，但实测它是黑盒长连接；如果启用，`MCPBrainAdapter` 有 600 秒硬超时，超时后回退到 direct BRAIN multi-simulation。长时间挖掘建议优先使用 direct multi-simulation，保留 MCP correlation tools。

### Live Mining Notes (USA D1 news12)

2026-05-29 使用本项目 Celery/LangGraph 链路实测 `USA TOP3000 D1 REGULAR maxTrade=ON news12`：

- 字段机制分类已生效：`news_pct_*`、`news_high/low_exc_stddev`、`news_ratio_vol`、`news_open_gap` 等会被分到 fast reaction、delayed drift、volatility、volume/liquidity、crowding/context 机制，避免全部落入 `General` 后策略塌缩。
- 生成质量改善：批次能覆盖不同机制、算子骨架、参数窗口和信号方向，重复自纠没有再吞掉有效批次。
- 根本效果瓶颈仍在因子本身：news12 短窗口事件信号换手接近 1.0，margin 约 0.14-3.9bp，risk neutralization 不达标；当前观察到的最好候选仍只是 OPTIMIZE，未满足提交门槛。
- 信号反转捕获已生效：负 Sharpe 且绝对值足够的表达式会被标记为 `NEGATIVE_SIGNAL_REVERSAL`，进入优化链而不是直接丢弃。
- 运行层瓶颈：BRAIN multi-simulation 在拥堵时可能超过 900 秒；超时后的 alpha 回查仍是待做项。

### CoSTEER Feedback Loop

**短循环** (单 Alpha 内):
```
生成 → 模拟 → 失败 → SELF_CORRECT → 重试（最多3次）
```

**长循环** (跨任务):
```
失败样本 → FeedbackAgent 聚类归因 → 更新 KnowledgeEntry → 优化 Prompt
```

### Knowledge Base (知识库)

知识库类型（`KnowledgeEntryType`）：
- **SUCCESS_PATTERN**: 成功的 Alpha 模式
- **FAILURE_PITFALL**: 失败教训
- **FIELD_BLACKLIST**: 问题字段黑名单
- **OPERATOR_STAT**: 算子使用统计

### Human-in-the-Loop

- 任意步骤可暂停/调整（`AgentMode.INTERACTIVE`）
- 👍/👎 反馈直接影响知识库（`HumanFeedback`）
- Alpha 优化候选自动识别

### Genetic Optimizer (遗传优化器)

基于遗传算法的 Alpha 表达式优化器，支持系统化的变异搜索：

```python
# 变异类型
- operator_substitution  # 算子替换 (ts_rank -> ts_zscore)
- window_parameter       # 窗口参数调整 (5 -> 20)
- add_wrapper           # 添加包装函数 (rank, decay)
- sign_flip             # 符号翻转
- structure_modification # 结构修改 (添加中性化)
```

关键特性：
- 多目标适应度函数 (Sharpe + Fitness + Turnover)
- 自适应变异率（根据成功历史调整）
- 种群多样性维护
- 精英保留策略
- 默认不额外消耗 BRAIN 模拟预算；需要在任务配置中显式启用：

```json
{
  "enable_genetic_optimization": true,
  "genetic_population_size": 12,
  "genetic_generations": 2,
  "genetic_max_simulations": 20
}
```

### Diversity Tracker (多样性追踪器)

防止系统陷入局部最优，鼓励探索：

```
功能:
1. 追踪已尝试的组合 (dataset, fields, operators, settings)
2. 计算多样性评分 (0-1, 越高越新颖)
3. 建议未充分探索的方向
4. 指纹去重防止重复
5. 主挖掘循环会将未充分探索的算子/方向注入下一轮 EvolutionStrategy
```

### External Knowledge (外部知识)

从外部来源自动导入高质量模式：

- **论坛同步**: 从 BRAIN 平台论坛提取高赞帖子中的 Alpha 模式
- **学术论文**: 预置 101 Formulaic Alphas 经典模式
- **模式验证**: 自动提取、验证、评分

### Metrics Tracker (指标追踪器)

全链路可观测性：

```
追踪层级:
- Session 级: 整体会话统计、知识库演进
- Round 级: 每轮 pass_rate、avg_sharpe、diversity_score
- Alpha 级: 单个 Alpha 的详细评估结果

输出:
- Debug 日志 (.cursor/debug.log)
- JSON 报告
- Logger 实时输出
```

---

## 🧪 Benchmark & Testing

### 综合测试套件 (推荐)

使用 `test_suite.py` 进行**真正的回归测试**，检测代码修改是否破坏功能：

```bash
# 完整测试 (单元 + 集成 + 回归 + 端到端)
python backend/tests/test_suite.py --all

# 仅单元测试 (快速)
python backend/tests/test_suite.py --unit

# 仅集成测试
python backend/tests/test_suite.py --integration

# 仅回归测试 (对比基准)
python backend/tests/test_suite.py --regression

# 保存当前结果为基准 (发布新版本时)
python backend/tests/test_suite.py --all --save-baseline
```

**测试输出示例:**

```
======================================================================
[TEST REPORT] Alpha Mining System Test Suite
======================================================================
Timestamp: 2026-01-29 15:57:30
Git Commit: 7075a16
Total: 18 | Passed: 18 | Failed: 0 | Rate: 100%

[UNIT] 7/7 passed
--------------------------------------------------
  [PASS] Alpha Syntax Validation - Accuracy: 83%
  [PASS] Threshold Calculation - 4/4 checks passed
  [PASS] Category Inference - Accuracy: 100%
  [PASS] Failure Classification - Accuracy: 83%
  [PASS] Mutation Operations - 8/8 checks passed
  [PASS] Diversity Scoring - 5/5 checks passed
  [PASS] Pattern Retrieval - 6/6 checks passed

[INTEGRATION] 3/3 passed
[REGRESSION] 5/5 passed
[E2E] 3/3 passed

[METRICS]
--------------------------------------------------
  syntax_validation_accuracy: 0.833
  category_inference_accuracy: 1.000 (baseline: 1.000, +0.000)
  failure_classification_accuracy: 0.833 (baseline: 0.833, +0.000)
  mutation_validity_rate: 1.000 (baseline: 1.000, +0.000)
  kb_total_entries: 59.000 (baseline: 59.000, +0.000)

======================================================================
[SUCCESS] All tests passed!
======================================================================
```

**关键特性:**
- ✅ **回归检测**: 自动对比基准，指标下降时报警
- ✅ **Git 集成**: 记录 commit hash，追踪问题来源
- ✅ **基准管理**: `baseline.json` 保存历史指标
- ✅ **分类明确**: 单元/集成/回归/端到端 四类测试

### 测试覆盖详情

| 类别 | 测试数 | 验证内容 |
|------|--------|---------|
| **Unit** | 7 | 语法验证、阈值计算、类别推断、失败分类、变异操作、多样性评分、模式检索 |
| **Integration** | 3 | Alpha 评估流程、RAG→生成流程、反馈循环集成 |
| **Regression** | 5 | 关键指标对比基准检测退化 |
| **E2E** | 3 | 数据库连接、知识库初始化、遗传优化循环 |

### 效果基准测试 (快速检查)

使用 `benchmark_test.py` 快速验证组件状态：

```bash
# 完整测试
python backend/benchmark_test.py --full

# 快速检查
python backend/benchmark_test.py --quick

# 初始化知识库
python backend/benchmark_test.py --seed
```

---

## 📈 Roadmap

### 已完成 ✅

- [x] **Phase 1**: 基础骨架 (Backend + Frontend + DB + SQLAlchemy 自动迁移)
- [x] **Phase 2**: Trace 可视化 + Mining Agent 核心
- [x] **Phase 3**: Brain 平台同步与集成 (Datasets, Operators, Fields, Alphas)
- [x] **Phase 4**: Celery 异步任务队列 (Mining, Sync, Feedback 定时任务)
- [x] **Phase 5**: 多区域支持 (USA, CHN, ASI, EUR 等)
- [x] **Phase 6**: LangGraph 工作流 + Evolution Strategy
- [x] **Phase 7**: Knowledge Base 知识库系统
- [x] **Phase 8**: 🆕 遗传优化器 (Genetic Optimizer) - 已可通过任务配置启用预算型优化
- [x] **Phase 9**: 🆕 多样性追踪器 (Diversity Tracker) - 已接入主挖掘循环和策略引导
- [x] **Phase 10**: 🆕 外部知识集成 (External Knowledge) - 已接入 Celery 定时导入，论坛同步保留 MCP 扩展入口
- [x] **Phase 11**: 🆕 指标追踪器 (Metrics Tracker) - 已接入主挖掘循环
- [x] **Phase 12**: 🆕 基准测试工具 (Benchmark Test) - 系统效果验证
- [x] **Phase 13**: 🆕 综合测试套件 (Test Suite) - 回归测试 + 自动化验证

### 进行中 🔄

- [ ] **Phase 14**: 高级分析仪表盘 (PnL 曲线, 区域对比, 知识库可视化)
- [x] **Phase 15**: Alpha 优化链 (Chain-of-Alpha) - 已接入局部表达式改写、参数 sweep 和可选遗传搜索

### 规划中 📋

- [ ] **Phase 16**: 多任务并行挖掘调度
- [ ] **Phase 17**: 强化学习策略优化 (RL-based Strategy)
- [ ] **Phase 18**: 生产环境部署优化 (监控, 日志, 告警)

---

## 📝 Changelog

### v2.1.1 (2026-05-29)

**挖掘链路修复:**

- 修复 news 类数据集字段蒸馏过度依赖 LLM 的问题，新增确定性机制分类与多机制 focused fields。
- 修复负 Sharpe 弱信号被直接丢弃的问题，支持 sign reversal candidate 进入优化链。
- 修复重复表达式进入 SELF_CORRECT 后产生等价改写/`+0` 空转的问题。
- 修复优化链单表达式调用 multi-simulation 的问题，严格保持 multi-sim-only。
- 改进 BRAIN/MCP simulation 错误报告，避免空错误字符串；MCP multi-create 增加硬超时和 fallback。
- 更新策略提示与演化逻辑，避免单字段、单骨架、硬凑比例策略塌缩。

**实测结论:**

- `news12` 在当前实测中主要受高换手、低 margin、risk neutralization 不达标限制；代码层面已能捕获反转和多样性，但数据机制本身仍需更低频、更平滑或换数据集机制探索。

### v2.1.0 (2025-01-29)

**新增功能:**

- ✨ **Genetic Optimizer** (`genetic_optimizer.py`)
  - 基于遗传算法的 Alpha 表达式优化
  - 支持算子替换、窗口参数变异、结构修改等 6 种变异类型
  - 多目标适应度函数 (Sharpe/Fitness/Turnover)
  - 自适应变异率和精英保留策略

- ✨ **Diversity Tracker** (`diversity_tracker.py`)
  - 追踪已尝试的 dataset/field/operator 组合
  - 计算多样性评分 (0-1)
  - 建议未充分探索的方向
  - 指纹去重防止重复探索

- ✨ **External Knowledge** (`external_knowledge.py`)
  - 论坛帖子模式提取与同步
  - 预置 101 Formulaic Alphas 经典模式
  - 自动模式验证和评分

- ✨ **Metrics Tracker** (`metrics_tracker.py`)
  - Session/Round/Alpha 三级指标追踪
  - Debug 日志输出到 `.cursor/debug.log`
  - JSON 报告生成

- ✨ **Benchmark Test** (`benchmark_test.py`)
  - 系统组件状态检查
  - 模拟测试效果评估
  - 改进建议自动生成
  - 支持 `--full`, `--quick`, `--seed` 三种模式

- ✨ **Comprehensive Test Suite** (`tests/test_suite.py`)
  - 18 个测试用例 (单元/集成/回归/端到端)
  - **真正的回归测试**: 自动对比基准检测代码退化
  - Git commit 追踪
  - 基准管理 (`baseline.json`)

- ✨ **Integration Tests** (`tests/test_integration.py`)
  - 9 个核心测试用例
  - 覆盖所有新增模块

**改进:**

- 🔧 `feedback_agent.py` - 增强失败分类，支持 10+ 种失败类型
- 🔧 `knowledge_seed.py` - 新增 101-Alpha 模式和区域优化配置
- 🔧 `rag_service.py` - 改进数据集类别推断逻辑

---

## 🤝 Contributing

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Alpha-GPT Paper](https://arxiv.org/abs/xxxx) - Human-AI Interaction Paradigm
- [RD-Agent](https://github.com/microsoft/RD-Agent) - CoSTEER Feedback Loop
- [WorldQuant BRAIN](https://platform.worldquantbrain.com) - Alpha Simulation Platform

---

<div align="center">

**Built with ❤️ for Quantitative Research**

</div>
