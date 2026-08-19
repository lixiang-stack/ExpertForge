# Layered Benchmark & Layered Evaluation Pipeline — Design Spec

Date: 2026-08-19
Status: Approved (brainstorming)

Reference: `draft_v3.md` §2（分层 Benchmark）、§5（分层 Evaluation Pipeline）、§6（降低 Evaluation 成本）

## Problem

draft_v3 的下一阶段目标是建立"低成本、可持续迭代的 Expert Evaluation 体系"。本次设计覆盖其中第一块：

1. **分层 Benchmark**（§2）：按执行成本分三层——Classification（100~500，程序化评估）、Routing（30~50，路由决策校验）、Full Expert（10~20，完整 Agent Pipeline + Judge），并采用失败驱动增长。
2. **分层 Evaluation Pipeline**（§5）：不同代码修改只运行必要的 Evaluation，辅以 Smoke Test / Full Regression 两级执行。
3. **降低成本**（§6）：程序化评估优先，LLM Judge 只用于难以程序化评价的内容，Reference 长期复用。

现有实现已具备：suite 数据集（8 个 YAML）、router 单次调用产出 domain/intent/complexity、strategy/orchestrate/model 指标、LLM-as-judge、cost/latency 指标、diff/baseline、`--suite`/`--max-per-suite`/`--skip-quality` CLI。本次设计在此基础上引入 **tier 概念**并对 CLI 做减法，使分层选择成为第一等公民。

## Design Decisions

| # | Question | Decision |
|---|---|---|
| 1 | tier 与现有 suite 的关系 | 每个 case 增加 `tier` 字段（classification / routing / full_expert），suite 内允许混层。 |
| 2 | suite 是否保留 | **不保留为模型/CLI 概念**。YAML 文件按 **intent 分组 + boundary.yaml** 仅作存储布局（与内部 strategy / orchestration 概念解耦），模型层视为扁平 case 池；删除 `--suite` 与 `metrics_by_suite`。 |
| 3 | CLI 深度选择 | `--tier <tier> [<tier>...]`；`--tier all` 为完整回归别名；**无 `--tier` 时默认只跑 `smoke: true` 的 case**（防 token 超耗）。 |
| 4 | 冗余 flag 处置 | 删除 `--max-per-suite`、`--skip-quality`、`--smoke`。默认即 smoke，无需显式开关；tier 体系取代 skip-quality 的省钱作用。 |
| 5 | `answer_quality` 字段 | 删除，由 tier 推导：full_expert ⇒ 完整 pipeline + judge；其余 ⇒ 仅 router 调用。 |
| 6 | Judge 范围 | 只对 full_expert tier 运行（draft §2.4：只评价最有价值的复杂 case）。 |
| 7 | Smoke 机制 | `smoke: true` 标记约 5 个跨层代表 case，作为默认 run 的选择集合。 |
| 8 | §6 成本控制 | 现有框架 + tier 设计已满足；仅需在 case schema 预留 `required_points` / `expert_expectations` 字段（本次不实现其评估，供后续子项目使用）。 |

## 1. Data Model

### 1.1 Case schema

```yaml
cases:
  - id: se-126
    tier: full_expert          # required: classification | routing | full_expert
    smoke: true                # optional: 加入默认 smoke 集合
    question: "..."
    expected: {domain, intent, complexity, strategy, orchestrate}
    reference: "..."           # optional（已有，长期复用，不重新生成）
    required_points: [...]      # optional（仅预留字段，本次不评估）
    expert_expectations: [...]  # optional（仅预留字段，本次不评估）
```

变更点：

- `tier` 必填，仅允许三个值；同一 YAML 文件内允许混层。
- `answer_quality` 字段**删除**，执行深度由 `tier` 推导。
- `smoke: true` 可选，全局约 5 个，必须覆盖三层（classification / routing / full_expert 各至少 1 个）。

### 1.2 加载模型

- 保留按目录扫描 `*.yaml` 的加载方式；每个文件仍是一个 `Suite`（存储单元），但 **选择与统计不再以 suite 为单位**。
- **文件按 intent 分组 + boundary.yaml 组织**（faq.yaml、concept_explain.yaml、troubleshooting.yaml、boundary.yaml…）。intent 是问题固有属性；strategy / orchestrate / tier 都是推导结果，仅作为每个 case 的字段，不参与文件命名。
- 加载后展平为 case 池，`tier` 是唯一深度选择轴；文件边界对运行语义无影响。
- 校验：tier 取值合法；`smoke: true` 的 case 的 expected 字段完整；full_expert case 的 expected 完整。

### 1.3 现有 case 迁移规则

**文件重组**：8 个策略/阶段命名文件（direct/teaching/debugging/analysis/code_snippet/classification/routing/orchestration）→ 按 intent 命名的 11 个文件 + boundary.yaml。

| intent / 角色 | 文件 | 现有 case |
|---|---|---|
| OOD 边界 | `boundary.yaml` | se-110 |
| faq | `faq.yaml` | se-001, se-003, se-120 |
| concept_explain | `concept_explain.yaml` | se-010, se-121, se-127 |
| tutorial | `tutorial.yaml` | se-020, se-122 |
| learning_guide | `learning_guide.yaml` | se-030, se-123 |
| summarization | `summarization.yaml` | se-040 |
| troubleshooting | `troubleshooting.yaml` | se-050, se-051, se-052, se-053, se-054, se-140 |
| performance_analysis | `performance_analysis.yaml` | se-060, se-062, se-124 |
| architecture_design | `architecture_design.yaml` | se-070, se-071, se-082, se-126, se-128 |
| comparison | `comparison.yaml` | se-081, se-125 |
| code_review | `code_review.yaml` | se-090 |
| generate_code | `generate_code.yaml` | se-100, se-101, se-102, se-103 |

**tier 赋值**（逐 case 标注，与文件组织无关）：

- `orchestrate: true` 的 case（se-052, se-071, se-102, se-126, se-128）→ `full_expert`
- 映射到 analysis/debugging 策略的 intent（troubleshooting / performance_analysis / architecture_design / comparison / code_review）→ `routing`（验证 strategy 选择）
- 其余 intent（faq / concept_explain / tutorial / learning_guide / summarization / generate_code）→ `classification`
- OOD 边界 → `classification`

迁移后 judge 只覆盖 full_expert tier（10~20 个）。原先按策略 suite 测"每个策略的答案质量"的做法被路由层（strategy_accuracy + per-strategy 分解）取代，答案质量只对最高价值的复杂 case 测量。这是对现有评价哲学的有意收缩，与 draft §2.4 一致。

## 2. Execution Semantics

### 2.1 tier → 执行深度

| tier | 执行 | 测量 | 每 case 成本 |
|---|---|---|---|
| `classification` | router 一次调用 | domain / intent / complexity（程序化比对） | 1 次 LLM |
| `routing` | router 一次调用 | + strategy / orchestrate / model（程序化比对） | 1 次 LLM |
| `full_expert` | 完整 pipeline（Planner→Workers→Aggregator→Evaluator）+ judge | + 答案质量（LLM judge） | 多次 LLM |

关键事实：classification 与 routing **执行相同**（router 单次调用产出全部决策），差异只在测量哪些字段与计入哪些指标。真正的执行分层只有"cheap（router-only）vs 昂贵（full pipeline）"两档。

### 2.2 选择语义

- 无 `--tier` → 只跑 `smoke: true` 的 case（默认快速验证）。
- `--tier <tier> [<tier>...]` → 跑所有处于所选 tier 的 case（跨文件）。
- `--tier all` → 等价于 `--tier classification routing full_expert`，完整回归。

### 2.3 模型路由 / 编排决策

strategy 由 intent 映射、orchestrate 由 policy 决定、model 由 router 决定——均在 router 单次调用内完成，不产生额外 LLM 调用。

## 3. CLI

```
# 快速验证（默认）：~5 个 smoke case，跨三层
python -m agent.evaluation run

# 分层执行
python -m agent.evaluation run --tier classification            # 仅分类，cheap
python -m agent.evaluation run --tier classification routing    # 分类 + 路由
python -m agent.evaluation run --tier full_expert               # 仅复杂专家 case
python -m agent.evaluation run --tier all                       # 完整回归

# 通用项不变
python -m agent.evaluation run --label my-run --results-dir out/ --config path.json
python -m agent.evaluation diff a.json b.json
python -m agent.evaluation baseline result.json
```

**删除的 flag**：`--max-per-suite`、`--skip-quality`、`--smoke`

**语义变化**：无 `--tier` 时默认只跑 smoke（原行为是跑全部，属破坏性变更，README 需更新）。

## 4. Metrics & Reporting

### 4.1 metrics 变更

- 删除 `metrics_by_suite`。
- 新增 `metrics_by_tier`：classification / routing / full_expert 各自的指标聚合。
- 新增 per-strategy 分解：`strategy_accuracy` 按 expected.strategy 分组（替代旧 suite 的策略维度统计）。
- 保留：domain / intent / complexity / strategy / orchestration / model 准确率、per-intent、per-complexity 分解、answer_quality、cost by path。

### 4.2 CaseResult 与报表

- `CaseResult` 增加 `tier` 字段，序列化进结果 JSON。
- 报表增加 `failed_cases` 段：列出每个失败 case（分类/路由不匹配、judge 低于阈值等）+ 所属 tier + 失败原因。

**目的：失败驱动增长闭环。** 每次 run 结束，直接看 failed_cases 就知道该往哪层补 boundary case（如 troubleshooting↔performance 混淆就补 classification 层 5~10 个），无需手工翻原始数据。

## 5. Cost Control（draft §6）

| §6 条目 | 状态 |
|---|---|
| 6.1 程序化 Evaluation 优先（分类直接比对） | 已实现（metrics 直接比较 expected vs actual） |
| 6.2 Required Points 程序化检查 | 归入后续 Expert Behavior 子项目（本次仅预留字段） |
| 6.3 LLM Judge 只用于难程序化内容 | tier 天然保证：judge 只跑 full_expert |
| 6.4 Reference 复用（不重新生成） | reference 已存于 case 文件，长期复用 |

## 6. Dataset Migration

- 将现有 8 个按策略/阶段命名的 YAML 重组为按 intent 命名的 11 个文件 + `boundary.yaml`（见 §1.3 映射表），对 ~33 个 case 补充 `tier` 字段，删除 `answer_quality`。
- 挑选约 5 个 case 标记 `smoke: true`（覆盖三层）。
- 迁移后运行 `run` 默认跑 smoke；`--tier all` 可验证全量迁移无损。

## Testing

- 数据集：tier 取值校验、smoke 标记校验（须覆盖三层）、扁平池加载（忽略文件边界）。
- runner：tier 选择、默认 smoke、judge 仅 full_expert。
- metrics：metrics_by_tier、per-strategy 分解。
- CLI：`--tier` / `--tier all` / 默认 smoke / 被删 flag 的删除。
- 迁移：现有全部 case 补 tier 后校验通过。
- 回归：`uv run pytest -q` 全绿。

## Out of Scope（后续子项目）

- Expert Behavior 评估（Context Awareness / Assumption Awareness / Trade-off Quality / Evidence Reasoning / Recommendation Quality / Uncertainty Calibration）。
- Required Points 覆盖率评估。
- 单 Agent vs 编排 ROI 对照实验。
- 混淆矩阵与分类优化。
- Clarification 能力。
- 多轮上下文分类。
- 工具 / Evidence 接口。

## Acceptance Criteria

1. `python -m agent.evaluation run`（无参数）只跑 smoke 集合，成本可控（≤ 约 10 次 LLM 调用）。
2. `--tier classification` 只跑分类层 case，每 case 仅 1 次 router 调用，无 judge。
3. `--tier all` 跑全量，metrics 含 `metrics_by_tier` 与 per-strategy 分解。
4. 报表输出 `failed_cases`（含 tier 与原因），可直接指导补 case。
5. 现有 case 迁移后所有测试通过；README 中 CLI 用法更新。