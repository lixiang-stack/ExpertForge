
## 1. 问题
    uv run python -m agent.evaluation compare --ids se-129

用于评估orchestration是否能够提高质量，结果发现启用orchestration并没有提升结果质量，部分case对比base甚至质量有所下降。

### 结果摘要
```
Compare: baseline vs orchestrated (1 cases)
------------------------------------------------------------------------------------------
Case         Intent               Base Q  Orch Q  Gain    Gain%   Base Tok  Orch Tok  +Tok    Tok%    Cost Eff  
------------------------------------------------------------------------------------------
se-129       architecture_design  5.00    4.50    -0.50   -10.00  10433     29808     19375   185.71  -0.000026 
------------------------------------------------------------------------------------------
Overall                           5.00    4.50    -0.50   -10.00  10433     29808     19375   185.71  -0.000026 

By intent:
  architecture_design  -0.50    gain / +19375  tokens / -0.000026 eff (1 cases)

By complexity:
  complex      -0.50    gain / +19375  tokens / -0.000026 eff (1 cases)
```
Base为未启用orchestration，Orch为启用orchestration。Orch对比Base token多消耗185.71%，但是质量却下降了10%。

### 结果文件
```
{
  "kind": "compare",
  "label": "compare",
  "model": "deepseek-v4-flash",
  "judge_model": "deepseek-v4-pro",
  "domain": "Software Engineering",
  "n_cases": 1,
  "n_compared": 1,
  "cases": [
    {
      "id": "se-129",
      "intent": "architecture_design",
      "complexity": "complex",
      "strategy": "planning",
      "baseline": {
        "scorecard": {
          "correctness": 5,
          "relevance": 5,
          "completeness": 5,
          "technical_depth": 5,
          "practical_usefulness": 5,
          "hallucination": 5
        },
        "quality": 5.0,
        "llm_calls": 1,
        "in_tokens": 489,
        "out_tokens": 9944,
        "total_tokens": 10433,
        "cache_tokens": 384,
        "latency_ms": 97490.8,
        "error": null
      },
      "orchestrated": {
        "scorecard": {
          "correctness": 4,
          "relevance": 5,
          "completeness": 5,
          "technical_depth": 4,
          "practical_usefulness": 5,
          "hallucination": 4
        },
        "quality": 4.5,
        "llm_calls": 6,
        "in_tokens": 16117,
        "out_tokens": 13691,
        "total_tokens": 29808,
        "cache_tokens": 2048,
        "latency_ms": 91273.9,
        "error": null
      },
      "quality_gain": -0.5,
      "quality_gain_pct": -10.0,
      "additional_tokens": 19375,
      "token_increase_pct": 185.71,
      "cost_efficiency": -2.6e-05
    }
  ],
  "aggregates": {
    "overall": {
      "mean_quality_gain": -0.5,
      "sum_additional_tokens": 19375,
      "cost_efficiency": -2.6e-05,
      "n_compared": 1
    },
    "by_intent": {
      "architecture_design": {
        "mean_quality_gain": -0.5,
        "sum_additional_tokens": 19375,
        "cost_efficiency": -2.6e-05,
        "n_compared": 1
      }
    },
    "by_complexity": {
      "complex": {
        "mean_quality_gain": -0.5,
        "sum_additional_tokens": 19375,
        "cost_efficiency": -2.6e-05,
        "n_compared": 1
      }
    }
  }
}
```
### 执行过程的可观测性
通过可观测性observability，执行过程如下：
* Base
```
in=489 out=9944 total=10433 time=97.5s
strategy.planning
model=deepseek-v4-flash in=489 out=9944 time=97.5s [ok]
```
* Orch
```
in=16117 out=13691 total=29808 time=91.3s
orchestration.planner
model=deepseek-v4-flash in=560 out=256 time=3.3s [ok]
task1: 支付系统总体架构与一致性设计
task2: 多区域部署与灾备方案设计
task3: 支付系统合规、安全与成本优化设计
orchestration.worker
worker1: 支付系统总体架构与一致性设计
model=deepseek-v4-flash in=492 out=2630 time=17.4s [ok]
worker2: 多区域部署与灾备方案设计
model=deepseek-v4-flash in=480 out=2983 time=22.7s [ok]
worker3: 支付系统合规、安全与成本优化设计
model=deepseek-v4-flash in=474 out=4001 time=27.7s [ok]
orchestration.aggregate
model=deepseek-v4-flash in=10123 out=3766 time=18.1s [ok]
{
  "evaluated": true,
  "evaluator_model": "deepseek-v4-pro"
}
orchestration.evaluator
model=deepseek-v4-pro in=3988 out=55 time=2.0s [ok]
{
  "scorecard": {
    "correctness": 4,
    "relevance": 5,
    "completeness": 5,
    "technical_depth": 4,
    "practical_usefulness": 4,
    "hallucination": 4
  },
  "passed": true
}
```

# 2. 根因分析

## 2.1 关键事实核查：Base 与 Orch 的 token 口径不可直接比较

对结果文件中两份答案实测字符数：

| | 可见答案字符数 | out tokens | 说明 |
|---|---|---|---|
| Base | **4809** | 9944 | thinking 开启，`completion_tokens` 包含隐藏的 reasoning tokens（按 Orch 的字符/token 比折算，可见内容仅约 2000 tokens，推理占约 80%） |
| Orch | **9016** | 13691 | 全部调用 `disable_thinking=True`，tokens 均为可见内容 |

结论：**Orch 最终答案比 Base 更长**。Orch 的质量下降不是"聚合压缩导致深度塌缩"——Base 的 9944 out tokens 中大部分（约八成）是不可见的推理过程。同时这说明两边的 token 统计口径不一致（RC1），且 Base 独享了扩展推理的质量加成。

## 2.2 核心根因 RC3：map-reduce 分解与耦合型任务存在模式错配

架构设计是**耦合优化问题**：部署模式、一致性模型、事务方案、数据分布互相决定。当前拓扑把强耦合维度拆给并行 worker，各自隐式做全局决策，再由聚合器拼接——这是没有可行合并算子的 divide-and-conquer，矛盾是结构性的。

Orch 答案中的实证（四处内部矛盾）：

| # | 矛盾 | 位置 |
|---|------|------|
| 1 | 部署模式自相矛盾：决策表与拓扑图为"两地三中心（**主备**）"、写流量强制主区域；路线图阶段二却写"同城**双活** + 异地灾备" | 决策表/拓扑 vs 路线图 |
| 2 | RPO 分层混乱：需求表称跨区域 RPO≤5分钟，复制策略表却给跨区域主备 ≤1秒（≤5分钟实为合规区域）；总结把分层 RPO 压平成单一"RPO≤1s" | 需求表 vs 复制策略表 vs 总结 |
| 3 | 事务方案混淆：正文推荐"TCC/Saga/本地消息表三者分工"，总结却合并为"采用TCC + 本地消息表，确保资金安全"（TCC 强一致与本地消息表最终一致被混为一谈） | 正文 vs 总结 |
| 4 | 架构风格矛盾：通篇是 9 服务微服务拆分，总结亦称"微服务拆分"；路线图阶段一却写"建议采用模块化单体起步，避免过早微服务化" | 架构章/总结 vs 路线图 |

Judge 扣分维度与此精确对应：`correctness` 5→4、`hallucination` 5→4、`technical_depth` 5→4 全是**连贯性失败**；`completeness` 保持 5——覆盖没问题，一致性崩了。

另有认识论纪律丢失：Base 开头有明确的"我下面的设计假设：…"（符合 expert_policy 的 "State important assumptions explicitly"）；Orch 把凭空发明的 KPI 表（峰值TPS 10万+、对账差异率<0.01%）当作需求事实呈现而未标注假设——这是 `hallucination` 扣分的直接来源。

## 2.3 次要根因 RC1：thinking 模式不一致（混淆变量）

- Base 走 `Strategy.process`，调用未传 `disable_thinking` → thinking 开启；
- Orch 全部 6 个调用传 `disable_thinking=True`（orchestrator.py），且 provider `supports_thinking_toggle: true` 生效。

即 Base 有扩展推理加持而 Orch 没有，既污染质量对比，也使 token 对比口径不一。

## 2.4 结论

单次调用在本 case 上并无瓶颈（Base 可见输出仅 4809 字符，远未触及上限，且已得 5.0）。问题不在 worker 协调不足，而在"**多作者生成 + 拼接**"这一拓扑本身：对耦合综合型任务，拆分生成立即产生全局决策矛盾，聚合器无法调和它没见证过的隐式决策。编排对质量的价值应来自**独立验证**，而非**碎片化生成**。

# 3. 优化方向

## 3.1 新拓扑 `critique`：单作者 + 并行评审 + 条件修订

```
question ──> draft        单次完整生成（与 baseline 完全同构：同一 system prompt、thinking 开启）
         ──> planner      变体：产出 2-4 个「评审视角」而非子任务（复用现有 json_schema 机制）
         ──> critics ×N   并行（复用 run_workers）：各自读 question+draft，
                          只报告缺陷，禁止重写。输出 JSON issues[{severity,description,suggestion}]
         ──> consolidate  纯代码合并 findings（无 LLM 调用）
         ──> revise       若有任何 issue：单作者调用（draft+findings → 终稿，要求显式标注假设）
                          若无 issue 或评审环节失败：直接返回 draft
         ──> evaluator    loop 保持现状不动
```

关键性质：

- **RC3 从构造上消失**：生成永远只有一个作者，不存在多作者拼接；critics 只增信息不改结构。
- **质量下限 = base**：draft 与 baseline 同构，修订只做定向修复。
- **RC1 同时修复**：draft 不传 `disable_thinking`（与 `Strategy.process` 一致）；planner/critics/revise 关闭 thinking 控成本。

诚实的代价预估：~38k tokens / ~155s（当前 orch 29.8k/91s，base 10.4k/97s）。用 token 买质量；验证通过后再做裁剪优化（见 §4 P2）。

## 3.2 配置（domains are data 原则）

```yaml
# orchestration.yaml 新增
topology: critique          # critique | map_reduce（缺省 map_reduce，向后兼容）
```

- **保留 map_reduce 路径**：配置切换即可回退，也为 §5 验证提供同条件 A/B 对比。
- 视角分解仍由 planner LLM 自适应产出（保持既有"领域自适应规划"设计），不改为静态配置。
- `expert_policy.md`、prompts、judge 均不需要改动。

## 3.3 可观测性集成

`patch.py` 新增 phase 映射（沿用现有命名风格）：

| 方法 | phase |
|------|-------|
| `Orchestrator._draft` | `orchestration.draft` |
| `Orchestrator._plan_perspectives` | `orchestration.planner`（复用，报告格式不变） |
| `Orchestrator._critic` | `orchestration.critic`（复用 `_wrap_worker` 模式记录 task/role 决策事件） |
| `Orchestrator._revise` | `orchestration.reviser` |

map_reduce 路径的 6 个既有映射不动。`run_workers` 已复制 contextvars，span 自然传播到 critic 线程，无需新机制。

## 3.4 错误处理降级链

原则：评审是增值环节，任何失败都退回 draft，绝不劣于 baseline。

```
draft 失败             → 异常上抛（由 chat 层处理，与现状一致）
perspectives 解析失败  → 返回 draft
单个 critic 失败       → findings 记为空，其余继续（run_workers 已保证）
全部 critic 失败       → 返回 draft
revise LLMError       → 返回 draft
critic JSON 解析失败   → 该视角 issues=[]（不中断其它视角）
```

# 4. 优化优先级

1. **P0 — critique 拓扑 + thinking 对齐**：一次改动同时消除 RC3 与 RC1。
2. **P1 — 验证**：se-129 复测 + 扩大 full_expert 样本（见 §5）。
3. **P2 — 成本裁剪**（仅当 P1 质量达标但成本不可接受时）：critics 去掉 expert_policy 上下文、按 intent 配置视角数、revise 输出预算控制。

# 5. 优化效果验证

- **命令**：`uv run python -m agent.evaluation compare --ids se-129`（topology=critique vs baseline），再跑全量 full_expert cases。
- **质量标准**：orch Q ≥ base Q（gain ≥ 0）；人工核对答案不再出现 §2.2 列出的四类矛盾（部署模式 / RPO 分层 / 事务方案 / 架构风格）。
- **成本预期**：token 增幅 ≤300% 视为可接受，在报告中如实呈现，裁剪留给 P2。
- **对照实验**：同一 case 分别以 `topology: map_reduce` 与 `topology: critique` 各跑一次，确认新拓扑优于旧拓扑。

# 6. 测试

单元测试（hermetic，fake client）：

- `test_orchestrator.py` 新增：
  - critique 拓扑全流程（draft → perspectives → critics → revise）；
  - 无 issue 时跳过 revise；
  - 各降级路径（perspectives 解析失败 / critic 全败 / revise 异常 → 返回 draft）；
  - draft 调用不带 `disable_thinking`，critic/planner/revise 调用带。
- `test_config.py`：`topology` 字段校验（非法值报 ConfigError，缺省 map_reduce）。
- `test_domain_agnostic.py`：确认新增配置项不破坏"加领域零代码"。
- `test_observability_patch.py`：新 phase 映射生效。
- 回归：现有 map_reduce 测试全部保持通过；`uv run pytest tests/unit -q`。
