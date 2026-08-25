# ExpertForge 架构总览

> 更新日期：2026-08-25（对应 `main` @ `3165436`，gate-first critique 已合并）
>
> 本文由代码分析生成；模块行为以源码为准。相关设计文档见 `docs/superpowers/specs/`，专项调查见 `docs/superpowers/investigations/`。

ExpertForge 是一个**领域数据驱动**的 LLM 专家系统：切换专家领域只需更换一个数据目录（`domain/<name>/`），无需改代码（`tests/unit/test_domain_agnostic.py` 强制此约束）。核心链路是「分类路由 → 单次专家应答 或 编排应答」，外围配套评测框架与零侵入观测层。

## 1. 分层组件图

```mermaid
flowchart TB
    subgraph entry["入口层"]
        CLI["python -m agent<br/>agent_cli.py：REPL / --ask 单发"]
        EVALCLI["python -m agent.evaluation<br/>run | diff | baseline | compare"]
        OBSCLI["python -m agent.observability<br/>trace JSONL → HTML 报告"]
    end

    subgraph pipeline["应答流水线（agent/chat.py）"]
        CHAT["Chat.respond<br/>拒答 / 单策略 / 编排 三分支"]
        ROUTER["Router.route<br/>router.py"]
        CLS["ClassificationService.classify<br/>单次 LLM 调用：in_domain + intent + complexity"]
        MR["model_router.resolve_model<br/>simple→model_low，其余→model_high"]
    end

    subgraph answer["应答执行"]
        STRAT["Strategy.process × N<br/>strategy.py：system=expert_policy+prompts/&lt;sid&gt;.md"]
        ORCH["Orchestrator.run<br/>orchestrator.py：topology 分派"]
    end

    subgraph infra["LLM 基础设施"]
        LLM["LLMClient.chat_completion<br/>llm.py：json_mode/json_schema 协商、thinking 开关、max_tokens"]
        CAPS["ProviderCapabilities<br/>capabilities.py + negotiate.py"]
    end

    subgraph data["领域数据（domains are data）"]
        DOM["domain/software_engineering/<br/>domain.json · intents.yaml · intent_mapping.yaml<br/>complexity.yaml · expert_policy.md<br/>orchestration.yaml · prompts/*.md"]
        LOAD["config.py（AgentConfig）<br/>domain_config.py（DomainConfig，含校验 ConfigError）"]
    end

    CLI --> CHAT
    CHAT -->|"route=None 时"| ROUTER --> CLS
    ROUTER --> CHAT
    CHAT --> MR
    CHAT -->|"orchestrate=False"| STRAT
    CHAT -->|"orchestrate=True"| ORCH
    STRAT --> LLM
    ORCH --> LLM
    LLM --> CAPS
    LOAD --> DOM
    LOAD -.->|"构造"| CHAT
```

要点：

- **入口**：`agent_cli.main` 完成 `load_config → load_domain_config → get_api_key`，构建 `LLMClient` 后先经 `observability.install()` 包装再交给 REPL / 单发问答。
- **路由**（router.py）：一次分类调用产出三要素；`orchestrate = policy.enabled ∧ complexity ≥ min_complexity ∧ intent ∈ policy.intents`（全部来自 `orchestration.yaml`）。
- **模型分层**（model_router.py）：仅按 complexity 二分到 `model_low` / `model_high`。

## 2. 请求主流程

```mermaid
flowchart TD
    Q["用户问题"] --> C{"classify()"}
    C -->|"不在域内 / 未知 intent"| REJ["拒答文案<br/>domain.out_of_domain_reply"]
    C -->|"intent ∈ mapping"| S["intent_mapping.yaml → strategy_id"]
    S --> O{"orchestration.yaml 判定"}
    O -->|"complexity 低 / intent 未列入"| SINGLE["Strategy.process（1 次调用）"]
    O -->|"达标"| ORCH["Orchestrator.run"]
    ORCH --> T{"orchestration.yaml<br/>topology"}
    T -->|"map_reduce"| MAPR["_plan → 并行 _worker×N → _aggregate"]
    T -->|"critique"| CRIT["_draft（保留 thinking）"]
    MAPR --> EV1["evaluator 回路：<br/>judge 打分 → 低分 re-aggregate ×max_rounds"]
    CRIT --> GATE{"judge(draft)<br/>全维度 ≥ min_dimension_score？"}
    GATE -->|"通过（scorecard None 同样视为通过）"| EARLY["直接返回 draft<br/>成本 ≈ base + 1 次 judge"]
    GATE -->|"存在低分维度"| PASS["_critique_pass：<br/>plan_perspectives（按 intent 上限）<br/>→ 并行 critics（无 expert_policy context）<br/>→ consolidate → 预算化 revise"]
    PASS --> EV2["round ≥1 judge：<br/>低分走 judge-feedback revise"]
    SINGLE --> ANS["答案 + 追加会话历史"]
    EV1 --> ANS
    EARLY --> ANS
    EV2 --> ANS
```

两种编排拓扑（orchestrator.py）：

| | map_reduce | critique（当前默认） |
|---|---|---|
| 作者 | 多 worker 各写一段 → aggregator 合成 | 单作者 draft，critic 只报缺陷不改写 |
| judge 位置 | 尾部回路（re-aggregate 改进） | **门控在前**（gate-first）；失败时进入管线，成功即早退 |
| 成本特征 | 多作者易矛盾（RC3，已弃用主线） | 通过路径 ≈ base + 1 次 judge；失败路径才付全套 |
| 失败降级 | planner 全败 → `_direct_answer` | critic/revise LLMError → 逐级回落 draft |

成本控制三件套（P2 措施 A/B/C，均配置化于 `orchestration.yaml` 的 `critique:` 块）：critics 不挂载生成端 context；视角数按 intent 设上限；revise 输出预算 = `clamp(ceil(draft_out × ratio), min, max)`。

## 3. 领域数据契约

```mermaid
flowchart LR
    subgraph D["domain/&lt;name&gt;/（纯数据）"]
        DJ["domain.json<br/>name/description/out_of_domain_reply"]
        IY["intents.yaml<br/>id+描述+正反例+边界"]
        IM["intent_mapping.yaml<br/>intent→strategy"]
        CY["complexity.yaml（可选）<br/>simple/medium/complex 定义"]
        EP["expert_policy.md（可选）"]
        OY["orchestration.yaml<br/>enabled/min_complexity/intents/max_workers<br/>topology/evaluator/critique"]
        PR["prompts/*.md<br/>每 strategy 一个文件"]
    end
    DC["agent/domain_config.py<br/>逐文件解析 + 校验，非法即 ConfigError"]
    D --> DC --> DM["DomainConfig（不可变数据类）"]
    DM --> RT["Router"]
    DM --> SG["Strategy 注册表"]
    DM --> OC["Orchestrator"]
```

新增领域 = 复制并修改这个目录，代码零改动。`CritiquePolicy`（视角上限、revise 预算）缺省即生效（default_max_perspectives=3 等），显式块可覆盖。

## 4. 评测框架（agent/evaluation/）

```mermaid
flowchart TB
    DS["datasets/&lt;domain&gt;/*.yaml<br/>case: question + expected(intent/complexity/strategy) + tier"] --> RUN
    subgraph RUN["__main__.py run（tier=classification/routing/full_expert/all）"]
        RC["RecordingClient<br/>包装任意 chat_completion，记账 tokens/latency"]
        JG["Judge.score<br/>6 维 scorecard（temp=0，LLMError→None 静默）"]
    end
    RUN --> R1["results/*.json + report 汇总"]
    CMP["compare（PR #21 引入）"] -->|"同一 case 双跑"| B["baseline：Strategy.process"]
    CMP --> O2["orchestrated：Orchestrator.run"]
    B & O2 --> RC
    B & O2 --> JG
    CMP --> OUT["quality_gain / token_increase_pct / cost_efficiency<br/>按 intent、complexity 聚合"]
    DF["diff / baseline 子命令"] --> HIST["历史结果对比"]
```

注意：`AGENT_API_KEY` 与 `AGENT_JUDGE_API_KEY` 相互独立；judge 缺 key 时评测命令直接 `Config error` 退出。

## 5. 观测层（agent/observability/）

```mermaid
flowchart LR
    INST["install(client, config, domain)"]
    INST -->|"包装 LLM 实例"| TC["TracedLLMClient：每次调用记 trace_llm_call"]
    INST -->|"monkey-patch ~14 个业务方法"| PATCH["patch.py<br/>Chat.respond / Router.route / Strategy.process<br/>Orchestrator._plan/_worker/_draft/_plan_perspectives/_critic/_revise/_evaluate ..."]
    PATCH --> STORE["TraceStore → JSONL（obs 目录）"]
    STORE --> REP["observability report → HTML（stage 分组时间线）"]
    DEC["decision 事件（planner tasks / critic roles / evaluator scorecard）<br/>仅在活跃 trace_span 上下文内记录"] --> STORE
```

设计铁律：观测永不破坏业务——所有写失败降级为 `warnings.warn`，未安装时全部为透明直通。已知约束：绕过 `Chat.respond` 直接调 `Orchestrator.run` 的代码（如 `evaluation/compare.py`）必须自建 `trace_span()`，否则 worker 决策事件静默缺失。

## 6. 关键不变量（改动前必读）

1. **Domains are data**：任何新领域/意图/策略调整只动 `domain/` 目录。
2. **观测与日志零侵入**：业务方法签名变更时必须同步 patch.py 的显式签名 wrapper。
3. **judge 失败语义统一为“视为通过”**：`Judge.score` 返回 None（LLMError/解析失败）在 gate 与尾部回路中都放行当前答案。
4. **map_reduce 为兼容保留路径**：主线质量修复都在 critique 拓扑上；两者共享 `_evaluate_loop` / worker_pool / 预算规则。
5. **单次抽样方差大**：thinking + temp 默认下补全长度可差数倍（见 [2026-08-25 方差调查](superpowers/investigations/2026-08-25-draft-output-variance-investigation.md)），per-case 单次 compare 百分比不具备结论力。
