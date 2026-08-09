# ExpertForge 迭代设计：分类体系与执行策略（Strategy Routing）

日期：2026-08-07
状态：已确认
相关草案：`draft.md`

## 背景与目标

现有 ExpertForge 只做 `Domain Classification` → 直接回答。本次迭代按 `draft.md` 引入分类体系与执行策略：将"业务类别"驱动的处理方式改为"执行策略（Execution Strategy）"驱动，使框架从单一领域 Agent 演进为**可配置的领域专家 AI Agent Framework**。

本次迭代范围采用**领域内、仅策略**策略：

- 实现 `Intent → Strategy` 映射 + 五个简单策略处理器。
- **重型 Orchestrator（Planner / Worker / Aggregator / Evaluator / Optimizer）全部延后**，仅以占位提示形式保留入口。

## 范围

**In scope（本次迭代）**

- 领域配置目录化（yaml + markdown prompts）。
- Intent 分类与 Complexity 分类（分两次 LLM 调用）。
- `intent → strategy` 确定性映射，配置化。
- 五个策略处理器：`direct`、`teaching`、`debugging`、`analysis`、`code_snippet`。
- Chat Interface（会话上下文、连续多轮）与 Single-shot Execution。
- `--ask` 单次问答入口与交互式 REPL。
- 模型路由：`classifier_model` + 每 strategy 可选 `model`。
- 一次性生成（非流式）。
- 分类调用采用单次 structured output（`response_format=json_object`），成功后不再重试。

**Out of scope（延后）**

- Orchestrator / Planner / Workers / Aggregator / Evaluator / Optimizer。
- complex 任务的真实多 worker 执行。
- Clarification 前置阶段（整体移除，不做澄清）。

## 设计原则

1. 处理器与**执行策略**绑定，而非业务类别。
2. 分类拆为 `Domain → Intent → Complexity`（`intent→strategy` 映射确定性）。
3. 所有映射与 prompt 模板配置化，域解耦。
4. 只有复杂 Strategy 才需要 Orchestrator；简单问题直接回答。
5. Single-shot Execution，处理器返回**完整字符串**（一次性生成，不做流式中间切片）。

## 架构概览

```
Question
   ├─ Domain Classifier → out-of-domain 拒绝
   ├─ Intent Classifier
   ├─ Complexity Classifier
   ├─ RouteResult { intent, complexity, strategy }
   │      │
   │      └─ strategy
   │             ├─ complex_unsupported → 占位提示
   │             └─ strategy processor → 一次性生成 → final answer
```

## 模块设计

### 领域配置目录结构与加载

`config.json` 只保留连接配置与指向领域配置目录的 `domain_dir`；领域内容拆分到独立目录。

```
config.json                     # base_url, model, classifier_model, domain_dir
domain/software_engineering/
  domain.json                   # { name, description, out_of_domain_reply }
  intents.yaml                  # intent 定义列表（id + description）
  intent_mapping.yaml           # intent -> strategy
  strategies.yaml               # 每个 strategy 元数据（id, model 可选, complexity_gate）
  prompts/
    direct.md
    teaching.md
    debugging.md
    analysis.md
    code_snippet.md
    unsupported_complex.md
```

- `config.py` 扩展：`AgentConfig` 增加 `domain_dir`；新增 `DomainConfig`（从 `domain_dir` 加载 intents / strategies / intent_mapping / prompts）。
- 缺文件或格式错误抛 `ConfigError`。
- 新增 `pyyaml` 依赖。

### 分类与路由

- **IntentClassifier**：`classify_intent(question, intents)` → `intent_id`。复用现有 `classifier.py` 的 LLM + JSON 解析模式。
- **ComplexityClassifier**：`classify_complexity(question, domain)` → `simple|medium|complex`。
- `Router.route(question)` → `RouteResult{ intent, complexity, strategy }`：
  - Domain 判拒；intent/complexity 各一次调用。
  - 映射查表 `intent→strategy`（确定性）。
  - complexity gate：若 strategy ∈ {teaching, debugging, analysis, code_snippet} 且 `complexity == complex` → 归为 `complex_unsupported`（占位，不调 LLM 生成实答）。
- 分类 LLM 调用采用**单次 structured output**：`response_format={"type": "json_object"}`，解析失败不再重试，直接落入默认值（reject / "" / medium）。

### 策略处理器

`processors/`

- `base.py`：`Processor` 基类（加载 prompt → 构建消息 → 一次性 `chat_completion` → 返回 `str`）。
- `direct.py`、`teaching.py`、`debugging.py`、`analysis.py`、`code_snippet.py`：各自从 `prompts/<id>.md` + 结构指令组装 system prompt，`process(question, context) -> str`。
- `registry.py`：strategy id → 处理器实例映射，由配置显式绑定（非反射自动发现）。
- 处理器**不做多轮、不管理历史、不循环追问**；会话上下文由 Chat Interface 传入。

### Chat Interface 与 Single-shot Execution

- `agent/chat.py` — `Chat` 引擎，持有 `AgentConfig`/`domain`/`router`/`processors` + `session_history`。
- `respond(question) -> str`：单问题一次推理一次生成，返回完整字符串，追加到 history。
- complex_unsupported → 输出占位提示。

### CLI

- 保留 `repl.py` 交互式 REPL，内部改走 `Chat` 引擎。
- 新增 `--ask "问题"` 单次入口：Chat 生成一次回答后退出。
- 输出整块回答（非流式）。

## 数据流

```
输入 question
→ Chat.respond
→ Router.route
→ processor.process(question, context) -> str
→ 追加历史，返回完整 answer
```

## 错误处理

- 分类结果无法解析 → 降级回退（reject / intent 空 / medium）。
- `response_format=json_object` 不被 API 支持 → `LLMError` 直接透传（fail-loud）。
- LLM 调用失败 → `LLMError` 透传给交互层显示。
- 配置缺字段/坏 yaml/json → `ConfigError`。

## 测试

- 配置加载与路由单测（domain_dir 拆载、yaml/markdown 解析、坏格式报错、intent→strategy 映射、complex→complex_unsupported）。
- 分类单测（intent/complexity prompt 构造与 JSON 解析/降级、单次调用）。
- 策略处理器单测（各 prompt 结构 + 一次性调用 + 返回完整串）。
- CLI 集成测试（`main` 从输入到路由到输出的完整串联）。
- 端到端冒烟（mock LLM，示例领域配置跑通一次真实意图）。
- 验收：`uv run pytest` 全绿；README 更新（安装、领域目录配置含示例）。

## 未来扩展（非本期）

- Orchestrator / Planner / Workers / Aggregator / Evaluator / Optimizer。
- 新领域只需替换领域配置目录（策略、prompt、intent、model），无需改核心框架。