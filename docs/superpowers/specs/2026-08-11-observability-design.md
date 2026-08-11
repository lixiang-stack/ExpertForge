# Observability Design (Token + Trace Observability Plugin)

**Date:** 2026-08-11
**Status:** Design agreed; pending user review
**Source requirement:** user request — 追踪 token 消耗；关键结果可视化、可追溯；用于系统运行分析与 LLM 成本分析。

## 1. Goal

为一个 OpenAI 兼容的领域专家 agent（ExpertForge）提供可插拔、低侵入的观测能力：

1. 追踪每次 LLM 调用的 token 用量与耗时。
2. 追溯全链路关键结果：分类/路由决策、策略处理、编排步骤（Planner → Worker → Aggregator）、最终回答。
3. 提供 CLI 汇总表、HTML 报告、终端实时展示三种消费形态，跨会话累积、历史聚合。
4. 不计算货币成本，只记录 token 用量。

**硬性约束：** 业务模块（`chat.py`/`orchestrator.py`/`classification.py`/`strategy.py`/`router.py`）源码零改动（唯一例外见 §4：`llm.py` 返回对象化，6 处调用点加 `.text`）。观测能力由可插拔插件包自动化提供，关闭时零行为差异、零开销。

## 2. Architecture

观测能力收进独立的 `agent/observability/` 插件包。唯一装配点是 `agent_cli.py` 组合根。

```
agent/
  observability/            # 新增插件包（可插拔）
    __init__.py             # 公开 install() 与 report 入口
    config.py               # ObservabilityConfig 读取与校验
    tracing.py              # TraceStore(JSONL 写入) + SpanStack(contextvars)
    client.py               # TracedLLMClient：自动记 token/耗时/phase
    patch.py                # 包装 6 个流程入口，自动开关 span
    report.py               # CLI 表格 + HTML 报告生成
```

**装配点**（`agent_cli.py` 内约 2 行）：

```python
client = LLMClient(...)
client, plugin = install_observability(client, config, domain)  # enabled 时返回包装后的
```

关闭时 `install_observability` 原样返回 `(client, None)`，行为与现状完全一致。

### 2.1 配置

`config.json` 新增可选段，默认关闭：

```json
{
  "observability": {
    "enabled": false,
    "data_dir": ".observability",
    "phase_map": {}
  }
}
```

- `enabled` — 是否激活观测。`false`（默认）时零开销。
- `data_dir` — JSONL 事件文件目录，默认 `.observability`。
- `phase_map` — 可选，覆盖默认"方法名 → 阶段名"映射表（§3.3）。

`AgentConfig` 增加一个可空字段 `observability`（`ObservabilityConfig | None`）。`observability.enabled` 为 false 或缺省时该字段为 `None`。

## 3. Data Model

**存储格式：** JSONL 追加写（每行一个事件），按天切分：`{data_dir}/trace-YYYY-MM-DD.jsonl`。

**Trace 边界：** 一次 `Chat.respond`（一个问题）= 一个 `trace_id`。每个事件共享 `trace_id` + 单调递增 `ts`（epoch 毫秒）。

### 3.1 事件类型

| 事件 | 字段 | 产生时机 |
|---|---|---|
| `trace_start` | question, domain, ts | 进入 `Chat.respond` |
| `llm_call` | phase, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, status, error | 每次 LLM 调用完成（含失败） |
| `decision` | phase, data | 分类结果、路由决策、Planner 输出 |
| `trace_end` | answer_len, total_tokens, total_llm_calls, total_latency_ms | 回答生成完毕 |

### 3.2 Token 采集

- 非流式：从 `resp.usage` 读 `prompt_tokens`/`completion_tokens`/`total_tokens`。
- 供应商不返回 `usage` 时，记 `usage: null`，报表明确区分"有/无 usage 数据"，**不当作 0**。
- 失败调用：记录 `status: "error"` + `error` 字符串，token 记 `null`（不可得）。

### 3.3 Phase 标签体系

```
classification → route → strategy.<id> → orchestration.planner
→ orchestration.worker.0 … orchestration.worker.n → orchestration.aggregate
```

- `strategy.<id>`：`<id>` 为策略名（如 `strategy.direct`、`strategy.debugging`）。
- `orchestration.worker.<n>`：worker 编号由包装器按调用次序自动分配。
- 各阶段由插件包装方法自动设置（§4）。

## 4. Instrumentation

**核心机制：`contextvars.ContextVar` 维护 span 栈**，业务代码无感知。

```python
# tracing.py
class SpanStack:
    trace_id: str   # 绑定到当前 trace
    phase: str      # 当前最深阶段，如 "orchestration.worker.1"
```

### 4.1 TracedLLMClient（agent/observability/client.py）

普通 client 的**透明包装**，代理所有 `chat_completion` 调用，读 span 栈取 phase 标签：

- 调用前记录 `monotonic()` 起始时间。
- 调用成功：记录 `llm_call`（phase、model、usage、latency_ms、status=ok）。
- 调用失败（`LLMError`）：记录 `llm_call`（status=error、error），随后**重新抛出原异常**（不吞）。

### 4.2 LLMClient 返回对象化（对业务的最小侵入）

现状：`LLMClient.chat_completion` 返回 `str`，丢弃了 `usage`。

改造：`agent/llm.py` 的 `chat_completion` 返回 `ChatCompletionResult(text, usage, model)`：

```python
@dataclass
class ChatCompletionResult:
    text: str
    usage: Usage | None      # prompt_tokens / completion_tokens / total_tokens
    model: str
```

业务调用点改 `.text`，共 6 处：
- `strategy.py:29` `client.chat_completion(...)` → `....text`（1 处）
- `orchestrator.py` `_plan`/`_worker`/`_aggregate`/`_direct_answer`（4 处）
- `classification.py:164` `client.chat_completion(...)` → `....text`（1 处）

这是对业务模块唯一的改动。`chat_completion_stream` 保持返回 `Iterator[str]` 不变，观测层第 1 版不包装流式调用，在 `install()` 时跳过并注释说明（主流程不使用流式，流式仅测试覆盖）。

### 4.3 插件装配（agent/observability/patch.py）

安装时包装 6 个流程入口，自动开关 span 栈，业务源码不动。用 `functools.wraps` 包装后替换实例/类属性。

| 包装的方法 | 开启的 span 阶段 | 额外记录 |
|---|---|---|
| `Chat.respond` | trace 根（trace_start/trace_end） | question, answer_len, 汇总 |
| `Router.route` | `classification` + `route` | classification JSON、RouteResult |
| `Orchestrator.run` | `orchestration` | 是否 degrade 到 direct |
| `Orchestrator._plan` | `orchestration.planner` | Planner JSON 输出 |
| `Orchestrator._worker` | `orchestration.worker.<n>` | 子任务标题 |
| `Orchestrator._aggregate` | `orchestration.aggregate` | — |

包装统一做三件事：push span → 调用原方法 → 记录 decision 事件 → pop span。

**phase_map 覆盖：** `patch.py` 维护默认映射表 `method → phase`。用户可通过 `config.observability.phase_map` 覆盖默认值（按类名.方法名键，如 `"Orchestrator._worker": "orchestration.worker"`）。

### 4.4 健壮性

- 某方法签名在重构中变化导致包装失败：`install()` 捕获异常，`warnings.warn` 后**跳过该项**，绝不阻断主流程——观测层失效是"降级"，不是"故障"。
- 观测层自身异常（磁盘满、写失败）一律 catch → `warnings.warn`，绝不冒泡到业务代码。
- 观测文件损坏的一行：跳过该行并计数，不中止解析。

## 5. Real-time Terminal Display

REPL 场景下，每次 `Chat.respond` 完成后在终端追加一行紧凑统计（复用 span 数据，不额外写文件）：

```
you > 什么是闭包？
expert > （回答）
[trace abc123] classification 0.3s/1.2k tok | route(direct) 0.1s/0.4k | strategy.direct 2.1s/3.5k | total 5.1k tok 2.5s
```

`--ask` 单发模式同样打印该行。

## 6. Reporters

**入口：** `python -m agent.observability report [--date YYYY-MM-DD] [--html]`

### 6.1 CLI 汇总表（默认）

- 按 trace 汇总：每个问题一行——阶段拆分、总 token、输入/输出拆分、总耗时、LLM 调用数。
- 顶部聚合：当日总 token、按 phase/model 分组小计、无 usage 数据占比。
- 纯文本表格（固定列宽），不依赖第三方库。

### 6.2 HTML 报告（`--html`）

- 自包含单文件，内联 CSS/SVG，不使用 CDN，离线可用。
- 模块：
  1. token 趋势图（按天/按 trace）
  2. model 分布
  3. phase 耗时瀑布图
  4. 可展开的 trace 详情（每阶段 token/耗时）
- 输出到 `data_dir/report.html`，浏览器直接打开。

## 7. Error Handling

- 观测层异常（写文件、包装、解析）全部降级：catch → `warnings.warn`，不影响业务。
- `usage` 缺失 → 记 `null`，报表区分，不当作 0。
- 失败 `llm_call` 记录 error 事件，异常继续由业务层处理（不吞）。

## 8. Testing

全走 mock，无需 API key。新增 `tests/test_observability_*.py`：

- `test_observability_config.py`：config 解析、enabled/disabled、phase_map 覆盖、缺省为 None。
- `test_tracing.py`：TraceStore 写入/读回、span 栈 push/pop、trace_id 唯一、按天切分。
- `test_observability_install.py`：
  - install 后业务对象被包装。
  - llm_call 事件带正确 phase/usage。
  - 异常时记录 error 事件且重新抛出（不吞）。
  - 包装失败时跳过该项并 warning，不阻断。
  - enabled=false 时行为与现状完全一致（零差异）。
- `test_report.py`：CLI 表与 HTML 对已知 JSONL 快照生成，断言关键数字。

回归：现有 `uv run pytest -q` 全绿。

## 9. Success Criteria

1. `uv run pytest -q` 全绿。
2. 业务模块除 `llm.py` 返回对象化（6 处 `.text` 调用点）外零改动。
3. `observability.enabled=false` 时行为与现在完全一致。
4. 启用后自动记录：每个 trace 的 classification/route/strategy/编排各阶段 token 与耗时，无需手写记录代码。
5. CLI 表、HTML 报告、终端实时行三种消费形态可用，跨会话累积。
6. 观测层任何异常降级为 warning，绝不阻断业务。
