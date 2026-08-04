# ExpertForge 设计文档

日期：2026-08-04

## 概述

ExpertForge 是一个**单领域可配置**的 AI 专家 Agent CLI 工具。用户提出问题时，Agent 先判定问题是否属于其配置的专家领域：属于则生成专业、多角度的回答；不属于则回复"不支持"，绝不越界回答。

定位示例（软件工程领域）：用户提出软件工程相关的技术问题时，Agent 以专业视角从多个角度解释技术问题。

## 目标与非目标

### 目标

- 以交互式 REPL 运行，支持多轮对话上下文
- 两阶段调用：领域分类器 + 回答生成器，边界判定可靠
- 单领域但可配置：领域描述为自然语言，换领域只需改配置
- 领域外问题明确拒绝，不产生越界内容

### 非目标

- 不支持多领域并存
- 不提供 Web/API 界面，仅 CLI
- 不引入大型 AI 框架

## 技术选型

- 语言：Python
- LLM 接口：`openai` SDK，通过可配置的 `base_url` 指向 OpenAI 兼容的第三方服务
- 配置：`config.json` + 环境变量（API Key 走环境变量）
- 交互：交互式 REPL（流式输出）
- 约定：发送给 LLM 的 prompt、通用错误与异常信息一律使用英文

## 架构与组件

```
agent/
├── agent_cli.py          # 入口：解析参数，启动 REPL
├── config.py             # 配置加载（config.json + 环境变量）
├── llm.py                # openai SDK 封装（chat 与流式 chat）
├── classifier.py         # 领域判定：LLM 分类调用 + 结果解析
├── generator.py          # 回答生成：构建领域 System Prompt + 多轮历史
├── repl.py               # 交互式对话循环（多轮、流式输出、退出命令）
└── tests/                # 单元测试
```

组件职责：

- **config.py**：读取 `config.json`（API 地址、模型、领域描述、拒绝语），API Key 从环境变量读取；缺失时给出友好错误。
- **llm.py**：封装 `openai` SDK，提供 `chat_completion()`（非流式）与 `chat_completion_stream()`（流式）；将 SDK 异常统一包装为 `LLMError`。
- **classifier.py**：用小 prompt 让 LLM 返回 JSON `{"in_domain": true/false, "reason": "..."}`，并解析。
- **generator.py**：把领域描述 + 半结构化引导（专业、多角度、可按问题调整结构）组装成 System Prompt，带多轮历史。
- **repl.py**：循环"输入→分类→生成/拒绝→输出"，支持流式打印和 `exit/quit` 退出。

## 数据流

单轮正常流程：

```
用户输入问题
  → classifier 调用：LLM 返回 {"in_domain": bool, "reason": "..."}
    → in_domain=false → 打印拒绝语（可附带 reason）
    → in_domain=true  → generator 调用：领域 System Prompt + 历史 + 当前问题
                        → 流式打印回答 → 追加到历史
  → 回到 REPL 等待下一条输入
```

多轮历史：每轮成功后，`(用户问题, 助手回答)` 追加到内存列表，作为后续 `generate()` 调用的 `messages`。历史上限为最近 20 轮，防止上下文超长。

## 领域判定

- 分类器以自然语言领域描述为唯一依据，由 LLM 判断问题是否在领域内。
- 分类器提示词（英文）让 LLM 返回 JSON 结构：`{"in_domain": boolean, "reason": "string"}`。
- 解析失败时：用更严格 prompt 重试一次；仍失败则按 `in_domain=false` 兜底处理（宁可误拒，不可越界回答），并提示"判定结果不可靠"。

## 回答生成

- System Prompt 由领域描述 + 半结构化引导组装：
  - 专业、权威的语气
  - 从多个角度解释问题
  - 允许 LLM 根据问题特性调整结构（半结构化，非固定模板）
- 生成调用携带多轮历史（最近 20 轮）与当前问题。

## 配置

配置文件 `config.json`：

```json
{
  "base_url": "https://api.example.com/v1",
  "model": "your-model-name",
  "classifier_model": null,
  "domain": {
    "name": "软件工程",
    "description": "涵盖软件设计、开发、测试、运维、性能优化等软件工程相关技术问题",
    "out_of_domain_reply": "这个问题不在我的专家领域（软件工程）范围内，暂时无法提供专业回答。"
  }
}
```

- `classifier_model` 为 null 时与 `model` 相同，可单独指定轻量模型。
- API Key 从环境变量 `AGENT_API_KEY` 读取。
- `base_url` 可由环境变量 `AGENT_BASE_URL` 覆盖。
- 换领域场景只需修改 `domain.description`。

## 错误处理

- 配置缺失 / API Key 未设置：启动时报错并提示配置方法，退出码非 0。
- API 调用失败（网络/超时/4xx/5xx）：REPL 内打印错误提示，不崩溃，可继续对话。
- 分类器返回无法解析：重试一次；仍失败按 `in_domain=false` 拒绝，并提示判定不可靠。
- 领域外问题：输出 `domain.out_of_domain_reply`（可自定义）。

## 测试策略

- `test_config.py`：配置加载、环境变量覆盖、缺失报错。
- `test_classifier.py`：mock HTTP 验证 JSON 解析、重试逻辑、失败兜底拒绝。
- `test_generator.py`：System Prompt 组装、历史截断、多轮消息结构。
- `test_repl.py`：用 fake LLM 客户端驱动完整流程（领域内回答 / 领域外拒绝 / API 报错不崩溃）。
- 手动验证：配置真实 API Key 后用示例问题体验 REPL。
