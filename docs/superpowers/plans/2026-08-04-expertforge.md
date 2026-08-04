# ExpertForge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个单领域可配置的 AI 专家 Agent CLI，两阶段判定领域边界：领域内生成专业多角度回答，领域外明确拒绝。

**Architecture:** 交互式 REPL 工具。用户提问 → `classifier` 用小调用判断是否属于配置的领域（返回 JSON）→ 领域外输出拒绝语；领域内由 `generator` 携带多轮历史流式生成回答。所有 HTTP 调用封装在 `llm_client`，走 OpenAI 兼容的 `/chat/completions` API。

**Tech Stack:** Python 3.10+（仅依赖 `requests`），测试用 `pytest`，无框架。

## Global Constraints

- Python >= 3.10，使用 `from __future__ import annotations`。
- 唯一第三方运行时依赖：`requests>=2.31`；开发依赖：`pytest>=8.0`。
- 配置文件键名必须与规格一致：`base_url`、`model`、`classifier_model`（null 时回退到 `model`）、`domain.name`、`domain.description`、`domain.out_of_domain_reply`。
- 环境变量：`AGENT_API_KEY`（必需）、`AGENT_BASE_URL`（可选，覆盖 `base_url`）。
- 分类器解析失败：重试一次；仍失败按 `in_domain=false` 拒绝，reason 含"判定结果不可靠"。
- 多轮历史上限：最近 20 轮。
- 所有用户可见文案使用中文。
- 退出命令：`exit` / `quit`。
- 测试运行方式：项目根目录执行 `./.venv/bin/python -m pytest`。

---

### Task 1: 项目脚手架

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `agent/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: `agent` 包（空 `__init__.py`）、可用的 venv、`pytest` 可发现 `tests/`。

- [ ] **Step 1: 创建 `requirements.txt`**

```text
requests>=2.31
pytest>=8.0
```

- [ ] **Step 2: 创建 `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
config.json
```

- [ ] **Step 3: 创建 `pytest.ini`**

```ini
[pytest]
testpaths = tests
```

- [ ] **Step 4: 创建空包**

创建 `agent/__init__.py` 与 `tests/__init__.py`，内容均为空文件。

- [ ] **Step 5: 创建 venv 并安装依赖**

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

- [ ] **Step 6: 验证**

Run: `./.venv/bin/python -m pytest`
Expected: 无测试被收集，退出码 0（`no tests ran`）。

- [ ] **Step 7: 提交**

```bash
git add requirements.txt .gitignore pytest.ini agent/__init__.py tests/__init__.py
git commit -m "chore: scaffold project"
```

---

### Task 2: 配置加载 config.py

**Files:**
- Create: `agent/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `class ConfigError(Exception)`
  - `@dataclass class AgentConfig`，字段：`base_url: str`、`model: str`、`classifier_model: str`、`domain_name: str`、`domain_description: str`、`out_of_domain_reply: str`
  - `def load_config(path: str | None = None) -> AgentConfig` —— `path` 缺省为 `"config.json"`；`AGENT_BASE_URL` 覆盖文件中的 `base_url`；`classifier_model` 为 null/缺失时回退到 `model`；`domain.name` 可缺省为空串；`out_of_domain_reply` 缺省时生成 `f"这个问题不在我的专家领域（{domain_name}）范围内，暂时无法提供专业回答。"`；缺少 `base_url`/`model`/`domain.description` 或文件不存在/非法 JSON 时抛 `ConfigError`。
  - `def get_api_key() -> str` —— 读 `AGENT_API_KEY`，缺失抛 `ConfigError`。

- [ ] **Step 1: 写失败测试**

```python
import json

import pytest

from agent.config import AgentConfig, ConfigError, get_api_key, load_config


def _write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_load_config_basic(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "classifier_model": None,
        "domain": {
            "name": "软件工程",
            "description": "软件工程相关技术问题",
            "out_of_domain_reply": "不支持。",
        },
    })
    cfg = load_config(path)
    assert isinstance(cfg, AgentConfig)
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.model == "model-a"
    assert cfg.classifier_model == "model-a"
    assert cfg.domain_name == "软件工程"
    assert cfg.domain_description == "软件工程相关技术问题"
    assert cfg.out_of_domain_reply == "不支持。"


def test_classifier_model_falls_back_to_model(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain": {"name": "软件工程", "description": "软件工程相关"},
    })
    cfg = load_config(path)
    assert cfg.classifier_model == "model-a"
    assert cfg.out_of_domain_reply == (
        "这个问题不在我的专家领域（软件工程）范围内，暂时无法提供专业回答。"
    )


def test_env_base_url_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BASE_URL", "https://env.example.com/v1")
    path = _write_config(tmp_path, {
        "base_url": "https://file.example.com/v1",
        "model": "m",
        "domain": {"description": "x"},
    })
    cfg = load_config(path)
    assert cfg.base_url == "https://env.example.com/v1"


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/path/config.json")


def test_invalid_json_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = tmp_path / "config.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_missing_base_url_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {"model": "m", "domain": {"description": "x"}})
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_model_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {"base_url": "https://x/v1", "domain": {"description": "x"}})
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_domain_description_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {"base_url": "https://x/v1", "model": "m", "domain": {}})
    with pytest.raises(ConfigError):
        load_config(path)


def test_get_api_key(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "secret")
    assert get_api_key() == "secret"


def test_get_api_key_missing(monkeypatch):
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        get_api_key()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'agent.config'`）。

- [ ] **Step 3: 实现 `agent/config.py`**

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass


class ConfigError(Exception):
    """配置无效或不完整时抛出。"""


DEFAULT_CONFIG_PATH = "config.json"


@dataclass
class AgentConfig:
    base_url: str
    model: str
    classifier_model: str
    domain_name: str
    domain_description: str
    out_of_domain_reply: str


def load_config(path: str | None = None) -> AgentConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ConfigError(
            f"配置文件不存在: {config_path}。请参照 config.example.json 创建。"
        )
    except json.JSONDecodeError as e:
        raise ConfigError(f"配置文件格式错误: {e}")

    if not isinstance(raw, dict):
        raise ConfigError("配置文件顶层必须是一个 JSON 对象")

    base_url = os.environ.get("AGENT_BASE_URL") or raw.get("base_url")
    if not base_url:
        raise ConfigError("缺少 base_url 配置或 AGENT_BASE_URL 环境变量")

    model = raw.get("model")
    if not model:
        raise ConfigError("缺少 model 配置")

    classifier_model = raw.get("classifier_model") or model

    domain = raw.get("domain")
    if not isinstance(domain, dict):
        raise ConfigError("缺少 domain 配置")

    domain_name = domain.get("name") or ""
    domain_description = domain.get("description")
    if not domain_description:
        raise ConfigError("缺少 domain.description 配置")

    out_of_domain_reply = domain.get("out_of_domain_reply") or (
        f"这个问题不在我的专家领域（{domain_name}）范围内，暂时无法提供专业回答。"
    )

    return AgentConfig(
        base_url=base_url,
        model=model,
        classifier_model=classifier_model,
        domain_name=domain_name,
        domain_description=domain_description,
        out_of_domain_reply=out_of_domain_reply,
    )


def get_api_key() -> str:
    api_key = os.environ.get("AGENT_API_KEY")
    if not api_key:
        raise ConfigError(
            "未设置 AGENT_API_KEY 环境变量，请先设置：export AGENT_API_KEY=你的Key"
        )
    return api_key
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS（11 项）。

- [ ] **Step 5: 提交**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat: add config loading with env overrides"
```

---

### Task 3: LLM HTTP 客户端 llm_client.py

**Files:**
- Create: `agent/llm_client.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Produces:
  - `class LLMError(Exception)`
  - `class LLMClient`：`__init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0)`
  - `LLMClient.chat_completion(self, messages: list[dict], *, model: str | None = None, temperature: float = 0.3) -> str` —— 非流式，返回完整内容；HTTP/解析错误抛 `LLMError`。
  - `LLMClient.chat_completion_stream(self, messages: list[dict], *, model: str | None = None, temperature: float = 0.7) -> Iterator[str]` —— 流式（SSE），逐块 yield content；错误抛 `LLMError`。
  - Consumes: `agent/config.py` 无（构造时直接传入字符串）。
  - 请求 URL 形如 `f"{base_url.rstrip('/')}/chat/completions"`，Header `Authorization: Bearer <api_key>`、`Content-Type: application/json`。

- [ ] **Step 1: 写失败测试**

```python
import json

import pytest
import requests

from agent.llm_client import LLMClient, LLMError


class FakeResponse:
    def __init__(self, payload=None, lines=None, status=200):
        self._payload = payload
        self._lines = lines
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


def test_chat_completion_returns_content(monkeypatch):
    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return FakeResponse(payload={"choices": [{"message": {"content": "你好"}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result == "你好"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer key"
    assert captured["payload"]["stream"] is False


def test_chat_completion_stream_yields_content(monkeypatch):
    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "世界"}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"content": "你好"}}]}),
        "data: [DONE]",
    ]
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(lines=lines))
    out = list(client.chat_completion_stream([{"role": "user", "content": "hi"}]))
    assert out == ["世界", "你好"]


def test_chat_completion_raises_llm_error_on_http_error(monkeypatch):
    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(status=500))
    with pytest.raises(LLMError):
        client.chat_completion([{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./.venv/bin/python -m pytest tests/test_llm_client.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现 `agent/llm_client.py`**

```python
from __future__ import annotations

import json
from typing import Iterator

import requests


class LLMError(Exception):
    """LLM API 调用失败时抛出。"""


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def chat_completion(
        self, messages: list[dict], *, model: str | None = None, temperature: float = 0.3
    ) -> str:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        try:
            response = requests.post(
                self._url(), headers=self._headers(), json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            raise LLMError(f"LLM API 调用失败: {e}") from e

    def chat_completion_stream(
        self, messages: list[dict], *, model: str | None = None, temperature: float = 0.7
    ) -> Iterator[str]:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        try:
            response = requests.post(
                self._url(),
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if choices:
                    content = choices[0].get("delta", {}).get("content")
                    if content:
                        yield content
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            raise LLMError(f"LLM API 调用失败: {e}") from e
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./.venv/bin/python -m pytest tests/test_llm_client.py -v`
Expected: PASS（3 项）。

- [ ] **Step 5: 提交**

```bash
git add agent/llm_client.py tests/test_llm_client.py
git commit -m "feat: add OpenAI-compatible LLM HTTP client"
```

---

### Task 4: 领域分类器 classifier.py

**Files:**
- Create: `agent/classifier.py`
- Test: `tests/test_classifier.py`

**Interfaces:**
- Produces:
  - `@dataclass class Classification`：`in_domain: bool`、`reason: str`
  - `def classify_question(client, question: str, domain_name: str, domain_description: str, *, model: str | None = None) -> Classification`
    - 调用 `client.chat_completion([{"role": "system", "content": prompt}], model=model)`。
    - 输出解析失败时用更严格 prompt 重试一次；仍失败返回 `Classification(in_domain=False, reason="判定结果不可靠：分类器输出无法解析")`。
    - `LLMError` 向上传播（由调用方处理）。
- Consumes: `LLMClient`、`LLMError`、`Classification`（Task 3、自身）。
- 后续 Task 6 使用 `classify_question` 与 `Classification`。

- [ ] **Step 1: 写失败测试**

```python
import pytest

from agent.classifier import Classification, classify_question
from agent.llm_client import LLMError


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None):
        self.calls.append((messages, model))
        return self.responses.pop(0)


def test_classify_in_domain():
    client = FakeClient(['{"in_domain": true, "reason": "属于软件工程"}'])
    result = classify_question(client, "什么是微服务？", "软件工程", "软件工程相关")
    assert isinstance(result, Classification)
    assert result.in_domain is True
    assert result.reason == "属于软件工程"


def test_classify_out_of_domain():
    client = FakeClient(['{"in_domain": false, "reason": "与领域无关"}'])
    result = classify_question(client, "今天天气如何？", "软件工程", "软件工程相关")
    assert result.in_domain is False


def test_retry_then_success():
    client = FakeClient(["不是JSON", '{"in_domain": true, "reason": "ok"}'])
    result = classify_question(client, "什么是Kafka？", "软件工程", "软件工程相关")
    assert result.in_domain is True
    assert len(client.calls) == 2


def test_retry_then_fallback_reject():
    client = FakeClient(["垃圾输出", "还是垃圾"])
    result = classify_question(client, "xxx", "软件工程", "软件工程相关")
    assert result.in_domain is False
    assert "不可靠" in result.reason
    assert len(client.calls) == 2


def test_propagates_llm_error():
    class FailingClient:
        def chat_completion(self, messages, model=None):
            raise LLMError("boom")

    with pytest.raises(LLMError):
        classify_question(FailingClient(), "q", "软件工程", "软件工程相关")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./.venv/bin/python -m pytest tests/test_classifier.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现 `agent/classifier.py`**

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm_client import LLMClient, LLMError


@dataclass
class Classification:
    in_domain: bool
    reason: str


_CLASSIFY_PROMPT = """你是领域边界判定器。根据给定的专家领域，判断用户问题是否属于该领域。

专家领域名称：{name}
专家领域描述：{description}

规则：
- 只输出一个 JSON 对象，不要输出任何其他内容。
- JSON 格式：{{"in_domain": true 或 false, "reason": "一句话说明判断理由"}}

用户问题：{question}
"""


def _build_prompt(name: str, description: str, question: str, strict: bool = False) -> str:
    prompt = _CLASSIFY_PROMPT.format(name=name, description=description, question=question)
    if strict:
        prompt += "\n再次强调：只输出上述 JSON 对象，禁止输出任何多余文字。"
    return prompt


def _parse_classification(text: str) -> Classification | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "in_domain" not in data:
        return None
    reason = data.get("reason")
    return Classification(
        in_domain=bool(data["in_domain"]),
        reason=reason if isinstance(reason, str) else "",
    )


def classify_question(
    client: LLMClient,
    question: str,
    domain_name: str,
    domain_description: str,
    *,
    model: str | None = None,
) -> Classification:
    for strict in (False, True):
        prompt = _build_prompt(domain_name, domain_description, question, strict=strict)
        text = client.chat_completion(
            [{"role": "system", "content": prompt}], model=model
        )
        result = _parse_classification(text)
        if result is not None:
            return result
    return Classification(in_domain=False, reason="判定结果不可靠：分类器输出无法解析")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./.venv/bin/python -m pytest tests/test_classifier.py -v`
Expected: PASS（5 项）。

- [ ] **Step 5: 提交**

```bash
git add agent/classifier.py tests/test_classifier.py
git commit -m "feat: add domain classifier with retry and reject fallback"
```

---

### Task 5: 回答生成器 generator.py

**Files:**
- Create: `agent/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Produces:
  - `def build_system_prompt(domain_name: str, domain_description: str) -> str` —— 组装"领域专家 + 半结构化引导"System Prompt，必须包含领域名、领域描述、"多角度"要求。
  - `def build_messages(system_prompt: str, history: list[tuple[str, str]], question: str, *, max_turns: int = 20) -> list[dict]` —— 返回 `[{"role":"system","content":system_prompt}]` + 最近 `max_turns` 轮的 user/assistant 对 + 末尾当前问题 user 消息。
- Consumes: 无（纯函数）。
- 后续 Task 6 使用 `build_system_prompt`、`build_messages`。

- [ ] **Step 1: 写失败测试**

```python
from agent.generator import build_messages, build_system_prompt


def test_build_system_prompt_contains_domain():
    prompt = build_system_prompt("软件工程", "软件工程相关技术问题")
    assert "软件工程" in prompt
    assert "软件工程相关技术问题" in prompt
    assert "多角度" in prompt


def test_build_messages_structure():
    messages = build_messages("sys", [("问1", "答1")], "问2")
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问1"},
        {"role": "assistant", "content": "答1"},
        {"role": "user", "content": "问2"},
    ]


def test_build_messages_truncates_history():
    history = [(f"q{i}", f"a{i}") for i in range(30)]
    messages = build_messages("sys", history, "final", max_turns=5)
    assert len(messages) == 1 + 10 + 1
    assert messages[1]["content"] == "q25"
    assert messages[-1]["content"] == "final"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./.venv/bin/python -m pytest tests/test_generator.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现 `agent/generator.py`**

```python
from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = """你是{name}领域的专家 Agent。

{description}

回答要求：
- 以专业、权威的语气回答问题。
- 从多个角度解释用户提出的问题，做到全面、有深度。
- 根据问题的特性灵活组织回答结构，不必套用固定模板。
- 只回答该领域内的问题。
"""


def build_system_prompt(domain_name: str, domain_description: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(name=domain_name, description=domain_description)


def build_messages(
    system_prompt: str,
    history: list[tuple[str, str]],
    question: str,
    *,
    max_turns: int = 20,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for user_text, assistant_text in history[-max_turns:]:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": question})
    return messages
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./.venv/bin/python -m pytest tests/test_generator.py -v`
Expected: PASS（3 项）。

- [ ] **Step 5: 提交**

```bash
git add agent/generator.py tests/test_generator.py
git commit -m "feat: add answer generator with history truncation"
```

---

### Task 6: 交互式对话循环 repl.py

**Files:**
- Create: `agent/repl.py`
- Test: `tests/test_repl.py`

**Interfaces:**
- Produces:
  - `def run_repl(client: LLMClient, config: AgentConfig) -> None`
    - 打印欢迎语（含领域名）；循环 `input("你 > ")`；`exit`/`quit` 退出；空输入跳过；`EOFError`/`KeyboardInterrupt` 退出。
    - 每轮：调 `classify_question(client, question, config.domain_name, config.domain_description, model=config.classifier_model)`。
    - `in_domain=false`：打印 `config.out_of_domain_reply`，若有 reason 再打印一行 `（{reason}）`。
    - `in_domain=true`：`build_messages` 组装 → `client.chat_completion_stream(messages, model=config.model)` 流式打印（前缀 `专家 > `），成功后把 `(question, 完整回答)` 追加到 `history`。
    - `LLMError` 在分类或生成时被捕获，打印 `[错误] {e}`，继续循环不崩溃。
- Consumes: `classify_question`、`Classification`（Task 4）、`build_system_prompt`、`build_messages`（Task 5）、`LLMError`（Task 3）、`AgentConfig`（Task 2）。

- [ ] **Step 1: 写失败测试**

```python
from agent.config import AgentConfig
from agent.llm_client import LLMError
from agent.repl import run_repl


class FakeClient:
    def __init__(self, classifications, streams):
        self.classifications = list(classifications)
        self.streams = list(streams)
        self.generate_calls = []

    def chat_completion(self, messages, model=None):
        return self.classifications.pop(0)

    def chat_completion_stream(self, messages, model=None):
        self.generate_calls.append(messages)
        for ch in self.streams.pop(0):
            yield ch


def _config():
    return AgentConfig(
        base_url="https://x",
        model="m",
        classifier_model="m",
        domain_name="软件工程",
        domain_description="软件工程相关",
        out_of_domain_reply="领域外，不支持。",
    )


def test_repl_in_domain_streams_answer(monkeypatch, capsys):
    inputs = iter(["什么是微服务？", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient(['{"in_domain": true, "reason": "ok"}'], ["流式回答"])
    run_repl(client, _config())
    out = capsys.readouterr().out
    assert "流式回答" in out
    assert len(client.generate_calls) == 1


def test_repl_out_of_domain_rejected(monkeypatch, capsys):
    inputs = iter(["今天天气如何？", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient(['{"in_domain": false, "reason": "与领域无关"}'], [])
    run_repl(client, _config())
    out = capsys.readouterr().out
    assert "领域外，不支持。" in out


def test_repl_api_error_does_not_crash(monkeypatch, capsys):
    inputs = iter(["问题", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    class ErrorClient:
        def chat_completion(self, messages, model=None):
            raise LLMError("网络错误")

        def chat_completion_stream(self, messages, model=None):
            raise AssertionError("不应被调用")

    run_repl(ErrorClient(), _config())
    out = capsys.readouterr().out
    assert "网络错误" in out


def test_repl_blank_input_skipped(monkeypatch, capsys):
    inputs = iter(["   ", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient([], [])
    run_repl(client, _config())
    out = capsys.readouterr().out
    assert "再见" in out
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./.venv/bin/python -m pytest tests/test_repl.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现 `agent/repl.py`**

```python
from __future__ import annotations

from .classifier import classify_question
from .config import AgentConfig
from .generator import build_messages, build_system_prompt
from .llm_client import LLMClient, LLMError


def run_repl(client: LLMClient, config: AgentConfig) -> None:
    system_prompt = build_system_prompt(config.domain_name, config.domain_description)
    history: list[tuple[str, str]] = []
    print(f"ExpertForge | 领域：{config.domain_name} | 输入 exit 或 quit 退出")

    while True:
        try:
            question = input("你 > ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n再见。")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("再见。")
            break

        try:
            result = classify_question(
                client,
                question,
                config.domain_name,
                config.domain_description,
                model=config.classifier_model,
            )
        except LLMError as e:
            print(f"[错误] {e}")
            continue

        if not result.in_domain:
            print(config.out_of_domain_reply)
            if result.reason:
                print(f"（{result.reason}）")
            continue

        messages = build_messages(system_prompt, history, question)
        print("专家 > ", end="", flush=True)
        answer_parts: list[str] = []
        try:
            for chunk in client.chat_completion_stream(messages, model=config.model):
                print(chunk, end="", flush=True)
                answer_parts.append(chunk)
            print()
            history.append((question, "".join(answer_parts)))
        except LLMError as e:
            print()
            print(f"[错误] {e}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `./.venv/bin/python -m pytest tests/test_repl.py -v`
Expected: PASS（4 项）。

- [ ] **Step 5: 提交**

```bash
git add agent/repl.py tests/test_repl.py
git commit -m "feat: add interactive REPL loop"
```

---

### Task 7: CLI 入口 + 示例配置 + README

**Files:**
- Create: `agent/agent_cli.py`
- Create: `agent/__main__.py`
- Create: `config.example.json`
- Modify: `README.md`
- Test: `tests/test_agent_cli.py`

**Interfaces:**
- Produces:
  - `def main(argv: list[str] | None = None) -> int` —— 支持 `-h/--help`（打印用法，返回 0）；第一个参数作为配置文件路径（可选）；`ConfigError` 时打印 `配置错误：{e}` 到 stderr 并返回 1；否则创建 `LLMClient` 并 `run_repl`，正常结束返回 0。
  - `agent/__main__.py` 调 `main()`，支持 `python -m agent`。
- Consumes: `load_config`、`get_api_key`、`ConfigError`（Task 2）、`LLMClient`（Task 3）、`run_repl`（Task 6）。

- [ ] **Step 1: 写失败测试**

```python
import pytest

from agent import agent_cli


def test_main_missing_config_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    code = agent_cli.main([str(tmp_path / "no-such.json")])
    assert code == 1
    err = capsys.readouterr().err
    assert "配置错误" in err


def test_main_missing_api_key_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"base_url": "https://x/v1", "model": "m", '
        '"domain": {"description": "d"}}',
        encoding="utf-8",
    )
    code = agent_cli.main([str(config_file)])
    assert code == 1
    err = capsys.readouterr().err
    assert "AGENT_API_KEY" in err


def test_main_help_exits_0(capsys):
    assert agent_cli.main(["-h"]) == 0
    out = capsys.readouterr().out
    assert "用法" in out


def test_main_runs_repl(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"base_url": "https://x/v1", "model": "m", '
        '"domain": {"name": "软件工程", "description": "d", '
        '"out_of_domain_reply": "领域外。"}}',
        encoding="utf-8",
    )

    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    class FakeClient:
        def chat_completion(self, messages, model=None):
            return '{"in_domain": true, "reason": "ok"}'

        def chat_completion_stream(self, messages, model=None):
            return iter(["你好"])

    monkeypatch.setattr(agent_cli, "LLMClient", lambda *a, **k: FakeClient())
    assert agent_cli.main([str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "ExpertForge" in out
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./.venv/bin/python -m pytest tests/test_agent_cli.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现 `agent/agent_cli.py`**

```python
from __future__ import annotations

import sys

from .config import ConfigError, get_api_key, load_config
from .llm_client import LLMClient
from .repl import run_repl


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if args and args[0] in {"-h", "--help"}:
        print("用法：python -m agent [配置文件路径]")
        return 0

    config_path = args[0] if args else None
    try:
        config = load_config(config_path)
        api_key = get_api_key()
    except ConfigError as e:
        print(f"配置错误：{e}", file=sys.stderr)
        return 1

    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model)
    try:
        run_repl(client, config)
    except KeyboardInterrupt:
        print("\n再见。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 创建 `agent/__main__.py`**

```python
import sys

from .agent_cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 运行测试确认通过**

Run: `./.venv/bin/python -m pytest tests/test_agent_cli.py -v`
Expected: PASS（4 项）。

- [ ] **Step 6: 创建 `config.example.json`**

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

- [ ] **Step 7: 更新 `README.md`**

覆盖：项目简介、安装（`python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt`）、配置（复制 `config.example.json` 为 `config.json` 并填写 `base_url`、`model`、`domain.description`；设置 `export AGENT_API_KEY=你的Key`）、运行（`./.venv/bin/python -m agent`）、退出命令说明。

- [ ] **Step 8: 全量回归测试**

Run: `./.venv/bin/python -m pytest -v`
Expected: 全部 PASS（25 项：config 11 + llm_client 3 + classifier 5 + generator 3 + repl 4 + agent_cli 4 = 30 项，如与清单不符以实际收集为准）。

- [ ] **Step 9: 端到端验证（无 API Key 时启动报错路径）**

Run: `env -u AGENT_API_KEY ./.venv/bin/python -m agent`
Expected: 退出码 1，stderr 打印 `配置错误：未设置 AGENT_API_KEY 环境变量...`（无 config.json 时则提示创建配置文件）。

- [ ] **Step 10: 提交**

```bash
git add agent/agent_cli.py agent/__main__.py config.example.json README.md tests/test_agent_cli.py
git commit -m "feat: add CLI entrypoint, example config, and README"
```

---

### 自审记录

- **规格覆盖**：组件（config/llm_client/classifier/generator/repl/agent_cli）逐一对应 Task 2-7；数据流（分类→拒绝/生成→历史追加）在 Task 6 实现；解析失败重试一次后拒绝在 Task 4 实现；历史上限 20 轮在 Task 5 实现；配置与环境变量在 Task 2 实现；错误处理（配置缺失、API 失败不崩溃）在 Task 2/6/7 实现；测试策略四类测试文件对应 Task 2/4/5/6。
- **占位符扫描**：所有步骤含完整代码与预期输出，无 TBD/TODO。
- **类型一致性**：`classify_question`、`Classification`、`build_system_prompt`、`build_messages`、`run_repl`、`load_config`、`get_api_key`、`LLMClient`、`LLMError`、`AgentConfig` 在后续任务中的引用签名与定义一致；`chat_completion_stream` 在两处均以生成器迭代使用。
