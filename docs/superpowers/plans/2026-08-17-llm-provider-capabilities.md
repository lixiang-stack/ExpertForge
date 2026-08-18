# LLM Provider Capabilities Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Abstract provider differences (structured-output support, thinking toggle, user-message requirement) behind a capability model with negotiation and robust JSON parsing, so callers state intent without binding to a provider.

> **Prerequisite / execution base:** Execute this plan directly on the PR #15 branch `fix/gemini-smoke-compat` (worktree `.worktrees/gemini-smoke-compat`, base commit `13c1744`), adding commits on top. PR #15 is NOT merged first — it becomes a combined PR containing both the Gemini smoke compatibility fix and the capability abstraction work. The plan builds on the PR #15 state: the `disable_thinking` config key, per-call user messages at classification/planner/judge, the `test_*_disable_thinking_*` tests, and the smoke-config preference. The plan's Tasks 4, 5, 7, 8, 9 explicitly modify or replace those PR #15 artifacts.

**Architecture:** Two small layers on top of the existing OpenAI-compat transport: `agent/capabilities.py` (user-declared capability data model — `ProviderCapabilities` frozen dataclass + `KNOWN_CAPABILITY_KEYS`, no detection, no default table) and `agent/negotiate.py` (pure function picking json_schema → json_object → no-request). `agent/parsing.py` is plain `json.loads` — unparseable output is treated as an error, never rescued. Capabilities are REQUIRED config (`provider` + `provider_capabilities`, validated), passed by the CLIs into `LLMClient`, which wires negotiation + capability-gated thinking + an unconditional user-message guard. Call sites (classification/planner/judge) state intent only; the question is removed from system prompts (user message only).

**Tech Stack:** Python, `openai` SDK (existing), `pytest`.

## Global Constraints

- TDD: write the failing test first, verify it fails, implement, verify it passes, commit.
- Run tests with `uv run pytest` from the repo root.
- **New modules live at `agent/` root** (`agent/capabilities.py`, `agent/negotiate.py`, `agent/parsing.py`). Do NOT create `agent/llm/` — `agent/llm.py` is a module, not a package.
- The json_schema `response_format` wrapper uses the fixed generic name `"structured_output"` (the identifier is irrelevant to providers; only the schema matters).
- `disable_thinking` remains a per-call parameter. The `thinking` `extra_body` is sent ONLY when `capabilities.supports_thinking_toggle` is True. No other use of `disable_thinking`.
- Do not change the OpenAI transport, streaming, or token accounting.
- When editing a file, keep surrounding code style and existing comment blocks intact unless a task explicitly removes them.

---
### Task 1: Provider capabilities module

**Files:**
- Create: `agent/capabilities.py`
- Create: `tests/test_capabilities.py`

**Interfaces:**
- Produces: `ProviderCapabilities` (frozen dataclass) and `KNOWN_CAPABILITY_KEYS` (tuple of valid flag names). No detection, no default table, no merge logic — capabilities are user-declared config data.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_capabilities.py
from agent.capabilities import KNOWN_CAPABILITY_KEYS, ProviderCapabilities


def test_defaults():
    caps = ProviderCapabilities(provider="deepseek")
    assert caps.provider == "deepseek"
    assert caps.supports_json_schema is False
    assert caps.supports_thinking_toggle is False
    assert caps.supports_tool_call is False


def test_explicit_flags():
    caps = ProviderCapabilities(
        provider="gemini",
        supports_json_schema=True,
        supports_thinking_toggle=True,
    )
    assert caps.supports_json_schema is True
    assert caps.supports_thinking_toggle is True
    assert caps.supports_tool_call is False  # untouched default


def test_known_capability_keys():
    assert set(KNOWN_CAPABILITY_KEYS) == {
        "supports_json_schema",
        "supports_thinking_toggle",
        "supports_tool_call",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_capabilities.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.capabilities'`

- [ ] **Step 3: Write the implementation**

```python
# agent/capabilities.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str
    supports_json_schema: bool = False
    supports_thinking_toggle: bool = False
    supports_tool_call: bool = False


KNOWN_CAPABILITY_KEYS = (
    "supports_json_schema",
    "supports_thinking_toggle",
    "supports_tool_call",
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_capabilities.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/capabilities.py tests/test_capabilities.py
git commit -m "feat: provider capabilities data model"
```

---
### Task 2: Structured-output negotiation

**Files:**
- Create: `agent/negotiate.py`
- Create: `tests/test_negotiate.py`

**Interfaces:**
- Consumes: `ProviderCapabilities` from `agent/capabilities.py` (Task 1).
- Produces: `negotiate_structured_output(caps: ProviderCapabilities, *, json_mode: bool, json_schema: dict | None) -> str | None` returning `"json_schema"`, `"json_object"`, or `None` (caller requested no structured output). `json_object` is the universal fallback; there is no capability-fallback "none".

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_negotiate.py
from agent.capabilities import ProviderCapabilities
from agent.negotiate import negotiate_structured_output

deepseek = ProviderCapabilities(provider="deepseek")
gemini = ProviderCapabilities(provider="gemini", supports_json_schema=True)


def test_schema_supported_uses_schema():
    assert negotiate_structured_output(gemini, json_mode=True, json_schema={"type": "object"}) == "json_schema"


def test_schema_unsupported_degrades_to_json_object():
    assert negotiate_structured_output(deepseek, json_mode=True, json_schema={"type": "object"}) == "json_object"


def test_json_mode_uses_json_object():
    assert negotiate_structured_output(deepseek, json_mode=True, json_schema=None) == "json_object"


def test_no_request_returns_none():
    assert negotiate_structured_output(gemini, json_mode=False, json_schema=None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_negotiate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.negotiate'`

- [ ] **Step 3: Write the implementation**

```python
# agent/negotiate.py
from __future__ import annotations

from .capabilities import ProviderCapabilities


def negotiate_structured_output(
    caps: ProviderCapabilities,
    *,
    json_mode: bool,
    json_schema: dict | None,
) -> str | None:
    """Pick the structured-output mechanism.

    json_object is the universal default for OpenAI-compat targets. json_schema
    is preferred when the provider declares it. Returns None only when the
    caller requested no structured output (plain-answer paths) — never as a
    capability fallback.
    """
    if json_schema is not None:
        return "json_schema" if caps.supports_json_schema else "json_object"
    if json_mode:
        return "json_object"
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_negotiate.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/negotiate.py tests/test_negotiate.py
git commit -m "feat: structured-output negotiation (json_schema preferred, json_object universal)"
```

---
### Task 3: Robust JSON parsing layer

**Files:**
- Create: `agent/parsing.py`
- Create: `tests/test_parsing.py`

**Interfaces:**
- Produces: `parse_json(text: str) -> dict | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parsing.py
from agent.parsing import parse_json


def test_pure_json():
    assert parse_json('{"ok": true}') == {"ok": True}


def test_nested_json():
    assert parse_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}


def test_non_object_list_returns_none():
    assert parse_json('[1, 2]') is None


def test_invalid_returns_none():
    assert parse_json("not json at all") is None


def test_prose_wrapped_treated_as_error():
    assert parse_json('Here is the result: {"ok": true} end.') is None


def test_empty_and_none_returns_none():
    assert parse_json("") is None
    assert parse_json(None) is None


def test_whitespace_only_returns_none():
    assert parse_json("   ") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_parsing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.parsing'`

- [ ] **Step 3: Write the implementation**

```python
# agent/parsing.py
from __future__ import annotations

import json


def parse_json(text: str) -> dict | None:
    """Parse a pure-JSON response into a dict.

    Structured-output modes (json_schema/json_object) guarantee pure JSON, so
    there is no extraction fallback: unparseable output is treated as an error
    (``None``) and callers degrade via their existing validation paths. The
    greedy ``re.search(r"\{.*\}")`` approach is never used.
    """
    if not text:
        return None
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_parsing.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/parsing.py tests/test_parsing.py
git commit -m "feat: pure-JSON parsing layer (unparseable treated as error)"
```

---
### Task 4: LLMClient capability integration

**Files:**
- Modify: `agent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `ProviderCapabilities` (Task 1), `negotiate_structured_output` (Task 2).
- Produces: `LLMClient(base_url, api_key, model, timeout=None, provider="", capability_overrides=None)` with `self.capabilities = ProviderCapabilities(provider=provider or "unknown", **capability_overrides or {})`; `chat_completion` keeps its existing signature and adds: unconditional user-message guard (raises `LLMError`), capability-gated `extra_body`, negotiated `response_format`.

- [ ] **Step 1: Update existing tests and add new failing tests**

In `tests/test_llm.py`:

1. In `test_chat_completion_disable_thinking_passes_extra_body` (line 44), pass the capability explicitly instead of relying on base_url:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_disable_thinking_passes_extra_body(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient(
        "https://api.example.com/v1", "key", "model-a",
        provider="deepseek", capability_overrides={"supports_thinking_toggle": True},
    )
    client.chat_completion([{"role": "user", "content": "hi"}], disable_thinking=True)

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
```

2. Replace `test_chat_completion_omits_extra_body_when_client_disabled` (line 58) with a capability-based test:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_omits_extra_body_when_provider_unsupported(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a", provider="gemini")
    client.chat_completion([{"role": "user", "content": "hi"}], disable_thinking=True)

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert "extra_body" not in kwargs
```

3. In `test_chat_completion_json_mode_with_disable_thinking` (line 130), pass the capability explicitly:

```python
    client = LLMClient(
        "https://api.example.com/v1", "key", "model-a",
        provider="deepseek", capability_overrides={"supports_thinking_toggle": True},
    )
```

4. In `test_chat_completion_json_schema_passes_response_format` (line 163), pass a schema-capable provider and use the wrapper name `"structured_output"`:

```python
    client = LLMClient(
        "https://api.example.com/v1", "key", "model-a",
        provider="gemini", capability_overrides={"supports_json_schema": True},
    )
    client.chat_completion([{"role": "user", "content": "hi"}], json_schema=schema)

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "structured_output", "schema": schema, "strict": False},
    }
```

5. In `test_chat_completion_json_schema_wins_over_json_mode` (line 186), pass a schema-capable provider:

```python
    client = LLMClient(
        "https://api.example.com/v1", "key", "model-a",
        provider="gemini", capability_overrides={"supports_json_schema": True},
    )
```

6. Add the new tests below the end of the file (after `test_chat_completion_zero_tokens_when_usage_absent`):

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_json_schema_degrades_to_json_object(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "{}"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a", provider="deepseek")
    client.chat_completion([{"role": "user", "content": "hi"}], json_schema={"type": "object"})

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@patch("agent.llm.OpenAI")
def test_chat_completion_without_user_message_raises(mock_openai):
    client = LLMClient("https://api.example.com/v1", "key", "model-a", provider="deepseek")
    with pytest.raises(LLMError):
        client.chat_completion([{"role": "system", "content": "rules"}])
    mock_openai.return_value.chat.completions.create.assert_not_called()


@patch("agent.llm.OpenAI")
def test_chat_completion_no_structured_output_omits_response_format(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a", provider="deepseek")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.text == "x"

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs


@patch("agent.llm.OpenAI")
def test_constructor_builds_capabilities_from_args(mock_openai):
    client = LLMClient(
        "https://api.example.com/v1", "key", "model-a",
        provider="gemini", capability_overrides={"supports_json_schema": True},
    )
    assert client.capabilities.provider == "gemini"
    assert client.capabilities.supports_json_schema is True


@patch("agent.llm.OpenAI")
def test_constructor_defaults_to_unknown(mock_openai):
    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    assert client.capabilities.provider == "unknown"
    assert client.capabilities.supports_json_schema is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL on the changed/new tests (missing `provider`/`capability_overrides` constructor params, `capabilities` attribute, thinking gating, guard, negotiation).

- [ ] **Step 3: Write the implementation**

Modify `agent/llm.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from openai import OpenAI, OpenAIError

from .capabilities import ProviderCapabilities
from .negotiate import negotiate_structured_output


class LLMError(Exception):
    """Raised when an LLM API call fails."""


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_tokens: int = 0


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float | None = None,
                 provider: str = "", capability_overrides: dict | None = None):
        kwargs: dict = {"api_key": api_key, "base_url": base_url}
        if timeout is not None:
            kwargs["timeout"] = timeout
        self.client = OpenAI(**kwargs)
        self.model = model
        self.capabilities = ProviderCapabilities(
            provider=provider or "unknown", **capability_overrides or {}
        )

    def chat_completion(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        disable_thinking: bool = False,
        json_mode: bool = False,
        json_schema: dict | None = None,
    ) -> ChatResult:
        if not any(m.get("role") == "user" for m in messages):
            raise LLMError(
                "Every chat_completion call must include at least one user message "
                "(all supported providers require or expect a user turn)."
            )
        try:
            mode = negotiate_structured_output(
                self.capabilities, json_mode=json_mode, json_schema=json_schema
            )
            kwargs = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if mode == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": json_schema,
                        "strict": False,
                    },
                }
            elif mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
            if disable_thinking and self.capabilities.supports_thinking_toggle:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = self.client.chat.completions.create(**kwargs)
            u = resp.usage
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None)
            return ChatResult(
                text=resp.choices[0].message.content or "",
                model=resp.model or (model or self.model),
                prompt_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
                completion_tokens=getattr(u, "completion_tokens", 0) if u else 0,
                total_tokens=getattr(u, "total_tokens", 0) if u else 0,
                cache_tokens=cached if isinstance(cached, int) else 0,
            )
        except OpenAIError as e:
            raise LLMError(f"LLM API call failed: {e}") from e

    def chat_completion_stream(
        self, messages: list[dict], *, model: str | None = None, temperature: float = 0.7
    ) -> Iterator[str]:
        try:
            stream = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                choices = chunk.choices
                if choices:
                    content = choices[0].delta.content
                    if content:
                        yield content
        except OpenAIError as e:
            raise LLMError(f"LLM API call failed: {e}") from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS (all tests, including the unchanged constructor/stream/token tests)

- [ ] **Step 5: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat: LLMClient capability-gated negotiation and thinking"
```

---
### Task 5: Classification — schema intent, shared parser, question dedup

**Files:**
- Modify: `agent/classification.py`
- Test: `tests/test_classification.py`

**Interfaces:**
- Consumes: `parse_json` from `agent/parsing.py` (Task 3).
- Produces: `build_classification_prompt(name, description, intents, complexity=None)` (no `question` parameter); `ClassificationService.classify` calls `chat_completion(..., json_schema=schema)`.

- [ ] **Step 1: Update tests and add failing assertions**

In `tests/test_classification.py`:

1. In `test_classify_single_call_returns_all_fields` (lines 56-61), change the intent assertions (json_mode is no longer passed; json_schema is now the intent):

```python
    messages, model, disable_thinking, json_mode, json_schema = client.calls[0]
    assert model == "cm"
    assert disable_thinking is True
    assert json_mode is False
    assert json_schema is not None
    assert "intent" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "what is a pointer?"
    assert "what is a pointer?" not in messages[0]["content"]
```

2. In `test_classify_out_of_domain_accepts_null` (lines 81-82), change the two assertions:

```python
    assert client.calls[0][3] is False   # json_mode is not passed; intent is json_schema
    assert client.calls[0][4] is not None  # json_schema intent; client negotiates the mechanism
```

3. Update `build_classification_prompt` call sites (lines 174-198, 238-244) to drop the `question` argument:

```python
    prompt = build_classification_prompt(
        "SE",
        "software engineering",
        [_rich_intent()],
    )
```
```python
    prompt = build_classification_prompt(
        "SE",
        "software engineering",
        [IntentDef("faq", "quick factual question")],
    )
```
```python
    prompt = build_classification_prompt(
        "SE", "software engineering",
        [IntentDef("faq", "quick factual question")],
        complexity=_complexity_policy(),
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_classification.py -v`
Expected: FAIL (json_schema intent not passed; `build_classification_prompt` still requires `question`; system prompt still contains the question).

- [ ] **Step 3: Write the implementation**

In `agent/classification.py`:

1. Change the imports: remove `import re`; add `from .parsing import parse_json`.

2. Remove the `User question: {question}` line from `_CLASSIFICATION_PROMPT` (the template ends after the JSON-format line).

3. Change `build_classification_prompt` signature and format call:

```python
def build_classification_prompt(
    name: str,
    description: str,
    intents: list[IntentDef],
    complexity: ComplexityPolicy | None = None,
) -> str:
    lines: list[str] = []
    for idef in intents:
        header = f"- {idef.id}: {idef.description}"
        if not (idef.positive_examples or idef.negative_examples or idef.boundaries):
            lines.append(header)
            continue
        lines.append(header)
        if idef.positive_examples:
            lines.append("  Positive examples:")
            lines.extend(f"    - {ex}" for ex in idef.positive_examples)
        if idef.negative_examples:
            lines.append("  Negative examples:")
            lines.extend(f"    - {ex}" for ex in idef.negative_examples)
        for b in idef.boundaries:
            lines.append(f"  Boundary: {b}")
    intents_block = "\n".join(lines)
    return _CLASSIFICATION_PROMPT.format(
        name=name,
        description=description,
        intents=intents_block,
        complexity_section=build_complexity_section(complexity),
    )
```

4. Delete `_parse` (lines 111-119) and replace `classify` (the whole method body including the commented json_schema TODO block) with:

```python
    def classify(self, question: str, *, model: str | None = None) -> ClassificationResult:
        intent_ids = list(self.domain.intents)
        schema = build_classification_schema(intent_ids)
        prompt = build_classification_prompt(
            self.domain.name, self.domain.description,
            list(self.domain.intents.values()),
            complexity=self.domain.complexity,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        result = self.client.chat_completion(
            messages,
            model=model,
            disable_thinking=True,
            json_schema=schema,
        )
        return validate_classification(parse_json(result.text), intent_ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_classification.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/classification.py tests/test_classification.py
git commit -m "refactor: classification uses json_schema intent, shared parser, dedup question"
```

---
### Task 6: Planner — schema intent, shared parser, question dedup

**Files:**
- Modify: `agent/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `parse_json` from `agent/parsing.py` (Task 3).
- Produces: `Orchestrator._plan` calls `chat_completion(..., json_schema=_planner_schema())`; `_parse_json` is deleted.

- [ ] **Step 1: Update tests and add failing assertions**

In `tests/test_orchestrator.py`:

1. In `test_run_normal_path_planner_workers_aggregator` (lines 71-77), update planner assertions:

```python
    # planner call expresses json_schema intent; json_mode is not passed
    planner_messages, planner_model, planner_dt, planner_jm, planner_schema = client.calls[0]
    assert planner_schema is not None
    assert planner_jm is False
    assert planner_dt is True
    assert planner_messages[1]["role"] == "user"
    assert planner_messages[1]["content"] == "huge task"
    assert "huge task" not in planner_messages[0]["content"]
```

2. Replace `test_run_planner_uses_json_object_main_path` (lines 138-151) with a schema-intent test:

```python
def test_run_planner_uses_json_schema_intent():
    """The planner expresses json_schema intent; the client negotiates the mechanism."""
    client = FakeClient([
        '{"tasks": [{"title": "t1", "instruction": "i1", "role": "R1"}]}',
        "worker1 output",
        "final answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "final answer"
    planner_messages, planner_model, planner_dt, planner_jm, planner_schema = client.calls[0]
    assert planner_schema is not None
    assert planner_jm is False
    assert planner_dt is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL (planner still passes json_mode, schema is None).

- [ ] **Step 3: Write the implementation**

In `agent/orchestrator.py`:

1. Change the imports: remove `import json` and `import re`; add `from .parsing import parse_json`.

2. Delete `_parse_json` (lines 13-21).

3. Remove the `User question: {question}` line from `_PLANNER_PROMPT`.

4. In `_plan`, replace the `messages = [...]` line and the commented TODO block with:

```python
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        result = self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_schema=_planner_schema()
        )
        data = parse_json(result.text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/orchestrator.py tests/test_orchestrator.py
git commit -m "refactor: planner uses json_schema intent, shared parser, dedup question"
```

---
### Task 7: Judge — shared parser, question dedup

**Files:**
- Modify: `agent/evaluation/judge.py`
- Test: `tests/test_evaluation_judge.py`

**Interfaces:**
- Consumes: `parse_json` from `agent/parsing.py` (Task 3).
- Produces: `build_judge_prompt(answer: str, *, reference: str | None = None) -> str` (no `question` param); `parse_scorecard` reuses `parse_json`.

- [ ] **Step 1: Update tests and add failing assertions**

In `tests/test_evaluation_judge.py`:

1. Replace `test_build_judge_prompt_contains_question_answer_and_dimensions` (lines 24-30) with:

```python
def test_build_judge_prompt_contains_answer_and_dimensions():
    prompt = build_judge_prompt("the answer", reference="ground truth")
    assert "q?" not in prompt
    assert "the answer" in prompt
    assert "ground truth" in prompt
    for d in JUDGE_DIMENSIONS:
        assert d in prompt
```

2. In `test_judge_returns_scorecard` (lines 62-67), add the dedup assertion:

```python
    messages, model, dt, jm, schema = client.calls[0]
    assert model == "judge-a"
    assert dt is True
    assert jm is True
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "q?"
    assert "q?" not in messages[0]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation_judge.py -v`
Expected: FAIL (prompt still contains `q?`; build_judge_prompt still takes question).

- [ ] **Step 3: Write the implementation**

In `agent/evaluation/judge.py`:

1. Change imports: remove `import json` and `import re`; add `from agent.parsing import parse_json`.

2. Remove the `Question: {question}` line from `_JUDGE_PROMPT` (the template starts with the evaluator role line; drop the question line and following blank line).

3. Change `build_judge_prompt` signature and format call:

```python
def build_judge_prompt(answer: str, *, reference: str | None = None) -> str:
    reference_block = (
        f"\nGround truth reference:\n{reference}" if reference else "\nNo reference provided."
    )
    return _JUDGE_PROMPT.format(
        answer=answer,
        reference_block=reference_block,
    )
```

4. Replace `parse_scorecard` (lines 48-64) with:

```python
def parse_scorecard(text: str | None) -> dict | None:
    data = parse_json(text) if text else None
    if data is None:
        return None
    for dim in JUDGE_DIMENSIONS:
        value = data.get(dim)
        if not isinstance(value, int) or not 1 <= value <= 5:
            return None
    return data
```

5. Update `Judge.score` message construction and the prompt call:

```python
    def score(self, question: str, answer: str, *, reference: str | None = None) -> dict | None:
        prompt = build_judge_prompt(answer, reference=reference)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_judge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/evaluation/judge.py tests/test_evaluation_judge.py
git commit -m "refactor: judge uses shared parser and dedups question"
```

---
### Task 8: Config keys — remove disable_thinking, require provider capabilities

**Files:**
- Modify: `agent/config.py`
- Test: `tests/test_config.py`
- Modify: `tests/test_agent_cli.py`, `tests/test_evaluation_cli.py` (fixtures only)
- Modify: `config.example.json`
- Modify: `config.json` (git-ignored user config — not committed)

**Interfaces:**
- Consumes: `KNOWN_CAPABILITY_KEYS` (Task 1).
- Produces: `AgentConfig.provider: str = ""` and `AgentConfig.provider_capabilities: dict[str, bool]`; `load_config` requires both (validated) and raises `ConfigError` on any violation. `disable_thinking` is removed.

- [ ] **Step 1: Add the failing validation tests and update fixtures**

In `tests/test_config.py`:

1. Update the `_write_config` helper to inject the required pair by default (so existing minimal-config tests keep passing):

```python
def _write_config(tmp_path, data):
    data = {
        "provider": "test",
        "provider_capabilities": {},
        **data,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)
```

2. In `test_default_config_falls_back_to_example` (line 145) and `test_explicit_missing_path_no_fallback` (line 165), add the pair to the written `config.example.json`:

```python
    (tmp_path / "config.example.json").write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider": "test",
        "provider_capabilities": {},
    }), encoding="utf-8")
```

3. Replace `test_load_config_disable_thinking_default_true`, `test_load_config_disable_thinking_false_parsed`, and `test_load_config_disable_thinking_non_bool_defaults_true` (lines 783-815) with:

```python
def test_load_config_provider_and_capabilities_parsed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider": "gemini",
        "provider_capabilities": {"supports_json_schema": True},
    })
    cfg = load_config(path)
    assert cfg.provider == "gemini"
    assert cfg.provider_capabilities == {
        "supports_json_schema": True,
    }


def test_load_config_missing_provider_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider_capabilities": {},
    }), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_load_config_missing_capabilities_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider": "gemini",
    }), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_load_config_unknown_capability_key_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider": "gemini",
        "provider_capabilities": {"supports_magic": True},
    }), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_load_config_non_boolean_capability_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider": "gemini",
        "provider_capabilities": {"supports_json_schema": "yes"},
    }), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))
```

In `tests/test_agent_cli.py`, update `_write_root_config` to inject the pair:

```python
def _write_root_config(tmp_path, domain_dir):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://x/v1",
        "model": "m",
        "domain_dir": domain_dir,
        "provider": "test",
        "provider_capabilities": {},
    }), encoding="utf-8")
    return str(path)
```

In `tests/test_evaluation_cli.py`, update `_suite_cli_env` to inject the pair into the written `config.json` (lines 139-144):

```python
    config_path = config_dir / "config.json"
    config_path.write_text(
        '{"base_url": "https://x", "model": "m", "domain_dir": "%s", '
        '"provider": "test", "provider_capabilities": {}}'
        % domain_dir,
        encoding="utf-8",
    )
```

Also update the direct config write in `test_main_missing_dataset_returns_error` (lines 378-382) — strict validation would otherwise raise before the dataset check and break its `assert "Dataset error" in err`:

```python
    config_path.write_text(
        f'{{"base_url": "https://x", "model": "m", "domain_dir": "{domain_dir}", '
        f'"provider": "test", "provider_capabilities": {{}}}}',
        encoding="utf-8",
    )
```

In `config.example.json`, add the DeepSeek capability block (the example's `base_url` is DeepSeek):

```json
{
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "model_low": "",
  "model_high": "",
  "domain_dir": "domain/software_engineering",
  "provider": "deepseek",
  "provider_capabilities": {
    "supports_json_schema": false,
    "supports_thinking_toggle": true
  },
  "orchestrator": {
    "max_workers": 4,
    "worker_timeout": 120
  },
  "evaluation": {
    "judge_model": "deepseek-v4-pro",
    "results_dir": "evaluation/results"
  },
  "observability": {
    "enabled": true,
    "data_dir": ".observability",
    "phase_map": {}
  }
}
```

In the user's local `config.json` (git-ignored, not committed), add the matching block. Its `base_url` is Gemini, so use:

```json
  "provider": "gemini",
  "provider_capabilities": {
    "supports_json_schema": true,
    "supports_thinking_toggle": false
  },
```

Also remove the stale `"disable_thinking": false` line from it if present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k "provider or capability" -v`
Expected: FAIL with `AttributeError: 'AgentConfig' object has no attribute 'provider'`

- [ ] **Step 3: Write the implementation**

In `agent/config.py`:

1. Add the import: `from .capabilities import KNOWN_CAPABILITY_KEYS`.

2. In `AgentConfig`, remove `disable_thinking: bool = True`; add:

```python
    provider: str = ""
    provider_capabilities: dict[str, bool] = field(default_factory=dict)
```

3. In `load_config`, remove the disable_thinking parse block and add (after the `domain_dir` check, before building the return):

```python
    provider = raw.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ConfigError("Missing 'provider' in config (e.g. 'deepseek' or 'gemini').")

    raw_caps = raw.get("provider_capabilities")
    if not isinstance(raw_caps, dict):
        raise ConfigError(
            "Missing 'provider_capabilities' in config; declare the provider's capabilities."
        )
    for key, value in raw_caps.items():
        if key not in KNOWN_CAPABILITY_KEYS:
            raise ConfigError(
                f"Unknown capability '{key}' in provider_capabilities. "
                f"Known capabilities: {', '.join(KNOWN_CAPABILITY_KEYS)}."
            )
        if not isinstance(value, bool):
            raise ConfigError(f"Capability '{key}' must be a boolean in provider_capabilities.")
    provider_capabilities = dict(raw_caps)
```

4. In the `AgentConfig(...)` return, replace `disable_thinking=disable_thinking,` with `provider=provider, provider_capabilities=provider_capabilities,`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_agent_cli.py tests/test_evaluation_cli.py tests/test_integration.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/config.py tests/test_config.py tests/test_agent_cli.py tests/test_evaluation_cli.py config.example.json
git commit -m "feat: require provider capabilities in config with validation"
```

---
### Task 9: CLI wiring

**Files:**
- Modify: `agent/agent_cli.py`, `agent/evaluation/__main__.py`
- Test: `tests/test_agent_cli.py`, `tests/test_evaluation_cli.py`

**Interfaces:**
- Consumes: `AgentConfig.provider`, `AgentConfig.provider_capabilities` (Task 8).
- Produces: both CLIs construct `LLMClient(..., provider=config.provider, capability_overrides=config.provider_capabilities)`.

- [ ] **Step 1: Update the two wiring tests and add failing assertions**

In `tests/test_agent_cli.py`, replace `test_main_passes_disable_thinking_from_config` (lines 95-119) with:

```python
def test_main_passes_provider_and_capability_overrides_from_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_path = _write_root_config(tmp_path, _write_domain(tmp_path))
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    data["provider"] = "gemini"
    data["provider_capabilities"] = {"supports_json_schema": True}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    captured = {}

    class FakeClient:
        def chat_completion(
            self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
        ):
            return ChatResult(
                text='{"in_domain": false, "intent": null, "complexity": null, "reason": "x"}',
                model=model or "m",
            )

    monkeypatch.setattr(agent_cli, "LLMClient",
                        lambda *a, **k: captured.update(k) or FakeClient())
    assert agent_cli.main([str(config_path), "--ask", "hi"]) == 0
    assert captured["provider"] == "gemini"
    assert captured["capability_overrides"] == {"supports_json_schema": True}
```

In `tests/test_evaluation_cli.py`, replace `test_main_passes_disable_thinking_from_config` (lines 240-276) with:

```python
def test_main_passes_provider_and_capability_overrides_from_config(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    data["provider"] = "gemini"
    data["provider_capabilities"] = {"supports_json_schema": True}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    monkeypatch.setenv("AGENT_API_KEY", "k")

    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def chat_completion(self, messages, model=None, temperature=0.3,
                            disable_thinking=False, json_mode=False, json_schema=None):
            return ChatResult(
                text='{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
                model=model or "m",
            )

        def chat_completion_stream(self, messages, **kwargs):
            return iter([])

    monkeypatch.setattr(eval_main, "LLMClient", FakeClient)
    import io
    import sys

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main([
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--label", "pc", "--results-dir", str(tmp_path / "r"), "--skip-quality",
    ])
    assert rc == 0
    assert captured["provider"] == "gemini"
    assert captured["capability_overrides"] == {"supports_json_schema": True}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent_cli.py tests/test_evaluation_cli.py -v`
Expected: FAIL on the two wiring tests (CLIs still pass `disable_thinking`).

- [ ] **Step 3: Write the implementation**

In `agent/agent_cli.py`, change the `LLMClient(...)` call from:

```python
client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model,
                       timeout=effective_timeout(config),
                       disable_thinking=config.disable_thinking)
```

to:

```python
client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model,
                       timeout=effective_timeout(config),
                       provider=config.provider,
                       capability_overrides=config.provider_capabilities)
```

In `agent/evaluation/__main__.py`, apply the identical change to its `LLMClient(...)` call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent_cli.py tests/test_evaluation_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/agent_cli.py agent/evaluation/__main__.py tests/test_agent_cli.py tests/test_evaluation_cli.py
git commit -m "feat: wire provider identity and capability overrides into CLIs"
```

---
### Task 10: Regression and live verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest -q`
Expected: PASS (all tests; the `disable_thinking` references in the config tests and CLI wiring tests are gone).

- [ ] **Step 2: Confirm no stale references**

Run: `rg -n "disable_thinking" agent/ tests/ config.example.json`
Expected: matches only for the per-call parameter — `agent/llm.py` (parameter + `extra_body` gating), `agent/classification.py`, `agent/orchestrator.py`, `agent/evaluation/judge.py` (per-call `disable_thinking=True`), `tests/test_llm.py`, and the fake-client signatures in `tests/test_agent_cli.py` / `tests/test_evaluation_cli.py`. No `config.disable_thinking`, no `AgentConfig.disable_thinking`, and no `"disable_thinking"` in `config.example.json`.

- [ ] **Step 3: Live Gemini smoke test**

Run (with `AGENT_API_KEY` available, e.g. sourced from the user's key file):

```bash
set -a; source /tmp/ef_api_key.sh; set +a
uv run pytest tests/test_smoke.py -v
```

Expected: 3 passed (single-question answer, out-of-domain reject, evaluation run writes a result). Gemini is schema-capable and requires a user message — both verified live.

- [ ] **Step 4: Final commit if any cleanup surfaced**

```bash
git add -A
git commit -m "chore: cleanup after provider capability refactor"
```

(Only run this step if Step 2 or Step 3 surfaced a stray change.)

---
## Self-Review Notes

**Spec coverage check:**
- ProviderCapabilities + `KNOWN_CAPABILITY_KEYS` (config-declared, no code-side defaults) → Task 1; required `provider`/`provider_capabilities` with strict validation → Task 8.
- Negotiation priority chain → Task 2, wired in Task 4.
- Pure-JSON parsing (json.loads only; unparseable → error) → Task 3; adopted at all call sites in Tasks 5-7.
- thinking gating + config key removal → Task 4 (gating), Task 8 (config), Task 9 (wiring).
- Question dedup → Tasks 5-7.
- Unconditional user-message guard → Task 4.
- Non-goals honored: no tool-call flow, no multi-SDK backends, no probing, no `_iter_json_objects`/brace-matching fallback.

**Path deviation from spec:** the spec's `agent/llm/capabilities.py` paths were illustrative; `agent/llm.py` is a module, so the new modules live at `agent/capabilities.py`, `agent/negotiate.py`, `agent/parsing.py`.