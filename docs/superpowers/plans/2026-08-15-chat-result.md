# ChatResult Return Value Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `LLMClient.chat_completion` return a rich `ChatResult` (text + actual model + token usage) and remove the private `_usage_local` thread-local, so `RecordingClient` and `TracedLLMClient` read usage from the return value instead of poking a private attribute.

**Architecture:** A `ChatResult` dataclass carries `text`, `model` (from `resp.model`), and token counts. Production callers consume `.text`. The two wrappers pass the `ChatResult` through unchanged while recording fields from it. The thread-local in `LLMClient` is deleted.

**Tech Stack:** Python 3, `openai` SDK, pytest.

## Global Constraints

- `ChatResult` fields: `text: str`, `model: str`, `prompt_tokens: int = 0`, `completion_tokens: int = 0`, `total_tokens: int = 0`, `cache_tokens: int = 0` (spec `2026-08-15-chat-result-design.md`).
- `model` = `resp.model or (model or self.model)`.
- Tokens default to 0 when `resp.usage` is absent; `cache_tokens` = `details.cached_tokens` when an int, else 0.
- `chat_completion` still raises `LLMError` on `OpenAIError` (unchanged).
- `chat_completion_stream` is unchanged and not observed.
- No new dependencies.
- The full suite is only green after Task 5; intermediate tasks keep their own module's tests green (the return-type change is atomic and breaks unrelated modules' fakes until updated).

---

### Task 1: ChatResult in LLMClient

**Files:**
- Modify: `agent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ChatResult` dataclass and `LLMClient.chat_completion(...) -> ChatResult`. `LLMClient` loses `_usage_local`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm.py`:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_returns_chat_result(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "你好"
    resp.model = "model-a"
    resp.usage = _usage(10, 5)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])

    assert result.text == "你好"
    assert result.model == "model-a"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15
    assert result.cache_tokens == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py::test_chat_completion_returns_chat_result -v`
Expected: FAIL with `AttributeError: 'str' object has no attribute 'text'`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `agent/llm.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from openai import OpenAI, OpenAIError


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
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

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
        try:
            kwargs = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if json_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification_result",
                        "schema": json_schema,
                        "strict": False,
                    },
                }
            elif json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if disable_thinking:
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

- [ ] **Step 4: Update the remaining `test_llm.py` assertions**

Replace `test_chat_completion_returns_content` (line ~19) with:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_returns_content(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "你好"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])

    assert result.text == "你好"
    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "model-a"
    assert kwargs["stream"] is False
    assert "extra_body" not in kwargs
```

Replace `test_chat_completion_disable_thinking_passes_extra_body` — only the mock needs `resp.model = "model-a"` and `resp.usage = None`:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_disable_thinking_passes_extra_body(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}], disable_thinking=True)

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
```

Replace `test_chat_completion_none_content_returns_empty_string`:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_none_content_returns_empty_string(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = None
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.text == ""
```

Replace `test_chat_completion_json_mode_passes_response_format`, `test_chat_completion_json_mode_with_disable_thinking`, `test_chat_completion_json_mode_off_by_default`, `test_chat_completion_json_schema_passes_response_format`, `test_chat_completion_json_schema_wins_over_json_mode` — in each, add `resp.model = "model-a"` and `resp.usage = None` to the mock (the assertions on `kwargs` stay unchanged).

Replace `test_chat_completion_records_thread_local_usage` and `test_usage_isolated_across_threads` with a single model-field test:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_model_falls_back_to_requested(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = None
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}], model="low-a")
    assert result.model == "low-a"
```

Replace `test_chat_completion_returns_text_unaffected`:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_returns_text_unaffected(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "你好"
    resp.model = "model-a"
    resp.usage = _usage(3, 4)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.text == "你好"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 4
```

Replace `test_chat_completion_records_cache_tokens` and `test_chat_completion_cache_tokens_zero_when_absent`:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_records_cache_tokens(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = _usage_with_cache(10, 5, cached=7)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.cache_tokens == 7


@patch("agent.llm.OpenAI")
def test_chat_completion_cache_tokens_zero_when_absent(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = _usage(10, 5)  # no prompt_tokens_details
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.cache_tokens == 0
```

Add `resp.model = "model-a"` and `resp.usage = None` to any remaining mocks in this file that set only `content`. Remove the now-unused `import threading` if no test references it.

- [ ] **Step 5: Run test_llm.py to verify it passes**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat: chat_completion returns ChatResult with usage and actual model"
```

---

### Task 2: Wrappers consume ChatResult

**Files:**
- Modify: `agent/evaluation/runner.py` (`RecordingClient`)
- Modify: `agent/observability/client.py` (`TracedLLMClient`)
- Test: `tests/test_evaluation_runner.py` (RecordingClient test only)
- Test: `tests/test_observability_client.py`

**Interfaces:**
- Consumes: `ChatResult` from `agent.llm` (Task 1).
- Produces: `RecordingClient.chat_completion(...) -> ChatResult` (pass-through), `TracedLLMClient.chat_completion(...) -> ChatResult` (pass-through).

- [ ] **Step 1: Write the failing test**

In `tests/test_evaluation_runner.py`, the `FakeClient` must return `ChatResult` with usage. Replace the class and the RecordingClient test:

```python
from agent.llm import ChatResult


class FakeClient:
    def __init__(self, responses, usage=None):
        self.responses = list(responses)
        self.models = []
        self.json_modes = []
        self.usage_queue = list(usage or [])

    def chat_completion(self, messages, model=None, temperature=0.3,
                        disable_thinking=False, json_mode=False, json_schema=None):
        self.models.append(model)
        self.json_modes.append(json_mode)
        prompt = completion = cached = 0
        if self.usage_queue:
            prompt, completion, cached = self.usage_queue.pop(0)
        return ChatResult(
            text=self.responses.pop(0),
            model=model or "m",
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cache_tokens=cached,
        )

    def _record_usage(self, prompt, completion, cached=0):
        """Set the usage seen by the NEXT chat_completion call."""
        self.usage_queue.append((prompt, completion, cached))


def test_recording_client_records_usage_and_latency():
    inner = FakeClient(["hello"], usage=[(10, 5, 3)])
    rc = RecordingClient(inner)
    out = rc.chat_completion([{"role": "user", "content": "hi"}], model="m2")
    assert out.text == "hello"
    assert rc.calls[0]["model"] == "m2"
    assert rc.calls[0]["prompt_tokens"] == 10
    assert rc.calls[0]["completion_tokens"] == 5
    assert rc.calls[0]["total_tokens"] == 15
    assert rc.calls[0]["cache_tokens"] == 3
    assert rc.calls[0]["latency_ms"] >= 0
    rc.reset()
    assert rc.calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_runner.py::test_recording_client_records_usage_and_latency -v`
Expected: FAIL (`'str' object has no attribute 'text'`, since RecordingClient returns raw str).

- [ ] **Step 3: Update `RecordingClient`**

Replace `agent/evaluation/runner.py` lines 16-59:

```python
from agent.llm import ChatResult, LLMClient


class RecordingClient:
    """Thin LLMClient wrapper that records per-call usage and latency.

    Reads token usage from the returned ChatResult; completely independent
    of observability.
    """

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.calls: list[dict] = []

    @property
    def model(self) -> str:
        return self._inner.model

    def reset(self) -> None:
        self.calls = []

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs) -> ChatResult:
        started = time.perf_counter()
        result = self._inner.chat_completion(
            messages, model=model, temperature=temperature, **kwargs
        )
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        self.calls.append({
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "cache_tokens": result.cache_tokens,
            "latency_ms": elapsed,
        })
        return result

    def chat_completion_stream(self, messages, *, model=None, temperature=0.7, **kwargs):
        yield from self._inner.chat_completion_stream(
            messages, model=model, temperature=temperature, **kwargs
        )
```

- [ ] **Step 4: Run RecordingClient test to verify it passes**

Run: `uv run pytest tests/test_evaluation_runner.py::test_recording_client_records_usage_and_latency -v`
Expected: PASS

- [ ] **Step 5: Write the failing TracedLLMClient test**

In `tests/test_observability_client.py`, replace `FakeInner` and the usage assertion tests:

```python
from agent.llm import ChatResult


class FakeInner:
    def __init__(self, usage=None, error=None, model="m"):
        self.model = model
        self._usage = usage
        self._error = error
        self.last_kwargs = None

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        self.last_kwargs = (model, messages)
        if self._error:
            raise self._error
        u = self._usage
        return ChatResult(
            text="the answer",
            model=model or self.model,
            prompt_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
            completion_tokens=getattr(u, "completion_tokens", 0) if u else 0,
            total_tokens=getattr(u, "total_tokens", 0) if u else 0,
            cache_tokens=0,
        )

    def chat_completion_stream(self, messages, **kwargs):
        for chunk in ["a", "b"]:
            yield chunk
```

In `test_records_ok_call_with_usage`, change the assertion to read from the pass-through result:

```python
def test_records_ok_call_with_usage(tmp_path):
    store = _make_store(tmp_path)
    inner = FakeInner(usage=_Usage(10, 5))
    traced = TracedLLMClient(inner, store)
    with trace_span() as tid, phase("classification"):
        result = traced.chat_completion([{"role": "user", "content": "hi"}], model="low-a")

    assert result.text == "the answer"
    assert inner.last_kwargs[0] == "low-a"
    calls = store.trace_llm_calls(tid)
    assert len(calls) == 1
    ev = calls[0]
    assert ev["status"] == "ok"
    assert ev["phase"] == "classification"
    assert ev["model"] == "low-a"
    assert ev["prompt_tokens"] == 10
    assert ev["completion_tokens"] == 5
    assert ev["total_tokens"] == 15
    assert ev["latency_ms"] >= 0
```

Replace `test_records_non_streaming_usage_is_none_when_missing` (usage now defaults to 0):

```python
def test_records_usage_zero_when_missing(tmp_path):
    store = _make_store(tmp_path)
    inner = FakeInner(usage=None)
    traced = TracedLLMClient(inner, store)
    with trace_span() as tid:
        traced.chat_completion([{"role": "user", "content": "hi"}])
    ev = store.trace_llm_calls(tid)[0]
    assert ev["prompt_tokens"] == 0
```

Remove the now-unused `import threading` in `tests/test_observability_client.py` if no longer referenced.

- [ ] **Step 6: Run TracedLLMClient test to verify it fails**

Run: `uv run pytest tests/test_observability_client.py::test_records_ok_call_with_usage -v`
Expected: FAIL (`'str' object has no attribute 'text'` on the pass-through result).

- [ ] **Step 7: Update `TracedLLMClient`**

Replace `agent/observability/client.py` lines 12-65:

```python
from agent.llm import ChatResult, LLMError

from .tracing import TraceStore, current_phase, current_trace_id, now_millis


class TracedLLMClient:
    """Transparent LLMClient wrapper that records token usage and latency."""

    def __init__(self, inner, store: TraceStore):
        self._inner = inner
        self._store = store

    @property
    def model(self) -> str:
        return self._inner.model

    def _base_event(self) -> dict:
        return {
            "type": "llm_call",
            "trace_id": current_trace_id(),
            "phase": current_phase(),
            "ts": now_millis(),
        }

    def _write(self, event: dict) -> None:
        try:
            self._store.write(event)
        except Exception as e:  # noqa: BLE001 - degrade, never break business
            warnings.warn(f"observability: failed to record llm_call: {e}")

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs) -> ChatResult:
        started = time.perf_counter()
        try:
            result = self._inner.chat_completion(
                messages, model=model, temperature=temperature, **kwargs
            )
        except LLMError as e:
            ev = self._base_event()
            ev.update({"model": model or self._inner.model, "prompt_tokens": None,
                       "completion_tokens": None, "total_tokens": None,
                       "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                       "status": "error", "error": str(e)})
            self._write(ev)
            raise
        ev = self._base_event()
        ev.update({"model": result.model, "prompt_tokens": result.prompt_tokens,
                   "completion_tokens": result.completion_tokens,
                   "total_tokens": result.total_tokens,
                   "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                   "status": "ok", "error": None})
        self._write(ev)
        return result

    def chat_completion_stream(self, messages, *, model=None, temperature=0.7, **kwargs) -> Iterator[str]:
        # Not observed in v1: the main flow never uses streaming.
        yield from self._inner.chat_completion_stream(
            messages, model=model, temperature=temperature, **kwargs
        )
```

- [ ] **Step 8: Run observability tests to verify they pass**

Run: `uv run pytest tests/test_observability_client.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add agent/evaluation/runner.py agent/observability/client.py \
  tests/test_evaluation_runner.py tests/test_observability_client.py
git commit -m "feat: wrappers read usage from ChatResult"
```

---

### Task 3: Production callers consume ChatResult

**Files:**
- Modify: `agent/classification.py:164-169`
- Modify: `agent/orchestrator.py:119,145,174,181`
- Modify: `agent/strategy.py:29`
- Modify: `agent/evaluation/judge.py:76`
- Test: `tests/test_classification.py`, `tests/test_orchestrator.py`, `tests/test_strategy.py`, `tests/test_evaluation_judge.py`

**Interfaces:**
- Consumes: `ChatResult` from `agent.llm` (Task 1).
- Produces: unchanged external behavior — callers that previously got a `str` still get a `str`.

- [ ] **Step 1: Update production callers**

`agent/classification.py:164-170`:

```python
        result = self.client.chat_completion(
            messages,
            model=model,
            disable_thinking=True,
            json_mode=True,
        )
        return validate_classification(_parse(result.text), intent_ids)
```

`agent/orchestrator.py:119-122`:

```python
        result = self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_mode=True
        )
        data = _parse_json(result.text)
```

`agent/orchestrator.py:145` (`_worker`), `:174` (`_aggregate`), `:181` (`_direct_answer`) — each return becomes:

```python
        return self.client.chat_completion(messages, model=model, disable_thinking=True).text
```

`agent/strategy.py:29` (`process`):

```python
    def process(self, client, question: str, history: list[tuple[str, str]], *, model: str | None = None) -> str:
        return client.chat_completion(self.build_messages(history, question), model=model).text
```

`agent/evaluation/judge.py:76-84`:

```python
        try:
            result = self.client.chat_completion(
                messages,
                model=self.model,
                disable_thinking=True,
                json_mode=True,
            )
        except LLMError:
            return None
        return parse_scorecard(result.text)
```

- [ ] **Step 2: Update the module fakes to return ChatResult**

In each test file, import `ChatResult` from `agent.llm` and wrap the returned string. Keep any recorded state (`.calls`, `.models`, `.json_modes`) unchanged.

`tests/test_classification.py` FakeClient:

```python
from agent.llm import ChatResult


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
        return ChatResult(text=self.responses.pop(0), model=model or "m")
```

`tests/test_orchestrator.py` FakeClient:

```python
from agent.llm import ChatResult


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
        return ChatResult(text=self.responses.pop(0), model=model or "m")
```

`tests/test_strategy.py` FakeClient:

```python
from agent.llm import ChatResult


class FakeClient:
    def __init__(self, text="answer"):
        self.text = text
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False):
        self.calls.append((messages, model))
        return ChatResult(text=self.text, model=model or "m")
```

`tests/test_evaluation_judge.py` FakeClient:

```python
from agent.llm import ChatResult


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False,
                        json_mode=False, json_schema=None):
        self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
        if self.error is not None:
            raise self.error
        return ChatResult(text=self.response, model=model or "m")
```

- [ ] **Step 3: Run the four module test files**

Run: `uv run pytest tests/test_classification.py tests/test_orchestrator.py tests/test_strategy.py tests/test_evaluation_judge.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add agent/classification.py agent/orchestrator.py agent/strategy.py \
  agent/evaluation/judge.py tests/test_classification.py tests/test_orchestrator.py \
  tests/test_strategy.py tests/test_evaluation_judge.py
git commit -m "feat: production callers consume ChatResult.text"
```

---

### Task 4: Remaining fakes across consumers

**Files:**
- Modify: `tests/test_router.py`, `tests/test_chat.py`, `tests/test_repl.py`, `tests/test_agent_cli.py`, `tests/test_domain_agnostic.py`, `tests/test_evaluation_cli.py`, `tests/test_observability_install.py`, `tests/test_observability_patch.py`, `tests/test_evaluation_runner.py` (run_evaluation tests, already updated in Task 2 for FakeClient)
- Test: the same files

**Interfaces:**
- Consumes: `ChatResult` from `agent.llm` (Task 1).
- Produces: nothing new.

- [ ] **Step 1: Update each fake to return ChatResult**

Add `from agent.llm import ChatResult` to each file and wrap the returned string. Keep all recorded state.

`tests/test_router.py` FakeClient:

```python
from agent.llm import ChatResult


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completion(
        self,
        messages,
        model=None,
        disable_thinking=False,
        json_mode=False,
        json_schema=None,
    ):
        return ChatResult(text=self.responses.pop(0), model=model or "m")
```

`tests/test_chat.py` FakeClient:

```python
from agent.llm import ChatResult


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.models = []

    def chat_completion(
        self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
    ):
        self.models.append(model)
        return ChatResult(text=self.responses.pop(0), model=model or "m")
```

`tests/test_repl.py` FakeClient:

```python
from agent.llm import ChatResult


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completion(
        self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
    ):
        return ChatResult(text=self.responses.pop(0), model=model or "m")
```

`tests/test_agent_cli.py` — both `FakeClient` classes (lines ~59 and ~81):

```python
from agent.llm import ChatResult


class FakeClient:  # first occurrence (test_main_launches_repl)
    def chat_completion(
        self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
    ):
        return ChatResult(
            text='{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
            model=model or "m",
        )


class FakeClient:  # second occurrence (test_main_ask_prints_answer)
    def __init__(self, *a, **k):
        self.responses = [
            '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
            "one-shot answer",
        ]

    def chat_completion(
        self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
    ):
        return ChatResult(text=self.responses.pop(0), model=model or "m")
```

`tests/test_domain_agnostic.py` FakeClient:

```python
from agent.llm import ChatResult


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        self.calls.append(messages)
        return ChatResult(text=self.responses.pop(0), model=model or "m")
```

`tests/test_evaluation_cli.py` FakeClient:

```python
from agent.llm import ChatResult


class FakeClient:
    def __init__(self, *args, **kwargs):
        self._usage_local = __import__("threading").local()

    def chat_completion(self, messages, model=None, temperature=0.3,
                        disable_thinking=False, json_mode=False, json_schema=None):
        self._usage_local.usage = None
        return ChatResult(
            text='{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
            model=model or "m",
        )

    def chat_completion_stream(self, messages, **kwargs):
        return iter([])
```

`tests/test_observability_install.py` FakeClient:

```python
from agent.llm import ChatResult


class FakeClient:
    model = "m"

    def __init__(self, responses):
        self._responses = list(responses)

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        return ChatResult(text=self._responses.pop(0), model=model or self.model)

    def chat_completion_stream(self, messages, **kwargs):
        return iter([])
```

`tests/test_observability_patch.py` FakeInner:

```python
from agent.llm import ChatResult


class FakeInner:
    def __init__(self, responses):
        self._responses = list(responses)
        self.seen_phases = []

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        self.seen_phases.append(current_phase())
        return ChatResult(text=self._responses.pop(0), model=model or "m")

    def chat_completion_stream(self, messages, **kwargs):
        return iter([])
```

Remove `import threading` from `tests/test_observability_install.py` and `tests/test_observability_patch.py` if no longer referenced.

- [ ] **Step 2: Run all affected test files**

Run: `uv run pytest tests/test_router.py tests/test_chat.py tests/test_repl.py tests/test_agent_cli.py tests/test_domain_agnostic.py tests/test_evaluation_cli.py tests/test_observability_install.py tests/test_observability_patch.py tests/test_evaluation_runner.py -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_router.py tests/test_chat.py tests/test_repl.py tests/test_agent_cli.py \
  tests/test_domain_agnostic.py tests/test_evaluation_cli.py tests/test_observability_install.py \
  tests/test_observability_patch.py tests/test_evaluation_runner.py
git commit -m "test: fakes return ChatResult"
```

---

### Task 5: Full regression and docs

**Files:**
- Modify: `docs/superpowers/specs/2026-08-15-chat-result-design.md` (mark implementation notes accurate — no changes expected)
- Verify: full suite

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Full regression**

Run: `uv run pytest -q`
Expected: PASS (all 192 + any new tests, 5 skipped, 4 warnings).

- [ ] **Step 2: Confirm no lingering `_usage_local` references in agent/**

Run: `rg -n "_usage_local" agent/`
Expected: no matches.

- [ ] **Step 3: Commit (if anything changed)**

```bash
git add -A
git commit -m "chore: verify ChatResult refactor (full suite green)" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- `ChatResult` with text/model/tokens/cache → Task 1 ✅
- `chat_completion` returns `ChatResult`, `model = resp.model or (model or self.model)`, tokens default 0, cache int-only → Task 1 ✅
- Remove `_usage_local` → Task 1 (implementation) + Task 5 Step 2 (verification) ✅
- `RecordingClient` reads return value → Task 2 ✅
- `TracedLLMClient` reads return value → Task 2 ✅
- Callers (`classification`, `orchestrator`, `strategy`, `judge`) use `.text` → Task 3 ✅
- Remaining fakes → Task 4 ✅
- `chat_completion_stream` unchanged / not observed → Task 1 (implementation unchanged) ✅
- Observability error path keeps `prompt_tokens: None` → Task 2 ✅

**Placeholder scan:** No TBD/TODO; every step has concrete code and expected output.

**Type consistency:** `ChatResult.text`, `.model`, `.prompt_tokens`, `.completion_tokens`, `.total_tokens`, `.cache_tokens` used identically across all tasks. `RecordingClient.chat_completion` and `TracedLLMClient.chat_completion` both return `ChatResult` (Task 2), matching consumer `.text` (Task 3).