# Observability Plugin (Token + Trace) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable, non-invasive observability plugin that automatically records LLM token usage, per-phase latency, and key pipeline results (classification/route/strategy/orchestration) to per-day JSONL files, with CLI table, HTML report, and real-time terminal display.

**Architecture:** `agent/llm.py` gains a thread-local `_usage_local` (2 lines, zero business churn). A new `agent/observability/` package provides `TraceStore` (JSONL) + `contextvars` span stack, a `TracedLLMClient` wrapper, class-level method wrapping in `patch.py`, and reporters in `report.py`. The only wiring point is `agent_cli.py` via `install()`; when `observability.enabled` is false, `install()` returns the client untouched.

**Tech Stack:** Python 3.10+, stdlib only for the plugin (`contextvars`, `threading`, `uuid`, `json`, `pathlib`), pyyaml/pytest/uv already in use.

## Global Constraints

- Business modules `chat.py`, `router.py`, `classification.py`, `strategy.py`, `orchestrator.py`, `repl.py` are **NOT modified** except `agent/llm.py` which gains exactly two lines: `self._usage_local = threading.local()` in `__init__` and `self._usage_local.usage = resp.usage` before the return of `chat_completion`.
- When `observability.enabled` is false/missing, `install()` returns `(client, None)` and behavior is identical to today. Existing tests must stay green.
- The observability layer must **never raise into business code**: every store write, patch application, and report read failure degrades to `warnings.warn` and continues.
- `uv run pytest -q` must pass after every task (currently 84 tests).
- `AgentConfig` gains `observability: ObservabilityConfig | None = None` — a keyword field with a default, so all existing `AgentConfig(...)` constructor calls keep working.
- No new third-party dependencies.

---

### Task 1: `LLMClient` thread-local usage recording

**Files:**
- Modify: `agent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: existing `LLMClient` (constructor `(base_url, api_key, model, timeout=60.0)`); `resp.usage` from the openai SDK response.
- Produces: `LLMClient._usage_local` — a `threading.local()` whose `.usage` attribute holds the most recent non-streaming call's `resp.usage` (or nothing). `chat_completion` still returns `str`. TracedLLMClient reads `_usage_local.usage` after each call from the same thread.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_llm.py`:

```python
import threading
from unittest.mock import MagicMock, patch

from agent.llm import LLMClient


def _usage(prompt, completion):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = prompt + completion
    return u


@patch("agent.llm.OpenAI")
def test_chat_completion_records_thread_local_usage(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.usage = _usage(10, 5)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}])

    assert client._usage_local.usage.prompt_tokens == 10
    assert client._usage_local.usage.completion_tokens == 5


@patch("agent.llm.OpenAI")
def test_usage_isolated_across_threads(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.usage = _usage(10, 5)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}])
    assert client._usage_local.usage.prompt_tokens == 10

    seen = {}

    def read_in_thread():
        seen["fresh_has_usage"] = hasattr(client._usage_local, "usage")
        client._usage_local.usage = _usage(99, 1)

    t = threading.Thread(target=read_in_thread)
    t.start()
    t.join()

    # A fresh thread has its own thread-local slot: it sees no usage set by the
    # main thread, and writes in the worker thread never leak back.
    assert seen["fresh_has_usage"] is False
    assert client._usage_local.usage.prompt_tokens == 10


@patch("agent.llm.OpenAI")
def test_chat_completion_returns_text_unaffected(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "你好"
    resp.usage = _usage(3, 4)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    text = client.chat_completion([{"role": "user", "content": "hi"}])

    assert text == "你好"
```

Note: these tests must be appended to `tests/test_llm.py` without duplicating its existing imports; the file already imports `MagicMock`, `patch`, `pytest`, `OpenAIError`, `LLMClient`, `LLMError`. `threading` and the `_usage` helper are the only additions. (If `_usage` collides with an existing name, prefix it `_llm_usage` and update the three call sites.)

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -q`
Expected: FAIL — `AttributeError: 'LLMClient' object has no attribute '_usage_local'`

- [ ] **Step 3: Implement** — in `agent/llm.py`:

```python
from __future__ import annotations

import threading
from typing import Iterator

from openai import OpenAI, OpenAIError


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self._usage_local = threading.local()
```

And in `chat_completion`, immediately before `return content or ""`:

```python
            self._usage_local.usage = resp.usage
```

Do NOT import `threading` inside the method; add it at module top as shown.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -q`
Expected: PASS (all tests, including pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat: record thread-local last_usage on LLMClient for token observability"
```

---

### Task 2: `ObservabilityConfig` parsing in `agent/config.py`

**Files:**
- Modify: `agent/config.py`
- Modify: `config.example.json`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `AgentConfig` dataclass (existing constructor sites pass keyword args).
- Produces:
  - `ObservabilityConfig` dataclass: `enabled: bool = False`, `data_dir: str = ".observability"`, `phase_map: dict[str, str] = field(default_factory=dict)`.
  - `AgentConfig` gains field `observability: ObservabilityConfig | None = None` (default keeps all existing constructions valid).
  - `load_config(path=None)` populates `observability=None` when the config JSON omits or disables the key; else a populated `ObservabilityConfig`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_config.py` (this file already imports `json`, `load_config`, and defines a `_write_config(tmp_path, data)` helper returning the config path — reuse it):

```python
def test_load_config_without_observability(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.observability is None


def test_load_config_observability_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "observability": {"enabled": False, "data_dir": "obs/"},
    })
    cfg = load_config(path)
    assert cfg.observability is not None
    assert cfg.observability.enabled is False
    assert cfg.observability.data_dir == "obs/"


def test_load_config_observability_enabled_with_phase_map(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "observability": {
            "enabled": True,
            "phase_map": {"Orchestrator._worker": "work"},
        },
    })
    cfg = load_config(path)
    assert cfg.observability.enabled is True
    assert cfg.observability.data_dir == ".observability"          # default
    assert cfg.observability.phase_map == {"Orchestrator._worker": "work"}


def test_load_config_observability_ignores_non_dict(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "observability": "nope",
    })
    cfg = load_config(path)
    assert cfg.observability is None
```

Check the existing `_write_config` in `tests/test_config.py` (defined at the top of the file) and the file's `monkeypatch` usage patterns before appending.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL — `AttributeError: 'AgentConfig' object has no attribute 'observability'`

- [ ] **Step 3: Implement** — `agent/config.py`:

```python
from dataclasses import dataclass, field
```

Add near `AgentConfig`:

```python
@dataclass
class ObservabilityConfig:
    enabled: bool = False
    data_dir: str = ".observability"
    phase_map: dict[str, str] = field(default_factory=dict)
```

Add the field to `AgentConfig`:

```python
@dataclass
class AgentConfig:
    base_url: str
    model: str
    classifier_model: str
    domain_dir: str
    model_low: str | None = None
    model_high: str | None = None
    observability: ObservabilityConfig | None = None
```

In `load_config`, after `domain_dir` validation, before `return AgentConfig(...)`:

```python
    raw_obs = raw.get("observability")
    observability = None
    if isinstance(raw_obs, dict) and raw_obs.get("enabled"):
        data_dir = raw_obs.get("data_dir") or ".observability"
        phase_map = raw_obs.get("phase_map")
        observability = ObservabilityConfig(
            enabled=True,
            data_dir=data_dir if isinstance(data_dir, str) else ".observability",
            phase_map=phase_map if isinstance(phase_map, dict) else {},
        )
```

Add `observability=observability` to the `AgentConfig(...)` return. Also add the optional `observability` block to `config.example.json`:

```json
{
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "model_low": "",
  "model_high": "",
  "domain_dir": "domain/software_engineering",
  "observability": {
    "enabled": false,
    "data_dir": ".observability",
    "phase_map": {}
  }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS. Then run the full suite: `uv run pytest -q` — all green (existing `AgentConfig(...)` call sites unaffected because the new field has a default).

- [ ] **Step 5: Commit**

```bash
git add agent/config.py config.example.json tests/test_config.py
git commit -m "feat: parse optional observability config block"
```

---

### Task 3: `agent/observability/tracing.py` — TraceStore + span stack

**Files:**
- Create: `agent/observability/__init__.py`
- Create: `agent/observability/tracing.py`
- Test: `tests/test_tracing.py`

**Interfaces:**
- Consumes: nothing from the app besides stdlib.
- Produces:
  - `class TraceStore: __init__(self, data_dir: str | Path); write(self, event: dict) -> None; close(self) -> None; trace_llm_calls(self, trace_id: str) -> list[dict]`. `write` never raises (wraps OSError in `warnings.warn`). File per day: `{data_dir}/trace-YYYY-MM-DD.jsonl`, appended with `ensure_ascii=False`.
  - `read_events(data_dir: str | Path, *, day: str | None = None) -> tuple[list[dict], int]` — returns `(valid_events, bad_line_count)`. When `day` given, only reads `trace-{day}.jsonl`.
  - `trace_span() -> Iterator[str]` context manager — pushes a fresh span (new `trace_id = uuid.uuid4().hex[:12]`), yields the id, pops on exit.
  - `phase(name: str) -> Iterator[None]` context manager — pushes phase onto the current span; no-op when no span is active.
  - `current_trace_id() -> str | None`, `current_phase() -> str | None`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_tracing.py`:

```python
import json
import threading
import warnings
from pathlib import Path

from agent.observability.tracing import (
    TraceStore,
    current_phase,
    current_trace_id,
    phase,
    read_events,
    trace_span,
)


def _write_snapshot(tmp_path, lines):
    day_dir = tmp_path / "observability"
    day_dir.mkdir(parents=True, exist_ok=True)
    p = day_dir / "trace-2026-08-11.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return day_dir


def test_read_events_ok_and_bad_lines(tmp_path):
    day_dir = _write_snapshot(tmp_path, [
        json.dumps({"type": "llm_call", "trace_id": "a", "phase": "classification"}),
        "{this is not json",
        json.dumps({"type": "llm_call", "trace_id": "b", "phase": "route"}),
    ])
    events, bad = read_events(day_dir, day="2026-08-11")
    assert bad == 1
    assert [e["phase"] for e in events] == ["classification", "route"]


def test_read_events_nonexistent_day_returns_empty(tmp_path):
    events, bad = read_events(tmp_path / "observability", day="2099-01-01")
    assert events == []
    assert bad == 0


def test_trace_store_writes_and_reads_back(tmp_path):
    store = TraceStore(tmp_path / "obs")
    store.write({"type": "trace_start", "trace_id": "abc", "question": "q"})
    store.write({"type": "llm_call", "trace_id": "abc", "phase": "classification",
                 "prompt_tokens": 3, "completion_tokens": 4})
    store.close()
    events, bad = read_events(tmp_path / "obs")
    assert bad == 0
    assert len(events) == 2
    assert events[0]["type"] == "trace_start"


def test_trace_store_write_never_raises(tmp_path, monkeypatch):
    store = TraceStore(tmp_path / "obs")
    monkeypatch.setattr(store, "_current_file", lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.write({"type": "llm_call"})  # _current_file returns None -> warn path
    assert any("observability" in str(w.message) for w in caught)


def test_trace_store_trace_llm_calls_accumulates_in_memory(tmp_path):
    store = TraceStore(tmp_path / "obs")
    store.write({"type": "trace_start", "trace_id": "abc"})
    store.write({"type": "llm_call", "trace_id": "abc", "phase": "classification"})
    store.write({"type": "llm_call", "trace_id": "abc", "phase": "route"})
    store.write({"type": "llm_call", "trace_id": "zzz", "phase": "classification"})
    calls = store.trace_llm_calls("abc")
    assert [c["phase"] for c in calls] == ["classification", "route"]


def test_span_stack_ctx_and_phase():
    with trace_span() as tid:
        assert tid == current_trace_id()
        assert current_phase() is None
        with phase("classification"):
            assert current_phase() == "classification"
        assert current_phase() is None
    assert current_trace_id() is None
    assert current_phase() is None


def test_phase_noop_without_span():
    with phase("classification"):
        assert current_phase() is None
```

Import names in the tests from `agent.observability.tracing`: `TraceStore`, `read_events`, `trace_span`, `phase`, `current_trace_id`, `current_phase`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tracing.py -q`
Expected: FAIL — `ModuleNotFoundError: agent.observability`. (So `agent/observability/__init__.py` must exist: make it an empty module in Step 3 when the package is created.)

- [ ] **Step 3: Implement `tracing.py`** — create `agent/observability/__init__.py` (empty for now) and `agent/observability/tracing.py`:

```python
from __future__ import annotations

import contextvars
import json
import threading
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


class TraceStore:
    """Append-only JSONL store, one file per day. Writes never raise."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            warnings.warn(f"observability: cannot create data dir: {e}")
        self._lock = threading.Lock()
        self._day: str | None = None
        self._file = None
        self._in_memory: dict[str, list[dict]] = {}
        self._memory_order: list[str] = []
        self._MAX_MEMORY_TRACES = 100

    def _current_file(self):
        day = datetime.now().strftime("%Y-%m-%d")
        if day != self._day:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
            self._day = day
            self._file = (self.data_dir / f"trace-{day}.jsonl").open("a", encoding="utf-8")
        return self._file

    def write(self, event: dict) -> None:
        try:
            line = json.dumps(event, ensure_ascii=False) + "\n"
            with self._lock:
                f = self._current_file()
                if f is None:
                    raise ValueError("unavailable file handle")
                f.write(line)
                f.flush()
                tid = event.get("trace_id")
                if event.get("type") == "llm_call" and tid is not None:
                    if tid not in self._in_memory:
                        self._in_memory[tid] = []
                        self._memory_order.append(tid)
                        if len(self._memory_order) > self._MAX_MEMORY_TRACES:
                            oldest = self._memory_order.pop(0)
                            self._in_memory.pop(oldest, None)
                    self._in_memory[tid].append(event)
        except Exception as e:  # noqa: BLE001 - observability must never break business
            warnings.warn(f"observability: failed to write trace event: {e}")

    def trace_llm_calls(self, trace_id: str) -> list[dict]:
        return list(self._in_memory.get(trace_id, []))

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


def read_events(data_dir: str | Path, *, day: str | None = None) -> tuple[list[dict], int]:
    base = Path(data_dir)
    pattern = f"trace-{day}.jsonl" if day else "trace-*.jsonl"
    bad = 0
    events: list[dict] = []
    for path in sorted(base.glob(pattern)):
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        bad += 1
                        continue
                    if isinstance(data, dict):
                        events.append(data)
                    else:
                        bad += 1
        except OSError:
            bad += 1
    return events, bad


@dataclass
class _Span:
    trace_id: str
    phases: list[str] = field(default_factory=list)


_span_var: contextvars.ContextVar[_Span | None] = contextvars.ContextVar(
    "observability_span", default=None
)


@contextmanager
def trace_span() -> Iterator[str]:
    trace_id = uuid.uuid4().hex[:12]
    token = _span_var.set(_Span(trace_id=trace_id))
    try:
        yield trace_id
    finally:
        _span_var.reset(token)


@contextmanager
def phase(name: str) -> Iterator[None]:
    span = _span_var.get()
    if span is None:
        yield
        return
    span.phases.append(name)
    try:
        yield
    finally:
        span.phases.pop()


def current_trace_id() -> str | None:
    span = _span_var.get()
    return span.trace_id if span else None


def current_phase() -> str | None:
    span = _span_var.get()
    if span is None or not span.phases:
        return None
    return span.phases[-1]


def now_millis() -> int:
    return int(datetime.now().timestamp() * 1000)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_tracing.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/observability/__init__.py agent/observability/tracing.py tests/test_tracing.py
git commit -m "feat: TraceStore JSONL store and contextvars span stack"
```

---

### Task 4: `TracedLLMClient` in `agent/observability/client.py`

**Files:**
- Create: `agent/observability/client.py`
- Test: `tests/test_observability_client.py`

**Interfaces:**
- Consumes: `TraceStore`, `current_trace_id()`, `current_phase()`, `now_millis()` from `tracing.py`; `LLMError` from `agent.llm`.
- Produces:
  - `class TracedLLMClient: __init__(self, inner, store: TraceStore)`. Proxies `model` attribute; `chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs) -> str` records an `llm_call` event (ok or error) and re-raises `LLMError`; `chat_completion_stream` delegates unobserved.
  - `llm_call` event shape:
    ```python
    {"type": "llm_call", "trace_id": ..., "phase": ..., "ts": ...,
     "model": ..., "prompt_tokens": int|None, "completion_tokens": int|None,
     "total_tokens": int|None, "latency_ms": float, "status": "ok"|"error", "error": str|None}
    ```
    Tokens come from `inner._usage_local.usage` if present (same thread — see §4.2 of the spec); else `None` (never 0).

- [ ] **Step 1: Write the failing test** — create `tests/test_observability_client.py`:

```python
import threading

import pytest

from agent.llm import LLMError
from agent.observability.client import TracedLLMClient
from agent.observability.tracing import TraceStore, current_phase, phase, trace_span


class FakeInner:
    def __init__(self, usage=None, error=None, model="m"):
        self.model = model
        self._usage_local = threading.local()
        self._usage_local.usage = usage
        self._error = error
        self.last_kwargs = None

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        self.last_kwargs = (model, messages)
        if self._error:
            raise self._error
        return "the answer"

    def chat_completion_stream(self, messages, **kwargs):
        for chunk in ["a", "b"]:
            yield chunk


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


def _make_store(tmp_path):
    return TraceStore(tmp_path / "obs")


def test_records_ok_call_with_usage(tmp_path):
    store = _make_store(tmp_path)
    inner = FakeInner(usage=_Usage(10, 5))
    traced = TracedLLMClient(inner, store)
    with trace_span() as tid, phase("classification"):
        text = traced.chat_completion([{"role": "user", "content": "hi"}], model="low-a")

    assert text == "the answer"
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


def test_records_error_and_reraises(tmp_path):
    store = _make_store(tmp_path)
    inner = FakeInner(error=LLMError("boom"))
    traced = TracedLLMClient(inner, store)
    with trace_span() as tid:
        with pytest.raises(LLMError):
            traced.chat_completion([{"role": "user", "content": "hi"}])

    ev = store.trace_llm_calls(tid)[0]
    assert ev["status"] == "error"
    assert "boom" in ev["error"]
    assert ev["prompt_tokens"] is None


def test_records_non_streaming_usage_is_none_when_missing(tmp_path):
    store = _make_store(tmp_path)
    inner = FakeInner(usage=None)
    traced = TracedLLMClient(inner, store)
    with trace_span() as tid:
        traced.chat_completion([{"role": "user", "content": "hi"}])
    ev = store.trace_llm_calls(tid)[0]
    assert ev["prompt_tokens"] is None


def test_stream_delegates_without_recording(tmp_path):
    store = _make_store(tmp_path)
    traced = TracedLLMClient(FakeInner(), store)
    out = list(traced.chat_completion_stream([{"role": "user", "content": "hi"}]))
    assert out == ["a", "b"]
    assert store.trace_llm_calls("anything") == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_observability_client.py -q`
Expected: FAIL — `ModuleNotFoundError: agent.observability.client`

- [ ] **Step 3: Implement `client.py`** — create `agent/observability/client.py`:

```python
from __future__ import annotations

import time
from typing import Iterator

from agent.llm import LLMError

from .tracing import TraceStore, current_phase, current_trace_id, now_millis


class TracedLLMClient:
    """Transparent LLMClient wrapper that records token usage and latency."""

    def __init__(self, inner, store: TraceStore):
        self._inner = inner
        self._store = store

    @property
    def model(self) -> str:
        return self._inner.model

    def _usage_tokens(self):
        usage = getattr(self._inner, "_usage_local", None)
        u = getattr(usage, "usage", None)
        if u is None:
            return None, None, None
        return getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None), getattr(u, "total_tokens", None)

    def _base_event(self) -> dict:
        return {
            "type": "llm_call",
            "trace_id": current_trace_id(),
            "phase": current_phase(),
            "ts": now_millis(),
        }

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs) -> str:
        started = time.perf_counter()
        try:
            text = self._inner.chat_completion(
                messages, model=model, temperature=temperature, **kwargs
            )
        except LLMError as e:
            ev = self._base_event()
            ev.update({"model": model or self._inner.model, "prompt_tokens": None,
                       "completion_tokens": None, "total_tokens": None,
                       "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                       "status": "error", "error": str(e)})
            self._store.write(ev)
            raise
        p, c, t = self._usage_tokens()
        ev = self._base_event()
        ev.update({"model": model or self._inner.model, "prompt_tokens": p,
                   "completion_tokens": c, "total_tokens": t,
                   "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                   "status": "ok", "error": None})
        self._store.write(ev)
        return text

    def chat_completion_stream(self, messages, *, model=None, temperature=0.7, **kwargs) -> Iterator[str]:
        # Not observed in v1: the main flow never uses streaming.
        yield from self._inner.chat_completion_stream(
            messages, model=model, temperature=temperature, **kwargs
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_observability_client.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/observability/client.py tests/test_observability_client.py
git commit -m "feat: TracedLLMClient records token usage and latency per call"
```

---

### Task 5: `agent/observability/patch.py` — automated method wrapping

**Files:**
- Create: `agent/observability/patch.py`
- Test: `tests/test_observability_patch.py`

**Interfaces:**
- Consumes: `TraceStore`, `trace_span`, `phase`, `current_phase`, `now_millis` from `tracing.py`; app classes `Chat`, `Router`, `ClassificationService`, `Strategy`, `Orchestrator`; `ObservabilityConfig`.
- Produces:
  - `DEFAULT_PHASES: dict[str, str]` — keys `Chat.respond`, `ClassificationService.classify`, `Router.route`, `Strategy.process`, `Orchestrator._plan`, `Orchestrator._worker`, `Orchestrator._aggregate`, `Orchestrator._direct_answer`.
  - `class Installed: __init__(self, store, phase_map); apply(self, config) -> None` — patches class methods with `functools.wraps` wrappers. Returns the list of successfully patched keys via `patched: list[str]`; failures are warned and skipped.
  - Wrapper behavior:
    - `Chat.respond` — enters `trace_span()`, records `trace_start`, calls original; on the way out records `trace_end` (answer_len = len of response text or 0; total_llm_calls from `store.trace_llm_calls(tid)`; total_tokens / total_latency_ms aggregated from those; for reject responses too). Record `decision`? No — `trace_start` and `trace_end` only.
      `trace_start` = `{"type": "trace_start", "trace_id": tid, "ts": ..., "question": question, "domain": domain name}`.
      `trace_end` = `{"type": "trace_end", "trace_id": tid, "ts": ..., "answer_len": len(text), "total_llm_calls": n, "total_tokens": int, "total_latency_ms": float, "reject": response.kind == "reject"}`.
    - `ClassificationService.classify` — `with phase("classification"):` original; record `decision` event with the `ClassificationResult` fields (`in_domain`, `intent`, `complexity`, `reason`).
    - `Router.route` — `with phase("route"):` original; record `decision` with `RouteResult` fields (`in_domain`, `strategy`, `intent`, `complexity`, `orchestrate`, `reject_reason`).
    - `Strategy.process` — `with phase(f"strategy.{self.strategy_id}"):` (phase is overridable via `phase_map["Strategy.process"]`).
    - `Orchestrator._plan` — `with phase(plan_name):`; record `decision` with `{"tasks": [...]}` from the parsed result or `{"degraded": True}` when it returns `None`.
    - `Orchestrator._worker` — `with phase(worker_name):` with a per-trace worker counter (`f"orchestration.worker.{n}"`); record `decision` with `{"task": title}` (task[0] of the tuple).
    - `Orchestrator._aggregate` — `with phase(aggregate_name):`.
    - `Orchestrator._direct_answer` — `with phase(direct_name):`.
  - Phase override resolution: `phase_map` keyed by the same names; `Strategy.process`/worker phase get `f"{base}.{suffix}"` where suffix is the strategy id / worker counter.

- [ ] **Step 1: Write the failing tests** — create `tests/test_observability_patch.py` (a `FakeInner` that records the phase at call time is essential — it lets us assert the phase string the wrapper set):

```python
import threading

import pytest

from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig, IntentDef, ObservabilityConfig, StrategyDef
from agent.observability import patch as patch_mod
from agent.observability.tracing import TraceStore, current_phase, read_events


@pytest.fixture(autouse=True)
def _reset_observability():
    yield
    patch_mod._ACTIVE = None  # keep class-level patching transparent for other test modules
```

The `FakeInner` and helpers below stay identical to the ones defined in the draft (records `current_phase()` at each call, pops from a response sequence). The three tests assert the wrapped pipeline records phases and decisions and preserves return values:

```python
class FakeInner:
    def __init__(self, responses):
        self._responses = list(responses)
        self._usage_local = threading.local()
        self.seen_phases = []

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        self.seen_phases.append(current_phase())
        return self._responses.pop(0)

    def chat_completion_stream(self, messages, **kwargs):
        return iter([])


_CLASSIFY = '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}'
_CLASSIFY_COMPLEX = '{"in_domain": true, "intent": "troubleshooting", "complexity": "complex", "reason": "ok"}'
_PLAN = '{"tasks": [{"title": "t1", "instruction": "i1"}, {"title": "t2", "instruction": "i2"}]}'


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="cm", domain_dir="d",
                       observability=ObservabilityConfig(enabled=True))


def _domain():
    return DomainConfig(
        name="sw", description="desc", out_of_domain_reply="Out.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies={"direct": StrategyDef("direct", default=True)},
        default_strategy="direct",
        prompts={"direct": "Direct prompt.", "unsupported_complex": "x."},
    )


def _domain_complex():
    return DomainConfig(
        name="sw", description="desc", out_of_domain_reply="Out.",
        intents={
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug"),
        },
        intent_mapping={"faq": "direct", "troubleshooting": "debugging"},
        strategies={
            "direct": StrategyDef("direct", default=True),
            "debugging": StrategyDef("debugging", complexity_gate=True),
        },
        default_strategy="direct",
        prompts={
            "direct": "Direct prompt.",
            "debugging": "Debugging prompt.",
            "unsupported_complex": "x.",
        },
    )


def _store(tmp_path):
    return TraceStore(tmp_path / "obs")


def test_install_wraps_and_records_pipeline(tmp_path):
    store = _store(tmp_path)
    client = FakeInner([_CLASSIFY, "the answer"])
    chat = Chat(client, _config(), _domain())
    patch_mod.Installed(store, phase_map={}).apply()
    resp = chat.respond("what is defer")

    assert resp.kind == "answer"
    assert resp.text == "the answer"
    assert client.seen_phases == ["classification", "strategy.direct"]
    events, _ = read_events(tmp_path / "obs")
    types = {e["type"] for e in events}
    assert {"trace_start", "llm_call", "decision", "trace_end"} <= types
    trace_id = None
    decision_phases = []
    for e in events:
        if e["type"] == "trace_start":
            trace_id = e["trace_id"]
        if e["type"] == "decision":
            decision_phases.append(e["phase"])
    assert trace_id is not None
    assert "classification" in decision_phases
    assert "route" in decision_phases
    trace_end = [e for e in events if e["type"] == "trace_end"][0]
    assert trace_end["answer_len"] == len("the answer")


def test_retains_original_return_values(tmp_path):
    client = FakeInner([_CLASSIFY, "the answer"])
    chat = Chat(client, _config(), _domain())
    patch_mod.Installed(_store(tmp_path), {}).apply()
    resp = chat.respond("what is defer")
    assert resp.text == "the answer"


def test_install_wraps_orchestration_phases(tmp_path):
    store = _store(tmp_path)
    inner = FakeInner([_CLASSIFY_COMPLEX, _PLAN, "w1", "w2", "final"])
    chat = Chat(inner, _config(), _domain_complex())
    patch_mod.Installed(store, {}).apply()
    resp = chat.respond("huge debugging task")

    assert resp.text == "final"
    assert "orchestration.planner" in inner.seen_phases
    assert "orchestration.aggregate" in inner.seen_phases
    assert any(p.startswith("orchestration.worker.") for p in inner.seen_phases)
    # worker numbering restarts per trace: exactly worker.1 and worker.2
    assert "orchestration.worker.1" in inner.seen_phases
    assert "orchestration.worker.2" in inner.seen_phases
```

The orchestration test's response sequence mirrors `tests/test_chat.py::test_respond_orchestrates_complex` (classification JSON → planner JSON → worker 1 → worker 2 → aggregate).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_observability_patch.py -q`
Expected: FAIL — `ModuleNotFoundError: agent.observability.patch`

- [ ] **Step 3: Implement `patch.py`** — create `agent/observability/patch.py`:

```python
from __future__ import annotations

import functools
import threading
import warnings
from dataclasses import dataclass, field

from agent.chat import Chat
from agent.classification import ClassificationService
from agent.orchestrator import Orchestrator
from agent.router import Router
from agent.strategy import Strategy

from .tracing import (
    TraceStore,
    current_trace_id,
    format_trace_line,
    now_millis,
    phase,
    trace_span,
)


DEFAULT_PHASES: dict[str, str] = {
    "Chat.respond": "trace",
    "ClassificationService.classify": "classification",
    "Router.route": "route",
    "Strategy.process": "strategy",
    "Orchestrator._plan": "orchestration.planner",
    "Orchestrator._worker": "orchestration.worker",
    "Orchestrator._aggregate": "orchestration.aggregate",
    "Orchestrator._direct_answer": "orchestration.direct",
}

# The active Installed (or None). Wrappers become transparent passthroughs when
# None, so class-level patching is safe even across tests/modules that never
# install observability.
_ACTIVE: "Installed | None" = None

_PATCH_MARKER = "__observability_patched__"


def _current_inst() -> "Installed | None":
    return _ACTIVE


@dataclass
class Installed:
    store: TraceStore
    phase_map: dict[str, str] = field(default_factory=dict)
    patched: list[str] = field(default_factory=list)
    _worker_counters: dict = field(default_factory=dict)
    _worker_lock: threading.Lock = field(default_factory=threading.Lock)

    def _phase(self, key: str) -> str:
        return self.phase_map.get(key, DEFAULT_PHASES[key])

    def _next_worker(self, trace_id: str) -> int:
        with self._worker_lock:
            n = self._worker_counters.get(trace_id, 0) + 1
            self._worker_counters[trace_id] = n
            return n

    def _record_decision(self, trace_id: str, ph: str, data: dict) -> None:
        self.store.write({"type": "decision", "trace_id": trace_id, "phase": ph,
                          "ts": now_millis(), "data": data})

    def _wrap(self, key: str, target, patch_name: str) -> None:
        factories = {
            "Chat.respond": Installed._wrap_respond,
            "ClassificationService.classify": Installed._wrap_classify,
            "Router.route": Installed._wrap_route,
            "Strategy.process": Installed._wrap_strategy,
            "Orchestrator._plan": Installed._wrap_plan,
            "Orchestrator._worker": Installed._wrap_worker,
            "Orchestrator._aggregate": Installed._wrap_aggregate,
            "Orchestrator._direct_answer": Installed._wrap_direct,
        }
        try:
            original = getattr(target, patch_name)
            if getattr(original, _PATCH_MARKER, None) == key:
                return  # idempotent: already wrapped by a previous install
            wrapper = factories[key](original, key)
            setattr(target, patch_name, functools.wraps(original)(wrapper))
            wrapper.__setattr__(_PATCH_MARKER, key)
            self.patched.append(key)
        except Exception as e:  # noqa: BLE001 - degrade, never block business
            warnings.warn(f"observability: failed to patch {key}: {e}")

    # Wrapper factories. `self` is the Installed instance (captured as `inst`);
    # the wrapper's first positional arg is the business instance (chat, cls,
    # strat, orch...). Each wrapper is a transparent passthrough when no install
    # is active.

    def _wrap_respond(self, original, key):
        inst = self

        def wrapper(chat, question):
            if _current_inst() is None:
                return original(chat, question)
            with trace_span() as tid:
                ph = inst._phase(key)
                inst.store.write({"type": "trace_start", "trace_id": tid, "phase": ph,
                                  "ts": now_millis(), "question": question,
                                  "domain": getattr(chat.domain, "name", None)})
                response = original(chat, question)
                calls = inst.store.trace_llm_calls(tid)
                total_tokens = sum(c.get("total_tokens") or 0 for c in calls)
                total_lat = sum(c.get("latency_ms") or 0 for c in calls)
                inst.store.write({"type": "trace_end", "trace_id": tid, "phase": ph,
                                  "ts": now_millis(), "answer_len": len(response.text),
                                  "total_llm_calls": len(calls), "total_tokens": total_tokens,
                                  "total_latency_ms": round(total_lat, 1),
                                  "reject": response.kind == "reject"})
                try:
                    if calls:
                        print(format_trace_line(tid, calls))
                except Exception:  # noqa: BLE001 - display must never break business
                    pass
                return response
        return wrapper

    def _wrap_classify(self, original, key):
        inst = self

        def wrapper(cls, question, *, model=None):
            if _current_inst() is None:
                return original(cls, question, model=model)
            with phase(inst._phase(key)):
                result = original(cls, question, model=model)
                tid = current_trace_id()
                if tid:
                    inst._record_decision(tid, inst._phase(key), {
                        "in_domain": result.in_domain, "intent": result.intent,
                        "complexity": result.complexity, "reason": result.reason})
                return result
        return wrapper

    def _wrap_route(self, original, key):
        inst = self

        def wrapper(rtr, question):
            if _current_inst() is None:
                return original(rtr, question)
            with phase(inst._phase(key)):
                result = original(rtr, question)
                tid = current_trace_id()
                if tid:
                    inst._record_decision(tid, inst._phase(key), {
                        "in_domain": result.in_domain, "strategy": result.strategy,
                        "intent": result.intent, "complexity": result.complexity,
                        "orchestrate": result.orchestrate, "reject_reason": result.reject_reason})
                return result
        return wrapper

    def _wrap_strategy(self, original, key):
        inst = self

        def wrapper(strat, client, question, history, *, model=None):
            if _current_inst() is None:
                return original(strat, client, question, history, model=model)
            with phase(f"{inst._phase(key)}.{strat.strategy_id}"):
                return original(strat, client, question, history, model=model)
        return wrapper

    def _wrap_plan(self, original, key):
        inst = self

        def wrapper(orch, question, strategy, context, model):
            if _current_inst() is None:
                return original(orch, question, strategy, context, model)
            with phase(inst._phase(key)):
                tasks = original(orch, question, strategy, context, model)
                tid = current_trace_id()
                if tid:
                    data = {"degraded": True} if tasks is None else {
                        "tasks": [{"title": t, "instruction": i} for t, i in tasks]}
                    inst._record_decision(tid, inst._phase(key), data)
                return tasks
        return wrapper

    def _wrap_worker(self, original, key):
        inst = self

        def wrapper(orch, question, task, context, model):
            if _current_inst() is None:
                return original(orch, question, task, context, model)
            base = inst._phase(key)
            n = inst._next_worker(current_trace_id() or "")
            with phase(f"{base}.{n}"):
                tid = current_trace_id()
                if tid:
                    inst._record_decision(tid, f"{base}.{n}", {"task": task[0]})
                return original(orch, question, task, context, model)
        return wrapper

    def _wrap_aggregate(self, original, key):
        inst = self

        def wrapper(orch, question, strategy, context, tasks, outputs, model):
            if _current_inst() is None:
                return original(orch, question, strategy, context, tasks, outputs, model)
            with phase(inst._phase(key)):
                return original(orch, question, strategy, context, tasks, outputs, model)
        return wrapper

    def _wrap_direct(self, original, key):
        inst = self

        def wrapper(orch, question, strategy, context, model):
            if _current_inst() is None:
                return original(orch, question, strategy, context, model)
            with phase(inst._phase(key)):
                return original(orch, question, strategy, context, model)
        return wrapper

    def apply(self) -> "Installed":
        global _ACTIVE
        targets = [
            ("Chat.respond", Chat, "respond"),
            ("ClassificationService.classify", ClassificationService, "classify"),
            ("Router.route", Router, "route"),
            ("Strategy.process", Strategy, "process"),
            ("Orchestrator._plan", Orchestrator, "_plan"),
            ("Orchestrator._worker", Orchestrator, "_worker"),
            ("Orchestrator._aggregate", Orchestrator, "_aggregate"),
            ("Orchestrator._direct_answer", Orchestrator, "_direct_answer"),
        ]
        for key, cls, method in targets:
            self._wrap(key, cls, method)
        _ACTIVE = self
        return self
```

Key robustness notes:

- **Closure capture**: every wrapper factory captures `inst = self` (the `Installed`) and names the business-instance positional arg distinctly (`chat`, `cls`, `rtr`, `strat`, `orch`). No `self` shadowing — the earlier drafts had this bug.
- **Transparent when inactive**: each wrapper first checks `_current_inst()`; when `None` it calls the original and returns, so class-level patching never alters behavior for code paths that didn't install observability.
- **Idempotent**: `_wrap` skips a method already carrying `_PATCH_MARKER`, so calling `install()` twice (e.g. across tests) never double-wraps.
- `_next_worker` uses `current_trace_id()` so worker numbering restarts per trace.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_observability_patch.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/observability/patch.py tests/test_observability_patch.py
git commit -m "feat: automated class-level wrapping of pipeline entry points"
```

---

### Task 6: `install()` + wiring in `agent_cli.py`

**Files:**
- Modify: `agent/observability/__init__.py`
- Modify: `agent/observability/tracing.py` (add `_fmt_tokens` + `format_trace_line`)
- Modify: `agent/agent_cli.py`
- Test: `tests/test_observability_install.py`

**Interfaces:**
- Consumes: `TracedLLMClient`, `Installed`, `TraceStore`, `ObservabilityConfig`, `LLMClient`.
- Produces:
  - `install(client, config, domain=None) -> tuple[client, Installed | None]` in `agent/observability/__init__.py`. When `config.observability` is None or disabled → returns `(client, None)`. Otherwise builds `TraceStore`, `TracedLLMClient`, `Installed(store, phase_map).apply()`, and returns `(traced, installed)`.
  - `format_trace_line(trace_id, calls) -> str` in `tracing.py` — `[trace <id>] <phase> <s>/<tok> ... | total <tok> tok <s>s`. Tokens formatted with 1k suffix; latency as seconds with 1 decimal.
  - Terminal display is **not** a separate hook — `Installed._wrap_respond` prints `format_trace_line(tid, calls)` after recording `trace_end` (implemented in Task 5).
  - `agent/agent_cli.py::main` — after building `client`, call `client, _obs = install(client, config, domain)` and use the returned client for `Chat(...)` both for `--ask` and `run_repl`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_observability_install.py`:

```python
import threading

import pytest

from agent.config import AgentConfig, ObservabilityConfig
from agent.observability import format_trace_line, install
from agent.observability import patch as patch_mod
from agent.observability.tracing import _fmt_tokens, read_events


@pytest.fixture(autouse=True)
def _reset_observability():
    yield
    patch_mod._ACTIVE = None  # keep class-level patching transparent for other test modules


class FakeClient:
    model = "m"

    def __init__(self, responses):
        self._responses = list(responses)
        self._usage_local = threading.local()

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        return self._responses.pop(0)

    def chat_completion_stream(self, messages, **kwargs):
        return iter([])


_CLASSIFY = '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}'


def _domain():
    from agent.config import DomainConfig, IntentDef, StrategyDef
    return DomainConfig(
        name="sw", description="desc", out_of_domain_reply="Out.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies={"direct": StrategyDef("direct", default=True)},
        default_strategy="direct",
        prompts={"direct": "Direct prompt.", "unsupported_complex": "x."},
    )


def _enabled_config(tmp_path):
    return AgentConfig(
        base_url="x", model="m", classifier_model="cm", domain_dir="d",
        observability=ObservabilityConfig(enabled=True, data_dir=str(tmp_path / "obs")),
    )


def test_install_disabled_returns_untouched(tmp_path):
    config = AgentConfig(base_url="x", model="m", classifier_model="cm", domain_dir="d")
    client = FakeClient([])
    out, plugin = install(client, config, None)
    assert out is client
    assert plugin is None


def test_install_enabled_wraps_client(tmp_path):
    client = FakeClient([])
    out, plugin = install(client, _enabled_config(tmp_path), None)
    assert out is not client
    assert plugin is not None
    assert out.model == "m"


def test_install_enabled_patches_pipeline(tmp_path):
    from agent.chat import Chat
    inner = FakeClient([_CLASSIFY, "the answer"])
    out, _ = install(inner, _enabled_config(tmp_path), _domain())
    resp = Chat(out, _enabled_config(tmp_path), _domain()).respond("hi")
    assert resp.kind == "answer"
    events, bad = read_events(tmp_path / "obs")
    assert bad == 0
    assert any(e["type"] == "trace_start" for e in events)


def test_fmt_tokens_and_trace_line():
    assert _fmt_tokens(None) == "?"
    assert _fmt_tokens(500) == "500"
    assert _fmt_tokens(1500) == "1.5k"
    calls = [
        {"phase": "classification", "total_tokens": 1234, "latency_ms": 330},
        {"phase": "strategy.direct", "total_tokens": 3500, "latency_ms": 2100},
    ]
    line = format_trace_line("abc", calls)
    assert line.startswith("[trace abc]")
    assert "classification 0.3s/1.2k" in line
    assert "strategy.direct 2.1s/3.5k" in line
    assert line.endswith("| total 4734 tok 2.4s")
```

Note: `install`'s `domain` argument is `None` in two of the tests — `install` must not dereference it.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_observability_install.py -q`
Expected: FAIL — `ImportError: cannot import name 'install'`

- [ ] **Step 3: Add `format_trace_line` to `tracing.py`** — append to `agent/observability/tracing.py`:

```python
def _fmt_tokens(n) -> str:
    if n is None:
        return "?"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def format_trace_line(trace_id: str, calls: list[dict]) -> str:
    parts = []
    for c in calls:
        ph = c.get("phase") or "?"
        tok = _fmt_tokens(c.get("total_tokens"))
        latency_s = (c.get("latency_ms") or 0) / 1000
        parts.append(f"{ph} {latency_s:.1f}s/{tok}")
    total_tokens = sum(c.get("total_tokens") or 0 for c in calls)
    total_s = sum(c.get("latency_ms") or 0 for c in calls) / 1000
    return f"[trace {trace_id}] " + " | ".join(parts) + \
        f" | total {total_tokens} tok {total_s:.1f}s"
```

- [ ] **Step 4: Implement `agent/observability/__init__.py`**:

```python
from __future__ import annotations

from .client import TracedLLMClient
from .patch import Installed
from .tracing import TraceStore, format_trace_line, read_events


def install(client, config, domain=None):
    obs = config.observability
    if obs is None or not obs.enabled:
        return client, None
    store = TraceStore(obs.data_dir)
    traced = TracedLLMClient(client, store)
    installed = Installed(store=store, phase_map=obs.phase_map).apply()
    return traced, installed


__all__ = [
    "TraceStore",
    "TracedLLMClient",
    "Installed",
    "format_trace_line",
    "read_events",
    "install",
]
```

Note: `Installed.apply()` takes no config argument (Task 5); the terminal display is already folded into `_wrap_respond` (Task 5), so `install()` needs no extra hook.

- [ ] **Step 5: Wire `agent/agent_cli.py`** — change the client construction block to:

```python
    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model)
    client, _obs_plugin = install(client, config, domain)
```

with `from agent.observability import install` at the top of `agent_cli.py`. Keep `install` returning `(client, None)` when disabled, so nothing else changes.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_observability_install.py tests/test_observability_patch.py tests/test_tracing.py -q`
Expected: PASS. Then `uv run pytest -q` — all green (existing tests unaffected; agent_cli integration is additive).

- [ ] **Step 7: Commit**

```bash
git add agent/observability/__init__.py agent/observability/tracing.py agent/observability/patch.py agent/agent_cli.py tests/test_observability_install.py
git commit -m "feat: install() wiring and terminal trace display"
```

---

### Task 7: `agent/observability/report.py` — CLI table + HTML report

**Files:**
- Create: `agent/observability/report.py`
- Create: `agent/observability/__main__.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `read_events` from `tracing.py`.
- Produces:
  - `summarize_traces(events: list[dict]) -> list[dict]` — group events by `trace_id`; each row: `trace_id`, `question` (from trace_start), `total_tokens`, `prompt_tokens`, `completion_tokens`, `llm_calls`, `total_latency_ms`, `reject` (bool), phases (sorted unique list).
  - `build_cli_report(events: list[dict]) -> str` — fixed-width text table + header aggregate (total tokens, per-model, per-phase subtotal, "no usage" count).
  - `build_html_report(events: list[dict]) -> str` — self-contained single file (inline CSS/SVG, no CDN) with: token trend (bar chart by trace sequence), model distribution, phase latency bar, and expandable per-trace detail (each phase token/latency).
  - `main(argv: list[str] | None = None) -> int` — argparse: `report [--data-dir PATH] [--day YYYY-MM-DD] [--html]`. Prints CLI report or writes `{data_dir}/report.html` and prints its path. `agent/observability/__main__.py` calls `report.main()` so `python -m agent.observability report ...` works.

- [ ] **Step 1: Write the failing tests** — create `tests/test_report.py`:

```python
import json

from agent.observability.report import build_cli_report, build_html_report, summarize_traces


def _events():
    return [
        {"type": "trace_start", "trace_id": "a", "question": "q1", "phase": "trace", "ts": 1},
        {"type": "llm_call", "trace_id": "a", "phase": "classification", "model": "m",
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "latency_ms": 100,
         "status": "ok"},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.direct", "model": "m",
         "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "latency_ms": 200,
         "status": "ok"},
        {"type": "trace_end", "trace_id": "a", "answer_len": 50, "total_tokens": 45,
         "total_llm_calls": 2, "reject": False, "phase": "trace", "ts": 300},
    ]


def test_summarize_traces_aggregates():
    rows = summarize_traces(_events())
    assert len(rows) == 1
    r = rows[0]
    assert r["trace_id"] == "a"
    assert r["question"] == "q1"
    assert r["total_tokens"] == 45
    assert r["llm_calls"] == 2
    assert r["phases"] == ["classification", "strategy.direct"]


def test_cli_report_contains_key_numbers(capsys):
    text = build_cli_report(_events())
    assert "a" in text
    assert "45" in text
    assert "classification" in text


def test_html_report_self_contained():
    html = build_html_report(_events())
    assert "<html" in html
    assert "classification" in html
    assert "<svg" in html
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_report.py -q`
Expected: FAIL — `ModuleNotFoundError: agent.observability.report`

- [ ] **Step 3: Implement `report.py`** — create `agent/observability/report.py`:

```python
from __future__ import annotations

import argparse
import html
import sys
from collections import Counter
from pathlib import Path

from .tracing import read_events


def summarize_traces(events: list[dict]) -> list[dict]:
    """One row per trace_id, aggregating llm_call/trace_start/trace_end events."""
    order: list[str] = []
    rows: dict[str, dict] = {}
    for e in events:
        tid = e.get("trace_id")
        if not tid:
            continue
        if tid not in rows:
            rows[tid] = {
                "trace_id": tid, "question": "", "total_tokens": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0,
                "total_latency_ms": 0.0, "reject": False, "phases": [],
                "phase_set": [],
            }
            order.append(tid)
        r = rows[tid]
        if e.get("type") == "trace_start":
            r["question"] = e.get("question", "")
        elif e.get("type") == "trace_end":
            r["reject"] = bool(e.get("reject"))
        elif e.get("type") == "llm_call":
            r["llm_calls"] += 1
            r["prompt_tokens"] += e.get("prompt_tokens") or 0
            r["completion_tokens"] += e.get("completion_tokens") or 0
            r["total_tokens"] += e.get("total_tokens") or 0
            r["total_latency_ms"] += e.get("latency_ms") or 0
            ph = e.get("phase") or "?"
            if ph not in r["phase_set"]:
                r["phase_set"].append(ph)
    result = []
    for tid in order:
        r = rows[tid]
        r["phases"] = r["phase_set"]
        del r["phase_set"]
        result.append(r)
    return result


def _clamped(s: str, width: int) -> str:
    return s[: width - 1] + "…" if len(s) > width else s.ljust(width)


def build_cli_report(events: list[dict]) -> str:
    rows = summarize_traces(events)
    total_tokens = sum(r["total_tokens"] for r in rows)
    no_usage = sum(1 for e in events if e.get("type") == "llm_call" and e.get("total_tokens") is None)
    model_counter = Counter(e.get("model") for e in events if e.get("type") == "llm_call")
    phase_counter = Counter(e.get("phase") for e in events if e.get("type") == "llm_call")

    lines = ["ExpertForge observability report", f"traces: {len(rows)}  total_tokens: {total_tokens}  no_usage: {no_usage}"]
    if model_counter:
        lines.append("by model: " + ", ".join(f"{m or '?'}={n}" for m, n in model_counter.items()))
    if phase_counter:
        lines.append("by phase: " + ", ".join(f"{p or '?'}={n}" for p, n in phase_counter.items()))
    lines.append("")
    header = ["trace", "question", "calls", "total_tok", "prompt", "completion", "latency(ms)", "reject"]
    lines.append(" | ".join(h.ljust(12) for h in header))
    lines.append("-" * 96)
    for r in rows:
        cols = [r["trace_id"], _clamped(r["question"], 12), str(r["llm_calls"]),
                str(r["total_tokens"]), str(r["prompt_tokens"]), str(r["completion_tokens"]),
                f'{r["total_latency_ms"]:.0f}', "yes" if r["reject"] else "no"]
        lines.append(" | ".join(c.ljust(12) for c in cols))
    return "\n".join(lines) + "\n"


def _svg_bar(rows: list[dict], width: int = 800) -> str:
    if not rows:
        return "<svg width=\"800\" height=\"40\"></svg>"
    max_tok = max(r["total_tokens"] for r in rows) or 1
    bars = []
    for i, r in enumerate(rows):
        w = max(2, int(r["total_tokens"] / max_tok * (width - 20)))
        bars.append(
            f'<rect x="{10}" y="{i * 8 + 2}" width="{w}" height="6" '
            f'fill="#4a90d9"><title>{html.escape(r["trace_id"])} '
            f'{r["total_tokens"]} tokens</title></rect>'
        )
    return f'<svg width="{width}" height="{len(rows) * 8 + 8}">' + "".join(bars) + "</svg>"


def build_html_report(events: list[dict]) -> str:
    rows = summarize_traces(events)
    details = []
    for r in rows:
        phase_rows = "\n".join(
            f"<tr><td>{html.escape(p)}</td></tr>"
            for p in r["phases"]
        )
        details.append(
            f"<details><summary>{html.escape(r['trace_id'])} — "
            f"{html.escape(r['question'])} — {r['total_tokens']} tok</summary>"
            f"<table><tr><th>phase</th></tr>{phase_rows}</table></details>"
        )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ExpertForge observability</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem}}details{{margin:.5rem 0}}</style>
</head><body>
<h1>ExpertForge observability report</h1>
<h2>Token trend by trace</h2>{_svg_bar(rows)}
<h2>Traces</h2>{''.join(details)}
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent.observability.report",
                                     description="Generate observability reports")
    parser.add_argument("--data-dir", default=".observability", help="trace JSONL directory")
    parser.add_argument("--day", default=None, help="YYYY-MM-DD filter")
    parser.add_argument("--html", action="store_true", help="write HTML report")
    args = parser.parse_args(argv)
    events, bad = read_events(args.data_dir, day=args.day)
    if bad:
        print(f"note: {bad} unreadable trace line(s) skipped", file=sys.stderr)
    if args.html:
        path = Path(args.data_dir) / "report.html"
        path.write_text(build_html_report(events), encoding="utf-8")
        print(f"Report written to {path}")
    else:
        print(build_cli_report(events))
    return 0
```

The behavior contract is fixed by `tests/test_report.py`: `summarize_traces` returns rows with `total_tokens`, `llm_calls`, `phases`, `question`; `build_cli_report` prints the trace id and token totals; `build_html_report` emits `<html>`, phase text, and `<svg>`.

Also create `agent/observability/__main__.py`:

```python
import sys

from .report import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_report.py -q`
Expected: PASS

- [ ] **Step 5: Smoke-test the entry point**

Run: `uv run python -m agent.observability report --data-dir tests/fixtures 2>/dev/null || true`
Expected: no crash (empty report or fixture output). Full CLI sanity: `uv run python -m agent.observability report --help` prints usage.

- [ ] **Step 6: Commit**

```bash
git add agent/observability/report.py agent/observability/__main__.py tests/test_report.py
git commit -m "feat: CLI table and HTML report for observability traces"
```

---

### Task 8: README docs + full regression + final commit

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: user-facing documentation.

- [ ] **Step 1: Document the feature** — in `README.md`, add a short "Observability" section after the "Run" section:

```markdown
## Observability

Optional token/cost-free usage tracking and trace visualization. Enable it in `config.json`:

```json
{
  "observability": { "enabled": true, "data_dir": ".observability", "phase_map": {} }
}
```

- Every LLM call's tokens and latency are recorded automatically (classification, routing, strategy, and orchestration phases) to per-day JSONL files under `data_dir`.
- During a REPL/`--ask` run a compact per-question trace line is printed after each answer.
- After a run, view reports:

```bash
uv run python -m agent.observability report               # CLI summary table
uv run python -m agent.observability report --html        # self-contained HTML report
uv run python -m agent.observability report --day 2026-08-11
```

`phase_map` optionally remaps the built-in phase names (see `agent/observability/patch.py::DEFAULT_PHASES`). Disabled by default; when disabled the agent behaves exactly as before.
```

(Adjust code-fence nesting so the inner JSON fences are indented correctly in the final README.)

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all green (existing 84 + new observability tests).

- [ ] **Step 3: Manual end-to-end sanity (no API key required)**

Run:
```bash
uv run python -c "
import threading
from agent.config import AgentConfig, ObservabilityConfig
from agent.observability import install
class C:
    model = 'm'
    def __init__(self):
        self._usage_local = threading.local()
    def chat_completion(self, messages, **kw):
        return 'x'
client, plugin = install(C(), AgentConfig(base_url='x', model='m', classifier_model='cm', domain_dir='d', observability=ObservabilityConfig(enabled=True, data_dir='/tmp/_obs_test')), None)
print(type(client).__name__, plugin is not None)
"
```
Expected: prints `TracedLLMClient True`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document observability plugin usage"
```

---

## Self-Review

- **Spec coverage:** §2 architecture → Tasks 3–6; §2.1 config → Task 2; §3 data model → Tasks 3–4; §3.3 phases → Task 5; §4.1/4.2 → Tasks 1, 4; §4.3 patch table → Task 5; §4.4 robustness (degrade/warn) → Tasks 3, 5, 6; §5 terminal display → Task 6; §6.1/6.2 reports → Task 7; §7 error handling → Tasks 3, 4, 5, 7; §8 tests → all tasks; §9 success criteria → verified in Task 8.
- **Placeholders:** none — every step has concrete code.
- **Type consistency:** `TraceStore.write(event: dict)`, `trace_span() -> Iterator[str]`, `phase(name)`, `current_trace_id()`/`current_phase()`, `TracedLLMClient(inner, store)`, `Installed(store, phase_map).apply()`, `install(client, config, domain=None) -> (client, plugin)`, `format_trace_line` (in `tracing.py`), `read_events(data_dir, day=None) -> (events, bad)`, `_fmt_tokens` (in `tracing.py`) are used consistently across tasks 3–7. Worker phases `f"orchestration.worker.{n}"` match §3.3. Config field name `observability` on `AgentConfig` matches §2.1. Wrappers capture `inst = self` to avoid `self` shadowing (Task 5).
- **Test isolation:** `tests/test_observability_patch.py` and `tests/test_observability_install.py` each install an autouse `_reset_observability` fixture that sets `patch_mod._ACTIVE = None` after every test, so the class-level wrappers stay transparent (passthrough) for every other test module in the same pytest process.
- **Known deviation:** terminal line appears immediately after `Chat.respond` returns, i.e. just *before* the REPL prints `expert > ...`. Cosmetic only; spec §5 example order is not contractually binding.