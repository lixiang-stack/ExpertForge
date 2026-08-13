# Observability Reporting Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the observability reporting layer: terminal shows a single in/out-token + time line per question; the report command becomes HTML-only with readable labeled charts; per-trace detail surfaces the full decision timeline.

**Architecture:** Split data preparation (pure, testable `report_data.py`) from rendering (HTML + CLI in `report.py`). The recording layer is untouched — no new event fields; all data already exists in the JSONL trace files. A new minimal `format_trace_summary` replaces the per-phase `format_trace_line` for the terminal line.

**Tech Stack:** Python 3.12 (repo standard), stdlib only (`dataclasses`, `argparse`, `html`, `pathlib`, `collections`), inline SVG/CSS in the HTML report (no CDN, no JS).

## Global Constraints

- **Token-only:** no monetary cost, no pricing table, anywhere.
- **Never raise into business code:** all observability degradation uses `warnings.warn`; the existing regression test `test_observability_failure_never_surfaces_into_business` must stay green.
- **No new event fields, no schema change:** `client.py`, `patch.py`, and the `TraceStore`/`read_events` event-write paths are unchanged except for the `format_trace_summary` rename in Task 3.
- **No raw prompt/response text** captured (structured-only).
- **HTML self-contained:** single file, inline CSS, no CDN, no JS libraries. All user content (question, task titles, reason strings, model names) passes through `html.escape`.
- **Terminal shows raw integers** for in/out tokens; no `_fmt_tokens` k-suffix is used for the terminal line.
- **Report command is HTML-only:** `build_cli_report` and the `--html` flag are removed; `main(argv) -> int` still strips a leading `report` token.
- **Dataclasses provide `as_dict()`** for JSON serialization.

---
## File Structure

| File | Responsibility |
|------|----------------|
| `agent/observability/report_data.py` (new) | Pure data prep: `TraceSummary`/`Step`/`ModelStat` dataclasses + `summarize_traces`/`build_timeline`/`model_stats`/`total_stats`. No rendering, no I/O. |
| `agent/observability/report.py` (modified) | HTML rendering + CLI. Deletes `build_cli_report`, `_svg_hbar`, `--html`; adds `_label_chart`, `build_html_report`, retains `main`. Re-exports `summarize_traces` from `report_data`. |
| `agent/observability/tracing.py` (modified) | `format_trace_line` → `format_trace_summary`; removes `_fmt_tokens`. |
| `agent/observability/__init__.py` (modified) | Export `format_trace_summary` in place of `format_trace_line`. |
| `agent/observability/patch.py` (modified) | Call site switches to `format_trace_summary`. |
| `tests/test_report_data.py` (new) | Pure data-prep tests. |
| `tests/test_report.py` (rewritten) | Renderer + CLI tests; asserts `build_cli_report` is gone. |
| `tests/test_tracing.py` (modified) | Replace `format_trace_line`/`_fmt_tokens` tests with `format_trace_summary` test. |
| `README.md` (modified) | Update report command docs + terminal-line wording. |

---
### Task 1: `report_data.py` — dataclasses + summarizes

**Files:**
- Create: `agent/observability/report_data.py`
- Create: `tests/test_report_data.py`

**Interfaces:**
- Consumes: event dicts exactly as written by `read_events()` (schema in `tracing.py`): `trace_start` (`trace_id`,`question`,`domain`,`phase`,`ts`), `llm_call` (`trace_id`,`phase`,`model`,`prompt_tokens`,`completion_tokens`,`total_tokens`,`latency_ms`,`status`,`error`,`ts`), `decision` (`trace_id`,`phase`,`data`,`ts`), `trace_end` (`trace_id`,`answer_len`,`total_llm_calls`,`total_tokens`,`total_latency_ms`,`reject`,`ts`).
- Produces: `TraceSummary`, `ModelStat` dataclasses; `summarize_traces(events) -> list[TraceSummary]`; `total_stats(events) -> dict`; `model_stats(events) -> list[ModelStat]`. (Used by Task 4's renderer.)

- [ ] **Step 1: Write the failing tests** — create `tests/test_report_data.py`:

```python
import pytest

from agent.observability.report_data import (
    ModelStat,
    TraceSummary,
    model_stats,
    summarize_traces,
    total_stats,
)


def _events():
    return [
        {"type": "trace_start", "trace_id": "a", "question": "q1", "domain": "sw",
         "phase": "trace", "ts": 1},
        {"type": "llm_call", "trace_id": "a", "phase": "classification", "model": "m1",
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
         "latency_ms": 100, "status": "ok", "ts": 10},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.direct", "model": "m2",
         "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
         "latency_ms": 200, "status": "ok", "ts": 20},
        {"type": "trace_end", "trace_id": "a", "answer_len": 50, "total_llm_calls": 2,
         "total_tokens": 45, "total_latency_ms": 300.0, "reject": False,
         "phase": "trace", "ts": 300},
        {"type": "llm_call", "trace_id": "b", "phase": "classification", "model": "m1",
         "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
         "latency_ms": 50, "status": "error", "error": "boom", "ts": 5},
    ]


def test_summarize_traces_aggregates():
    rows = summarize_traces(_events())
    assert len(rows) == 2
    a = rows[0]
    assert a.trace_id == "a"
    assert a.question == "q1"
    assert a.domain == "sw"
    assert a.in_tokens == 30
    assert a.out_tokens == 15
    assert a.total_tokens == 45
    assert a.llm_calls == 2
    assert a.total_latency_ms == 300.0
    assert a.reject is False
    assert a.has_error is False
    b = rows[1]
    assert b.has_error is True
    assert b.in_tokens == 0
    assert b.out_tokens == 0
    assert b.total_latency_ms == 50.0


def test_summarize_missing_usage_counts_zero():
    rows = summarize_traces([
        {"type": "llm_call", "trace_id": "x", "phase": "route", "model": "m1",
         "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
         "latency_ms": 10, "status": "ok"},
    ])
    assert rows[0].in_tokens == 0
    assert rows[0].out_tokens == 0
    assert rows[0].llm_calls == 1


def test_total_stats():
    st = total_stats(_events())
    assert st["traces"] == 2
    assert st["llm_calls"] == 3
    assert st["in_tokens"] == 30
    assert st["out_tokens"] == 15
    assert st["total_tokens"] == 45
    assert st["total_latency_ms"] == 350.0
    assert st["has_error"] is True


def test_model_stats_sorted_and_aggregated():
    ms = model_stats(_events())
    assert [m.model for m in ms] == ["m2", "m1"]  # by total tokens desc (30, 15)
    assert ms[0].calls == 1
    assert ms[0].in_tokens == 20
    assert ms[0].out_tokens == 10
    assert ms[1].calls == 2
    assert ms[1].in_tokens == 10


def test_summary_as_dict():
    s = summarize_traces(_events())[0].as_dict()
    assert s["trace_id"] == "a"
    assert s["in_tokens"] == 30


def test_model_stat_is_dataclass():
    assert isinstance(model_stats(_events())[0], ModelStat)
    assert isinstance(summarize_traces(_events())[0], TraceSummary)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_report_data.py -q`
Expected: FAIL — `ModuleNotFoundError: agent.observability.report_data`

- [ ] **Step 3: Implement `report_data.py`**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TraceSummary:
    trace_id: str = ""
    question: str = ""
    domain: str | None = None
    in_tokens: int = 0
    out_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    total_latency_ms: float = 0.0
    reject: bool = False
    has_error: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelStat:
    model: str = ""
    calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    total_latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def summarize_traces(events: list[dict]) -> list[TraceSummary]:
    """One TraceSummary per trace_id, first-seen order. Never raises."""
    order: list[str] = []
    rows: dict[str, TraceSummary] = {}
    for e in events:
        tid = e.get("trace_id")
        if not tid:
            continue
        if tid not in rows:
            rows[tid] = TraceSummary(trace_id=tid)
            order.append(tid)
        r = rows[tid]
        typ = e.get("type")
        if typ == "trace_start":
            if not r.question:
                r.question = e.get("question", "")
            if r.domain is None:
                r.domain = e.get("domain")
        elif typ == "trace_end":
            r.reject = bool(e.get("reject"))
            r.total_latency_ms = float(e.get("total_latency_ms") or r.total_latency_ms)
        elif typ == "llm_call":
            r.llm_calls += 1
            r.in_tokens += e.get("prompt_tokens") or 0
            r.out_tokens += e.get("completion_tokens") or 0
            r.total_tokens += e.get("total_tokens") or 0
            r.total_latency_ms += e.get("latency_ms") or 0
            if e.get("status") == "error":
                r.has_error = True
    return [rows[tid] for tid in order]


def total_stats(events: list[dict]) -> dict:
    """Header aggregates: traces, llm_calls, in/out/total tokens, latency, has_error."""
    rows = summarize_traces(events)
    return {
        "traces": len(rows),
        "llm_calls": sum(r.llm_calls for r in rows),
        "in_tokens": sum(r.in_tokens for r in rows),
        "out_tokens": sum(r.out_tokens for r in rows),
        "total_tokens": sum(r.total_tokens for r in rows),
        "total_latency_ms": round(sum(r.total_latency_ms for r in rows), 1),
        "has_error": any(r.has_error for r in rows),
    }


def model_stats(events: list[dict]) -> list[ModelStat]:
    """One ModelStat per model, sorted by total (in+out) tokens descending."""
    acc: dict[str, ModelStat] = {}
    for e in events:
        if e.get("type") != "llm_call":
            continue
        m = e.get("model") or "?"
        st = acc.setdefault(m, ModelStat(model=m))
        st.calls += 1
        st.in_tokens += e.get("prompt_tokens") or 0
        st.out_tokens += e.get("completion_tokens") or 0
        st.total_latency_ms += e.get("latency_ms") or 0
    return sorted(acc.values(), key=lambda s: s.in_tokens + s.out_tokens, reverse=True)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_report_data.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/observability/report_data.py tests/test_report_data.py
git commit -m "feat: pure observability data prep (summaries, model stats, totals)"
```

---
### Task 2: `report_data.py` — chronological timeline

**Files:**
- Modify: `agent/observability/report_data.py` (add `Step` + `build_timeline`)
- Modify: `tests/test_report_data.py` (append timeline tests)

**Interfaces:**
- Consumes: same event dicts as Task 1.
- Produces: `Step` dataclass; `build_timeline(events) -> dict[str, list[Step]]` — per trace_id, steps sorted by `ts` ascending. (Used by Task 4's timeline cards.)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_report_data.py`:

```python
from agent.observability.report_data import Step, build_timeline


def _decision_events():
    return [
        {"type": "trace_start", "trace_id": "a", "phase": "trace", "ts": 1},
        {"type": "decision", "trace_id": "a", "phase": "classification", "ts": 10,
         "data": {"intent": "question", "complexity": "low", "reason": "simple"}},
        {"type": "llm_call", "trace_id": "a", "phase": "classification", "model": "m1",
         "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
         "latency_ms": 100, "status": "ok", "ts": 20},
        {"type": "decision", "trace_id": "a", "phase": "route", "ts": 30,
         "data": {"strategy": "s1", "orchestrate": True}},
        {"type": "decision", "trace_id": "a", "phase": "orchestration.planner", "ts": 35,
         "data": {"tasks": [{"title": "t1", "instruction": "i1"}]}},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.1", "model": "m2",
         "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
         "latency_ms": 50, "status": "ok", "ts": 40},
        {"type": "trace_end", "trace_id": "a", "answer_len": 30, "reject": False,
         "total_latency_ms": 150.0, "phase": "trace", "ts": 50},
    ]


def test_build_timeline_orders_by_ts():
    tl = build_timeline(_decision_events())
    assert list(tl) == ["a"]
    steps = tl["a"]
    assert [s.kind for s in steps] == ["decision", "llm_call", "decision", "decision",
                                       "llm_call", "result"]
    assert [s.phase for s in steps] == ["classification", "classification", "route",
                                        "orchestration.planner", "orchestration.worker.1", "trace"]
    assert steps[0].detail["type"] == "classification"
    assert steps[0].detail["data"]["intent"] == "question"
    assert steps[2].detail["type"] == "route"
    assert steps[3].detail["data"]["tasks"][0]["title"] == "t1"
    assert steps[4].detail["in_tokens"] == 2
    assert steps[5].detail["reject"] is False


def test_build_timeline_worker_decision_type():
    tl = build_timeline([{"type": "decision", "trace_id": "a", "phase": "orchestration.worker.2",
                          "ts": 1, "data": {"task": "fix bug"}}])
    steps = tl["a"]
    assert steps[0].kind == "decision"
    assert steps[0].detail["type"] == "worker"
    assert steps[0].detail["data"]["task"] == "fix bug"


def test_build_timeline_ignores_unknown_and_missing_trace():
    assert build_timeline([{"type": "wat", "trace_id": "x", "ts": 1},
                           {"type": "llm_call", "ts": 1}]) == {}


def test_build_timeline_sorted_ts_sequence():
    tl = build_timeline(_decision_events())
    steps = tl["a"]
    assert [s.ts for s in steps] == sorted(s.ts for s in steps)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_report_data.py -q`
Expected: FAIL — `ImportError` for `Step`/`build_timeline`

- [ ] **Step 3: Implement timeline**

Append to `agent/observability/report_data.py`:

```python
@dataclass
class Step:
    ts: int = 0
    kind: str = ""  # "decision" | "llm_call" | "result"
    phase: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _decision_type(phase: str) -> str:
    if phase == "classification":
        return "classification"
    if phase == "route":
        return "route"
    if phase == "orchestration.planner":
        return "plan"
    if phase.startswith("orchestration.worker"):
        return "worker"
    return "decision"


def build_timeline(events: list[dict]) -> dict[str, list[Step]]:
    """Per trace_id, chronological ordered steps. Unknown events and missing
    trace_ids are skipped. Never raises."""
    by_tid: dict[str, list[Step]] = {}
    for e in events:
        tid = e.get("trace_id")
        if not tid:
            continue
        typ = e.get("type")
        ts = e.get("ts") or 0
        ph = e.get("phase") or "?"
        if typ == "decision":
            step = Step(ts=ts, kind="decision", phase=ph,
                        detail={"type": _decision_type(ph), "data": e.get("data") or {}})
        elif typ == "llm_call":
            step = Step(ts=ts, kind="llm_call", phase=ph, detail={
                "model": e.get("model"), "in_tokens": e.get("prompt_tokens"),
                "out_tokens": e.get("completion_tokens"), "total_tokens": e.get("total_tokens"),
                "latency_ms": e.get("latency_ms"), "status": e.get("status"),
                "error": e.get("error")})
        elif typ == "trace_end":
            step = Step(ts=ts, kind="result", phase=ph, detail={
                "answer_len": e.get("answer_len"), "total_llm_calls": e.get("total_llm_calls"),
                "total_tokens": e.get("total_tokens"), "total_latency_ms": e.get("total_latency_ms"),
                "reject": e.get("reject")})
        else:
            continue
        if tid not in by_tid:
            by_tid[tid] = []
        by_tid[tid].append(step)
    for steps in by_tid.values():
        steps.sort(key=lambda s: s.ts)
    return {tid: steps for tid, steps in by_tid.items() if steps}
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_report_data.py -q`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/observability/report_data.py tests/test_report_data.py
git commit -m "feat: chronological per-trace timeline from events"
```

---
### Task 3: terminal line — `format_trace_summary`

**Files:**
- Modify: `agent/observability/tracing.py` (replace `format_trace_line`, remove `_fmt_tokens`)
- Modify: `agent/observability/__init__.py:5,22` (re-export)
- Modify: `agent/observability/patch.py:17,126` (call site)
- Modify: `tests/test_tracing.py` (replace imports + `test_fmt_tokens_and_trace_line`)

**Interfaces:**
- Consumes: `calls: list[dict]` — the llm_call events for one trace (as returned by `TraceStore.trace_llm_calls`, used at `patch.py:113`), each with `prompt_tokens`, `completion_tokens`, `latency_ms`.
- Produces: `format_trace_summary(trace_id: str, calls: list[dict]) -> str` returning `"[trace {id}] in={in} out={out} {s}s"`.

- [ ] **Step 1: Write the failing test** — in `tests/test_tracing.py`, replace the import of `format_trace_line` with `format_trace_summary` and replace the `test_fmt_tokens_and_trace_line` function:

```python
from agent.observability.tracing import (
    TraceStore,
    current_phase,
    current_trace_id,
    format_trace_summary,
    phase,
    read_events,
    trace_span,
)
```

```python
def test_format_trace_summary():
    calls = [
        {"phase": "classification", "prompt_tokens": 10, "completion_tokens": 5, "latency_ms": 300},
        {"phase": "strategy.direct", "prompt_tokens": 123, "completion_tokens": 45, "latency_ms": 2100},
    ]
    line = format_trace_summary("abc", calls)
    assert line == "[trace abc] in=133 out=50 2.4s"


def test_format_trace_summary_missing_usage_zero():
    line = format_trace_summary("abc", [{"phase": "route", "prompt_tokens": None,
                                          "completion_tokens": None, "latency_ms": None}])
    assert line == "[trace abc] in=0 out=0 0.0s"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tracing.py -k format_trace_summary -q`
Expected: FAIL — `ImportError: cannot import name 'format_trace_summary'`

- [ ] **Step 3: Implement** — in `agent/observability/tracing.py`, replace `_fmt_tokens` and `format_trace_line` (lines ~158-176) with:

```python
def format_trace_summary(trace_id: str, calls: list[dict]) -> str:
    in_tokens = sum(c.get("prompt_tokens") or 0 for c in calls)
    out_tokens = sum(c.get("completion_tokens") or 0 for c in calls)
    total_s = sum(c.get("latency_ms") or 0 for c in calls) / 1000
    return f"[trace {trace_id}] in={in_tokens} out={out_tokens} {total_s:.1f}s"
```

In `agent/observability/__init__.py`, change line 5 to `from .tracing import TraceStore, format_trace_summary, read_events` and in `__all__` replace `"format_trace_line"` with `"format_trace_summary"`.

In `agent/observability/patch.py`, change line 17 to import `format_trace_summary` instead of `format_trace_line`, and line 126 to `print(format_trace_summary(tid, calls))`.

- [ ] **Step 4: Run to verify they pass** — run the tracing + patch + install suites:

Run: `uv run pytest tests/test_tracing.py tests/test_observability_patch.py tests/test_observability_install.py -q`
Expected: PASS (all green, `_fmt_tokens`/`format_trace_line` gone)

- [ ] **Step 5: Commit**

```bash
git add agent/observability/tracing.py agent/observability/__init__.py agent/observability/patch.py tests/test_tracing.py
git commit -m "feat: terminal line shows in/out tokens + elapsed time only"
```

---
### Task 4: HTML renderer + HTML-only CLI

**Files:**
- Rewrite: `agent/observability/report.py`
- Rewrite: `tests/test_report.py`

**Interfaces:**
- Consumes: `summarize_traces`, `build_timeline`, `model_stats`, `total_stats` from `report_data.py` (Tasks 1-2); `read_events` from `tracing.py`.
- Produces: `build_html_report(events, *, default_collapsed=True) -> str`; `main(argv=None) -> int` (HTML-only). Re-exports `summarize_traces` for backward import-compat (the existing `tests/test_report.py::test_summarize_traces_aggregates` style import).

- [ ] **Step 1: Rewrite the tests** — replace `tests/test_report.py` entirely:

```python
import json

import pytest

from agent.observability.report import build_html_report, main, summarize_traces
from agent.observability.report_data import TraceSummary


def _events():
    return [
        {"type": "trace_start", "trace_id": "a", "question": "q1", "domain": "sw",
         "phase": "trace", "ts": 1},
        {"type": "decision", "trace_id": "a", "phase": "classification", "ts": 5,
         "data": {"intent": "question", "complexity": "low", "reason": "self explanatory"}},
        {"type": "llm_call", "trace_id": "a", "phase": "classification", "model": "m",
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
         "latency_ms": 100, "status": "ok", "ts": 10},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.direct", "model": "m",
         "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
         "latency_ms": 200, "status": "ok", "ts": 20},
        {"type": "trace_end", "trace_id": "a", "answer_len": 50, "total_llm_calls": 2,
         "total_tokens": 45, "total_latency_ms": 300.0, "reject": False,
         "phase": "trace", "ts": 300},
    ]


def test_summarize_traces_reexported():
    rows = summarize_traces(_events())
    assert isinstance(rows[0], TraceSummary)
    assert rows[0].in_tokens == 30
    assert rows[0].out_tokens == 15


def test_html_has_summary_and_labels():
    html = build_html_report(_events())
    for token in ("<html", "traces", "LLM calls", "q1", "strategy.direct", "<svg"):
        assert token in html


def test_html_decision_payload_visible():
    html = build_html_report(_events())
    assert "intent=question" in html
    assert "complexity=low" in html


def test_html_details_collapsed_by_default():
    html = build_html_report(_events())
    assert "<details>" in html
    assert "<details open" not in html
    html2 = build_html_report(_events(), default_collapsed=False)
    assert "<details open" in html2


def test_html_escapes_user_content():
    events = _events()
    events[0]["question"] = "<script>alert('x')</script>"
    events[1]["data"]["reason"] = "</b><script>"
    html = build_html_report(events)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_main_writes_report_and_prints_path(tmp_path, capsys):
    day_file = tmp_path / "trace-2026-08-11.jsonl"
    day_file.write_text("\n".join(json.dumps(e) for e in _events()) + "\n", encoding="utf-8")
    code = main(["report", "--data-dir", str(tmp_path), "--day", "2026-08-11"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Report written to" in out
    assert (tmp_path / "report.html").exists()
    assert "<html" in (tmp_path / "report.html").read_text(encoding="utf-8")


def test_build_cli_report_removed():
    with pytest.raises(ImportError):
        from agent.observability.report import build_cli_report
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_report.py -q`
Expected: FAIL — old `report.py` still has `build_cli_report` (import fails on `test_build_cli_report_removed`? No — the old module still imports it fine; failures instead come from `TraceSummary`/`in_tokens` and `_label_chart` absence). Either way: not all passing.

- [ ] **Step 3: Rewrite `report.py`**

```python
from __future__ import annotations

import argparse
import html
import sys
import warnings
from pathlib import Path

from .report_data import build_timeline, model_stats, summarize_traces, total_stats
from .tracing import read_events


def _short(s, n=28):
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _render_decision(detail: dict) -> str:
    typ = detail.get("type")
    data = detail.get("data") or {}
    if typ == "plan":
        tasks = data.get("tasks")
        if isinstance(tasks, list):
            return ", ".join(
                f'{i + 1}. {_short(html.escape(str(t.get("title") or "")))} '
                f'({_short(html.escape(str(t.get("instruction") or "")))})'
                for i, t in enumerate(tasks)
            )
        return ""
    parts = []
    for k in ("in_domain", "intent", "complexity", "reason", "strategy",
              "orchestrate", "reject_reason", "task"):
        if k in data:
            parts.append(f"{k}={html.escape(str(data[k]))}")
    return ", ".join(parts)


def _label_chart(items: list[tuple[str, float]], title: str, caption: str, *, unit: str) -> str:
    """Labeled horizontal-bar SVG with visible text labels and value annotations."""
    header = f'<h3>{html.escape(title)}</h3>'
    if not items:
        return header + f'<p class="caption">{html.escape(caption)} — no data</p>'
    mx = max(v for _, v in items) or 1
    boxes = []
    for i, (label, value) in enumerate(items):
        y = 14 + i * 18
        w = max(2, int(value / mx * 400))
        boxes.append(
            f'<text x="4" y="{y}" font-size="11">{_short(html.escape(label), 30)}</text>'
            f'<rect x="250" y="{y - 9}" width="{w}" height="11" fill="#4a90d9"/>'
            f'<text x="656" y="{y}" font-size="11">{value:g} {html.escape(unit)}</text>'
        )
    height = 14 + len(items) * 18
    svg = f'<svg width="720" height="{height}" xmlns="http://www.w3.org/2000/svg">{"".join(boxes)}</svg>'
    return header + svg + f'<p class="caption">{html.escape(caption)}</p>'


def _timeline_html(timeline: dict[str, list], trace_id: str) -> str:
    steps = timeline.get(trace_id, [])
    if not steps:
        return "<p>No steps recorded.</p>"
    rows = []
    for s in steps:
        if s.kind == "decision":
            rows.append(f'<li class="decision"><b>{html.escape(s.phase)}</b> {_render_decision(s.detail)}</li>')
        elif s.kind == "llm_call":
            d = s.detail
            badge = "error" if d.get("status") == "error" else "ok"
            err = f' <span class="err">{html.escape(str(d.get("error") or ""))}</span>' if d.get("error") else ""
            rows.append(
                f'<li class="call"><b>{html.escape(s.phase)}</b> '
                f'model={html.escape(str(d.get("model") or ""))} '
                f'in={d.get("in_tokens")} out={d.get("out_tokens")} total={d.get("total_tokens")} '
                f'{float(d.get("latency_ms") or 0):.0f}ms [{badge}]{err}</li>'
            )
        else:  # result
            d = s.detail
            rows.append(
                f'<li class="result"><b>result</b> answer={d.get("answer_len")} '
                f'reject={bool(d.get("reject"))} calls={d.get("total_llm_calls")} '
                f'tokens={d.get("total_tokens")} {float(d.get("total_latency_ms") or 0):.0f}ms</li>'
            )
    return "<ol>" + "".join(rows) + "</ol>"


def build_html_report(events: list[dict], *, default_collapsed: bool = True) -> str:
    summaries = summarize_traces(events)
    total = total_stats(events)
    timeline = build_timeline(events)
    open_attr = "" if default_collapsed else " open"

    trend_items = [(f"{_short(r.question)}", r.total_tokens) for r in summaries]
    model_items = [(m.model, m.in_tokens + m.out_tokens) for m in model_stats(events)]
    phase_lat: dict[str, float] = {}
    for steps in timeline.values():
        for s in steps:
            if s.kind == "llm_call":
                ph = s.phase or "?"
                phase_lat[ph] = phase_lat.get(ph, 0.0) + float(s.detail.get("latency_ms") or 0)
    phase_items = sorted(phase_lat.items(), key=lambda kv: kv[1], reverse=True)

    cards = []
    for r in summaries:
        cards.append(
            f'<details{open_attr}><summary>{html.escape(r.trace_id)} — '
            f'{html.escape(_short(r.question))} in={r.in_tokens} out={r.out_tokens} '
            f'total={r.total_tokens} {r.total_latency_ms / 1000:.1f}s'
            f'</summary>{_timeline_html(timeline, r.trace_id)}</details>'
        )

    meta = (f'{total["traces"]} traces, {total["llm_calls"]} LLM calls, '
            f'in={total["in_tokens"]} out={total["out_tokens"]} total={total["total_tokens"]} tokens, '
            f'{total["total_latency_ms"] / 1000:.1f}s total')

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ExpertForge observability</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1024px}}
details{{margin:.5rem 0;border:1px solid #ddd;padding:.5rem;border-radius:4px}}
summary{{cursor:pointer}}ol{{margin:.3rem 0 0}}li{{margin:.15rem 0}}
.caption{{color:#666;font-size:.85rem;margin:.2rem 0 1rem}}
.decision{{color:#1a5276}}.call{{color:#333}}.err{{color:#c0392b}}.result{{color:#7d6608}}
</style></head><body>
<h1>ExpertForge observability report</h1>
<p>{meta}</p>
{_label_chart(trend_items, "Token usage by trace", "Total tokens consumed per question.", unit="tokens")}
{_label_chart(model_items, "Model distribution", "Total in+out tokens per model.", unit="tokens")}
{_label_chart(phase_items, "Phase latency", "Summed latency per phase.", unit="ms")}
<h2>Traces</h2>
{''.join(cards)}
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "report":
        argv = argv[1:]
    parser = argparse.ArgumentParser(prog="agent.observability.report",
                                     description="Generate observability HTML report")
    parser.add_argument("--data-dir", default=".observability", help="trace JSONL directory")
    parser.add_argument("--day", default=None, help="YYYY-MM-DD filter")
    args = parser.parse_args(argv)
    events, bad = read_events(args.data_dir, day=args.day)
    if bad:
        print(f"note: {bad} unreadable trace line(s) skipped", file=sys.stderr)
    try:
        report = build_html_report(events)
        path = Path(args.data_dir) / "report.html"
        path.write_text(report, encoding="utf-8")
        print(f"Report written to {path}")
    except Exception as e:  # noqa: BLE001 - degrade, never crash
        warnings.warn(f"observability: failed to write HTML report: {e}")
        return 1
    return 0
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_report.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Smoke-test the HTML against real data**

Run: `uv run python -m agent.observability report --data-dir .observability` (the existing sample data dir)
Expected: prints `Report written to .observability/report.html`; opening the file shows the summary strip, three labeled charts with visible text, and collapsed trace cards. Also `uv run python -m agent.observability report --help` exits 0.

- [ ] **Step 6: Commit**

```bash
git add agent/observability/report.py tests/test_report.py
git commit -m "feat: HTML-only report with labeled charts and decision timelines"
```

---
### Task 5: README + full regression

**Files:**
- Modify: `README.md:94-102`

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Update README** — replace the observability bullets and command block:

```markdown
- Every LLM call's tokens and latency are recorded automatically (classification, routing, strategy, and orchestration phases) to per-day JSONL files under `data_dir`.
- During a REPL/`--ask` run a compact per-question line is printed after each answer, showing input tokens, output tokens, and elapsed time.
- After a run, generate the HTML report (self-contained, with per-trace step timelines):

```bash
uv run python -m agent.observability report               # writes data_dir/report.html
uv run python -m agent.observability report --day 2026-08-11
```
```

(Note the inner code fences: the outer Markdown fence ends before the inner ```` ``` ```` block — write it with the correct nesting so the README renders the three bullets and one code block.)

- [ ] **Step 2: Full regression**

Run: `uv run pytest -q`
Expected: all green (118 passed + new report_data/report tests, 4 skipped)

- [ ] **Step 3: End-to-end sanity (no API key)** — verify install → succinct terminal line wiring compiles end-to-end:

Run:
```bash
uv run python -c "
import threading
from agent.config import AgentConfig
from agent.observability import install
class C:
    model = 'm'
    def __init__(self):
        self._usage_local = threading.local()
    def chat_completion(self, messages, **kw):
        return 'x'
client, plugin = install(C(), AgentConfig(base_url='x', model='m', classifier_model='cm', domain_dir='d', observability=None), None)
print(type(client).__name__, plugin is None)
"
```
Expected: prints `C True` (observability disabled ⇒ passthrough). Also confirm `from agent.observability.tracing import format_trace_summary` imports cleanly.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: report is HTML-only; terminal lines show in/out tokens + time"
```

---
## Self-Review

**1. Spec coverage:**
- Remove CLI table + `--html` → Task 4 (deletes `build_cli_report`, `_svg_hbar`; `main` HTML-only).
- Terminal only in/out tokens + time, per question, trace-id retained → Task 3 (`format_trace_summary`, call site `patch.py`).
- HTML max detail: classification/route/plan/worker decisions as ordered timeline, collapsed by default → Task 2 (`build_timeline`) + Task 4 (`_timeline_html`, `<details>` default collapsed).
- Charts labeled + captioned → Task 4 `_label_chart` (visible `<text>` per bar + caption).
- Structured only, no raw text → no event schema change; renderer consumes existing fields only.
- Token-only (no cost) → no pricing anywhere.
- 3 charts → token usage by trace, model distribution (in+out), phase latency (summed ms) → Task 4.

**2. Placeholder scan:** no TBD/TODO; every step has real code and runnable commands.

**3. Type consistency:** `TraceSummary`/`ModelStat`/`Step` defined in Tasks 1-2 with `as_dict()`; consumed in Task 4 by the same names and field attributes (`r.in_tokens`, `s.kind`, `s.detail`). `format_trace_summary(trace_id, calls)` defined in Task 3, called at `patch.py:126`, re-exported in `__init__.py`. `summarize_traces` returns `list[TraceSummary]` in Tasks 1/2, re-exported in Task 4 via `from .report_data import ... summarize_traces`. `read_events(data_dir, *, day=None) -> (events, bad)` unchanged and used in Task 4 `main`. Field names match the event schema (tracing.py recorders). Commit messages match repo style (`feat:`/`docs:`).