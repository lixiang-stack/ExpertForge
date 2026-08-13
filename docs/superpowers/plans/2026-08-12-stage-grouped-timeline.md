# Stage-Grouped Trace Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the observability HTML report's per-trace detail from a flat chronological list into stage-grouped sections (classification → route → strategy → planner → workers → aggregate → 总结), and prefix every displayed duration with `time=`.

**Architecture:** Pure grouping logic (`group_stages`) added to `report_data.py` turns the existing flat `build_timeline` step list into ordered `Stage` groups (workers collapsed into one stage with per-worker sub-blocks). The renderer in `report.py` consumes the stages and emits the stage-grouped HTML with `time=` labels on all durations. No event-schema or recording-layer changes.

**Tech Stack:** Python 3.12 (repo standard), stdlib only (`dataclasses`, `html`, `pathlib`), inline HTML/CSS.

## Global Constraints

- **Token-only:** no monetary cost anywhere.
- **Never raise into business code:** pure functions degrade on malformed input (`.get()` defaults, skip unknowns); render/write failures degrade with `warnings.warn` + non-zero exit.
- **No schema change, no new event fields, no recording-layer changes.**
- **HTML self-contained:** single file, inline CSS, no CDN/JS. All user content (question, task titles, reason strings, model names) passes through `html.escape`.
- **Every displayed duration is prefixed `time=`** and uses seconds with 1 decimal: per-call rows, trace card `<summary>` header, the report's top summary strip total, and the 总结 block.
- **Stage headers:** `classification阶段`, `route结果`, `strategy.<id>阶段`, `orchestration.planner阶段`, `orchestration.worker阶段`, `orchestration.aggregate阶段`, `总结`. Stages appear in first-seen order.
- **Planner tasks render title-only** (`task<N>: <title>`), instruction omitted.
- **Workers collapse into one `orchestration.worker` stage** with `worker<N>: <task title>` sub-blocks.
- **Charts and top summary-strip aggregation are unchanged** except the `time=` label on the total duration.

---
## File Structure

| File | Responsibility |
|------|----------------|
| `agent/observability/report_data.py` (modify) | Add `Stage`/`WorkerGroup` dataclasses + `group_stages(timeline)`. |
| `agent/observability/report.py` (modify) | Replace `_timeline_html`/`_render_decision` with stage-based rendering; add `time=` labels to call rows, card summaries, meta strip, 总结. |
| `tests/test_report_data.py` (modify) | Append `group_stages` tests. |
| `tests/test_report.py` (modify) | Update renderer tests to stage structure + `time=` assertions; add worker-stage test. |

---
### Task 1: `report_data.py` — stage grouping

**Files:**
- Modify: `agent/observability/report_data.py` (append `Stage`, `WorkerGroup`, `group_stages`)
- Modify: `tests/test_report_data.py` (append tests)

**Interfaces:**
- Consumes: `Step` dataclass and `build_timeline(events) -> dict[str, list[Step]]` (already in `report_data.py`; each `Step` has `.ts`, `.kind` ("decision"|"llm_call"|"result"), `.phase`, `.detail` dict).
- Produces: `Stage`/`WorkerGroup` dataclasses; `group_stages(timeline: dict[str, list[Step]]) -> dict[str, list[Stage]]`. Used by Task 2's renderer.

- [ ] **Step 1: Append the failing tests** — append to `tests/test_report_data.py`:

```python
from agent.observability.report_data import build_timeline, group_stages


def _simple_events():
    return [
        {"type": "trace_start", "trace_id": "a", "question": "q", "phase": "trace", "ts": 1},
        {"type": "llm_call", "trace_id": "a", "phase": "classification", "model": "m",
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
         "latency_ms": 100, "status": "ok", "ts": 10},
        {"type": "decision", "trace_id": "a", "phase": "classification", "ts": 11,
         "data": {"in_domain": True, "intent": "q", "complexity": "low", "reason": "r"}},
        {"type": "decision", "trace_id": "a", "phase": "route", "ts": 12,
         "data": {"strategy": "teaching", "orchestrate": False}},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.teaching", "model": "m",
         "prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50,
         "latency_ms": 200, "status": "ok", "ts": 20},
        {"type": "trace_end", "trace_id": "a", "answer_len": 100, "total_llm_calls": 2,
         "total_tokens": 65, "total_latency_ms": 300.0, "reject": False, "ts": 30},
    ]


def _worker_events():
    return [
        {"type": "decision", "trace_id": "a", "phase": "orchestration.worker.1", "ts": 1,
         "data": {"task": "t1"}},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.1", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
         "latency_ms": 10, "status": "ok", "ts": 2},
        {"type": "decision", "trace_id": "a", "phase": "orchestration.worker.2", "ts": 3,
         "data": {"task": "t2"}},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.2", "model": "m",
         "prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9,
         "latency_ms": 20, "status": "ok", "ts": 4},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.2", "model": "m",
         "prompt_tokens": 6, "completion_tokens": 7, "total_tokens": 13,
         "latency_ms": 30, "status": "ok", "ts": 5},
    ]


def test_group_stages_orders_stages_first_seen():
    stages = group_stages(build_timeline(_simple_events()))["a"]
    assert [s.title for s in stages] == ["classification", "route", "strategy.teaching", "result"]


def test_group_stages_classification_holds_decision_and_call():
    stages = group_stages(build_timeline(_simple_events()))["a"]
    classification = stages[0]
    assert [s.kind for s in classification.steps] == ["llm_call", "decision"]
    assert classification.steps[1].detail["data"]["intent"] == "q"


def test_group_stages_result_stage():
    stages = group_stages(build_timeline(_simple_events()))["a"]
    result = stages[-1]
    assert result.title == "result"
    assert result.steps[0].kind == "result"
    assert result.steps[0].detail["answer_len"] == 100


def test_group_stages_workers_collapse_into_one_stage():
    stages = group_stages(build_timeline(_worker_events()))["a"]
    assert len(stages) == 1
    assert stages[0].title == "orchestration.worker"
    assert [w.number for w in stages[0].workers] == [1, 2]
    assert stages[0].workers[0].task_title == "t1"
    assert stages[0].workers[1].task_title == "t2"
    assert len(stages[0].workers[0].steps) == 1
    assert len(stages[0].workers[1].steps) == 2


def test_group_stages_empty_timeline():
    assert group_stages({}) == {}


def test_group_stages_unknown_phase_own_stage():
    stages = group_stages(build_timeline([
        {"type": "llm_call", "trace_id": "a", "phase": "custom.phase", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
         "latency_ms": 5, "status": "ok", "ts": 1},
    ]))["a"]
    assert [s.title for s in stages] == ["custom.phase"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_report_data.py -k group_stages -q`
Expected: FAIL — `ImportError: cannot import name 'group_stages'`

- [ ] **Step 3: Implement** — append to `agent/observability/report_data.py`:

```python
@dataclass
class WorkerGroup:
    number: int = 0
    task_title: str = ""
    steps: list[Step] = field(default_factory=list)


@dataclass
class Stage:
    title: str = ""
    steps: list[Step] = field(default_factory=list)
    workers: list[WorkerGroup] = field(default_factory=list)


def _stage_title(step: Step) -> str:
    ph = step.phase or "?"
    if step.kind == "result":
        return "result"
    if ph == "classification":
        return "classification"
    if ph == "route":
        return "route"
    if ph.startswith("strategy."):
        return ph
    if ph == "orchestration.planner":
        return "orchestration.planner"
    if ph.startswith("orchestration.worker."):
        return "orchestration.worker"
    if ph == "orchestration.aggregate":
        return "orchestration.aggregate"
    return ph


def _worker_number(phase: str) -> int:
    try:
        return int(phase.rsplit(".", 1)[1])
    except (IndexError, ValueError):
        return 0


def _worker_task(step: Step) -> str:
    if step.kind == "decision":
        return str((step.detail.get("data") or {}).get("task") or "")
    return ""


def group_stages(timeline: dict[str, list[Step]]) -> dict[str, list[Stage]]:
    """Per trace, group flat steps into ordered Stage groups. Worker phases
    collapse into one stage with per-worker sub-blocks. Never raises."""
    result: dict[str, list[Stage]] = {}
    for tid, steps in timeline.items():
        by_title: dict[str, Stage] = {}
        order: list[str] = []
        for s in steps:
            title = _stage_title(s)
            if title not in by_title:
                by_title[title] = Stage(title=title)
                order.append(title)
            stage = by_title[title]
            if title == "orchestration.worker":
                n = _worker_number(s.phase)
                wg = next((w for w in stage.workers if w.number == n), None)
                if wg is None:
                    wg = WorkerGroup(number=n, task_title=_worker_task(s))
                    stage.workers.append(wg)
                if s.kind != "decision":
                    wg.steps.append(s)
            else:
                stage.steps.append(s)
        result[tid] = [by_title[t] for t in order]
    return result
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_report_data.py -q`
Expected: PASS (16 passed = 10 existing + 6 new)

- [ ] **Step 5: Commit**

```bash
git add agent/observability/report_data.py tests/test_report_data.py
git commit -m "feat: stage-grouped trace timeline data (group_stages)"
```

---
### Task 2: `report.py` — stage-grouped rendering + `time=` labels

**Files:**
- Modify: `agent/observability/report.py` (imports, delete `_render_decision`/`_timeline_html`, add `_call_row_html`/`_decision_kv`/`_stage_li`/`_trace_body_html`, update `build_html_report`)
- Modify: `tests/test_report.py`

**Interfaces:**
- Consumes: `group_stages`, `Stage`, `WorkerGroup` from `report_data.py` (Task 1); `build_timeline` (existing); `TraceSummary` (existing).
- Produces: updated `build_html_report(events, *, default_collapsed=True) -> str` with stage-grouped trace cards and `time=` labels on all durations.

- [ ] **Step 1: Update the tests** — in `tests/test_report.py`, replace the `_events()` fixture's classification decision to include `in_domain`, and update/add tests:

```python
def _events():
    return [
        {"type": "trace_start", "trace_id": "a", "question": "q1", "domain": "sw",
         "phase": "trace", "ts": 1},
        {"type": "decision", "trace_id": "a", "phase": "classification", "ts": 5,
         "data": {"in_domain": True, "intent": "question", "complexity": "low",
                  "reason": "self explanatory"}},
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


def _worker_events():
    return [
        {"type": "trace_start", "trace_id": "b", "question": "q2", "domain": "sw",
         "phase": "trace", "ts": 1},
        {"type": "decision", "trace_id": "b", "phase": "orchestration.worker.1", "ts": 2,
         "data": {"task": "task alpha"}},
        {"type": "llm_call", "trace_id": "b", "phase": "orchestration.worker.1", "model": "m",
         "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
         "latency_ms": 50, "status": "ok", "ts": 3},
        {"type": "decision", "trace_id": "b", "phase": "orchestration.worker.2", "ts": 4,
         "data": {"task": "task beta"}},
        {"type": "llm_call", "trace_id": "b", "phase": "orchestration.worker.2", "model": "m",
         "prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11,
         "latency_ms": 60, "status": "ok", "ts": 5},
        {"type": "trace_end", "trace_id": "b", "answer_len": 30, "total_llm_calls": 2,
         "total_tokens": 18, "total_latency_ms": 110.0, "reject": False,
         "phase": "trace", "ts": 6},
    ]


def test_html_has_summary_and_labels():
    html = build_html_report(_events())
    for token in ("<html", "traces", "LLM calls", "q1", "strategy.direct", "<svg",
                  "classification阶段", "time="):
        assert token in html


def test_html_decision_payload_visible():
    html = build_html_report(_events())
    assert "结构化输出" in html
    assert "intent=question" in html
    assert "complexity=low" in html


def test_html_worker_stage_nested():
    html = build_html_report(_worker_events())
    assert "orchestration.worker阶段" in html
    assert "worker1: task alpha" in html
    assert "worker2: task beta" in html
    assert html.count("class=\"worker\"") == 2


def test_html_has_time_labels():
    html = build_html_report(_events())
    assert "time=0.1s" in html  # classification call 100ms
    assert "time=0.2s" in html  # strategy.direct call 200ms
    assert "time=0.3s" in html  # trace summary header (300ms)
    assert "time=0.3s total" in html  # top summary strip
```

Keep the other existing tests unchanged (`test_summarize_traces_reexported`, `test_html_details_collapsed_by_default`, `test_html_escapes_user_content`, `test_main_writes_report_and_prints_path`, `test_build_cli_report_removed`, `test_main_write_failure_degrades`).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_report.py -q`
Expected: FAIL — the new assertions (`classification阶段`, `time=`, `worker1: task alpha`, `结构化输出`) are not satisfied by the current flat renderer.

- [ ] **Step 3: Implement** — in `agent/observability/report.py`:

Change the import line (line 9) to:

```python
from .report_data import Stage, build_timeline, group_stages, model_stats, summarize_traces, total_stats
```

Delete `_render_decision` (lines 18-38) and `_timeline_html` (lines 61-86). Add these functions in their place:

```python
def _call_row_html(step) -> str:
    d = step.detail
    badge = "error" if d.get("status") == "error" else "ok"
    err = f' <span class="err">{html.escape(str(d.get("error") or ""))}</span>' if d.get("error") else ""
    secs = float(d.get("latency_ms") or 0) / 1000
    return (f'<li class="call">model={html.escape(str(d.get("model") or ""))} '
            f'in={d.get("in_tokens")} out={d.get("out_tokens")} time={secs:.1f}s [{badge}]{err}</li>')


def _decision_kv(data: dict) -> str:
    return ", ".join(
        f"{k}={html.escape(str(v))}" for k, v in data.items() if v is not None
    )


def _stage_li(stage: Stage, summary) -> str:
    title = stage.title
    if title == "route":
        label = "route结果"
    elif title == "result":
        label = "总结"
    else:
        label = f"{title}阶段"
    header = f'<b>{html.escape(label)}</b>'
    if title == "result":
        result_step = next((s for s in stage.steps if s.kind == "result"), None)
        answer_len = (result_step.detail or {}).get("answer_len") if result_step else ""
        return (f'<li class="stage">{header} answer_len={answer_len} reject={summary.reject}, '
                f'total in={summary.in_tokens} out={summary.out_tokens} tokens, '
                f'time={summary.total_latency_ms / 1000:.1f}s</li>')
    body: list[str] = []
    if stage.workers:
        for w in stage.workers:
            calls = "".join(_call_row_html(s) for s in w.steps)
            body.append(
                f'<li class="worker"><b>worker{w.number}: {html.escape(w.task_title)}</b>'
                f'<ol>{calls}</ol></li>'
            )
    else:
        for s in stage.steps:
            if s.kind == "decision":
                data = s.detail.get("data") or {}
                if title == "classification":
                    kv = _decision_kv({k: data.get(k) for k in
                                       ("in_domain", "intent", "complexity", "reason")})
                    body.append(f'<li class="decision">结构化输出: {kv}</li>')
                elif title == "route":
                    kv = _decision_kv({k: data.get(k) for k in
                                       ("in_domain", "intent", "complexity", "strategy",
                                        "orchestrate", "reject_reason")})
                    body.append(f'<li class="decision">{kv}</li>')
                elif title == "orchestration.planner":
                    if data.get("degraded"):
                        body.append('<li class="decision">planning degraded</li>')
                    else:
                        tasks = data.get("tasks")
                        if isinstance(tasks, list):
                            for i, t in enumerate(tasks):
                                if isinstance(t, dict):
                                    body.append(
                                        f'<li class="decision">task{i + 1}: '
                                        f'{html.escape(str(t.get("title") or ""))}</li>')
                else:
                    body.append(f'<li class="decision">{_decision_kv(data)}</li>')
            elif s.kind == "llm_call":
                body.append(_call_row_html(s))
    if body:
        return f'<li class="stage">{header}<ol>{"".join(body)}</ol></li>'
    return f'<li class="stage">{header}</li>'


def _trace_body_html(stages: list[Stage], summary) -> str:
    if not stages:
        return "<p>No steps recorded.</p>"
    return "<ol class='stages'>" + "".join(_stage_li(s, summary) for s in stages) + "</ol>"
```

Update `build_html_report` — change the timeline line and the card-building loop:

```python
    timeline = build_timeline(events)
    grouped = group_stages(timeline)
```

```python
    cards = []
    for r in summaries:
        stages = grouped.get(r.trace_id, [])
        cards.append(
            f'<details{open_attr}><summary>{html.escape(r.trace_id)} — '
            f'{html.escape(_short(r.question))} in={r.in_tokens} out={r.out_tokens} '
            f'total={r.total_tokens} time={r.total_latency_ms / 1000:.1f}s'
            f'</summary>{_trace_body_html(stages, r)}</details>'
        )
```

Update the meta strip line (line 114-116) to prefix the total duration:

```python
    meta = (f'{total["traces"]} traces, {total["llm_calls"]} LLM calls, '
            f'in={total["in_tokens"]} out={total["out_tokens"]} total={total["total_tokens"]} tokens, '
            f'time={total["total_latency_ms"] / 1000:.1f}s total')
```

Note: `phase_lat` in `build_html_report` still iterates the flat `timeline.values()` for the phase-latency chart — leave that chart loop as-is (it already reads `s.detail["latency_ms"]` from flat steps).

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_report.py tests/test_report_data.py -q`
Expected: PASS (report tests + data tests all green)

- [ ] **Step 5: Smoke-test against real data**

Run: `uv run python -m agent.observability report --data-dir .observability`
Expected: prints `Report written to .observability/report.html`. Open the file: each trace card shows stage sections (`classification阶段`, `route结果`, `strategy.*阶段`, `orchestration.worker阶段` with `workerN:` blocks, `总结`), every duration labeled `time=<s>s`, planner tasks as `taskN:` lines, and the top strip shows `time=<s>s total`.

- [ ] **Step 6: Full suite**

Run: `uv run pytest -q`
Expected: all green (134 passed, 4 skipped baseline + any new tests)

- [ ] **Step 7: Commit**

```bash
git add agent/observability/report.py tests/test_report.py
git commit -m "feat: stage-grouped trace rendering with time= labels"
```

---
## Self-Review

**1. Spec coverage:**
- `time=` prefix on all durations → Task 2 (`_call_row_html` time=, card summary time=, meta `time=...s total`, 总结 time=).
- Stage grouping (classification/route/strategy.X/planner/worker/aggregate/总结) → Task 1 `group_stages` + Task 2 `_stage_li` headers (`route结果`, `总结`, `*阶段`).
- Structured output inline (`结构化输出: k=v`; route bare kv; planner `taskN:`; worker `workerN: title`) → Task 2.
- Workers one stage with sub-blocks → Task 1 `WorkerGroup` + Task 2 `_stage_li` worker branch.
- Charts/summary-strip unchanged except `time=` → Task 2 leaves chart loops and aggregation untouched.
- Pure, never-raise; html.escape; token-only → preserved (no schema change; `_decision_kv`/`_stage_li` escape user strings; `group_stages` is pure).

**2. Placeholder scan:** no TBD/TODO; every step has real code and runnable commands.

**3. Type consistency:** `Stage`/`WorkerGroup`/`group_stages` defined in Task 1, consumed in Task 2 by the same names; `Stage.workers`, `WorkerGroup.number/task_title/steps`, `Stage.title/steps` match between the dataclass and renderer. `_trace_body_html(stages, summary)` uses `TraceSummary` fields `reject/in_tokens/out_tokens/total_latency_ms`. `build_timeline`/`build_html_report` signatures unchanged. The Task 2 test count assumes existing tests stay green with the new structure — `test_html_escapes_user_content` still passes because `_stage_li` escapes the decision reason via `_decision_kv`.