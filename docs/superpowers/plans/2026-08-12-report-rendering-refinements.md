# Report HTML Rendering Refinements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine the observability HTML report's rendering: remove all auto-numbering from the per-trace timeline, render classification/route decisions as pretty multi-line JSON, and merge worker/strategy phases in the phase-latency chart.

**Architecture:** All three changes are confined to the renderer in `agent/observability/report.py` and its tests in `tests/test_report.py`. No data-model, schema, or recording-layer changes. A JSON renderer helper replaces the `k=v` decision formatter; the phase-latency aggregation loop normalizes phase keys before summing.

**Tech Stack:** Python 3.12 (repo standard), stdlib (`json`, `html`), inline HTML/CSS.

## Global Constraints

- **Token-only:** no monetary cost anywhere.
- **Never raise into business code:** pure helpers degrade on malformed input; render/write failures degrade with `warnings.warn` + non-zero exit.
- **No schema change, no new event fields, no recording-layer changes.**
- **HTML self-contained:** single file, inline CSS, no CDN/JS. All user content (question, task titles, reason strings, model names, decision values) passes through `html.escape`.
- **No auto-numbering in the trace timeline** (no `1. 2. 3.` markers at any level).
- **Classification AND route decisions render as pretty JSON** in `<pre class="decision-json">` (route no longer bare `k=v`).
- **Phase-latency chart normalizes keys:** `orchestration.worker.N` → `orchestration.worker`; `strategy.X` → `strategy`. All other phases keep exact names.
- Planner `task<N>:` lines, worker `worker<N>:` blocks, degraded-plan, and 总结 blocks unchanged except the numbering removal.

---
## File Structure

| File | Responsibility |
|------|----------------|
| `agent/observability/report.py` (modify) | Remove `<ol>` numbering; add `_decision_json_html`; use JSON for classification/route/else decisions; normalize `phase_lat` keys; CSS `ol`→`ul`. |
| `tests/test_report.py` (modify) | Update decision-payload assertions to JSON; add no-`<ol>` and phase-merge tests. |

---
### Task 1: report rendering refinements

**Files:**
- Modify: `agent/observability/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Consumes: `build_html_report(events, *, default_collapsed=True) -> str` (existing), `Stage`, `group_stages`, `summarize_traces`, `total_stats`, `model_stats`, `build_timeline` (all existing in `report_data.py`); `Step.detail` shapes as produced by `build_timeline` (decision steps have `detail = {"type": ..., "data": {...}}`).
- Produces: updated `build_html_report` with no list numbers, JSON decisions, and merged phase-latency bars.

- [ ] **Step 1: Update the tests** — in `tests/test_report.py`:

Replace `test_html_decision_payload_visible` with JSON assertions:

```python
def test_html_decision_payload_json():
    html = build_html_report(_events())
    assert "结构化输出" in html
    assert "&quot;intent&quot;: &quot;question&quot;" in html
    assert "&quot;complexity&quot;: &quot;low&quot;" in html
    assert "intent=question" not in html
```

Add a no-numbering test:

```python
def test_html_no_list_numbers():
    html = build_html_report(_events())
    assert "<ol" not in html
    assert "class='stages'" in html
```

Add a phase-latency merge test (worker + strategy phases collapse):

```python
def test_html_phase_latency_merges_workers_and_strategies():
    events = [
        {"type": "trace_start", "trace_id": "a", "question": "q", "domain": "sw",
         "phase": "trace", "ts": 1},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.1", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
         "latency_ms": 30, "status": "ok", "ts": 2},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.2", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
         "latency_ms": 70, "status": "ok", "ts": 3},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.teaching", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
         "latency_ms": 100, "status": "ok", "ts": 4},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.analysis", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
         "latency_ms": 200, "status": "ok", "ts": 5},
        {"type": "trace_end", "trace_id": "a", "answer_len": 1, "total_llm_calls": 4,
         "total_tokens": 8, "total_latency_ms": 400.0, "reject": False,
         "phase": "trace", "ts": 6},
    ]
    html = build_html_report(events)
    assert "100 ms" in html   # orchestration.worker = 30 + 70
    assert "300 ms" in html   # strategy = 100 + 200
    assert "30 ms" not in html
    assert "70 ms" not in html
```

Keep all other existing tests unchanged (`test_summarize_traces_reexported`, `test_html_has_summary_and_labels`, `test_html_worker_stage_nested`, `test_html_has_time_labels`, `test_html_details_collapsed_by_default`, `test_html_escapes_user_content`, `test_main_writes_report_and_prints_path`, `test_build_cli_report_removed`, `test_main_write_failure_degrades`).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_report.py -q`
Expected: FAIL — the current renderer still emits `<ol>` and `k=v` decisions and separate worker/strategy bars.

- [ ] **Step 3: Implement** — in `agent/observability/report.py`:

Add `import json` at the top (after the existing `import html`):

```python
import json
```

Replace `_decision_kv` (lines ~27-30) with a JSON renderer:

```python
def _decision_json_html(data: dict) -> str:
    text = json.dumps(data if isinstance(data, dict) else {}, indent=2, ensure_ascii=False)
    return "<pre class='decision-json'>" + html.escape(text) + "</pre>"
```

In `_stage_li`, replace the classification/route branches (lines ~60-68) with a single JSON branch, and the generic else branch (line ~81) with the JSON renderer:

```python
            if s.kind == "decision":
                data = s.detail.get("data") or {}
                if title == "classification" or title == "route":
                    body.append(f'<li class="decision">结构化输出{_decision_json_html(data)}</li>')
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
                    body.append(f'<li class="decision">{_decision_json_html(data)}</li>')
```

Remove all auto-numbering: in `_stage_li`, change the worker-branch `<ol>` (line ~54) and the body `<ol>` (line ~85) to `<ul>`; in `_trace_body_html`, change `<ol class='stages'>` (line ~92) to `<ul class='stages'>`. In the CSS block, change `ol{{margin:.3rem 0 0}}li{{margin:.15rem 0}}` to:

```css
ul{{margin:.3rem 0 0;list-style:none;padding-left:0}}li{{margin:.15rem 0}}ul ul{{padding-left:1.2rem}}
```

In `build_html_report`, normalize the phase-latency aggregation (lines ~124-130):

```python
    def _chart_phase(ph: str) -> str:
        if ph.startswith("orchestration.worker."):
            return "orchestration.worker"
        if ph.startswith("strategy."):
            return "strategy"
        return ph

    phase_lat: dict[str, float] = {}
    for steps in timeline.values():
        for s in steps:
            if s.kind == "llm_call":
                ph = _chart_phase(s.phase or "?")
                phase_lat[ph] = phase_lat.get(ph, 0.0) + float(s.detail.get("latency_ms") or 0)
    phase_items = sorted(phase_lat.items(), key=lambda kv: kv[1], reverse=True)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_report.py tests/test_report_data.py -q`
Expected: PASS (report tests + data tests all green)

- [ ] **Step 5: Smoke-test against real data**

Run: `uv run python -m agent.observability report --data-dir .observability`
Expected: prints `Report written to .observability/report.html`. Open the file: trace cards show no `1. 2. 3.` numbering; classification/route decisions render as indented JSON; the Phase latency chart shows one `orchestration.worker` bar and one `strategy` bar (no `orchestration.worker.1` / `strategy.teaching` bars).

- [ ] **Step 6: Full suite**

Run: `uv run pytest -q`
Expected: all green (142 passed, 4 skipped baseline, adjusted for the replaced/added tests)

- [ ] **Step 7: Commit**

```bash
git add agent/observability/report.py tests/test_report.py
git commit -m "feat: report rendering refinements (no step numbers, JSON decisions, merged phase latency)"
```

---
## Self-Review

**1. Spec coverage:**
- Remove auto-numbering at all three levels → Task 1 (`<ol>`→`<ul>` in `_trace_body_html`, `_stage_li` body + worker branch; CSS `list-style:none`).
- Decisions as pretty JSON (classification + route, no `k=v`) → Task 1 (`_decision_json_html` via `json.dumps(indent=2, ensure_ascii=False)` + `html.escape` in `<pre class='decision-json'>`; route joins the JSON branch).
- Phase-latency merge: `orchestration.worker.N`→`orchestration.worker`, `strategy.X`→`strategy` → Task 1 (`_chart_phase` in the aggregation loop).
- Planner/worker/总结 unchanged except numbering → Task 1 leaves those branches intact.
- Tests → `test_html_decision_payload_json`, `test_html_no_list_numbers`, `test_html_phase_latency_merges_workers_and_strategies`.

**2. Placeholder scan:** no TBD/TODO; every step has real code and runnable commands.

**3. Type consistency:** `_decision_json_html(data: dict) -> str` defined and used in the three decision branches; `_chart_phase` defined and used in the one aggregation loop; `build_html_report` signature unchanged; `_decision_kv` removed with no remaining references (the three decision branches and the else branch all now use `_decision_json_html`). The phase-merge test's value assertions (`100 ms`, `300 ms`) match the SVG `{value:g} ms` format in `_label_chart`. Existing tests that reference `strategy.direct` (trace-body stage header) and `time=` remain valid since those paths are unchanged.

**Note:** `html.escape` (default `quote=True`) escapes JSON quotes to `&quot;`, so the decision-JSON tests assert the escaped form (`&quot;intent&quot;: &quot;question&quot;`); browsers render it as `"intent": "question"`. The stage list is `<ul class='stages'>` (single quotes), so the no-numbering test asserts `class='stages'`.