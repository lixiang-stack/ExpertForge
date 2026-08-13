# Report HTML Rendering Refinements

**Date:** 2026-08-12
**Status:** Approved design (pending implementation plan)

## Problem Statement

Three rendering issues in the generated `report.html`:

1. **Auto-numbering in the per-trace timeline.** The trace body uses `<ol>` (ordered lists) at three levels — the stage list, the decision/call rows inside each stage, and the call rows nested under each worker — so every step is prefixed `1. 2. 3.`. The steps should have no numbers at all.

2. **Structured decisions rendered as `k=v, k=v`.** Classification and route decisions are shown as comma-separated `key=value` strings (e.g. `in_domain=True, intent=concept_explain, ...`). Structured output should instead be displayed as pretty-printed JSON. Route currently renders bare `k=v` (no `结构化输出` prefix) — it should use the same pretty-JSON treatment as classification.

3. **Phase-latency chart splits workers and strategies.** The chart aggregates by exact phase string, so `orchestration.worker.1`/`.2`/`.3` become separate bars and `strategy.teaching`/`strategy.analysis`/`strategy.direct` separate bars. Workers should merge into one `orchestration.worker` bar and strategies into one `strategy` bar.

## Design

Scope is confined to `agent/observability/report.py` rendering and `tests/test_report.py`. No data-model, schema, or recording-layer changes.

### 1. Remove all auto-numbering in the trace timeline

- Replace every `<ol>` in the trace-body renderer with `<ul>` plus CSS `list-style: none` on the relevant selectors, so no `1. 2. 3.` markers appear anywhere in a trace card.
- Levels affected:
  - the outer stage list (`_trace_body_html`),
  - the decision/call rows inside each stage (`_stage_li` body),
  - the call rows nested under each worker (`_stage_li` worker branch).
- Visual grouping is preserved by stage headers + indentation (nested `<ul>`), not by numbers.
- The `task<N>:` labels in the planner stage and `worker<N>:` labels in the worker stage are explicit text, not list numbering — they remain unchanged.

### 2. Decisions as pretty JSON

- Replace `_decision_kv` with a JSON renderer for the classification and route decision rows.
- Render as: `json.dumps(data, indent=2, ensure_ascii=False)`, then `html.escape` the entire JSON string, wrapped in a `<pre class="decision-json">` element.
- Both classification and route use this treatment (route no longer renders bare `k=v`).
- Planner `task<N>:` lines, worker `worker<N>: <task title>` blocks, degraded-plan `planning degraded`, and the `result`/`总结` block are unchanged.
- The decision keys rendered come from the decision step's `data` dict as recorded (classification: in_domain/intent/complexity/reason; route: in_domain/intent/complexity/strategy/orchestrate/reject_reason). `json.dumps` of the full `data` dict is the content — no key-filtering needed beyond what the recorder wrote.
- `None` values render as JSON `null` (per `ensure_ascii=False`, standard `json.dumps`).

### 3. Phase-latency chart normalization

- In the `phase_lat` aggregation loop in `build_html_report`, normalize phase keys before summing:
  - `orchestration.worker.N` → `orchestration.worker`
  - `strategy.X` → `strategy`
- All other phases (classification, route, orchestration.planner, orchestration.aggregate) keep their exact phase string.
- Result: the chart shows one `orchestration.worker` bar (sum of all worker calls' latency) and one `strategy` bar (sum of all strategy calls' latency).

## Data Flow

Unchanged: `read_events` → `report_data` (summarize/build_timeline/group_stages) → `report.py` renderer → `report.html`. Only the renderer's output formatting changes.

## Error Handling

- `json.dumps` on the decision `data` dict never raises for the recorded shapes (plain dict of primitives/lists). Any malformed `data` (non-dict) degrades to `{}` → renders `{}`.
- Existing never-raise render discipline unchanged.

## Testing

- **`tests/test_report.py`**:
  - Update `test_html_has_summary_and_labels` / structure assertions if they referenced `<ol>`; assert no `<ol` present in the trace body (numbers gone).
  - Update decision-payload assertions: classification and route render as JSON — assert `{"in_domain": true, ...}`-style content (`json.dumps` output substrings like `"intent": "question"`), not `intent=question`.
  - Add an assertion that the phase-latency chart contains a single `orchestration.worker` label and a single `strategy` label (no `orchestration.worker.1` / `strategy.teaching` bars) when fed worker + strategy events.
  - Existing tests unaffected by the JSON/numbering change (escaping, collapse-default, main-writes-file, write-failure) stay green.
- Full suite stays green.

## Out of Scope

- Any data-model or recording-layer change.
- Changing the trace card `<summary>` header, the top summary strip, or the token-trend / model-distribution charts.
- Planner task / worker block content or labels (other than the numbering removal already covered).