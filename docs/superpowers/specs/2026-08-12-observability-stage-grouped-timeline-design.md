# Observability HTML Report: Stage-Grouped Trace Timeline

**Date:** 2026-08-12
**Status:** Approved design (pending implementation plan)

## Problem Statement

Two issues in the generated `report.html`:

1. **Time values lack a `time=` label.** Durations are rendered as bare numbers (e.g. `2483ms`, `15.5s`, `115.3s total`), making it unclear the number is a time. Every duration must be prefixed with `time=`. (The time *math* is correct — this is purely a labeling change; the user confirmed no calculation fix is needed.)

2. **The per-trace detail is not stage-grouped and is hard to read.** The current trace body is a flat chronological `<ol>` mixing decision rows and call rows. It should present the processing flow as clear stages (classification → route → strategy → planner → workers → aggregate → summary), each showing its structured decision output and its model calls, so a reader can follow exactly what each step did.

## Design Principles

- **Flow clarity:** a trace reads top-to-bottom as the actual processing pipeline, stage by stage.
- **Structured detail:** decision outputs (classification intent/complexity/reason, route strategy/orchestrate, planner tasks, worker task titles) are shown inline under their stage.
- **Simple wins:** no new event fields, no schema change, no recording-layer changes. Pure presentation rework on the existing events + `time=` relabeling.
- **Token-only:** no monetary cost anywhere (unchanged).
- **Never raise into business:** all rendering degrades with `warnings.warn` (unchanged).

## Architecture

Pure grouping logic added to `report_data.py`; the renderer in `report.py` consumes it. No other modules change.

### 1. `report_data.py` — stage grouping (pure)

New dataclass and function:

- **`Stage`** dataclass: `{title: str, steps: list[Step]}` (reuses the existing `Step` dataclass from Task 2 of the redesign).
- **`group_stages(timeline: dict[str, list[Step]]) -> dict[str, list[Stage]]`** — per trace, converts the flat chronological step list into ordered stages by grouping steps by their phase. Pure; never raises.

Stage derivation (first-seen order preserved):

| Phase | Stage title | Steps kept in stage |
|---|---|---|
| `classification` | `classification` | decision + llm_calls |
| `route` | `route` | decision + llm_calls |
| `strategy.<id>` | `strategy.<id>` | llm_calls |
| `orchestration.planner` | `orchestration.planner` | decision (tasks) + llm_calls |
| `orchestration.worker.<n>` | `orchestration.worker` (one stage) | llm_calls per worker, with worker-number + task-title grouping |
| `orchestration.aggregate` | `orchestration.aggregate` | llm_calls |
| `trace_end` | `result` (总结) | the trace_end payload |

- Worker steps are grouped under a single `orchestration.worker` stage; each worker `<n>` becomes a sub-block keyed by its phase suffix with its decision task title + its llm_calls.
- Any other phase → its own stage titled by the phase string.
- Stages appear in first-seen order of their phase.

### 2. `report.py` — stage-grouped rendering + `time=` labels

Rewrite the trace-card body renderer (`_timeline_html` → stage-based renderer) and add `time=` labels. Structure per trace (matching the approved example):

```
classification阶段
  model=deepseek-v4-flash in=358 out=47 time=2.5s [ok]
  结构化输出: in_domain=True, intent=concept_explain, complexity=simple, reason=...
route结果
  in_domain=True, intent=concept_explain, complexity=simple, strategy=teaching, orchestrate=False, reject_reason=
strategy.teaching阶段
  model=deepseek-v4-flash in=178 out=1521 time=13.0s [ok]
orchestration.planner阶段
  model=deepseek-v4-flash in=224 out=477 time=5.7s [ok]
  task1: Go 调度器与内存模型的协同机制
  task2: Go 内存模型与垃圾回收（GC）的交互
orchestration.worker阶段
  worker1: Go 调度器与内存模型的协同机制
    model=deepseek-v4-flash in=239 out=1837 time=14.6s [ok]
  worker2: Go 内存模型与垃圾回收（GC）的交互
    model=deepseek-v4-flash in=250 out=2238 time=20.3s [ok]
orchestration.aggregate阶段
  model=deepseek-v4-flash in=5665 out=2845 time=15.0s [ok]
总结
  answer_len=5918 reject=False, total in=8929 out=12916 tokens, time=71.3s
```

Rendering rules:

- **Stage header:** stage title + `阶段`; `route` renders as `route结果`; `result` renders as `总结`.
- **llm_call rows:** `model=<name> in=<n> out=<n> time=<s>s [ok]` or `[error]` + error string. Time in seconds, 1 decimal.
- **Decision payloads:** `结构化输出: k=v, k=v, ...` (classification: in_domain/intent/complexity/reason; route: in_domain/intent/complexity/strategy/orchestrate/reject_reason). Planner stage renders `task<N>: <title>` lines (task instruction omitted — title only, per approved design). Worker stage renders each worker as `worker<N>: <task title>` then its model rows indented beneath.
- **总结:** answer_len, reject flag, total in/out tokens, `time=<s>s` (from `trace_end.total_latency_ms`).
- Trace card `<summary>` header duration and the report's top summary strip duration also get the `time=` prefix.
- `<details>` cards remain collapsed by default; worker sub-blocks are plain indented text (no extra nesting).
- All user content stays `html.escape`'d.

### 3. Top-level summary strip and charts — unchanged

The three labeled charts and the header summary numbers stay exactly as they are now, except the total-duration value gets the `time=` prefix.

## Data Flow

1. Agent run writes events (unchanged).
2. `report.py` reads events → `report_data.build_timeline` (existing) → new `report_data.group_stages` → renderer emits stage-grouped HTML.
3. `python -m agent.observability report` writes `report.html` (unchanged CLI).

## Error Handling

- `group_stages` is pure; malformed steps/events degrade (`.get()` defaults, skip unknowns) and never raise.
- Render/write failures degrade with `warnings.warn` + non-zero exit (unchanged from current design).

## Testing

- **`tests/test_report_data.py`** — new `group_stages` tests:
  - classification + route + strategy grouped into stages in first-seen order;
  - worker phases collapse into one `orchestration.worker` stage with per-worker sub-blocks ordered by worker number;
  - planner stage carries the task list; `trace_end` becomes a `result` stage;
  - malformed/unexpected phase falls back to its own stage; empty timeline → no stages.
- **`tests/test_report.py`** — renderer tests updated:
  - stage headers present (`classification阶段`, `orchestration.worker阶段`, `总结`);
  - `time=` present on call rows, trace summary header, and top summary strip;
  - `task1:` and `worker1:` labels present; error row shows `[error]`;
  - existing assertions updated from the old flat format (e.g. `strategy.direct`, `intent=question` still present but under the new structure);
  - escaping / collapse-default / main-writes-file tests remain.
- Full suite stays green.

## Out of Scope

- Monetary cost / pricing.
- Recording real elapsed wall-clock time (user confirmed current time math is correct; only `time=` labeling is requested).
- Capturing prompt/response text.
- Changes to event schema or the recording layer.
- Changes to the top-level charts or summary-strip aggregation (beyond the `time=` label).