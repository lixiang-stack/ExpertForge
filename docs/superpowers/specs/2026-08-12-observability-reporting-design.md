# Observability Reporting Redesign

**Date:** 2026-08-12
**Status:** Approved design (pending implementation plan)

## Problem Statement

The current observability reporting layer does not match how it is meant to be used:

1. **CLI table mode** (`python -m agent.observability report` default output) is not wanted. The report command should produce HTML only.
2. **Terminal trace line** shows a per-phase breakdown (`[trace abc] classification 0.1s/1k | route ... | total N tok Ss`). The terminal should show only simple usage numbers: input tokens, output tokens, and elapsed time.
3. **HTML overview charts** (Token trend by trace, Model distribution, Phase latency (ms)) render as thin hover-only SVG rectangles with no visible labels, so "看不出是在做什么". They must become readable labeled charts with text labels and value annotations.
4. **HTML per-trace detail** currently shows only a phase → tokens → latency table. It should surface the full decision data already recorded in the events (classification result, route decision, planner tasks, worker task titles) as an ordered step-by-step timeline so a user can trace what each step did.

## Design Principles

- **Terminal = simple.** A single concise line per answered question. No tables, no per-phase detail, no cost (token-only by explicit user decision).
- **HTML = detailed.** Structured-only event data rendered as a full timeline, with large detail collapsed by default.
- **No new event fields.** All needed data (token usage, latency, model, decision payloads) is already written to the JSONL trace files. No raw prompt/response text is captured.
- **Never raise into business code.** Same discipline as the current plugin: all observability degradation uses `warnings.warn`.
- **Token-only.** No monetary cost calculation anywhere.

## Architecture

Two modules replace the current single `report.py` data+renderer, matching the repo's existing pattern of a pure data module per report/rendering:

### 1. `agent/observability/report_data.py` — pure data preparation

No rendering, no I/O (reads nothing, writes nothing). All functions are pure aggregates over the event list returned by `read_events()`, tolerant of malformed events (`.get()` defaults, unknown types ignored).

Dataclasses (each with `as_dict()` for JSON serialization):

- **`TraceSummary`** — one per `trace_id`, first-seen order. Fields: `trace_id`, `question`, `domain`, `in_tokens`, `out_tokens`, `total_tokens`, `llm_calls`, `total_latency_ms`, `reject`, `has_error`.
  - `in_tokens` = sum of `prompt_tokens` across the trace's `llm_call` events; `out_tokens` = sum of `completion_tokens`; `total_tokens` = sum of `total_tokens` (kept even though not always equal to in+out, because providers may report total differently).
- **`Step`** — one entry in a trace's ordered timeline. Fields: `ts`, `kind` (`"decision"` | `"llm_call"` | `"result"`), `phase`, plus kind-specific payload:
  - decision payload: `{type: "classification"|"route"|"plan"|"worker", detail: dict}` where detail carries the decision record's `data`.
  - llm_call payload: `{model, in_tokens, out_tokens, total_tokens, latency_ms, status, error}`.
  - result payload: `{answer_len, total_llm_calls, total_tokens, total_latency_ms, reject}` (from `trace_end`).
- **`ModelStat`** — one per model. Fields: `model`, `calls`, `in_tokens`, `out_tokens`, `total_latency_ms`.

Functions:

- `summarize_traces(events) -> list[TraceSummary]`
- `build_timeline(events) -> dict[str, list[Step]]` — per `trace_id`, steps sorted by `ts` ascending, so decisions and llm_calls interleave in true chronological order.
- `model_stats(events) -> list[ModelStat]` — sorted by total tokens descending.
- `total_stats(events) -> dict` — header aggregates: `{traces, llm_calls, in_tokens, out_tokens, total_tokens, total_latency_ms, has_error}`.

### 2. `agent/observability/report.py` — rendering + CLI

Deletes `build_cli_report`, `_svg_hbar` (replaced), and the `--html` flag. Keeps `main(argv)` as the CLI entry point. `summarize_traces` is re-exported here from `report_data` for import-compatibility with existing tests.

Functions:

- `build_html_report(events, *, default_collapsed=True) -> str` — single self-contained HTML file, inline CSS only (no CDN, no JS libraries). Structure:
  1. **Header:** title + summary meta (trace count, date range).
  2. **Summary strip:** total traces, total LLM calls, total in tokens, total out tokens, total tokens, total elapsed time — plain readable numbers.
  3. **Three labeled charts** (SVG with visible `<text>` labels and value annotations, not `<title>`-only), each followed by a one-line caption:
     - **Per-trace token usage**: one bar per trace, label = truncated question text, value = total tokens.
     - **Model distribution**: one bar per model, label = model name, value = in+out tokens.
     - **Phase latency**: one bar per phase (chronological), label = phase name, value = summed latency ms.
  4. **Per-trace cards:** each trace is a `<details>` **collapsed by default**. `<summary>` = trace-id, truncated question, in/out/total tokens, total time. Expanded body = ordered timeline from `build_timeline()`:
     - decision step → bold phase + rendered payload (classification: intent/complexity/reason; route: strategy/orchestrate/reject_reason; plan: numbered task list with title + instruction; worker: `worker.N` + task title).
     - llm_call step → phase, model, in/out tokens, latency, status badge.
     - result step → answer length, reject flag.
- `_label_chart(items, title, caption) -> str` — shared labeled-bar SVG generator (label + value text visible per bar).
- `main(argv=None) -> int` — argparse with `--data-dir` (default `.observability`) and `--day` (YYYY-MM-DD). Always writes `report.html` into `--data-dir` and prints its path. A leading `report` token is still stripped for `python -m agent.observability report ...`.

All user content (question, task titles, reason strings, model names) passes through `html.escape`. Render/write failures degrade to `warnings.warn` and a non-zero exit — never a raw traceback.

### 3. `agent/observability/tracing.py` — terminal line

Replace `format_trace_line(trace_id, calls)` with `format_trace_summary(trace_id, calls) -> str`:

```
[trace {id}] in={in_tokens} out={out_tokens} {total_s}s
```

- `in_tokens` = sum of `prompt_tokens` over the trace's llm_calls; `out_tokens` = sum of `completion_tokens`; `total_s` = elapsed seconds = sum of llm_call `latency_ms` / 1000, one decimal (signature receives only `calls`, so totals are computed from the calls, as the current `format_trace_line` does).
- Raw integers (no `_fmt_tokens` k-suffix) so the numbers are unambiguous.
- `_fmt_tokens` is removed if its only remaining use was `format_trace_line`; otherwise retained.
- Call-site stays in `patch.py::_wrap_respond` (already wrapped so display errors never break business).
- `agent/observability/__init__.py` re-exports `format_trace_summary` in place of `format_trace_line` (`__all__` updated); any caller of the old name is updated (only `patch.py` and `__init__.py` reference it today).

## Data Flow

1. Agent run: `TracedLLMClient` records `llm_call` events; wrappers record `trace_start`/`decision`/`trace_end`; `TraceStore` writes all to per-day JSONL.
2. `patch.py` prints `format_trace_summary(...)` after each answered question.
3. User runs `python -m agent.observability report [--day YYYY-MM-DD]` → `read_events()` → `report_data` builds summaries/timeline/model stats → `build_html_report` renders → writes `report.html`.

## Error Handling

- Data prep: pure, never raises on malformed events.
- Rendering/writing: wrap in try/except → `warnings.warn` + return non-zero exit code; the error message names what failed.
- Terminal print: existing guard in `patch.py` (try/except pass).
- No changes to the never-raise-into-business constraint; all new code observes it.

## Testing

- **`tests/test_report_data.py`** (new) — pure unit tests:
  - summarize: in/out split across models and phases; missing-usage (None) events counted as 0; `has_error` set when any call errored.
  - timeline: interleaved decision + llm_call events sorted by `ts`; events without `trace_id` ignored; unknown decision types skipped.
  - model_stats: aggregation + sorting.
  - total_stats: header numbers.
- **`tests/test_report.py`** (rewritten) — renderer:
  - summary strip numbers present in HTML; labeled-chart text present (truncated question, model name, phase names);
  - `<details>` collapsed-by-default (no `open` attribute) with timeline content inside;
  - HTML-escaping of a hostile question/title string;
  - `main()` with a temp data dir writes `report.html` and prints path; `--day` filters;
  - assert `build_cli_report` no longer importable.
- **`tests/test_tracing.py`** — add `format_trace_summary` format test (in/out sums, one-decimal seconds, trace-id prefix).
- Full suite stays green (currently 118 passed, 4 skipped).

## Out of Scope

- Monetary cost / pricing.
- Capturing prompt/response text.
- Streaming `chat_completion_stream` observability.
- Chart interactivity (beyond native `<details>` collapse) or external JS/CSS.
- Changes to event schema or the recording layer (`client.py`, `patch.py`, `tracing.py` event writes).

## Known Deviations / Rationale

- **Chart value semantics:** Phase latency bars use summed latency per phase; Per-trace bars use total tokens; Model distribution uses in+out tokens — each labeled with its unit, so "看不出" ambiguity is resolved by labeling, not by new aggregation.
- **Cost dropped** per explicit user decision (no pricing table; token-only).