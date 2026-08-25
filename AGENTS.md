# AGENTS.md

## Commands

```bash
uv sync                                   # install deps
uv run pytest -q                          # unit suite only (hermetic, no API key)
uv run pytest tests/unit/test_config.py -q            # one file
uv run pytest tests/unit/test_config.py::test_name -q # one test
uv run pytest tests/live -v               # live E2E; needs AGENT_API_KEY, auto-skips if unset
```

- `pytest.ini` sets `testpaths = tests/unit`, so bare `pytest` never runs live tests.
- No lint or typecheck tool is configured.

## Entry points

- `uv run python -m agent [--ask "..."] [config.json]` — REPL / single-shot agent.
- `uv run python -m agent.evaluation {run|diff|baseline|compare}` — benchmark CLI.
  `run --tier classification|routing|full_expert|all` (default: curated smoke cases);
  `compare` runs baseline-vs-orchestrated on `full_expert` cases (`--ids` filters).
- `uv run python -m agent.observability report` — trace JSONL → HTML report.

## Environment & config gotchas

- Two independent keys: `AGENT_API_KEY` (main client) and `AGENT_JUDGE_API_KEY`
  (judge). Evaluation commands exit 1 with `Config error:` when the judge key is
  missing even though the main client works.
- `load_config()` falls back to `config.example.json` when `config.json` is absent —
  running without a personal config silently uses whatever API the example points at.
- `provider_capabilities` is mandatory and validated: unknown capability keys raise
  `ConfigError`. Judge capabilities fall back to top-level ones when unset.
- Judge model must match the judge `base_url`'s supported models; a mismatch surfaces
  only as a swallowed `LLMError`.

## Architecture

- **Domains are data, not code**: an expert domain is a directory under `domain/`
  (`domain.json`, `intents.yaml`, `intent_mapping.yaml`, `orchestration.yaml`,
  `expert_policy.md`, `prompts/*.md`). Adding a domain requires no code changes;
  `tests/unit/test_domain_agnostic.py` enforces this.
- **Flow**: `Router.route` → `RouteResult` → `Chat.respond` → single
  `Strategy.process` call, or gated questions go through `Orchestrator.run`
  (planner → parallel workers via `worker_pool.run_workers` → aggregator → optional
  evaluator/re-aggregate loop). Critique topology is gate-first when
  `evaluator.enabled`: the draft is judged before critics run, so passing drafts
  return at ~baseline cost.
- **Evaluation tiers** (`classification`/`routing`/`full_expert`) control per-case
  depth; datasets live in `evaluation/datasets/<domain>/*.yaml`.
- **Judge failures are silent by design**: `Judge.score` returns `None` on `LLMError`
  or on any scorecard dimension that isn't an int 1–5 (`parse_scorecard`). Quality
  metrics showing N/A usually means the judge call failed — check base_url/model/key
  compatibility before suspecting the pipeline.
- **Observability** (`agent/observability/`): `install()` wraps the LLM client and
  monkey-patches ~10 business methods. Decision events (planner tasks, worker
  task/role) are only recorded inside an active `trace_span()` contextvar;
  `Chat.respond` creates one, but code calling `Orchestrator.run` directly (e.g.
  `evaluation/compare.py`) must wrap itself in `trace_span()` or worker decisions are
  silently skipped. `run_workers` copies contextvars into worker threads, so spans do
  propagate to workers.
- `RecordingClient` wraps anything exposing `chat_completion` and records token
  usage; it is the cost-accounting mechanism for both evaluation runner and compare.

## Conventions

- Python ≥ 3.10; every module starts with `from __future__ import annotations`.
- Logging goes through `get_logger("<component>")` + structlog-style kwargs
  (`logger.info("event", key=value)`), configured via the `logging` config block.
- Observability and logging must never break business logic: all their failure paths
  degrade to `warnings.warn`, never raise.
