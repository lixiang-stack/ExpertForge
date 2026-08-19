# Code-Quality Refactor Tasks — Design Spec

Date: 2026-08-19
Status: Approved (brainstorming)

## Problem

Five code-quality tasks surfaced during review of the ExpertForge repo:

1. `live/` test directory sits at repo root instead of living with the rest of the test tree.
2. Dead code exists (an empty `agent/processors/` package, never-called streaming methods, unused params/fields, unused test imports).
3. The evaluation judge cannot be configured as a full model client (different `base_url`/`provider`/`provider_capabilities`); today the `evaluation` block only carries `judge_model` + `results_dir`.
4. `load_domain_config` in `agent/config.py` is ~155 lines, parses 7 files with all filenames hardcoded in the function body, and mixes two concerns (top-level config vs domain loading).
5. The agent has no application logging. Observability (the traces/metrics pillar) is opt-in and non-invasive, but there is no logs pillar.

Secondary: `README.md` is long (~217 lines) and buries its key content.

## Design Decisions

| # | Question | Decision |
|---|---|---|
| 1 | `live/` placement | Move to `tests/live/`, sibling of `tests/unit/`. `testpaths = tests/unit` already excludes live tests from the default run; the `live` marker stays. |
| 2 | Dead-code removal scope | Remove confirmed production dead code (empty `processors/` package, 3 streaming methods, unused `resolve_model` param, 2 unused `.model` properties, `Installed.patched`, unused test imports, stale comments). **Keep** `test.py` and `draft*.md` (user: do not delete). Keep-with-reason: `supports_tool_call` (reserved capability), Orchestrator `strategy` params (observability patch wrappers pass them through), `total_stats()["has_error"]` (read-model API, test-asserted). |
| 3 | Evaluation judge config | Keep in `config.json` (single source of truth, no sync risk with the pipeline model stack). Expand the `evaluation` block with a nested `judge` object carrying the full model-client set (`base_url`, `model`, `provider`, `provider_capabilities`, `timeout`); each field falls back to the top-level value. Flat `judge_model` is removed (superseded by `judge.model`). |
| 4 | `load_domain_config` refactor | Extract to a dedicated `agent/domain_config.py` with one small parser per domain file + a `DOMAIN_FILE_CONTRACT` constant; `load_domain_config` becomes a ~25-line orchestrator in dependency order. Domain dataclasses and `ConfigError` stay in `config.py` (minimize import churn). No re-export from `config.py`; the 5 import sites move to `agent.domain_config`. |
| 5 | Logging library | **`structlog`** (new dependency) — the Python standard for structured logging; emits JSONL, composes with stdlib `logging`. Not `loguru` (archived late 2025), not hand-rolled (user wants an existing library). |
| 6 | Logging placement | Logs are conceptually part of observability (logs/metrics/traces pillars), but the component is **independent**: `agent/loggers.py`, own `logging` config block, zero imports to/from the observability package. Rationale: lifecycle (logs from process start, observability is opt-in), mechanism (direct-call instrumentation vs AOP patching), and failure mode (logs are read when things break). |
| 7 | README | Slim to ~half length; highlight install/configure/run/test; compress Observability/Evaluation/Baseline sections; fold in `tests/live` path + `logging` config. |

## 1. Move `live/` → `tests/live/`

- `git mv live tests/live` (keeps history; `live/__init__.py` moves too).
- Fix `REPO_ROOT` in `tests/live/test_smoke.py` and `tests/live/test_integration.py`: `parents[1]` → `parents[2]`.
- `from tests.helpers import ...` keeps working unchanged.
- `pytest.ini`: unchanged (`testpaths = tests/unit`; `live` marker).
- Update README ("Live tests") and both test-file docstrings: `uv run pytest live -v` → `uv run pytest tests/live -v`.
- Rename `test_integration_medium_question_uses_processor` → `test_integration_medium_question_uses_strategy`.

## 2. Dead code removal

Removals:

1. Delete `agent/processors/` (untracked, `.pyc`-only leftover; source removed in `fc6cbe4`).
2. Remove the 3 never-called-in-production streaming methods and their tests/stubs:
   - `LLMClient.chat_completion_stream` (`agent/llm.py`), plus now-unused `Iterator` import.
   - `TracedLLMClient.chat_completion_stream` (`agent/observability/client.py`).
   - `RecordingClient.chat_completion_stream` (`agent/evaluation/runner.py`).
   - Remove streaming tests in `tests/unit/test_llm.py`, `test_observability_client.py` and the `chat_completion_stream` stubs in test fakes (`test_llm.py`, `test_observability_install.py`, `test_observability_client.py`, `test_evaluation_cli.py`, `test_observability_patch.py`).
3. Remove unused `domain` param from `resolve_model` (`agent/model_router.py`) + its `DomainConfig` import; update callers `agent/chat.py` and `agent/evaluation/runner.py`.
4. Remove unused `TracedLLMClient.model` and `RecordingClient.model` properties.
5. Remove `Installed.patched` field and its `append` in `agent/observability/patch.py`.
6. Clean unused test imports (ruff F401): `test_config.py:9`, `test_evaluation_cli.py:3`, `test_evaluation_dataset.py:4,6,7`, `test_observability_client.py:5`, `test_report_data.py:1,95,163` (+ dedupe `build_timeline` re-import), `test_strategy.py:1`, `test_tracing.py:5`, `test_worker_pool.py:5`.
7. Fix stale comments: `tests/unit/test_chat.py:91` ("second call: generator" → strategy).

Kept (do not touch): `test.py`, `draft.md`, `draft_v1.md`, `draft_v2.md` (user directive). Kept with reason: `ProviderCapabilities.supports_tool_call` (reserved capability contract), Orchestrator `strategy` params (`_plan`/`_aggregate`/`_direct_answer`/`_reaggregate` — observability patch wrappers pass `strategy` through), `total_stats()["has_error"]` (read-model API, asserted by `test_report_data.py`), `TraceStore.close()`/`as_dict()` methods (public read API used by tests).

Verification: `uv run pytest -q` green after removal.

## 3. Evaluation judge model config

New `evaluation` block in `config.json` / `config.example.json`:

```json
"evaluation": {
  "results_dir": "evaluation/results",
  "judge": {
    "base_url": null,
    "model": "gemini-3.5-flash-lite",
    "provider": "gemini",
    "provider_capabilities": { "supports_json_schema": true, "supports_thinking_toggle": false },
    "timeout": 120
  }
}
```

Each `judge` field is optional; absent/null falls back to the top-level value. `judge.model` supersedes flat `judge_model`.

### config.py

- New `JudgeConfig(base_url, model, provider, provider_capabilities, timeout)` dataclass (all `None`-defaulted).
- `EvaluationConfig` becomes `{results_dir="evaluation/results", judge: JudgeConfig | None = None}`.
- `load_config` parses/validates the nested block: `provider_capabilities` keys checked against `KNOWN_CAPABILITY_KEYS` and values must be booleans when present; `timeout` positive when present.

### evaluation/runner.py

- Signature: `run_evaluation(config, domain, suite, client, judge_client=None, *, skip_quality=False)`.
- `judge_client is None` → judge runs on the pipeline `client` (today's behavior, single recorder).
- `judge_client` provided → wrap it in a `RecordingClient` (`judge_recorder`); per case reset both recorders; cost totals = `_sum_calls(recorder.calls + judge_recorder.calls)`; `actual_model` still taken from the pipeline recorder's last call.
- Judge model = `config.evaluation.judge.model or config.model`.

### orchestrator.py

- `Orchestrator._judge_model()` (`agent/orchestrator.py:102-104`) reads the nested judge config: `(cfg.judge.model if cfg and cfg.judge and cfg.judge.model else None) or self.config.model`. `observability/patch.py:248` reads through this method — no change needed there.
- The result-record field `judge_model` (`agent/evaluation/report.py:45,54,113`) is the **output schema key**, not a config key — it stays unchanged.

### evaluation/__main__.py

- Build the pipeline client as today. When a `judge` block exists, build a dedicated judge `LLMClient` from resolved judge params (fallback to top-level) and pass it to `run_evaluation`.
- Record metadata: `judge_model = (config.evaluation.judge.model if config.evaluation and config.evaluation.judge else None) or config.model` (output field unchanged).

### Tests

- Update `tests/unit/test_config.py:636` (`judge_model` → nested `judge`), add validation tests for the nested block.
- Update `tests/unit/test_evaluation_runner.py:38` (`EvaluationConfig(judge=JudgeConfig(model="judge-a"))`).
- Update `tests/unit/test_evaluation_cli.py:293` (expected serialization) and `tests/unit/test_orchestrator.py:300-302` (`EvaluationConfig(judge_model=...)` → `JudgeConfig`).
- `tests/unit/test_evaluation_report.py` record-field assertions are unchanged (output schema).
- Add coverage: judge on a different `base_url` → judge calls recorded separately and summed into per-case totals.

## 4. Refactor `load_domain_config`

New module **`agent/domain_config.py`**:

- Imports `ConfigError`, `COMPLEXITY_LEVELS`, and the domain dataclasses (`IntentDef`, `ComplexityLevelDef`, `ComplexityPolicy`, `EvaluatorPolicy`, `OrchestrationPolicy`, `DomainConfig`) from `.config`. No import cycle: `config.py` does not import `domain_config`.
- `DOMAIN_FILE_CONTRACT = ("domain.json", "intents.yaml", "orchestration.yaml", "complexity.yaml", "intent_mapping.yaml", "prompts/*.md", "expert_policy.md")` — the documented domain directory contract.
- Moved helpers: `_read_json`, `_read_yaml`, `_read_prompt`, `_str_list`.
- One parser per file: `_parse_domain_json`, `_parse_intents`, `_parse_orchestration` (needs intents), `_parse_complexity`, `_parse_intent_mapping` (needs intents + strategies), `_load_prompts`, `_load_expert_policy`.
- `load_domain_config(domain_dir)` → thin orchestrator (~25 lines) calling parsers in dependency order; all existing validation preserved verbatim.

`agent/config.py`: delete the old `load_domain_config` and its helpers; `DomainConfig` and other domain dataclasses remain importable from `agent.config`. `load_domain_config` is **not** re-exported; update the 5 import sites to `from agent.domain_config import load_domain_config`:
- `agent/agent_cli.py`
- `agent/evaluation/__main__.py`
- `tests/live/test_integration.py`
- `tests/unit/test_config.py`
- `tests/unit/test_domain_agnostic.py`

## 5. Logging component (structlog, independent)

### Dependency

Add `structlog>=24.1` to `pyproject.toml` `dependencies`.

### New module `agent/loggers.py`

Imports only stdlib + structlog (no `agent.*` imports → zero coupling with business or observability).

- `setup_logging(cfg: LoggingConfig | None)` — idempotent, process-level.
  - `None` or `enabled=False` → attach a no-op (NullHandler) so logger calls are cheap no-ops and never emit to stderr.
  - Enabled → configure a structlog logger bound to a stdlib `FileHandler` writing JSONL to `cfg.file` (parent dirs created), level from `cfg.level`. Structlog processor pipeline: `TimeStamper` → component/level/event fields → `JSONRenderer`.
- `get_logger(name)` — thin facade returning a structlog logger with the component name attached; guarantees a handler exists even before `setup_logging` (keeps tests quiet).

JSON line shape: `{"event": ..., "level": ..., "logger": ..., "ts": ISO-8601, ...kwargs}` — kwargs passed at call sites become JSON fields.

### config.py

- `LoggingConfig(enabled=False, level="INFO", file="logs/agent.jsonl")`.
- `AgentConfig.logging: LoggingConfig | None = None`.
- `load_config` parses/validates the block (level in `{"DEBUG","INFO","WARNING","ERROR","CRITICAL"}`, `file` a non-empty string).

### config.json / config.example.json / .gitignore / README

- Add `"logging": { "enabled": false, "level": "INFO", "file": "logs/agent.jsonl" }`.
- Add `logs/` to `.gitignore`.

### Instrumentation (1–3 lines per site, only via `get_logger`)

- **Lifecycle** (`agent_cli.main`): `setup_logging` once at start; `event="startup"` (mode, config path), `event="config_error"`, `event="shutdown"`.
- **Routing** (`Router.route`): `event="routing"` with in_domain, intent, complexity, strategy, orchestrate, reject_reason.
- **Answer** (`Chat.respond`): `event="answer"` with question, kind, answer_len; `event="error"` on exception.
- **LLM errors** (`LLMClient.chat_completion` except-branch): `event="llm_error"` with model + error (failures only — no token/latency duplication with observability).
- **Orchestration** (`Orchestrator.run`): `event="orchestration"` with phase + outcome; worker exception → `event="worker_error"`.
- **Evaluation** (`evaluation/__main__.main` + `runner.run_evaluation`): `setup_logging` at start; `event="eval_run_start/end"` (suites, skip_quality); per-case failures → `event="eval_case_error"` (warning).

### Tests

New `tests/unit/test_loggers.py`:
- disabled/None → no file created, calls don't raise.
- enabled → parseable JSON lines with expected fields (event, level, logger, ts).
- level filtering (e.g. `WARNING` → INFO lines not written).
- `setup_logging` idempotent; `get_logger` returns component-scoped logger.
- `setup_logging` called with a `tmp_path` file; no cross-test leakage (each test configures + tears down).

## 6. README simplification

Slim `README.md` from ~217 to roughly ~100 lines, keeping headers prominent:
- Keep: intro, install (`uv sync`), configure (config keys incl. new `evaluation.judge` + `logging`), domain directory summary (1 paragraph), run commands, test commands (unit + `tests/live`), observability (2–3 lines + report command).
- Compress: Evaluation section to a short summary + the `run`/`diff`/`baseline` one-liners; Baseline tracking to a few lines; drop the verbose prose while keeping every command example.
- Mention the new `logging` config block under a Logging subsection.

## Testing

- `uv run pytest -q` — all unit tests green after each task.
- `uv run ruff check agent/ tests/` (if ruff available) — no F401s in production.
- Live tests (`uv run pytest tests/live -v`) remain opt-in via `AGENT_API_KEY`; run smoke once with a real key to confirm the move + config changes did not break end-to-end paths.
