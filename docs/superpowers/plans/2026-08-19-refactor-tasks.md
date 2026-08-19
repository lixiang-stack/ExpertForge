# Code-Quality Refactor Tasks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the five approved refactor tasks: move `live/` under `tests/`, remove dead code, add a full judge-model config to the `evaluation` block, extract `load_domain_config` into `agent/domain_config.py`, and add a structlog-based JSONL logging component — plus slim the README.

**Architecture:** Independent, sequential changes. Live tests move to `tests/live/`. Dead code is removed in three focused passes. Evaluation config grows a nested `judge` model-client block with fallback to top-level values. Domain parsing moves to a dedicated module with one parser per file. Logging is a standalone `agent/loggers.py` (stdlib + structlog only, no `agent.*` imports) used through a `get_logger()` facade, configured via a new `logging` block in `config.json`.

**Tech Stack:** Python >= 3.10, pytest, openai, pyyaml, structlog (new), uv.

## Global Constraints

- Python >= 3.10 (per `pyproject.toml`).
- The ONLY new dependency is `structlog>=24.1`.
- Do NOT delete `test.py`, `draft.md`, `draft_v1.md`, `draft_v2.md`.
- Do NOT touch: `ProviderCapabilities.supports_tool_call`, the 4 Orchestrator `strategy` params, `total_stats()["has_error"]`, `TraceStore.close()` / `as_dict()` methods.
- `pytest.ini` stays as-is (`testpaths = tests/unit`; `live` marker). Live tests run via `uv run pytest tests/live -v`.
- After every task, `uv run pytest -q` must pass.
- The result-record output key `judge_model` (`agent/evaluation/report.py`) is the output schema and stays unchanged.

---

### Task 1: Move `live/` → `tests/live/`

**Files:**
- Move: `live/` → `tests/live/`
- Modify: `tests/live/test_smoke.py`, `tests/live/test_integration.py`

**Interfaces:**
- Produces: `tests/live/test_smoke.py` and `tests/live/test_integration.py`, both with `REPO_ROOT = Path(__file__).resolve().parents[2]`.

- [ ] **Step 1: Move the directory**

Run: `git mv live tests/live`
Expected: `live/` gone; `tests/live/{__init__.py,test_smoke.py,test_integration.py}` present.

- [ ] **Step 2: Fix `REPO_ROOT` and docstrings in both moved files**

In `tests/live/test_smoke.py` and `tests/live/test_integration.py`:

Replace `REPO_ROOT = Path(__file__).resolve().parents[1]` with:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
```

In both file docstrings, replace `uv run pytest live -v` with `uv run pytest tests/live -v`.

- [ ] **Step 3: Rename the stale integration test**

In `tests/live/test_integration.py`, rename `test_integration_medium_question_uses_processor` to `test_integration_medium_question_uses_strategy`.

- [ ] **Step 4: Verify live tests collect (and skip without a key)**

Run: `uv run pytest tests/live -q`
Expected: all tests SKIPPED (no `AGENT_API_KEY`), 0 errors, 0 failures — collection works.

- [ ] **Step 5: Verify unit suite still green**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/live live
git commit -m "test: move live tests into tests/live next to unit"
```

---

### Task 2: Remove the three never-called streaming methods

**Files:**
- Modify: `agent/llm.py` (remove `chat_completion_stream`, `from typing import Iterator`)
- Modify: `agent/observability/client.py` (remove `chat_completion_stream`, `from typing import Iterator`)
- Modify: `agent/evaluation/runner.py` (remove `chat_completion_stream`)
- Modify: `tests/unit/test_llm.py`, `tests/unit/test_observability_client.py`, `tests/unit/test_observability_install.py`, `tests/unit/test_observability_patch.py`, `tests/unit/test_evaluation_cli.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LLMClient` has only `chat_completion`. `TracedLLMClient` and `RecordingClient` have only `chat_completion`.

- [ ] **Step 1: Remove the production methods**

`agent/llm.py`: delete the `chat_completion_stream` method (it is the last method of `LLMClient`) and delete `from typing import Iterator` (line 4).

`agent/observability/client.py`: delete the `chat_completion_stream` method and `from typing import Iterator` (line 5).

`agent/evaluation/runner.py`: delete the `chat_completion_stream` method.

- [ ] **Step 2: Remove the streaming tests and stubs**

- `tests/unit/test_llm.py`: delete `test_chat_completion_stream_yields_content`.
- `tests/unit/test_observability_client.py`: delete `test_stream_delegates_without_recording` and the `chat_completion_stream` method from `FakeInner`.
- `tests/unit/test_observability_install.py`: delete the `chat_completion_stream` method from `FakeClient`.
- `tests/unit/test_observability_patch.py`: delete the `chat_completion_stream` method from `FakeInner`.
- `tests/unit/test_evaluation_cli.py`: delete the `chat_completion_stream` method from each of the three nested `FakeClient` classes.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass. Run `rg -n "chat_completion_stream" --glob '*.py' .` — no hits.

- [ ] **Step 4: Commit**

```bash
git add agent tests/unit
git commit -m "refactor: remove unused chat_completion_stream methods"
```

---

### Task 3: Remove unused `resolve_model` param and dead properties/field

**Files:**
- Modify: `agent/model_router.py`, `agent/chat.py`, `agent/evaluation/runner.py`
- Modify: `agent/observability/client.py`, `agent/observability/patch.py`
- Modify: `tests/unit/test_model_router.py`, `tests/unit/test_observability_install.py`

**Interfaces:**
- Produces: `resolve_model(config: AgentConfig, route: RouteResult, default: str) -> str` (no `domain` param).

- [ ] **Step 1: Update production code**

`agent/model_router.py` — new full content:

```python
from __future__ import annotations

from .config import AgentConfig
from .router import RouteResult


def resolve_model(config: AgentConfig, route: RouteResult, default: str) -> str:
    if route.complexity == "simple":
        return config.model_low or default
    return config.model_high or default
```

`agent/chat.py` line 38 — replace `resolve_model(self.config, self.domain, route, self.config.model)` with `resolve_model(self.config, route, self.config.model)`.

`agent/evaluation/runner.py` line 134 — replace `resolve_model(config, domain, route, config.model)` with `resolve_model(config, route, config.model)`.

`agent/observability/client.py` — remove the `model` property:

```python
    @property
    def model(self) -> str:
        return self._inner.model
```

`agent/evaluation/runner.py` — remove the `model` property:

```python
    @property
    def model(self) -> str:
        return self._inner.model
```

`agent/observability/patch.py` — remove `patched: list[str] = field(default_factory=list)` from `Installed`, and remove the line `self.patched.append(key)` inside `_wrap`.

- [ ] **Step 2: Update tests**

`tests/unit/test_model_router.py` — replace the whole file with:

```python
from agent.config import AgentConfig
from agent.model_router import resolve_model
from agent.router import RouteResult


def _config(model_low=None, model_high=None):
    return AgentConfig(
        base_url="https://x", model="m", classifier_model="c",
        domain_dir="d", model_low=model_low, model_high=model_high,
    )


def _route(strategy="direct", complexity=None):
    return RouteResult(
        in_domain=True, strategy=strategy,
        intent="faq", complexity=complexity,
    )


def test_simple_uses_model_low():
    result = resolve_model(_config("low-a", "high-a"), _route(complexity="simple"), "default")
    assert result == "low-a"


def test_simple_missing_model_low_falls_back_to_default():
    result = resolve_model(_config(), _route(complexity="simple"), "default")
    assert result == "default"


def test_medium_uses_model_high():
    result = resolve_model(_config("low-a", "high-a"), _route(complexity="medium"), "default")
    assert result == "high-a"


def test_complex_uses_model_high():
    result = resolve_model(_config("low-a", "high-a"), _route(complexity="complex"), "default")
    assert result == "high-a"


def test_none_complexity_uses_model_high():
    result = resolve_model(_config("low-a", "high-a"), _route(complexity=None), "default")
    assert result == "high-a"


def test_medium_missing_model_high_falls_back_to_default():
    result = resolve_model(_config("low-a", None), _route(complexity="medium"), "default")
    assert result == "default"
```

`tests/unit/test_observability_install.py` — in `test_install_enabled_wraps_client`, delete the line `assert out.model == "m"`.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add agent tests/unit
git commit -m "refactor: drop unused resolve_model domain param and dead client/patched fields"
```

---

### Task 4: Delete `agent/processors/`, clean unused test imports, fix stale comments

**Files:**
- Delete: `agent/processors/`
- Modify: `tests/unit/test_evaluation_dataset.py`, `tests/unit/test_observability_client.py`, `tests/unit/test_report_data.py`, `tests/unit/test_strategy.py`, `tests/unit/test_tracing.py`, `tests/unit/test_worker_pool.py`, `tests/unit/test_evaluation_cli.py`, `tests/unit/test_chat.py`

**Interfaces:**
- Consumes: nothing.
- Produces: no file references `agent.processors`.

- [ ] **Step 1: Delete the leftover package**

Run: `rm -rf agent/processors`
Expected: directory gone. It is untracked (`.pyc`-only), so no `git rm` needed.

- [ ] **Step 2: Clean unused test imports**

- `tests/unit/test_evaluation_dataset.py`: remove `COMPLEXITY_LEVELS,`, `EvalCase,`, and `Suite,` from the `from agent.evaluation.dataset import (...)` block.
- `tests/unit/test_observability_client.py`: change line 5 to `from agent.observability.tracing import TraceStore, phase, trace_span` (drop `current_phase`).
- `tests/unit/test_report_data.py`: delete `import pytest` (line 1) and delete the entire line `from agent.observability.report_data import Step, build_timeline` (line 95). Keep line 163 `from agent.observability.report_data import build_timeline, group_stages`.
- `tests/unit/test_strategy.py`: change line 1 to `from agent.config import DomainConfig`.
- `tests/unit/test_tracing.py`: delete `from pathlib import Path` (line 5).
- `tests/unit/test_worker_pool.py`: change line 5 to `from agent.worker_pool import WorkerTask, run_workers`.
- `tests/unit/test_evaluation_cli.py`: delete `import pytest` (line 3).

- [ ] **Step 3: Fix the stale comment**

`tests/unit/test_chat.py` line 91 — replace `# first call: classification (model=cm); second call: generator (model=low)` with `# first call: classification (model=cm); second call: strategy (model=low)`.

- [ ] **Step 4: Verify and run suite**

Run: `rg -n "agent\.processors|chat_completion_stream" --glob '*.py' -g '!.venv/**' -g '!.git/**' -g '!.worktrees/**' .`
Expected: no hits.

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A agent tests/unit
git commit -m "chore: remove leftover processors package and unused test imports"
```

---

### Task 5: Full judge-model client config in the `evaluation` block

**Files:**
- Modify: `agent/config.py`, `config.example.json`
- Modify: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `ProviderCapabilities` (already in `agent/config.py`).
- Produces: `JudgeConfig(base_url, model, provider, provider_capabilities, timeout)` and `EvaluationConfig(results_dir, judge)` where `judge: JudgeConfig | None`. The flat `judge_model` field is removed.
- Config shape (all optional, each falls back to the top-level value, then to defaults):

```json
{
  "evaluation": {
    "results_dir": "evaluation/results",
    "judge": {
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4o",
      "provider": "openai",
      "provider_capabilities": { "supports_tool_call": true },
      "timeout": 60
    }
  }
}
```

- [ ] **Step 1: Add `JudgeConfig` and update `EvaluationConfig` in `agent/config.py`**

Add `from dataclasses import dataclass, field` import if not present. Add the new dataclass next to `EvaluationConfig`:

```python
@dataclass
class JudgeConfig:
    base_url: str
    model: str
    provider: str
    provider_capabilities: ProviderCapabilities | None = None
    timeout: int = 60
```

Replace `EvaluationConfig` with:

```python
@dataclass
class EvaluationConfig:
    results_dir: str = "evaluation/results"
    judge: JudgeConfig | None = None
```

- [ ] **Step 2: Rewrite the `evaluation` block parsing in `load_config`**

Replace the `evaluation` parsing block in `load_config` with:

```python
    eval_block = raw.get("evaluation") or {}
    eval_results_dir = eval_block.get("results_dir", "evaluation/results")

    judge_block = eval_block.get("judge") or {}
    judge_base_url = judge_block.get("base_url") or base_url
    judge_model = judge_block.get("model") or model
    judge_provider = judge_block.get("provider") or provider
    judge_caps = judge_block.get("provider_capabilities")
    judge_caps_obj = (
        ProviderCapabilities(**judge_caps) if isinstance(judge_caps, dict) else None
    )
    judge_config = (
        JudgeConfig(
            base_url=judge_base_url,
            model=judge_model,
            provider=judge_provider,
            provider_capabilities=judge_caps_obj,
            timeout=judge_block.get("timeout", 60),
        )
        if judge_block
        else None
    )

    evaluation = EvaluationConfig(results_dir=eval_results_dir, judge=judge_config)
```

Wire `evaluation` into the `AgentConfig(...)` constructor call.

- [ ] **Step 3: Update `config.example.json`**

Add the `judge` block from the config shape above to the `evaluation` section.

- [ ] **Step 4: Update tests**

`tests/unit/test_config.py`:
- Delete the tests asserting `evaluation.judge_model` exists (e.g. the judge_model assertions around line 636) and replace with `evaluation.judge` assertions.
- Update the config with a full `judge` block to assert each field, including fallback behavior when `judge` omits `base_url`/`model`/`provider`.
- Update the default-config test: `evaluation.judge is None`.
- Remove the `evaluation.judge_model` line from the config fixture used by other tests if present.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass. Run `rg -n "judge_model" agent tests/unit --glob '*.py'` and confirm the ONLY remaining hits are in `agent/evaluation/report.py` (output schema) and `tests/unit/test_evaluation_cli.py` line 293 (baseline fixture — leave as-is).

- [ ] **Step 6: Commit**

```bash
git add agent tests/unit config.example.json
git commit -m "feat(config): nested evaluation.judge model client config"
```

---

### Task 6: Judge client wiring — runner, CLI, orchestrator

**Files:**
- Modify: `agent/evaluation/runner.py`, `agent/evaluation/__main__.py`, `agent/orchestrator.py`
- Modify: `tests/unit/test_evaluation_runner.py`, `tests/unit/test_orchestrator.py`

**Interfaces:**
- Consumes: `JudgeConfig` (Task 5).
- Produces: `run_evaluation(config, domain, suite, client, judge_client=None, *, skip_quality=False)`. The recorder for the judge client is `judge_recorder`; its call list is appended to the total. `Orchestrator._judge_name()` (renamed in Task 5 from `_judge_model`) already reads the nested judge config. The result-record `judge_model` output key (report.py) is unchanged.

NOTE: Task 5 already updated the flat `judge_model` consumers in `__main__.py`, `runner.py`, `orchestrator.py` (`_judge_name`), `patch.py`, and already switched `tests/unit/test_evaluation_runner.py` and `tests/unit/test_orchestrator.py` to `JudgeConfig`. This task is the remaining wiring only: the `judge_client` parameter + call-list summing in `run_evaluation`, and building/passing the judge client in `__main__.py`.

- [ ] **Step 1: Update `run_evaluation` in `agent/evaluation/runner.py`**

Ensure the signature is:

```python
def run_evaluation(
    config: EvaluationConfig,
    domain: str,
    suite: Suite,
    client: LLMClient,
    judge_client: LLMClient | None = None,
    *,
    skip_quality: bool = False,
) -> list[dict[str, str]]:
```

In the body, wrap the judge client in a recorder and sum both recorders' call lists:

```python
    judge_recorder = RecordingClient(judge_client) if judge_client is not None else None
    # ...after running all cases...
    call_list = client.call_list + (judge_recorder.call_list if judge_recorder else [])
```

Pass `call_list` to `build_timeline` (and the calls-per-model counts) instead of `client.call_list`. Confirm the current signature already has `skip_quality` (the plan assumed it may not — verify `agent/evaluation/runner.py` and `agent/evaluation/__main__.py` before changing).

- [ ] **Step 2: Build the judge client in `agent/evaluation/__main__.py`**

`__main__.py` already reads `config.evaluation.judge` for the `judge_model=` metadata (Task 5). Extend the `run_evaluation(...)` call so it also builds a judge client only when the judge block exists:

```python
    judge_client = None
    if config.evaluation.judge is not None:
        judge_client = LLMClient(config.evaluation.judge.base_url, config.evaluation.judge.model)
    run_evaluation(config.evaluation, args.domain, suite, client, judge_client=judge_client, skip_quality=args.skip_quality)
```

- [ ] **Step 3: Verify `Orchestrator._judge_name()` in `agent/orchestrator.py`**

Already done in Task 5 (reads `self.config.evaluation.judge.model` with `self.config.model` fallback). Verify it is present; do NOT rename it back.

- [ ] **Step 4: Update/add tests**

- `tests/unit/test_evaluation_runner.py`: confirm existing tests use `JudgeConfig` (done in Task 5). ADD coverage for the new `judge_client` parameter: calling `run_evaluation` with a judge client records its calls into the timeline/call list, and passing no judge client keeps current behavior.
- `tests/unit/test_orchestrator.py`: already updated to `JudgeConfig` + `_judge_name()` assertion (Task 5) — verify.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass. Run `rg -n "judge_model" agent tests/unit --glob '*.py'` and confirm the ONLY remaining hits are the output-schema interface and baseline fixture: `agent/evaluation/report.py`, `agent/evaluation/__main__.py:69` (the `judge_model=` kwarg to `serialize_results`), `tests/unit/test_evaluation_report.py`, and `tests/unit/test_evaluation_cli.py:282` (baseline fixture). No `judge_model` references may exist in `agent/config.py` or as an attribute read in `agent/orchestrator.py`.

- [ ] **Step 6: Commit**

```bash
git add agent tests/unit
git commit -m "refactor(evaluation): wire dedicated judge client and nested judge config"
```

---

### Task 7: Extract `load_domain_config` into `agent/domain_config.py`

**Files:**
- Add: `agent/domain_config.py`
- Modify: `agent/config.py`
- Modify: `agent/agent_cli.py`, `agent/evaluation/__main__.py`, `tests/live/test_integration.py`, `tests/unit/test_config.py`, `tests/unit/test_domain_agnostic.py`

**Interfaces:**
- Consumes: `ConfigError`, `COMPLEXITY_LEVELS`, and the dataclasses `DomainConfig`, `IntentDef`, `ComplexityPolicy`, `OrchestrationPolicy`, `IntentMapping`, `AnswerMapping`, `OptionMapping`, `LinkMapping` from `agent/config.py`.
- Produces: `load_domain_config(path: str | Path) -> DomainConfig` in `agent/domain_config.py`. `agent/config.py` does NOT re-export it. `DOMAIN_FILE_CONTRACT` constant documents the required keys.

- [ ] **Step 1: Create `agent/domain_config.py`**

```python
from __future__ import annotations

import json
import os
from pathlib import Path

from .config import (
    COMPLEXITY_LEVELS,
    AnswerMapping,
    ComplexityPolicy,
    ConfigError,
    DomainConfig,
    IntentDef,
    IntentMapping,
    LinkMapping,
    OptionMapping,
    OrchestrationPolicy,
)

DOMAIN_FILE_CONTRACT = (
    "domain.json must contain: 'intents' (list of {id, description}), "
    "'orchestration' (direct|chain|agent or {chain: [agents]}), "
    "'complexity' (map of intent id -> simple|medium|complex), "
    "optional 'answers' (list of {id, option_ids, text, link, include_metadata}), "
    "'policy' (list of {task, instructions, condition}). "
    "Chain 'agents' must include the {id, file, description} fields."
)


def _parse_domain_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in domain file '{path}': {exc}") from exc


def _parse_intents(raw: dict, path: Path) -> list[IntentDef]:
    intents = raw.get("intents")
    if not isinstance(intents, list):
        raise ConfigError(f"domain file '{path}': 'intents' must be a list")
    out = []
    for entry in intents:
        if not isinstance(entry, dict):
            raise ConfigError(f"domain file '{path}': intent entry must be an object")
        intent_id = entry.get("id")
        description = entry.get("description")
        if not intent_id or not isinstance(intent_id, str):
            raise ConfigError(f"domain file '{path}': intent is missing a string 'id'")
        if description is not None and not isinstance(description, str):
            raise ConfigError(f"domain file '{path}': intent '{intent_id}' description must be a string")
        out.append(IntentDef(id=intent_id, description=description))
    return out


def _parse_orchestration(raw: dict, path: Path) -> OrchestrationPolicy:
    value = raw.get("orchestration")
    if isinstance(value, str):
        if value not in ("direct", "chain", "agent"):
            raise ConfigError(
                f"domain file '{path}': unknown orchestration '{value}' "
                "(expected 'direct', 'chain' or 'agent')"
            )
        return OrchestrationPolicy(mode=value, agents=[])
    if isinstance(value, dict):
        mode = value.get("mode", "chain")
        agents = value.get("chain") or []
        parsed_agents = []
        for agent in agents:
            if not isinstance(agent, dict):
                raise ConfigError(f"domain file '{path}': chain agent must be an object")
            parsed_agents.append(
                {
                    "id": agent.get("id", agent.get("name")),
                    "file": agent.get("file"),
                    "description": agent.get("description"),
                }
            )
        return OrchestrationPolicy(mode=mode, agents=parsed_agents)
    raise ConfigError(
        f"domain file '{path}': 'orchestration' must be 'direct', 'chain', 'agent' "
        "or an object with a 'chain' list"
    )


def _parse_complexity(raw: dict, path: Path) -> ComplexityPolicy:
    complexity = raw.get("complexity")
    if complexity is None:
        return ComplexityPolicy(intents={})
    if not isinstance(complexity, dict):
        raise ConfigError(f"domain file '{path}': 'complexity' must be an object")
    levels = {}
    for intent_id, level in complexity.items():
        if level not in COMPLEXITY_LEVELS:
            raise ConfigError(
                f"domain file '{path}': unknown complexity '{level}' for intent '{intent_id}'"
            )
        levels[intent_id] = level
    return ComplexityPolicy(intents=levels)


def _parse_intent_mapping(raw: dict, path: Path) -> IntentMapping:
    mapping = raw.get("intent_mapping") or raw.get("intent_map")
    if mapping is None:
        return IntentMapping(entries={})
    if not isinstance(mapping, dict):
        raise ConfigError(f"domain file '{path}': 'intent_mapping' must be an object")
    entries = {key: value.get("answer_id") if isinstance(value, dict) else value for key, value in mapping.items()}
    return IntentMapping(entries=entries)


def _load_prompts(raw: dict, path: Path) -> dict:
    prompts = raw.get("prompts") or {}
    if not isinstance(prompts, dict):
        raise ConfigError(f"domain file '{path}': 'prompts' must be an object")
    return prompts


def _load_expert_policy(raw: dict, path: Path) -> list[AnswerMapping]:
    answers = raw.get("answers") or []
    policy = raw.get("policy") or []
    if not isinstance(answers, list) or not isinstance(policy, list):
        raise ConfigError(f"domain file '{path}': 'answers' and 'policy' must be lists")
    mappings = []
    for entry in answers:
        if not isinstance(entry, dict):
            raise ConfigError(f"domain file '{path}': answer entry must be an object")
        link = entry.get("link")
        mappings.append(
            AnswerMapping(
                id=entry.get("id"),
                option_ids=entry.get("option_ids") or [],
                text=entry.get("text"),
                link=link if isinstance(link, str) else None,
                include_metadata=entry.get("include_metadata"),
            )
        )
    return mappings, policy


def load_domain_config(path: str | Path) -> DomainConfig:
    path = Path(path)
    if not path.is_absolute():
        path = Path(os.getcwd()) / path
    raw = _parse_domain_json(path)
    intents = _parse_intents(raw, path)
    orchestration = _parse_orchestration(raw, path)
    complexity = _parse_complexity(raw, path)
    intent_mapping = _parse_intent_mapping(raw, path)
    prompts = _load_prompts(raw, path)
    answers, policy = _load_expert_policy(raw, path)
    return DomainConfig(
        intents=intents,
        orchestration=orchestration,
        complexity=complexity,
        intent_mapping=intent_mapping,
        prompts=prompts,
        answers=answers,
        policy=policy,
    )
```

NOTE: if `DomainConfig` does not have all of the fields above, adapt to its actual field set; the essential behavior is: JSON load with clear `ConfigError`s and per-file parsers. `_load_expert_policy` returns a 2-tuple — if the original `load_domain_config` handled `answers` and `policy` separately, split accordingly.

- [ ] **Step 2: Delete the old implementation from `agent/config.py`**

Remove `load_domain_config` and its private helpers (`_parse_*`, `_load_prompts`, `_load_expert_policy`), and remove now-unused imports (`json`, `os`, `Path`, `COMPLEXITY_LEVELS` if unused elsewhere). Keep `ConfigError` and all dataclasses. Do NOT add a `load_domain_config` re-export.

- [ ] **Step 3: Update the five import sites**

- `agent/agent_cli.py` line 6: `from .config import ConfigError, get_api_key, load_config, load_domain_config` → remove `load_domain_config`, add `from .domain_config import load_domain_config`.
- `agent/evaluation/__main__.py` line 8: `from agent.config import ConfigError, get_api_key, load_config, load_domain_config` → remove `load_domain_config`, add `from agent.domain_config import load_domain_config`.
- `tests/live/test_integration.py` line 20: `from agent.config import load_config, load_domain_config` → change to `from agent.config import load_config` + `from agent.domain_config import load_domain_config`.
- `tests/unit/test_config.py`: keep the `load_domain_config` tests here (do NOT move them) but change the import to `from agent.domain_config import load_domain_config`; keep `from agent.config import ConfigError` if used.
- `tests/unit/test_domain_agnostic.py` line 4: `from agent.config import AgentConfig, load_domain_config` → `from agent.config import AgentConfig` + `from agent.domain_config import load_domain_config`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent tests/live tests/unit
git commit -m "refactor: extract load_domain_config into agent/domain_config.py"
```

---

### Task 8: Logging dependency, config, and `agent/loggers.py`

**Files:**
- Modify: `pyproject.toml`, `agent/config.py`, `config.example.json`, `.gitignore`
- Add: `agent/loggers.py`, `tests/unit/test_loggers.py`

**Interfaces:**
- Consumes: nothing from `agent.*` (loggers.py is importable without the rest of the app).
- Produces: `setup_logging(cfg: LoggingConfig) -> None` (idempotent) and `get_logger(name: str) -> structlog.stdlib.BoundLogger` returning `structlog.get_logger(f"agent.{name}")`.
- `LoggingConfig(enabled: bool = False, level: str = "INFO", file: str = "logs/agent.jsonl")`.
- Config shape: top-level `"logging": {"enabled": false, "level": "INFO", "file": "logs/agent.jsonl"}`.
- `.gitignore` gains `logs/`.

- [ ] **Step 1: Add the dependency**

`pyproject.toml` dependencies: add `"structlog>=24.1"`.

- [ ] **Step 2: Add `LoggingConfig` to `agent/config.py`**

Add near the other small dataclasses (before `AgentConfig`):

```python
@dataclass
class LoggingConfig:
    enabled: bool = False
    level: str = "INFO"
    file: str = "logs/agent.jsonl"
```

Add `LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")` near the top of the module (used by validation).

In `load_config`, add a `logging` block parser with validation that rejects unknown levels:

```python
    logging_block = raw.get("logging") or {}
    logging_level = logging_block.get("level", "INFO")
    if logging_level not in LOG_LEVELS:
        raise ConfigError(f"Unknown log level '{logging_level}'. Valid: {', '.join(LOG_LEVELS)}")
    logging_config = LoggingConfig(
        enabled=bool(logging_block.get("enabled", False)),
        level=logging_level,
        file=logging_block.get("file", "logs/agent.jsonl"),
    )
```

Wire `logging=logging_config` into the `AgentConfig(...)` constructor call. Keep `AgentConfig` sorted fields consistent.

- [ ] **Step 3: Add `LoggingConfig` to `config.example.json`**

Add the `logging` block above to the config example.

- [ ] **Step 4: Update `.gitignore`**

Add `logs/` under the existing entries.

- [ ] **Step 5: Create `agent/loggers.py`**

```python
from __future__ import annotations

import logging
import sys

import structlog

from .config import LoggingConfig

_log_setup_done = False


def _configure_structlog() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", key="ts"),
            structlog.processors.add_logger_name,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _configure_agent_stdlib_logger() -> None:
    logger = logging.getLogger("agent")
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


def setup_logging(cfg: LoggingConfig) -> None:
    """(Re)configure the 'agent' logger and structlog. Idempotent: safe to call repeatedly."""
    global _log_setup_done

    if not _log_setup_done:
        _configure_structlog()
        _configure_agent_stdlib_logger()
        _log_setup_done = True

    logger = logging.getLogger("agent")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()

    if not cfg.enabled:
        logger.addHandler(logging.NullHandler())
        return

    formatter = logging.Formatter("%(message)s")
    if cfg.file in ("-", "stdout"):
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(cfg.file, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.setLevel(getattr(logging, cfg.level.upper()))
    logger.addHandler(handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the 'agent.*' namespace."""
    return structlog.get_logger(f"agent.{name}")
```

Note: with `cache_logger_on_first_use=True` and structlog's stdlib `LoggerFactory`, processors render into the stdlib record's message and the FileHandler writes the JSON line to the file.

- [ ] **Step 6: Write `tests/unit/test_loggers.py`**

An `autouse` fixture resets global logging state before each test:

```python
import json
import logging
from pathlib import Path

import pytest
import structlog

from agent.config import LoggingConfig
from agent.loggers import get_logger, setup_logging


@pytest.fixture(autouse=True)
def _reset_logging_state():
    yield
    logger = logging.getLogger("agent")
    logger.handlers = []
    logger.propagate = True
    structlog.reset_defaults()


def test_disabled_writes_nothing(tmp_path):
    cfg = LoggingConfig(enabled=False, level="DEBUG", file=str(tmp_path / "a.jsonl"))
    setup_logging(cfg)
    get_logger("test").info("hello")
    assert list(tmp_path.glob("*.jsonl")) == []


def test_enabled_writes_jsonl_lines(tmp_path):
    log_file = tmp_path / "agent.jsonl"
    cfg = LoggingConfig(enabled=True, level="DEBUG", file=str(log_file))
    setup_logging(cfg)
    get_logger("test").info("hello", extra=1)
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "hello"
    assert record["extra"] == 1
    assert record["logger"] == "agent.test"


def test_stdout_when_file_is_dash(tmp_path, capsys):
    cfg = LoggingConfig(enabled=True, level="INFO", file="-")
    setup_logging(cfg)
    get_logger("test").info("to stdout")
    assert "to stdout" in capsys.readouterr().out


def test_setup_logging_is_idempotent(tmp_path):
    log_file = tmp_path / "agent.jsonl"
    cfg = LoggingConfig(enabled=True, level="INFO", file=str(log_file))
    setup_logging(cfg)
    setup_logging(cfg)
    get_logger("test").info("one line")
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
```

Adjust `tmp_path`/`capsys` usage to the actual pytest version in use (pytest>=8.0 is fine).

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (including the new `tests/unit/test_loggers.py`).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml agent tests/unit config.example.json .gitignore
git commit -m "feat(logging): structlog-based JSONL logging with config-gated setup"
```

---

### Task 9: Instrument the codebase with logging

**Files:**
- Modify: `agent/agent_cli.py`, `agent/router.py`, `agent/chat.py`, `agent/llm.py`, `agent/orchestrator.py`, `agent/evaluation/__main__.py`, `agent/evaluation/runner.py`

**Interfaces:**
- Consumes: `setup_logging`, `get_logger` from `agent/loggers.py`.
- Produces: lifecycle, routing, error/warning, and evaluation-lifecycle log events. No functional behavior changes.

**Log categories to cover:**
- Lifecycle: startup / shutdown in `agent_cli.py`.
- Routing: decision in `Router.route`.
- Answers/errors: in `Chat.respond` (success path and exception path).
- LLM errors: exception path in `LLMClient.chat_completion`.
- Orchestration: run start, per-worker failures in `Orchestrator.run`.
- Evaluation lifecycle: run start/end in `evaluation/__main__.py`, per-case errors in `evaluation/runner.py`.

- [ ] **Step 1: `agent/agent_cli.py`**

Import `from .loggers import get_logger, setup_logging`. In `main()`, after `cfg` is loaded, call `setup_logging(cfg.logging)`. Add a logger `logger = get_logger("cli")`. Log `logger.info("agent startup", domain=cfg.domain, config=cfg.config_file)` before the main loop and `logger.info("agent shutdown")` on clean exit. Wrap `run_interactive`/main loop body in try/except that logs `logger.exception("agent crashed")` before re-raising (or logging and exiting 1). Also use `setup_logging` inside the `--chat` and `--evaluate` branches so all entry paths configure logging.

- [ ] **Step 2: `agent/router.py`**

In `Router.route`, add `logger = get_logger("router")` and after the strategy/intent are resolved, `logger.info("routing decision", intent=result.intent, complexity=result.complexity, strategy=result.strategy)`. (Keep `self._cm_processed` / other existing logic untouched.)

- [ ] **Step 3: `agent/chat.py`**

Add `logger = get_logger("chat")`. In `respond()`, after a successful answer, `logger.info("answer generated", strategy=route.strategy, chain_length=len(route.agents) if route.agents else 0)`. In the `except` path, `logger.exception("answer generation failed")` before re-raising.

- [ ] **Step 4: `agent/llm.py`**

In `LLMClient.chat_completion`, wrap the `chat.completions.create` call in try/except; on exception call `logger.exception("llm error", model=self.model, endpoint=self.base_url)` before re-raising. Add `logger = get_logger("llm")`.

- [ ] **Step 5: `agent/orchestrator.py`**

Add `logger = get_logger("orchestrator")`. In `run()`, log `logger.info("orchestration start", strategy=..., chain_length=...)` at the top and, per worker failure, `logger.warning("worker failure", agent=..., error=...)`. Keep existing exception handling.

- [ ] **Step 6: `agent/evaluation/__main__.py`**

After building config, `setup_logging(cfg.logging)`. Add `logger = get_logger("evaluation")` and log `logger.info("eval run start", domain=args.domain, suite=suite_name)` before `run_evaluation` and `logger.info("eval run end", cases=len(results))` after.

- [ ] **Step 7: `agent/evaluation/runner.py`**

Add `logger = get_logger("evaluation")`. In the per-case loop, wrap the quality-check/answer execution so failures log `logger.warning("eval case error", case=case_id, error=str(exc))` (or exception) and continue per existing behavior. Do NOT add trace/token instrumentation — observability owns that.

- [ ] **Step 8: Verify and run suite**

Run: `rg -n "get_logger\(" agent --glob '*.py'` — confirm the expected files use it and no logger leaks into `agent/loggers.py` itself.

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add agent
git commit -m "feat(logging): instrument lifecycle, routing, errors, and evaluation events"
```

---

### Task 10: Slim the README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a README that is substantially shorter (~100 lines, from ~217) and keeps: project purpose, quick start, main features, config (`config.json` incl. new `evaluation.judge` and `logging` blocks), basic usage commands, and a short "development" section (tests, live tests, lint). Remove verbose prose, long examples, and redundant sections.

- [ ] **Step 1: Rewrite `README.md`**

Keep the existing title and a 1-2 sentence intro. Structure:

```
# ExpertForge
<purpose + one-paragraph intro>

## Features
- short bullet list

## Quick Start
- uv install / setup
- AGENT_API_KEY
- config.json
- run agent (interactive), run once, evaluate

## Configuration
- brief table/bullets for config.json sections, including:
  - `evaluation.judge`: full judge-model client config (base_url, model, provider, provider_capabilities, timeout)
  - `logging`: enabled/level/file JSONL logging
- note: README documents but does not enumerate every field

## Development
- uv run pytest -q
- uv run pytest tests/live -v  (needs AGENT_API_KEY)
- lint: <existing command if any>
```

Do NOT add emojis. Do NOT add new sections beyond what exists unless needed.

- [ ] **Step 2: Verify length and content**

Check the README is roughly 100 lines and mentions the new `evaluation.judge` and `logging` config blocks.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: slim README"
```

---

## Notes

- The five tasks correspond to the user's requests: (1) move `live/` into `tests/`, (2) remove dead code, (3) evaluation judge model config, (4) extract `load_domain_config`, (5) observability/logging.
- Logging is implemented as an independent component (`agent/loggers.py`) rather than inside `agent/observability/`; observability owns traces/token stats, logging owns structured JSONL events.
- The result-record output key `judge_model` in `agent/evaluation/report.py` is the output schema and stays unchanged.
- `tests/unit/test_evaluation_cli.py` line 293 (`"judge_model": None`) is a baseline fixture and is intentionally left unchanged.
