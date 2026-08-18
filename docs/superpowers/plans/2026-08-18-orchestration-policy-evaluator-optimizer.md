# Orchestration Policy + Evaluator / Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `complexity_gate` orchestration decision with a required per-domain `orchestration.yaml` policy, and add an Evaluator (reusing the existing LLM-as-judge) plus an Optimizer (feedback-driven re-aggregation) to the orchestrated pipeline.

**Architecture:** The policy lives in `domain/<name>/orchestration.yaml` (parsed into `DomainConfig.orchestration`). `Router.route` decides orchestration solely from that policy. After aggregation, `Orchestrator.run` runs the Judge on the answer and, if any dimension is below `min_dimension_score`, re-aggregates the worker outputs with judge feedback up to `max_rounds`. Legacy machinery removed: `strategies.yaml`, `default_strategy`, `StrategyDef`, `unsupported_complex.md`, `worker_timeout`, `OrchestratorConfig`, `effective_timeout`.

**Tech Stack:** Python 3 (sync, `concurrent.futures` worker pool), PyYAML, pytest (`uv run pytest -q`).

## Global Constraints

- `uv run pytest -q` must stay green after every task.
- All tests run without an API key.
- Orchestration decision is ONLY: `enabled AND complexity_rank(result.complexity) >= complexity_rank(min_complexity) AND result.intent ∈ intents`.
- `orchestration.yaml` is required; missing/invalid → `ConfigError`.
- `config.timeout` is the only global timeout; no per-worker wall-clock timeout.
- Do not rebuild the Judge — reuse `agent/evaluation/judge.py` as-is.
- No comments in code unless already present in the file being edited.

---

### Task 1: OrchestrationPolicy data model + `orchestration.yaml` loader

**Files:**
- Modify: `agent/config.py`
- Create: `domain/software_engineering/orchestration.yaml`
- Test: `tests/unit/test_config.py`, `tests/unit/test_domain_agnostic.py`

**Interfaces:**
- Consumes: existing `COMPLEXITY_LEVELS`, `DomainConfig`, `ConfigError`, `_read_yaml`, `intents` dict in `load_domain_config`.
- Produces:
  - `EvaluatorPolicy(enabled: bool = True, min_dimension_score: int = 3, max_rounds: int = 1)`
  - `OrchestrationPolicy(enabled: bool = True, min_complexity: str = "complex", intents: list[str], max_workers: int = 4, evaluator: EvaluatorPolicy)`
  - `DomainConfig.orchestration: OrchestrationPolicy | None` (set by loader, `None` only for direct constructions)

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_config.py` a module-level fixture constant and orchestration tests:

```python
ORCHESTRATION_YAML = (
    "enabled: true\n"
    "min_complexity: complex\n"
    "intents:\n"
    "  - faq\n"
    "max_workers: 4\n"
    "evaluator:\n"
    "  enabled: true\n"
    "  min_dimension_score: 3\n"
    "  max_rounds: 1\n"
)
```

In `_write_domain(tmp_path, **overrides)`, right after `(base / "prompts").mkdir(parents=True)`, add:

```python
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
```

Every other test in `test_config.py` that calls `load_domain_config` must:
1. Declare at least intent `faq`. Replace every `(base / "intents.yaml").write_text("", encoding="utf-8")` (appears ~13 times: in `test_load_domain_config_out_of_domain_reply_default`, the complexity tests `test_load_domain_config_complexity_policy` through `test_load_domain_config_complexity_reordered`, and the intent-example default tests) with:

```python
    (base / "intents.yaml").write_text("- id: faq\n  description: quick question\n", encoding="utf-8")
```

2. Add right after its `(base / "prompts").mkdir(parents=True)` line:

```python
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
```

`test_load_domain_config_bad_yaml` (intents.yaml is invalid YAML) is the one exception — it raises `ConfigError` while parsing `intents.yaml`, before orchestration is reached, so it needs no orchestration.yaml. Fixtures already declaring `faq` (e.g. the intent-example default tests) only need the orchestration.yaml line.

Add these new tests:

```python
def test_load_domain_config_orchestration_policy(tmp_path):
    domain = load_domain_config(_write_domain(tmp_path))
    oc = domain.orchestration
    assert oc is not None
    assert oc.enabled is True
    assert oc.min_complexity == "complex"
    assert oc.intents == ["faq"]
    assert oc.max_workers == 4
    assert oc.evaluator.enabled is True
    assert oc.evaluator.min_dimension_score == 3
    assert oc.evaluator.max_rounds == 1


def test_load_domain_config_orchestration_missing_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_orchestration_empty_intents_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    (base / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: complex\nintents: []\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_orchestration_unknown_intent_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    (base / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: complex\nintents:\n  - bogus\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_orchestration_bad_min_complexity_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    (base / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: impossible\nintents:\n  - faq\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_orchestration_bad_evaluator_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    (base / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: complex\nintents:\n  - faq\n"
        "evaluator:\n  enabled: true\n  min_dimension_score: 9\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))
```

In `tests/unit/test_domain_agnostic.py`, inside `_write_finance_domain`, after the `prompts` directory creation, add:

```python
    (base / "orchestration.yaml").write_text(
        "enabled: true\n"
        "min_complexity: complex\n"
        "intents:\n"
        "  - portfolio_review\n"
        "  - risk_check\n"
        "max_workers: 2\n"
        "evaluator:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py::test_load_domain_config_orchestration_policy tests/unit/test_domain_agnostic.py::test_custom_strategy_orchestrates_complex -v`
Expected: FAIL — `AttributeError: 'DomainConfig' object has no attribute 'orchestration'` (policy tests) and `ConfigError: orchestration.yaml not found` (domain-agnostic test).

- [ ] **Step 3: Implement**

In `agent/config.py`:

1. Add the two dataclasses after `ComplexityPolicy`:

```python
@dataclass
class EvaluatorPolicy:
    enabled: bool = True
    min_dimension_score: int = 3
    max_rounds: int = 1


@dataclass
class OrchestrationPolicy:
    enabled: bool = True
    min_complexity: str = "complex"
    intents: list[str] = field(default_factory=list)
    max_workers: int = 4
    evaluator: EvaluatorPolicy = field(default_factory=EvaluatorPolicy)
```

2. Add the field to `DomainConfig`:

```python
@dataclass
class DomainConfig:
    name: str
    description: str
    out_of_domain_reply: str
    intents: dict[str, IntentDef]
    intent_mapping: dict[str, str]
    strategies: dict[str, StrategyDef]
    default_strategy: str
    prompts: dict[str, str]
    complexity: ComplexityPolicy | None = None
    expert_policy: str = ""
    orchestration: OrchestrationPolicy | None = None
```

3. In `load_domain_config`, insert the parse block right after the `intents` dict is built and before the `complexity` block:

```python
    orchestration = None
    orch_path = base / "orchestration.yaml"
    if not orch_path.is_file():
        raise ConfigError(f"orchestration.yaml not found: {orch_path}")
    orch_data = _read_yaml(orch_path)
    if not isinstance(orch_data, dict):
        raise ConfigError(f"orchestration.yaml must contain a mapping: {orch_path}")
    orch_intents = orch_data.get("intents")
    if not isinstance(orch_intents, list) or not orch_intents:
        raise ConfigError(f"orchestration.yaml 'intents' must be a non-empty list: {orch_path}")
    if not all(isinstance(i, str) and i in intents for i in orch_intents):
        raise ConfigError(f"orchestration.yaml 'intents' references unknown intent: {orch_path}")
    min_complexity = orch_data.get("min_complexity", "complex")
    if min_complexity not in COMPLEXITY_LEVELS:
        raise ConfigError(f"Unknown 'min_complexity' {min_complexity!r} in {orch_path}")
    max_workers = orch_data.get("max_workers", 4)
    if not isinstance(max_workers, int) or max_workers <= 0:
        raise ConfigError(f"orchestration.yaml 'max_workers' must be a positive int: {orch_path}")
    ev = orch_data.get("evaluator") or {}
    if not isinstance(ev, dict):
        raise ConfigError(f"orchestration.yaml 'evaluator' must be a mapping: {orch_path}")
    min_score = ev.get("min_dimension_score", 3)
    max_rounds = ev.get("max_rounds", 1)
    if not isinstance(min_score, int) or not 1 <= min_score <= 5:
        raise ConfigError(f"orchestration.yaml 'min_dimension_score' must be an int in 1..5: {orch_path}")
    if not isinstance(max_rounds, int) or max_rounds < 0:
        raise ConfigError(f"orchestration.yaml 'max_rounds' must be a non-negative int: {orch_path}")
    orchestration = OrchestrationPolicy(
        enabled=bool(orch_data.get("enabled", True)),
        min_complexity=min_complexity,
        intents=orch_intents,
        max_workers=max_workers,
        evaluator=EvaluatorPolicy(
            enabled=bool(ev.get("enabled", True)),
            min_dimension_score=min_score,
            max_rounds=max_rounds,
        ),
    )
```

4. Pass it into the returned `DomainConfig(...)` constructor as `orchestration=orchestration`.

Create `domain/software_engineering/orchestration.yaml`:

```yaml
enabled: true
min_complexity: complex
intents:
  - architecture_design
  - performance_analysis
  - troubleshooting
  - code_review
max_workers: 4
evaluator:
  enabled: true
  min_dimension_score: 3
  max_rounds: 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_domain_agnostic.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/config.py domain/software_engineering/orchestration.yaml tests/unit/test_config.py tests/unit/test_domain_agnostic.py
git commit -m "feat: orchestration policy config with per-domain orchestration.yaml"
```

---

### Task 2: Router policy-based orchestration decision

**Files:**
- Modify: `agent/router.py`
- Test: `tests/unit/test_router.py`, `tests/unit/test_chat.py`, `tests/unit/test_observability_patch.py`, `tests/unit/test_domain_agnostic.py`

**Interfaces:**
- Consumes: `DomainConfig.orchestration: OrchestrationPolicy | None` (Task 1).
- Produces: `RouteResult.orchestrate` computed from the policy. `complexity_gate` is no longer read.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_router.py`, replace the `strategies`/`default_strategy` fixture block with a policy and rewrite the orchestration tests:

```python
def _domain(**overrides):
    default = {
        "name": "软件工程",
        "description": "sw",
        "out_of_domain_reply": "Out.",
        "intents": {
            "concept_explain": IntentDef("concept_explain", "explain"),
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug"),
            "architecture_design": IntentDef("architecture_design", "arch"),
        },
        "intent_mapping": {
            "concept_explain": "teaching",
            "faq": "direct",
            "troubleshooting": "debugging",
            "architecture_design": "analysis",
        },
        "strategies": {
            "teaching": StrategyDef("teaching", complexity_gate=True, default=True),
            "direct": StrategyDef("direct"),
            "debugging": StrategyDef("debugging", complexity_gate=True),
            "analysis": StrategyDef("analysis", complexity_gate=True),
        },
        "default_strategy": "teaching",
        "orchestration": OrchestrationPolicy(
            enabled=True, min_complexity="complex",
            intents=["architecture_design", "troubleshooting"],
            max_workers=4, evaluator=EvaluatorPolicy(enabled=True),
        ),
        "prompts": {},
    }
    default.update(overrides)
    return DomainConfig(**default)
```

Update the import line: `from agent.config import AgentConfig, DomainConfig, EvaluatorPolicy, IntentDef, OrchestrationPolicy, StrategyDef`.

Rewrite the two orchestration tests:

```python
def test_route_policy_orchestrates_complex_in_intent():
    client = FakeClient([_combined(True, "architecture_design", "complex")])
    result = Router(client, _config(), _domain()).route("design a big system")
    assert result.strategy == "analysis"
    assert result.orchestrate is True


def test_route_policy_disabled_never_orchestrates():
    domain = _domain(orchestration=OrchestrationPolicy(
        enabled=False, min_complexity="simple", intents=["architecture_design"]))
    client = FakeClient([_combined(True, "architecture_design", "complex")])
    result = Router(client, _config(), domain).route("design a big system")
    assert result.orchestrate is False


def test_route_policy_complexity_below_min_not_orchestrated():
    client = FakeClient([_combined(True, "architecture_design", "medium")])
    result = Router(client, _config(), _domain()).route("design something")
    assert result.orchestrate is False


def test_route_policy_intent_not_in_list_not_orchestrated():
    client = FakeClient([_combined(True, "faq", "complex")])
    result = Router(client, _config(), _domain()).route("q")
    assert result.strategy == "direct"
    assert result.orchestrate is False
```

Keep `test_route_unknown_intent_falls_back_to_default` unchanged — it still passes because `default_strategy` still exists in this task (it is removed in Task 4).

In `tests/unit/test_chat.py`, update `_domain()`:

```python
def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out of domain.",
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
        orchestration=OrchestrationPolicy(
            enabled=True, min_complexity="complex", intents=["troubleshooting"],
            max_workers=4, evaluator=EvaluatorPolicy(enabled=False),
        ),
        prompts={
            "direct": "Direct answer prompt.",
            "debugging": "Debugging prompt.",
            "unsupported_complex": "Needs orchestrator.",
        },
    )
```

Update the import: `from agent.config import AgentConfig, DomainConfig, EvaluatorPolicy, IntentDef, OrchestrationPolicy, StrategyDef`.

In `tests/unit/test_observability_patch.py`, update `_domain_complex()`:

```python
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
        orchestration=OrchestrationPolicy(
            enabled=True, min_complexity="complex", intents=["troubleshooting"],
            max_workers=4, evaluator=EvaluatorPolicy(enabled=False),
        ),
        prompts={
            "direct": "Direct prompt.",
            "debugging": "Debugging prompt.",
            "unsupported_complex": "x.",
        },
    )
```

Update the import: `from agent.config import AgentConfig, DomainConfig, EvaluatorPolicy, IntentDef, ObservabilityConfig, OrchestrationPolicy, StrategyDef`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_router.py tests/unit/test_chat.py tests/unit/test_observability_patch.py tests/unit/test_domain_agnostic.py -v`
Expected: FAIL — the router still reads `complexity_gate` (not the policy), so `test_route_policy_disabled_never_orchestrates` fails (it expects `orchestrate is False` but the old code returns `True` for a complex `architecture_design` gated strategy).

- [ ] **Step 3: Implement**

In `agent/router.py`:

```python
from .classification import ClassificationService
from .config import COMPLEXITY_LEVELS, AgentConfig, DomainConfig
from .llm import LLMClient


_COMPLEXITY_RANK = {level: i for i, level in enumerate(COMPLEXITY_LEVELS)}


class Router:
    ...
    def route(self, question: str) -> RouteResult:
        result = self.classifier.classify(question, model=self.config.classifier_model)
        if not result.in_domain:
            return RouteResult(
                in_domain=False, strategy="reject", reject_reason=result.reason
            )
        intent_id = result.intent
        strategy = self.domain.intent_mapping.get(intent_id, self.domain.default_strategy)
        orchestrate = False
        policy = self.domain.orchestration
        if policy is not None:
            orchestrate = (
                policy.enabled
                and _COMPLEXITY_RANK.get(result.complexity, -1)
                >= _COMPLEXITY_RANK.get(policy.min_complexity, 0)
                and result.intent in policy.intents
            )
        return RouteResult(
            in_domain=True,
            strategy=strategy,
            intent=intent_id,
            complexity=result.complexity,
            orchestrate=orchestrate,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_router.py tests/unit/test_chat.py tests/unit/test_observability_patch.py tests/unit/test_domain_agnostic.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/router.py tests/unit/test_router.py tests/unit/test_chat.py tests/unit/test_observability_patch.py
git commit -m "feat: orchestration decision from orchestration policy"
```

---

### Task 3: Remove the worker pool wall-clock timeout

**Files:**
- Modify: `agent/worker_pool.py`, `agent/orchestrator.py`
- Test: `tests/unit/test_worker_pool.py`

**Interfaces:**
- Consumes: none (internal).
- Produces: `run_workers(tasks, run_one, *, max_workers: int = 4) -> list[WorkerResult]` — no `timeout` parameter.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_worker_pool.py`, drop `timeout=...` from every `run_workers(...)` call (in `test_run_workers_returns_results_in_input_order`, `test_run_workers_caps_concurrency`, `test_run_workers_exception_captured_not_raised`, `test_run_workers_all_failed_returns_all_errors`, `test_run_workers_propagates_caller_context`, `test_run_workers_empty_tasks`) and delete `test_run_workers_worker_timeout_marks_failure`. Add:

```python
def test_run_workers_rejects_timeout_parameter():
    import pytest
    with pytest.raises(TypeError):
        run_workers(_tasks(1), lambda t: "x", max_workers=1, timeout=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_worker_pool.py -v`
Expected: FAIL on `test_run_workers_rejects_timeout_parameter` — the `TypeError` is not raised because `run_workers` still accepts a `timeout` keyword.

- [ ] **Step 3: Implement**

In `agent/worker_pool.py`, replace the whole file with:

```python
"""Parallel worker execution with bounded concurrency."""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class WorkerTask:
    title: str
    instruction: str
    role: str


@dataclass
class WorkerResult:
    task: WorkerTask
    text: str | None = None
    error: str | None = None


def run_workers(
    tasks: list[WorkerTask],
    run_one,
    *,
    max_workers: int = 4,
) -> list[WorkerResult]:
    """Run run_one(task) per task concurrently, capped at max_workers.
    Never raises on worker failure; results are returned in input order.
    `run_one` is injected so this module needs no LLM. All LLM calls are
    bounded by the client's config.timeout, so there is no wall-clock timeout
    here. Each work item runs within its own copy of the caller's contextvars
    context (the same approach asyncio's executor/task machinery uses):
    ThreadPoolExecutor threads start with an empty context, so this keeps
    contextvars (e.g. an observability span/trace_id) visible inside the worker
    threads. A single Context cannot be entered by more than one thread at
    once, so each submitted work item gets a fresh copy rather than sharing
    one."""
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            i: executor.submit(
                lambda t=task, c=contextvars.copy_context(): c.run(run_one, t)
            )
            for i, task in enumerate(tasks)
        }
        results: list[WorkerResult] = []
        for i, task in enumerate(tasks):
            try:
                text = futures[i].result()
            except Exception as e:  # noqa: BLE001 - worker failure is captured, not propagated
                results.append(WorkerResult(task=task, text=None, error=str(e) or repr(e)))
            else:
                results.append(WorkerResult(task=task, text=text))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return results
```

Key changes from the current version: drop the `timeout` parameter, drop the
`FutureTimeoutError` import and the `futures[i].result(timeout=timeout)` call
(now `futures[i].result()`), and drop the `error="timeout"` branch. Everything
else — `contextvars.copy_context()`, input-order guarantee, empty-message
`repr(e)` behavior — is preserved verbatim.

In `agent/orchestrator.py`, change the `run_workers` call in `run()`:

```python
        oc = self.config.orchestrator or OrchestratorConfig()
        results = run_workers(
            tasks,
            lambda task: self._worker(question, task, context, model),
            max_workers=oc.max_workers,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_worker_pool.py tests/unit/test_orchestrator.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/worker_pool.py agent/orchestrator.py tests/unit/test_worker_pool.py
git commit -m "refactor: drop worker pool wall-clock timeout (config.timeout governs)"
```

---

### Task 4: Delete `strategies.yaml`, `default_strategy`, `StrategyDef`, `unsupported_complex.md`, per-strategy model

**Files:**
- Modify: `agent/config.py`, `agent/model_router.py`, `agent/router.py`
- Delete: `domain/software_engineering/strategies.yaml`, `domain/software_engineering/prompts/unsupported_complex.md`
- Test: `tests/unit/test_config.py`, `tests/unit/test_domain_agnostic.py`, `tests/unit/test_router.py`, `tests/unit/test_chat.py`, `tests/unit/test_repl.py`, `tests/unit/test_classification.py`, `tests/unit/test_observability_install.py`, `tests/unit/test_observability_patch.py`, `tests/unit/test_orchestrator.py`, `tests/unit/test_evaluation_runner.py`

**Interfaces:**
- Consumes: existing `prompts/` directory layout, `intents` + `intent_mapping` dicts.
- Produces:
  - `DomainConfig.strategies: list[str]` (sorted strategy ids derived from `prompts/*.md`).
  - `DomainConfig` WITHOUT `default_strategy`.
  - `Router.route` rejects an unknown/None intent (no fallback strategy).
  - `resolve_model(config, domain, route, default)` ignores `domain.strategies`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_config.py`:
1. Remove `StrategyDef` from the import.
2. In `_write_domain`, delete the `(base / "strategies.yaml").write_text(...)` and `(base / "prompts" / "unsupported_complex.md").write_text(...)` lines.
3. Rewrite `test_load_domain_config_basic`:

```python
def test_load_domain_config_basic(tmp_path):
    domain = load_domain_config(_write_domain(tmp_path))
    assert isinstance(domain, DomainConfig)
    assert domain.name == "软件工程"
    assert domain.description == "software engineering"
    assert domain.out_of_domain_reply == "Out of domain."
    assert set(domain.intents) == {"concept_explain", "faq"}
    assert domain.intent_mapping == {"concept_explain": "teaching", "faq": "direct"}
    assert domain.strategies == ["direct", "teaching"]
    assert "teach self-contained" in domain.prompts["teaching"]
    assert domain.orchestration is not None
```

4. In every other `load_domain_config` fixture in `test_config.py`, apply all three mechanical edits:
   - Delete the `(base / "strategies.yaml").write_text(...)` line.
   - Delete the `(base / "prompts" / "unsupported_complex.md").write_text(...)` line.
   - Ensure the fixture's `intent_mapping.yaml` maps every declared intent (the fixtures now declaring `faq` need `"faq: direct\n"` instead of `""`), so the new mapping-coverage check passes. Every fixture already writes `prompts/direct.md`, so the `direct` strategy resolves.
5. Replace `test_load_domain_config_resolves_default_strategy` with:

```python
def test_load_domain_config_strategies_derived_from_prompts(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "teaching.md").write_text("t", encoding="utf-8")
    domain = load_domain_config(str(base))
    assert domain.strategies == ["direct", "teaching"]
```

6. Add the intent-mapping coverage test:

```python
def test_load_domain_config_unmapped_intent_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n"
        "- id: tutorial\n  description: walkthrough\n",
        encoding="utf-8",
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))
```

7. Note: deleting the `test_load_config_orchestrator_*` tests is NOT part of this task (that is Task 5).

In `tests/unit/test_domain_agnostic.py`:
- Delete the `strategies.yaml` and `unsupported_complex.md` write lines from `_write_finance_domain`.
- Replace `assert domain.default_strategy == "advise"` with `assert domain.strategies == ["advise", "risk_assessment"]`.

In `tests/unit/test_router.py`:
- Replace the `strategies`/`default_strategy` fixture keys with `strategies=["teaching", "direct", "debugging", "analysis"]` and delete `default_strategy` from the fixture; delete `StrategyDef` from the import.
- Replace `test_route_unknown_intent_falls_back_to_default` with:

```python
def test_route_unknown_intent_rejects():
    client = FakeClient([_combined(True, "bogus", "simple")])  # validation sets intent to None
    result = Router(client, _config(), _domain()).route("q")
    assert result.in_domain is False
    assert result.strategy == "reject"
```

In `tests/unit/test_chat.py`, `tests/unit/test_repl.py`, `tests/unit/test_classification.py`, `tests/unit/test_observability_install.py`, `tests/unit/test_observability_patch.py`, `tests/unit/test_orchestrator.py`, `tests/unit/test_evaluation_runner.py`: convert each direct `DomainConfig(...)` fixture from

```python
        strategies={"direct": StrategyDef("direct", default=True)},
        default_strategy="direct",
```

to

```python
        strategies=["direct"],
```

and remove `StrategyDef` from the `from agent.config import ...` line. Keep the `prompts` dict keys aligned with the strategies list (drop the `"unsupported_complex"` prompt key where present; keep `prompts["direct"]` etc.).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'default_strategy'` (once `default_strategy` is removed from the dataclass) and fixture/loader mismatches.

- [ ] **Step 3: Implement**

In `agent/config.py`:
1. Delete the `StrategyDef` dataclass.
2. `DomainConfig` becomes:

```python
@dataclass
class DomainConfig:
    name: str
    description: str
    out_of_domain_reply: str
    intents: dict[str, IntentDef]
    intent_mapping: dict[str, str]
    strategies: list[str]
    prompts: dict[str, str]
    complexity: ComplexityPolicy | None = None
    expert_policy: str = ""
    orchestration: OrchestrationPolicy | None = None
```

3. In `load_domain_config`, replace the strategies parsing + default resolution block. After `intent_mapping` is built, add the coverage check:

```python
    for intent_id in intents:
        if intent_id not in intent_mapping:
            raise ConfigError(
                f"intent_mapping.yaml is missing a strategy for intent '{intent_id}'"
            )
```

4. Derive strategy ids and prompts from the prompts directory (replacing the `strategies.yaml` reading and the `default_strategy` resolution):

```python
    prompt_dir = base / "prompts"
    strategies = sorted(p.stem for p in prompt_dir.glob("*.md"))
    if not strategies:
        raise ConfigError(f"No strategy prompt files found in {prompt_dir}")
    prompts: dict[str, str] = {}
    for sid in strategies:
        prompts[sid] = _read_prompt(prompt_dir / f"{sid}.md")
    for intent_id, strategy_id in intent_mapping.items():
        if strategy_id not in strategies:
            raise ConfigError(
                f"Mapping for intent '{intent_id}' references unknown strategy "
                f"'{strategy_id}': no {strategy_id}.md in {prompt_dir}"
            )
```

5. Remove the `prompts["unsupported_complex"]` line.
6. Return `DomainConfig(...)` without `default_strategy`.

In `agent/model_router.py`:

```python
def resolve_model(
    config: AgentConfig,
    domain: DomainConfig,
    route: RouteResult,
    default: str,
) -> str:
    if route.complexity == "simple":
        return config.model_low or default
    return config.model_high or default
```

(`domain` is kept in the signature for call-site compatibility; the strategy-model override is removed.)

In `agent/router.py`, replace the strategy lookup:

```python
        intent_id = result.intent
        if not intent_id or intent_id not in self.domain.intent_mapping:
            return RouteResult(
                in_domain=False, strategy="reject",
                reject_reason=f"Unknown intent: {intent_id}",
            )
        strategy = self.domain.intent_mapping[intent_id]
```

(Delete the `default_strategy` fallback.)

Delete the files:

```bash
rm domain/software_engineering/strategies.yaml domain/software_engineering/prompts/unsupported_complex.md
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add -A agent/config.py agent/model_router.py agent/router.py domain/software_engineering tests/unit
git commit -m "refactor: drop strategies.yaml, default_strategy, unsupported_complex, per-strategy model"
```

---

### Task 5: Remove `OrchestratorConfig`, `effective_timeout`, config.json `orchestrator` block; `max_workers` from policy

**Files:**
- Modify: `agent/config.py`, `agent/agent_cli.py`, `agent/evaluation/__main__.py`, `agent/orchestrator.py`, `config.example.json`
- Test: `tests/unit/test_config.py`, `tests/unit/test_evaluation_runner.py`, `tests/unit/test_orchestrator.py`

**Interfaces:**
- Consumes: `OrchestrationPolicy.max_workers` (Task 1).
- Produces: `AgentConfig` WITHOUT `orchestrator`; `config.timeout` as the sole timeout; `Orchestrator.run` reads `max_workers` from `domain.orchestration`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_config.py`, delete the four `test_load_config_orchestrator_*` tests and remove any `orchestrator` key from `_write_config` call sites.

In `tests/unit/test_evaluation_runner.py`:
- Remove `StrategyDef` and `effective_timeout` from the import.
- Delete the three `test_effective_timeout_*` tests.
- Update the `_domain()` fixture (see Task 4 for `strategies=["direct"]`).

In `tests/unit/test_orchestrator.py`, update `_domain()` to include the orchestration policy so `max_workers` has a source:

```python
def _domain():
    return DomainConfig(
        name="sw",
        description="software engineering",
        out_of_domain_reply="Out.",
        intents={"troubleshooting": IntentDef("troubleshooting", "debug")},
        intent_mapping={"troubleshooting": "debugging"},
        strategies=["debugging"],
        prompts={"debugging": "Debugging system prompt."},
        orchestration=OrchestrationPolicy(
            enabled=True, min_complexity="complex", intents=["troubleshooting"],
            max_workers=4, evaluator=EvaluatorPolicy(enabled=False),
        ),
    )
```

Update the import: `from agent.config import AgentConfig, DomainConfig, EvaluatorPolicy, IntentDef, OrchestrationPolicy`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -q`
Expected: FAIL — `AttributeError: 'AgentConfig' object has no attribute 'orchestrator'` (once removed) and `OrchestratorConfig` import errors.

- [ ] **Step 3: Implement**

In `agent/config.py`:
1. Delete `OrchestratorConfig`.
2. Delete the `orchestrator` field from `AgentConfig`.
3. Delete the `raw_orch = raw.get("orchestrator") ...` block in `load_config`.
4. Delete `effective_timeout` entirely.

In `agent/agent_cli.py`:

```python
from .config import ConfigError, get_api_key, load_config, load_domain_config
...
    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model,
                       timeout=config.timeout,
                       provider=config.provider,
                       capability_overrides=config.provider_capabilities)
```

In `agent/evaluation/__main__.py`, same import change and `timeout=config.timeout` at L49.

In `agent/orchestrator.py`:

```python
        policy = self.domain.orchestration
        results = run_workers(
            tasks,
            lambda task: self._worker(question, task, context, model),
            max_workers=policy.max_workers if policy else 4,
        )
```

(Delete the `oc = self.config.orchestrator or OrchestratorConfig()` line and the `OrchestratorConfig` import.)

In `config.example.json`, delete the `"orchestrator": {...}` block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/config.py agent/agent_cli.py agent/evaluation/__main__.py agent/orchestrator.py config.example.json tests/unit
git commit -m "refactor: config.timeout governs all calls; max_workers from orchestration policy"
```

---

### Task 6: Evaluator / Optimizer loop

**Files:**
- Modify: `agent/orchestrator.py`
- Test: `tests/unit/test_orchestrator.py`

**Interfaces:**
- Consumes: `Judge` from `agent.evaluation.judge` (unchanged), `EvaluatorPolicy` (Task 1), `WorkerResult`/`WorkerTask`/`run_workers`.
- Produces:
  - `Orchestrator._judge_model() -> str`
  - `Orchestrator._evaluate(judge, question, answer) -> dict | None`
  - `Orchestrator._reaggregate(question, strategy, context, results, previous, feedback, round_no, model) -> str`
  - `Orchestrator.run(...)` returns the aggregated answer directly when the evaluator is disabled or passes; otherwise returns the best version after ≤ `max_rounds` re-aggregations.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_orchestrator.py`, update `_domain()` to accept an evaluator override:

```python
def _domain(evaluator=None):
    return DomainConfig(
        name="sw",
        description="software engineering",
        out_of_domain_reply="Out.",
        intents={"troubleshooting": IntentDef("troubleshooting", "debug")},
        intent_mapping={"troubleshooting": "debugging"},
        strategies=["debugging"],
        prompts={"debugging": "Debugging system prompt."},
        orchestration=OrchestrationPolicy(
            enabled=True, min_complexity="complex", intents=["troubleshooting"],
            max_workers=4,
            evaluator=evaluator or EvaluatorPolicy(enabled=True, min_dimension_score=3, max_rounds=1),
        ),
    )
```

Add scorecard constants and a call-raising client near the other fixtures:

```python
_PLAN_JSON = '{"tasks": [{"title": "t1", "instruction": "i1", "role": "R1"}, {"title": "t2", "instruction": "i2", "role": "R2"}]}'
_SCORECARD_PASS = ('{"correctness": 4, "relevance": 4, "completeness": 4, '
                   '"technical_depth": 4, "practical_usefulness": 4, "hallucination": 4}')
_SCORECARD_LOW = ('{"correctness": 2, "relevance": 4, "completeness": 4, '
                  '"technical_depth": 4, "practical_usefulness": 4, "hallucination": 4}')


class CallRaisingClient(FakeClient):
    def __init__(self, responses, raise_on_call):
        super().__init__(responses)
        self.raise_on_call = raise_on_call

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        if len(self.calls) == self.raise_on_call:
            self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
            raise LLMError("boom")
        return super().chat_completion(
            messages, model=model, disable_thinking=disable_thinking,
            json_mode=json_mode, json_schema=json_schema,
        )
```

Add these tests:

```python
def test_run_evaluator_passes_returns_aggregated():
    client = FakeClient([_PLAN_JSON, "w1", "w2", "final answer", _SCORECARD_PASS])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "final answer"
    assert len(client.calls) == 5
    judge_messages = client.calls[4][0]
    assert judge_messages[-1]["content"] == "huge task"


def test_run_evaluator_fail_optimizes_once():
    client = FakeClient([_PLAN_JSON, "w1", "w2", "draft answer", _SCORECARD_LOW,
                         "improved answer", _SCORECARD_PASS])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "improved answer"
    assert len(client.calls) == 7
    reaggregate_messages = client.calls[5][0]
    assert "correctness: 2/5" in reaggregate_messages[0]["content"]
    assert "Previous draft:\ndraft answer" in reaggregate_messages[-1]["content"]
    assert "Sub-task (R1): t1" in reaggregate_messages[-1]["content"]


def test_run_evaluator_fail_exhausts_max_rounds():
    client = FakeClient([_PLAN_JSON, "w1", "w2", "draft answer", _SCORECARD_LOW,
                         "attempt 1", _SCORECARD_LOW])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "attempt 1"
    assert len(client.calls) == 7


def test_run_evaluator_disabled_skips_judge():
    domain = _domain(evaluator=EvaluatorPolicy(enabled=False))
    client = FakeClient([_PLAN_JSON, "w1", "w2", "final answer"])
    result = Orchestrator(client, _config(), domain).run("huge task", _route(), "high-a")
    assert result == "final answer"
    assert len(client.calls) == 4


def test_run_judge_parse_failure_treated_as_pass():
    client = FakeClient([_PLAN_JSON, "w1", "w2", "final answer", "not json"])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "final answer"
    assert len(client.calls) == 5


def test_run_judge_llm_error_treated_as_pass():
    client = CallRaisingClient([_PLAN_JSON, "w1", "w2", "final answer", "unused"],
                               raise_on_call=4)
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "final answer"


def test_run_reaggregate_llm_error_returns_previous():
    client = CallRaisingClient([_PLAN_JSON, "w1", "w2", "draft answer", _SCORECARD_LOW,
                                "unused"], raise_on_call=5)
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "draft answer"


def test_run_judge_uses_judge_model_from_config():
    config = _config()
    config.evaluation = EvaluationConfig(judge_model="judge-a")
    client = FakeClient([_PLAN_JSON, "w1", "w2", "final answer", _SCORECARD_PASS])
    Orchestrator(client, config, _domain()).run("huge task", _route(), "high-a")
    judge_call = client.calls[4]
    assert judge_call[1] == "judge-a"
```

Update the import to include `EvaluationConfig`:

```python
from agent.config import AgentConfig, DomainConfig, EvaluationConfig, EvaluatorPolicy, IntentDef, OrchestrationPolicy
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_orchestrator.py -v`
Expected: FAIL — `AttributeError: 'Orchestrator' object has no attribute '_judge_model'` and no evaluator call happens (call counts differ).

- [ ] **Step 3: Implement**

In `agent/orchestrator.py`:

```python
from __future__ import annotations

from .config import AgentConfig, DomainConfig
from .evaluation.judge import Judge
from .llm import LLMClient
from .parsing import parse_json
from .strategy import build_registry
from .router import RouteResult
from .worker_pool import WorkerResult, WorkerTask, run_workers
```

Rewrite `run()` and add the new methods (keep `_plan`, `_worker`, `_aggregate`, `_direct_answer` as-is):

```python
    def run(self, question: str, route: RouteResult, model: str) -> str:
        context = self._strategy_context(route.strategy)
        tasks = self._plan(question, route.strategy, context, model)
        if tasks is None:
            return self._direct_answer(question, route.strategy, context, model)
        policy = self.domain.orchestration
        results = run_workers(
            tasks,
            lambda task: self._worker(question, task, context, model),
            max_workers=policy.max_workers if policy else 4,
        )
        if all(r.error for r in results):
            return self._direct_answer(question, route.strategy, context, model)
        answer = self._aggregate(question, route.strategy, context, results, model)
        if not (policy and policy.evaluator.enabled):
            return answer
        return self._evaluate_loop(
            question, route.strategy, context, results, answer, model, policy.evaluator
        )

    def _judge_model(self) -> str:
        cfg = self.config.evaluation
        return (cfg.judge_model if cfg and cfg.judge_model else None) or self.config.model

    def _evaluate_loop(
        self, question: str, strategy: str, context: str,
        results: list[WorkerResult], answer: str, model: str, evaluator,
    ) -> str:
        judge = Judge(self.client, self._judge_model())
        threshold = evaluator.min_dimension_score
        for round_no in range(evaluator.max_rounds + 1):
            scorecard = self._evaluate(judge, question, answer)
            if scorecard is None:
                return answer
            if all(score >= threshold for score in scorecard.values()):
                return answer
            if round_no == evaluator.max_rounds:
                return answer
            feedback = [f"{dim}: {score}/5" for dim, score in scorecard.items() if score < threshold]
            try:
                answer = self._reaggregate(
                    question, strategy, context, results, answer, feedback, round_no, model
                )
            except LLMError:
                return answer
        return answer

    def _evaluate(self, judge: Judge, question: str, answer: str) -> dict | None:
        return judge.score(question, answer)

    def _reaggregate(
        self, question: str, strategy: str, context: str,
        results: list[WorkerResult], previous: str, feedback: list[str],
        round_no: int, model: str,
    ) -> str:
        sections = []
        for r in results:
            label = f"Sub-task ({r.task.role}): {r.task.title}"
            if r.error:
                sections.append(f"{label}\n[worker failed: {r.error}]")
            else:
                sections.append(f"{label}\n{r.text}")
        user_content = (
            f"User question: {question}\n\n"
            f"Sub-task results:\n\n" + "\n\n".join(sections) +
            f"\n\nPrevious draft:\n{previous}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"{context}\n\n"
                    "You are synthesizing sub-task results into one coherent final "
                    "answer to the user's original question. Some sub-task results "
                    "may be missing due to worker failure; produce the best answer "
                    "from what is available.\n\n"
                    "A previous draft scored too low on these judge dimensions; "
                    "produce an improved draft that addresses them:\n"
                    + "\n".join(f"- {f}" for f in feedback)
                ),
            },
            {"role": "user", "content": user_content},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True).text
```

Update the `LLMError` import: `from .llm import LLMClient, LLMError`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_orchestrator.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat: evaluator/optimizer loop after aggregation"
```

---

### Task 7: Observability records for evaluator and optimizer

**Files:**
- Modify: `agent/observability/patch.py`
- Test: `tests/unit/test_observability_patch.py`

**Interfaces:**
- Consumes: `Orchestrator._evaluate`, `Orchestrator._reaggregate`, `Orchestrator._judge_model` (Task 6).
- Produces: decision events `orchestration.evaluator` `{"scorecard", "passed"}` and `orchestration.optimizer` `{"round", "feedback"}`; `orchestration.aggregate` gains `{"evaluated", "evaluator_model"}`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_observability_patch.py`:
1. Make `_domain_complex()` accept an evaluator override:

```python
def _domain_complex(evaluator=None):
    return DomainConfig(
        name="sw", description="desc", out_of_domain_reply="Out.",
        intents={
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug"),
        },
        intent_mapping={"faq": "direct", "troubleshooting": "debugging"},
        strategies=["direct", "debugging"],
        orchestration=OrchestrationPolicy(
            enabled=True, min_complexity="complex", intents=["troubleshooting"],
            max_workers=4,
            evaluator=evaluator or EvaluatorPolicy(enabled=False),
        ),
        prompts={
            "direct": "Direct prompt.",
            "debugging": "Debugging prompt.",
        },
    )
```

2. Add scorecard constants and the new test:

```python
_SCORECARD_LOW = ('{"correctness": 2, "relevance": 4, "completeness": 4, '
                  '"technical_depth": 4, "practical_usefulness": 4, "hallucination": 4}')
_SCORECARD_PASS = ('{"correctness": 4, "relevance": 4, "completeness": 4, '
                   '"technical_depth": 4, "practical_usefulness": 4, "hallucination": 4}')


def test_orchestration_evaluator_and_optimizer_recorded(tmp_path):
    store = _store(tmp_path)
    inner = FakeInner([_CLASSIFY_COMPLEX, _PLAN, "w1", "w2", "draft",
                       _SCORECARD_LOW, "improved", _SCORECARD_PASS])
    chat = Chat(inner, _config(), _domain_complex(
        evaluator=EvaluatorPolicy(enabled=True, min_dimension_score=3, max_rounds=1)))
    patch_mod.Installed(store, {}).apply()
    resp = chat.respond("huge debugging task")
    assert resp.text == "improved"
    events, _ = read_events(tmp_path / "obs")
    eval_events = [e for e in events
                   if e["type"] == "decision" and e["phase"] == "orchestration.evaluator"]
    opt_events = [e for e in events
                  if e["type"] == "decision" and e["phase"] == "orchestration.optimizer"]
    assert len(eval_events) == 2
    assert eval_events[0]["data"]["passed"] is False
    assert eval_events[1]["data"]["passed"] is True
    assert len(opt_events) == 1
    assert opt_events[0]["data"]["round"] == 0
    assert "correctness: 2/5" in opt_events[0]["data"]["feedback"]


def test_orchestration_aggregate_records_evaluated_flag(tmp_path):
    store = _store(tmp_path)
    inner = FakeInner([_CLASSIFY_COMPLEX, _PLAN, "w1", "w2", "final"])
    chat = Chat(inner, _config(), _domain_complex())
    patch_mod.Installed(store, {}).apply()
    resp = chat.respond("huge debugging task")
    assert resp.text == "final"
    events, _ = read_events(tmp_path / "obs")
    agg = [e for e in events
           if e["type"] == "decision" and e["phase"] == "orchestration.aggregate"][0]
    assert agg["data"]["evaluated"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_observability_patch.py -v`
Expected: FAIL — no `orchestration.evaluator` / `orchestration.optimizer` decision phases exist.

- [ ] **Step 3: Implement**

In `agent/observability/patch.py`:

1. Add to `DEFAULT_PHASES`:

```python
    "Orchestrator._evaluate": "orchestration.evaluator",
    "Orchestrator._reaggregate": "orchestration.optimizer",
```

2. Add to the `_wrap` factories dict:

```python
            "Orchestrator._evaluate": self._wrap_evaluate,
            "Orchestrator._reaggregate": self._wrap_reaggregate,
```

3. Add the wrapper factories:

```python
    def _wrap_evaluate(self, original, key):
        def wrapper(orch, judge, question, answer):
            inst = _current_inst()
            if inst is None:
                return original(orch, judge, question, answer)
            with phase(inst._phase(key)):
                scorecard = original(orch, judge, question, answer)
                tid = current_trace_id()
                if tid:
                    policy = orch.domain.orchestration
                    threshold = policy.evaluator.min_dimension_score if policy else 3
                    passed = scorecard is not None and all(
                        s >= threshold for s in scorecard.values()
                    )
                    inst._record_decision(tid, inst._phase(key), {
                        "scorecard": scorecard, "passed": bool(passed)})
                return scorecard
        return wrapper

    def _wrap_reaggregate(self, original, key):
        def wrapper(orch, question, strategy, context, results, previous, feedback, round_no, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, strategy, context, results, previous, feedback, round_no, model)
            with phase(inst._phase(key)):
                answer = original(orch, question, strategy, context, results, previous, feedback, round_no, model)
                tid = current_trace_id()
                if tid:
                    inst._record_decision(tid, inst._phase(key), {
                        "round": round_no, "feedback": feedback})
                return answer
        return wrapper
```

4. Update `_wrap_aggregate` to record the evaluated flag:

```python
    def _wrap_aggregate(self, original, key):
        def wrapper(orch, question, strategy, context, results, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, strategy, context, results, model)
            with phase(inst._phase(key)):
                answer = original(orch, question, strategy, context, results, model)
                tid = current_trace_id()
                if tid:
                    policy = orch.domain.orchestration
                    inst._record_decision(tid, inst._phase(key), {
                        "evaluated": bool(policy and policy.evaluator.enabled),
                        "evaluator_model": orch._judge_model()})
                return answer
        return wrapper
```

5. Add the two keys to the `apply()` target list:

```python
            ("Orchestrator._evaluate", Orchestrator, "_evaluate"),
            ("Orchestrator._reaggregate", Orchestrator, "_reaggregate"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_observability_patch.py tests/unit/test_observability_install.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/observability/patch.py tests/unit/test_observability_patch.py
git commit -m "feat: observability records for evaluator/optimizer phases"
```

---

### Task 8: Full-suite verification

**Files:**
- Test: whole suite

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`
Expected: all PASS.

- [ ] **Step 2: Verify no stale references**

Run: `rg -n "strategies\.yaml|unsupported_complex|StrategyDef|OrchestratorConfig|default_strategy|effective_timeout|complexity_gate|worker_timeout" agent/ domain/ config.example.json tests/ || true`
Expected: no matches (except any mention inside `docs/` history).

- [ ] **Step 3: Manual load check**

Run: `uv run python -c "from agent.config import load_domain_config; d = load_domain_config('domain/software_engineering'); print(d.orchestration); print(d.strategies)"`
Expected: prints the parsed `OrchestrationPolicy` and `['analysis', 'code_snippet', 'debugging', 'direct', 'teaching']`.

- [ ] **Step 4: Commit any remaining changes**

```bash
git add -A
git commit -m "chore: verify orchestration policy + evaluator/optimizer implementation" || true
```

---

## Out of scope

- Re-running workers inside the Optimizer.
- More than `max_rounds` evaluator iterations.
- Reintroducing per-worker wall-clock timeouts or per-strategy model overrides.
- Non-software-engineering domain policy content.
- Async refactors.
