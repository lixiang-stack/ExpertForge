# Evaluation Framework (P0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the P0 evaluation system: a golden dataset, a runner that drives the real pipeline per case, classification/routing/answer-quality/cost metrics, per-run JSON results, and an A/B diff — all decoupled from observability.

**Architecture:** A new `agent/evaluation/` package (sibling to `agent/observability/`) whose runner calls `Router.route()` and `Chat.respond()` directly and collects metrics from their return values plus its own recording client wrapper (reads `LLMClient._usage_local`). Golden datasets live under a top-level `evaluation/datasets/`, per-run results under `evaluation/results/` (gitignored). One small core change: `Chat.respond(question, *, route=None)` so evaluation can pass a pre-computed route. `cache_tokens` is added to `LLMClient`.

**Tech Stack:** Python 3.10+, openai SDK, pyyaml, pytest, uv (all already in use).

## Global Constraints

- `git` working tree starts on `main`.
- `uv run pytest -q` must pass (currently 62 tests) after every task.
- Evaluation never reads `TraceStore` / observability events; it is fully decoupled from observability.
- `Chat.respond(question, *, route: RouteResult | None = None) -> ChatResponse` — when `route` is given, skip `Router.route`; default behavior unchanged.
- `LLMClient.chat_completion` stores `_usage_local.cache_tokens` (0 when provider does not report it) alongside the existing `_usage_local.usage`.
- `AgentConfig` gains `evaluation: EvaluationConfig | None = None` (`EvaluationConfig(judge_model: str | None = None, results_dir: str = "evaluation/results")`). `judge_model` is evaluation-only, so it lives inside the `evaluation` block.
- The initial dataset `evaluation/datasets/software_engineering.yaml` uses the real intent ids and strategy ids from `domain/software_engineering/` (`intents.yaml`, `strategies.yaml`, `intent_mapping.yaml`).
- Per-run result files go to `{results_dir}/{YYYY-MM-DD}-{label}.json`.
- Out-of-domain cases use `expected.domain: "other"`, `expected.intent: null`, `expected.complexity: null`, `expected.strategy: reject`, `expected.orchestrate: false`.

---

### Task 1: `cache_tokens` in `LLMClient`

**Files:**
- Modify: `agent/llm.py:51`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: none.
- Produces: `LLMClient._usage_local.cache_tokens` — an `int`, set after every successful `chat_completion`, `0` when the provider does not report cache tokens. Later tasks (`evaluation/runner.py`) read it from the same thread-local.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_llm.py`:

```python
def _usage_with_cache(prompt, completion, cached=0):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = prompt + completion
    details = MagicMock()
    details.cached_tokens = cached
    u.prompt_tokens_details = details
    return u


@patch("agent.llm.OpenAI")
def test_chat_completion_records_cache_tokens(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.usage = _usage_with_cache(10, 5, cached=7)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}])

    assert client._usage_local.cache_tokens == 7


@patch("agent.llm.OpenAI")
def test_chat_completion_cache_tokens_zero_when_absent(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.usage = _usage(10, 5)  # no prompt_tokens_details
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}])

    assert client._usage_local.cache_tokens == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py -k cache -v`
Expected: `FAIL` — `AttributeError: 'thread._local' object has no attribute 'cache_tokens'`.

- [ ] **Step 3: Implement** — in `agent/llm.py`, after the line `self._usage_local.usage = resp.usage`, add:

```python
            details = getattr(resp.usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None)
            self._usage_local.cache_tokens = cached if isinstance(cached, int) else 0
```

The block reads `resp.usage.prompt_tokens_details.cached_tokens` when present, else `0`. `_usage_local.cache_tokens` persists on the thread-local like `_usage_local.usage`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm.py -k cache -v`
Expected: PASS (2 new).

- [ ] **Step 5: Full regression**

Run: `uv run pytest -q`
Expected: all pass (62 + 2 = 64).

- [ ] **Step 6: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat: record cache tokens in LLMClient usage"
```

---

### Task 2: Config — `evaluation` block with `judge_model`

**Files:**
- Modify: `agent/config.py`
- Test: `tests/test_config.py`
- Modify: `config.example.json`

**Interfaces:**
- Consumes: none.
- Produces:
  - `EvaluationConfig` dataclass: `judge_model: str | None = None`, `results_dir: str = "evaluation/results"`.
  - `AgentConfig.evaluation: EvaluationConfig | None = None`.
  - `load_config` populates it from JSON key `evaluation` (object `{"judge_model": "...", "results_dir": "..."}`, absent or non-dict → None). `judge_model` is evaluation-only (used by the judge), so it lives inside the `evaluation` block, not at the top level.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_config.py`:

```python
def test_load_config_evaluation_judge_model_and_results_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "evaluation": {"judge_model": "judge-a", "results_dir": "eval/results"},
    })
    cfg = load_config(path)
    assert cfg.evaluation is not None
    assert cfg.evaluation.judge_model == "judge-a"
    assert cfg.evaluation.results_dir == "eval/results"


def test_load_config_evaluation_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.evaluation is None


def test_load_config_evaluation_ignores_non_dict(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "evaluation": "nope",
    })
    cfg = load_config(path)
    assert cfg.evaluation is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k "evaluation" -v`
Expected: `FAIL` — `AttributeError: 'AgentConfig' object has no attribute 'evaluation'`.

- [ ] **Step 3: Implement** — in `agent/config.py`:

Add the dataclass next to `ObservabilityConfig` (after line 23):

```python
@dataclass
class EvaluationConfig:
    judge_model: str | None = None
    results_dir: str = "evaluation/results"
```

Add the field to `AgentConfig` (after `observability`):

```python
    evaluation: EvaluationConfig | None = None
```

In `load_config`, after the observability block, add:

```python
    raw_eval = raw.get("evaluation")
    evaluation = None
    if isinstance(raw_eval, dict):
        judge_model = raw_eval.get("judge_model")
        judge_model = judge_model if isinstance(judge_model, str) and judge_model else None
        results_dir = raw_eval.get("results_dir") or "evaluation/results"
        evaluation = EvaluationConfig(
            judge_model=judge_model,
            results_dir=results_dir if isinstance(results_dir, str) else "evaluation/results",
        )
```

And pass it into the `AgentConfig(...)` constructor:

```python
        evaluation=evaluation,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all existing + 3 new).

- [ ] **Step 5: Update `config.example.json`** to document the new keys:

```json
{
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "model_low": "",
  "model_high": "",
  "domain_dir": "domain/software_engineering",
  "evaluation": {
    "judge_model": "deepseek-v4-flash",
    "results_dir": "evaluation/results"
  },
  "observability": {
    "enabled": true,
    "data_dir": ".observability",
    "phase_map": {}
  }
}
```

- [ ] **Step 6: Full regression + smoke**

Run: `uv run pytest -q`
Expected: all pass (64 + 3 = 67).

Run: `env -u AGENT_API_KEY uv run python -m agent --ask "What is Go defer?"`
Expected: stdout/stderr contains `Config error: AGENT_API_KEY environment variable is not set. ...` and exit code 1.

- [ ] **Step 7: Commit**

```bash
git add agent/config.py tests/test_config.py config.example.json
git commit -m "feat: add evaluation config with judge_model"
```

---

### Task 3: `Chat.respond` accepts a pre-computed route

**Files:**
- Modify: `agent/chat.py:29-30`
- Test: `tests/test_chat.py`

**Interfaces:**
- Consumes: `RouteResult` from `agent/router.py` (already exists).
- Produces: `Chat.respond(question: str, *, route: RouteResult | None = None) -> ChatResponse`. When `route` is `None`, behavior is unchanged (calls `self.router.route(question)`). When provided, the classification LLM call is skipped and the given route is used.

- [ ] **Step 1: Write the failing test** — append to `tests/test_chat.py`:

```python
def test_respond_with_precomputed_route_skips_classification():
    from agent.router import RouteResult

    client = FakeClient(["the answer"])
    chat = Chat(client, _config(), _domain())
    route = RouteResult(in_domain=True, strategy="direct", intent="faq", complexity="simple")
    resp = chat.respond("what is defer", route=route)
    assert resp.kind == "answer"
    assert resp.text == "the answer"
    assert client.models == ["m"]  # only the answer call; classification was skipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat.py::test_respond_with_precomputed_route_skips_classification -v`
Expected: `FAIL` — `TypeError: respond() got an unexpected keyword argument 'route'`.

- [ ] **Step 3: Implement** — in `agent/chat.py`, change the import line and `respond`:

```python
from .router import RouteResult, Router
```

```python
    def respond(self, question: str, *, route: RouteResult | None = None) -> ChatResponse:
        if route is None:
            route = self.router.route(question)
        if not route.in_domain:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chat.py -v`
Expected: PASS (all existing + 1 new).

- [ ] **Step 5: Full regression**

Run: `uv run pytest -q`
Expected: all pass (68).

- [ ] **Step 6: Commit**

```bash
git add agent/chat.py tests/test_chat.py
git commit -m "feat: Chat.respond accepts pre-computed route"
```

---

### Task 4: Dataset loader

**Files:**
- Create: `agent/evaluation/__init__.py`
- Create: `agent/evaluation/dataset.py`
- Create: `tests/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: `yaml` (already a dependency).
- Produces:
  - `DatasetError(Exception)`.
  - `EvalCase` dataclass: `id: str, question: str, category: str, expected_domain: str, expected_intent: str | None, expected_complexity: str | None, expected_strategy: str, expected_orchestrate: bool, answer_quality: bool, reference: str | None`.
  - `Dataset` dataclass: `domain: str, cases: list[EvalCase]`.
  - `load_dataset(path: str) -> Dataset` — raises `DatasetError` on missing file, bad YAML, or any validation failure.
  - `CATEGORIES = ("knowledge", "problem_solving", "evaluation", "generation", "boundary")`.
  - `COMPLEXITY_LEVELS = ("simple", "medium", "complex")`.
  - `is_in_domain(case: EvalCase, dataset: Dataset) -> bool` — `case.expected_domain == dataset.domain`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_evaluation_dataset.py`:

```python
import pytest

from agent.evaluation.dataset import (
    COMPLEXITY_LEVELS,
    Dataset,
    DatasetError,
    EvalCase,
    load_dataset,
)


def _dataset_path(tmp_path, yaml_text):
    path = tmp_path / "se.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return str(path)


_VALID = """
domain: software_engineering
cases:
  - id: se-001
    question: "What is dependency injection?"
    category: knowledge
    expected:
      domain: software_engineering
      intent: concept_explain
      complexity: simple
      strategy: teaching
      orchestrate: false
    answer_quality: true
    reference: "Dependency injection passes dependencies into a component."
  - id: se-002
    question: "Recommend a restaurant in Tokyo."
    category: boundary
    expected:
      domain: other
      intent: null
      complexity: null
      strategy: reject
      orchestrate: false
"""


def test_load_dataset_valid(tmp_path):
    ds = load_dataset(_dataset_path(tmp_path, _VALID))
    assert isinstance(ds, Dataset)
    assert ds.domain == "software_engineering"
    assert len(ds.cases) == 2
    c = ds.cases[0]
    assert c.id == "se-001"
    assert c.question == "What is dependency injection?"
    assert c.category == "knowledge"
    assert c.expected_domain == "software_engineering"
    assert c.expected_intent == "concept_explain"
    assert c.expected_complexity == "simple"
    assert c.expected_strategy == "teaching"
    assert c.expected_orchestrate is False
    assert c.answer_quality is True
    assert c.reference == "Dependency injection passes dependencies into a component."


def test_out_of_domain_case_fields():
    from agent.evaluation.dataset import is_in_domain

    import tempfile
    path = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    with open(path, "w", encoding="utf-8") as f:
        f.write(_VALID)
    ds = load_dataset(path)
    c = ds.cases[1]
    assert c.expected_domain == "other"
    assert c.expected_intent is None
    assert c.expected_complexity is None
    assert c.expected_strategy == "reject"
    assert is_in_domain(c, ds) is False
    assert is_in_domain(ds.cases[0], ds) is True


def test_load_dataset_answer_quality_defaults_true(tmp_path):
    path = tmp_path / "se.yaml"
    path.write_text(
        'domain: software_engineering\n'
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    category: knowledge\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )
    ds = load_dataset(str(path))
    assert ds.cases[0].answer_quality is True
    assert ds.cases[0].reference is None
    assert ds.cases[0].expected_orchestrate is False


def test_load_dataset_missing_file():
    with pytest.raises(DatasetError):
        load_dataset("/nonexistent/se.yaml")


def test_load_dataset_bad_yaml(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path, ":: not: [valid"))


def test_load_dataset_missing_domain(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path, "cases: []\n"))


def test_load_dataset_missing_cases(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path, "domain: se\n"))


def test_load_dataset_unknown_intent(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path,
            'domain: software_engineering\n'
            'cases:\n'
            '  - id: a\n'
            '    question: "q"\n'
            '    category: knowledge\n'
            '    expected:\n'
            '      domain: software_engineering\n'
            '      intent: bogus\n'
            '      complexity: simple\n'
            '      strategy: direct\n'))


def test_load_dataset_invalid_complexity(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path,
            'domain: software_engineering\n'
            'cases:\n'
            '  - id: a\n'
            '    question: "q"\n'
            '    category: knowledge\n'
            '    expected:\n'
            '      domain: software_engineering\n'
            '      intent: faq\n'
            '      complexity: huge\n'
            '      strategy: direct\n'))


def test_load_dataset_unknown_category(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_dataset_path(tmp_path,
            'domain: software_engineering\n'
            'cases:\n'
            '  - id: a\n'
            '    question: "q"\n'
            '    category: weird\n'
            '    expected:\n'
            '      domain: software_engineering\n'
            '      intent: faq\n'
            '      complexity: simple\n'
            '      strategy: direct\n'))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_dataset.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'agent.evaluation'`.

- [ ] **Step 3: Implement** — create `agent/evaluation/__init__.py`:

```python
"""Evaluation subsystem: golden dataset runner, metrics, and A/B diff.

Deliberately decoupled from observability: reads only pipeline return
values and its own client wrapper.
"""
```

Create `agent/evaluation/dataset.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CATEGORIES = ("knowledge", "problem_solving", "evaluation", "generation", "boundary")
COMPLEXITY_LEVELS = ("simple", "medium", "complex")
OUT_OF_DOMAIN = "other"
REJECT_STRATEGY = "reject"


class DatasetError(Exception):
    """Raised when a golden dataset is missing or invalid."""


@dataclass
class EvalCase:
    id: str
    question: str
    category: str
    expected_domain: str
    expected_intent: str | None
    expected_complexity: str | None
    expected_strategy: str
    expected_orchestrate: bool
    answer_quality: bool
    reference: str | None


@dataclass
class Dataset:
    domain: str
    cases: list[EvalCase]


def is_in_domain(case: EvalCase, dataset: Dataset) -> bool:
    return case.expected_domain == dataset.domain


def _read_yaml(path: Path) -> object:
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise DatasetError(f"Invalid dataset YAML: {path}: {e}")


def _validate_case(raw: object, dataset_domain: str) -> EvalCase:
    if not isinstance(raw, dict):
        raise DatasetError(f"Dataset case must be a mapping, got: {raw!r}")
    cid = raw.get("id")
    question = raw.get("question")
    category = raw.get("category")
    if not isinstance(cid, str) or not cid:
        raise DatasetError(f"Case missing string 'id': {raw!r}")
    if not isinstance(question, str) or not question:
        raise DatasetError(f"Case {cid} missing string 'question'")
    if category not in CATEGORIES:
        raise DatasetError(f"Case {cid} has unknown category {category!r}")
    expected = raw.get("expected")
    if not isinstance(expected, dict):
        raise DatasetError(f"Case {cid} missing 'expected' mapping")
    exp_domain = expected.get("domain")
    if not isinstance(exp_domain, str) or not exp_domain:
        raise DatasetError(f"Case {cid} missing expected.domain")
    if exp_domain not in (dataset_domain, OUT_OF_DOMAIN):
        raise DatasetError(f"Case {cid} expected.domain {exp_domain!r} must be "
                           f"{dataset_domain!r} or {OUT_OF_DOMAIN!r}")
    in_domain = exp_domain == dataset_domain
    intent = expected.get("intent")
    complexity = expected.get("complexity")
    strategy = expected.get("strategy")
    orchestrate = expected.get("orchestrate", False)
    if not isinstance(strategy, str) or not strategy:
        raise DatasetError(f"Case {cid} missing expected.strategy")
    if orchestrate not in (True, False):
        raise DatasetError(f"Case {cid} expected.orchestrate must be a boolean")
    if in_domain:
        if not isinstance(intent, str) or not intent:
            raise DatasetError(f"In-domain case {cid} missing expected.intent")
        if complexity not in COMPLEXITY_LEVELS:
            raise DatasetError(f"Case {cid} invalid complexity {complexity!r}")
    else:
        intent = None
        complexity = None
        if strategy != REJECT_STRATEGY:
            raise DatasetError(f"Out-of-domain case {cid} expected.strategy must be "
                               f"{REJECT_STRATEGY!r}")
        if orchestrate is not False:
            raise DatasetError(f"Out-of-domain case {cid} must have orchestrate: false")
    answer_quality = raw.get("answer_quality", True)
    if answer_quality not in (True, False):
        raise DatasetError(f"Case {cid} answer_quality must be a boolean")
    reference = raw.get("reference")
    return EvalCase(
        id=cid,
        question=question,
        category=category,
        expected_domain=exp_domain,
        expected_intent=intent,
        expected_complexity=complexity,
        expected_strategy=strategy,
        expected_orchestrate=bool(orchestrate),
        answer_quality=bool(answer_quality),
        reference=reference if isinstance(reference, str) else None,
    )


def load_dataset(path: str) -> Dataset:
    raw = _read_yaml(Path(path))
    if not isinstance(raw, dict):
        raise DatasetError(f"Dataset top-level must be a mapping: {path}")
    domain = raw.get("domain")
    if not isinstance(domain, str) or not domain:
        raise DatasetError(f"Dataset missing string 'domain': {path}")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise DatasetError(f"Dataset 'cases' must be a list: {path}")
    return Dataset(domain=domain, cases=[_validate_case(c, domain) for c in cases_raw])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation_dataset.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Full regression**

Run: `uv run pytest -q`
Expected: all pass (68 + 11 = 79).

- [ ] **Step 6: Commit**

```bash
git add agent/evaluation/ tests/test_evaluation_dataset.py
git commit -m "feat: golden dataset loader with validation"
```

---

### Task 5: LLM-as-judge

**Files:**
- Create: `agent/evaluation/judge.py`
- Create: `tests/test_evaluation_judge.py`

**Interfaces:**
- Consumes: `LLMClient` / `LLMError` from `agent.llm`; the `json_mode` path.
- Produces:
  - `JUDGE_DIMENSIONS = ("correctness", "relevance", "completeness", "technical_depth", "practical_usefulness", "hallucination")`.
  - `build_judge_prompt(question: str, answer: str, *, reference: str | None = None) -> str`.
  - `parse_scorecard(text: str) -> dict | None` — extracts the JSON object and validates all six dimensions are ints in 1..5; returns `None` when unparseable/invalid.
  - `class Judge` with `__init__(self, client: LLMClient, model: str)` and `score(self, question: str, answer: str, *, reference: str | None = None) -> dict | None`. Uses `chat_completion(..., json_mode=True, disable_thinking=True)`. Returns `None` if the call raises `LLMError` or the scorecard is invalid. Does not retry.

- [ ] **Step 1: Write the failing tests** — create `tests/test_evaluation_judge.py`:

```python
from agent.evaluation.judge import (
    JUDGE_DIMENSIONS,
    Judge,
    build_judge_prompt,
    parse_scorecard,
)


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False,
                        json_mode=False, json_schema=None):
        self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
        if self.error is not None:
            raise self.error
        return self.response


def test_build_judge_prompt_contains_question_answer_and_dimensions():
    prompt = build_judge_prompt("q?", "the answer", reference="ground truth")
    assert "q?" in prompt
    assert "the answer" in prompt
    assert "ground truth" in prompt
    for d in JUDGE_DIMENSIONS:
        assert d in prompt


def test_parse_scorecard_valid():
    text = ('{"correctness": 4, "relevance": 5, "completeness": 3, '
            '"technical_depth": 4, "practical_usefulness": 5, "hallucination": 2}')
    sc = parse_scorecard(text)
    assert sc is not None
    assert sc["correctness"] == 4
    assert sc["hallucination"] == 2


def test_parse_scorecard_unparseable():
    assert parse_scorecard("not json") is None
    assert parse_scorecard(None) is None


def test_parse_scorecard_missing_or_out_of_range():
    assert parse_scorecard('{"correctness": 4}') is None
    assert parse_scorecard(
        '{"correctness": 4, "relevance": 5, "completeness": 3, '
        '"technical_depth": 4, "practical_usefulness": 5, "hallucination": 99}'
    ) is None


def test_judge_returns_scorecard():
    client = FakeClient(
        '{"correctness": 5, "relevance": 4, "completeness": 4, '
        '"technical_depth": 5, "practical_usefulness": 4, "hallucination": 5}'
    )
    sc = Judge(client, "judge-a").score("q?", "answer")
    assert sc["correctness"] == 5
    messages, model, dt, jm, schema = client.calls[0]
    assert model == "judge-a"
    assert dt is True
    assert jm is True


def test_judge_error_returns_none():
    from agent.llm import LLMError

    sc = Judge(FakeClient(error=LLMError("boom")), "judge-a").score("q?", "answer")
    assert sc is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_judge.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'agent.evaluation.judge'`.

- [ ] **Step 3: Implement** — create `agent/evaluation/judge.py`:

```python
from __future__ import annotations

import json
import re

from agent.llm import LLMClient, LLMError

JUDGE_DIMENSIONS = (
    "correctness",
    "relevance",
    "completeness",
    "technical_depth",
    "practical_usefulness",
    "hallucination",
)

_JUDGE_PROMPT = """You are a strict evaluator of technical answers.

Question: {question}

Agent answer:
{answer}
{reference_block}
Score the answer on each dimension from 1 (worst) to 5 (best):
- correctness: factual accuracy and technical truth
- relevance: how directly it addresses the question
- completeness: whether all important aspects are covered
- technical_depth: depth and sophistication of the explanation
- practical_usefulness: how actionable and useful the answer is
- hallucination: 1 = many unsupported claims, 5 = no unsupported claims

Output ONLY a single JSON object:
{{"correctness": 1-5, "relevance": 1-5, "completeness": 1-5, "technical_depth": 1-5, "practical_usefulness": 1-5, "hallucination": 1-5}}
"""


def build_judge_prompt(question: str, answer: str, *, reference: str | None = None) -> str:
    reference_block = (
        f"\nGround truth reference:\n{reference}" if reference else "\nNo reference provided."
    )
    return _JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        reference_block=reference_block,
    )


def parse_scorecard(text: str | None) -> dict | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    for dim in JUDGE_DIMENSIONS:
        value = data.get(dim)
        if not isinstance(value, int) or not 1 <= value <= 5:
            return None
    return data


class Judge:
    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def score(self, question: str, answer: str, *, reference: str | None = None) -> dict | None:
        prompt = build_judge_prompt(question, answer, reference=reference)
        messages = [{"role": "system", "content": prompt}]
        try:
            text = self.client.chat_completion(
                messages,
                model=self.model,
                disable_thinking=True,
                json_mode=True,
            )
        except LLMError:
            return None
        return parse_scorecard(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation_judge.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Full regression**

Run: `uv run pytest -q`
Expected: all pass (79 + 7 = 86).

- [ ] **Step 6: Commit**

```bash
git add agent/evaluation/judge.py tests/test_evaluation_judge.py
git commit -m "feat: LLM-as-judge scoring with fallback"
```

---

### Task 6: Runner with recording client

**Files:**
- Create: `agent/evaluation/runner.py`
- Create: `tests/test_evaluation_runner.py`

**Interfaces:**
- Consumes:
  - `Router`, `Chat` from `agent.chat`; `RouteResult` from `agent.router`.
  - `LLMClient` from `agent.llm`.
  - `AgentConfig`, `DomainConfig` from `agent.config`.
  - `load_dataset`, `EvalCase`, `Dataset` from `.dataset`.
  - `resolve_model` from `agent.model_router`.
  - `Judge` from `.judge` (Task 5).
- Produces:
  - `RecordingClient` — wraps `LLMClient`; same `chat_completion` signature (passes through `**kwargs`); exposes `calls: list[dict]` where each entry has `model, prompt_tokens, completion_tokens, total_tokens, cache_tokens, latency_ms`; `reset()` clears the list; `model` property delegates to the inner client.
  - `CaseResult` dataclass: `case: EvalCase, in_domain: bool, intent: str | None, complexity: str | None, strategy: str, orchestrate: bool, answer: str | None, actual_model: str | None, expected_model: str | None, scorecard: dict | None, llm_calls: int, in_tokens: int, out_tokens: int, total_tokens: int, cache_tokens: int, latency_ms: float`.
  - `run_evaluation(config: AgentConfig, domain: DomainConfig, dataset: Dataset, client: LLMClient, *, skip_quality: bool = False) -> list[CaseResult]`.

`run_evaluation` per case (sequential):
1. `recorder = RecordingClient(client)` — wrap; the SAME `client` instance is shared, but router/chat/judge receive the `recorder` so every LLM call is captured.
2. `router = Router(recorder, config, domain)`, `judge = Judge(recorder, config.evaluation.judge_model if config.evaluation else config.model)` built once.
3. For each case: `recorder.reset()`; `chat = Chat(recorder, config, domain)` (fresh per case so history never leaks across cases); `route = router.route(case.question)`; `expected_model = resolve_model(config, domain, route, config.model)`; if `case.answer_quality and not skip_quality`: `resp = chat.respond(case.question, route=route)`; `answer = resp.text`; else `answer = None`. If judged, `scorecard = judge.score(case.question, answer, reference=case.reference)`; else `None`. `actual_model` = the model of the last recorded call (or `None` if no calls). Aggregate recorded calls into the cost fields.

- [ ] **Step 1: Write the failing tests** — create `tests/test_evaluation_runner.py`:

```python
import json

from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig, EvaluationConfig, IntentDef, StrategyDef
from agent.evaluation.dataset import Dataset, EvalCase, load_dataset
from agent.evaluation.runner import RecordingClient, run_evaluation
from agent.llm import LLMClient


def _dataset():
    return Dataset(domain="software_engineering", cases=[
        EvalCase(
            id="se-001", question="what is defer",
            category="knowledge", expected_domain="software_engineering",
            expected_intent="faq", expected_complexity="simple",
            expected_strategy="direct", expected_orchestrate=False,
            answer_quality=True, reference="short",
        ),
        EvalCase(
            id="se-002", question="recommend a restaurant",
            category="boundary", expected_domain="other",
            expected_intent=None, expected_complexity=None,
            expected_strategy="reject", expected_orchestrate=False,
            answer_quality=False, reference=None,
        ),
    ])


def _domain():
    return DomainConfig(
        name="sw", description="desc", out_of_domain_reply="Out.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies={"direct": StrategyDef("direct", default=True)},
        default_strategy="direct",
        prompts={"direct": "Direct prompt.", "unsupported_complex": "x."},
    )


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="cm",
                       domain_dir="d", model_low="low-a", model_high="high-a",
                       evaluation=EvaluationConfig(judge_model="judge-a"))


class FakeClient:
    def __init__(self, responses, usage=None):
        self.responses = list(responses)
        self._usage_local = __import__("threading").local()
        self.models = []
        self.json_modes = []
        self.usage_queue = list(usage or [])

    def chat_completion(self, messages, model=None, temperature=0.3,
                        disable_thinking=False, json_mode=False, json_schema=None):
        self.models.append(model)
        self.json_modes.append(json_mode)
        if self.usage_queue:
            self._set_usage(*self.usage_queue.pop(0))
        return self.responses.pop(0)

    def _set_usage(self, prompt, completion, cached=0):
        class U:
            pass
        u = U()
        u.prompt_tokens = prompt
        u.completion_tokens = completion
        u.total_tokens = prompt + completion
        details = U()
        details.cached_tokens = cached
        u.prompt_tokens_details = details
        self._usage_local.usage = u
        self._usage_local.cache_tokens = cached

    def _record_usage(self, prompt, completion, cached=0):
        """Set the usage seen by the NEXT chat_completion call."""
        self.usage_queue.append((prompt, completion, cached))


def test_recording_client_records_usage_and_latency():
    inner = FakeClient(["hello"], usage=[(10, 5, 3)])
    rc = RecordingClient(inner)
    out = rc.chat_completion([{"role": "user", "content": "hi"}], model="m2")
    assert out == "hello"
    assert rc.calls[0]["model"] == "m2"
    assert rc.calls[0]["prompt_tokens"] == 10
    assert rc.calls[0]["completion_tokens"] == 5
    assert rc.calls[0]["total_tokens"] == 15
    assert rc.calls[0]["cache_tokens"] == 3
    assert rc.calls[0]["latency_ms"] >= 0
    rc.reset()
    assert rc.calls == []


def test_run_evaluation_answers_and_judges():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        "the answer",
        '{"correctness": 4, "relevance": 5, "completeness": 3, '
        '"technical_depth": 4, "practical_usefulness": 5, "hallucination": 4}',
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ])
    client._record_usage(10, 5, cached=2)   # classification
    client._record_usage(20, 8, cached=4)   # answer generation
    client._record_usage(5, 2, cached=1)    # judge
    results = run_evaluation(_config(), _domain(), _dataset(), client)
    assert len(results) == 2
    r0 = results[0]
    assert r0.case.id == "se-001"
    assert r0.in_domain is True
    assert r0.intent == "faq"
    assert r0.complexity == "simple"
    assert r0.strategy == "direct"
    assert r0.orchestrate is False
    assert r0.answer == "the answer"
    assert r0.actual_model == "judge-a"  # judge call is last
    assert r0.expected_model == "low-a"
    assert r0.scorecard is not None
    assert r0.scorecard["correctness"] == 4
    assert r0.llm_calls == 3
    assert r0.in_tokens == 35
    assert r0.out_tokens == 15
    assert r0.total_tokens == 50
    assert r0.cache_tokens == 7


def test_run_evaluation_rejects_out_of_domain():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ])
    results = run_evaluation(_config(), _domain(), _dataset(), client, skip_quality=True)
    r1 = results[1]
    assert r1.in_domain is False
    assert r1.strategy == "reject"
    assert r1.answer is None
    assert r1.scorecard is None
    assert r1.llm_calls == 1
    assert r1.actual_model == "cm"


def test_run_evaluation_skip_quality_skips_answer():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ])
    results = run_evaluation(_config(), _domain(), _dataset(), client, skip_quality=True)
    r0 = results[0]
    assert r0.answer is None
    assert r0.scorecard is None
    assert r0.llm_calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_runner.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'agent.evaluation.runner'`.

- [ ] **Step 3: Implement** — create `agent/evaluation/runner.py`:

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig
from agent.llm import LLMClient
from agent.model_router import resolve_model
from agent.router import Router

from .dataset import EvalCase, Dataset
from .judge import Judge


class RecordingClient:
    """Thin LLMClient wrapper that records per-call usage and latency.

    Reads the same thread-local `_usage_local` that `LLMClient` populates;
    completely independent of observability.
    """

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.calls: list[dict] = []

    @property
    def model(self) -> str:
        return self._inner.model

    def reset(self) -> None:
        self.calls = []

    def _usage(self):
        usage = getattr(self._inner, "_usage_local", None)
        return getattr(usage, "usage", None)

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs) -> str:
        started = time.perf_counter()
        text = self._inner.chat_completion(
            messages, model=model, temperature=temperature, **kwargs
        )
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        u = self._usage()
        usage_local = getattr(self._inner, "_usage_local", None)
        self.calls.append({
            "model": model or self._inner.model,
            "prompt_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) if u else 0,
            "total_tokens": getattr(u, "total_tokens", 0) if u else 0,
            "cache_tokens": getattr(usage_local, "cache_tokens", 0) if usage_local else 0,
            "latency_ms": elapsed,
        })
        return text

    def chat_completion_stream(self, messages, *, model=None, temperature=0.7, **kwargs):
        yield from self._inner.chat_completion_stream(
            messages, model=model, temperature=temperature, **kwargs
        )


@dataclass
class CaseResult:
    case: EvalCase
    in_domain: bool
    intent: str | None
    complexity: str | None
    strategy: str
    orchestrate: bool
    answer: str | None
    actual_model: str | None
    expected_model: str | None
    scorecard: dict | None
    llm_calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    total_tokens: int = 0
    cache_tokens: int = 0
    latency_ms: float = 0.0


def _sum_calls(calls: list[dict]) -> dict:
    return {
        "llm_calls": len(calls),
        "in_tokens": sum(c["prompt_tokens"] for c in calls),
        "out_tokens": sum(c["completion_tokens"] for c in calls),
        "total_tokens": sum(c["total_tokens"] for c in calls),
        "cache_tokens": sum(c["cache_tokens"] for c in calls),
        "latency_ms": round(sum(c["latency_ms"] for c in calls), 1),
    }


def run_evaluation(
    config: AgentConfig,
    domain: DomainConfig,
    dataset: Dataset,
    client: LLMClient,
    *,
    skip_quality: bool = False,
) -> list[CaseResult]:
    recorder = RecordingClient(client)
    router = Router(recorder, config, domain)
    judge = Judge(recorder,
                  config.evaluation.judge_model if config.evaluation else config.model)
    results: list[CaseResult] = []
    for case in dataset.cases:
        recorder.reset()
        chat = Chat(recorder, config, domain)  # fresh history per case
        route = router.route(case.question)
        expected_model = resolve_model(config, domain, route, config.model)
        answer = None
        scorecard = None
        if case.answer_quality and not skip_quality:
            resp = chat.respond(case.question, route=route)
            answer = resp.text
            scorecard = judge.score(case.question, answer, reference=case.reference)
        costs = _sum_calls(recorder.calls)
        actual_model = recorder.calls[-1]["model"] if recorder.calls else None
        results.append(CaseResult(
            case=case,
            in_domain=route.in_domain,
            intent=route.intent,
            complexity=route.complexity,
            strategy=route.strategy,
            orchestrate=route.orchestrate,
            answer=answer,
            actual_model=actual_model,
            expected_model=expected_model,
            scorecard=scorecard,
            **costs,
        ))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation_runner.py -v`
Expected: PASS (the `agent.evaluation.judge` module already exists from Task 5).

- [ ] **Step 5: Full regression**

Run: `uv run pytest -q`
Expected: all pass (86 + 4 = 90).

- [ ] **Step 6: Commit**

```bash
git add agent/evaluation/runner.py tests/test_evaluation_runner.py
git commit -m "feat: evaluation runner with recording client"
```

---

### Task 7: Metrics computation

**Files:**
- Create: `agent/evaluation/metrics.py`
- Create: `tests/test_evaluation_metrics.py`

**Interfaces:**
- Consumes: `CaseResult` from `.runner`; `Dataset`, `is_in_domain` from `.dataset`.
- Produces:
  - `compute_metrics(dataset: Dataset, results: list[CaseResult]) -> dict` returning:
    ```python
    {
      "n_cases": int,
      "classification": {
        "domain_accuracy": float | None,
        "intent_accuracy": float | None,
        "complexity_accuracy": float | None,
        "per_intent": {intent: float},      # ordered by first appearance
      },
      "routing": {
        "strategy_accuracy": float | None,
        "orchestration_accuracy": float | None,
        "model_routing_accuracy": float | None,
      },
      "answer_quality": {dimension: float},  # means over judged cases; empty dict when none
      "cost": {
        "llm_calls": int, "in_tokens": int, "out_tokens": int,
        "total_tokens": int, "cache_tokens": int, "latency_ms": float,
        "by_path": {
          "simple": {same cost keys}, "medium": {...}, "complex": {...},
        },
      },
    }
    ```
  - `_accuracy(correct: int, total: int) -> float | None` — `None` when `total == 0`, else `round(correct / total, 4)`.

Rules (from spec):
- `domain_accuracy` over ALL cases: predicted `in_domain` equals `is_in_domain(case, dataset)`.
- `intent_accuracy` / `complexity_accuracy` over in-domain cases only; predicted value equals expected.
- Per-intent breakdown over in-domain cases grouped by `case.expected_intent`.
- `strategy_accuracy` over ALL cases: `result.strategy == case.expected_strategy`.
- `orchestration_accuracy` over ALL cases: `result.orchestrate == case.expected_orchestrate`.
- `model_routing_accuracy` over cases with both `actual_model` and `expected_model`: equal.
- `answer_quality` means over cases with a `scorecard`.
- `cost.by_path` groups by `case.expected_complexity` for in-domain cases (out-of-domain excluded), keys `simple`/`medium`/`complex` (only present when the group is non-empty).

- [ ] **Step 1: Write the failing tests** — create `tests/test_evaluation_metrics.py`:

```python
from agent.evaluation.dataset import Dataset, EvalCase
from agent.evaluation.metrics import _accuracy, compute_metrics
from agent.evaluation.runner import CaseResult


def _case(cid, domain="software_engineering", intent="faq", complexity="simple",
          strategy="direct", orchestrate=False):
    return EvalCase(
        id=cid, question=f"q {cid}", category="knowledge",
        expected_domain=domain, expected_intent=intent,
        expected_complexity=complexity, expected_strategy=strategy,
        expected_orchestrate=orchestrate, answer_quality=True, reference=None,
    )


def _result(case, *, in_domain=True, intent=None, complexity=None, strategy=None,
            orchestrate=False, actual_model=None, expected_model=None, scorecard=None,
            in_tokens=10, out_tokens=5, cache_tokens=1, latency=10.0):
    return CaseResult(
        case=case, in_domain=in_domain,
        intent=case.expected_intent if intent is None else intent,
        complexity=case.expected_complexity if complexity is None else complexity,
        strategy=case.expected_strategy if strategy is None else strategy,
        orchestrate=orchestrate, answer="a", actual_model=actual_model,
        expected_model=expected_model, scorecard=scorecard,
        llm_calls=2, in_tokens=in_tokens, out_tokens=out_tokens,
        total_tokens=in_tokens + out_tokens, cache_tokens=cache_tokens, latency_ms=latency,
    )


def _m(cases, results):
    return compute_metrics(Dataset(domain="software_engineering", cases=cases), results)


def test_accuracy_none_when_empty():
    assert _accuracy(0, 0) is None
    assert _accuracy(3, 4) == 0.75


def test_perfect_classification_and_routing():
    cases = [_case("a"), _case("b")]
    results = [_result(cases[0]), _result(cases[1])]
    m = _m(cases, results)
    assert m["n_cases"] == 2
    assert m["classification"]["domain_accuracy"] == 1.0
    assert m["classification"]["intent_accuracy"] == 1.0
    assert m["classification"]["complexity_accuracy"] == 1.0
    assert m["routing"]["strategy_accuracy"] == 1.0
    assert m["routing"]["orchestration_accuracy"] == 1.0
    assert m["routing"]["model_routing_accuracy"] is None  # no actual/expected models


def test_wrong_intent_counts_intent_only():
    cases = [_case("a", intent="faq"), _case("b", intent="faq")]
    results = [
        _result(cases[0], intent="faq"),
        _result(cases[1], intent="concept_explain"),
    ]
    m = _m(cases, results)
    assert m["classification"]["intent_accuracy"] == 0.5
    assert m["classification"]["domain_accuracy"] == 1.0  # both in-domain
    assert m["classification"]["complexity_accuracy"] == 1.0
    assert m["classification"]["per_intent"]["faq"] == 0.5


def test_out_of_domain_affects_domain_and_strategy_only():
    cases = [_case("a"), _case("ood", domain="other", intent=None, complexity=None,
                      strategy="reject")]
    results = [
        _result(cases[0]),
        _result(cases[1], in_domain=False, intent=None, complexity=None, strategy="reject"),
    ]
    m = _m(cases, results)
    assert m["classification"]["domain_accuracy"] == 1.0
    # intent/complexity exclude the out-of-domain case
    assert m["classification"]["intent_accuracy"] == 1.0
    assert m["classification"]["complexity_accuracy"] == 1.0
    assert m["routing"]["strategy_accuracy"] == 1.0
    # by_path excludes out-of-domain
    assert "simple" in m["cost"]["by_path"]
    assert set(m["cost"]["by_path"]) == {"simple"}


def test_out_of_domain_wrong_in_domain_prediction():
    cases = [_case("ood", domain="other", intent=None, complexity=None, strategy="reject")]
    results = [_result(cases[0], in_domain=True, strategy="direct")]
    m = _m(cases, results)
    assert m["classification"]["domain_accuracy"] == 0.0
    assert m["routing"]["strategy_accuracy"] == 0.0


def test_orchestration_accuracy():
    cases = [_case("a", orchestrate=True), _case("b", orchestrate=False)]
    results = [_result(cases[0], orchestrate=True), _result(cases[1], orchestrate=False)]
    m = _m(cases, results)
    assert m["routing"]["orchestration_accuracy"] == 1.0


def test_model_routing_accuracy():
    cases = [_case("a"), _case("b")]
    results = [
        _result(cases[0], actual_model="low-a", expected_model="low-a"),
        _result(cases[1], actual_model="high-a", expected_model="low-a"),
    ]
    m = _m(cases, results)
    assert m["routing"]["model_routing_accuracy"] == 0.5


def test_answer_quality_means():
    cases = [_case("a"), _case("b")]
    results = [
        _result(cases[0], scorecard={"correctness": 4, "relevance": 5, "completeness": 3,
                                     "technical_depth": 4, "practical_usefulness": 5,
                                     "hallucination": 4}),
        _result(cases[1], scorecard=None),
    ]
    m = _m(cases, results)
    assert m["answer_quality"]["correctness"] == 4.0
    assert m["answer_quality"]["relevance"] == 5.0


def test_cost_aggregates_and_by_path():
    cases = [
        _case("s", complexity="simple"),
        _case("m", complexity="medium"),
        _case("c", complexity="complex"),
    ]
    results = [_result(cases[0], in_tokens=10, out_tokens=5, cache_tokens=1, latency=10.0),
               _result(cases[1], in_tokens=20, out_tokens=8, cache_tokens=2, latency=20.0),
               _result(cases[2], in_tokens=30, out_tokens=12, cache_tokens=3, latency=30.0)]
    m = _m(cases, results)
    cost = m["cost"]
    assert cost["llm_calls"] == 6
    assert cost["in_tokens"] == 60
    assert cost["out_tokens"] == 25
    assert cost["total_tokens"] == 85
    assert cost["cache_tokens"] == 6
    assert set(cost["by_path"]) == {"simple", "medium", "complex"}
    assert cost["by_path"]["simple"]["in_tokens"] == 10
    assert cost["by_path"]["complex"]["total_tokens"] == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_metrics.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'agent.evaluation.metrics'`.

- [ ] **Step 3: Implement** — create `agent/evaluation/metrics.py`:

```python
from __future__ import annotations

from agent.evaluation.dataset import Dataset, is_in_domain
from agent.evaluation.judge import JUDGE_DIMENSIONS
from agent.evaluation.runner import CaseResult

_COST_KEYS = ("llm_calls", "in_tokens", "out_tokens", "total_tokens",
              "cache_tokens", "latency_ms")


def _accuracy(correct: int, total: int) -> float | None:
    if total == 0:
        return None
    return round(correct / total, 4)


def _zero_cost() -> dict:
    return {k: 0 for k in _COST_KEYS}


def _add_cost(acc: dict, r: CaseResult) -> None:
    for k in _COST_KEYS:
        acc[k] += getattr(r, k, 0)


def compute_metrics(dataset: Dataset, results: list[CaseResult]) -> dict:
    n = len(results)
    domain_correct = 0
    intent_total = 0
    intent_correct = 0
    complexity_total = 0
    complexity_correct = 0
    strategy_correct = 0
    orchestration_correct = 0
    model_total = 0
    model_correct = 0
    per_intent: dict[str, list[bool]] = {}
    per_intent_order: list[str] = []
    judged: list[dict] = []
    total_cost = _zero_cost()
    by_path: dict[str, dict] = {}

    for r in results:
        c = r.case
        expected_in = is_in_domain(c, dataset)
        if r.in_domain == expected_in:
            domain_correct += 1
        if r.strategy == c.expected_strategy:
            strategy_correct += 1
        if r.orchestrate == c.expected_orchestrate:
            orchestration_correct += 1
        if r.actual_model is not None and r.expected_model is not None:
            model_total += 1
            if r.actual_model == r.expected_model:
                model_correct += 1
        if expected_in:
            intent_total += 1
            if r.intent == c.expected_intent:
                intent_correct += 1
                per_intent.setdefault(c.expected_intent, []).append(True)
            else:
                per_intent.setdefault(c.expected_intent, []).append(False)
            if c.expected_intent not in per_intent_order:
                per_intent_order.append(c.expected_intent)
            complexity_total += 1
            if r.complexity == c.expected_complexity:
                complexity_correct += 1
            path = c.expected_complexity
            if path not in by_path:
                by_path[path] = _zero_cost()
            _add_cost(by_path[path], r)
        if r.scorecard is not None:
            judged.append(r.scorecard)
        _add_cost(total_cost, r)

    total_cost["latency_ms"] = round(total_cost["latency_ms"], 1)
    for path in by_path:
        by_path[path]["latency_ms"] = round(by_path[path]["latency_ms"], 1)

    per_intent_accuracy = {}
    for iid in per_intent_order:
        marks = per_intent[iid]
        per_intent_accuracy[iid] = _accuracy(sum(marks), len(marks))

    answer_quality = {}
    if judged:
        for dim in JUDGE_DIMENSIONS:
            answer_quality[dim] = round(
                sum(j[dim] for j in judged) / len(judged), 2
            )

    return {
        "n_cases": n,
        "classification": {
            "domain_accuracy": _accuracy(domain_correct, n),
            "intent_accuracy": _accuracy(intent_correct, intent_total),
            "complexity_accuracy": _accuracy(complexity_correct, complexity_total),
            "per_intent": per_intent_accuracy,
        },
        "routing": {
            "strategy_accuracy": _accuracy(strategy_correct, n),
            "orchestration_accuracy": _accuracy(orchestration_correct, n),
            "model_routing_accuracy": _accuracy(model_correct, model_total),
        },
        "answer_quality": answer_quality,
        "cost": {"by_path": by_path, **total_cost},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation_metrics.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Full regression**

Run: `uv run pytest -q`
Expected: all pass (90 + 10 = 100).

- [ ] **Step 6: Commit**

```bash
git add agent/evaluation/metrics.py tests/test_evaluation_metrics.py
git commit -m "feat: evaluation metrics aggregation"
```

---

### Task 8: Report — terminal summary + JSON write

**Files:**
- Create: `agent/evaluation/report.py`
- Create: `tests/test_evaluation_report.py`

**Interfaces:**
- Consumes: `metrics` dict from `.metrics`; `CaseResult` list from `.runner`.
- Produces:
  - `serialize_results(cases: list[CaseResult], metrics: dict, *, domain: str, label: str, model: str, judge_model: str | None, skip_quality: bool, dataset_path: str) -> dict` — the run record written to JSON. Includes `domain`, `label`, `model`, `judge_model`, `skip_quality`, `dataset`, `metrics`, and `cases` (each as a dict with case metadata + predictions + scorecard + cost).
  - `write_result(results_dir: str, record: dict, *, label: str) -> str` — writes `{results_dir}/{YYYY-MM-DD}-{label}.json` (creates the dir), returns the absolute path.
  - `format_summary(record: dict) -> str` — a terminal summary string.

- [ ] **Step 1: Write the failing tests** — create `tests/test_evaluation_report.py`:

```python
import json

from agent.evaluation.dataset import Dataset, EvalCase
from agent.evaluation.metrics import compute_metrics
from agent.evaluation.report import format_summary, serialize_results, write_result
from agent.evaluation.runner import CaseResult


def _case(cid):
    return EvalCase(
        id=cid, question=f"q {cid}", category="knowledge",
        expected_domain="software_engineering", expected_intent="faq",
        expected_complexity="simple", expected_strategy="direct",
        expected_orchestrate=False, answer_quality=True, reference=None,
    )


def _result(case):
    return CaseResult(
        case=case, in_domain=True, intent="faq", complexity="simple",
        strategy="direct", orchestrate=False, answer="the answer",
        actual_model="low-a", expected_model="low-a",
        scorecard={"correctness": 4, "relevance": 5, "completeness": 4,
                   "technical_depth": 4, "practical_usefulness": 5, "hallucination": 5},
        llm_calls=2, in_tokens=10, out_tokens=5, total_tokens=15,
        cache_tokens=1, latency_ms=10.0,
    )


def _record():
    cases = [_case("a")]
    results = [_result(cases[0])]
    ds = Dataset(domain="software_engineering", cases=cases)
    m = compute_metrics(ds, results)
    return serialize_results(
        results, m, domain="software_engineering", label="run1", model="m",
        judge_model="judge-a", skip_quality=False,
        dataset_path="evaluation/datasets/software_engineering.yaml",
    )


def test_serialize_results_contains_expected_keys():
    rec = _record()
    assert rec["domain"] == "software_engineering"
    assert rec["label"] == "run1"
    assert rec["model"] == "m"
    assert rec["judge_model"] == "judge-a"
    assert rec["skip_quality"] is False
    assert rec["dataset"] == "evaluation/datasets/software_engineering.yaml"
    assert rec["metrics"]["n_cases"] == 1
    case = rec["cases"][0]
    assert case["id"] == "a"
    assert case["question"] == "q a"
    assert case["intent"] == "faq"
    assert case["complexity"] == "simple"
    assert case["strategy"] == "direct"
    assert case["scorecard"]["correctness"] == 4
    assert case["llm_calls"] == 2
    assert case["in_tokens"] == 10


def test_write_result_creates_json(tmp_path):
    rec = _record()
    path = write_result(str(tmp_path / "results"), rec, label="run1")
    assert path.endswith("run1.json")
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["label"] == "run1"
    assert loaded["metrics"]["n_cases"] == 1


def test_format_summary_contains_key_sections():
    text = format_summary(_record())
    assert "run1" in text
    assert "classification" in text.lower()
    assert "routing" in text.lower()
    assert "answer_quality" in text.lower()
    assert "cost" in text.lower()
    assert "domain_accuracy" in text
    assert "intent_accuracy" in text
    assert "strategy_accuracy" in text
    assert "model_routing_accuracy" in text
    assert "correctness" in text
    assert "total_tokens" in text
    assert "simple" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_report.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'agent.evaluation.report'`.

- [ ] **Step 3: Implement** — create `agent/evaluation/report.py`:

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _case_record(r) -> dict:
    return {
        "id": r.case.id,
        "question": r.case.question,
        "category": r.case.category,
        "expected_domain": r.case.expected_domain,
        "expected_intent": r.case.expected_intent,
        "expected_complexity": r.case.expected_complexity,
        "expected_strategy": r.case.expected_strategy,
        "expected_orchestrate": r.case.expected_orchestrate,
        "in_domain": r.in_domain,
        "intent": r.intent,
        "complexity": r.complexity,
        "strategy": r.strategy,
        "orchestrate": r.orchestrate,
        "answer": r.answer,
        "actual_model": r.actual_model,
        "expected_model": r.expected_model,
        "scorecard": r.scorecard,
        "llm_calls": r.llm_calls,
        "in_tokens": r.in_tokens,
        "out_tokens": r.out_tokens,
        "total_tokens": r.total_tokens,
        "cache_tokens": r.cache_tokens,
        "latency_ms": r.latency_ms,
    }


def serialize_results(
    cases,
    metrics,
    *,
    domain: str,
    label: str,
    model: str,
    judge_model: str | None,
    skip_quality: bool,
    dataset_path: str,
) -> dict:
    return {
        "domain": domain,
        "label": label,
        "model": model,
        "judge_model": judge_model,
        "skip_quality": skip_quality,
        "dataset": dataset_path,
        "metrics": metrics,
        "cases": [_case_record(r) for r in cases],
    }


def write_result(results_dir: str, record: dict, *, label: str) -> str:
    base = Path(results_dir)
    base.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    path = base / f"{day}-{label}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _fmt_accuracy(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _fmt_cost(cost: dict) -> str:
    return (
        f"llm_calls={cost.get('llm_calls', 0)} "
        f"in={cost.get('in_tokens', 0)} out={cost.get('out_tokens', 0)} "
        f"total={cost.get('total_tokens', 0)} cache={cost.get('cache_tokens', 0)} "
        f"latency={cost.get('latency_ms', 0)}ms"
    )


def format_summary(record: dict) -> str:
    m = record["metrics"]
    cls = m["classification"]
    routing = m["routing"]
    aq = m["answer_quality"]
    cost = m["cost"]
    lines = [
        f"Evaluation run: {record['label']}  (domain={record['domain']}, "
        f"cases={m['n_cases']}, model={record['model']}, "
        f"judge_model={record['judge_model'] or record['model']})",
        "",
        "Classification:",
        f"  domain_accuracy     {_fmt_accuracy(cls['domain_accuracy'])}",
        f"  intent_accuracy     {_fmt_accuracy(cls['intent_accuracy'])}",
        f"  complexity_accuracy {_fmt_accuracy(cls['complexity_accuracy'])}",
    ]
    if cls["per_intent"]:
        lines.append("  per_intent:")
        for iid, acc in cls["per_intent"].items():
            lines.append(f"    {iid}: {_fmt_accuracy(acc)}")
    lines += [
        "",
        "Routing:",
        f"  strategy_accuracy        {_fmt_accuracy(routing['strategy_accuracy'])}",
        f"  orchestration_accuracy   {_fmt_accuracy(routing['orchestration_accuracy'])}",
        f"  model_routing_accuracy   {_fmt_accuracy(routing['model_routing_accuracy'])}",
        "",
        "Answer quality (judged cases):",
    ]
    if aq:
        for dim, mean in aq.items():
            lines.append(f"  {dim}: {mean}")
    else:
        lines.append("  (none)")
    lines += ["", "Cost / latency (total):", f"  {_fmt_cost(cost)}", "  by_path:"]
    for path, pcost in cost["by_path"].items():
        lines.append(f"    {path}: {_fmt_cost(pcost)}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation_report.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full regression**

Run: `uv run pytest -q`
Expected: all pass (100 + 3 = 103).

- [ ] **Step 6: Commit**

```bash
git add agent/evaluation/report.py tests/test_evaluation_report.py
git commit -m "feat: evaluation report and summary"
```

---

### Task 9: A/B diff

**Files:**
- Create: `agent/evaluation/diff.py`
- Create: `tests/test_evaluation_diff.py`

**Interfaces:**
- Consumes: run records (dicts) from `.report`.
- Produces:
  - `diff_runs(record_a: dict, record_b: dict) -> str` — a comparison string.
  - `load_result(path: str) -> dict` — loads a JSON result file, raises `ValueError` on bad JSON.

The diff compares, per metric: accuracy numbers, answer-quality means, and the aggregate cost block. For numeric accuracy values where both sides are non-`None`, print `a -> b (delta +x.xx)`. When a value is `None`, print `n/a`. Print a per-metric line for: `domain_accuracy`, `intent_accuracy`, `complexity_accuracy`, `strategy_accuracy`, `orchestration_accuracy`, `model_routing_accuracy`, each answer-quality dimension, and each cost aggregate key.

- [ ] **Step 1: Write the failing tests** — create `tests/test_evaluation_diff.py`:

```python
import json

from agent.evaluation.diff import diff_runs, load_result


def _run(label, domain_acc, intent_acc, correctness, total_tokens):
    return {
        "label": label,
        "model": "m",
        "metrics": {
            "n_cases": 2,
            "classification": {
                "domain_accuracy": domain_acc,
                "intent_accuracy": intent_acc,
                "complexity_accuracy": 1.0,
                "per_intent": {"faq": 1.0},
            },
            "routing": {
                "strategy_accuracy": 1.0,
                "orchestration_accuracy": 1.0,
                "model_routing_accuracy": 1.0,
            },
            "answer_quality": {"correctness": correctness},
            "cost": {
                "llm_calls": 4, "in_tokens": 100, "out_tokens": 50,
                "total_tokens": total_tokens, "cache_tokens": 10,
                "latency_ms": 500.0, "by_path": {},
            },
        },
    }


def test_diff_shows_deltas():
    a = _run("a", 1.0, 0.5, 4.0, 100)
    b = _run("b", 0.75, 0.75, 4.5, 120)
    text = diff_runs(a, b)
    assert "domain_accuracy" in text
    assert "0.75" in text
    assert "intent_accuracy" in text
    assert "correctness" in text
    assert "total_tokens" in text


def test_diff_handles_none_accuracy():
    a = _run("a", None, 0.5, 4.0, 100)
    b = _run("b", 0.75, None, None, 120)
    text = diff_runs(a, b)
    assert "n/a" in text


def test_load_result_roundtrip(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_run("x", 1.0, 1.0, 5.0, 10)), encoding="utf-8")
    rec = load_result(str(path))
    assert rec["label"] == "x"


def test_load_result_bad_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ not json", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        load_result(str(path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_diff.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'agent.evaluation.diff'`.

- [ ] **Step 3: Implement** — create `agent/evaluation/diff.py`:

```python
from __future__ import annotations

import json


def load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Result file must contain a JSON object: {path}")
    return data


def _num(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _diff_value(a, b) -> str:
    if a is None or b is None:
        return f"{_num(a)} -> {_num(b)}"
    return f"{_num(a)} -> {_num(b)} (delta {b - a:+.2f})"


def diff_runs(record_a: dict, record_b: dict) -> str:
    ma = record_a["metrics"]
    mb = record_b["metrics"]
    lines = [
        f"Comparing {record_a['label']} (model={record_a.get('model')}) -> "
        f"{record_b['label']} (model={record_b.get('model')})",
        "",
        "Classification:",
    ]
    clsa = ma["classification"]
    clsb = mb["classification"]
    for key in ("domain_accuracy", "intent_accuracy", "complexity_accuracy"):
        lines.append(f"  {key}: {_diff_value(clsa[key], clsb[key])}")
    lines += ["", "Routing:"]
    ra = ma["routing"]
    rb = mb["routing"]
    for key in ("strategy_accuracy", "orchestration_accuracy", "model_routing_accuracy"):
        lines.append(f"  {key}: {_diff_value(ra[key], rb[key])}")
    lines += ["", "Answer quality:"]
    aqa = ma["answer_quality"]
    aqb = mb["answer_quality"]
    for dim in sorted(set(aqa) | set(aqb)):
        lines.append(f"  {dim}: {_diff_value(aqa.get(dim), aqb.get(dim))}")
    lines += ["", "Cost:"]
    ca = ma["cost"]
    cb = mb["cost"]
    for key in ("llm_calls", "in_tokens", "out_tokens", "total_tokens",
                "cache_tokens", "latency_ms"):
        lines.append(f"  {key}: {_diff_value(ca[key], cb[key])}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation_diff.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Full regression**

Run: `uv run pytest -q`
Expected: all pass (103 + 4 = 107).

- [ ] **Step 6: Commit**

```bash
git add agent/evaluation/diff.py tests/test_evaluation_diff.py
git commit -m "feat: A/B evaluation diff"
```

---

### Task 10: CLI — `python -m agent.evaluation`

**Files:**
- Create: `agent/evaluation/__main__.py`
- Create: `tests/test_evaluation_cli.py`

**Interfaces:**
- Consumes: `load_config`, `load_domain_config`, `get_api_key`, `ConfigError` from `agent.config`; `LLMClient` from `agent.llm`; `.dataset.load_dataset`; `.runner.run_evaluation`; `.metrics.compute_metrics`; `.report.{serialize_results, write_result, format_summary}`; `.diff.{diff_runs, load_result}`.
- Produces:
  - `main(argv: list[str] | None = None) -> int`.
  - Subcommand `run`: flags `--dataset` (default `evaluation/datasets/{basename(domain_dir)}.yaml`), `--label` (default `"run"`), `--skip-quality`, `--config` (default `None` → `load_config()` default resolution), `--results-dir` (precedence: flag > `config.evaluation.results_dir` > `evaluation/results`). Prints the terminal summary and the written result path. Returns 0.
  - Subcommand `diff <run-a.json> <run-b.json>`: loads both, prints `diff_runs`. Returns 0.
  - On `ConfigError` / `DatasetError` / `FileNotFoundError` / `ValueError`: print to stderr, return 1.

- [ ] **Step 1: Write the failing tests** — create `tests/test_evaluation_cli.py`:

```python
import pytest

from agent.evaluation import __main__ as eval_main


def test_main_run_prints_summary_and_writes_file(tmp_path, monkeypatch):
    domain_dir = tmp_path / "software_engineering"
    domain_dir.mkdir()
    (domain_dir / "domain.json").write_text(
        '{"name": "sw", "description": "d", "out_of_domain_reply": "Out."}',
        encoding="utf-8",
    )
    (domain_dir / "intents.yaml").write_text("- id: faq\n  description: quick\n", encoding="utf-8")
    (domain_dir / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (domain_dir / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (domain_dir / "prompts").mkdir()
    (domain_dir / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (domain_dir / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")

    dataset_dir = tmp_path / "evaluation" / "datasets"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "software_engineering.yaml").write_text(
        'domain: software_engineering\n'
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    category: knowledge\n'
        '    answer_quality: false\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        f'{{"base_url": "https://x", "model": "m", "domain_dir": "{domain_dir}"}}',
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._usage_local = __import__("threading").local()

        def chat_completion(self, messages, model=None, temperature=0.3,
                            disable_thinking=False, json_mode=False, json_schema=None):
            self._usage_local.usage = None
            return '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}'

        def chat_completion_stream(self, messages, **kwargs):
            return iter([])

    monkeypatch.setattr(eval_main, "LLMClient", FakeClient)
    monkeypatch.setenv("AGENT_API_KEY", "k")

    import io
    import sys

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main([
        "run",
        "--config", str(config_path),
        "--dataset", str(dataset_dir / "software_engineering.yaml"),
        "--label", "x",
        "--results-dir", str(results_dir),
        "--skip-quality",
    ])
    assert rc == 0
    text = out.getvalue()
    assert "domain_accuracy" in text
    assert "x.json" in text


def test_main_diff(tmp_path, monkeypatch):
    import json

    results_dir = tmp_path / "r"
    results_dir.mkdir()
    a = results_dir / "a.json"
    b = results_dir / "b.json"
    record = {
        "label": "x", "model": "m",
        "metrics": {
            "classification": {"domain_accuracy": 1.0, "intent_accuracy": 1.0,
                               "complexity_accuracy": 1.0, "per_intent": {}},
            "routing": {"strategy_accuracy": 1.0, "orchestration_accuracy": 1.0,
                        "model_routing_accuracy": 1.0},
            "answer_quality": {"correctness": 4.0},
            "cost": {"llm_calls": 2, "in_tokens": 10, "out_tokens": 5,
                     "total_tokens": 15, "cache_tokens": 0, "latency_ms": 20.0,
                     "by_path": {}},
        },
    }
    a.write_text(json.dumps(record), encoding="utf-8")
    record["metrics"]["classification"]["domain_accuracy"] = 0.5
    b.write_text(json.dumps(record), encoding="utf-8")

    import io
    import sys

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main(["diff", str(a), str(b)])
    assert rc == 0
    assert "domain_accuracy" in out.getvalue()


def test_main_missing_config_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    rc = eval_main.main(["run", "--config", "/nonexistent/config.json"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Config error" in err


def test_main_bad_dataset_returns_1(tmp_path, monkeypatch, capsys):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        f'{{"base_url": "https://x", "model": "m", "domain_dir": "{tmp_path}"}}',
        encoding="utf-8",
    )
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    rc = eval_main.main(["run", "--config", str(config_path),
                         "--dataset", "/nonexistent/dataset.yaml"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Dataset error" in err
```

Note: the CLI resolves the default dataset path from the config's `domain_dir` basename — with `--dataset` given explicitly, that logic is skipped.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_cli.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'agent.evaluation.__main__'`.

- [ ] **Step 3: Implement** — create `agent/evaluation/__main__.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.config import ConfigError, get_api_key, load_config, load_domain_config
from agent.llm import LLMClient

from .dataset import DatasetError, load_dataset
from .diff import diff_runs, load_result
from .metrics import compute_metrics
from .report import format_summary, serialize_results, write_result
from .runner import run_evaluation


def _default_dataset(domain_dir: str) -> str:
    return f"evaluation/datasets/{Path(domain_dir).name}.yaml"


def _cmd_run(args) -> int:
    try:
        config = load_config(args.config)
        domain = load_domain_config(config.domain_dir)
        api_key = get_api_key()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1
    dataset_path = args.dataset or _default_dataset(config.domain_dir)
    try:
        dataset = load_dataset(dataset_path)
    except DatasetError as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1
    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model)
    results = run_evaluation(config, domain, dataset, client, skip_quality=args.skip_quality)
    metrics = compute_metrics(dataset, results)
    judge_model = (config.evaluation.judge_model if config.evaluation else None) or config.model
    record = serialize_results(
        results, metrics,
        domain=dataset.domain, label=args.label, model=config.model,
        judge_model=judge_model, skip_quality=args.skip_quality,
        dataset_path=dataset_path,
    )
    results_dir = args.results_dir
    if results_dir is None:
        results_dir = "evaluation/results"
        eval_cfg = getattr(config, "evaluation", None)
        if eval_cfg is not None:
            results_dir = eval_cfg.results_dir
    path = write_result(results_dir, record, label=args.label)
    print(format_summary(record))
    print(f"Result written to: {path}")
    return 0


def _cmd_diff(args) -> int:
    try:
        a = load_result(args.run_a)
        b = load_result(args.run_b)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Diff error: {e}", file=sys.stderr)
        return 1
    print(diff_runs(a, b))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the golden dataset")
    run_p.add_argument("--dataset", default=None, help="path to dataset YAML")
    run_p.add_argument("--label", default="run", help="run label for the result file")
    run_p.add_argument("--skip-quality", action="store_true",
                       help="classification/routing/cost only, no answer generation")
    run_p.add_argument("--config", default=None, help="path to agent config.json")
    run_p.add_argument("--results-dir", default=None,
                       help="directory for result JSONs (default: config evaluation.results_dir, "
                            "else evaluation/results)")
    run_p.set_defaults(func=_cmd_run)

    diff_p = sub.add_parser("diff", help="compare two run results")
    diff_p.add_argument("run_a", help="first result JSON")
    diff_p.add_argument("run_b", help="second result JSON")
    diff_p.set_defaults(func=_cmd_diff)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation_cli.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Manual smoke of CLI**

Run: `env -u AGENT_API_KEY uv run python -m agent.evaluation run --label nosuch --skip-quality`
Expected: exit code 1 with a `Config error` message (no key). Then run with a key present if you have one (skipped otherwise).

- [ ] **Step 6: Full regression**

Run: `uv run pytest -q`
Expected: all pass (107 + 4 = 111).

- [ ] **Step 7: Commit**

```bash
git add agent/evaluation/__main__.py tests/test_evaluation_cli.py
git commit -m "feat: evaluation CLI (run and diff)"
```

---

### Task 11: Golden dataset for software_engineering

**Files:**
- Create: `evaluation/datasets/software_engineering.yaml`
- Test: `tests/test_evaluation_dataset.py` (one new test that loads the real file)

**Interfaces:**
- Consumes: real intent ids from `domain/software_engineering/intents.yaml`; real strategy ids from `strategies.yaml` and `intent_mapping.yaml`.
- Produces: the committed golden dataset, ~40 cases covering all §4.3 categories + boundary cases, loadable by `load_dataset`.

Strategy map (from the real domain config):
- `faq`, `summarization` → `direct`
- `concept_explain`, `tutorial`, `learning_guide` → `teaching`
- `troubleshooting` → `debugging`
- `comparison`, `performance_analysis`, `architecture_design`, `code_review` → `analysis`
- `generate_code` → `code_snippet`
- All except `direct` have `complexity_gate: true`, so `complex` cases on those strategies orchestrate (`expected.orchestrate: true`).

- [ ] **Step 1: Write the failing test** — append to `tests/test_evaluation_dataset.py`:

```python
def test_load_committed_software_engineering_dataset():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    path = repo / "evaluation" / "datasets" / "software_engineering.yaml"
    ds = load_dataset(str(path))
    assert ds.domain == "software_engineering"
    assert len(ds.cases) >= 40
    categories = {c.category for c in ds.cases}
    assert {"knowledge", "problem_solving", "evaluation", "generation", "boundary"} <= categories
    intents = {c.expected_intent for c in ds.cases}
    assert {"faq", "concept_explain", "tutorial", "learning_guide", "summarization",
            "troubleshooting", "performance_analysis", "comparison", "architecture_design",
            "code_review", "generate_code"} <= intents
    strategies = {c.expected_strategy for c in ds.cases}
    assert {"direct", "teaching", "debugging", "analysis", "code_snippet"} <= strategies
    assert {"simple", "medium", "complex"} <= {c.expected_complexity for c in ds.cases}
    assert any(c.expected_orchestrate for c in ds.cases)
    assert any(c.expected_domain == "other" for c in ds.cases)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_dataset.py::test_load_committed_software_engineering_dataset -v`
Expected: `FAIL` — `DatasetError: Dataset file not found: ...`.

- [ ] **Step 3: Create the dataset** — create `evaluation/datasets/software_engineering.yaml`:

```yaml
domain: software_engineering
cases:
  # ---- Knowledge: FAQ ----
  - id: se-001
    question: "What does HTTP status code 503 mean?"
    category: knowledge
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-002
    question: "What is the default port for MySQL?"
    category: knowledge
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-003
    question: "What is the time complexity of quicksort in the average case?"
    category: knowledge
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-004
    question: "What is a database index?"
    category: knowledge
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-005
    question: "How do you create a git branch?"
    category: knowledge
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-006
    question: "What does the SOLID acronym stand for?"
    category: knowledge
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true

  # ---- Knowledge: Concept Explanation ----
  - id: se-010
    question: "What is dependency injection?"
    category: knowledge
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true
    reference: "Dependency injection is a technique where a component receives its dependencies from outside rather than constructing them itself, improving testability and decoupling."
  - id: se-011
    question: "Explain how a virtual memory page table works together with the TLB."
    category: knowledge
    expected: {domain: software_engineering, intent: concept_explain, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-012
    question: "Why is immutability important in functional programming?"
    category: knowledge
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-013
    question: "Explain the difference between concurrency and parallelism."
    category: knowledge
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-014
    question: "Explain how a relational database query planner works and why an index changes the plan."
    category: knowledge
    expected: {domain: software_engineering, intent: concept_explain, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true

  # ---- Knowledge: Tutorial ----
  - id: se-020
    question: "Teach me step by step how to build a REST API with FastAPI and connect it to PostgreSQL."
    category: knowledge
    expected: {domain: software_engineering, intent: tutorial, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-021
    question: "Give me a step-by-step tutorial on setting up Docker Compose for a Node.js app with a Redis service."
    category: knowledge
    expected: {domain: software_engineering, intent: tutorial, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true

  # ---- Knowledge: Learning Guide ----
  - id: se-030
    question: "Create a structured learning path to go from Python basics to backend web development."
    category: knowledge
    expected: {domain: software_engineering, intent: learning_guide, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-031
    question: "Design a multi-month study plan for becoming a reliable systems engineer."
    category: knowledge
    expected: {domain: software_engineering, intent: learning_guide, complexity: complex, strategy: teaching, orchestrate: true}
    answer_quality: true

  # ---- Knowledge: Summarization ----
  - id: se-040
    question: "Summarize the key ideas of this article into three bullet points: <article>Effective debugging requires reproducing the failure deterministically, isolating the smallest failing input, and forming hypotheses that you can test rather than guessing at fixes.</article>"
    category: knowledge
    expected: {domain: software_engineering, intent: summarization, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: false

  # ---- Problem Solving: Troubleshooting ----
  - id: se-050
    question: "My database connection pool is exhausted under load and new requests hang. How do I diagnose it?"
    category: problem_solving
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: debugging, orchestrate: false}
    answer_quality: true
  - id: se-051
    question: "Why does my C++ program crash with a segmentation fault when accessing index 0 of an empty vector?"
    category: problem_solving
    expected: {domain: software_engineering, intent: troubleshooting, complexity: simple, strategy: debugging, orchestrate: false}
    answer_quality: true
  - id: se-052
    question: "A distributed system fails intermittently with timeout errors across several services. Investigate the root cause and propose a fix."
    category: problem_solving
    expected: {domain: software_engineering, intent: troubleshooting, complexity: complex, strategy: debugging, orchestrate: true}
    answer_quality: true
  - id: se-053
    question: "My Node.js app crashes with an out-of-memory error only in production. How should I debug this?"
    category: problem_solving
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: debugging, orchestrate: false}
    answer_quality: true

  # ---- Problem Solving: Performance Analysis ----
  - id: se-060
    question: "Analyze why my API response time degrades as concurrent users increase, and identify the bottleneck."
    category: problem_solving
    expected: {domain: software_engineering, intent: performance_analysis, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-061
    question: "Analyze end-to-end latency bottlenecks in a system spanning a CDN, application servers, and a data warehouse."
    category: problem_solving
    expected: {domain: software_engineering, intent: performance_analysis, complexity: complex, strategy: analysis, orchestrate: true}
    answer_quality: true
  - id: se-062
    question: "Why is a bulk INSERT of 10 million rows slower than expected, and how can it be tuned?"
    category: problem_solving
    expected: {domain: software_engineering, intent: performance_analysis, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true

  # ---- Problem Solving: Architecture Design ----
  - id: se-070
    question: "Design the module structure for a monorepo with several shared packages and clear dependency boundaries."
    category: problem_solving
    expected: {domain: software_engineering, intent: architecture_design, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-071
    question: "Design a scalable microservices architecture for an e-commerce platform covering orders, payments, and inventory."
    category: problem_solving
    expected: {domain: software_engineering, intent: architecture_design, complexity: complex, strategy: analysis, orchestrate: true}
    answer_quality: true
  - id: se-072
    question: "Design a caching layer for a read-heavy news website, including invalidation strategy."
    category: problem_solving
    expected: {domain: software_engineering, intent: architecture_design, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true

  # ---- Evaluation: Comparison ----
  - id: se-080
    question: "Compare gRPC and REST for inter-service communication in a microservices environment."
    category: evaluation
    expected: {domain: software_engineering, intent: comparison, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-081
    question: "Compare the Go defer statement with C++ RAII for resource management."
    category: evaluation
    expected: {domain: software_engineering, intent: comparison, complexity: simple, strategy: analysis, orchestrate: false}
    answer_quality: true

  # ---- Evaluation: Code Review ----
  - id: se-090
    question: "Review this Python function for correctness and style: <code>def sum(a, b): return a + b  # never used</code>"
    category: evaluation
    expected: {domain: software_engineering, intent: code_review, complexity: simple, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-091
    question: "Perform a security-focused code review of this authentication snippet: <code>if user.password == request.password: login(user)</code>"
    category: evaluation
    expected: {domain: software_engineering, intent: code_review, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true

  # ---- Generation: Code Generation ----
  - id: se-100
    question: "Write a Python function that checks whether a string is a palindrome."
    category: generation
    expected: {domain: software_engineering, intent: generate_code, complexity: simple, strategy: code_snippet, orchestrate: false}
    answer_quality: true
  - id: se-101
    question: "Write a Python function that reads a CSV file, validates each row, and returns a summary of invalid rows."
    category: generation
    expected: {domain: software_engineering, intent: generate_code, complexity: medium, strategy: code_snippet, orchestrate: false}
    answer_quality: true
  - id: se-102
    question: "Build a complete CLI tool in Python with argument parsing, a config file, and unit tests."
    category: generation
    expected: {domain: software_engineering, intent: generate_code, complexity: complex, strategy: code_snippet, orchestrate: true}
    answer_quality: true

  # ---- Boundary: SE / non-SE ----
  - id: se-110
    question: "Recommend a good restaurant in Tokyo."
    category: boundary
    expected: {domain: other, intent: null, complexity: null, strategy: reject, orchestrate: false}
    answer_quality: false
  - id: se-111
    question: "Explain the causes of the French Revolution."
    category: boundary
    expected: {domain: other, intent: null, complexity: null, strategy: reject, orchestrate: false}
    answer_quality: false
  - id: se-112
    question: "What is the capital of Australia?"
    category: boundary
    expected: {domain: other, intent: null, complexity: null, strategy: reject, orchestrate: false}
    answer_quality: false

  # ---- Boundary: FAQ / Concept Explain ----
  - id: se-120
    question: "What is a hash function?"
    category: boundary
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-121
    question: "Why are hash tables fast for key lookups?"
    category: boundary
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true

  # ---- Boundary: Tutorial / Learning Guide ----
  - id: se-130
    question: "Teach me Git branching with hands-on commands."
    category: boundary
    expected: {domain: software_engineering, intent: tutorial, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-131
    question: "Plan out a full curriculum to master Kubernetes from beginner to advanced."
    category: boundary
    expected: {domain: software_engineering, intent: learning_guide, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true

  # ---- Boundary: Troubleshooting / Performance Analysis ----
  - id: se-140
    question: "My service became slow right after deploying the new caching layer. Why might that be?"
    category: boundary
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: debugging, orchestrate: false}
    answer_quality: true
  - id: se-141
    question: "Compare the performance of an in-process cache versus a distributed cache for a read-heavy workload."
    category: boundary
    expected: {domain: software_engineering, intent: performance_analysis, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true

  # ---- Boundary: Comparison / Architecture Design ----
  - id: se-150
    question: "Should we use event-driven or request-response architecture for our notification service?"
    category: boundary
    expected: {domain: software_engineering, intent: architecture_design, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-151
    question: "Compare queues and topics as messaging primitives."
    category: boundary
    expected: {domain: software_engineering, intent: comparison, complexity: simple, strategy: analysis, orchestrate: false}
    answer_quality: true

  # ---- Boundary: Medium / Complex ----
  - id: se-160
    question: "How should we scale our PostgreSQL database to handle 10x the current read volume?"
    category: boundary
    expected: {domain: software_engineering, intent: architecture_design, complexity: complex, strategy: analysis, orchestrate: true}
    answer_quality: true
  - id: se-161
    question: "Explain the full request lifecycle of a React application from URL entry to paint."
    category: boundary
    expected: {domain: software_engineering, intent: concept_explain, complexity: complex, strategy: teaching, orchestrate: true}
    answer_quality: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation_dataset.py::test_load_committed_software_engineering_dataset -v`
Expected: PASS.

- [ ] **Step 5: Count the cases**

Run: `rg -c '^  - id:' evaluation/datasets/software_engineering.yaml`
Expected: 42 (>= 40).

- [ ] **Step 6: Full regression**

Run: `uv run pytest -q`
Expected: all pass (111 + 1 = 112).

- [ ] **Step 7: Commit**

```bash
git add evaluation/datasets/software_engineering.yaml tests/test_evaluation_dataset.py
git commit -m "feat: software_engineering golden dataset"
```

---

### Task 12: README docs + final review

**Files:**
- Modify: `README.md`
- Test: `tests/test_smoke.py` (add one real-API evaluation smoke test)

**Interfaces:**
- Consumes: the CLI from Task 10.
- Produces: documentation for the evaluation subsystem and a live smoke test (skips without `AGENT_API_KEY`).

- [ ] **Step 1: Add README section** — append a section after the Observability section:

```markdown
## Evaluation

Golden-dataset evaluation for classification, routing, answer quality, and cost:

```bash
uv run python -m agent.evaluation run                      # full run (all metrics)
uv run python -m agent.evaluation run --skip-quality       # classification/routing/cost only
uv run python -m agent.evaluation run --label my-run       # named result file
uv run python -m agent.evaluation run --results-dir out/   # override the results dir
```

Datasets live in `evaluation/datasets/` (one YAML file per domain). Each run writes
a timestamped JSON result to `evaluation/results/` (gitignored). Compare two runs:

```bash
uv run python -m agent.evaluation diff evaluation/results/2026-08-14-a.json \
                                   evaluation/results/2026-08-14-b.json
```

Answer-quality judging uses `evaluation.judge_model` from `config.json` (falls back to `model`).
Evaluation is independent of observability: it reads pipeline return values and its
own usage recorder, so disabling observability does not affect evaluation.
```

- [ ] **Step 2: Add a live smoke test** — append to `tests/test_smoke.py`:

```python
def test_smoke_evaluation_writes_result(live_config, tmp_path, monkeypatch):
    """A tiny dataset slice runs end-to-end and produces a result file."""
    import agent.evaluation.__main__ as eval_main

    dataset = tmp_path / "smoke-dataset.yaml"
    dataset.write_text(
        'domain: software_engineering\n'
        'cases:\n'
        '  - id: smoke-1\n'
        '    question: "What is Go defer?"\n'
        '    category: knowledge\n'
        '    answer_quality: false\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    monkeypatch.setenv("AGENT_API_KEY", os.environ["AGENT_API_KEY"])
    rc = eval_main.main(["run", "--config", live_config,
                         "--dataset", str(dataset),
                         "--label", "smoke",
                         "--results-dir", str(results_dir),
                         "--skip-quality"])
    assert rc == 0
    files = list(results_dir.glob("*-smoke.json"))
    assert files, "expected a result file to be written"
    assert json.loads(files[0].read_text(encoding="utf-8"))["metrics"]["n_cases"] == 1
```

- [ ] **Step 3: Run the unit suite**

Run: `uv run pytest -q`
Expected: all pass (112 tests).

- [ ] **Step 4: Run smoke tests (requires AGENT_API_KEY)**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS (if the key is set; otherwise skipped).

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_smoke.py
git commit -m "docs: evaluation usage; test: live evaluation smoke"
```

---

### Task 13: Review the plan against the spec

- [ ] **Step 1: Spec coverage check (written spec `docs/superpowers/specs/2026-08-14-evaluation-framework-design.md`)**
  - §2.1 module layout (`agent/evaluation/*`, `evaluation/datasets/`, `evaluation/results/`) → Tasks 4-10.
  - §2.2 decoupling from observability → Task 6 (`RecordingClient` reads `_usage_local`; nothing reads `TraceStore`).
  - §2.3 `Chat.respond(question, *, route=None)` → Task 3.
  - §3 dataset schema + validation + coverage → Task 4 + Task 11.
  - §4.1 classification metrics + per-intent + out-of-domain rules → Task 7.
  - §4.2 routing metrics (incl. reject/orchestrate false/model excluded) → Task 7.
  - §4.3 answer quality judge (6 dims, separate judge_model, reference) → Task 5 + Task 2.
  - §4.4 cost/latency + by_path → Task 7.
  - §5 runner flow (route reuse, sequential) → Task 6.
  - §6 config (`judge_model`, `evaluation` block) → Task 2.
  - §7 CLI (`run`, `diff`, terminal + JSON) → Tasks 8, 9, 10.
  - §8 cache_tokens in `LLMClient` → Task 1.
  - §9 testing (unit with FakeClient, real-API smoke) → Tasks 4-12.

- [ ] **Step 2: Run the full suite one final time** and report the pass count (expected 112).

- [ ] **Step 3: Verify the example config and README consistency** — `config.example.json` includes the `evaluation` block with `judge_model` and `results_dir`; README documents `run`/`diff` and the independence from observability.
