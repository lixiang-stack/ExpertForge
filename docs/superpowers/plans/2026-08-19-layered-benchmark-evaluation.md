# Layered Benchmark & Layered Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `tier` axis (classification / routing / full_expert) to the evaluation datasets, make `tier` the single execution-depth selector, default runs to the curated smoke set, and drop the obsolete `--suite` / `--max-per-suite` / `--skip-quality` flags.

**Architecture:** The dataset keeps its per-strategy YAML files purely as storage layout; the model layer becomes a flat case pool where each case carries `tier` (and optional `smoke`). Execution depth is derived from tier (`full_expert` ⇒ full pipeline + judge; otherwise router-only). The CLI selects by `--tier` (or defaults to `smoke: true` cases), and metrics/reports group by tier instead of by suite.

**Tech Stack:** Python 3.x, PyYAML, argparse, pytest, ruff.

## Global Constraints

- `tier` values are exactly `("classification", "routing", "full_expert")` (importable as `TIERS` from `agent.evaluation.dataset`).
- `full_expert` ⇒ in-domain only (an out-of-domain case cannot be `full_expert`).
- `smoke: true` is optional; the default run (no `--tier`) selects only `smoke: true` cases.
- `answer_quality` field is removed from datasets and `EvalCase`; execution depth derives solely from `tier`.
- `--suite`, `--max-per-suite`, `--skip-quality` are removed from the CLI.
- `required_points` / `expert_expectations` are reserved optional list-of-str fields on cases; they are parsed and stored but never evaluated in this plan.
- All unit tests pass with `uv run pytest -q` (no API key).
- No new dependencies.

---

### Task 1: Dataset model — `tier` + `smoke` + reserved fields

**Files:**
- Modify: `agent/evaluation/dataset.py`
- Test: `tests/unit/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: `COMPLEXITY_LEVELS` from `agent.config`, `yaml`, `Path`.
- Produces: `TIERS: tuple[str, ...]` and `FULL_EXPERT: str` constants; `EvalCase` dataclass with fields `(id, question, expected_domain, expected_intent, expected_complexity, expected_strategy, expected_orchestrate, tier, smoke, reference, required_points, expert_expectations)`. `load_suites(path) -> list[Suite]` keeps its signature.

- [ ] **Step 1: Update existing dataset tests and add new failing tests**

In `tests/unit/test_evaluation_dataset.py`, replace the `_VALID` fixture:

```python
_VALID = """
domain: software_engineering
cases:
  - id: se-001
    question: "What is dependency injection?"
    tier: classification
    expected:
      domain: software_engineering
      intent: concept_explain
      complexity: simple
      strategy: teaching
      orchestrate: false
    reference: "Dependency injection passes dependencies into a component."
  - id: se-002
    question: "Recommend a restaurant in Tokyo."
    tier: classification
    expected:
      domain: other
      intent: null
      complexity: null
      strategy: reject
      orchestrate: false
"""
```

Update `test_load_suites_valid` to drop the `answer_quality` assertion and assert the new fields:

```python
    assert c.tier == "classification"
    assert c.smoke is False
```

Replace `test_load_suites_answer_quality_defaults_true` with:

```python
def test_load_suites_defaults(tmp_path):
    path = _dataset_path(tmp_path,
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    tier: classification\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n')
    c = load_suites(str(path))[0].cases[0]
    assert c.smoke is False
    assert c.reference is None
    assert c.required_points is None
    assert c.expert_expectations is None
    assert c.expected_orchestrate is False
```

Add new tests:

```python
def test_tier_required(tmp_path):
    with pytest.raises(DatasetError):
        load_suites(_dataset_path(tmp_path,
            'cases:\n'
            '  - id: a\n'
            '    question: "q"\n'
            '    expected:\n'
            '      domain: software_engineering\n'
            '      intent: faq\n'
            '      complexity: simple\n'
            '      strategy: direct\n'))


def test_tier_invalid(tmp_path):
    with pytest.raises(DatasetError):
        load_suites(_dataset_path(tmp_path,
            'cases:\n'
            '  - id: a\n'
            '    question: "q"\n'
            '    tier: bogus\n'
            '    expected:\n'
            '      domain: software_engineering\n'
            '      intent: faq\n'
            '      complexity: simple\n'
            '      strategy: direct\n'))


def test_full_expert_must_be_in_domain(tmp_path):
    with pytest.raises(DatasetError):
        load_suites(_dataset_path(tmp_path,
            'cases:\n'
            '  - id: a\n'
            '    question: "q"\n'
            '    tier: full_expert\n'
            '    expected:\n'
            '      domain: other\n'
            '      intent: null\n'
            '      complexity: null\n'
            '      strategy: reject\n'
            '      orchestrate: false\n'))


def test_smoke_and_reserved_fields_parsed(tmp_path):
    path = _dataset_path(tmp_path,
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    tier: full_expert\n'
        '    smoke: true\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: architecture_design\n'
        '      complexity: complex\n'
        '      strategy: analysis\n'
        '    required_points:\n'
        '      - identify bottleneck\n'
        '    expert_expectations:\n'
        '      - compare alternatives\n')
    c = load_suites(str(path))[0].cases[0]
    assert c.tier == "full_expert"
    assert c.smoke is True
    assert c.required_points == ["identify bottleneck"]
    assert c.expert_expectations == ["compare alternatives"]
```

- [ ] **Step 2: Run the dataset tests to verify they fail**

Run: `uv run pytest tests/unit/test_evaluation_dataset.py -v`
Expected: FAIL — `EvalCase.__init__()` unexpected keyword `tier` / `smoke`, and validation errors not raised.

- [ ] **Step 3: Implement the dataset model changes**

In `agent/evaluation/dataset.py`:

```python
OUT_OF_DOMAIN = "other"
REJECT_STRATEGY = "reject"
TIERS = ("classification", "routing", "full_expert")
FULL_EXPERT = "full_expert"
```

Replace the `EvalCase` dataclass:

```python
@dataclass
class EvalCase:
    id: str
    question: str
    expected_domain: str
    expected_intent: str | None
    expected_complexity: str | None
    expected_strategy: str
    expected_orchestrate: bool
    tier: str
    smoke: bool = False
    reference: str | None = None
    required_points: list[str] | None = None
    expert_expectations: list[str] | None = None
```

Add a helper after `_read_yaml`:

```python
def _str_list(value):
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise DatasetError(f"Expected a list of strings, got: {value!r}")
    return value
```

In `_validate_case`, replace the `answer_quality` block (currently lines 91-93) and the `EvalCase(...)` construction (lines 95-105) with:

```python
    tier = raw.get("tier")
    if tier not in TIERS:
        raise DatasetError(f"Case {cid} tier must be one of {TIERS}, got {tier!r}")
    smoke = raw.get("smoke", False)
    if smoke not in (True, False):
        raise DatasetError(f"Case {cid} smoke must be a boolean")
    if tier == FULL_EXPERT and not in_domain:
        raise DatasetError(f"Out-of-domain case {cid} cannot be tier 'full_expert'")
    reference = raw.get("reference")
    return EvalCase(
        id=cid,
        question=question,
        expected_domain=exp_domain,
        expected_intent=intent,
        expected_complexity=complexity,
        expected_strategy=strategy,
        expected_orchestrate=bool(orchestrate),
        tier=tier,
        smoke=bool(smoke),
        reference=reference if isinstance(reference, str) else None,
        required_points=_str_list(raw.get("required_points")),
        expert_expectations=_str_list(raw.get("expert_expectations")),
    )
```

- [ ] **Step 4: Run the dataset tests to verify they pass**

Run: `uv run pytest tests/unit/test_evaluation_dataset.py -v`
Expected: PASS (all dataset tests, including the committed-suite and boundary tests).

- [ ] **Step 5: Commit**

```bash
git add agent/evaluation/dataset.py tests/unit/test_evaluation_dataset.py
git commit -m "feat: add tier and smoke fields to evaluation dataset model"
```

---

### Task 2: Migrate the dataset YAML files to `tier` + smoke markers

**Files:**
- Modify: `evaluation/datasets/software_engineering/classification.yaml`
- Modify: `evaluation/datasets/software_engineering/routing.yaml`
- Modify: `evaluation/datasets/software_engineering/direct.yaml`
- Modify: `evaluation/datasets/software_engineering/teaching.yaml`
- Modify: `evaluation/datasets/software_engineering/debugging.yaml`
- Modify: `evaluation/datasets/software_engineering/analysis.yaml`
- Modify: `evaluation/datasets/software_engineering/code_snippet.yaml`
- Modify: `evaluation/datasets/software_engineering/orchestration.yaml`
- Test: `tests/unit/test_evaluation_dataset.py` (`test_load_committed_software_engineering_suites`)

**Interfaces:**
- Consumes: the `TIERS`/`smoke` schema from Task 1.
- Produces: every case has a valid `tier`; no case has `answer_quality`; exactly 5 cases have `smoke: true` covering all three tiers.

- [ ] **Step 1: Rewrite each dataset file with `tier` (remove `answer_quality`)**

Per-file tier assignment (case id → tier):

| File | Cases |
|---|---|
| `classification.yaml` | se-110, se-120, se-140, se-121, se-122, se-123, se-124, se-125, se-127 → `classification`; se-126, se-128 → `full_expert` |
| `routing.yaml` | se-050, se-060, se-090 → `routing` |
| `direct.yaml` | se-001, se-003, se-040 → `classification` |
| `teaching.yaml` | se-010, se-020, se-030 → `classification` |
| `debugging.yaml` | se-051, se-053, se-054 → `routing` |
| `analysis.yaml` | se-062, se-070, se-081, se-082 → `routing` |
| `code_snippet.yaml` | se-100, se-101, se-103 → `classification` |
| `orchestration.yaml` | se-052, se-071, se-102 → `full_expert` |

Smoke markers (add `smoke: true` to exactly these): se-110, se-120, se-050, se-052, se-071.

Example — `classification.yaml` becomes (truncated; apply the same pattern to every case, dropping the `answer_quality` line and adding `tier`):

```yaml
cases:
  - id: se-110
    tier: classification
    smoke: true
    question: "Recommend a good restaurant in Tokyo."
    expected: {domain: other, intent: null, complexity: null, strategy: reject, orchestrate: false}
  - id: se-120
    tier: classification
    smoke: true
    question: "What is a hash function?"
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
  - id: se-140
    tier: classification
    question: "My service became slow right after deploying the new caching layer. Why might that be?"
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: debugging, orchestrate: false}
  - id: se-121
    tier: classification
    question: "Why does dependency injection reduce coupling?"
    expected: {domain: software_engineering, intent: concept_explain, complexity: medium, strategy: teaching, orchestrate: false}
  - id: se-122
    tier: classification
    question: "Walk me through setting up a React project step by step."
    expected: {domain: software_engineering, intent: tutorial, complexity: medium, strategy: teaching, orchestrate: false}
  - id: se-123
    tier: classification
    question: "Create a month-long learning path to go from zero to competent in Python."
    expected: {domain: software_engineering, intent: learning_guide, complexity: medium, strategy: teaching, orchestrate: false}
  - id: se-124
    tier: classification
    question: "Analyze why my API response time degrades as concurrent users increase, and identify the bottleneck."
    expected: {domain: software_engineering, intent: performance_analysis, complexity: medium, strategy: analysis, orchestrate: false}
  - id: se-125
    tier: classification
    question: "Compare gRPC vs REST for microservices."
    expected: {domain: software_engineering, intent: comparison, complexity: medium, strategy: analysis, orchestrate: false}
  - id: se-126
    tier: full_expert
    question: "Design the architecture of a system that must handle millions of events per second."
    expected: {domain: software_engineering, intent: architecture_design, complexity: complex, strategy: analysis, orchestrate: true}
  - id: se-127
    tier: classification
    question: "Walk me through the 12-factor app principles, one by one, in full detail."
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
  - id: se-128
    tier: full_expert
    question: "Design a distributed rate limiter for millions of QPS with multi-region deployment."
    expected: {domain: software_engineering, intent: architecture_design, complexity: complex, strategy: analysis, orchestrate: true}
```

`orchestration.yaml` becomes:

```yaml
cases:
  - id: se-052
    tier: full_expert
    smoke: true
    question: "A distributed system fails intermittently with timeout errors across several services. Investigate the root cause and propose a fix."
    expected: {domain: software_engineering, intent: troubleshooting, complexity: complex, strategy: debugging, orchestrate: true}
  - id: se-071
    tier: full_expert
    smoke: true
    question: "Design a scalable microservices architecture for an e-commerce platform covering orders, payments, and inventory."
    expected: {domain: software_engineering, intent: architecture_design, complexity: complex, strategy: analysis, orchestrate: true}
  - id: se-102
    tier: full_expert
    question: "Build a complete CLI tool in Python with argument parsing, a config file, and unit tests."
    expected: {domain: software_engineering, intent: generate_code, complexity: complex, strategy: code_snippet, orchestrate: true}
```

Apply the same pattern to the remaining files per the table: every case gains `tier: classification|routing` and loses its `answer_quality` line; `routing.yaml` adds `smoke: true` to se-050 only.

- [ ] **Step 2: Extend the committed-suite test to assert tiers and smoke**

In `tests/unit/test_evaluation_dataset.py`, extend `test_load_committed_software_engineering_suites` with:

```python
    tiers = {c.id: c.tier for s in suites for c in s.cases}
    assert set(tiers.values()) == {"classification", "routing", "full_expert"}
    smoke = [c for s in suites for c in s.cases if c.smoke]
    assert len(smoke) == 5
    assert {c.tier for c in smoke} == {"classification", "routing", "full_expert"}
```

- [ ] **Step 3: Run the dataset tests to verify the migrated data loads**

Run: `uv run pytest tests/unit/test_evaluation_dataset.py -v`
Expected: PASS — all cases load with valid tiers; exactly 5 smoke cases spanning three tiers.

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest -q`
Expected: PASS (existing tests still pass; answer-quality-related assertions were updated in Task 1).

- [ ] **Step 5: Commit**

```bash
git add evaluation/datasets/software_engineering/ tests/unit/test_evaluation_dataset.py
git commit -m "chore: migrate evaluation datasets to tier field and smoke markers"
```

---

### Task 3: Runner — tier-based execution depth

**Files:**
- Modify: `agent/evaluation/runner.py`
- Test: `tests/unit/test_evaluation_runner.py`

**Interfaces:**
- Consumes: `EvalCase.tier` and `FULL_EXPERT` from `agent.evaluation.dataset` (Task 1).
- Produces: `run_evaluation(config, domain, suite, client, judge_client=None) -> list[CaseResult]` — **`skip_quality` parameter removed**; the quality phase runs iff `case.tier == "full_expert"`. `CaseResult` gains `tier: str` (default `""`, set to `case.tier`).

- [ ] **Step 1: Update runner tests**

In `tests/unit/test_evaluation_runner.py`:

`_dataset()` becomes:

```python
def _dataset():
    return Suite(name="direct", domain="software_engineering", cases=[
        EvalCase(
            id="se-001", question="what is defer",
            expected_domain="software_engineering",
            expected_intent="faq", expected_complexity="simple",
            expected_strategy="direct", expected_orchestrate=False,
            tier="full_expert", reference="short",
        ),
        EvalCase(
            id="se-002", question="recommend a restaurant",
            expected_domain="other",
            expected_intent=None, expected_complexity=None,
            expected_strategy="reject", expected_orchestrate=False,
            tier="classification", reference=None,
        ),
    ])
```

In `test_run_evaluation_uses_dedicated_judge_client`, both `EvalCase(...)` constructions gain `tier="full_expert",` (replace the `answer_quality=True,` line) and the `answer_quality=False,` in the second construction becomes `tier="full_expert",`.

Replace `test_run_evaluation_skip_quality_skips_answer` and remove the `skip_quality=True` argument from `test_run_evaluation_rejects_out_of_domain` and `test_run_evaluation_records_suite`:

```python
def test_run_evaluation_rejects_out_of_domain():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ])
    results = run_evaluation(_config(), _domain(), _dataset(), client)
    r1 = results[1]
    assert r1.in_domain is False
    assert r1.strategy == "reject"
    assert r1.answer is None
    assert r1.scorecard is None
    assert r1.llm_calls == 1
    assert r1.actual_model is None  # out-of-domain: no answer call


def test_run_evaluation_classification_tier_skips_answer():
    suite = Suite(name="direct", domain="software_engineering", cases=[
        EvalCase(id="se-001", question="what is defer",
                 expected_domain="software_engineering",
                 expected_intent="faq", expected_complexity="simple",
                 expected_strategy="direct", expected_orchestrate=False,
                 tier="classification", reference="short"),
    ])
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
    ])
    client._record_usage(10, 5, cached=2)
    results = run_evaluation(_config(), _domain(), suite, client)
    r0 = results[0]
    assert r0.answer is None
    assert r0.scorecard is None
    assert r0.llm_calls == 1
```

Update `test_run_evaluation_records_suite` to drop `skip_quality=True` and add a tier assertion, and add a new tier test:

```python
def test_run_evaluation_records_suite_and_tier():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ])
    results = run_evaluation(_config(), _domain(), _dataset(), client)
    assert results[0].suite == "direct"
    assert results[0].tier == "full_expert"
    assert results[1].tier == "classification"
```

- [ ] **Step 2: Run the runner tests to verify they fail**

Run: `uv run pytest tests/unit/test_evaluation_runner.py -v`
Expected: FAIL — `skip_quality` unexpected keyword argument / `EvalCase` missing `tier` / `CaseResult` has no `tier`.

- [ ] **Step 3: Implement the runner changes**

In `agent/evaluation/runner.py`:

Change the import line:

```python
from .dataset import EvalCase, FULL_EXPERT, Suite
```

Add `tier` to `CaseResult` (after `suite: str = ""`):

```python
    suite: str = ""
    tier: str = ""
```

Remove the `skip_quality` parameter from `run_evaluation`:

```python
def run_evaluation(
    config: AgentConfig,
    domain: DomainConfig,
    suite: Suite,
    client: LLMClient,
    judge_client: LLMClient | None = None,
) -> list[CaseResult]:
```

Replace the quality-phase condition and the docstring's step 5:

```python
            if case.tier == FULL_EXPERT:
                resp = chat.respond(case.question, route=route)
```

Set the tier in the `CaseResult(...)` construction (next to `suite=suite.name,`):

```python
            suite=suite.name,
            tier=case.tier,
```

Update the module docstring lines that mention `skip_quality`:

```python
      5. Quality phase (only if ``case.tier == "full_expert"``):
         ``chat.respond`` runs the full pipeline, then the judge scores it.
```

```python
      - The router's classification call is always made; the answer-pipeline
        and judge calls happen only for ``full_expert`` tier cases.
```

- [ ] **Step 4: Run the runner tests to verify they pass**

Run: `uv run pytest tests/unit/test_evaluation_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/evaluation/runner.py tests/unit/test_evaluation_runner.py
git commit -m "feat: drive evaluation execution depth from case tier"
```

---

### Task 4: Metrics — `metrics_by_tier`, per-strategy breakdown, failure detection

**Files:**
- Modify: `agent/evaluation/metrics.py`
- Test: `tests/unit/test_evaluation_metrics.py`

**Interfaces:**
- Consumes: `Suite`, `CaseResult`, `EvalCase` (Task 1/3).
- Produces:
  - `compute_metrics(suite, results)` — `routing` dict now includes `per_strategy` (`dict[str, float | None]`).
  - `compute_metrics_by_tier(cases: list[EvalCase], results: list[CaseResult], *, domain: str) -> dict[str, dict]` — keys `"classification"`, `"routing"`, `"full_expert"` in that order; empty tiers yield zeroed metrics.
  - `case_failures(case: EvalCase, result: CaseResult, domain: str) -> list[str]`
  - `failed_cases(results: list[CaseResult], domain: str) -> list[dict]` — entries `{id, tier, suite, question, reasons}`.

- [ ] **Step 1: Add failing tests**

In `tests/unit/test_evaluation_metrics.py`, update the `_case` helper to accept `tier`:

```python
def _case(cid, domain="software_engineering", intent="faq", complexity="simple",
          strategy="direct", orchestrate=False, tier="classification"):
    return EvalCase(
        id=cid, question=f"q {cid}",
        expected_domain=domain, expected_intent=intent,
        expected_complexity=complexity, expected_strategy=strategy,
        expected_orchestrate=orchestrate, tier=tier, reference=None,
    )
```

Update the import line:

```python
from agent.evaluation.dataset import Suite, EvalCase
from agent.evaluation.metrics import _accuracy, case_failures, compute_metrics, compute_metrics_by_tier, failed_cases
from agent.evaluation.runner import CaseResult
```

Add these tests:

```python
def test_per_strategy_accuracy():
    cases = [_case("a", strategy="direct"), _case("b", strategy="teaching"),
             _case("c", strategy="direct")]
    results = [
        _result(cases[0], strategy="direct"),
        _result(cases[1], strategy="teaching"),
        _result(cases[2], strategy="teaching"),  # wrong
    ]
    m = _m(cases, results)
    ps = m["routing"]["per_strategy"]
    assert ps["direct"] == 1.0
    assert ps["teaching"] == 0.5


def test_metrics_by_tier():
    cases = [_case("a", tier="classification"), _case("b", tier="routing"),
             _case("c", tier="full_expert")]
    results = [_result(cases[0]), _result(cases[1]), _result(cases[2])]
    by_tier = compute_metrics_by_tier(cases, results, domain="software_engineering")
    assert list(by_tier) == ["classification", "routing", "full_expert"]
    assert by_tier["classification"]["n_cases"] == 1
    assert by_tier["routing"]["n_cases"] == 1
    assert by_tier["full_expert"]["n_cases"] == 1


def test_metrics_by_tier_empty_tier_zeroed():
    cases = [_case("a", tier="classification")]
    results = [_result(cases[0])]
    by_tier = compute_metrics_by_tier(cases, results, domain="software_engineering")
    assert by_tier["full_expert"]["n_cases"] == 0
    assert by_tier["full_expert"]["classification"]["domain_accuracy"] is None


def test_case_failures():
    case = _case("a", tier="classification")
    assert case_failures(case, _result(case), "software_engineering") == []
    bad = _result(case, intent="concept_explain", strategy="teaching")
    reasons = case_failures(case, bad, "software_engineering")
    assert any("intent mismatch" in r for r in reasons)
    assert any("strategy mismatch" in r for r in reasons)


def test_failed_cases_lists_only_failures():
    cases = [_case("a", tier="classification"), _case("b", tier="routing")]
    results = [_result(cases[0]), _result(cases[1], intent="concept_explain")]
    failed = failed_cases(results, "software_engineering")
    assert len(failed) == 1
    assert failed[0]["id"] == "b"
    assert failed[0]["tier"] == "routing"
    assert any("intent mismatch" in r for r in failed[0]["reasons"])
```

- [ ] **Step 2: Run the metrics tests to verify they fail**

Run: `uv run pytest tests/unit/test_evaluation_metrics.py -v`
Expected: FAIL — `ImportError` / `KeyError: 'per_strategy'` / `compute_metrics_by_tier` not defined.

- [ ] **Step 3: Implement the metrics changes**

In `agent/evaluation/metrics.py`, change the import line:

```python
from agent.evaluation.dataset import Suite, TIERS, is_in_domain
```

Add `per_strategy` tracking inside `compute_metrics`. Insert near the other counters:

```python
    per_strategy: dict[str, list[bool]] = {}
    per_strategy_order: list[str] = []
```

Replace the strategy-accuracy lines in the loop:

```python
        if r.strategy == c.expected_strategy:
            strategy_correct += 1
            per_strategy.setdefault(c.expected_strategy, []).append(True)
        else:
            per_strategy.setdefault(c.expected_strategy, []).append(False)
        if c.expected_strategy not in per_strategy_order:
            per_strategy_order.append(c.expected_strategy)
```

After the loop, next to the other accuracy dicts:

```python
    per_strategy_accuracy = {}
    for sid in per_strategy_order:
        marks = per_strategy[sid]
        per_strategy_accuracy[sid] = _accuracy(sum(marks), len(marks))
```

Update the routing block in the return:

```python
        "routing": {
            "strategy_accuracy": _accuracy(strategy_correct, n),
            "per_strategy": per_strategy_accuracy,
            "orchestration_accuracy": _accuracy(orchestration_correct, n),
            "model_routing_accuracy": _accuracy(model_correct, model_total),
        },
```

Append the new functions at the end of `metrics.py`:

```python
def compute_metrics_by_tier(cases, results, *, domain: str) -> dict[str, dict]:
    by_tier: dict[str, dict] = {}
    for tier in TIERS:
        tier_cases = [c for c in cases if c.tier == tier]
        tier_ids = {c.id for c in tier_cases}
        tier_results = [r for r in results if r.case.id in tier_ids]
        by_tier[tier] = compute_metrics(
            Suite(name=tier, domain=domain, cases=tier_cases), tier_results
        )
    return by_tier


def case_failures(case, result, domain: str) -> list[str]:
    reasons = []
    if result.error:
        reasons.append(f"error: {result.error}")
    expected_in = case.expected_domain == domain
    if result.in_domain != expected_in:
        reasons.append("domain mismatch")
    if expected_in:
        if result.intent != case.expected_intent:
            reasons.append(f"intent mismatch (expected {case.expected_intent}, got {result.intent})")
        if result.complexity != case.expected_complexity:
            reasons.append("complexity mismatch")
    if result.strategy != case.expected_strategy:
        reasons.append("strategy mismatch")
    if result.orchestrate != case.expected_orchestrate:
        reasons.append("orchestration mismatch")
    if (result.actual_model is not None and result.expected_model is not None
            and result.actual_model != result.expected_model):
        reasons.append("model routing mismatch")
    if result.scorecard is not None and any(v < 3 for v in result.scorecard.values()):
        reasons.append("judge score below threshold (<3)")
    return reasons


def failed_cases(results, domain: str) -> list[dict]:
    out = []
    for r in results:
        reasons = case_failures(r.case, r, domain)
        if reasons:
            out.append({
                "id": r.case.id,
                "tier": r.case.tier,
                "suite": r.suite,
                "question": r.case.question,
                "reasons": reasons,
            })
    return out
```

- [ ] **Step 4: Run the metrics tests to verify they pass**

Run: `uv run pytest tests/unit/test_evaluation_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/evaluation/metrics.py tests/unit/test_evaluation_metrics.py
git commit -m "feat: tier-grouped metrics, per-strategy breakdown, and failure detection"
```

---

### Task 5: Report — new record schema, per-tier summary, failed cases

**Files:**
- Modify: `agent/evaluation/report.py`
- Test: `tests/unit/test_evaluation_report.py`

**Interfaces:**
- Consumes: metrics + failure helpers from Task 4.
- Produces: `serialize_results(cases, metrics, metrics_by_tier, *, domain, label, model, judge_model, tiers, smoke_only, dataset_path, failed_cases) -> dict`. Record keys: `domain, label, model, judge_model, smoke_only, dataset, tiers, metrics, metrics_by_tier, failed_cases, cases`. `_case_record` includes `tier`. `format_summary(record)` prints a Per-tier section and a Failed-cases section. `write_result` / `slim_record` / `write_baseline` unchanged.

- [ ] **Step 1: Update report tests**

In `tests/unit/test_evaluation_report.py`, update `_case` and `_record`:

```python
def _case(cid):
    return EvalCase(
        id=cid, question=f"q {cid}",
        expected_domain="software_engineering", expected_intent="faq",
        expected_complexity="simple", expected_strategy="direct",
        expected_orchestrate=False, tier="classification", reference=None,
    )
```

Update `_result` to set `tier` (it currently omits it, leaving the `""` default):

```python
def _result(case):
    return CaseResult(
        case=case, in_domain=True, intent="faq", complexity="simple",
        strategy="direct", orchestrate=False, answer="the answer",
        actual_model="low-a", expected_model="low-a",
        scorecard={"correctness": 4, "relevance": 5, "completeness": 4,
                   "technical_depth": 4, "practical_usefulness": 5, "hallucination": 5},
        suite="direct", tier="classification", llm_calls=2, in_tokens=10,
        out_tokens=5, total_tokens=15, cache_tokens=1, latency_ms=10.0,
    )
```

```python
def _record():
    cases = [_case("a")]
    results = [_result(cases[0])]
    suite = Suite(name="direct", domain="software_engineering", cases=cases)
    m = compute_metrics(suite, results)
    by_tier = compute_metrics_by_tier(cases, results, domain="software_engineering")
    return serialize_results(
        results, m, by_tier, domain="software_engineering", label="run1",
        model="m", judge_model="judge-a", tiers=["classification"], smoke_only=True,
        dataset_path="evaluation/datasets/software_engineering",
        failed_cases=[],
    )
```

Update the imports line:

```python
from agent.evaluation.metrics import compute_metrics, compute_metrics_by_tier
```

Update `test_serialize_results_contains_expected_keys` — replace the `skip_quality` and `metrics` assertions:

```python
    assert rec["smoke_only"] is True
    assert rec["tiers"] == ["classification"]
    assert rec["metrics"]["n_cases"] == 1
```

Replace `test_serialize_results_has_suites_and_metrics_by_suite`:

```python
def test_serialize_results_has_tiers_and_metrics_by_tier():
    rec = _record()
    assert rec["tiers"] == ["classification"]
    assert rec["metrics_by_tier"]["classification"]["n_cases"] == 1
    assert rec["failed_cases"] == []
    case = rec["cases"][0]
    assert case["suite"] == "direct"
    assert case["tier"] == "classification"
```

Replace `test_format_summary_has_per_suite_section`:

```python
def test_format_summary_has_per_tier_section():
    text = format_summary(_record())
    assert "Per-tier" in text
    assert "classification" in text
```

Update `test_case_record_includes_error` and `test_case_record_error_none_by_default` to the new `serialize_results` signature:

```python
def test_case_record_includes_error():
    case = _case("a")
    r = _result(case)
    r.error = "LLMError: boom"
    record = serialize_results(
        [r], {}, {}, domain="software_engineering", label="run1",
        model="m", judge_model="judge-a", tiers=[], smoke_only=True,
        dataset_path="evaluation/datasets/software_engineering", failed_cases=[],
    )
    assert record["cases"][0]["error"] == "LLMError: boom"
```

Update `test_format_summary_shows_failed_cases`:

```python
def test_format_summary_shows_failed_cases():
    case = _case("a")
    r = _result(case)
    r.error = "LLMError: boom"
    suite = Suite(name="direct", domain="software_engineering", cases=[case])
    m = compute_metrics(suite, [r])
    by_tier = compute_metrics_by_tier([case], [r], domain="software_engineering")
    failed = failed_cases([r], "software_engineering")
    record = serialize_results(
        [r], m, by_tier, domain="software_engineering", label="run1",
        model="m", judge_model="judge-a", tiers=["classification"], smoke_only=True,
        dataset_path="evaluation/datasets/software_engineering", failed_cases=failed,
    )
    text = format_summary(record)
    assert "Failed cases: 1" in text
    assert "LLMError: boom" in text
```

Add the `failed_cases` import:

```python
from agent.evaluation.metrics import compute_metrics, compute_metrics_by_tier, failed_cases
```

- [ ] **Step 2: Run the report tests to verify they fail**

Run: `uv run pytest tests/unit/test_evaluation_report.py -v`
Expected: FAIL — `serialize_results` unexpected keyword arguments / missing keys / `Per-tier` missing.

- [ ] **Step 3: Implement the report changes**

In `agent/evaluation/report.py`, update `_case_record` to add `tier` after `suite`:

```python
        "suite": r.suite,
        "tier": r.tier,
```

Replace `serialize_results`:

```python
def serialize_results(
    cases,
    metrics,
    metrics_by_tier,
    *,
    domain: str,
    label: str,
    model: str,
    judge_model: str | None,
    tiers: list[str],
    smoke_only: bool,
    dataset_path: str,
    failed_cases: list[dict],
) -> dict:
    return {
        "domain": domain,
        "label": label,
        "model": model,
        "judge_model": judge_model,
        "smoke_only": smoke_only,
        "dataset": dataset_path,
        "tiers": tiers,
        "metrics": metrics,
        "metrics_by_tier": metrics_by_tier,
        "failed_cases": failed_cases,
        "cases": [_case_record(r) for r in cases],
    }
```

Replace `format_summary` (keep `_fmt_accuracy` and `_fmt_cost` as-is):

```python
def format_summary(record: dict) -> str:
    m = record["metrics"]
    cls = m["classification"]
    routing = m["routing"]
    aq = m["answer_quality"]
    cost = m["cost"]
    if record.get("smoke_only"):
        selection = "smoke"
    else:
        selection = "tiers: " + ",".join(record.get("tiers", []))
    lines = [
        f"Evaluation run: {record['label']}  (domain={record['domain']}, "
        f"cases={m['n_cases']}, model={record['model']}, "
        f"judge_model={record['judge_model'] or record['model']}, selection={selection})",
    ]
    if m.get("n_failed"):
        lines.append(f"Failed cases: {m['n_failed']}")
    lines += [
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
    if cls.get("per_complexity") or {}:
        lines.append("  per_complexity:")
        for level, acc in cls["per_complexity"].items():
            lines.append(f"    {level}: {_fmt_accuracy(acc)}")
    lines += [
        "",
        "Routing:",
        f"  strategy_accuracy        {_fmt_accuracy(routing['strategy_accuracy'])}",
    ]
    if routing.get("per_strategy"):
        lines.append("  per_strategy:")
        for sid, acc in routing["per_strategy"].items():
            lines.append(f"    {sid}: {_fmt_accuracy(acc)}")
    lines += [
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
    lines += ["", "Per-tier:"]
    for tname in record.get("tiers", []):
        tm = record["metrics_by_tier"].get(tname, {})
        lines.append(
            f"  {tname}: n={tm.get('n_cases', 0)} "
            f"domain={_fmt_accuracy(tm.get('classification', {}).get('domain_accuracy'))} "
            f"intent={_fmt_accuracy(tm.get('classification', {}).get('intent_accuracy'))} "
            f"strategy={_fmt_accuracy(tm.get('routing', {}).get('strategy_accuracy'))} "
            f"{_fmt_cost(tm.get('cost', {}))}"
        )
    failed = record.get("failed_cases") or []
    if failed:
        lines += ["", "Failed cases:"]
        for fc in failed:
            lines.append(f"  {fc['id']} [{fc['tier']}]: " + "; ".join(fc["reasons"]))
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the report tests to verify they pass**

Run: `uv run pytest tests/unit/test_evaluation_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/evaluation/report.py tests/unit/test_evaluation_report.py
git commit -m "feat: tier-based result schema and report summary with failed cases"
```

---

### Task 6: CLI — `--tier` selector, default smoke, remove obsolete flags

**Files:**
- Modify: `agent/evaluation/__main__.py`
- Test: `tests/unit/test_evaluation_cli.py`, `tests/live/test_smoke.py`

**Interfaces:**
- Consumes: `TIERS` from `dataset`; `compute_metrics_by_tier`, `failed_cases` from `metrics`; `run_evaluation` (no `skip_quality`) from `runner`; new `serialize_results` signature from `report`.
- Produces: `run` command with `--tier {classification,routing,full_expert,all}` (repeatable via `nargs="+"`), defaulting to `smoke: true` cases; no `--suite` / `--max-per-suite` / `--skip-quality`.

- [ ] **Step 1: Update CLI tests**

In `tests/unit/test_evaluation_cli.py`, update `_suite_cli_env` YAML to add `tier` (and remove `answer_quality`):

```python
    (suite_dir / "direct.yaml").write_text(
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    tier: classification\n'
        '    smoke: true\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n'
        '  - id: a2\n'
        '    question: "q2"\n'
        '    tier: routing\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )
    (suite_dir / "teaching.yaml").write_text(
        'cases:\n'
        '  - id: b\n'
        '    question: "q"\n'
        '    tier: classification\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )
```

In `test_main_run_prints_summary_and_writes_file`, the `direct.yaml` case gains `tier: classification` and `smoke: true` (drop `answer_quality: false`), and remove `--skip-quality` from the argv list.

Replace `test_main_run_suite_selection`, `test_main_run_max_per_suite`, and `test_main_max_per_suite_lt_1_returns_1` with:

```python
def test_main_run_default_is_smoke(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    out = _run_with_fake(monkeypatch, [
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--label", "smoke", "--results-dir", str(tmp_path / "r"),
    ])
    assert "cases=1" in out  # only the smoke case runs
    assert "selection=smoke" in out
    assert "Per-tier" in out
    assert "classification: n=1" in out


def test_main_run_tier_selection(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    out = _run_with_fake(monkeypatch, [
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--tier", "classification", "--label", "cls", "--results-dir", str(tmp_path / "r"),
    ])
    assert "cases=2" in out  # a + b
    assert "selection=tiers: classification" in out


def test_main_run_tier_all(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    out = _run_with_fake(monkeypatch, [
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--tier", "all", "--label", "all", "--results-dir", str(tmp_path / "r"),
    ])
    assert "cases=3" in out


def test_main_run_no_matching_tier_returns_1(tmp_path, monkeypatch, capsys):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    rc = eval_main.main([
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--tier", "full_expert", "--results-dir", str(tmp_path / "r"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "No cases match the selection" in err
```

Remove `--skip-quality` from the argv lists in `test_main_passes_provider_and_capability_overrides_from_config`, `test_main_judge_client_gets_capabilities_from_judge_config`, `test_main_judge_client_capabilities_fall_back_to_top_level`, and `test_main_judge_client_missing_key_returns_1`.

Update `_result_record` to the new schema:

```python
def _result_record(domain_accuracy):
    return {
        "domain": "software_engineering",
        "label": "run",
        "model": "m",
        "judge_model": None,
        "smoke_only": False,
        "dataset": "evaluation/datasets/software_engineering",
        "tiers": ["classification", "routing"],
        "metrics": {
            "n_cases": 2,
            "classification": {"domain_accuracy": domain_accuracy, "intent_accuracy": 1.0,
                               "complexity_accuracy": 1.0, "per_intent": {}},
            "routing": {"strategy_accuracy": 1.0, "per_strategy": {},
                        "orchestration_accuracy": 1.0, "model_routing_accuracy": 1.0},
            "answer_quality": {},
            "cost": {"llm_calls": 2, "in_tokens": 10, "out_tokens": 5,
                     "total_tokens": 15, "cache_tokens": 0, "latency_ms": 20.0,
                     "by_path": {}},
        },
        "metrics_by_tier": {
            "classification": {"n_cases": 1, "classification": {"domain_accuracy": domain_accuracy},
                               "routing": {}, "cost": {}},
            "routing": {"n_cases": 1, "classification": {"domain_accuracy": domain_accuracy},
                        "routing": {}, "cost": {}},
            "full_expert": {"n_cases": 0, "classification": {}, "routing": {}, "cost": {}},
        },
        "failed_cases": [],
        "cases": [{"id": "a", "suite": "direct", "tier": "classification"},
                  {"id": "b", "suite": "teaching", "tier": "routing"}],
    }
```

Update `test_main_baseline_writes_slim_file` assertion:

```python
    assert baseline["metrics_by_tier"]["classification"]["n_cases"] == 1
```

In `tests/live/test_smoke.py`, update the dataset YAML and drop `--skip-quality`:

```python
    dataset.write_text(
        'cases:\n'
        '  - id: smoke-1\n'
        '    question: "What is Go defer?"\n'
        '    tier: classification\n'
        '    smoke: true\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )
```

```python
    rc = eval_main.main(["run", "--config", live_config,
                         "--dataset", str(dataset_dir),
                         "--label", "smoke",
                         "--results-dir", str(results_dir)])
```

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run: `uv run pytest tests/unit/test_evaluation_cli.py -v`
Expected: FAIL — `--suite`/`--skip-quality` unknown arguments; summary missing `selection=` / `Per-tier`; wrong `cases=` counts.

- [ ] **Step 3: Implement the CLI changes**

In `agent/evaluation/__main__.py`, update imports:

```python
from .dataset import DatasetError, Suite, TIERS, load_suites
from .metrics import compute_metrics, compute_metrics_by_tier, failed_cases
```

Update the module docstring to remove the old flags and document `--tier`:

```python
"""Evaluation CLI.

Commands:
  run          run the benchmark
    --tier T [T ...]  tiers to run: classification, routing, full_expert, or all
                      (default: the curated smoke: true cases)
    --label NAME       run label for the result file
    --results-dir DIR  override the results dir (default: config evaluation.results_dir)
    --config PATH      path to agent config.json
  diff A B     compare two run results (paths printed by each run)
  baseline RUN  record a metrics-only baseline from a run result; prints delta vs existing

Example:
  uv run python -m agent.evaluation run
  uv run python -m agent.evaluation run --tier classification
  uv run python -m agent.evaluation run --tier classification routing
  uv run python -m agent.evaluation run --tier full_expert
  uv run python -m agent.evaluation run --tier all
  uv run python -m agent.evaluation diff evaluation/results/a.json evaluation/results/b.json
  uv run python -m agent.evaluation baseline evaluation/results/2026-08-15-a.json
"""
```

Rewrite `_cmd_run`:

```python
def _cmd_run(args) -> int:
    try:
        config = load_config(args.config)
        domain = load_domain_config(config.domain_dir)
        api_key = get_api_key()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1
    if config.logging is not None:
        setup_logging(config.logging)
    logger = get_logger("evaluation")
    dataset_path = args.dataset or _default_dataset(config.domain_dir)
    try:
        suites = load_suites(dataset_path)
    except DatasetError as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1
    if not suites:
        print("No dataset suites found", file=sys.stderr)
        return 1
    all_cases = [c for s in suites for c in s.cases]
    if args.tier:
        if "all" in args.tier:
            tiers = list(TIERS)
        else:
            tiers = list(dict.fromkeys(args.tier))
        selected = [c for c in all_cases if c.tier in tiers]
        smoke_only = False
    else:
        selected = [c for c in all_cases if c.smoke]
        tiers = sorted({c.tier for c in selected}, key=TIERS.index)
        smoke_only = True
    if not selected:
        print("No cases match the selection", file=sys.stderr)
        return 1
    pool = Suite(name="pool", domain=suites[0].domain, cases=selected)
    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model,
                       timeout=config.timeout,
                       provider=config.provider,
                       capability_overrides=config.provider_capabilities)
    judge_client = None
    if config.evaluation is not None and config.evaluation.judge is not None:
        try:
            judge_caps = config.evaluation.judge.provider_capabilities
            judge_capability_overrides = (
                {k: getattr(judge_caps, k) for k in KNOWN_CAPABILITY_KEYS}
                if judge_caps is not None
                else config.provider_capabilities
            )
            judge_client = LLMClient(
                base_url=config.evaluation.judge.base_url,
                api_key=get_judge_api_key(),
                model=config.evaluation.judge.model,
                timeout=config.evaluation.judge.timeout,
                provider=config.evaluation.judge.provider,
                capability_overrides=judge_capability_overrides,
            )
        except ConfigError as e:
            print(f"Config error: {e}", file=sys.stderr)
            return 1
    logger.info("eval run start", domain=domain.name, tiers=tiers, smoke_only=smoke_only,
                cases=len(selected))
    results = run_evaluation(config, domain, pool, client, judge_client=judge_client)
    logger.info("eval run end", domain=domain.name, cases=len(results))
    metrics = compute_metrics(pool, results)
    metrics_by_tier = compute_metrics_by_tier(pool.cases, results, domain=pool.domain)
    failed = failed_cases(results, pool.domain)
    judge_name = resolve_judge_model(config)
    record = serialize_results(
        results, metrics, metrics_by_tier,
        domain=pool.domain, label=args.label, model=config.model,
        judge_model=judge_name, tiers=tiers, smoke_only=smoke_only,
        dataset_path=dataset_path, failed_cases=failed,
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
```

In `main`, replace the three removed flags with the `--tier` argument:

```python
    run_p.add_argument("--tier", nargs="+",
                       choices=("classification", "routing", "full_expert", "all"),
                       default=None,
                       help="tiers to run; 'all' means all tiers (default: smoke cases)")
```

- [ ] **Step 4: Run the CLI tests to verify they pass**

Run: `uv run pytest tests/unit/test_evaluation_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest -q`
Expected: PASS (report, metrics, runner, dataset tests all green after the earlier tasks).

- [ ] **Step 6: Commit**

```bash
git add agent/evaluation/__main__.py tests/unit/test_evaluation_cli.py tests/live/test_smoke.py
git commit -m "feat: tier-based CLI selection with default smoke run"
```

---

### Task 7: README and final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the CLI surface from Task 6.

- [ ] **Step 1: Update the README evaluation commands**

In `README.md`, replace the "run" examples block (currently lines ~54-56 and ~92-102) with the tier-based commands:

```markdown
Evaluate against the golden dataset (needs `AGENT_API_KEY`):

```bash
uv run python -m agent.evaluation run                        # default: curated smoke cases (~5)
uv run python -m agent.evaluation run --tier classification  # classification tier only (cheap)
uv run python -m agent.evaluation run --tier classification routing   # classification + routing
uv run python -m agent.evaluation run --tier full_expert     # full expert cases (full pipeline + judge)
uv run python -m agent.evaluation run --tier all             # full regression (all tiers)
```
```

And inside the `<details>` block, replace the evaluation-command list:

```markdown
Evaluation runs (all need `AGENT_API_KEY`; a judge block also needs `AGENT_JUDGE_API_KEY`):

```bash
uv run python -m agent.evaluation run                        # smoke: curated ~5 cases (default)
uv run python -m agent.evaluation run --tier classification  # classification tier only (cheap)
uv run python -m agent.evaluation run --tier classification routing  # classification + routing
uv run python -m agent.evaluation run --tier full_expert     # full expert cases (full pipeline + judge)
uv run python -m agent.evaluation run --tier all             # full regression
uv run python -m agent.evaluation run --label my-run         # named result file
uv run python -m agent.evaluation run --results-dir out/     # override the results dir
uv run python -m agent.evaluation run --config path.json     # custom agent config
```
```

Also update the intro paragraph ("Golden-dataset evaluation" bullet) to mention tiers:

```markdown
- **Layered evaluation**: cases carry a `tier` (classification / routing /
  full_expert) controlling execution depth and cost; runs select by `--tier`
  or default to a curated smoke set, with per-tier metrics and failure-driven
  case growth.
```

- [ ] **Step 2: Run the full unit suite and lint**

Run: `uv run pytest -q`
Expected: PASS.

Run: `uv run ruff check agent/ tests/`
Expected: no lint errors (fix any F401/imports the refactors left behind).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update evaluation CLI docs for tier-based selection"
```

---

## Verification

After all tasks: `uv run pytest -q` green; `uv run ruff check agent/ tests/` clean. Then run one real smoke evaluation (`AGENT_API_KEY` + optional `AGENT_JUDGE_API_KEY` set) and confirm `uv run python -m agent.evaluation run` executes only the ~5 smoke cases and prints a Per-tier summary plus the result file path.