# Suite-Based Evaluation Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single flat dataset YAML with a directory of independent suites, and support running all cases per suite or only 1 case per suite.

**Architecture:** Remove the `Dataset` dataclass; introduce a single `Suite` dataclass (`name`, `domain`, `cases`). `load_suites(path)` returns all suites from a dataset directory (domain derived from the directory name). Each suite runs independently through the existing `run_evaluation`, results are merged into one result file with per-suite and global metrics. CLI gains `--suite` and `--max-per-suite` flags.

**Tech Stack:** Python 3.10+, argparse, PyYAML, pytest, uv.

## Global Constraints

- `EvalCase.category` field and the `CATEGORIES` constant are **removed** (per spec §3).
- The `Dataset` dataclass and `load_dataset()` are **removed**; only `Suite` remains (per spec §3, §4).
- `Suite.domain` is **derived from the dataset directory name** (per spec §2). Suite YAML files contain only `cases` — no top-level `domain`.
- `is_in_domain(case, suite)` compares `case.expected_domain == suite.domain`.
- Each dataset case appears in **at most one suite** — no cross-file sharing (per spec §2).
- `--max-per-suite N` selects the **first N cases** in suite order (per spec §6).
- Test command: `uv run pytest <test-file> -q`. All paths relative to the worktree root.

---

### Task 1: Add `Suite` and `load_suites` (dataset.py)

**Files:**
- Modify: `agent/evaluation/dataset.py`
- Test: `tests/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: existing `_read_yaml`, `_validate_case`, `EvalCase`, `DatasetError`.
- Produces (used by later tasks):
  - `@dataclass class Suite: name: str; domain: str; cases: list[EvalCase]`
  - `def load_suites(path: str) -> list[Suite]`
  - `def is_in_domain(case: EvalCase, suite: Suite) -> bool`

This task is **additive**: `Dataset`/`load_dataset`/`category`/`CATEGORIES` are kept temporarily so the full suite stays green. They are removed in Task 4.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluation_dataset.py` (keep the existing imports; add `Suite`, `load_suites` to the import list):

```python
def _suite_dir(tmp_path, name="software_engineering"):
    d = tmp_path / name
    d.mkdir()
    (d / "direct.yaml").write_text(
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
    (d / "teaching.yaml").write_text(
        'cases:\n'
        '  - id: b\n'
        '    question: "q2"\n'
        '    category: knowledge\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: concept_explain\n'
        '      complexity: medium\n'
        '      strategy: teaching\n',
        encoding="utf-8",
    )
    return d


def test_load_suites_directory(tmp_path):
    d = _suite_dir(tmp_path)
    suites = load_suites(str(d))
    assert len(suites) == 2
    assert suites[0].name == "direct"
    assert suites[1].name == "teaching"
    assert suites[0].domain == "software_engineering"
    assert suites[1].domain == "software_engineering"
    assert len(suites[0].cases) == 1
    assert suites[0].cases[0].id == "a"


def test_load_suites_empty_directory(tmp_path):
    d = tmp_path / "software_engineering"
    d.mkdir()
    with pytest.raises(DatasetError):
        load_suites(str(d))


def test_load_suites_directory_missing_cases(tmp_path):
    d = _suite_dir(tmp_path)
    (d / "bad.yaml").write_text("not_a_mapping: true\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_suites(str(d))


def test_load_suites_single_file(tmp_path):
    d = _suite_dir(tmp_path)
    p = d / "direct.yaml"
    suites = load_suites(str(p))
    assert len(suites) == 1
    assert suites[0].name == "direct"
    assert suites[0].domain == "software_engineering"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation_dataset.py -q`
Expected: FAIL with `ImportError: cannot import name 'load_suites'`

- [ ] **Step 3: Implement `Suite` and `load_suites`**

In `agent/evaluation/dataset.py`, add after the `EvalCase` dataclass:

```python
@dataclass
class Suite:
    name: str
    domain: str
    cases: list[EvalCase]
```

Change `is_in_domain` to accept a suite:

```python
def is_in_domain(case: EvalCase, suite: Suite) -> bool:
    return case.expected_domain == suite.domain
```

Add at the end of the module:

```python
def _load_yaml_cases(raw: object, path: str) -> list[object]:
    if not isinstance(raw, dict):
        raise DatasetError(f"Dataset must be a mapping: {path}")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise DatasetError(f"Dataset 'cases' must be a list: {path}")
    return cases_raw


def load_suites(path: str) -> list[Suite]:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.yaml"))
        if not files:
            raise DatasetError(f"Dataset directory has no suite YAML files: {path}")
        domain = p.name
        return [
            Suite(name=f.stem, domain=domain,
                  cases=[_validate_case(c, domain) for c in _load_yaml_cases(_read_yaml(f), str(f))])
            for f in files
        ]
    domain = p.parent.name
    return [
        Suite(name=p.stem, domain=domain,
              cases=[_validate_case(c, domain) for c in _load_yaml_cases(_read_yaml(p), path)])
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_dataset.py -q`
Expected: PASS (all tests, including the new ones)

- [ ] **Step 5: Commit**

```bash
git add agent/evaluation/dataset.py tests/test_evaluation_dataset.py
git commit -m "feat: add Suite and load_suites for suite-based datasets"
```

---

### Task 2: Adopt `Suite` in runner and metrics

**Files:**
- Modify: `agent/evaluation/runner.py`, `agent/evaluation/metrics.py`
- Test: `tests/test_evaluation_runner.py`, `tests/test_evaluation_metrics.py`

**Interfaces:**
- Consumes: `Suite` from `agent.evaluation.dataset`.
- Produces (used by later tasks):
  - `CaseResult` gains `suite: str = ""` field, placed in the defaulted-fields section (after `scorecard`), **not** immediately after `case` — a defaulted field cannot precede the non-defaulted fields. The default keeps existing test helpers that construct `CaseResult` without a suite (metrics/report tests) green; `run_evaluation` always populates it.
  - `run_evaluation(config, domain, suite: Suite, client, *, skip_quality=False) -> list[CaseResult]`
  - `compute_metrics(suite: Suite, results: list[CaseResult]) -> dict`

`category` still exists on `EvalCase` at this point; it is removed in Task 4. The runner/metrics code does not read `category`, so no change needed for it here.

- [ ] **Step 1: Write the failing tests**

In `tests/test_evaluation_runner.py`, replace the `_dataset()` helper with a suite builder and update the import:

```python
from agent.evaluation.dataset import Suite, EvalCase

def _dataset():
    return Suite(name="direct", domain="software_engineering", cases=[...])
```

(the two `EvalCase(...)` literals stay exactly as they are — the `category=` arguments are harmless until Task 4 and can be left in place).

Add a test asserting the suite field propagates:

```python
def test_run_evaluation_records_suite():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ])
    results = run_evaluation(_config(), _domain(), _dataset(), client, skip_quality=True)
    assert results[0].suite == "direct"
    assert results[1].suite == "direct"
```

In `tests/test_evaluation_metrics.py`, change the import and `_m` helper:

```python
from agent.evaluation.dataset import Suite, EvalCase

def _m(cases, results):
    return compute_metrics(Suite(name="direct", domain="software_engineering", cases=cases), results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation_runner.py tests/test_evaluation_metrics.py -q`
Expected: FAIL (TypeError: `run_evaluation()` got an unexpected keyword argument / `CaseResult.__init__()` missing `suite`)

- [ ] **Step 3: Implement**

In `agent/evaluation/runner.py`:
- Change import: `from .dataset import EvalCase, Suite`
- Add `suite: str = ""` to the `CaseResult` dataclass in the defaulted-fields section (see code block below).
- Change signature: `def run_evaluation(config, domain, suite: Suite, client, *, skip_quality=False)`
- Change the loop: `for case in suite.cases:`
- Pass `suite=suite.name` into the `CaseResult(...)` construction.

Add the field to the `CaseResult` dataclass in the defaulted-fields section (after `scorecard`, alongside `llm_calls=0`), **not** immediately after `case` — `suite` must carry a default (`""`), and a defaulted field cannot precede the non-defaulted fields (`in_domain`, `intent`, `complexity`, `strategy`, `orchestrate`, `answer`, `actual_model`, `expected_model`, `scorecard`):

```python
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
    suite: str = ""
    llm_calls: int = 0
    # ... remaining defaulted fields unchanged
```

In `agent/evaluation/metrics.py`:
- Change import: `from agent.evaluation.dataset import Suite, is_in_domain`
- Change signature: `def compute_metrics(suite: Suite, results: list[CaseResult]) -> dict`
- Inside the loop, `expected_in = is_in_domain(c, suite)` (only this call site uses `dataset`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_runner.py tests/test_evaluation_metrics.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

Run: `uv run pytest -q`
Expected: PASS (196 passed, 5 skipped) — `__main__.py` still calls `load_dataset`/`compute_metrics(dataset, ...)`, which still works because `Dataset` is unchanged.

- [ ] **Step 6: Commit**

```bash
git add agent/evaluation/runner.py agent/evaluation/metrics.py tests/test_evaluation_runner.py tests/test_evaluation_metrics.py
git commit -m "feat: adopt Suite in runner and metrics"
```

---

### Task 3: Switch report and CLI to suites with merged results

**Files:**
- Modify: `agent/evaluation/report.py`, `agent/evaluation/__main__.py`
- Test: `tests/test_evaluation_report.py`, `tests/test_evaluation_cli.py`

**Interfaces:**
- Consumes: `load_suites` from `agent.evaluation.dataset`; `run_evaluation(config, domain, suite, client, *, skip_quality)`; `compute_metrics(suite, results)`; `CaseResult.suite`.
- Produces:
  - `serialize_results(cases, metrics, metrics_by_suite, *, domain, label, model, judge_model, skip_quality, dataset_path, suites) -> dict`
  - `format_summary(record) -> str` (global + per-suite sections)
  - `_cmd_run` iterates suites, merges results, writes one JSON.

The result JSON gains `suites: [...]`, `metrics_by_suite: {...}`; each case record gains `"suite"`. The `category` key is still serialized at this point (removed in Task 4). `Dataset`/`load_dataset` are still used by `__main__.py` until Task 4 — this task switches it to `load_suites` now.

- [ ] **Step 1: Write the failing tests**

In `tests/test_evaluation_report.py`, update the `_record()` helper. Replace the `Dataset(...)` construction and `serialize_results(...)` call. Also update the `_result()` helper to pass `suite="direct"` (otherwise the new `case["suite"]` assertion sees `""`):

```python
from agent.evaluation.dataset import Suite, EvalCase

def _result(case):
    return CaseResult(
        case=case, in_domain=True, intent="faq", complexity="simple",
        strategy="direct", orchestrate=False, answer="the answer",
        actual_model="low-a", expected_model="low-a",
        scorecard={"correctness": 4, "relevance": 5, "completeness": 4,
                   "technical_depth": 4, "practical_usefulness": 5, "hallucination": 5},
        suite="direct", llm_calls=2, in_tokens=10, out_tokens=5, total_tokens=15,
        cache_tokens=1, latency_ms=10.0,
    )

def _record():
    cases = [_case("a")]
    results = [_result(cases[0])]
    suite = Suite(name="direct", domain="software_engineering", cases=cases)
    m = compute_metrics(suite, results)
    metrics_by_suite = {"direct": m}
    return serialize_results(
        results, m, metrics_by_suite, domain="software_engineering", label="run1",
        model="m", judge_model="judge-a", skip_quality=False,
        dataset_path="evaluation/datasets/software_engineering",
        suites=["direct"],
    )
```

Add tests for the new keys and per-suite summary:

```python
def test_serialize_results_has_suites_and_metrics_by_suite():
    rec = _record()
    assert rec["suites"] == ["direct"]
    assert rec["metrics_by_suite"]["direct"]["n_cases"] == 1
    case = rec["cases"][0]
    assert case["suite"] == "direct"


def test_format_summary_has_per_suite_section():
    text = format_summary(_record())
    assert "Per-suite" in text
    assert "direct" in text
```

In `tests/test_evaluation_cli.py`, update `test_main_run_prints_summary_and_writes_file` to build a dataset **directory** instead of a single file. Replace the dataset setup block (lines building `dataset_dir / "software_engineering.yaml"`):

```python
    dataset_dir = tmp_path / "evaluation" / "datasets"
    dataset_dir.mkdir(parents=True)
    suite_dir = dataset_dir / "software_engineering"
    suite_dir.mkdir()
    (suite_dir / "direct.yaml").write_text(
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
```

and change the `--dataset` argument to point at the directory:

```python
        "--dataset", str(suite_dir),
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation_report.py tests/test_evaluation_cli.py -q`
Expected: FAIL (KeyError `metrics_by_suite` / TypeError on `serialize_results` / CLI still writes old format)

- [ ] **Step 3: Implement report.py**

- `_case_record`: add `"suite": r.suite,` as the second key (after `"id"`).
- `serialize_results` — change signature to accept `metrics_by_suite` and `suites`:

```python
def serialize_results(
    cases,
    metrics,
    metrics_by_suite,
    *,
    domain: str,
    label: str,
    model: str,
    judge_model: str | None,
    skip_quality: bool,
    dataset_path: str,
    suites: list[str],
) -> dict:
    return {
        "domain": domain,
        "label": label,
        "model": model,
        "judge_model": judge_model,
        "skip_quality": skip_quality,
        "dataset": dataset_path,
        "suites": suites,
        "metrics": metrics,
        "metrics_by_suite": metrics_by_suite,
        "cases": [_case_record(r) for r in cases],
    }
```

- `format_summary`: append a per-suite section after the cost block (before `return`):

```python
    lines += ["", "Per-suite:"]
    for sname in record.get("suites", []):
        sm = record["metrics_by_suite"].get(sname, {})
        lines.append(
            f"  {sname}: n={sm.get('n_cases', 0)} "
            f"domain={_fmt_accuracy(sm['classification']['domain_accuracy'])} "
            f"intent={_fmt_accuracy(sm['classification']['intent_accuracy'])} "
            f"strategy={_fmt_accuracy(sm['routing']['strategy_accuracy'])} "
            f"{_fmt_cost(sm.get('cost', {}))}"
        )
```

- [ ] **Step 4: Implement __main__.py**

In `agent/evaluation/__main__.py`:
- Change import: `from .dataset import Suite, DatasetError, load_suites`
- `_default_dataset` returns a directory:

```python
def _default_dataset(domain_dir: str) -> str:
    return f"evaluation/datasets/{Path(domain_dir).name}"
```

- Replace the body of `_cmd_run` from `dataset = load_dataset(dataset_path)` onward with:

```python
    try:
        suites = load_suites(dataset_path)
    except DatasetError as e:
        print(f"Dataset error: {e}", file=sys.stderr)
        return 1
    client = LLMClient(base_url=config.base_url, api_key=api_key, model=config.model)
    results_by_suite: dict[str, list] = {}
    for s in suites:
        results_by_suite[s.name] = run_evaluation(
            config, domain, s, client, skip_quality=args.skip_quality
        )
    metrics_by_suite = {
        s.name: compute_metrics(s, results_by_suite[s.name]) for s in suites
    }
    all_results = [r for rs in results_by_suite.values() for r in rs]
    all_cases = [c for s in suites for c in s.cases]
    merged = Suite(name="all", domain=suites[0].domain, cases=all_cases)
    metrics = compute_metrics(merged, all_results)
    judge_model = (config.evaluation.judge_model if config.evaluation else None) or config.model
    record = serialize_results(
        all_results, metrics, metrics_by_suite,
        domain=merged.domain, label=args.label, model=config.model,
        judge_model=judge_model, skip_quality=args.skip_quality,
        dataset_path=dataset_path, suites=[s.name for s in suites],
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

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_report.py tests/test_evaluation_cli.py -q`
Expected: PASS

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (runner/metrics/report/CLI all consistent; dataset tests still use `Dataset`/`load_dataset` which still exist)

- [ ] **Step 7: Commit**

```bash
git add agent/evaluation/report.py agent/evaluation/__main__.py tests/test_evaluation_report.py tests/test_evaluation_cli.py
git commit -m "feat: merged suite results with per-suite metrics"
```

---

### Task 4: Remove `Dataset`, `load_dataset`, and `category`

**Files:**
- Modify: `agent/evaluation/dataset.py`, `agent/evaluation/report.py`
- Test: `tests/test_evaluation_dataset.py`, `tests/test_evaluation_report.py`, `tests/test_evaluation_runner.py`, `tests/test_evaluation_metrics.py`, `tests/test_evaluation_cli.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: final state — `dataset.py` exports only `COMPLEXITY_LEVELS`, `OUT_OF_DOMAIN`, `REJECT_STRATEGY`, `DatasetError`, `EvalCase` (no `category`), `Suite`, `is_in_domain`, `load_suites`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_evaluation_dataset.py`:
- Remove `Dataset` and `load_dataset` from the import.
- Update the `_dataset_path` helper to put the file inside a `software_engineering/` directory (single-file domain now comes from the parent directory name):

```python
def _dataset_path(tmp_path, yaml_text):
    d = tmp_path / "software_engineering"
    d.mkdir(exist_ok=True)
    path = d / "se.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return str(path)
```

- In `_VALID`, remove both `category: knowledge` and `category: boundary` lines.
- Convert `test_load_dataset_valid` to suites:

```python
def test_load_dataset_valid(tmp_path):
    suites = load_suites(_dataset_path(tmp_path, _VALID))
    assert len(suites) == 1
    s = suites[0]
    assert s.name == "se"
    assert s.domain == "software_engineering"
    assert len(s.cases) == 2
    c = s.cases[0]
    assert c.id == "se-001"
    assert c.question == "What is dependency injection?"
    assert c.expected_domain == "software_engineering"
    assert c.expected_intent == "concept_explain"
    assert c.expected_complexity == "simple"
    assert c.expected_strategy == "teaching"
    assert c.expected_orchestrate is False
    assert c.answer_quality is True
    assert c.reference == "Dependency injection passes dependencies into a component."
```

- Convert `test_out_of_domain_case_fields`:

```python
def test_out_of_domain_case_fields(tmp_path):
    suites = load_suites(_dataset_path(tmp_path, _VALID))
    s = suites[0]
    c = s.cases[1]
    assert c.expected_domain == "other"
    assert c.expected_intent is None
    assert c.expected_complexity is None
    assert c.expected_strategy == "reject"
    assert is_in_domain(c, s) is False
    assert is_in_domain(s.cases[0], s) is True
```

- Convert `test_load_dataset_answer_quality_defaults_true` to `load_suites(str(path))[0]` and remove its `category: knowledge` line.
- Convert `test_load_dataset_missing_file` to `load_suites("/nonexistent/se.yaml")`.
- Convert `test_load_dataset_bad_yaml` to `load_suites(_dataset_path(tmp_path, ":: not: [valid"))`.
- Convert `test_load_dataset_missing_cases` — the new error text is `"Dataset 'cases' must be a list"`; keep the `pytest.raises(DatasetError)` assertion, drop the `domain:` line from the input YAML.
- Convert `test_load_dataset_invalid_complexity` to `load_suites(_dataset_path(...))` and remove its `category: knowledge` line.
- Remove `test_load_dataset_missing_domain` (no top-level domain anymore) and `test_load_dataset_unknown_category` (category removed).
- Remove `category: knowledge` from the inline YAML strings in `_suite_dir` and `test_load_suites_directory_missing_cases`.
- Update `test_load_committed_software_engineering_dataset` to load the directory (this file is created in Task 5; until then the test will fail — see note at Step 2).

Note: `test_load_committed_software_engineering_dataset` is expected to fail until Task 5 creates the dataset directory. To keep the suite green, **temporarily** comment out that one test with a `# TODO(Task 5): restore when suite dataset dir lands` marker. It will be restored in Task 5.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation_dataset.py tests/test_evaluation_runner.py tests/test_evaluation_metrics.py tests/test_evaluation_report.py tests/test_evaluation_cli.py -q`
Expected: the dataset tests fail on removed imports (that is correct); the other files may fail only if they still construct `Dataset` or read `category` — update them:

- `tests/test_evaluation_runner.py`, `tests/test_evaluation_metrics.py`, `tests/test_evaluation_report.py`: remove the `category="..."` argument from every `EvalCase(...)` construction, and import `Suite` instead of `Dataset`.
- `tests/test_evaluation_report.py`: remove the `assert case["category"]...` assertion if present (the `_record`/`_case` helpers never asserted it, but check).
- `tests/test_evaluation_cli.py`: remove `category: knowledge` from the inline dataset YAML.

- [ ] **Step 3: Implement dataset.py final state**

- Remove the `CATEGORIES` constant.
- Remove `category: str` from `EvalCase`.
- In `_validate_case`, remove the `category = raw.get("category")` and `if category not in CATEGORIES:` lines, and remove `category=category,` from the returned `EvalCase(...)`.
- Remove the `Dataset` dataclass and the `load_dataset` function entirely (the `_load_yaml_cases`/`load_suites` added in Task 1 replace them).

- [ ] **Step 4: Implement report.py final state**

In `_case_record`, remove the `"category": r.case.category,` line.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_dataset.py tests/test_evaluation_runner.py tests/test_evaluation_metrics.py tests/test_evaluation_report.py tests/test_evaluation_cli.py -q`
Expected: PASS (with `test_load_committed_software_engineering_dataset` commented out)

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (the committed-dataset test is temporarily skipped via its `# TODO(Task 5)` marker)

- [ ] **Step 7: Commit**

```bash
git add agent/evaluation/dataset.py agent/evaluation/report.py tests/test_evaluation_dataset.py tests/test_evaluation_runner.py tests/test_evaluation_metrics.py tests/test_evaluation_report.py tests/test_evaluation_cli.py
git commit -m "refactor: remove Dataset and EvalCase.category"
```

---

### Task 5: Create the suite dataset directory and restore the committed-dataset test

**Files:**
- Delete: `evaluation/datasets/software_engineering.yaml`
- Create: `evaluation/datasets/software_engineering/{classification,routing,direct,teaching,debugging,analysis,code_snippet,orchestration}.yaml`
- Modify: `tests/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: `load_suites` (directory mode), `Suite`/`EvalCase` (no `category`).
- Produces: the committed dataset directory, covering all intents, strategies, complexity levels, orchestration, and out-of-domain cases across the combined suites.

All 46 existing cases are redistributed, each into exactly one suite:

- `classification.yaml`: se-110, se-111, se-112, se-120, se-121, se-130, se-131, se-140, se-141, se-150, se-151
- `routing.yaml`: se-050, se-060, se-080, se-090
- `direct.yaml`: se-001, se-002, se-003, se-004, se-005, se-006, se-040
- `teaching.yaml`: se-010, se-011, se-012, se-013, se-014, se-020, se-021, se-030
- `debugging.yaml`: se-051, se-053
- `analysis.yaml`: se-062, se-070, se-072, se-081, se-091
- `code_snippet.yaml`: se-100, se-101
- `orchestration.yaml`: se-031, se-052, se-061, se-071, se-102, se-160, se-161

(Total = 46 cases; every case appears exactly once; `orchestrate: true` cases all live in `orchestration.yaml`; OOD `other` cases live in `classification.yaml`.)

- [ ] **Step 1: Write the failing test**

Replace the commented-out `test_load_committed_software_engineering_dataset` in `tests/test_evaluation_dataset.py`:

```python
def test_load_committed_software_engineering_suites():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    path = repo / "evaluation" / "datasets" / "software_engineering"
    suites = load_suites(str(path))
    names = [s.name for s in suites]
    assert names == ["analysis", "classification", "code_snippet", "debugging",
                     "direct", "orchestration", "routing", "teaching"]
    assert all(s.domain == "software_engineering" for s in suites)
    all_cases = [c for s in suites for c in s.cases]
    assert len(all_cases) >= 40
    ids = [c.id for c in all_cases]
    assert len(ids) == len(set(ids))  # no cross-suite duplication
    intents = {c.expected_intent for c in all_cases}
    assert {"faq", "concept_explain", "tutorial", "learning_guide", "summarization",
            "troubleshooting", "performance_analysis", "comparison", "architecture_design",
            "code_review", "generate_code"} <= intents
    strategies = {c.expected_strategy for c in all_cases}
    assert {"direct", "teaching", "debugging", "analysis", "code_snippet"} <= strategies
    assert {"simple", "medium", "complex"} <= {c.expected_complexity for c in all_cases}
    assert any(c.expected_orchestrate for c in all_cases)
    assert any(c.expected_domain == "other" for c in all_cases)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_evaluation_dataset.py::test_load_committed_software_engineering_suites -q`
Expected: FAIL (directory not found / no such file)

- [ ] **Step 3: Create the suite dataset files**

Delete the old single file:

```bash
git rm evaluation/datasets/software_engineering.yaml
```

Create the 8 suite YAML files with the exact content below (each file is `cases:` + a YAML list of cases; each case keeps its original `id`, `question`, `expected`, `answer_quality`, `reference` — minus any `category` line).

`evaluation/datasets/software_engineering/classification.yaml`:

```yaml
cases:
  - id: se-110
    question: "Recommend a good restaurant in Tokyo."
    expected: {domain: other, intent: null, complexity: null, strategy: reject, orchestrate: false}
    answer_quality: false
  - id: se-111
    question: "Explain the causes of the French Revolution."
    expected: {domain: other, intent: null, complexity: null, strategy: reject, orchestrate: false}
    answer_quality: false
  - id: se-112
    question: "What is the capital of Australia?"
    expected: {domain: other, intent: null, complexity: null, strategy: reject, orchestrate: false}
    answer_quality: false
  - id: se-120
    question: "What is a hash function?"
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-121
    question: "Why are hash tables fast for key lookups?"
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-130
    question: "Teach me Git branching with hands-on commands."
    expected: {domain: software_engineering, intent: tutorial, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-131
    question: "Plan out a full curriculum to master Kubernetes from beginner to advanced."
    expected: {domain: software_engineering, intent: learning_guide, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-140
    question: "My service became slow right after deploying the new caching layer. Why might that be?"
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: debugging, orchestrate: false}
    answer_quality: true
  - id: se-141
    question: "Compare the performance of an in-process cache versus a distributed cache for a read-heavy workload."
    expected: {domain: software_engineering, intent: performance_analysis, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-150
    question: "Should we use event-driven or request-response architecture for our notification service?"
    expected: {domain: software_engineering, intent: architecture_design, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-151
    question: "Compare queues and topics as messaging primitives."
    expected: {domain: software_engineering, intent: comparison, complexity: simple, strategy: analysis, orchestrate: false}
    answer_quality: true
```

`evaluation/datasets/software_engineering/routing.yaml`:

```yaml
cases:
  - id: se-050
    question: "My database connection pool is exhausted under load and new requests hang. How do I diagnose it?"
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: debugging, orchestrate: false}
    answer_quality: true
  - id: se-060
    question: "Analyze why my API response time degrades as concurrent users increase, and identify the bottleneck."
    expected: {domain: software_engineering, intent: performance_analysis, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-080
    question: "Compare gRPC and REST for inter-service communication in a microservices environment."
    expected: {domain: software_engineering, intent: comparison, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-090
    question: "Review this Python function for correctness and style: <code>def sum(a, b): return a + b  # never used</code>"
    expected: {domain: software_engineering, intent: code_review, complexity: simple, strategy: analysis, orchestrate: false}
    answer_quality: true
```

`evaluation/datasets/software_engineering/direct.yaml`:

```yaml
cases:
  - id: se-001
    question: "What does HTTP status code 503 mean?"
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-002
    question: "What is the default port for MySQL?"
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-003
    question: "What is the time complexity of quicksort in the average case?"
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-004
    question: "What is a database index?"
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-005
    question: "How do you create a git branch?"
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-006
    question: "What does the SOLID acronym stand for?"
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - id: se-040
    question: "Summarize the key ideas of this article into three bullet points: <article>Effective debugging requires reproducing the failure deterministically, isolating the smallest failing input, and forming hypotheses that you can test rather than guessing at fixes.</article>"
    expected: {domain: software_engineering, intent: summarization, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: false
```

`evaluation/datasets/software_engineering/teaching.yaml`:

```yaml
cases:
  - id: se-010
    question: "What is dependency injection?"
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true
    reference: "Dependency injection is a technique where a component receives its dependencies from outside rather than constructing them itself, improving testability and decoupling."
  - id: se-011
    question: "Explain how a virtual memory page table works together with the TLB."
    expected: {domain: software_engineering, intent: concept_explain, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-012
    question: "Why is immutability important in functional programming?"
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-013
    question: "Explain the difference between concurrency and parallelism."
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-014
    question: "Explain how a relational database query planner works and why an index changes the plan."
    expected: {domain: software_engineering, intent: concept_explain, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-020
    question: "Teach me step by step how to build a REST API with FastAPI and connect it to PostgreSQL."
    expected: {domain: software_engineering, intent: tutorial, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-021
    question: "Give me a step-by-step tutorial on setting up Docker Compose for a Node.js app with a Redis service."
    expected: {domain: software_engineering, intent: tutorial, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-030
    question: "Create a structured learning path to go from Python basics to backend web development."
    expected: {domain: software_engineering, intent: learning_guide, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
```

`evaluation/datasets/software_engineering/debugging.yaml`:

```yaml
cases:
  - id: se-051
    question: "Why does my C++ program crash with a segmentation fault when accessing index 0 of an empty vector?"
    expected: {domain: software_engineering, intent: troubleshooting, complexity: simple, strategy: debugging, orchestrate: false}
    answer_quality: true
  - id: se-053
    question: "My Node.js app crashes with an out-of-memory error only in production. How should I debug this?"
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: debugging, orchestrate: false}
    answer_quality: true
```

`evaluation/datasets/software_engineering/analysis.yaml`:

```yaml
cases:
  - id: se-062
    question: "Why is a bulk INSERT of 10 million rows slower than expected, and how can it be tuned?"
    expected: {domain: software_engineering, intent: performance_analysis, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-070
    question: "Design the module structure for a monorepo with several shared packages and clear dependency boundaries."
    expected: {domain: software_engineering, intent: architecture_design, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-072
    question: "Design a caching layer for a read-heavy news website, including invalidation strategy."
    expected: {domain: software_engineering, intent: architecture_design, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-081
    question: "Compare the Go defer statement with C++ RAII for resource management."
    expected: {domain: software_engineering, intent: comparison, complexity: simple, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-091
    question: "Perform a security-focused code review of this authentication snippet: <code>if user.password == request.password: login(user)</code>"
    expected: {domain: software_engineering, intent: code_review, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
```

`evaluation/datasets/software_engineering/code_snippet.yaml`:

```yaml
cases:
  - id: se-100
    question: "Write a Python function that checks whether a string is a palindrome."
    expected: {domain: software_engineering, intent: generate_code, complexity: simple, strategy: code_snippet, orchestrate: false}
    answer_quality: true
  - id: se-101
    question: "Write a Python function that reads a CSV file, validates each row, and returns a summary of invalid rows."
    expected: {domain: software_engineering, intent: generate_code, complexity: medium, strategy: code_snippet, orchestrate: false}
    answer_quality: true
```

`evaluation/datasets/software_engineering/orchestration.yaml`:

```yaml
cases:
  - id: se-031
    question: "Design a multi-month study plan for becoming a reliable systems engineer."
    expected: {domain: software_engineering, intent: learning_guide, complexity: complex, strategy: teaching, orchestrate: true}
    answer_quality: true
  - id: se-052
    question: "A distributed system fails intermittently with timeout errors across several services. Investigate the root cause and propose a fix."
    expected: {domain: software_engineering, intent: troubleshooting, complexity: complex, strategy: debugging, orchestrate: true}
    answer_quality: true
  - id: se-061
    question: "Analyze end-to-end latency bottlenecks in a system spanning a CDN, application servers, and a data warehouse."
    expected: {domain: software_engineering, intent: performance_analysis, complexity: complex, strategy: analysis, orchestrate: true}
    answer_quality: true
  - id: se-071
    question: "Design a scalable microservices architecture for an e-commerce platform covering orders, payments, and inventory."
    expected: {domain: software_engineering, intent: architecture_design, complexity: complex, strategy: analysis, orchestrate: true}
    answer_quality: true
  - id: se-102
    question: "Build a complete CLI tool in Python with argument parsing, a config file, and unit tests."
    expected: {domain: software_engineering, intent: generate_code, complexity: complex, strategy: code_snippet, orchestrate: true}
    answer_quality: true
  - id: se-160
    question: "How should we scale our PostgreSQL database to handle 10x the current read volume?"
    expected: {domain: software_engineering, intent: architecture_design, complexity: complex, strategy: analysis, orchestrate: true}
    answer_quality: true
  - id: se-161
    question: "Explain the full request lifecycle of a React application from URL entry to paint."
    expected: {domain: software_engineering, intent: concept_explain, complexity: complex, strategy: teaching, orchestrate: true}
    answer_quality: true
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_evaluation_dataset.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add evaluation/datasets tests/test_evaluation_dataset.py
git commit -m "feat: split dataset into per-suite yaml files"
```

---

### Task 6: CLI suite selection and per-suite cap

**Files:**
- Modify: `agent/evaluation/__main__.py`
- Test: `tests/test_evaluation_cli.py`

**Interfaces:**
- Consumes: `load_suites`, `Suite` from `agent.evaluation.dataset` (already imported in Task 3).
- Produces: `run` subcommand flags `--suite` (multiple, default all) and `--max-per-suite N` (default unlimited).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluation_cli.py` (reuse the suite-dir + config setup pattern from `test_main_run_prints_summary_and_writes_file`; extract a helper `_make_env(tmp_path)` if needed):

```python
def _suite_cli_env(tmp_path):
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
    suite_dir = dataset_dir / "software_engineering"
    suite_dir.mkdir()
    for name in ("direct", "teaching"):
        (suite_dir / f"{name}.yaml").write_text(
            'cases:\n'
            '  - id: a\n'
            '    question: "q"\n'
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
    return config_path, suite_dir


def test_main_run_suite_selection(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    out = _run_with_fake(monkeypatch, [
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--suite", "direct", "--label", "sel", "--results-dir", str(tmp_path / "r"),
        "--skip-quality",
    ])
    assert "Per-suite" in out
    assert "direct" in out
    assert "teaching" not in out


def test_main_run_max_per_suite(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    out = _run_with_fake(monkeypatch, [
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--max-per-suite", "1", "--label", "mx", "--results-dir", str(tmp_path / "r"),
        "--skip-quality",
    ])
    assert "Per-suite" in out
```

Add the `_run_with_fake` helper (same FakeClient + stdout monkeypatch pattern as the existing test):

```python
def _run_with_fake(monkeypatch, argv):
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
    import io
    import sys
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main(argv)
    assert rc == 0
    return out.getvalue()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation_cli.py -q`
Expected: FAIL (unrecognized arguments: `--suite` / `--max-per-suite`)

- [ ] **Step 3: Implement**

In `agent/evaluation/__main__.py`, add the two flags to the `run` subparser:

```python
    run_p.add_argument("--suite", nargs="+", default=None,
                       help="suites to run by name (default: all suites)")
    run_p.add_argument("--max-per-suite", type=int, default=None,
                       help="cap cases per suite (default: unlimited)")
```

In `_cmd_run`, after `suites = load_suites(dataset_path)` and before the run loop, add selection and truncation:

```python
    if args.suite:
        wanted = set(args.suite)
        suites = [s for s in suites if s.name in wanted]
        if not suites:
            print(f"No suites matched: {', '.join(args.suite)}", file=sys.stderr)
            return 1
    if args.max_per_suite is not None:
        suites = [Suite(name=s.name, domain=s.domain, cases=s.cases[:args.max_per_suite])
                  for s in suites]
```

(Import `Suite` is already added in Task 3.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_cli.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add agent/evaluation/__main__.py tests/test_evaluation_cli.py
git commit -m "feat: support --suite selection and --max-per-suite"
```

---

### Task 7: Full regression

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS (all tests; count is 196 + new tests − removed tests)

- [ ] **Step 2: Grep for stale references**

Run: `rg "load_dataset|Dataset\(|\bDataset\b|category" agent/evaluation -n`
Expected: no matches for `load_dataset`, `Dataset`, or `category` in `agent/evaluation/` (only `DatasetError` is allowed).

- [ ] **Step 3: Manual smoke of the CLI help**

Run: `uv run python -m agent.evaluation run --help`
Expected: output lists `--suite` and `--max-per-suite`.