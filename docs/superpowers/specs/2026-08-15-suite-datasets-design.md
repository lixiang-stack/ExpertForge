# Suite-Based Evaluation Datasets — Design

Date: 2026-08-15
Status: Approved

## 1. Motivation

The evaluation framework currently stores the golden dataset in a single
`evaluation/datasets/software_engineering.yaml` file: a flat `cases` list of
46 cases, each tagged with a `category` (knowledge / problem_solving /
evaluation / generation / boundary).

Running the whole dataset is slow and costly (46 cases × 3-4 LLM calls each:
router + answer + judge). We want to support:

- Running all cases of a given *type*, and
- Running only 1 case per type as a quick smoke run.

The requirement evolved into organizing the dataset as a **directory of
independent suites**, each a small YAML focused on one evaluation dimension.
A suite is the unit of selection and execution.

## 2. Data Organization

- Delete `evaluation/datasets/software_engineering.yaml`.
- Create directory `evaluation/datasets/software_engineering/` containing
  eight independent suite YAML files:
  - `classification.yaml`
  - `routing.yaml`
  - `direct.yaml`
  - `teaching.yaml`
  - `debugging.yaml`
  - `analysis.yaml`
  - `code_snippet.yaml`
  - `orchestration.yaml`
- Each suite file contains only a `cases` list (no top-level `domain`).
  The domain is derived from the dataset directory name.
- Suites are self-contained and independent: no cross-file references or
  shared case definitions. A case appears in at most one suite.
- The existing 46 cases are redistributed into suites by dimension and
  trimmed where sensible (classification-focused cases into
  `classification.yaml`, `direct` strategy cases into `direct.yaml`,
  `orchestrate: true` cases into `orchestration.yaml`, etc.).

### Suite file shape

```yaml
cases:
  - id: se-001
    question: "..."
    expected: {domain: software_engineering, intent: faq, complexity: simple, strategy: direct, orchestrate: false}
    answer_quality: true
  - ...
```

## 3. Data Model

Replace the `Dataset` dataclass with a single `Suite` dataclass.

```python
@dataclass
class Suite:
    name: str          # YAML file basename (e.g. "classification")
    domain: str        # derived from the dataset directory name
    cases: list[EvalCase]
```

- `EvalCase` is unchanged except that the `category` field is **removed**
  (along with the `CATEGORIES` constant and its validation in
  `_validate_case`, and the `category` serialization in `report.py`).
- `is_in_domain(case, suite)` compares `case.expected_domain == suite.domain`.
- All type annotations and construction sites that referenced `Dataset` now
  reference `Suite`:
  - `dataset.py` (`load_dataset` → `load_suites`)
  - `runner.py` (`run_evaluation`, `RecordingClient` unaffected)
  - `metrics.py` (`compute_metrics`)
  - `report.py` (`serialize_results`)
  - `__main__.py`
  - tests

## 4. Loading API

Replace `load_dataset(path) -> Dataset` with:

```python
def load_suites(path: str) -> list[Suite]:
```

- If `path` is a directory: iterate `*.yaml` files in it, each becomes a
  `Suite(name=<file basename>, domain=<directory name>, cases=...)`.
- If `path` is a single YAML file: return one suite with
  `name=<file basename>` and `domain` derived from its parent directory name.
- Validation reuses the existing `_read_yaml` / `_validate_case` logic,
  minus the `category` checks.

## 5. Execution Semantics

- `run_evaluation(config, domain, suite, client, *, skip_quality=False)`
  keeps its signature; it is invoked once per selected suite.
  Parameter type changes from `Dataset` to `Suite`.
- `CaseResult` gains a `suite: str` field (populated from the suite name),
  used for grouping in results and summary.
- A dispatch layer iterates the selected suites, runs each, and collects
  the per-suite results.

## 6. CLI

`python -m agent.evaluation run` gains:

- `--suite NAME [NAME ...]` — select suites by name; default: all suites
  in the dataset directory.
- `--max-per-suite N` — cap the number of cases run per suite (default:
  unlimited). `--max-per-suite 1` gives the smoke run (1 case per suite).
  Selection takes the first N cases in suite order.
- `--dataset` — now points at the dataset directory
  (default `evaluation/datasets/<config.domain_dir name>/`).

## 7. Results and Metrics

One run writes a **single merged result JSON**:

```json
{
  "domain": "software_engineering",
  "label": "...",
  "model": "...",
  "judge_model": "...",
  "skip_quality": false,
  "dataset": "<dataset dir>",
  "suites": ["classification", "routing", "..."],
  "metrics": { /* global aggregate */ },
  "metrics_by_suite": { "classification": {...}, "routing": {...} },
  "cases": [ /* each case record gains "suite": "<suite name>" */ ]
}
```

- `compute_metrics` is invoked once per suite (→ `metrics_by_suite`) and
  once globally over the merged case list (→ `metrics`).
- `format_summary` prints the global summary plus a per-suite summary.
- `diff` operates on the merged result file; no change to its behavior.

## 8. Testing

- Dataset tests: replace `load_dataset` coverage with `load_suites`
  (directory loading, suite naming, single-file fallback, domain derivation,
  `category` removal).
- Runner tests: `suite` field on `CaseResult`, `--max-per-suite` truncation
  (first N cases).
- Report/CLI tests: merged result file with `suites` + `metrics_by_suite`,
  per-suite summary output, `--suite` selection, `--max-per-suite` behavior.
- Existing `Dataset(...)` construction sites in tests become
  `Suite(name=..., domain=..., cases=...)`.

## 9. Out of Scope

- Cross-suite case sharing or references.
- Per-suite result files (results are merged into one file).
- Preserving the old single-file dataset format.