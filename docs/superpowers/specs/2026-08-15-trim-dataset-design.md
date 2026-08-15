# Trim Evaluation Dataset to Most-Representative Cases

## Goal

Reduce the committed `software_engineering` evaluation dataset from 46 cases to 22 by keeping only the most representative cases per suite, so evaluation runs stay fast while still exercising every suite's key behaviors.

## Scope

- **Files changed:** the 8 suite YAML files under `evaluation/datasets/software_engineering/`, and one assertion in `tests/test_evaluation_dataset.py` (`test_load_committed_software_engineering_suites`).
- **No code changes** to `agent/evaluation/` (dataset loader, runner, metrics, report, CLI all unaffected).

## Selection (46 → 22)

Each suite keeps 1-3 cases that maximize coverage of the suite's distinct intents and complexity levels. Canonical/representative examples preferred (e.g. the FAQ with a reference, the tutorial that spans a stack).

| Suite | Count | Cases kept |
|-------|-------|------------|
| classification | 3 | se-110 (OOD reject), se-120 (faq), se-140 (troubleshooting) |
| routing | 3 | se-050 (troubleshooting), se-060 (performance_analysis), se-090 (code_review) |
| direct | 3 | se-001 (faq), se-003 (faq), se-040 (summarization) |
| teaching | 3 | se-010 (concept_explain, keeps `reference`), se-020 (tutorial), se-030 (learning_guide) |
| debugging | 2 | se-051 (troubleshooting/simple), se-053 (troubleshooting/medium) |
| analysis | 3 | se-062 (performance_analysis), se-070 (architecture_design), se-081 (comparison) |
| code_snippet | 2 | se-100 (generate_code/simple), se-101 (generate_code/medium) |
| orchestration | 3 | se-052 (troubleshooting), se-071 (architecture_design), se-102 (generate_code) |

**Total: 22 cases.**

Per-case data is unchanged: `id`, `question`, `expected`, `answer_quality`, and `reference` (where present) are kept verbatim. Removed cases are deleted from their YAML file; IDs of kept cases are NOT renumbered.

## Coverage preservation

Verified against the trimmed set:

- **Intents (11):** faq, concept_explain, tutorial, learning_guide, summarization, troubleshooting, performance_analysis, comparison, architecture_design, code_review, generate_code — all present.
- **Strategies (5):** direct, teaching, debugging, analysis, code_snippet — all present (+ `reject` via OOD).
- **Complexity:** simple, medium, complex — all present (complex via the orchestration suite).
- **Orchestration:** `orchestrate: true` cases remain (se-052, se-071, se-102).
- **Out-of-domain:** `domain: other` case remains (se-110).
- **Answer-quality flags:** `answer_quality: false` preserved (se-040, se-110).

## Test update

In `tests/test_evaluation_dataset.py`, `test_load_committed_software_engineering_suites`:

- Change `assert len(all_cases) >= 40` → `assert len(all_cases) >= 20` (expected 22).
- All other assertions (8 suite names in order, single shared domain, unique ids, intent superset, strategy superset, complexity superset, any orchestrate, any OOD) remain as-is and still pass with the trimmed dataset.

## Non-goals

- No loader/runner/metrics/report/CLI changes.
- No renumbering of case IDs.
- No new suite files, no removed suite files.
- No changes to `--suite` / `--max-per-suite` semantics.

## Verification

- `uv run pytest tests/test_evaluation_dataset.py -q` — green.
- `uv run pytest -q` — green (200 passed, 5 skipped baseline; total case count falls, no test asserts the old 46).