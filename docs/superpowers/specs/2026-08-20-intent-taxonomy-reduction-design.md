# Intent Taxonomy Reduction + Strategy Overhaul — Design Spec

Date: 2026-08-20
Status: Approved (brainstorming)
Reference: `draft_v4.md`

## Problem

The current `software_engineering` domain has 11 intents with overlapping
boundaries (`faq`, `summarization`, `code_review`, `generate_code`,
`performance_analysis` blur into `concept_explain`, `code_task`,
`troubleshooting`). The taxonomy has grown with features rather than being
driven by routing value, and strategy routing conflates intent with processing
style. The goal is a smaller, stable, easy-to-label, easy-to-evaluate taxonomy.

## Goals

1. Reduce to 7 core intents with clear semantic boundaries.
2. Separate intent from strategy: intent answers "what does the user want to
   do"; strategy answers "how should the system process it".
3. Keep `orchestration` as a pipeline gate, not a strategy.
4. Reorganize the evaluation dataset by intent + `boundary.yaml`.
5. Feed the enriched definitions into runtime classification; evaluate via
   boundary cases rather than hardcoded classifier logic.

## Section 1 — Taxonomy & Strategy Mapping

### 1.1 Intent set

The `software_engineering` domain keeps exactly 7 core intents:

```
concept_explain
tutorial
learning_guide
code_task
troubleshooting
architecture_design
comparison
```

Removed intents:

| Intent | Disposition |
|---|---|
| `faq` | merged into `concept_explain` |
| `summarization` | removed (generic LLM capability, not an SE expert intent) |
| `code_review` | merged into `code_task` |
| `generate_code` | merged into `code_task` |
| `performance_analysis` | merged into `troubleshooting` |

`code_task` consolidates generate / explain / review / modify / refactor /
transform / optimize over source code. These do NOT become first-class intents.

### 1.2 Strategy set

Prompt files become `{direct, teaching, analysis, planning}`:

- `direct.md` — kept
- `teaching.md` — kept
- `analysis.md` — kept
- `planning.md` — NEW (for learning_guide + architecture_design)
- `code_snippet.md` — DELETED
- `debugging.md` — DELETED

`orchestration` remains the pipeline gate flag, not a strategy prompt.

### 1.3 intent → strategy mapping (`intent_mapping.yaml`)

| Intent | Strategy |
|---|---|
| concept_explain | teaching |
| tutorial | teaching |
| learning_guide | planning |
| code_task | direct |
| troubleshooting | analysis |
| architecture_design | planning |
| comparison | analysis |

### 1.4 Orchestration gate (`orchestration.yaml`)

```
enabled: true
min_complexity: complex
intents: [architecture_design, troubleshooting, code_task]
max_workers: 4
evaluator: {enabled: true, min_dimension_score: 3, max_rounds: 1}
```

## Section 2 — Dataset Reorganization & Case Migration

### 2.1 File layout

Target layout in `evaluation/datasets/software_engineering/`:

| File | Cases |
|---|---|
| `concept_explain.yaml` | se-010, se-121, se-127, se-001, se-003, se-120 |
| `tutorial.yaml` | se-020, se-122 |
| `learning_guide.yaml` | se-030, se-123 |
| `code_task.yaml` | se-090, se-100, se-101, se-102, se-103 (NEW FILE) |
| `troubleshooting.yaml` | se-050, se-051, se-052, se-053, se-054, se-140, se-060, se-062, se-124 |
| `architecture_design.yaml` | se-070, se-071, se-082, se-126, se-128 |
| `comparison.yaml` | se-081, se-125 |
| `boundary.yaml` | se-110 (OOD) + new boundary cases |

Deleted files: `faq.yaml`, `summarization.yaml`, `performance_analysis.yaml`,
`code_review.yaml`, `generate_code.yaml`.

### 2.2 Case migration table

`expected` field updates per case:

| Case | intent change | strategy change |
|---|---|---|
| se-001, se-003, se-120 | faq → concept_explain | direct → teaching |
| se-040 | DELETED (summarization) | — |
| se-060, se-062, se-124 | performance_analysis → troubleshooting | analysis → analysis |
| se-090 | code_review → code_task | analysis → direct |
| se-100, se-101, se-102, se-103 | generate_code → code_task | code_snippet → direct |
| se-030, se-123 | learning_guide (unchanged) | teaching → planning |
| se-070, se-071, se-082, se-126, se-128 | architecture_design (unchanged) | analysis → planning |
| se-050, se-051, se-052, se-053, se-054, se-140 | troubleshooting (unchanged) | debugging → analysis |
| se-110 | unchanged (OOD) | reject |

`tier` is unchanged for all surviving cases. se-052, se-071, se-102, se-126,
se-128 remain `full_expert` with `orchestrate: true` — all three orchestrated
intents (architecture_design, troubleshooting, code_task) are still in the gate.

## Section 3 — New Content

### 3.1 `intents.yaml`

Rewritten to exactly 7 intents, each with `description`, `positive_examples`,
`negative_examples`, and `boundaries` in the existing format. `code_task`
consolidates the code-operation behaviors and its boundaries distinguish it
from `troubleshooting` (object = source code vs existing problem).

### 3.2 `planning.md` (new strategy prompt)

Structures answers for long-term learning paths (`learning_guide`) and system
design (`architecture_design`): goals → constraints → phases/options →
trade-offs → decision. Content drafted to fit the existing prompt style.

### 3.3 `boundary.yaml` — boundary cases

Add boundary cases covering the four high-confusion pairs, each
`tier: classification`, with correct `expected`:

- concept_explain ↔ tutorial (dependency injection: "What is…" vs "How do I
  use…" vs "Show me step by step…")
- code_task ↔ troubleshooting (explain/refactor a Go function vs "My Go
  service crashes…")
- tutorial ↔ code_task ("Teach me step by step how to build a REST API" vs
  "Implement a REST API for me")
- troubleshooting ↔ architecture_design ("My service can't handle 10K
  concurrent users, find the bottleneck" vs "Design a service that handles
  10K concurrent users")

`boundary.yaml` is evaluation-only, NOT an intent. OOD case se-110 stays here.

### 3.4 Classifier

No code change in `agent/classification.py`. It already renders `intents.yaml`
definitions into the prompt. Boundary knowledge lives in the taxonomy and
dataset, not in classifier program logic (no keyword-based intent detection).

### 3.5 Tests

Unit tests that hardcode removed names must be updated to surviving names:
- `tests/unit/test_classification.py` uses `faq` as a sample intent id.
- `tests/unit/test_chat.py` uses `debugging`/`code_snippet` strategies.
- `tests/unit/test_evaluation_runner.py`, `test_observability_install.py`,
  `test_config.py`, `test_evaluation_diff.py` reference `faq`/`direct` etc. as
  generic fixtures — update only where a removed strategy/intent is required.
- `tests/live/test_smoke.py` writes a temp dataset with
  `intent: faq` / `strategy: direct`; update to `concept_explain` /
  `teaching` so it stays consistent with the 7-intent taxonomy.

The domain loader (`agent/domain_config.py`) already validates that every
mapped strategy has a prompt file, so removing `code_snippet.md` and
`debugging.md` must be mirrored in `intent_mapping.yaml`.

## Out of Scope

- New intents beyond the 7.
- `code_task_type`, `problem_type`, `sub_intent`, intent hierarchy/graph
  (deferred until evaluation proves routing value).
- `taxonomy.yaml` config layer (existing `intents.yaml` already expresses the
  needed structure).
- Changes to orchestration prompt templates or the orchestrator pipeline.
- Other domains.

## Acceptance Criteria

1. `domain/software_engineering/intents.yaml` contains exactly the 7 intents.
2. `intent_mapping.yaml` maps all 7 intents to surviving strategies.
3. Only `direct.md`, `teaching.md`, `analysis.md`, `planning.md` exist in
   `prompts/`.
4. `orchestration.yaml` intents = `[architecture_design, troubleshooting,
   code_task]`.
5. Dataset files reorganized per §2.1; all cases carry valid expected fields
   referencing surviving intent/strategy names; se-040 removed.
6. `boundary.yaml` contains OOD + boundary cases; boundary cases are not an
   intent.
7. `uv run pytest -q` passes.
8. `--tier all` loads without dataset validation errors.