# Intent Taxonomy Reduction + Strategy & Evaluation Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the `software_engineering` domain from 11 intents to 7 core intents, consolidate strategies to `{direct, teaching, analysis, planning}`, keep orchestration as an independent pipeline gate, and reorganize the evaluation dataset + tests to match.

**Architecture:** All changes are declarative (YAML/MD config + dataset) plus test updates. No production code changes: `agent/classification.py`, `agent/router.py`, `agent/domain_config.py`, and the evaluation pipeline stay untouched. The domain loader already validates cross-references (intent→mapping→prompt→orchestration), so the taxonomy change is driven entirely by the domain directory and dataset files.

**Tech Stack:** Python 3.x, PyYAML, pytest. Test command: `uv run pytest -q`.

## Global Constraints

- Final intents: `concept_explain`, `tutorial`, `learning_guide`, `code_task`, `troubleshooting`, `architecture_design`, `comparison`.
- Final strategies (prompt files): `direct.md`, `teaching.md`, `analysis.md`, `planning.md`. Delete `code_snippet.md` + `debugging.md`.
- Final intent→strategy map: concept_explain→teaching, tutorial→teaching, learning_guide→planning, code_task→direct, troubleshooting→analysis, architecture_design→planning, comparison→analysis.
- `orchestration.yaml` intents = `[architecture_design, troubleshooting, code_task]`; orchestration stays a pipeline gate, never a strategy prompt.
- Removed dataset files: `faq.yaml`, `summarization.yaml`, `performance_analysis.yaml`, `code_review.yaml`, `generate_code.yaml`. Case `se-040` is deleted.
- All surviving cases keep their existing `tier`. `se-052`, `se-071`, `se-102`, `se-126`, `se-128` remain `full_expert` + `orchestrate: true`.
- New boundary cases use `tier: classification`.
- No keyword-based / hardcoded intent detection added to `agent/classification.py`.
- Full test suite passes: `uv run pytest -q` (no API key needed).
- No new dependencies.

**Dataset schema (for reference in Tasks 4–8):** each file is a mapping with a `cases:` list. Each case: `id`, `question`, `tier` (`classification`|`routing`|`full_expert`), optional `smoke: true`, and `expected: {domain, intent, complexity, strategy, orchestrate}`. Loaded by `agent.evaluation.dataset.load_suites(path)` → `list[Suite]`, each with `.name`, `.domain`, `.cases` (EvalCase with `.expected_intent`, `.expected_strategy`, `.tier`, `.smoke`, ...).

---

### Task 1: Rewrite `intents.yaml` to the 7 core intents

**Files:**
- Modify: `domain/software_engineering/intents.yaml`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing new — file is parsed by `agent/domain_config.py:_parse_intents` into `dict[str, IntentDef]`.
- Produces: exactly 7 `IntentDef` entries with `description`, `positive_examples`, `negative_examples`, `boundaries`; used at runtime by `agent/classification.py` for the classification prompt.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_config.py`, add a test that loads the real committed domain and asserts the intent set (reuse existing `load_domain_config` import):

```python
from pathlib import Path

def test_real_domain_has_exactly_seven_intents():
    repo = Path(__file__).resolve().parents[2]
    domain = load_domain_config(str(repo / "domain" / "software_engineering"))
    assert set(domain.intents) == {
        "concept_explain", "tutorial", "learning_guide", "code_task",
        "troubleshooting", "architecture_design", "comparison",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py::test_real_domain_has_exactly_seven_intents -v`
Expected: FAIL — the real domain currently has 11 intents.

- [ ] **Step 3: Rewrite `intents.yaml`**

Replace the entire file with:

```yaml
- id: concept_explain
  description: Explain a software engineering concept, mechanism, principle, or factual topic
  positive_examples:
    - "What is dependency injection?"
    - "Why does TCP use a three-way handshake?"
    - "What is the difference between a process and a thread?"
  negative_examples:
    - "Show me step by step how to build a REST API."
    - "Give me a roadmap for becoming a Go backend expert."
    - "Write a Go function that parses this JSON."
  boundaries:
    - "Use concept_explain when the primary goal is understanding a concept; code examples included in an explanation do not make it a code_task."
    - "Prefer tutorial when the user wants step-by-step instructions."
    - "Prefer code_task when the primary object of the request is concrete source code."
- id: tutorial
  description: Teach the user how to perform a specific task, usually with step-by-step instructions
  positive_examples:
    - "How do I configure Redis persistence?"
    - "Show me step by step how to build a REST API."
    - "How do I set up a Go project with MongoDB?"
  negative_examples:
    - "Implement a REST API for me."
    - "How should I learn distributed systems?"
  boundaries:
    - "Prefer tutorial when the user wants to learn how to perform the task; prefer code_task when the user asks you to perform the task directly."
    - "Prefer learning_guide over tutorial for a long-term roadmap, curriculum, or study plan."
- id: learning_guide
  description: Provide a structured, longer-term learning path, roadmap, curriculum, or study plan
  positive_examples:
    - "How should I learn distributed systems?"
    - "Give me a roadmap for becoming a Go backend expert."
    - "What should I learn before studying Kubernetes?"
  negative_examples:
    - "Show me step by step how to build a REST API."
    - "Design a distributed rate limiter."
  boundaries:
    - "Use learning_guide for roadmap, learning sequence, curriculum, milestones, prerequisites, or long-term study planning."
    - "Prefer tutorial for a single step-by-step task."
- id: code_task
  description: Perform an operation involving concrete source code (generate, explain, review, modify, refactor, transform, optimize)
  positive_examples:
    - "Write a Go function that parses this JSON."
    - "Explain what this function does."
    - "Review this function for bugs."
    - "Refactor this code to remove duplication."
  negative_examples:
    - "What is dependency injection?"
    - "My Go service crashes after several hours. Find the cause."
  boundaries:
    - "Use code_task when the primary object of the request is concrete source code; a general concept explanation stays concept_explain even if code examples are included."
    - "Prefer troubleshooting when the user reports an existing problem and wants the cause diagnosed or fixed."
- id: troubleshooting
  description: Diagnose and resolve an existing technical problem, including performance problems
  positive_examples:
    - "Why is my MongoDB connection failing?"
    - "My Go service crashes under load. Find the cause."
    - "The application has intermittent memory leaks. How can I find the root cause?"
  negative_examples:
    - "Design a service that can handle 10K concurrent users."
    - "Refactor this Go function to remove duplication."
  boundaries:
    - "Use troubleshooting when an existing system has a problem to diagnose; use architecture_design when the user asks how to design a system to meet a target."
- id: architecture_design
  description: Design or redesign a software system, architecture, component structure, or major technical solution
  positive_examples:
    - "Design a distributed rate limiter."
    - "How should I architect a multi-region service?"
    - "How should this monolith be decomposed?"
  negative_examples:
    - "My service cannot handle 10K concurrent users. Find the bottleneck."
    - "Compare Kafka vs Pulsar."
  boundaries:
    - "Use architecture_design when the primary goal is designing a solution; use troubleshooting when diagnosing an existing failure."
- id: comparison
  description: Compare alternatives and understand trade-offs
  positive_examples:
    - "Redis vs Memcached?"
    - "Kafka vs Pulsar?"
    - "What are the trade-offs between REST and gRPC?"
  negative_examples:
    - "Design a message processing system using Pulsar."
    - "What is a hash function?"
  boundaries:
    - "Use comparison when the user contrasts two or more alternatives; use architecture_design when the user asks to design a system."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py::test_real_domain_has_exactly_seven_intents -v`
Expected: PASS. Note: the full suite will fail until Tasks 2–3 land (mapping/prompts still reference removed names), which is expected.

- [ ] **Step 5: Commit**

```bash
git add domain/software_engineering/intents.yaml tests/unit/test_config.py
git commit -m "feat: reduce software_engineering intents to 7 core intents"
```

---

### Task 2: Update `intent_mapping.yaml` + `orchestration.yaml`

**Files:**
- Modify: `domain/software_engineering/intent_mapping.yaml`
- Modify: `domain/software_engineering/orchestration.yaml`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `intents.yaml` (Task 1) — the 7 surviving intent ids.
- Produces: `intent_mapping: dict[str, str]` (7 entries) and `OrchestrationPolicy(intents=[architecture_design, troubleshooting, code_task])` used by `agent/router.py:route`.

- [ ] **Step 1: Update `intent_mapping.yaml`**

Replace the entire file with:

```yaml
concept_explain: teaching
tutorial: teaching
learning_guide: planning
code_task: direct
troubleshooting: analysis
architecture_design: planning
comparison: analysis
```

- [ ] **Step 2: Update `orchestration.yaml`**

Replace the entire file with:

```yaml
enabled: true
min_complexity: complex
intents:
  - architecture_design
  - troubleshooting
  - code_task
max_workers: 4
evaluator:
  enabled: true
  min_dimension_score: 3
  max_rounds: 1
```

- [ ] **Step 3: Verify domain still loads**

Run: `uv run python -c "from agent.domain_config import load_domain_config; d = load_domain_config('domain/software_engineering'); print(d.intent_mapping); print(d.orchestration.intents)"`

Note: this will fail with "references unknown strategy" until Task 3 creates `planning.md` and deletes the stale prompts. Expected failure now.

- [ ] **Step 4: Commit**

```bash
git add domain/software_engineering/intent_mapping.yaml domain/software_engineering/orchestration.yaml
git commit -m "feat: remap intents to strategies, narrow orchestration gate"
```

---

### Task 3: Strategy prompts — add `planning.md`, delete `code_snippet.md` + `debugging.md`

**Files:**
- Create: `domain/software_engineering/prompts/planning.md`
- Delete: `domain/software_engineering/prompts/code_snippet.md`
- Delete: `domain/software_engineering/prompts/debugging.md`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `intent_mapping.yaml` (Task 2) — the four surviving strategy ids.
- Produces: `domain.prompts` dict with exactly the four keys `direct`, `teaching`, `analysis`, `planning`. `agent/domain_config.py:_parse_prompts` reads every `*.md` in `prompts/`; `load_domain_config` validates every mapped strategy has a prompt file.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_config.py`, add a test asserting the strategy prompt set:

```python
def test_real_domain_prompts_are_exactly_four_strategies():
    repo = Path(__file__).resolve().parents[2]
    domain = load_domain_config(str(repo / "domain" / "software_engineering"))
    assert set(domain.strategies) == {"direct", "teaching", "analysis", "planning"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py::test_real_domain_prompts_are_exactly_four_strategies -v`
Expected: FAIL — the real domain currently has 5 prompts: analysis, code_snippet, debugging, direct, teaching.

- [ ] **Step 3: Create `planning.md`**

Create `domain/software_engineering/prompts/planning.md`, adapting the structure/voice of the existing prompts (`direct.md`, `teaching.md`, `analysis.md`). It must adapt its output structure to the intent (learning_guide vs architecture_design). Recommended content:

```markdown
# Planning Strategy

## Role
You act as a planning expert. You produce structured plans, roadmaps,
architectures, and step-by-step designs rather than executing the work directly.

## Behavior
- Break the request into phases, milestones, prerequisites, and concrete deliverables.
- Call out dependencies, risks, and decision points with trade-offs.
- Produce artifacts such as roadmaps, module breakdowns, architecture sketches, or study curriculums.
- Adapt the output structure to the intent:
  - learning_guide: learning goals -> prerequisites -> learning phases -> practice -> milestones
  - architecture_design: goals -> constraints -> options -> trade-offs -> decision -> implementation phases
- Do not write full production code unless explicitly required; keep examples minimal and illustrative.
```

- [ ] **Step 4: Delete `code_snippet.md` and `debugging.md`**

```bash
git rm domain/software_engineering/prompts/code_snippet.md domain/software_engineering/prompts/debugging.md
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py::test_real_domain_prompts_are_exactly_four_strategies -v`
Expected: PASS.

- [ ] **Step 6: Verify the full domain loads cleanly**

Run: `uv run python -c "from agent.domain_config import load_domain_config; d = load_domain_config('domain/software_engineering'); print('ok', d.intent_mapping, d.orchestration.intents)"`
Expected: prints `ok` plus the 7-entry mapping and `['architecture_design', 'troubleshooting', 'code_task']`.

- [ ] **Step 7: Commit**

```bash
git add domain/software_engineering/prompts/planning.md
git commit -m "feat: add planning strategy prompt, drop code_snippet and debugging prompts"
```

---

### Task 4: Delete 5 dataset files, create `code_task.yaml`

**Files:**
- Delete: `evaluation/datasets/software_engineering/faq.yaml`
- Delete: `evaluation/datasets/software_engineering/summarization.yaml`
- Delete: `evaluation/datasets/software_engineering/performance_analysis.yaml`
- Delete: `evaluation/datasets/software_engineering/code_review.yaml`
- Delete: `evaluation/datasets/software_engineering/generate_code.yaml`
- Create: `evaluation/datasets/software_engineering/code_task.yaml`
- Test: `tests/unit/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: `agent.evaluation.dataset.load_suites` — suite name is the file stem, so deleted files disappear and the new file appears as a suite named `code_task`.
- Produces: suite `code_task` containing `se-090` (from code_review) and `se-100`, `se-101`, `se-102`, `se-103` (from generate_code), all with `intent: code_task`, `strategy: direct`, existing tiers, and `se-102` keeping `tier: full_expert` + `orchestrate: true`.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_evaluation_dataset.py`, add a test against the real committed dataset directory:

```python
def test_committed_suite_names_match_new_taxonomy():
    suites = load_suites("evaluation/datasets/software_engineering")
    names = [s.name for s in suites]
    assert "code_task" in names
    for removed in ("faq", "summarization", "performance_analysis", "code_review", "generate_code"):
        assert removed not in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_evaluation_dataset.py::test_committed_suite_names_match_new_taxonomy -v`
Expected: FAIL — old files still present, `code_task.yaml` missing.

- [ ] **Step 3: Delete the 5 dataset files**

Use `git rm` for the 5 files listed above. Case data is migrated in Tasks 5–7; no cases are lost except `se-040` (in `summarization.yaml`, deleted entirely per spec §5/§11).

- [ ] **Step 4: Create `code_task.yaml`**

Create `evaluation/datasets/software_engineering/code_task.yaml` using the real `cases:`/`expected:` schema. Move these cases verbatim (only `intent`/`strategy` change; keep `tier`, `question`, `complexity`, `orchestrate`):

```yaml
cases:
  - id: se-090
    tier: routing
    question: "Review this Python function for correctness and style: <code>def sum(a, b): return a + b  # never used</code>"
    expected: {domain: software_engineering, intent: code_task, complexity: simple, strategy: direct, orchestrate: false}
  - id: se-100
    tier: classification
    question: "Write a Python function that checks whether a string is a palindrome."
    expected: {domain: software_engineering, intent: code_task, complexity: simple, strategy: direct, orchestrate: false}
  - id: se-101
    tier: classification
    question: "Write a Python function that reads a CSV file, validates each row, and returns a summary of invalid rows."
    expected: {domain: software_engineering, intent: code_task, complexity: medium, strategy: direct, orchestrate: false}
  - id: se-102
    tier: full_expert
    question: "Build a complete CLI tool in Python with argument parsing, a config file, and unit tests."
    expected: {domain: software_engineering, intent: code_task, complexity: complex, strategy: direct, orchestrate: true}
  - id: se-103
    tier: classification
    question: "Write a Python function that reads a file, processes each line, writes results to an output file, and closes the file properly even when an error occurs."
    expected: {domain: software_engineering, intent: code_task, complexity: medium, strategy: direct, orchestrate: false}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_evaluation_dataset.py::test_committed_suite_names_match_new_taxonomy -v`
Expected: PASS.

- [ ] **Step 6: Verify the suite loads against the new domain**

Run: `uv run python -c "from agent.evaluation.dataset import load_suites; s=[x for x in load_suites('evaluation/datasets/software_engineering') if x.name=='code_task'][0]; print(len(s.cases))"`
Expected: prints `5`.

- [ ] **Step 7: Commit**

```bash
git add -A evaluation/datasets/software_engineering
git commit -m "feat: reorganize dataset to 7 intent files, add code_task.yaml"
```

---

### Task 5: Migrate `faq` cases into `concept_explain.yaml`

**Files:**
- Modify: `evaluation/datasets/software_engineering/concept_explain.yaml`
- Test: `tests/unit/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: `faq.yaml` (deleted in Task 4).
- Produces: cases `se-001`, `se-003`, `se-120` in suite `concept_explain` with `intent: concept_explain`, `strategy: teaching`, unchanged tiers. `se-120` keeps `smoke: true`.

- [ ] **Step 1: Write the failing test**

```python
def test_faq_cases_migrated_to_concept_explain():
    suites = load_suites("evaluation/datasets/software_engineering")
    cases = {c.id: c for s in suites for c in s.cases}
    for cid in ("se-001", "se-003", "se-120"):
        assert cases[cid].expected_intent == "concept_explain"
        assert cases[cid].expected_strategy == "teaching"
    assert cases["se-120"].smoke is True
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `faq.yaml` was deleted, so the ids are missing entirely.

- [ ] **Step 3: Migrate the cases**

Append to `concept_explain.yaml`:

```yaml
  - id: se-001
    tier: classification
    question: "What does HTTP status code 503 mean?"
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
  - id: se-003
    tier: classification
    question: "What is the time complexity of quicksort in the average case?"
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
  - id: se-120
    tier: classification
    smoke: true
    question: "What is a hash function?"
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_evaluation_dataset.py::test_faq_cases_migrated_to_concept_explain -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/datasets/software_engineering/concept_explain.yaml
git commit -m "feat: migrate faq cases to concept_explain"
```

---

### Task 6: Migrate `performance_analysis` cases into `troubleshooting.yaml`

**Files:**
- Modify: `evaluation/datasets/software_engineering/troubleshooting.yaml`
- Test: `tests/unit/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: `performance_analysis.yaml` (deleted in Task 4).
- Produces: cases `se-060`, `se-062`, `se-124` in suite `troubleshooting` with `intent: troubleshooting`, `strategy: analysis` (unchanged), unchanged tiers.

- [ ] **Step 1: Write the failing test**

```python
def test_performance_cases_migrated_to_troubleshooting():
    suites = load_suites("evaluation/datasets/software_engineering")
    cases = {c.id: c for s in suites for c in s.cases}
    for cid in ("se-060", "se-062", "se-124"):
        assert cases[cid].expected_intent == "troubleshooting"
        assert cases[cid].expected_strategy == "analysis"
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — the ids are missing because `performance_analysis.yaml` was deleted in Task 4.

- [ ] **Step 3: Migrate the cases**

Append to `troubleshooting.yaml`:

```yaml
  - id: se-060
    tier: routing
    question: "Analyze why my API response time degrades as concurrent users increase, and identify the bottleneck."
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: analysis, orchestrate: false}
  - id: se-062
    tier: routing
    question: "Why is a bulk INSERT of 10 million rows slower than expected, and how can it be tuned?"
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: analysis, orchestrate: false}
  - id: se-124
    tier: routing
    question: "Analyze why my API response time degrades as concurrent users increase, and identify the bottleneck."
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: analysis, orchestrate: false}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_evaluation_dataset.py::test_performance_cases_migrated_to_troubleshooting -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/datasets/software_engineering/troubleshooting.yaml
git commit -m "feat: migrate performance_analysis cases to troubleshooting"
```

---

### Task 7: Update strategy fields in surviving files

**Files:**
- Modify: `evaluation/datasets/software_engineering/learning_guide.yaml`
- Modify: `evaluation/datasets/software_engineering/architecture_design.yaml`
- Modify: `evaluation/datasets/software_engineering/troubleshooting.yaml`
- Test: `tests/unit/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: the four surviving strategy ids (Task 3).
- Produces: `learning_guide` cases `se-030`, `se-123` → `strategy: planning`; `architecture_design` cases `se-070`, `se-071`, `se-082`, `se-126`, `se-128` → `strategy: planning`; `troubleshooting` cases `se-050`, `se-051`, `se-052`, `se-053`, `se-054`, `se-140` → `strategy: analysis` (from `debugging`). All other fields (intent, tier, orchestrate, smoke) unchanged; `se-071`, `se-126`, `se-128` keep `orchestrate: true`, `se-050`/`se-052`/`se-071` keep `smoke: true`.

- [ ] **Step 1: Write the failing test**

```python
def test_strategy_fields_updated_to_surviving_set():
    suites = load_suites("evaluation/datasets/software_engineering")
    cases = {c.id: c for s in suites for c in s.cases}
    assert cases["se-030"].expected_strategy == "planning"
    assert cases["se-123"].expected_strategy == "planning"
    for cid in ("se-070", "se-071", "se-082", "se-126", "se-128"):
        assert cases[cid].expected_strategy == "planning"
    for cid in ("se-050", "se-051", "se-052", "se-053", "se-054", "se-140"):
        assert cases[cid].expected_strategy == "analysis"
    for cid in ("se-071", "se-126", "se-128"):
        assert cases[cid].expected_orchestrate is True
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — cases still use `teaching`/`analysis`/`debugging` strategies.

- [ ] **Step 3: Update strategies**

Edit the three files in place, changing only `strategy` values:
- `learning_guide.yaml`: `se-030`, `se-123` → `strategy: planning`.
- `architecture_design.yaml`: `se-070`, `se-071`, `se-082`, `se-126`, `se-128` → `strategy: planning`.
- `troubleshooting.yaml`: `se-050`, `se-051`, `se-052`, `se-053`, `se-054`, `se-140` → `strategy: analysis`.

Keep intent, tier, complexity, orchestrate, and smoke exactly as they are.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_evaluation_dataset.py::test_strategy_fields_updated_to_surviving_set -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/datasets/software_engineering/learning_guide.yaml evaluation/datasets/software_engineering/architecture_design.yaml evaluation/datasets/software_engineering/troubleshooting.yaml
git commit -m "feat: update surviving cases to the 4-strategy set"
```

---

### Task 8: Add boundary cases to `boundary.yaml`

**Files:**
- Modify: `evaluation/datasets/software_engineering/boundary.yaml`
- Test: `tests/unit/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: the 7-intent taxonomy (Task 1) and `expected` schema of `load_suites`.
- Produces: boundary cases with `tier: classification` covering the four minimal pairs from spec §12.1, plus the retained OOD case `se-110` (`expected.strategy: reject`).

- [ ] **Step 1: Write the failing test**

```python
def test_boundary_cases_cover_four_minimal_pairs():
    suites = load_suites("evaluation/datasets/software_engineering")
    boundary = next(s for s in suites if s.name == "boundary")
    by_id = {c.id: c for c in boundary.cases}
    assert by_id["se-110"].expected_strategy == "reject"
    assert by_id["se-110"].tier == "classification"
    assert {c.expected_intent for c in boundary.cases} == {
        "concept_explain", "tutorial", "code_task", "troubleshooting", "architecture_design",
    }
    assert all(c.tier == "classification" for c in boundary.cases)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — boundary.yaml only has `se-110`.

- [ ] **Step 3: Add boundary cases**

Append minimal-pair cases to `boundary.yaml` (one representative per side of each of the four pairs; ids must be unique repo-wide — use the `se-200` range):

```yaml
  - id: se-200
    tier: classification
    question: "How do I use dependency injection in a Go application?"
    expected: {domain: software_engineering, intent: tutorial, complexity: medium, strategy: teaching, orchestrate: false}
  - id: se-201
    tier: classification
    question: "Show me step by step how to add dependency injection to my Go service."
    expected: {domain: software_engineering, intent: tutorial, complexity: medium, strategy: teaching, orchestrate: false}
  - id: se-202
    tier: classification
    question: "Explain what this Go function does."
    expected: {domain: software_engineering, intent: code_task, complexity: simple, strategy: direct, orchestrate: false}
  - id: se-203
    tier: classification
    question: "My Go service crashes after several hours. Find the cause."
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: analysis, orchestrate: false}
  - id: se-204
    tier: classification
    question: "Teach me step by step how to build a REST API."
    expected: {domain: software_engineering, intent: tutorial, complexity: medium, strategy: teaching, orchestrate: false}
  - id: se-205
    tier: classification
    question: "Implement a REST API for me."
    expected: {domain: software_engineering, intent: code_task, complexity: medium, strategy: direct, orchestrate: false}
  - id: se-206
    tier: classification
    question: "My service cannot handle 10K concurrent users. Find the bottleneck."
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: analysis, orchestrate: false}
  - id: se-207
    tier: classification
    question: "Design a service that can handle 10K concurrent users."
    expected: {domain: software_engineering, intent: architecture_design, complexity: complex, strategy: planning, orchestrate: false}
```

Note: `se-200`/`se-201` cover pair A (concept_explain↔tutorial with the concept side being the existing se-010/se-121 family), `se-202`/`se-203` pair B, `se-204`/`se-205` pair C, `se-206`/`se-207` pair D. Adjust ids if any collide with cases added in other tasks.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_evaluation_dataset.py::test_boundary_cases_cover_four_minimal_pairs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/datasets/software_engineering/boundary.yaml
git commit -m "feat: add boundary minimal-pair cases for the 7-intent taxonomy"
```

---

### Task 9: Update `test_evaluation_dataset.py` committed-suite assertions

**Files:**
- Modify: `tests/unit/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: the reorganized dataset from Tasks 4–8.
- Produces: assertions that reflect the new committed suite (8 files: 7 intent files + boundary.yaml).

- [ ] **Step 1: Update `test_load_committed_software_engineering_suites`**

Change the name/intent/strategy assertions:

```python
def test_load_committed_software_engineering_suites():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    path = repo / "evaluation" / "datasets" / "software_engineering"
    suites = load_suites(str(path))
    names = [s.name for s in suites]
    assert "boundary" in names
    assert {"concept_explain", "tutorial", "learning_guide", "code_task",
            "troubleshooting", "architecture_design", "comparison"} <= set(names)
    assert all(s.domain == "software_engineering" for s in suites)
    all_cases = [c for s in suites for c in s.cases]
    assert len(all_cases) >= 20
    ids = [c.id for c in all_cases]
    assert len(ids) == len(set(ids))  # no cross-file duplication
    intents = {c.expected_intent for c in all_cases}
    assert {"concept_explain", "tutorial", "learning_guide", "code_task",
            "troubleshooting", "architecture_design", "comparison"} <= intents
    strategies = {c.expected_strategy for c in all_cases}
    assert {"direct", "teaching", "analysis", "planning"} <= strategies
    assert {"simple", "medium", "complex"} <= {c.expected_complexity for c in all_cases}
    assert any(c.expected_orchestrate for c in all_cases)
    assert any(c.expected_domain == "other" for c in all_cases)
    tiers = {c.id: c.tier for s in suites for c in s.cases}
    assert set(tiers.values()) == {"classification", "routing", "full_expert"}
    smoke = [c for s in suites for c in s.cases if c.smoke]
    assert len(smoke) == 5
    assert {c.tier for c in smoke} == {"classification", "routing", "full_expert"}
    assert {c.id for c in smoke} == {"se-110", "se-120", "se-050", "se-052", "se-071"}
```

- [ ] **Step 2: Update `test_policy_behavior_cases_present`**

Replace expected strategies:

```python
def test_policy_behavior_cases_present():
    suites = load_suites("evaluation/datasets/software_engineering")
    cases = {c.id: c for s in suites for c in s.cases}
    assert cases["se-054"].expected_strategy == "analysis"
    assert "no logs" in cases["se-054"].question
    assert cases["se-082"].expected_strategy == "planning"
    assert "monolith" in cases["se-082"].question
    assert cases["se-103"].expected_strategy == "direct"
    assert "closes the file" in cases["se-103"].question
```

- [ ] **Step 3: Run the full unit suite to verify nothing else breaks**

Run: `uv run pytest tests/unit -q`
Expected: PASS (the synthetic-fixture tests using `faq`/`debugging`/`code_snippet` in `test_chat.py`, `test_evaluation_runner.py`, `test_config.py`, `test_observability_install.py`, `test_classification.py`, `test_evaluation_diff.py` construct DomainConfig/Suite objects directly and do NOT load the committed domain, so they remain valid).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_evaluation_dataset.py
git commit -m "test: update evaluation dataset assertions for the new taxonomy"
```

---

### Task 10: Update `tests/live/test_smoke.py` fixture

**Files:**
- Modify: `tests/live/test_smoke.py`

**Interfaces:**
- Consumes: new taxonomy — the temporary evaluation dataset must use a surviving intent/strategy.
- Produces: the smoke dataset case uses `intent: concept_explain`, `strategy: teaching` (spec §15).

- [ ] **Step 1: Update the dataset fixture**

In `test_smoke_evaluation_writes_result`, change the embedded YAML:

```yaml
        '      intent: concept_explain\n'
        '      complexity: simple\n'
        '      strategy: teaching\n',
```

- [ ] **Step 2: Sanity check the test file compiles**

Run: `uv run python -m py_compile tests/live/test_smoke.py`
Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
git add tests/live/test_smoke.py
git commit -m "test: update smoke test fixture for the new taxonomy"
```

---

### Task 11: Final verification — full test suite + dataset validation

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (no API key needed).

- [ ] **Step 2: Validate the dataset loads against the new domain**

Run: `uv run python -c "from agent.domain_config import load_domain_config; from agent.evaluation.dataset import load_suites; load_domain_config('domain/software_engineering'); suites = load_suites('evaluation/datasets/software_engineering'); print(len([c for s in suites for c in s.cases]))"`
Expected: prints the total case count (33 original minus se-040 = 32, plus 8 new boundary cases = 40).

- [ ] **Step 3: Verify no stale references**

Search the repo for removed names (excluding plan/spec docs):

```bash
grep -rn "intent: faq\|intent: summarization\|intent: performance_analysis\|intent: code_review\|intent: generate_code\|strategy: debugging\|strategy: code_snippet" --include="*.yaml" --include="*.py" --include="*.md" domain/ evaluation/ tests/ agent/ | grep -v "docs/superpowers" || echo "clean"
```

Expected: prints `clean`.

- [ ] **Step 4: Final commit if anything remains**

```bash
git add -A
git commit -m "chore: finalize intent taxonomy reduction"
```

- [ ] **Step 5: Report results**

Summarize the before/after state: 11→7 intents, 5→4 strategies, 8→8 dataset files (7 intent files + boundary.yaml), total case count, and full-suite result.