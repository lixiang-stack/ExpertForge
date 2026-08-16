# Intent Taxonomy Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clarify the 11 software-engineering intents' definitions and boundaries by enriching `intents.yaml`, feeding examples + boundaries into the runtime classification prompt, and expanding the classification dataset so evaluation surfaces intent confusion.

**Architecture:** `intents.yaml` is the single source of truth. `agent/config.py` parses each intent's `description`, `positive_examples`, `negative_examples`, `boundaries` into an extended `IntentDef`. `agent/classification.py` renders those fields into the classification prompt at runtime. `evaluation/datasets/software_engineering/classification.yaml` gains ~6 new boundary cases across the four high-conflict pairs (FAQ vs Concept, Tutorial vs Learning Guide, Troubleshooting vs Performance Analysis, Comparison vs Architecture Design). Evaluation metrics (`intent_accuracy`, `per_intent`) already exist and surface confusion once boundary cases exist.

**Tech Stack:** Python 3.10+, PyYAML, pytest. No new dependencies.

## Global Constraints

- Do NOT add new intents; the 11 intents are unchanged.
- Do NOT change `intent_mapping.yaml`, `strategies.yaml`, or any strategy prompt.
- `IntentDef` new fields are optional and default to empty lists — existing domains must load unchanged.
- Follow existing TDD pattern: write failing test → verify fail → implement → verify pass → commit.
- Project uses `uv`; run tests with `uv run pytest <path>`.
- No code comments unless they explain an existing pattern being preserved.

---

### Task 1: Extend `IntentDef` and loader in `agent/config.py`

**Files:**
- Modify: `agent/config.py:137-141` (dataclass), `agent/config.py:210-217` (loader)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `IntentDef(id: str, description: str)` positional constructor (used in `tests/test_config.py` and `tests/test_classification.py`).
- Produces: `IntentDef(id, description, positive_examples=[], negative_examples=[], boundaries=[])` where the three list fields are `list[str]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_load_domain_config_intent_definition_fields(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: concept_explain\n"
        "  description: explain a concept\n"
        "  positive_examples:\n"
        "    - Why does DI reduce coupling?\n"
        "  negative_examples:\n"
        "    - My app crashes.\n"
        "  boundaries:\n"
        "    - Prefer concept_explain over faq when the user wants understanding.\n",
        encoding="utf-8",
    )
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    domain = load_domain_config(str(base))
    intent = domain.intents["concept_explain"]
    assert intent.positive_examples == ["Why does DI reduce coupling?"]
    assert intent.negative_examples == ["My app crashes."]
    assert intent.boundaries == [
        "Prefer concept_explain over faq when the user wants understanding."
    ]


def test_load_domain_config_intent_fields_default_empty(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    domain = load_domain_config(str(base))
    intent = domain.intents["faq"]
    assert intent.positive_examples == []
    assert intent.negative_examples == []
    assert intent.boundaries == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k "intent_definition_fields or intent_fields_default_empty" -v`
Expected: FAIL — `IntentDef` has no attribute `positive_examples`.

- [ ] **Step 3: Implement**

Add `from dataclasses import field` to the imports in `agent/config.py` (verify whether it is already imported first). Update the dataclass:

```python
@dataclass
class IntentDef:
    id: str
    description: str
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
```

Update the loader loop:

```python
        iid = item["id"]
        intents[iid] = IntentDef(
            id=iid,
            description=item.get("description") or "",
            positive_examples=_str_list(item.get("positive_examples")),
            negative_examples=_str_list(item.get("negative_examples")),
            boundaries=_str_list(item.get("boundaries")),
        )
```

Add a helper near `_read_yaml`:

```python
def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -k "intent_definition_fields or intent_fields_default_empty" -v`
Expected: PASS. Then run the full file:
Run: `uv run pytest tests/test_config.py -v`
Expected: all PASS (existing tests still construct `IntentDef` positionally with two args — covered by defaults).

- [ ] **Step 5: Commit**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat: extend IntentDef with examples and boundaries"
```

---

### Task 2: Enrich `intents.yaml` with full definitions

**Files:**
- Modify: `domain/software_engineering/intents.yaml`

**Interfaces:**
- Consumes: schema from Task 1 (`positive_examples`, `negative_examples`, `boundaries` lists).
- Produces: enriched YAML that the Task 1 loader parses; descriptions must remain compatible with `intent_mapping.yaml` intent ids.

- [ ] **Step 1: Replace the file content**

Write the full enriched file:

```yaml
- id: concept_explain
  description: Explain a software engineering concept, mechanism, or design rationale ("why" questions)
  positive_examples:
    - "Why does dependency injection reduce coupling?"
    - "How does garbage collection work?"
  negative_examples:
    - "My application crashes with this error."
    - "What is the syntax for this function?"
  boundaries:
    - "Prefer concept_explain over faq when the user wants understanding of a mechanism, not a short factual answer."
- id: tutorial
  description: Walk through a topic step by step so the user can follow along
  positive_examples:
    - "Walk me through setting up a React project step by step."
    - "Show me how to write a basic REST endpoint."
  negative_examples:
    - "Create a month-long learning path to go from zero to competent in Python."
    - "What is a hash function?"
  boundaries:
    - "Prefer tutorial over learning_guide when the user wants to follow concrete steps now, not a long-term plan."
- id: learning_guide
  description: Create a structured learning path or study plan
  positive_examples:
    - "Create a month-long learning path to go from zero to competent in Python."
    - "What order should I study databases, networking, and operating systems in?"
  negative_examples:
    - "Walk me through setting up a React project step by step."
    - "Why does dependency injection reduce coupling?"
  boundaries:
    - "Prefer learning_guide over tutorial when the user wants a plan or curriculum over a period of time."
- id: faq
  description: Quick factual or definitional question with a short answer
  positive_examples:
    - "What is a hash function?"
    - "What does HTTP status code 503 mean?"
  negative_examples:
    - "Why does dependency injection reduce coupling?"
    - "Design the architecture of a system that must handle millions of events per second."
  boundaries:
    - "Prefer faq over concept_explain when the user only needs a short factual or definitional answer."
- id: summarization
  description: Summarize content the user provided
  positive_examples:
    - "Summarize the key ideas of this article into three bullet points: <article>...</article>"
  negative_examples:
    - "Compare gRPC and REST for microservices."
    - "Explain how dependency injection works."
  boundaries:
    - "Prefer summarization when the user provides content and asks to condense it; do not use it for outside knowledge."
- id: troubleshooting
  description: Diagnose and resolve a specific reported problem
  positive_examples:
    - "My service became slow right after deploying the new caching layer. Why might that be?"
    - "My database connection pool is exhausted under load and new requests hang. How do I diagnose it?"
  negative_examples:
    - "Analyze why API response time degrades as concurrent users increase and identify the bottleneck."
    - "Why does dependency injection reduce coupling?"
  boundaries:
    - "Prefer troubleshooting over performance_analysis when the user reports a concrete failure or symptom to fix."
- id: comparison
  description: Compare two or more technologies, approaches, or options
  positive_examples:
    - "Compare gRPC vs REST for microservices."
    - "SQL vs NoSQL for an analytics workload?"
  negative_examples:
    - "Design the architecture of a system that must handle millions of events per second."
    - "Why does dependency injection reduce coupling?"
  boundaries:
    - "Prefer comparison when the user explicitly contrasts options; prefer architecture_design when the user asks to design a system."
- id: performance_analysis
  description: Analyze performance characteristics, bottlenecks, or scaling behavior
  positive_examples:
    - "Analyze why API response time degrades as concurrent users increase and identify the bottleneck."
    - "What is the performance profile of this query at scale?"
  negative_examples:
    - "My service became slow right after deploying the new caching layer. Why might that be?"
    - "Review this Python function for correctness and style."
  boundaries:
    - "Prefer performance_analysis over troubleshooting when the user asks to analyze a performance trend or bottleneck, not fix a reported failure."
- id: architecture_design
  description: Design a system, component, or architecture to meet requirements
  positive_examples:
    - "Design the architecture of a system that must handle millions of events per second."
    - "How should I structure a multi-tenant SaaS backend?"
  negative_examples:
    - "Compare gRPC vs REST for microservices."
    - "What is a hash function?"
  boundaries:
    - "Prefer architecture_design when the user asks to design or structure a system; prefer comparison when they only contrast options."
- id: generate_code
  description: Write code to accomplish a task
  positive_examples:
    - "Write a Python function that retries an HTTP request with exponential backoff."
    - "Generate a SQL query to find duplicate emails."
  negative_examples:
    - "Review this Python function for correctness and style."
    - "Why does dependency injection reduce coupling?"
  boundaries:
    - "Prefer generate_code when the user asks for code to be produced; prefer code_review when they provide code and ask for feedback."
- id: code_review
  description: Review code the user provided for correctness, style, or improvement
  positive_examples:
    - "Review this Python function for correctness and style: <code>def sum(a, b): return a + b</code>"
    - "What issues do you see in this pull request diff?"
  negative_examples:
    - "Write a Python function that retries an HTTP request with exponential backoff."
    - "What is a hash function?"
  boundaries:
    - "Prefer code_review when the user provides code and asks for feedback; prefer generate_code when they ask for new code."
```

- [ ] **Step 2: Verify the YAML loads and every intent has required fields**

Run a quick check:

```bash
uv run python - <<'EOF'
from agent.config import load_domain_config
d = load_domain_config("domain/software_engineering")
missing = [i for i, idef in d.intents.items() if not idef.description or not idef.positive_examples]
assert not missing, f"intents missing description/positive_examples: {missing}"
print("intents:", sorted(d.intents))
for iid, idef in d.intents.items():
    print(f"  {iid}: +{len(idef.positive_examples)} -{len(idef.negative_examples)} b{len(idef.boundaries)}")
EOF
```

Expected: prints all 11 intents, each with ≥1 positive example; no assertion error.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (the enriched YAML is only consumed by classification prompt rendering, which Task 3 changes).

- [ ] **Step 4: Commit**

```bash
git add domain/software_engineering/intents.yaml
git commit -m "feat: enrich software engineering intent definitions"
```

---

### Task 3: Render examples and boundaries in the classification prompt

**Files:**
- Modify: `agent/classification.py:60-72` (`build_classification_prompt`), `agent/classification.py:132-135` (call site)
- Test: `tests/test_classification.py`

**Interfaces:**
- Consumes: `IntentDef` from Task 1 (with `positive_examples`, `negative_examples`, `boundaries`).
- Produces: `build_classification_prompt(name: str, description: str, intents: list[IntentDef], question: str) -> str` — signature changes from `intent_items: list[tuple[str, str]]` to `intents: list[IntentDef]`.
- Note: `build_classification_prompt` is not directly imported by any test today (verified via grep). The only caller is `ClassificationService.classify` at `agent/classification.py:132-135`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classification.py`:

```python
from agent.classification import build_classification_prompt


def _rich_intent():
    return IntentDef(
        id="concept_explain",
        description="explain a concept",
        positive_examples=["Why does DI reduce coupling?"],
        negative_examples=["My app crashes."],
        boundaries=["Prefer concept_explain over faq when the user wants understanding."],
    )


def test_build_classification_prompt_renders_examples_and_boundaries():
    prompt = build_classification_prompt(
        "SE",
        "software engineering",
        [_rich_intent()],
        "why DI?",
    )
    assert "concept_explain: explain a concept" in prompt
    assert "Why does DI reduce coupling?" in prompt
    assert "My app crashes." in prompt
    assert "Boundary: Prefer concept_explain over faq when the user wants understanding." in prompt


def test_build_classification_prompt_omits_empty_sections():
    prompt = build_classification_prompt(
        "SE",
        "software engineering",
        [IntentDef("faq", "quick factual question")],
        "what is a hash?",
    )
    assert "faq: quick factual question" in prompt
    assert "Positive examples" not in prompt
    assert "Negative examples" not in prompt
    assert "Boundary:" not in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_classification.py -k "renders_examples_and_boundaries or omits_empty_sections" -v`
Expected: FAIL — `build_classification_prompt` not imported / prompt renders only `- id: desc`.

- [ ] **Step 3: Implement**

Update `build_classification_prompt` (keep the existing `_CLASSIFICATION_PROMPT` template unchanged):

```python
def build_classification_prompt(
    name: str,
    description: str,
    intents: list[IntentDef],
    question: str,
) -> str:
    lines: list[str] = []
    for idef in intents:
        header = f"- {idef.id}: {idef.description}"
        if not (idef.positive_examples or idef.negative_examples or idef.boundaries):
            lines.append(header)
            continue
        lines.append(header)
        if idef.positive_examples:
            lines.append("  Positive examples:")
            lines.extend(f"    - {ex}" for ex in idef.positive_examples)
        if idef.negative_examples:
            lines.append("  Negative examples:")
            lines.extend(f"    - {ex}" for ex in idef.negative_examples)
        for b in idef.boundaries:
            lines.append(f"  Boundary: {b}")
    intents_block = "\n".join(lines)
    return _CLASSIFICATION_PROMPT.format(
        name=name,
        description=description,
        intents=intents_block,
        complexity_levels=", ".join(COMPLEXITY_LEVELS),
        question=question,
    )
```

Add `IntentDef` to the import from `.config` at the top of `agent/classification.py`:

```python
from .config import DomainConfig, IntentDef
```

Update the call site in `ClassificationService.classify` (`agent/classification.py:132-139`):

```python
        intent_ids = list(self.domain.intents)
        prompt = build_classification_prompt(
            self.domain.name, self.domain.description, list(self.domain.intents.values()), question
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_classification.py -v`
Expected: all PASS, including the existing `test_classify_single_call_returns_all_fields` (it only asserts `"intent" in content`).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/classification.py tests/test_classification.py
git commit -m "feat: render intent examples and boundaries in classification prompt"
```

---

### Task 4: Add boundary cases to the classification dataset

**Files:**
- Modify: `evaluation/datasets/software_engineering/classification.yaml`

**Interfaces:**
- Consumes: dataset schema `cases: [{id, question, expected: {domain, intent, complexity, strategy, orchestrate}, answer_quality}]`; the runner in `agent/evaluation/runner.py` reads `expected.intent`, `expected.complexity`, etc.
- Produces: `se-1xx` cases that target the four boundary pairs. `intent_accuracy`/`per_intent` in `agent/evaluation/metrics.py` use `expected_intent` only — no metric change needed.

- [ ] **Step 1: Append boundary cases**

Read the current file, then append before the trailing marker so `cases:` stays the single top-level key. Add these six cases after the existing `se-140` entry (keep the existing `se-110`, `se-120`, `se-140`):

```yaml
  - id: se-121
    question: "Why does dependency injection reduce coupling?"
    expected: {domain: software_engineering, intent: concept_explain, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-122
    question: "Walk me through setting up a React project step by step."
    expected: {domain: software_engineering, intent: tutorial, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-123
    question: "Create a month-long learning path to go from zero to competent in Python."
    expected: {domain: software_engineering, intent: learning_guide, complexity: medium, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-124
    question: "Analyze why my API response time degrades as concurrent users increase, and identify the bottleneck."
    expected: {domain: software_engineering, intent: performance_analysis, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-125
    question: "Compare gRPC vs REST for microservices."
    expected: {domain: software_engineering, intent: comparison, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
  - id: se-126
    question: "Design the architecture of a system that must handle millions of events per second."
    expected: {domain: software_engineering, intent: architecture_design, complexity: complex, strategy: analysis, orchestrate: true}
    answer_quality: true
```

- [ ] **Step 2: Validate the dataset loads**

Load-check with the actual API (`load_suites(path) -> list[Suite]`; each `Suite` has `.name`, `.domain`, `.cases`; each `EvalCase` has `.id`, `.expected_domain`, `.expected_intent`):

```bash
uv run python - <<'EOF'
from agent.evaluation.dataset import load_suites
suites = load_suites("evaluation/datasets/software_engineering/classification.yaml")
suite = suites[0]
ids = [c.id for c in suite.cases]
print(suite.name, len(ids), ids)
assert len(ids) == 9, ids
assert all(c.expected_intent for c in suite.cases if c.expected_domain == "software_engineering")
EOF
```

Expected: prints `classification 9 [...]` including `se-121`..`se-126`; no assertion error.

- [ ] **Step 3: Run the evaluation dataset tests**

Run: `uv run pytest tests/test_evaluation_dataset.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add evaluation/datasets/software_engineering/classification.yaml
git commit -m "feat: add intent boundary cases to classification dataset"
```

---

### Task 5: Full-suite verification

**Files:**
- No source changes.

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Run the classifier smoke test against the real domain config**

Run: `uv run pytest tests/test_integration.py tests/test_smoke.py -v`
Expected: all PASS (confirms the enriched intents.yaml still drives routing end-to-end).

- [ ] **Step 3: Run lint/typecheck if configured**

Check `pyproject.toml` for a lint/typecheck config (none found in dev deps beyond pytest). If a tool exists, run it; otherwise skip.

- [ ] **Step 4: Commit any stragglers**

```bash
git status --short
```

If clean, nothing to commit. If there are unrelated changes, leave them; do not commit.

---

## Self-Review

**Spec coverage:**
- §3.1 data model → Task 1 ✓
- §3.2 enriched intents.yaml → Task 2 ✓
- §3.3 prompt rendering → Task 3 ✓
- §3.4 dataset boundary cases → Task 4 ✓
- §3.5 no metric changes → verified, no task needed ✓
- §3.6 tests → Tasks 1 and 3 ✓
- §5.4 acceptance criteria → description (Task 2), positive examples (Task 2), negative examples on high-conflict intents (Task 2), dataset covers boundaries (Task 4), evaluation detects confusion via existing `intent_accuracy`/`per_intent` (Task 4) ✓

**Placeholder scan:** All steps contain concrete code and exact YAML; no TBD/TODO; the only conditional is the dataset-load check in Task 4 Step 2, which explicitly instructs inspecting `dataset.py` if attribute names differ.

**Type consistency:** `IntentDef` fields (`positive_examples`, `negative_examples`, `boundaries`) are defined in Task 1 and consumed identically in Tasks 2 and 3. `build_classification_prompt` signature (`list[IntentDef]`) is defined in Task 3; Task 3's own test and call site use it consistently. Dataset case IDs `se-121`..`se-126` are used only in Task 4. Intent ids match `intent_mapping.yaml` (verified against the existing file).