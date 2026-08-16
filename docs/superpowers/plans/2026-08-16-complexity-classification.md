# Complexity Classification Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make complexity classification judge Reasoning Complexity, Scope, Trade-off, and Coordination Cost (not answer length) via a domain-configurable `complexity.yaml`, rendered into the single-call classification prompt, with dataset boundary cases and a per-level metric.

**Architecture:** A per-domain `domain/<name>/complexity.yaml` defines the three levels (simple/medium/complex) each with a description, four dimension features, positive/negative examples, and boundaries. `load_domain_config` parses it into `ComplexityPolicy`; `build_classification_prompt` renders it into the existing single-call classification prompt (missing file → today's default text). The classification dataset gains two boundary cases; `metrics.py` adds `per_complexity`.

**Tech Stack:** Python 3.13, PyYAML, pytest. Config loading follows the existing `intents.yaml` pattern in `agent/config.py`.

## Global Constraints

- Complexity levels are exactly `simple | medium | complex` (`COMPLEXITY_LEVELS` in `agent/classification.py` stays the enum source of truth).
- Classification remains a **single LLM call** (draft_v2 §3 non-goal 3). No new LLM calls.
- `validate_classification` fallback stays: invalid complexity → `medium`.
- Routing, `model_router`, `complexity_gate`, and Orchestrator behavior are **unchanged**.
- No generic shared "levels" loader; no edits to other domain directories or strategy prompts.
- Optional `complexity.yaml` missing → domain loads with `complexity=None` and renders default complexity text.
- Existing domain dirs without `complexity.yaml` must load unchanged (backward compatible).

---
## File Structure

- `agent/config.py` — add `ComplexityLevelDef`, `ComplexityPolicy` dataclasses; `DomainConfig.complexity` field; loader logic.
- `agent/classification.py` — add `build_complexity_section`; render policy in `build_classification_prompt`; wire `ClassificationService`.
- `domain/software_engineering/complexity.yaml` — new policy content.
- `evaluation/datasets/software_engineering/classification.yaml` — add se-127, se-128 boundary cases.
- `agent/evaluation/metrics.py` — add `per_complexity`.
- `agent/evaluation/report.py` — render `per_complexity` in `format_summary`.
- `agent/evaluation/diff.py` — diff `per_complexity`.
- Tests: `tests/test_config.py`, `tests/test_classification.py`, `tests/test_evaluation_metrics.py`, `tests/test_evaluation_report.py`, `tests/test_evaluation_diff.py`.

---

### Task 1: Data model + loader in `agent/config.py`

**Files:**
- Modify: `agent/config.py` (add dataclasses near `IntentDef` ~line 140; add loader logic in `load_domain_config` ~line 201)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `_read_yaml`, `_str_list`, `ConfigError` in `agent/config.py`.
- Produces:
  - `ComplexityLevelDef` dataclass: `level: str`, `description: str`, `dimensions: list[str]`, `positive_examples: list[str]`, `negative_examples: list[str]`, `boundaries: list[str]` (lists default to empty).
  - `ComplexityPolicy` dataclass: `levels: list[ComplexityLevelDef]`.
  - `DomainConfig.complexity: ComplexityPolicy | None = None`.
  - `load_domain_config(domain_dir)` reads `domain_dir/complexity.yaml`; missing file → `complexity=None`; invalid → `ConfigError`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py` (after `test_load_domain_config_bad_yaml`):

```python
from agent.config import ComplexityPolicy


def test_load_domain_config_complexity_policy(tmp_path):
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
    (base / "complexity.yaml").write_text(
        "- level: simple\n"
        "  description: single concept\n"
        "  dimensions:\n"
        "    - 'Reasoning: single step'\n"
        "    - 'Scope: single concept'\n"
        "  positive_examples:\n"
        "    - 'What is dependency injection?'\n"
        "  negative_examples:\n"
        "    - 'Design a distributed rate limiter'\n"
        "  boundaries:\n"
        "    - 'Prefer medium when multiple concepts'\n",
        encoding="utf-8",
    )
    domain = load_domain_config(str(base))
    assert isinstance(domain.complexity, ComplexityPolicy)
    assert len(domain.complexity.levels) == 1
    level = domain.complexity.levels[0]
    assert level.level == "simple"
    assert level.description == "single concept"
    assert level.dimensions == ["Reasoning: single step", "Scope: single concept"]
    assert level.positive_examples == ["What is dependency injection?"]
    assert level.negative_examples == ["Design a distributed rate limiter"]
    assert level.boundaries == ["Prefer medium when multiple concepts"]


def test_load_domain_config_complexity_missing_is_none(tmp_path):
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
    domain = load_domain_config(str(base))
    assert domain.complexity is None


def test_load_domain_config_complexity_invalid_level(tmp_path):
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
    (base / "complexity.yaml").write_text("- level: bogus\n  description: d\n",
                                          encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))
```

Note: `json`, `ConfigError`, `load_domain_config` are already imported in the test file's header.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -k complexity -v`
Expected: FAIL — `ComplexityPolicy` import error / `DomainConfig.complexity` attribute missing.

- [ ] **Step 3: Add dataclasses**

In `agent/config.py`, after `IntentDef`:

```python
@dataclass
class ComplexityLevelDef:
    level: str
    description: str
    dimensions: list[str] = field(default_factory=list)
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)


@dataclass
class ComplexityPolicy:
    levels: list[ComplexityLevelDef]
```

Add field to `DomainConfig`:

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
```

- [ ] **Step 4: Add loader logic**

In `load_domain_config`, after the intents loop (before `mapping_data`), add:

```python
    complexity = None
    complexity_path = base / "complexity.yaml"
    if complexity_path.is_file():
        complexity_data = _read_yaml(complexity_path)
        if not isinstance(complexity_data, list):
            raise ConfigError(
                f"complexity.yaml must contain a list: {complexity_path}"
            )
        levels: list[ComplexityLevelDef] = []
        for item in complexity_data:
            if not isinstance(item, dict) or not isinstance(item.get("level"), str):
                raise ConfigError(
                    f"Invalid complexity level entry in {complexity_path}: {item}"
                )
            if item["level"] not in ("simple", "medium", "complex"):
                raise ConfigError(
                    f"Unknown complexity level {item['level']!r} in {complexity_path}"
                )
            levels.append(ComplexityLevelDef(
                level=item["level"],
                description=item.get("description") or "",
                dimensions=_str_list(item.get("dimensions")),
                positive_examples=_str_list(item.get("positive_examples")),
                negative_examples=_str_list(item.get("negative_examples")),
                boundaries=_str_list(item.get("boundaries")),
            ))
        complexity = ComplexityPolicy(levels=levels)
```

Pass `complexity=complexity` in the returned `DomainConfig(...)` at the end of `load_domain_config`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -k complexity -v`
Expected: PASS (3 tests). Then run full config suite: `pytest tests/test_config.py -v`
Expected: PASS (all existing + new).

- [ ] **Step 6: Commit**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat: complexity policy data model and loader"
```

---

### Task 2: `build_complexity_section` + prompt rendering

**Files:**
- Modify: `agent/classification.py` (add `build_complexity_section`; modify `build_classification_prompt`; wire `ClassificationService.classify`)
- Test: `tests/test_classification.py`

**Interfaces:**
- Consumes: `ComplexityPolicy`, `ComplexityLevelDef` from `agent.config` (Task 1).
- Produces:
  - `build_complexity_section(policy: ComplexityPolicy | None) -> str`
  - `build_classification_prompt(name, description, intents, question, complexity=None)` — new optional 5th param.
  - `ClassificationService.classify` passes `self.domain.complexity`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_classification.py`:

```python
from agent.config import ComplexityLevelDef, ComplexityPolicy


def _complexity_policy():
    return ComplexityPolicy(levels=[
        ComplexityLevelDef(
            level="simple",
            description="single clear concept, single fact",
            dimensions=["Reasoning: single step", "Scope: single concept",
                        "Trade-off: none", "Coordination: none"],
            positive_examples=["What is dependency injection?"],
            negative_examples=["Design a distributed rate limiter"],
            boundaries=["Prefer medium over simple when multiple concepts"],
        ),
        ComplexityLevelDef(
            level="complex",
            description="multiple subsystems, multiple constraints",
            dimensions=["Reasoning: multi-step", "Scope: multiple subsystems"],
            positive_examples=["Design a distributed rate limiter for millions of QPS"],
            negative_examples=["What is dependency injection?"],
            boundaries=["Prefer complex when task decomposition is required"],
        ),
    ])


def test_build_complexity_section_renders_levels():
    section = build_complexity_section(_complexity_policy())
    assert "simple: single clear concept, single fact" in section
    assert "Reasoning: single step" in section
    assert "Trade-off: none" in section
    assert "What is dependency injection?" in section
    assert "Design a distributed rate limiter" in section
    assert "Boundary: Prefer medium over simple when multiple concepts" in section
    assert "complex: multiple subsystems, multiple constraints" in section


def test_build_complexity_section_none_renders_default():
    section = build_complexity_section(None)
    assert "short direct answer" in section


def test_build_classification_prompt_renders_complexity_policy():
    prompt = build_classification_prompt(
        "SE", "software engineering",
        [IntentDef("faq", "quick factual question")],
        "what is a hash?",
        complexity=_complexity_policy(),
    )
    assert "single clear concept, single fact" in prompt
    assert "Design a distributed rate limiter for millions of QPS" in prompt


def test_classify_passes_domain_complexity_to_prompt():
    domain = _domain()
    domain.complexity = _complexity_policy()
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
    ])
    ClassificationService(client, domain).classify("q")
    messages, model, disable_thinking, json_mode, json_schema = client.calls[0]
    assert "single clear concept, single fact" in messages[0]["content"]
```

Note: `build_classification_prompt` and `IntentDef` are already imported in the test file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_classification.py -k complexity -v`
Expected: FAIL — `build_complexity_section` not defined / signature mismatch.

- [ ] **Step 3: Implement `build_complexity_section`**

In `agent/classification.py`, add:

```python
def build_complexity_section(policy: ComplexityPolicy | None) -> str:
    if policy is None:
        return (
            "simple (short direct answer), medium (needs structured explanation), "
            "complex (large scope, multiple steps or subsystems)"
        )
    blocks: list[str] = []
    for level in policy.levels:
        lines = [f"- {level.level}: {level.description}"]
        for dim in level.dimensions:
            lines.append(f"  {dim}")
        if level.positive_examples:
            lines.append("  Positive examples:")
            lines.extend(f"    - {ex}" for ex in level.positive_examples)
        if level.negative_examples:
            lines.append("  Negative examples:")
            lines.extend(f"    - {ex}" for ex in level.negative_examples)
        for b in level.boundaries:
            lines.append(f"  Boundary: {b}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)
```

- [ ] **Step 4: Update `build_classification_prompt` and the prompt template**

Change the signature:

```python
def build_classification_prompt(
    name: str,
    description: str,
    intents: list[IntentDef],
    question: str,
    complexity: ComplexityPolicy | None = None,
) -> str:
```

Replace the hardcoded complexity line in `_CLASSIFICATION_PROMPT`:

```python
- Also judge task complexity as one of {complexity_levels}:
  {complexity_section}
```

Remove `complexity_levels` from the format call in `build_classification_prompt` and add `complexity_section=build_complexity_section(complexity)`.

- [ ] **Step 5: Wire `ClassificationService.classify`**

In `ClassificationService.classify`, change the call:

```python
        prompt = build_classification_prompt(
            self.domain.name, self.domain.description,
            list(self.domain.intents.values()), question,
            complexity=self.domain.complexity,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_classification.py -v`
Expected: PASS (all new + existing, including prompt-assertion tests).

- [ ] **Step 7: Commit**

```bash
git add agent/classification.py tests/test_classification.py
git commit -m "feat: render complexity policy into classification prompt"
```

---

### Task 3: Software Engineering `complexity.yaml` content

**Files:**
- Create: `domain/software_engineering/complexity.yaml`

**Interfaces:**
- Consumes: schema from Task 1 (`level`, `description`, `dimensions`, `positive_examples`, `negative_examples`, `boundaries`).
- Produces: the actual policy rendered by `build_complexity_section` (Task 2) and loaded by `load_domain_config` (Task 1).

- [ ] **Step 1: Write the file**

Create `domain/software_engineering/complexity.yaml`:

```yaml
- level: simple
  description: Single clear concept, single fact, or simple code change; no obvious trade-off; no multi-step reasoning
  dimensions:
    - "Reasoning: single step, recalls one concept or fact"
    - "Scope: one concept, one function, or one file"
    - "Trade-off: none or negligible"
    - "Coordination: none required"
  positive_examples:
    - "What is dependency injection?"
    - "Walk me through the 12-factor app principles, one by one, in full detail."
  negative_examples:
    - "Design a distributed rate limiter for millions of QPS with multi-region deployment."
    - "Compare mutex and RWMutex in Go."
  boundaries:
    - "Prefer medium over simple when the question connects multiple related concepts or requires reasoning across them."
- level: medium
  description: Multiple related concepts; requires some reasoning; bounded trade-offs; completable by one expert; no task decomposition
  dimensions:
    - "Reasoning: a few connected steps across related concepts"
    - "Scope: a feature or a focused area of a system"
    - "Trade-off: bounded, a few options with clear consequences"
    - "Coordination: a single expert suffices"
  positive_examples:
    - "Compare mutex and RWMutex in Go."
    - "Why does my API response time degrade as concurrent users increase?"
  negative_examples:
    - "What is dependency injection?"
    - "Design a distributed rate limiter for millions of QPS with multi-region deployment."
  boundaries:
    - "Prefer complex over medium when the task spans multiple subsystems or needs task decomposition."
- level: complex
  description: Multiple subsystems; multiple constraints; architecture-level decision; multiple viable approaches; clear trade-offs; requires task decomposition; needs multiple independent analysis perspectives
  dimensions:
    - "Reasoning: multi-step reasoning across subsystems and constraints"
    - "Scope: multiple subsystems or system-wide architecture"
    - "Trade-off: significant, multiple viable approaches with real consequences"
    - "Coordination: requires task decomposition or multiple analysis perspectives"
  positive_examples:
    - "Design a distributed rate limiter for millions of QPS with multi-region deployment."
  negative_examples:
    - "Walk me through the 12-factor app principles, one by one, in full detail."
    - "What is dependency injection?"
  boundaries:
    - "Prefer complex when the question demands architecture-level decisions under multiple constraints."
```

- [ ] **Step 2: Verify loader accepts it**

Run: `python -c "from agent.config import load_domain_config; d=load_domain_config('domain/software_engineering'); print(len(d.complexity.levels)); print([l.level for l in d.complexity.levels])"`
Expected: prints `3` and `['simple', 'medium', 'complex']`.

- [ ] **Step 3: Commit**

```bash
git add domain/software_engineering/complexity.yaml
git commit -m "feat: software engineering complexity policy"
```

---

### Task 4: Dataset boundary cases

**Files:**
- Modify: `evaluation/datasets/software_engineering/classification.yaml`

**Interfaces:**
- Consumes: dataset schema in `agent/evaluation/dataset.py` (`cases`, `id`, `question`, `expected`).
- Produces: two boundary cases that exercise §6.5.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation_dataset.py` (append after existing tests):

```python
def test_complexity_boundary_cases_present():
    suites = load_suites("evaluation/datasets/software_engineering")
    cases = {c.id: c for s in suites for c in s.cases}
    assert cases["se-127"].expected_complexity == "simple"
    assert cases["se-128"].expected_complexity == "complex"
    assert "12-factor" in cases["se-127"].question
    assert "distributed rate limiter" in cases["se-128"].question
```

Check the existing import header of `tests/test_evaluation_dataset.py` for `load_suites` before writing.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation_dataset.py -k complexity_boundary -v`
Expected: FAIL — KeyError `se-127`.

- [ ] **Step 3: Add the two cases**

Append to `evaluation/datasets/software_engineering/classification.yaml`:

```yaml
  - id: se-127
    question: "Walk me through the 12-factor app principles, one by one, in full detail."
    expected: {domain: software_engineering, intent: concept_explain, complexity: simple, strategy: teaching, orchestrate: false}
    answer_quality: true
  - id: se-128
    question: "Design a distributed rate limiter for millions of QPS with multi-region deployment."
    expected: {domain: software_engineering, intent: architecture_design, complexity: complex, strategy: analysis, orchestrate: true}
    answer_quality: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation_dataset.py -k complexity_boundary -v`
Expected: PASS. Then run `pytest tests/test_evaluation_dataset.py -v` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/datasets/software_engineering/classification.yaml tests/test_evaluation_dataset.py
git commit -m "feat: complexity boundary cases in classification dataset"
```

---

### Task 5: `per_complexity` metric

**Files:**
- Modify: `agent/evaluation/metrics.py`
- Test: `tests/test_evaluation_metrics.py`

**Interfaces:**
- Consumes: existing `_accuracy`, `is_in_domain` in `metrics.py`.
- Produces: `metrics["classification"]["per_complexity"]` — dict `{level: accuracy}` keyed in expected order.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation_metrics.py`:

```python
def test_per_complexity_accuracy():
    cases = [_case("a", complexity="simple"), _case("b", complexity="medium"),
             _case("c", complexity="complex")]
    results = [
        _result(cases[0], complexity="simple"),
        _result(cases[1], complexity="complex"),  # wrong
        _result(cases[2], complexity="complex"),
    ]
    m = _m(cases, results)
    pc = m["classification"]["per_complexity"]
    assert pc["simple"] == 1.0
    assert pc["medium"] == 0.0
    assert pc["complex"] == 1.0
    assert list(pc) == ["simple", "medium", "complex"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation_metrics.py::test_per_complexity_accuracy -v`
Expected: FAIL — `per_complexity` KeyError.

- [ ] **Step 3: Implement**

In `compute_metrics`, next to `per_intent`, add:

```python
    per_complexity: dict[str, list[bool]] = {}
    per_complexity_order: list[str] = []
```

In the `if expected_in:` block, mirror the intent logic for complexity:

```python
            if r.complexity == c.expected_complexity:
                per_complexity.setdefault(c.expected_complexity, []).append(True)
            else:
                per_complexity.setdefault(c.expected_complexity, []).append(False)
            if c.expected_complexity not in per_complexity_order:
                per_complexity_order.append(c.expected_complexity)
```

After `per_intent_accuracy`:

```python
    per_complexity_accuracy = {}
    for level in per_complexity_order:
        marks = per_complexity[level]
        per_complexity_accuracy[level] = _accuracy(sum(marks), len(marks))
```

Add `"per_complexity": per_complexity_accuracy,` to the classification dict (after `per_intent`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluation_metrics.py -v`
Expected: PASS (all existing + new).

- [ ] **Step 5: Commit**

```bash
git add agent/evaluation/metrics.py tests/test_evaluation_metrics.py
git commit -m "feat: per-complexity accuracy metric"
```

---

### Task 6: Render `per_complexity` in report and diff

**Files:**
- Modify: `agent/evaluation/report.py` (`format_summary`)
- Modify: `agent/evaluation/diff.py` (`diff_runs`)
- Test: `tests/test_evaluation_report.py`, `tests/test_evaluation_diff.py`

**Interfaces:**
- Consumes: `metrics["classification"]["per_complexity"]` from Task 5.
- Produces: human-readable `per_complexity` output in summary and diff.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluation_report.py`:

```python
def test_format_summary_has_per_complexity():
    text = format_summary(_record())
    assert "per_complexity" in text
    assert "simple:" in text
```

Add to `tests/test_evaluation_diff.py`:

```python
def test_diff_shows_per_complexity():
    a = _run("a", 1.0, 0.5, 4.0, 100)
    a["metrics"]["classification"]["per_complexity"] = {"simple": 1.0, "medium": 0.5}
    b = _run("b", 1.0, 0.5, 4.0, 100)
    b["metrics"]["classification"]["per_complexity"] = {"simple": 1.0, "medium": 1.0}
    text = diff_runs(a, b)
    assert "per_complexity" in text
    assert "medium" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation_report.py::test_format_summary_has_per_complexity tests/test_evaluation_diff.py::test_diff_shows_per_complexity -v`
Expected: FAIL — no `per_complexity` text.

- [ ] **Step 3: Update `format_summary`**

In `agent/evaluation/report.py`, after the `per_intent` block:

```python
    if cls["per_complexity"]:
        lines.append("  per_complexity:")
        for level, acc in cls["per_complexity"].items():
            lines.append(f"    {level}: {_fmt_accuracy(acc)}")
```

- [ ] **Step 4: Update `diff_runs`**

In `agent/evaluation/diff.py`, after the top-level classification keys loop:

```python
    pca = clsa.get("per_complexity") or {}
    pcb = clsb.get("per_complexity") or {}
    if pca or pcb:
        lines.append("  per_complexity:")
        for level in sorted(set(pca) | set(pcb)):
            lines.append(f"    {level}: {_diff_value(pca.get(level), pcb.get(level))}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_evaluation_report.py tests/test_evaluation_diff.py -v`
Expected: PASS (all new + existing).

- [ ] **Step 6: Commit**

```bash
git add agent/evaluation/report.py agent/evaluation/diff.py tests/test_evaluation_report.py tests/test_evaluation_diff.py
git commit -m "feat: render per-complexity in report and diff"
```

---

### Task 7: Full regression

**Files:** none new.

**Interfaces:** consumes all prior tasks.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: PASS (all tests, including domain-agnostic, evaluation CLI, and smoke tests).

- [ ] **Step 2: Verify the software-engineering domain loads and renders**

Run: `python -c "from agent.config import load_domain_config; from agent.classification import build_classification_prompt, build_complexity_section; d=load_domain_config('domain/software_engineering'); print(build_complexity_section(d.complexity))"`
Expected: prints the three-level policy text.

- [ ] **Step 3: Commit any stragglers**

```bash
git status --short
git add -A
git commit -m "chore: complexity classification regression"
```
(If nothing changed, skip the commit.)