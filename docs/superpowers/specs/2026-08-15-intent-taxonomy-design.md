# Software Engineering Intent Taxonomy 优化 — Design

Date: 2026-08-15
Status: Draft

## 1. Goal

Per `draft_v2.md` §5 (P0): do not add new intents. Instead, clarify the
definition and boundaries of the existing 11 intents, feed the enriched
definitions into runtime classification, expand the classification dataset to
cover the main boundary pairs, and let evaluation surface intent confusion.

Current intents:

```
concept_explain
tutorial
learning_guide
faq
summarization
troubleshooting
comparison
performance_analysis
architecture_design
generate_code
code_review
```

## 2. Approach

Minimal enrichment (Approach A): the enriched per-intent definition lives in
`intents.yaml` as the single source of truth, is parsed into `IntentDef` at
config load, and is rendered (examples + boundaries) into the classification
prompt at runtime. No global decision-policy section; no new intents; no
intent-to-strategy mapping changes.

Boundary modeling uses a simple string list per intent (`boundaries: [string,
...]`) as in the §5.2 example, rendered verbatim into the prompt.

## 3. Component Changes

### 3.1 `agent/config.py` — data model

Extend the `IntentDef` dataclass (currently `id`, `description`) with three
optional fields, all defaulting to empty lists for backward compatibility:

```python
@dataclass
class IntentDef:
    id: str
    description: str
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
```

Extend the loader loop (`load_domain_config`) to read `positive_examples`,
`negative_examples`, and `boundaries` from each intent dict, defaulting to `[]`
when absent. Existing domains without these fields load unchanged — no
migration required.

### 3.2 `intents.yaml` — enriched definitions

Each of the 11 intents is expanded to:

```yaml
- id: concept_explain
  description: Explain a concept, design rationale, or "why" question
  positive_examples:
    - "Why does dependency injection reduce coupling?"
  negative_examples:
    - "My application crashes with this error."
  boundaries:
    - "Prefer concept_explain over faq when the user wants understanding, not just a short factual answer."
```

Every intent has a `description`. Every intent has at least one
`positive_example`. The high-conflict intents (the four boundary pairs) get
`negative_examples` and `boundaries`.

### 3.3 `agent/classification.py` — prompt rendering

`build_classification_prompt` currently takes `intent_items: list[tuple[str, str]]`
and renders `- {iid}: {desc}`. Change it to accept the full `IntentDef`
objects and render a per-intent block:

```
- concept_explain: Explain a concept, design rationale, or "why" question
  Positive examples:
    - Why does dependency injection reduce coupling?
  Negative examples:
    - My application crashes with this error.
  Boundary: Prefer concept_explain over faq when the user wants understanding, not just a short factual answer.
```

Rendering rules:
- Render `positive_examples` / `negative_examples` only when non-empty.
- Render each `boundaries` entry prefixed with `Boundary:`.
- Leave the rest of the prompt (in_domain / complexity rules, JSON output
  format) unchanged.

Update the single caller (`ClassificationService.classify`) to pass `IntentDef`
objects instead of `(id, description)` tuples. Update existing classification
tests that assert on the rendered prompt.

### 3.4 `evaluation/datasets/software_engineering/classification.yaml`

Add ~8 boundary test cases covering the four high-conflict pairs, each labeled
with the correct intent. Follow the existing `se-1xx` id sequence. A boundary
case that gets misclassified drops `intent_accuracy` / `per_intent` and thereby
surfaces the confusion.

| Pair | Boundary case | Correct intent |
|------|---------------|----------------|
| FAQ vs Concept | existing se-120 "What is a hash function?" | `faq` |
| FAQ vs Concept | "Why does dependency injection reduce coupling?" | `concept_explain` |
| Tutorial vs Learning Guide | "Walk me through setting up a React project step by step." | `tutorial` |
| Tutorial vs Learning Guide | "Create a month-long learning path from zero to competent in Python." | `learning_guide` |
| Troubleshooting vs Performance Analysis | existing se-140 "slow after caching layer" | `troubleshooting` |
| Troubleshooting vs Performance Analysis | "Analyze why API response time degrades as concurrency increases." | `performance_analysis` |
| Comparison vs Architecture Design | "Compare gRPC vs REST for microservices." | `comparison` |
| Comparison vs Architecture Design | "Design the architecture of a system handling millions of events per second." | `architecture_design` |

### 3.5 Evaluation metrics

No metric code changes. `intent_accuracy` and `per_intent` already exist in
`agent/evaluation/metrics.py`; the expanded dataset is what makes confusion
visible.

### 3.6 Tests

- `test_config.py`: loader parses the new fields; defaults to empty lists when
  absent.
- `test_classification.py`: prompt renders examples and boundaries; a definition
  with no optional fields renders without those sections.

## 4. Acceptance Criteria (§5.4)

- Every intent has a clear `description`. ✓ (all 11 in `intents.yaml`)
- Every intent has at least one `positive_examples`. ✓
- Every high-conflict intent has `negative_examples`. ✓ (four pairs)
- Classification dataset covers the main boundaries. ✓ (~8 cases across 4 pairs)
- Evaluation can detect intent confusion. ✓ (via `intent_accuracy`/`per_intent`
  on the expanded dataset)

## 5. Out of Scope

- Adding new intents.
- Changing intent→strategy mapping (`intent_mapping.yaml`).
- A global decision-policy prompt section (Approach B) — deferred.
- Complexity classification (§6).
- Editing strategy prompts / other domain directories.
