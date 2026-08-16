# Complexity Classification 优化 — Design

Date: 2026-08-16
Status: Draft

## 1. Goal

Per `draft_v2.md` §6 (P0): complexity classification no longer depends on
"answer length". Instead it is judged by four dimensions — **Reasoning
Complexity, Scope, Trade-off, Coordination Cost**. The complexity decision
policy becomes domain-configurable, is rendered into the single-call
classification prompt, the classification dataset gains explicit boundary
cases, and evaluation surfaces which level is confused.

This mirrors the completed §5 (intent taxonomy) cycle: enriched definitions →
runtime rendering → dataset → evaluation.

## 2. Approach

**Approach A** (chosen): a per-domain `complexity.yaml` under
`domain/<name>/`, parsed into a `ComplexityPolicy` at config load, and
rendered into the classification prompt at runtime. Missing file → fall back to
today's default one-line complexity text (backward compatible). Each level
(simple/medium/complex) carries a description, the four dimension features,
positive/negative examples, and boundary rules. The four dimensions are
judgment *features*, not scored dimensions — the model judges the overall
level in the existing single call.

Rationale:
- Per-domain configurability (chosen over a global rubric) supports the
  upcoming legal domain: adding a domain is just adding files, zero code.
- Rejected a generic "levels.yaml" loader shared with `intents.yaml`: the two
  schemas differ (complexity has `dimensions`, intents do not), a unified
  loader would distort both. Defer shared extraction until real duplication.
- Rejected dimension scoring (score each dimension, map to level): weights are
  inherently fuzzy, it splits one fuzzy judgment into four plus a mapping rule,
  and conflicts with the single-call design goal (draft_v2 §3 non-goal 3).

## 3. Component Changes

### 3.1 `agent/config.py` — data model

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
    levels: list[ComplexityLevelDef]  # ordered simple → medium → complex
```

`DomainConfig` gains an optional field `complexity: ComplexityPolicy | None`.
`load_domain_config` reads `complexity.yaml` from the domain directory:
- File missing → `complexity=None` (fall back to default prompt text).
- File present but structurally invalid (non-list, entry missing `level`, or
  unknown level value) → raise `ConfigError`, mirroring the `intents.yaml`
  loader pattern.

`COMPLEXITY_LEVELS` in `agent/classification.py` stays the single source of
truth for the level enum; the policy levels must match it.

### 3.2 `domain/software_engineering/complexity.yaml` — policy content

A three-level list aligned with §6.2–6.4:

- **simple**: single clear concept, single fact, simple code change, no
  obvious trade-off, no multi-step reasoning.
- **medium**: multiple related concepts, some reasoning required, bounded
  trade-offs, completable by one expert, no task decomposition.
- **complex**: multiple subsystems, multiple constraints, architecture-level
  decision, multiple viable approaches, clear trade-offs, requires task
  decomposition, needs multiple independent analysis perspectives.

Each level includes the four `dimensions` (Reasoning/Scope/Trade-off/
Coordination) as judgment features, `positive_examples`, `negative_examples`,
and `boundaries`. Examples echo the dataset boundary cases (se-127/se-128).

### 3.3 `agent/classification.py` — prompt rendering

- Add `build_complexity_section(policy: ComplexityPolicy | None) -> str`:
  - With policy: render each level as
    `- <level>: <description>` followed by `Dimensions: ...`,
    `Positive examples:`, `Negative examples:`, `Boundary:` lines.
  - Without policy: render the current default text
    (`simple (short direct answer), medium (needs structured explanation),
    complex (large scope, multiple steps or subsystems)`).
- `build_classification_prompt` gains an optional `complexity` parameter;
  the hardcoded complexity rule line is replaced by
  `build_complexity_section(...)`.
- `ClassificationService.classify` passes `self.domain.complexity`.
- Behavior unchanged: single-call classification (§3 non-goal 3),
  `validate_classification` fallback (invalid complexity → `medium`) intact.

### 3.4 Dataset — boundary cases

Add to `evaluation/datasets/software_engineering/classification.yaml`
(following the existing `se-1xx` sequence):

| ID | Question | Expected complexity | Validates |
|----|----------|--------------------:|-----------|
| se-127 | "Walk me through the 12-factor app principles, one by one, in full detail." | simple | Long answer but no reasoning/trade-off → simple (length ≠ complexity) |
| se-128 | "Design a distributed rate limiter for millions of QPS with multi-region deployment." | complex | Short question but multi-subsystem/multi-constraint → complex (short-but-complex) |

Notes:
- se-126 (architecture_design, complex) already exercises "short-but-complex";
  se-128 adds the §6.4 canonical example.
- Focus on the two chosen boundary directions (length≠complexity,
  short-but-complex). simple/medium and medium/complex boundary cases are
  deferred unless evaluation shows confusion there.

### 3.5 `agent/evaluation/metrics.py` — per-level metric

Add `per_complexity` (modeled on `per_intent`): accuracy per expected
complexity level, so evaluation surfaces which level is misclassified.
`complexity_accuracy` (overall) is unchanged.

`agent/evaluation/report.py` `format_summary` renders `per_complexity`
alongside `per_intent` (same block style). `agent/evaluation/diff.py` keeps the
top-level `complexity_accuracy` diff line and additionally diffs
`per_complexity` per level when present in both runs.

### 3.6 Tests

- `test_config.py`: `complexity.yaml` parses into `ComplexityPolicy`;
  missing file → `complexity=None`; structurally invalid → `ConfigError`.
- `test_classification.py`: `build_complexity_section` renders dimensions,
  examples, boundaries; no policy → default text; prompt integration.
- Metrics test: `per_complexity` statistics correct per level.
- Regression: all existing evaluation/classification tests pass.

## 4. Acceptance Criteria (§6.5)

- Stable simple/medium/complex separation → `per_complexity` + boundary cases.
- Obvious architecture questions classified complex → se-126/se-128.
- Not auto-classified complex because the answer would be long → se-127.
- Evaluation dataset has explicit boundary cases → se-127/se-128.
- `complexity_accuracy` does not regress (target: no decrease).

## 5. Out of Scope

- Changes to routing, `model_router`, `complexity_gate`, or Orchestrator
  behavior (classification accuracy improves routing automatically).
- A generic shared "levels" loader across intents/complexity.
- Other domain directories / strategy prompts.
- simple/medium and medium/complex dataset boundary cases (deferred).