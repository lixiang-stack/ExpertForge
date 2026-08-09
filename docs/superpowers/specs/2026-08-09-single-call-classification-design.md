# Single-Call Classification Service Design

**Date:** 2026-08-09
**Status:** Design agreed; pending user review before implementation
**Source requirement:** `draft_v1.md` sections 6.1, 6.2, 7, 8, 14, 17.1, 17.4, 19

## 1. Goal

Bring the current three-call classification (Domain, Intent, Complexity issued as
three separate LLM calls in `Router.route`) in line with `draft_v1.md`: classify all
three dimensions in a **single LLM call** through a unified **ClassificationService**,
with a **schema-enforced structured output** and a dedicated **validation layer** that
fall back to safe defaults without retrying.

Out of scope (kept for later iterations): complexity-based model routing (§12) and the
Orchestrator pipeline (§13). The `RouteResult` shape must stay unchanged so `Chat`,
`repl`, and `agent_cli` need zero edits.

## 2. Components

### 2.1 New module: `agent/classification.py`

```
ClassificationService                    (public)
  ├── __init__(client, domain)
  ├── classify(question) -> ClassificationResult     (one LLM call)
  ├── _build_prompt(question, degrade)                (semantics + rules)
  └── _validate(result) -> ClassificationResult      (field-level fallback)

ClassificationResult (dataclass)         (public)
  in_domain: bool
  intent: str | None
  complexity: str | None
  reason: str
```

- `ClassificationService` owns the single-call logic. It reads everything it needs from
  the existing `DomainConfig` (`name`, `description`, `intents`), so `domain.json`,
  `intents.yaml`, `strategies.yaml`, and `intent_mapping.yaml` are **not modified**.
- `Agent/classifier.py` is **deleted** along with its three `classify_*` functions and
  the three legacy prompt constants.

### 2.2 LLM client: `agent/llm.py` extension

`chat_completion` gains `json_schema: dict | None = None`:

- when provided, sends
  `response_format={"type": "json_schema", "json_schema": {"name": "classification_result", "schema": <schema>, "strict": False}}`
- otherwise behaves exactly as today (`json_mode` → `{"type": "json_object"}`)
- `json_schema` and `json_mode` are mutually exclusive; `json_schema` wins.

## 3. JSON Schema (built from config, not hardcoded)

```json
{
  "type": "object",
  "properties": {
    "in_domain":  { "type": "boolean" },
    "intent":     { "type": ["string", "null"], "enum": [<all intent ids from intents.yaml>, null] },
    "complexity": { "type": ["string", "null"], "enum": ["simple", "medium", "complex", null] },
    "reason":     { "type": "string" }
  },
  "required": ["in_domain", "intent", "complexity", "reason"]
}
```

- `intent` enum is generated from `domain.intents` keys (config-driven per `draft_v1.md` §7).
- `intent`/`complexity` are **nullable**, because they are meaningless when
  `in_domain=false` (user decision: nullable, not required always).
- The prompt carries semantics (domain intent descriptions and classification rules);
  the schema carries structure. They remain decoupled (§8, §17.4).

## 4. Degradation path (JSON Schema → json_object)

1. Try a single call with `json_schema`.
2. If the provider rejects/errors on `response_format=json_schema` (capability problem,
   not a classification failure), **degrade once**: retry the same prompt with
   `json_mode=True` plus a brief JSON output instruction block appended to the prompt
   (field names, enums, nullability). This preserves `prompt/schema` separation on the
   happy path and falls back to prompt-embedded schema only when needed.
3. If that too fails (auth/network/server), re-raise `LLMError` — do not swallow the
   failure, do not return a fake result.

Deployment note: this degradation is specifically about API capability, and is distinct
from validation fallback below.

## 5. Validation and fallback (never retries)

Parsing happens with the existing regex + `json.loads` approach. Validation is per-field:

| Case                                            | Result                                     |
|-------------------------------------------------|--------------------------------------------|
| Not valid JSON / not an object                  | `ClassificationResult(in_domain=False, reason="Unreliable classification: ...")`; routes to reject |
| `in_domain` not a bool                          | same fallback as above                     |
| `in_domain=true`, `intent` not in config ids    | `intent=None`                              |
| `in_domain=true`, `complexity` not in enum      | `complexity="medium"`                      |
| `in_domain=false`, invalid intent/complexity    | left as returned by the model; ignored downstream |

Notes:
- No retry. `ClassificationService.classify` returns a result object whose fields are
  already valid for the next stage.
- `complexity` never exceeds `"simple"`/`"medium"`/`"complex"` after validation.

## 6. Router integration (`agent/router.py`)

- Construct `ClassificationService(self.client, self.domain)` in `Router.__init__`
  (or lazily in `route`).
- `route()` replaces the three `classify_question` / `classify_intent` /
  `classify_complexity` calls with a single `service.classify(question)`.
- Mapping to the existing `RouteResult`:
  | Service result                  | `RouteResult`                     |
  |---------------------------------|-----------------------------------|
  | `in_domain=false`               | `in_domain=False, strategy="reject", reject_reason=result.reason` |
  | `in_domain=true`, `intent=None` | `in_domain=True, strategy=DEFAULT_STRATEGY ("direct")` |
  | `in_domain=true`                | strategy from `intent_mapping`, `complexity` from result (None → no gate trigger) |

- The complexity-gate logic (`complexity == "complex"` → `COMPLEX_UNSUPPORTED`) stays.
- `classifier_model` from `AgentConfig` is still passed for the classification call.

`Chat`, `repl`, `agent_cli` unchanged.

## 7. Testing

Rewrite `tests/test_classifier.py` → `tests/test_classification.py`:

- FakeClient gains `json_schema=None` parameter (and `json_mode`), records calls for
  assertions.
- `test_classify_one_call`: one response returns all four fields; assert exactly 1 call
  and `json_schema is not None`.
- `test_out_of_domain_accepts_null`: `{"in_domain": false, "intent": null, "complexity": null, "reason": "..."}` passes validation, returns those nulls.
- Field fallback cases (invalid JSON garbage, non-bool `in_domain`, intent not in
  allowed set, complexity not in enum → medium).
- Degrade path: fake client raises `LLMError` when `json_schema` set → service retries
  with `json_mode=True` → if both fail, raises `LLMError`.
- `tests/test_router.py`: FakeClient now returns a single combined object; update the
  response fixtures used in each test case (one object instead of three).
- `tests/test_llm.py`: add `json_schema` passthrough → `response_format` JSON-schema
  case; default (no `json_schema`) unchanged.
- `tests/test_chat.py`, `tests/test_repl.py`, `tests/test_agent_cli.py`: extend each
  FakeClient with a `json_schema` param; assertions otherwise unchanged.

Run `uv run pytest -q` (expect all green) and the no-key smoke
(`env -u AGENT_API_KEY uv run python -m agent --ask "..."`).

## 8. Docs

- Update `README.md` classification module mentions.
- Add this spec; then add an implementation plan under `docs/superpowers/plans/`.

## 9. Success criteria

1. `uv run pytest -q` all green after each committed task.
2. Classification is exactly **1** LLM call on the happy path.
3. `json_schema` used when the provider supports it; JSON degraded otherwise.
4. Invalid/unsupported outputs fall back within `ClassificationService`, never reaching
   `Router` in an invalid shape; no unbreakable crash.
5. `Chat`, `repl`, `agent_cli` are untouched.
6. Schema built from `domain.intents`, so adding/removing an intent needs no code change.