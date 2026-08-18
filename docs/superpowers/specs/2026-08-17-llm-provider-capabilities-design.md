# LLM Provider Capabilities Abstraction — Design Spec

Date: 2026-08-17
Status: Approved (brainstorming)

## Problem

The agent currently supports DeepSeek and Gemini, both accessed through their OpenAI-compatible endpoints. PR #15 added Gemini compatibility, but exposed three structural weaknesses:

1. **Structured output is provider-bound.** `json_schema` paths exist in code (`build_classification_schema`, `_planner_schema`) but are commented out; every structured call site hardcodes `json_mode=True`. The mechanism (json_schema vs json_object) is chosen by callers, not by what the provider supports. Classification should not be coupled to one provider's implementation.

2. **JSON parsing is regex-greedy.** Three copies of `re.search(r"\{.*\}", text, re.DOTALL)` + `json.loads` exist (`classification._parse`, `orchestrator._parse_json`, `judge.parse_scorecard`). The greedy match is used as the primary parse strategy; it mis-handles prose containing a second `{`.

3. **`disable_thinking` leaks a provider-specific param into global config.** The `thinking` toggle is a DeepSeek-specific field; Gemini rejects it with `400`. Modeling it as a top-level config key is wrong. Separately, the user question is duplicated across the system prompt and a user message at three call sites (classification/planner/judge), wasting input tokens.

## Design Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Abstraction boundary | Single OpenAI-compat adapter layer. Capability declarations + structured-output negotiation + parsing layers on top of one transport. No multi-SDK backends; no protocol interface with a single implementation (YAGNI — extract the interface when a second non-compat backend actually appears). |
| 2 | How capabilities are determined | Declared in config with validation: `provider` and `provider_capabilities` are required together, keys checked against a known set, values must be booleans; invalid config fails fast with `ConfigError`. No code-side provider table, no `base_url` sniffing, no runtime probing. |
| 3 | `disable_thinking` placement | Capability-gated: per-call semantic parameter stays (`disable_thinking=True` on structured paths), the client sends the `thinking` field only when the provider declares `supports_thinking_toggle`. Top-level config key removed. |
| 4 | Question duplication | Remove the question from system prompts at classification/planner/judge; keep it only in the user message. Saves ~one question-worth of input tokens per call and forces a user message to exist. |
| 5 | `json_object` support | Treated as a universal default for OpenAI-compat targets (DeepSeek, Gemini, OpenAI all accept `response_format: {"type": "json_object"}`; Gemini verified live via the judge path). Not a configurable capability. `json_schema` is the only structured-output capability that is configured. |
| 6 | `requires_user_message` | Not a configurable capability. Every call site always passes a user message, so the client enforces it with an unconditional guard (a missing user message is a bug, not a provider choice). |
| 7 | Negotiation outcomes | Only `json_schema` / `json_object` / no-request. There is no capability-fallback "none": `json_object` is universal, so a structured-output request always resolves to a real mechanism. The only "none" is a caller that requested no structured output (plain-answer paths), represented by returning `None`. |

## Architecture

Three small layers, each independently testable, all on top of the existing `openai.OpenAI` transport:

```
classification / orchestrator._plan / judge        (callers state intent only)
        │  json_schema=schema | json_mode=True | disable_thinking=True
        ▼
LLMClient.chat_completion                           (single adapter)
        │  negotiate_structured_output(caps, ...) → json_schema | json_object | None
        │  unconditional user-message guard
        │  thinking field gated by supports_thinking_toggle
        ▼
openai.OpenAI (OpenAI-compat transport)

parse_json(text)                                    (parsing layer)
        │  json.loads(whole text)
        ▼
dict | None      (unparseable → None → caller degradation)
```

### New modules

**`agent/capabilities.py`**

```python
@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str                                # e.g. "deepseek" | "gemini" | user-chosen name
    supports_json_schema: bool = False
    supports_thinking_toggle: bool = False       # accepts extra_body thinking field
    supports_tool_call: bool = False             # reserved; no tool-call flow implemented

KNOWN_CAPABILITY_KEYS = (
    "supports_json_schema",
    "supports_thinking_toggle",
    "supports_tool_call",
)
```

No `detect_provider`, no default table, no merge logic: capabilities are user-declared config data, validated at load time. `config.example.json` ships recommended tables for deepseek/gemini as a reference.

**`agent/negotiate.py`**

```python
def negotiate_structured_output(
    caps, *, json_mode: bool, json_schema: dict | None
) -> str | None:
    # json_object is universal; json_schema is preferred when configured.
    if json_schema is not None:
        return "json_schema" if caps.supports_json_schema else "json_object"
    if json_mode:
        return "json_object"
    return None  # caller requested no structured output
```

Pure local decision, no network, no probing, single request emitted.

**`agent/parsing.py`**

- `parse_json(text: str) -> dict | None` — strip then `json.loads` the whole text; return the object when it is a dict, else `None`. Structured-output modes (json_schema/json_object) guarantee pure JSON, so there is no extraction fallback: unparseable output is treated as an error (`None` → caller degradation). No greedy regex and no brace-matching scanner.

### `LLMClient` changes

- Constructor: drop `disable_thinking`; add `provider: str = ""`, `capability_overrides: dict | None = None`; compute `self.capabilities = ProviderCapabilities(provider=provider or "unknown", **capability_overrides or {})`. There are no code-side defaults to merge against — the config is the source of truth.
- `chat_completion`: keep signature (`json_mode`, `json_schema`, `disable_thinking` per-call params). Internally:
  - Unconditional user-message guard: if `messages` has no `role == "user"` entry, raise `LLMError` with an actionable message (fail fast instead of a Gemini 400).
  - mode = `negotiate_structured_output(...)`; when non-`None`, build `response_format` accordingly (`json_schema` wrapper uses a fixed generic name like `"structured_output"` — the identifier is irrelevant to the provider, only the schema matters — with `schema`/`strict: False`; `json_object` → `{"type": "json_object"}`); when `None`, omit `response_format` entirely.
  - `extra_body={"thinking": {"type": "disabled"}}` only when `disable_thinking and caps.supports_thinking_toggle`.
- `chat_completion_stream` unchanged (no structured output involved).

### Call-site intent changes

| Call site | Before | After |
|---|---|---|
| `classification.classify` | `json_mode=True` | `json_schema=schema` |
| `orchestrator._plan` | `json_mode=True` | `json_schema=_planner_schema()` |
| `judge.score` | `json_mode=True` | unchanged |

Effective behavior: DeepSeek (no `json_schema`) degrades to `json_object`, identical to today; Gemini uses real schema-constrained output.

### Parsing call-site changes

- `classification._parse` → delete, use `parse_json`.
- `orchestrator._parse_json` → delete, use `parse_json`.
- `judge.parse_scorecard` → extract object via `parse_json`; keep the 1–5 integer dimension validation.

### Prompt dedup (question only in user message)

- `_CLASSIFICATION_PROMPT`: remove the `User question: {question}` line; `build_classification_prompt` drops the `question` parameter.
- `_PLANNER_PROMPT`: remove the `User question: {question}` line; `_plan` keeps `question` for the user message.
- `_JUDGE_PROMPT`: remove the `Question: {question}` line; answer/reference stay in system; user message carries the question.

### Config changes

- `AgentConfig`: remove `disable_thinking`; add `provider: str = ""` and `provider_capabilities: dict[str, bool]` (required together).
- `load_config` validation (fail fast with `ConfigError`):
  - `provider` (non-empty str) and `provider_capabilities` (dict) must both be present; if only one is present, error.
  - Every key in `provider_capabilities` must be in `KNOWN_CAPABILITY_KEYS`; every value must be a boolean; otherwise error with an actionable message.
- `config.example.json`: drop `disable_thinking`; ship recommended `provider` + `provider_capabilities` blocks matching the example's `base_url` (deepseek) as a reference; users edit the block to match their own provider.
- User-local `config.json` (git-ignored) gains the same two keys (updated during implementation); any stale `disable_thinking` key is ignored.

### Wiring

- `agent/agent_cli.py` and `agent/evaluation/__main__.py`: `LLMClient(...)` calls replace `disable_thinking=config.disable_thinking` with `provider=config.provider, capability_overrides=config.provider_capabilities`.

## Error Handling

- Provider 400s surface as `LLMError` (unchanged).
- Invalid capability config raises `ConfigError` at startup (missing/inconsistent `provider`/`provider_capabilities`, unknown keys, non-boolean values).
- The unconditional user-message guard raises `LLMError` with an actionable message (a missing user message is a bug in a call site, not a provider choice) before hitting the network.
- `parse_json` returns `None` on unparseable output (treated as an error); callers degrade via their existing validation paths (e.g. `validate_classification` produces an "Unreliable classification" result; planner degrades to direct; judge returns `None`).

## Testing

1. `tests/test_capabilities.py` — `ProviderCapabilities` construction and `KNOWN_CAPABILITY_KEYS` contents.
2. `tests/test_negotiate.py` — json_schema requested + configured → `json_schema`; json_schema requested + not configured → `json_object`; json_mode → `json_object`; no request → `None`.
3. `tests/test_parsing.py` — pure JSON, nested object, non-dict → `None`, invalid → `None`, empty/`None`/whitespace → `None`, prose-wrapped → `None` (unparseable is treated as an error).
4. `tests/test_llm.py` — mock the OpenAI client: assert `response_format`/`extra_body` are gated by the configured capabilities (thinking sent when `supports_thinking_toggle`, omitted otherwise; schema negotiation correct; `json_schema` requested but unconfigured degrades to `json_object`; unconditional user-message guard raises).
5. `tests/test_classification.py`, `tests/test_orchestrator.py`, `tests/test_evaluation_judge.py` — callers request `json_schema` intent (classification/planner); system prompt no longer contains the question; user message contains it; judge keeps `json_mode`.
6. `tests/test_config.py` — `provider`/`provider_capabilities` parsing and validation (missing-only-one, unknown key, non-boolean value all raise `ConfigError`); `disable_thinking` key removal.
7. `tests/test_agent_cli.py`, `tests/test_evaluation_cli.py` — constructor-argument assertions updated to `provider`/`capability_overrides`.
8. Full unit suite regression; live Gemini smoke (`AGENT_API_KEY`, `tests/test_smoke.py`).

## Non-Goals

- Tool-call flow (only the capability field is declared).
- Multi-SDK native backends and a protocol interface with a single implementation.
- Runtime capability probing.
- Any model-quality/performance parity across providers (that is the model-routing layer's concern).