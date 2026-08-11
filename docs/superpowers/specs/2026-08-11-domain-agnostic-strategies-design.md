# Domain-Agnostic Strategies Design (Decouple Processors from Software Engineering)

**Date:** 2026-08-11
**Status:** Design agreed; pending user review
**Source requirement:** Review feedback — the agent's strategy/processor layer is
hardcoded to software-engineering concepts (`direct`/`teaching`/`debugging`/
`analysis`/`code_snippet`), so swapping in a new expert domain would make that code
completely unusable.

## 1. Goal

Make the strategy layer **fully data-driven and domain-agnostic**. Today the only
thing that differs between the five processor subclasses is the answer `structure`
string, but that string (and the strategy whitelist) lives in Python code:

- `agent/processors/registry.py:11` hardcodes `PROCESSOR_CLASSES`, so a domain that
  declares a custom strategy in `strategies.yaml` gets no processor and hits
  `chat.py:38` ("No processor for strategy").
- Each subclass hardcodes SE-flavored structures (e.g. debugging = "Problem
  analysis / Possible causes / Verification steps / Fix suggestions").
- `router.py:9` hardcodes `DEFAULT_STRATEGY = "direct"`.

The strategy abstraction is already mostly data-driven (`strategies.yaml` declares
id/model/complexity_gate, `intents.yaml` + `intent_mapping.yaml` map intents to
strategies, `prompts/{sid}.md` provide templates with `{name}`/`{description}`/
`{structure}` placeholders). The remaining code-bound piece is the **structure** and
the **whitelist**. This spec removes both.

Approach chosen: **Fully data-driven** — delete the `agent/processors/` package, use
fully self-contained prompt files (no placeholders at all — no `{name}`/
`{description}`/`{structure}`), and carry the default strategy as a `default: true`
flag in `strategies.yaml`.

## 2. Config format changes

### 2.1 Self-contained prompt files

`domain/software_engineering/prompts/{sid}.md` no longer contain the `{structure}`
placeholder. Each file's `{structure}` line is replaced by the answer structure text
currently hardcoded in the corresponding processor subclass:

- `direct.md` → (no structure; `DirectAnswerProcessor.structure` is empty).
- `teaching.md` → the teaching structure ("Concept / Why it is designed this way /
  How it works / Concrete example / Common misconceptions / Summary").
- `debugging.md` → the debugging structure ("Problem analysis / Possible causes /
  Verification steps / Fix suggestions / Best practices").
- `analysis.md` → the analysis structure ("Comparison dimensions / Key differences /
  Trade-offs / Recommendation").
- `code_snippet.md` → the code-snippet structure ("Approach / Code snippet / Key
  points and caveats / How to extend or adapt it").

The prompts are now **fully self-contained**: there are no placeholders at all.
Each SE prompt file replaces the old `{name}`/`{description}` placeholders with the
domain's literal name ("Software Engineering") and description (from `domain.json`).
`Strategy.build_system_prompt()` performs no template injection — it returns the
prompt text verbatim. The `domain_name`/`domain_description` constructor arguments
are dropped from `Strategy`.

### 2.2 `strategies.yaml` gains a `default` flag

```yaml
direct:
  default: true
  complexity_gate: false
teaching:
  complexity_gate: true
debugging:
  complexity_gate: true
analysis:
  complexity_gate: true
code_snippet:
  complexity_gate: true
```

- `StrategyDef` gains `default: bool = False`.
- `load_domain_config` validates that **exactly one** strategy has `default: true`;
  zero or multiple raises `ConfigError`.
- `DomainConfig` gains `default_strategy: str` (the id of the default strategy),
  resolved at load time. The `Router` reads it directly.

### 2.3 Unchanged config behavior

- `intents.yaml`, `intent_mapping.yaml`, and per-strategy prompt loading
  (`prompts/{sid}.md`) are unchanged.
- `unsupported_complex.md` is still loaded eagerly (backward compatibility with
  existing domain dirs); it remains unused at runtime.

## 3. New module: `agent/strategy.py`

`agent/processors/` (base.py, direct.py, teaching.py, debugging.py, analysis.py,
code_snippet.py, registry.py, `__init__.py`) is **deleted** and replaced by a single
file `agent/strategy.py`:

```python
class Strategy:
    def __init__(self, strategy_id: str, prompt_template: str):
        ...

    def build_system_prompt(self) -> str:
        # returns prompt_template verbatim — no placeholder injection

    def build_messages(self, history, question, *, max_turns=20) -> list[dict]:
        # same as today's Processor.build_messages

    def process(self, client, question, history, *, model=None) -> str:
        # same as today's Processor.process

def build_registry(domain: DomainConfig) -> dict[str, Strategy]:
    return {
        sid: Strategy(sid, domain.prompts[sid])
        for sid in domain.strategies
    }
```

- `strategy_id` and `structure` concepts are removed; `Strategy` has zero domain
  knowledge and performs no template injection.
- `build_registry` iterates `domain.strategies` (data-driven). No `PROCESSOR_CLASSES`
  whitelist: every strategy declared in `strategies.yaml` gets a `Strategy`.

## 4. Router changes (`agent/router.py`)

- Delete `DEFAULT_STRATEGY = "direct"`.
- `route()` uses `self.domain.default_strategy` as the fallback for unmapped intents:
  `strategy = intent_mapping.get(intent_id, domain.default_strategy)`.
- Routing, complexity gate, and orchestrate-flag logic are otherwise unchanged.

## 5. Wiring changes

### `agent/chat.py`

- Import `build_registry` from `agent.strategy` instead of `agent.processors.registry`.
- Delete the `"No processor for strategy"` error branch (`chat.py:36-38`) — every
  declared strategy now has a processor by construction. `respond()` calls
  `self.processors[route.strategy].process(...)` directly.

### `agent/orchestrator.py`

- `build_registry` import path updated to `agent.strategy`.
- `_strategy_context`, `_plan`, `_worker`, `_aggregate`, `_direct_answer` unchanged —
  they already work through `{name}`/`{description}` and the strategy prompt and are
  inherently domain-agnostic.

### `agent/config.py`

- `StrategyDef` gains `default: bool = False`.
- `DomainConfig` gains `default_strategy: str`.
- `load_domain_config` parses the `default` flag and validates exactly-one-default.

## 6. Error handling

- **Missing default strategy (zero or multiple)** → `ConfigError` at domain load,
  before any LLM call.
- **Declared strategy with missing prompt file** → existing `ConfigError` from
  `load_domain_config` (unchanged).
- LLM errors in `Strategy.process` and Orchestrator stages propagate as today
  (no swallowing).

## 7. Testing

### Rename `tests/test_processors.py` → `tests/test_strategy.py`

- `build_system_prompt` returns the prompt text verbatim (no `{name}`/`{description}`/
  `{structure}` substitution; assert a placeholder is never left unresolved and the
  self-contained structure text is present in the prompt).
- `build_messages` (history truncation, message shape) and `process` (delegates to
  client) behave as today.

### `tests/test_config.py`

- Update fixtures to the new `strategies.yaml` format (default flag).
- New tests: zero defaults → `ConfigError`; multiple defaults → `ConfigError`;
  `DomainConfig.default_strategy` resolves to the flagged strategy.

### `tests/test_router.py`

- Use `domain.default_strategy` as the fallback; assert unmapped intent falls back
  to the flagged strategy (not the hardcoded `"direct"`).

### `tests/test_chat.py` / `tests/test_repl.py` / `tests/test_agent_cli.py` /
`tests/test_orchestrator.py`

- Update imports and fake-domain fixtures to the new format (strategy defaults,
  generic `Strategy`).

### New dogfood test (proof of domain-agnosticism)

- Add a synthetic non-SE domain fixture (e.g. `finance`) with custom strategy ids
  (`advise`, `risk_assessment`), custom intents/intent mapping, and self-contained
  prompt files. Assert the full path — routing → registry → answer — works with **no
  code changes**. This is the regression guard that would have failed before the
  refactor (`chat.py:38` "No processor for strategy").

## 8. Docs

- Update `README.md`: drop the processor/strategy-classes description; describe
  strategies as data (`strategies.yaml` + `prompts/{sid}.md`), including the
  `default` flag.
- Migrate `domain/software_engineering/` config files (prompts + strategies.yaml).
- Add this spec; then an implementation plan under `docs/superpowers/plans/`.

## 9. Success criteria

1. `uv run pytest -q` all green.
2. `agent/processors/` no longer exists; `agent/strategy.py` is the only strategy
   module.
3. `grep -rn "PROCESSOR_CLASSES\|DEFAULT_STRATEGY\|No processor for strategy"`
   returns no matches.
4. A domain with fully custom strategy ids routes and answers with zero code changes
   (covered by the dogfood test).
5. SE domain behavior unchanged: same strategies, same answer structures, same
   orchestration.
