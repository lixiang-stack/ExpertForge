# Design: LLMClient.chat_completion returns ChatResult (decouple wrappers from _usage_local)

Date: 2026-08-15

## Problem

`LLMClient.chat_completion()` returns only `str`. Token usage is stashed in a private
thread-local (`_usage_local`), forcing two wrappers — `RecordingClient`
(`agent/evaluation/runner.py`) and `TracedLLMClient` (`agent/observability/client.py`) —
to poke at a private attribute to read usage. This is fragile (not a contract) and
records the *requested* model rather than the *actual* model from the response.

## Goal

Return a rich `ChatResult` from `chat_completion` that carries the text, the actual
model (from `resp.model`), and token usage. Remove the `_usage_local` thread-local
entirely. Both wrappers read usage from the return value.

## Design

### `agent/llm.py`

Introduce:

```python
@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_tokens: int = 0
```

`chat_completion(...) -> ChatResult`:
- `text = resp.choices[0].message.content or ""`
- `model = resp.model or (model or self.model)`
- tokens from `resp.usage` (defaults to 0 when absent)
- `cache_tokens` from `resp.usage.prompt_tokens_details.cached_tokens` when an int, else 0
- on `OpenAIError` → raise `LLMError` (unchanged)
- remove `_usage_local` (thread-local) entirely

### Callers (production)

- `agent/classification.py:164` — `text = res.text` before JSON parse
- `agent/orchestrator.py:119` — `text = res.text` before JSON parse
- `agent/orchestrator.py:145,174,181` (`_worker`/`_aggregate`/`_direct_answer`) — return `.text`
- `agent/strategy.py:29` — `process` returns `.text`
- `agent/evaluation/judge.py:76` — `text = res.text` before `parse_scorecard`
- `agent/evaluation/runner.py` (`RecordingClient`) — read fields from returned
  `ChatResult`; drop the `getattr(self._inner, "_usage_local", ...)` reads
- `agent/observability/client.py` (`TracedLLMClient`) — read fields from returned
  `ChatResult`; drop `_usage_tokens()` private-attr read

### Tests

- `tests/test_llm.py` — assert return type fields instead of `_usage_local`;
  delete thread-isolation test; add `resp.model` assertion
- All `FakeClient.chat_completion` doubles that are consumed via `.text` must
  return a `ChatResult` (or the caller). ~10 fakes across
  `tests/test_{classification,router,chat,strategy,orchestrator,repl,agent_cli,domain_agnostic,evaluation_*.py}`
- `RecordingClient` / `TracedLLMClient` tests updated to the new read path

## Out of scope

- `chat_completion_stream` (unchanged; not observed in v1)
- Observability event schema (same fields, now sourced from return value)

## Testing strategy

TDD: update `test_llm.py` expectations first, then implementation, then each
module's fake, then full regression (current suite: 192 passed / 5 skipped).

## Risks

- Fake doubles scattered across ~10 files: each consumed result must be a
  `ChatResult`. Mitigation: every production caller uses `.text`, so any fake
  still returning `str` fails loudly in tests and is fixed in-place.