from __future__ import annotations

import time
import warnings
from typing import Iterator

from agent.llm import ChatResult, LLMError

from .tracing import TraceStore, current_phase, current_trace_id, now_millis


class TracedLLMClient:
    """Transparent LLMClient wrapper that records token usage and latency."""

    def __init__(self, inner, store: TraceStore):
        self._inner = inner
        self._store = store

    @property
    def model(self) -> str:
        return self._inner.model

    def _base_event(self) -> dict:
        return {
            "type": "llm_call",
            "trace_id": current_trace_id(),
            "phase": current_phase(),
            "ts": now_millis(),
        }

    def _write(self, event: dict) -> None:
        try:
            self._store.write(event)
        except Exception as e:  # noqa: BLE001 - degrade, never break business
            warnings.warn(f"observability: failed to record llm_call: {e}")

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs) -> ChatResult:
        started = time.perf_counter()
        try:
            result = self._inner.chat_completion(
                messages, model=model, temperature=temperature, **kwargs
            )
        except LLMError as e:
            ev = self._base_event()
            ev.update({"model": model or self._inner.model, "prompt_tokens": None,
                       "completion_tokens": None, "total_tokens": None,
                       "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                       "status": "error", "error": str(e)})
            self._write(ev)
            raise
        ev = self._base_event()
        ev.update({"model": result.model, "prompt_tokens": result.prompt_tokens,
                   "completion_tokens": result.completion_tokens,
                   "total_tokens": result.total_tokens,
                   "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                   "status": "ok", "error": None})
        self._write(ev)
        return result

    def chat_completion_stream(self, messages, *, model=None, temperature=0.7, **kwargs) -> Iterator[str]:
        # Not observed in v1: the main flow never uses streaming.
        yield from self._inner.chat_completion_stream(
            messages, model=model, temperature=temperature, **kwargs
        )
