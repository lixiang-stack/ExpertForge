import pytest

from agent.llm import ChatResult, LLMError
from agent.observability.client import TracedLLMClient
from agent.observability.tracing import TraceStore, phase, trace_span


class FakeInner:
    def __init__(self, usage=None, error=None, model="m"):
        self.model = model
        self._usage = usage
        self._error = error
        self.last_kwargs = None

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        self.last_kwargs = (model, messages)
        if self._error:
            raise self._error
        u = self._usage
        return ChatResult(
            text="the answer",
            model=model or self.model,
            prompt_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
            completion_tokens=getattr(u, "completion_tokens", 0) if u else 0,
            total_tokens=getattr(u, "total_tokens", 0) if u else 0,
            cache_tokens=0,
        )


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


def _make_store(tmp_path):
    return TraceStore(tmp_path / "obs")


def test_records_ok_call_with_usage(tmp_path):
    store = _make_store(tmp_path)
    inner = FakeInner(usage=_Usage(10, 5))
    traced = TracedLLMClient(inner, store)
    with trace_span() as tid, phase("classification"):
        result = traced.chat_completion([{"role": "user", "content": "hi"}], model="low-a")

    assert result.text == "the answer"
    assert inner.last_kwargs[0] == "low-a"
    calls = store.trace_llm_calls(tid)
    assert len(calls) == 1
    ev = calls[0]
    assert ev["status"] == "ok"
    assert ev["phase"] == "classification"
    assert ev["model"] == "low-a"
    assert ev["prompt_tokens"] == 10
    assert ev["completion_tokens"] == 5
    assert ev["total_tokens"] == 15
    assert ev["latency_ms"] >= 0


def test_records_error_and_reraises(tmp_path):
    store = _make_store(tmp_path)
    inner = FakeInner(error=LLMError("boom"))
    traced = TracedLLMClient(inner, store)
    with trace_span() as tid:
        with pytest.raises(LLMError):
            traced.chat_completion([{"role": "user", "content": "hi"}])

    ev = store.trace_llm_calls(tid)[0]
    assert ev["status"] == "error"
    assert "boom" in ev["error"]
    assert ev["prompt_tokens"] is None


def test_records_usage_zero_when_missing(tmp_path):
    store = _make_store(tmp_path)
    inner = FakeInner(usage=None)
    traced = TracedLLMClient(inner, store)
    with trace_span() as tid:
        traced.chat_completion([{"role": "user", "content": "hi"}])
    ev = store.trace_llm_calls(tid)[0]
    assert ev["prompt_tokens"] == 0
