import threading

import pytest

from agent.llm import LLMError
from agent.observability.client import TracedLLMClient
from agent.observability.tracing import TraceStore, current_phase, phase, trace_span


class FakeInner:
    def __init__(self, usage=None, error=None, model="m"):
        self.model = model
        self._usage_local = threading.local()
        self._usage_local.usage = usage
        self._error = error
        self.last_kwargs = None

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        self.last_kwargs = (model, messages)
        if self._error:
            raise self._error
        return "the answer"

    def chat_completion_stream(self, messages, **kwargs):
        for chunk in ["a", "b"]:
            yield chunk


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
        text = traced.chat_completion([{"role": "user", "content": "hi"}], model="low-a")

    assert text == "the answer"
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


def test_records_non_streaming_usage_is_none_when_missing(tmp_path):
    store = _make_store(tmp_path)
    inner = FakeInner(usage=None)
    traced = TracedLLMClient(inner, store)
    with trace_span() as tid:
        traced.chat_completion([{"role": "user", "content": "hi"}])
    ev = store.trace_llm_calls(tid)[0]
    assert ev["prompt_tokens"] is None


def test_stream_delegates_without_recording(tmp_path):
    store = _make_store(tmp_path)
    traced = TracedLLMClient(FakeInner(), store)
    out = list(traced.chat_completion_stream([{"role": "user", "content": "hi"}]))
    assert out == ["a", "b"]
    assert store.trace_llm_calls("anything") == []
