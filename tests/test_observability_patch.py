import threading

import pytest

from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig, IntentDef, ObservabilityConfig, StrategyDef
from agent.observability import patch as patch_mod
from agent.observability.client import TracedLLMClient
from agent.observability.tracing import TraceStore, current_phase, read_events


@pytest.fixture(autouse=True)
def _reset_observability():
    yield
    patch_mod._ACTIVE = None  # keep class-level patching transparent for other test modules


class FakeInner:
    def __init__(self, responses):
        self._responses = list(responses)
        self._usage_local = threading.local()
        self.seen_phases = []

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        self.seen_phases.append(current_phase())
        return self._responses.pop(0)

    def chat_completion_stream(self, messages, **kwargs):
        return iter([])


_CLASSIFY = '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}'
_CLASSIFY_COMPLEX = '{"in_domain": true, "intent": "troubleshooting", "complexity": "complex", "reason": "ok"}'
_PLAN = '{"tasks": [{"title": "t1", "instruction": "i1"}, {"title": "t2", "instruction": "i2"}]}'


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="cm", domain_dir="d",
                       observability=ObservabilityConfig(enabled=True))


def _domain():
    return DomainConfig(
        name="sw", description="desc", out_of_domain_reply="Out.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies={"direct": StrategyDef("direct", default=True)},
        default_strategy="direct",
        prompts={"direct": "Direct prompt.", "unsupported_complex": "x."},
    )


def _domain_complex():
    return DomainConfig(
        name="sw", description="desc", out_of_domain_reply="Out.",
        intents={
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug"),
        },
        intent_mapping={"faq": "direct", "troubleshooting": "debugging"},
        strategies={
            "direct": StrategyDef("direct", default=True),
            "debugging": StrategyDef("debugging", complexity_gate=True),
        },
        default_strategy="direct",
        prompts={
            "direct": "Direct prompt.",
            "debugging": "Debugging prompt.",
            "unsupported_complex": "x.",
        },
    )


def _store(tmp_path):
    return TraceStore(tmp_path / "obs")


def test_install_wraps_and_records_pipeline(tmp_path):
    store = _store(tmp_path)
    inner = FakeInner([_CLASSIFY, "the answer"])
    chat = Chat(TracedLLMClient(inner, store), _config(), _domain())
    patch_mod.Installed(store, phase_map={}).apply()
    resp = chat.respond("what is defer")

    assert resp.kind == "answer"
    assert resp.text == "the answer"
    assert inner.seen_phases == ["classification", "strategy.direct"]
    events, _ = read_events(tmp_path / "obs")
    types = {e["type"] for e in events}
    assert {"trace_start", "llm_call", "decision", "trace_end"} <= types
    trace_id = None
    decision_phases = []
    for e in events:
        if e["type"] == "trace_start":
            trace_id = e["trace_id"]
        if e["type"] == "decision":
            decision_phases.append(e["phase"])
    assert trace_id is not None
    assert "classification" in decision_phases
    assert "route" in decision_phases
    trace_end = [e for e in events if e["type"] == "trace_end"][0]
    assert trace_end["answer_len"] == len("the answer")


def test_retains_original_return_values(tmp_path):
    client = FakeInner([_CLASSIFY, "the answer"])
    chat = Chat(client, _config(), _domain())
    patch_mod.Installed(_store(tmp_path), {}).apply()
    resp = chat.respond("what is defer")
    assert resp.text == "the answer"


def test_observability_failure_never_surfaces_into_business(tmp_path, monkeypatch):
    store = _store(tmp_path)
    inner = FakeInner([_CLASSIFY, "the answer"])
    chat = Chat(TracedLLMClient(inner, store), _config(), _domain())

    def _boom(*args, **kwargs):
        raise RuntimeError("observability layer exploded")

    monkeypatch.setattr(store, "write", _boom)
    patch_mod.Installed(store, {}).apply()
    resp = chat.respond("what is defer")
    assert resp.kind == "answer"
    assert resp.text == "the answer"


def test_install_wraps_orchestration_phases(tmp_path):
    store = _store(tmp_path)
    inner = FakeInner([_CLASSIFY_COMPLEX, _PLAN, "w1", "w2", "final"])
    chat = Chat(inner, _config(), _domain_complex())
    patch_mod.Installed(store, {}).apply()
    resp = chat.respond("huge debugging task")

    assert resp.text == "final"
    assert "orchestration.planner" in inner.seen_phases
    assert "orchestration.aggregate" in inner.seen_phases
    assert any(p.startswith("orchestration.worker.") for p in inner.seen_phases)
    # worker numbering restarts per trace: exactly worker.1 and worker.2
    assert "orchestration.worker.1" in inner.seen_phases
    assert "orchestration.worker.2" in inner.seen_phases