import pytest

from agent.config import AgentConfig, ObservabilityConfig
from agent.llm import ChatResult
from agent.observability import install
from agent.observability import patch as patch_mod
from agent.observability.tracing import read_events


@pytest.fixture(autouse=True)
def _reset_observability():
    yield
    patch_mod._ACTIVE = None  # keep class-level patching transparent for other test modules


class FakeClient:
    model = "m"

    def __init__(self, responses):
        self._responses = list(responses)

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        return ChatResult(text=self._responses.pop(0), model=model or self.model)


_CLASSIFY = '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}'


def _domain():
    from agent.config import DomainConfig, IntentDef
    return DomainConfig(
        name="sw", description="desc", out_of_domain_reply="Out.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies=["direct"],
        prompts={"direct": "Direct prompt."},
    )


def _enabled_config(tmp_path):
    return AgentConfig(
        base_url="x", model="m", classifier_model="cm", domain_dir="d",
        observability=ObservabilityConfig(enabled=True, data_dir=str(tmp_path / "obs")),
    )


def test_install_disabled_returns_untouched(tmp_path):
    config = AgentConfig(base_url="x", model="m", classifier_model="cm", domain_dir="d")
    client = FakeClient([])
    out, plugin = install(client, config, None)
    assert out is client
    assert plugin is None


def test_install_enabled_wraps_client(tmp_path):
    client = FakeClient([])
    out, plugin = install(client, _enabled_config(tmp_path), None)
    assert out is not client
    assert plugin is not None


def test_install_enabled_patches_pipeline(tmp_path):
    from agent.chat import Chat
    inner = FakeClient([_CLASSIFY, "the answer"])
    out, _ = install(inner, _enabled_config(tmp_path), _domain())
    resp = Chat(out, _enabled_config(tmp_path), _domain()).respond("hi")
    assert resp.kind == "answer"
    events, bad = read_events(tmp_path / "obs")
    assert bad == 0
    assert any(e["type"] == "trace_start" for e in events)