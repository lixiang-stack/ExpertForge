from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef


def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out of domain.",
        intents={
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug"),
        },
        intent_mapping={"faq": "direct", "troubleshooting": "debugging"},
        strategies={
            "direct": StrategyDef("direct"),
            "debugging": StrategyDef("debugging", complexity_gate=True),
        },
        prompts={
            "direct": "Direct {name} {description} {structure}",
            "debugging": "Debug {name} {description} {structure}",
            "unsupported_complex": "Needs orchestrator.",
        },
    )


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="cm", domain_dir="d")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False):
        return self.responses.pop(0)


def test_respond_reject():
    chat = Chat(FakeClient(['{"in_domain": false, "reason": "unrelated"}']), _config(), _domain())
    resp = chat.respond("weather?")
    assert resp.kind == "reject"
    assert resp.text == "Out of domain. (unrelated)"


def test_respond_answer_appends_history():
    chat = Chat(FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "faq", "reason": "ok"}',
        '{"complexity": "simple", "reason": "ok"}',
        "the answer",
    ]), _config(), _domain())
    resp = chat.respond("what is defer")
    assert resp.kind == "answer"
    assert resp.text == "the answer"
    assert chat.history == [("what is defer", "the answer")]


def test_respond_unsupported_complex():
    chat = Chat(FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "complex", "reason": "ok"}',
    ]), _config(), _domain())
    resp = chat.respond("huge debugging task")
    assert resp.kind == "unsupported"
    assert resp.text == "Needs orchestrator."
