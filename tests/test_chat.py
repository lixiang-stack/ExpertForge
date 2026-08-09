from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef


def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out of domain.",
        intents={
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug", needs_clarification=True),
        },
        intent_mapping={"faq": "direct", "troubleshooting": "debugging"},
        strategies={
            "direct": StrategyDef("direct"),
            "debugging": StrategyDef("debugging", complexity_gate=True),
        },
        prompts={
            "direct": "Direct {name} {description} {structure}",
            "debugging": "Debug {name} {description} {structure}",
            "clarify": "What do you mean by {question} ({intent}/{complexity})?",
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


def test_respond_clarification_then_answer():
    chat = Chat(FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "medium", "reason": "ok"}',
        "clarify question",
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "medium", "reason": "ok"}',
        "the final answer",
    ]), _config(), _domain())
    resp = chat.respond("my go program hangs")
    assert resp.kind == "clarification"
    assert "clarify question" in resp.text
    resp2 = chat.answer_clarification("it hangs on startup")
    assert resp2.kind == "answer"
    assert resp2.text == "the final answer"
    assert chat.history == [
        ("my go program hangs\n\nAdditional context: it hangs on startup", "the final answer")
    ]


def test_respond_skips_clarification_when_disallowed():
    chat = Chat(FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "medium", "reason": "ok"}',
        "direct answer",
    ]), _config(), _domain())
    resp = chat.respond("my program hangs", allow_clarification=False)
    assert resp.kind == "answer"
    assert resp.text == "direct answer"


def test_respond_unsupported_complex():
    chat = Chat(FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "complex", "reason": "ok"}',
    ]), _config(), _domain())
    resp = chat.respond("huge debugging task")
    assert resp.kind == "unsupported"
    assert resp.text == "Needs orchestrator."
