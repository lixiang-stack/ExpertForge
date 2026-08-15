from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.llm import ChatResult


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
            "direct": StrategyDef("direct", default=True),
            "debugging": StrategyDef("debugging", complexity_gate=True),
        },
        default_strategy="direct",
        prompts={
            "direct": "Direct answer prompt.",
            "debugging": "Debugging prompt.",
            "unsupported_complex": "Needs orchestrator.",
        },
    )


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="cm", domain_dir="d")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.models = []

    def chat_completion(
        self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
    ):
        self.models.append(model)
        return ChatResult(text=self.responses.pop(0), model=model or "m")


def test_respond_reject():
    chat = Chat(
        FakeClient(['{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}']),
        _config(),
        _domain(),
    )
    resp = chat.respond("weather?")
    assert resp.kind == "reject"
    assert resp.text == "Out of domain. (unrelated)"


def test_respond_answer_appends_history():
    chat = Chat(FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        "the answer",
    ]), _config(), _domain())
    resp = chat.respond("what is defer")
    assert resp.kind == "answer"
    assert resp.text == "the answer"
    assert chat.history == [("what is defer", "the answer")]


def test_respond_orchestrates_complex():
    client = FakeClient([
        '{"in_domain": true, "intent": "troubleshooting", "complexity": "complex", "reason": "ok"}',
        '{"tasks": [{"title": "t1", "instruction": "i1"}, {"title": "t2", "instruction": "i2"}]}',
        "worker1 output",
        "worker2 output",
        "final answer",
    ])
    chat = Chat(client, _config(), _domain())
    resp = chat.respond("huge debugging task")
    assert resp.kind == "answer"
    assert resp.text == "final answer"
    assert chat.history == [("huge debugging task", "final answer")]


def test_respond_uses_complexity_routed_model():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        "the answer",
    ])
    chat = Chat(client, AgentConfig(base_url="https://x", model="m", classifier_model="cm",
                                    domain_dir="d", model_low="low-a", model_high="high-a"),
                _domain())
    resp = chat.respond("what is defer")
    assert resp.kind == "answer"
    # first call: classification (model=cm); second call: generator (model=low)
    assert client.models == ["cm", "low-a"]


def test_respond_with_precomputed_route_skips_classification():
    from agent.router import RouteResult

    client = FakeClient(["the answer"])
    chat = Chat(client, _config(), _domain())
    route = RouteResult(in_domain=True, strategy="direct", intent="faq", complexity="simple")
    resp = chat.respond("what is defer", route=route)
    assert resp.kind == "answer"
    assert resp.text == "the answer"
    assert client.models == ["m"]  # only the answer call; classification was skipped
