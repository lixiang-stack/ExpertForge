from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.router import COMPLEX_UNSUPPORTED, Router


def _domain(**overrides):
    default = {
        "name": "软件工程",
        "description": "sw",
        "out_of_domain_reply": "Out.",
        "intents": {
            "concept_explain": IntentDef("concept_explain", "explain"),
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug", needs_clarification=True),
            "architecture_design": IntentDef("architecture_design", "arch"),
        },
        "intent_mapping": {
            "concept_explain": "teaching",
            "faq": "direct",
            "troubleshooting": "debugging",
            "architecture_design": "analysis",
        },
        "strategies": {
            "teaching": StrategyDef("teaching", complexity_gate=True),
            "direct": StrategyDef("direct"),
            "debugging": StrategyDef("debugging", complexity_gate=True),
            "analysis": StrategyDef("analysis", complexity_gate=True),
        },
        "prompts": {},
    }
    default.update(overrides)
    return DomainConfig(**default)


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="cm", domain_dir="d")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False):
        return self.responses.pop(0)


def test_route_in_domain_simple_strategy():
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "concept_explain", "reason": "ok"}',
        '{"complexity": "simple", "reason": "ok"}',
    ])
    result = Router(client, _config(), _domain()).route("q")
    assert result.in_domain is True
    assert result.strategy == "teaching"
    assert result.intent == "concept_explain"
    assert result.complexity == "simple"
    assert result.needs_clarification is False


def test_route_out_of_domain():
    client = FakeClient(['{"in_domain": false, "reason": "unrelated"}'])
    result = Router(client, _config(), _domain()).route("weather?")
    assert result.in_domain is False
    assert result.strategy == "reject"
    assert result.reject_reason == "unrelated"


def test_route_unknown_intent_defaults_to_direct():
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "", "reason": "unreliable"}',
        '{"intent": "", "reason": "unreliable"}',
        '{"complexity": "simple", "reason": "ok"}',
    ])
    result = Router(client, _config(), _domain()).route("q")
    assert result.strategy == "direct"


def test_route_needs_clarification():
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "troubleshooting", "reason": "ok"}',
        '{"complexity": "medium", "reason": "ok"}',
    ])
    result = Router(client, _config(), _domain()).route("my program hangs")
    assert result.needs_clarification is True


def test_route_complex_gated_to_unsupported():
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "architecture_design", "reason": "ok"}',
        '{"complexity": "complex", "reason": "ok"}',
    ])
    result = Router(client, _config(), _domain()).route("design a big system")
    assert result.strategy == COMPLEX_UNSUPPORTED
    assert result.needs_clarification is False


def test_route_complex_ungated_strategy_stays():
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "faq", "reason": "ok"}',
        '{"complexity": "complex", "reason": "ok"}',
    ])
    result = Router(client, _config(), _domain()).route("q")
    assert result.strategy == "direct"
