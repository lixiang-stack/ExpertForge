import json

from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.router import Router


def _domain(**overrides):
    default = {
        "name": "软件工程",
        "description": "sw",
        "out_of_domain_reply": "Out.",
        "intents": {
            "concept_explain": IntentDef("concept_explain", "explain"),
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug"),
            "architecture_design": IntentDef("architecture_design", "arch"),
        },
        "intent_mapping": {
            "concept_explain": "teaching",
            "faq": "direct",
            "troubleshooting": "debugging",
            "architecture_design": "analysis",
        },
        "strategies": {
            "teaching": StrategyDef("teaching", complexity_gate=True, default=True),
            "direct": StrategyDef("direct"),
            "debugging": StrategyDef("debugging", complexity_gate=True),
            "analysis": StrategyDef("analysis", complexity_gate=True),
        },
        "default_strategy": "teaching",
        "prompts": {},
    }
    default.update(overrides)
    return DomainConfig(**default)


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="cm", domain_dir="d")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completion(
        self,
        messages,
        model=None,
        disable_thinking=False,
        json_mode=False,
        json_schema=None,
    ):
        return self.responses.pop(0)


def _combined(in_domain, intent, complexity, reason="ok"):
    return (
        '{{"in_domain": {}, "intent": {}, "complexity": {}, "reason": "{}"}}'
    ).format(
        "true" if in_domain else "false",
        json.dumps(intent),
        json.dumps(complexity),
        reason,
    )


def test_route_in_domain_maps_strategy_and_keeps_fields():
    client = FakeClient([_combined(True, "concept_explain", "simple")])
    result = Router(client, _config(), _domain()).route("q")
    assert result.in_domain is True
    assert result.strategy == "teaching"
    assert result.intent == "concept_explain"
    assert result.complexity == "simple"
    assert result.orchestrate is False


def test_route_out_of_domain_rejects():
    client = FakeClient([_combined(False, None, None, "unrelated")])
    result = Router(client, _config(), _domain()).route("weather?")
    assert result.in_domain is False
    assert result.strategy == "reject"
    assert result.reject_reason == "unrelated"


def test_route_unknown_intent_falls_back_to_default():
    client = FakeClient([_combined(True, "bogus", "simple")])  # intent bogus → validation sets None
    result = Router(client, _config(), _domain()).route("q")
    assert result.strategy == "teaching"
    assert result.orchestrate is False


def test_route_complex_gated_sets_orchestrate():
    client = FakeClient([_combined(True, "architecture_design", "complex")])
    result = Router(client, _config(), _domain()).route("design a big system")
    assert result.strategy == "analysis"
    assert result.orchestrate is True


def test_route_complex_ungated_strategy_stays():
    client = FakeClient([_combined(True, "faq", "complex")])
    result = Router(client, _config(), _domain()).route("q")
    assert result.strategy == "direct"
    assert result.orchestrate is False