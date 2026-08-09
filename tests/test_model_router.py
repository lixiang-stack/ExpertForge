from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.model_router import resolve_model
from agent.router import RouteResult


def _config(model_low=None, model_high=None):
    return AgentConfig(
        base_url="https://x", model="m", classifier_model="c",
        domain_dir="d", model_low=model_low, model_high=model_high,
    )


def _domain(strategy_model=None):
    strategies = {
        "direct": StrategyDef("direct", model=strategy_model),
        "teaching": StrategyDef("teaching"),
    }
    return DomainConfig(
        name="sw",
        description="d",
        out_of_domain_reply="Out.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies=strategies,
        prompts={},
    )


def _route(strategy="direct", complexity=None):
    return RouteResult(
        in_domain=True, strategy=strategy,
        intent="faq", complexity=complexity,
    )


def test_simple_uses_model_low():
    domain = _domain()
    result = resolve_model(_config("low-a", "high-a"), domain, _route(complexity="simple"), "default")
    assert result == "low-a"


def test_simple_missing_model_low_falls_back_to_default():
    domain = _domain()
    result = resolve_model(_config(), _domain(), _route(complexity="simple"), "default")
    assert result == "default"


def test_medium_uses_model_high():
    result = resolve_model(_config("low-a", "high-a"), _domain(), _route(complexity="medium"), "default")
    assert result == "high-a"


def test_complex_uses_model_high():
    result = resolve_model(_config("low-a", "high-a"), _domain(), _route(complexity="complex"), "default")
    assert result == "high-a"


def test_none_complexity_uses_model_high():
    result = resolve_model(_config("low-a", "high-a"), _domain(), _route(complexity=None), "default")
    assert result == "high-a"


def test_medium_missing_model_high_falls_back_to_default():
    result = resolve_model(_config("low-a", None), _domain(), _route(complexity="medium"), "default")
    assert result == "default"


def test_strategy_model_overrides_complexity():
    result = resolve_model(
        _config("low-a", "high-a"), _domain(strategy_model="strat-a"),
        _route(complexity="simple"), "default",
    )
    assert result == "strat-a"