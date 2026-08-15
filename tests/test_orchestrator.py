from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.llm import ChatResult
from agent.orchestrator import Orchestrator
from agent.router import RouteResult


def _domain():
    return DomainConfig(
        name="sw",
        description="software engineering",
        out_of_domain_reply="Out.",
        intents={"troubleshooting": IntentDef("troubleshooting", "debug")},
        intent_mapping={"troubleshooting": "debugging"},
        strategies={"debugging": StrategyDef("debugging", complexity_gate=True, default=True)},
        default_strategy="debugging",
        prompts={
            "debugging": "Debugging system prompt.",
        },
    )


def _config():
    return AgentConfig(
        base_url="https://x", model="m", classifier_model="cm", domain_dir="d",
        model_low="low-a", model_high="high-a",
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
        return ChatResult(text=self.responses.pop(0), model=model or "m")


def _route():
    return RouteResult(
        in_domain=True, strategy="debugging", intent="troubleshooting",
        complexity="complex", orchestrate=True,
    )


def test_run_normal_path_planner_workers_aggregator():
    client = FakeClient([
        '{"tasks": [{"title": "t1", "instruction": "i1"}, {"title": "t2", "instruction": "i2"}]}',
        "worker1 output",
        "worker2 output",
        "final answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "final answer"
    assert len(client.calls) == 4
    # planner call uses json_object (json_mode=True); json_schema preserved but unused
    planner_messages, planner_model, planner_dt, planner_jm, planner_schema = client.calls[0]
    assert planner_schema is None
    assert planner_jm is True
    assert planner_dt is True
    for _, model, dt, jm, schema in client.calls:
        assert model == "high-a"


def test_run_planner_invalid_json_degrades_to_direct():
    client = FakeClient([
        "not json",
        "direct answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "direct answer"
    assert len(client.calls) == 2


def test_run_planner_missing_tasks_degrades_to_direct():
    client = FakeClient([
        '{"other": 1}',
        "direct answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "direct answer"
    assert len(client.calls) == 2


def test_run_planner_malformed_item_degrades_to_direct():
    client = FakeClient([
        '{"tasks": [{"title": "t1", "instruction": 5}]}',
        "direct answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "direct answer"
    assert len(client.calls) == 2


def test_run_planner_empty_tasks_degrades_to_direct():
    client = FakeClient([
        '{"tasks": []}',
        "direct answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "direct answer"
    assert len(client.calls) == 2


def test_run_worker_empty_output_still_aggregates():
    client = FakeClient([
        '{"tasks": [{"title": "t1", "instruction": "i1"}, {"title": "t2", "instruction": "i2"}]}',
        "worker1 output",
        "",
        "final answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "final answer"
    assert len(client.calls) == 4
    # aggregator user message contains both worker outputs including the empty one
    agg_messages = client.calls[3][0]
    agg_user = agg_messages[-1]["content"]
    assert "worker1 output" in agg_user


def test_run_planner_uses_json_object_main_path():
    """The planner's main path uses json_object (json_mode=True); json_schema is
    preserved but not exercised until a provider supports it (see code TODO)."""
    client = FakeClient([
        '{"tasks": [{"title": "t1", "instruction": "i1"}]}',
        "worker1 output",
        "final answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "final answer"
    planner_messages, planner_model, planner_dt, planner_jm, planner_schema = client.calls[0]
    assert planner_schema is None
    assert planner_jm is True
    assert planner_dt is True
