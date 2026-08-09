from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.llm import LLMError
from agent.repl import run_repl


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="m", domain_dir="d")


def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out of domain.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies={"direct": StrategyDef("direct")},
        prompts={
            "direct": "Direct {name} {description} {structure}",
            "clarify": "clarify",
            "unsupported_complex": "unsupported",
        },
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False):
        return self.responses.pop(0)


def test_repl_answers(monkeypatch, capsys):
    inputs = iter(["What is defer?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient([
        '{"in_domain": true, "reason": "ok"}',
        '{"intent": "faq", "reason": "ok"}',
        '{"complexity": "simple", "reason": "ok"}',
        "the answer",
    ])
    run_repl(client, _config(), _domain())
    out = capsys.readouterr().out
    assert "the answer" in out


def test_repl_rejects(monkeypatch, capsys):
    inputs = iter(["What is the weather?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient(['{"in_domain": false, "reason": "unrelated"}'])
    run_repl(client, _config(), _domain())
    out = capsys.readouterr().out
    assert "Out of domain." in out


def test_repl_error_does_not_crash(monkeypatch, capsys):
    inputs = iter(["question", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    class ErrorClient:
        def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False):
            raise LLMError("network error")

    run_repl(ErrorClient(), _config(), _domain())
    out = capsys.readouterr().out
    assert "network error" in out
