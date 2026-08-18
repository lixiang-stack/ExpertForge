from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.llm import ChatResult, LLMError
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
        strategies={"direct": StrategyDef("direct", default=True)},
        default_strategy="direct",
        prompts={
            "direct": "Direct answer prompt.",
            "unsupported_complex": "unsupported",
        },
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_completion(
        self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
    ):
        return ChatResult(text=self.responses.pop(0), model=model or "m")


def test_repl_answers(monkeypatch, capsys):
    inputs = ["What is defer?", "exit"]
    monkeypatch.setattr("builtins.input", lambda prompt="": inputs.pop(0))
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        "the answer",
    ])
    run_repl(client, _config(), _domain())
    out = capsys.readouterr().out
    assert "the answer" in out


def test_repl_rejects(monkeypatch, capsys):
    inputs = ["What is the weather?", "exit"]
    monkeypatch.setattr("builtins.input", lambda prompt="": inputs.pop(0))
    client = FakeClient([
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}'
    ])
    run_repl(client, _config(), _domain())
    out = capsys.readouterr().out
    assert "Out of domain." in out


def test_repl_error_does_not_crash(monkeypatch, capsys):
    inputs = ["question", "exit"]
    monkeypatch.setattr("builtins.input", lambda prompt="": inputs.pop(0))

    class ErrorClient:
        def chat_completion(
            self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
        ):
            raise LLMError("network error")

    run_repl(ErrorClient(), _config(), _domain())
    out = capsys.readouterr().out
    assert "network error" in out
