from agent.config import AgentConfig
from agent.llm import LLMError
from agent.repl import run_repl


class FakeClient:
    def __init__(self, classifications, streams):
        self.classifications = list(classifications)
        self.streams = list(streams)
        self.generate_calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False):
        return self.classifications.pop(0)

    def chat_completion_stream(self, messages, model=None):
        self.generate_calls.append(messages)
        for ch in self.streams.pop(0):
            yield ch


def _config():
    return AgentConfig(
        base_url="https://x",
        model="m",
        classifier_model="m",
        domain_name="软件工程",
        domain_description="software engineering",
        out_of_domain_reply="Out of domain, not supported.",
    )


def test_repl_in_domain_streams_answer(monkeypatch, capsys):
    inputs = iter(["What is microservices?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient(['{"in_domain": true, "reason": "ok"}'], ["流式回答"])
    run_repl(client, _config())
    out = capsys.readouterr().out
    assert "流式回答" in out
    assert len(client.generate_calls) == 1


def test_repl_out_of_domain_rejected(monkeypatch, capsys):
    inputs = iter(["What is the weather?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient(['{"in_domain": false, "reason": "unrelated"}'], [])
    run_repl(client, _config())
    out = capsys.readouterr().out
    assert "Out of domain, not supported." in out


def test_repl_api_error_does_not_crash(monkeypatch, capsys):
    inputs = iter(["question", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    class ErrorClient:
        def chat_completion(self, messages, model=None, disable_thinking=False):
            raise LLMError("network error")

        def chat_completion_stream(self, messages, model=None):
            raise AssertionError("should not be called")

    run_repl(ErrorClient(), _config())
    out = capsys.readouterr().out
    assert "network error" in out


def test_repl_blank_input_skipped(monkeypatch, capsys):
    inputs = iter(["   ", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient([], [])
    run_repl(client, _config())
    out = capsys.readouterr().out
    assert "Bye." in out


def test_repl_second_turn_includes_history(monkeypatch, capsys):
    inputs = iter(["q1", "q2", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    client = FakeClient(
        ['{"in_domain": true, "reason": "ok"}', '{"in_domain": true, "reason": "ok"}'],
        ["A1", "A2"],
    )
    run_repl(client, _config())
    messages = client.generate_calls[1]
    user_contents = [m["content"] for m in messages if m["role"] == "user"]
    assistant_contents = [m["content"] for m in messages if m["role"] == "assistant"]
    assert "q1" in user_contents
    assert "q2" in user_contents
    assert "A1" in assistant_contents
