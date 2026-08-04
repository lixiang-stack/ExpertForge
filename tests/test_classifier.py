import pytest

from agent.classifier import Classification, classify_question
from agent.llm import LLMError


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False):
        self.calls.append((messages, model, disable_thinking))
        return self.responses.pop(0)


def test_classify_in_domain():
    client = FakeClient(['{"in_domain": true, "reason": "in software engineering"}'])
    result = classify_question(client, "What is microservices?", "软件工程", "software engineering")
    assert isinstance(result, Classification)
    assert result.in_domain is True
    assert result.reason == "in software engineering"
    assert client.calls[0][2] is True


def test_classify_out_of_domain():
    client = FakeClient(['{"in_domain": false, "reason": "unrelated"}'])
    result = classify_question(client, "What is the weather?", "软件工程", "software engineering")
    assert result.in_domain is False


def test_retry_then_success():
    client = FakeClient(["not JSON", '{"in_domain": true, "reason": "ok"}'])
    result = classify_question(client, "What is Kafka?", "软件工程", "software engineering")
    assert result.in_domain is True
    assert len(client.calls) == 2


def test_retry_then_fallback_reject():
    client = FakeClient(["garbage", "more garbage"])
    result = classify_question(client, "xxx", "软件工程", "software engineering")
    assert result.in_domain is False
    assert "Unreliable" in result.reason
    assert len(client.calls) == 2


def test_propagates_llm_error():
    class FailingClient:
        def chat_completion(self, messages, model=None, disable_thinking=False):
            raise LLMError("boom")

    with pytest.raises(LLMError):
        classify_question(FailingClient(), "q", "软件工程", "software engineering")
