import pytest

from agent.classifier import (
    Classification,
    classify_complexity,
    classify_intent,
    classify_question,
)
from agent.llm import LLMError


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False):
        self.calls.append((messages, model, disable_thinking, json_mode))
        return self.responses.pop(0)


def test_classify_in_domain():
    client = FakeClient(['{"in_domain": true, "reason": "in software engineering"}'])
    result = classify_question(client, "What is microservices?", "软件工程", "software engineering")
    assert isinstance(result, Classification)
    assert result.in_domain is True
    assert result.reason == "in software engineering"
    assert len(client.calls) == 1
    assert client.calls[0][2] is True
    assert client.calls[0][3] is True


def test_classify_out_of_domain():
    client = FakeClient(['{"in_domain": false, "reason": "unrelated"}'])
    result = classify_question(client, "What is the weather?", "软件工程", "software engineering")
    assert result.in_domain is False
    assert len(client.calls) == 1


def test_single_call_parse_failure_falls_back_reject():
    client = FakeClient(["garbage"])
    result = classify_question(client, "xxx", "软件工程", "software engineering")
    assert result.in_domain is False
    assert "Unreliable" in result.reason
    assert len(client.calls) == 1


def test_string_bool_not_coerced_to_true():
    client = FakeClient(['{"in_domain": "false", "reason": "unrelated"}'])
    result = classify_question(client, "What is the weather?", "软件工程", "software engineering")
    assert result.in_domain is False
    assert len(client.calls) == 1


def test_propagates_llm_error():
    class FailingClient:
        def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False):
            raise LLMError("boom")

    with pytest.raises(LLMError):
        classify_question(FailingClient(), "q", "软件工程", "software engineering")


def test_classify_intent_success():
    client = FakeClient(['{"intent": "concept_explain", "reason": "why"}'])
    result = classify_intent(client, "why is interface like this", "软件工程", "sw", ["concept_explain", "faq"])
    assert result.intent_id == "concept_explain"
    assert result.reason == "why"
    assert len(client.calls) == 1
    assert client.calls[0][3] is True


def test_classify_intent_unknown_falls_back_empty():
    client = FakeClient(['{"intent": "bogus", "reason": "x"}'])
    result = classify_intent(client, "q", "软件工程", "sw", ["concept_explain", "faq"])
    assert result.intent_id == ""
    assert "Unreliable" in result.reason
    assert len(client.calls) == 1


def test_classify_intent_unreliable_falls_back_empty():
    client = FakeClient(["garbage"])
    result = classify_intent(client, "q", "软件工程", "sw", ["concept_explain", "faq"])
    assert result.intent_id == ""
    assert "Unreliable" in result.reason
    assert len(client.calls) == 1


def test_classify_complexity_success():
    client = FakeClient(['{"complexity": "complex", "reason": "big scope"}'])
    result = classify_complexity(client, "design a redis cluster", "软件工程", "sw")
    assert result.level == "complex"
    assert len(client.calls) == 1


def test_classify_complexity_invalid_level_defaults_medium():
    client = FakeClient(['{"complexity": "huge", "reason": "x"}'])
    result = classify_complexity(client, "q", "软件工程", "sw")
    assert result.level == "medium"
    assert len(client.calls) == 1


def test_classify_complexity_unreliable_defaults_medium():
    client = FakeClient(["garbage"])
    result = classify_complexity(client, "q", "软件工程", "sw")
    assert result.level == "medium"