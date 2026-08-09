import pytest

from agent.classification import (
    ClassificationResult,
    ClassificationService,
    build_classification_schema,
    validate_classification,
)
from agent.config import DomainConfig, IntentDef
from agent.llm import LLMError


def _domain():
    return DomainConfig(
        name="软件工程",
        description="software engineering",
        out_of_domain_reply="Out.",
        intents={
            "concept_explain": IntentDef("concept_explain", "explain a concept"),
            "faq": IntentDef("faq", "quick factual question"),
        },
        intent_mapping={},
        strategies={},
        prompts={},
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
        return self.responses.pop(0)


def _classify(text):
    client = FakeClient([text])
    service = ClassificationService(client, _domain())
    result = service.classify("what is a pointer?", model="cm")
    return result, client


def test_classify_single_call_returns_all_fields():
    result, client = _classify(
        '{"in_domain": true, "intent": "concept_explain", "complexity": "medium", "reason": "why question"}'
    )
    assert isinstance(result, ClassificationResult)
    assert result.in_domain is True
    assert result.intent == "concept_explain"
    assert result.complexity == "medium"
    assert result.reason == "why question"
    assert len(client.calls) == 1
    messages, model, disable_thinking, json_mode, json_schema = client.calls[0]
    assert model == "cm"
    assert disable_thinking is True
    assert json_mode is False
    assert json_schema is not None
    assert "intent" in json_schema["properties"]


def test_classify_garbage_text_falls_back_reject():
    result, client = _classify("garbage that is not json")
    assert result.in_domain is False
    assert result.intent is None
    assert result.complexity is None
    assert result.reason.startswith("Unreliable")
    assert len(client.calls) == 1


def test_classify_out_of_domain_accepts_null():
    result, client = _classify(
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}'
    )
    assert result.in_domain is False
    assert result.intent is None
    assert result.complexity is None
    assert len(client.calls) == 1
    assert client.calls[0][4] is not None


def test_validate_non_bool_in_domain_falls_back_reject():
    result = validate_classification(
        {"in_domain": "false", "intent": "faq", "complexity": "simple", "reason": "x"},
        ["concept_explain", "faq"],
    )
    assert result.in_domain is False
    assert result.intent is None
    assert result.reason.startswith("Unreliable")


def test_validate_invalid_json_falls_back_reject():
    result = validate_classification(None, ["concept_explain", "faq"])
    assert result.in_domain is False
    assert result.reason.startswith("Unreliable")


def test_validate_null_in_domain_falls_back_reject():
    result = validate_classification(
        {"in_domain": None, "intent": "faq", "complexity": "simple", "reason": "x"},
        ["concept_explain", "faq"],
    )
    assert result.in_domain is False


def test_validate_unknown_intent_falls_back_none():
    result = validate_classification(
        {"in_domain": True, "intent": "bogus", "complexity": "simple", "reason": "x"},
        ["concept_explain", "faq"],
    )
    assert result.in_domain is True
    assert result.intent is None
    assert result.complexity == "simple"


def test_validate_invalid_complexity_falls_back_medium():
    result = validate_classification(
        {"in_domain": True, "intent": "faq", "complexity": "huge", "reason": "x"},
        ["concept_explain", "faq"],
    )
    assert result.in_domain is True
    assert result.intent == "faq"
    assert result.complexity == "medium"


def test_schema_enum_derived_from_intent_ids():
    schema = build_classification_schema(["faq", "concept_explain"])
    assert schema["properties"]["intent"]["enum"] == ["faq", "concept_explain", None]
    assert "complexity" in schema["properties"]
    assert schema["required"] == ["in_domain", "intent", "complexity", "reason"]


def test_degrade_to_json_mode_when_schema_unsupported():
    class SchemaRejectingClient:
        def __init__(self):
            self.calls = []

        def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
            self.calls.append((messages, disable_thinking, json_mode, json_schema))
            if json_schema is not None:
                raise LLMError("response_format json_schema not supported")
            return '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}'

    client = SchemaRejectingClient()
    result = ClassificationService(client, _domain()).classify("q")
    assert result.in_domain is True
    assert result.intent == "faq"
    assert len(client.calls) == 2
    messages, disable_thinking, json_mode, json_schema = client.calls[0]
    assert json_schema is not None
    messages, disable_thinking, json_mode, json_schema = client.calls[1]
    assert json_schema is None
    assert json_mode is True
    assert "JSON only" in messages[0]["content"] or '"in_domain"' in messages[0]["content"]


def test_degrade_second_call_also_fails_propagates():
    class AlwaysFailingClient:
        def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
            raise LLMError("boom")

    with pytest.raises(LLMError):
        ClassificationService(AlwaysFailingClient(), _domain()).classify("q")