import pytest

from agent.classification import (
    ClassificationResult,
    ClassificationService,
    build_classification_schema,
    validate_classification,
)
from agent.config import ComplexityLevelDef, ComplexityPolicy, DomainConfig, IntentDef
from agent.llm import ChatResult, LLMError


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
        strategies=[],
        prompts={},
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
        return ChatResult(text=self.responses.pop(0), model=model or "m")


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
    assert "intent" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "what is a pointer?"
    assert "what is a pointer?" not in messages[0]["content"]


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
    assert client.calls[0][3] is False   # json_mode is not passed; intent is json_schema
    assert client.calls[0][4] is not None  # json_schema intent; client negotiates the mechanism


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


def test_classify_uses_json_schema_intent():
    """The main path passes json_schema as the structured-output intent; the
    client negotiates the actual mechanism (json_schema vs json_object)."""
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
    ])
    result = ClassificationService(client, _domain()).classify("q")
    assert result.in_domain is True
    assert len(client.calls) == 1
    messages, model, disable_thinking, json_mode, json_schema = client.calls[0]
    assert json_mode is False
    assert json_schema is not None


def test_classify_api_failure_propagates():
    class AlwaysFailingClient:
        def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
            raise LLMError("boom")

    with pytest.raises(LLMError):
        ClassificationService(AlwaysFailingClient(), _domain()).classify("q")


from agent.classification import build_classification_prompt, build_complexity_section


def _rich_intent():
    return IntentDef(
        id="concept_explain",
        description="explain a concept",
        positive_examples=["Why does DI reduce coupling?"],
        negative_examples=["My app crashes."],
        boundaries=["Prefer concept_explain over faq when the user wants understanding."],
    )


def test_build_classification_prompt_renders_examples_and_boundaries():
    prompt = build_classification_prompt(
        "SE",
        "software engineering",
        [_rich_intent()],
    )
    assert "concept_explain: explain a concept" in prompt
    assert "Why does DI reduce coupling?" in prompt
    assert "My app crashes." in prompt
    assert "Boundary: Prefer concept_explain over faq when the user wants understanding." in prompt


def test_build_classification_prompt_omits_empty_sections():
    prompt = build_classification_prompt(
        "SE",
        "software engineering",
        [IntentDef("faq", "quick factual question")],
    )
    assert "faq: quick factual question" in prompt
    assert "Positive examples" not in prompt
    assert "Negative examples" not in prompt
    assert "Boundary:" not in prompt


def _complexity_policy():
    return ComplexityPolicy(levels=[
        ComplexityLevelDef(
            level="simple",
            description="single clear concept, single fact",
            dimensions=["Reasoning: single step", "Scope: single concept",
                        "Trade-off: none", "Coordination: none"],
            positive_examples=["What is dependency injection?"],
            negative_examples=["Design a distributed rate limiter"],
            boundaries=["Prefer medium over simple when multiple concepts"],
        ),
        ComplexityLevelDef(
            level="complex",
            description="multiple subsystems, multiple constraints",
            dimensions=["Reasoning: multi-step", "Scope: multiple subsystems"],
            positive_examples=["Design a distributed rate limiter for millions of QPS"],
            negative_examples=["What is dependency injection?"],
            boundaries=["Prefer complex when task decomposition is required"],
        ),
    ])


def test_build_complexity_section_renders_levels():
    section = build_complexity_section(_complexity_policy())
    assert "simple: single clear concept, single fact" in section
    assert "Reasoning: single step" in section
    assert "Trade-off: none" in section
    assert "What is dependency injection?" in section
    assert "Design a distributed rate limiter" in section
    assert "Boundary: Prefer medium over simple when multiple concepts" in section
    assert "complex: multiple subsystems, multiple constraints" in section


def test_build_complexity_section_none_renders_default():
    section = build_complexity_section(None)
    assert "short direct answer" in section


def test_build_classification_prompt_renders_complexity_policy():
    prompt = build_classification_prompt(
        "SE", "software engineering",
        [IntentDef("faq", "quick factual question")],
        complexity=_complexity_policy(),
    )
    assert "single clear concept, single fact" in prompt
    assert "Design a distributed rate limiter for millions of QPS" in prompt


def test_classify_passes_domain_complexity_to_prompt():
    domain = _domain()
    domain.complexity = _complexity_policy()
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
    ])
    ClassificationService(client, domain).classify("q")
    messages, model, disable_thinking, json_mode, json_schema = client.calls[0]
    assert "single clear concept, single fact" in messages[0]["content"]