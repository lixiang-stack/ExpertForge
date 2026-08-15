from agent.evaluation.judge import (
    JUDGE_DIMENSIONS,
    Judge,
    build_judge_prompt,
    parse_scorecard,
)


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False,
                        json_mode=False, json_schema=None):
        self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
        if self.error is not None:
            raise self.error
        return self.response


def test_build_judge_prompt_contains_question_answer_and_dimensions():
    prompt = build_judge_prompt("q?", "the answer", reference="ground truth")
    assert "q?" in prompt
    assert "the answer" in prompt
    assert "ground truth" in prompt
    for d in JUDGE_DIMENSIONS:
        assert d in prompt


def test_parse_scorecard_valid():
    text = ('{"correctness": 4, "relevance": 5, "completeness": 3, '
            '"technical_depth": 4, "practical_usefulness": 5, "hallucination": 2}')
    sc = parse_scorecard(text)
    assert sc is not None
    assert sc["correctness"] == 4
    assert sc["hallucination"] == 2


def test_parse_scorecard_unparseable():
    assert parse_scorecard("not json") is None
    assert parse_scorecard(None) is None


def test_parse_scorecard_missing_or_out_of_range():
    assert parse_scorecard('{"correctness": 4}') is None
    assert parse_scorecard(
        '{"correctness": 4, "relevance": 5, "completeness": 3, '
        '"technical_depth": 4, "practical_usefulness": 5, "hallucination": 99}'
    ) is None


def test_judge_returns_scorecard():
    client = FakeClient(
        '{"correctness": 5, "relevance": 4, "completeness": 4, '
        '"technical_depth": 5, "practical_usefulness": 4, "hallucination": 5}'
    )
    sc = Judge(client, "judge-a").score("q?", "answer")
    assert sc["correctness"] == 5
    messages, model, dt, jm, schema = client.calls[0]
    assert model == "judge-a"
    assert dt is True
    assert jm is True


def test_judge_error_returns_none():
    from agent.llm import LLMError

    sc = Judge(FakeClient(error=LLMError("boom")), "judge-a").score("q?", "answer")
    assert sc is None
