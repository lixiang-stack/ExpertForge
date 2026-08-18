from agent.config import AgentConfig, DomainConfig, EvaluationConfig, IntentDef
from agent.evaluation.dataset import Suite, EvalCase
from agent.evaluation.runner import RecordingClient, run_evaluation


def _dataset():
    return Suite(name="direct", domain="software_engineering", cases=[
        EvalCase(
            id="se-001", question="what is defer",
            expected_domain="software_engineering",
            expected_intent="faq", expected_complexity="simple",
            expected_strategy="direct", expected_orchestrate=False,
            answer_quality=True, reference="short",
        ),
        EvalCase(
            id="se-002", question="recommend a restaurant",
            expected_domain="other",
            expected_intent=None, expected_complexity=None,
            expected_strategy="reject", expected_orchestrate=False,
            answer_quality=False, reference=None,
        ),
    ])


def _domain():
    return DomainConfig(
        name="sw", description="desc", out_of_domain_reply="Out.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies=["direct"],
        prompts={"direct": "Direct prompt."},
    )


def _config():
    return AgentConfig(base_url="https://x", model="m", classifier_model="cm",
                       domain_dir="d", model_low="low-a", model_high="high-a",
                       evaluation=EvaluationConfig(judge_model="judge-a"))


from agent.llm import ChatResult, LLMError


class FakeClient:
    def __init__(self, responses, usage=None, fail_on_call=None):
        self.responses = list(responses)
        self.models = []
        self.json_modes = []
        self.usage_queue = list(usage or [])
        self.fail_on_call = fail_on_call
        self.call_count = 0

    def chat_completion(self, messages, model=None, temperature=0.3,
                        disable_thinking=False, json_mode=False, json_schema=None):
        self.call_count += 1
        if self.call_count == self.fail_on_call:
            raise LLMError("boom")
        self.models.append(model)
        self.json_modes.append(json_mode)
        prompt = completion = cached = 0
        if self.usage_queue:
            prompt, completion, cached = self.usage_queue.pop(0)
        return ChatResult(
            text=self.responses.pop(0),
            model=model or "m",
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cache_tokens=cached,
        )

    def _record_usage(self, prompt, completion, cached=0):
        """Set the usage seen by the NEXT chat_completion call."""
        self.usage_queue.append((prompt, completion, cached))


def test_recording_client_records_usage_and_latency():
    inner = FakeClient(["hello"], usage=[(10, 5, 3)])
    rc = RecordingClient(inner)
    out = rc.chat_completion([{"role": "user", "content": "hi"}], model="m2")
    assert out.text == "hello"
    assert rc.calls[0]["model"] == "m2"
    assert rc.calls[0]["prompt_tokens"] == 10
    assert rc.calls[0]["completion_tokens"] == 5
    assert rc.calls[0]["total_tokens"] == 15
    assert rc.calls[0]["cache_tokens"] == 3
    assert rc.calls[0]["latency_ms"] >= 0
    rc.reset()
    assert rc.calls == []


def test_run_evaluation_answers_and_judges():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        "the answer",
        '{"correctness": 4, "relevance": 5, "completeness": 3, '
        '"technical_depth": 4, "practical_usefulness": 5, "hallucination": 4}',
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ])
    client._record_usage(10, 5, cached=2)   # classification
    client._record_usage(20, 8, cached=4)   # answer generation
    client._record_usage(5, 2, cached=1)    # judge
    results = run_evaluation(_config(), _domain(), _dataset(), client)
    assert len(results) == 2
    r0 = results[0]
    assert r0.case.id == "se-001"
    assert r0.in_domain is True
    assert r0.intent == "faq"
    assert r0.complexity == "simple"
    assert r0.strategy == "direct"
    assert r0.orchestrate is False
    assert r0.answer == "the answer"
    assert r0.actual_model == "low-a"  # answer call model (judge excluded)
    assert r0.expected_model == "low-a"
    assert r0.scorecard is not None
    assert r0.scorecard["correctness"] == 4
    assert r0.llm_calls == 3
    assert r0.in_tokens == 35
    assert r0.out_tokens == 15
    assert r0.total_tokens == 50
    assert r0.cache_tokens == 7


def test_run_evaluation_rejects_out_of_domain():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ])
    results = run_evaluation(_config(), _domain(), _dataset(), client, skip_quality=True)
    r1 = results[1]
    assert r1.in_domain is False
    assert r1.strategy == "reject"
    assert r1.answer is None
    assert r1.scorecard is None
    assert r1.llm_calls == 1
    assert r1.actual_model is None  # out-of-domain: no answer call


def test_run_evaluation_skip_quality_skips_answer():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ])
    results = run_evaluation(_config(), _domain(), _dataset(), client, skip_quality=True)
    r0 = results[0]
    assert r0.answer is None
    assert r0.scorecard is None
    assert r0.llm_calls == 1


def test_run_evaluation_records_suite():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ])
    results = run_evaluation(_config(), _domain(), _dataset(), client, skip_quality=True)
    assert results[0].suite == "direct"
    assert results[1].suite == "direct"


def test_run_evaluation_records_answer_error_and_continues():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        "the answer",
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ], fail_on_call=2)  # the answer-generation call fails
    results = run_evaluation(_config(), _domain(), _dataset(), client)
    assert len(results) == 2
    r0 = results[0]
    assert r0.error == "LLMError: boom"
    assert r0.answer is None
    assert r0.scorecard is None
    assert r0.in_domain is True          # route succeeded, answer phase failed
    r1 = results[1]
    assert r1.error is None              # later case still processed
    assert r1.strategy == "reject"


def test_run_evaluation_records_route_stage_error():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        "the answer",
        '{"in_domain": false, "intent": null, "complexity": null, "reason": "unrelated"}',
    ], fail_on_call=4)  # the second case's classification call fails
    results = run_evaluation(_config(), _domain(), _dataset(), client)
    assert len(results) == 2
    r1 = results[1]
    assert r1.error == "LLMError: boom"
    assert r1.in_domain is False
    assert r1.strategy is None
    assert r1.expected_model is None