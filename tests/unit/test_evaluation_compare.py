import pytest
from agent.config import AgentConfig, DomainConfig, EvaluationConfig, IntentDef, JudgeConfig, OrchestrationPolicy, EvaluatorPolicy
from agent.evaluation.dataset import EvalCase
from agent.llm import ChatResult, LLMError
from agent.router import RouteResult


_FAKE_PLAN = '{"tasks": [{"title": "t1", "instruction": "i1", "role": "R1"}, {"title": "t2", "instruction": "i2", "role": "R2"}]}'
_FAKE_SCORE_PASS = ('{"correctness": 4, "relevance": 4, "completeness": 4, '
                    '"technical_depth": 4, "practical_usefulness": 4, "hallucination": 4}')
class FakeClient:
    def __init__(self, responses, usage=None):
        self.responses = list(responses)
        self.usage_queue = list(usage or [])
        self.call_count = 0

    def chat_completion(self, messages, model=None, temperature=0.3,
                        disable_thinking=False, json_mode=False, json_schema=None):
        self.call_count += 1
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


def _domain():
    return DomainConfig(
        name="sw", description="software engineering", out_of_domain_reply="Out.",
        intents={"troubleshooting": IntentDef("troubleshooting", "debug")},
        intent_mapping={"troubleshooting": "debugging"},
        strategies=["debugging"],
        prompts={"debugging": "Debugging system prompt."},
        orchestration=OrchestrationPolicy(
            enabled=True, min_complexity="complex", intents=["troubleshooting"],
            max_workers=4,
            evaluator=EvaluatorPolicy(enabled=True, min_dimension_score=3, max_rounds=1),
        ),
    )


def _config():
    return AgentConfig(
        base_url="https://x", model="m", classifier_model="cm", domain_dir="d",
        model_low="low-a", model_high="high-a",
        evaluation=EvaluationConfig(
            judge=JudgeConfig(base_url="https://j", model="judge-a", provider="p")),
    )


def _cases():
    return [
        EvalCase(
            id="se-001", question="my service crashes under load",
            expected_domain="software_engineering",
            expected_intent="troubleshooting", expected_complexity="complex",
            expected_strategy="debugging", expected_orchestrate=True,
            tier="full_expert", reference="check memory",
        ),
        EvalCase(
            id="se-002", question="database connection pool exhausted",
            expected_domain="software_engineering",
            expected_intent="troubleshooting", expected_complexity="complex",
            expected_strategy="debugging", expected_orchestrate=True,
            tier="full_expert", reference=None,
        ),
    ]


def test_compare_baseline_single_call():
    from agent.evaluation.compare import run_compare

    # 1 baseline response + 5 orchestrated responses + 2 judge responses
    client = FakeClient(
        [
            "baseline answer",          # baseline: 1 call
            _FAKE_PLAN,                 # orch: planner
            "worker1",                  # orch: worker 1
            "worker2",                  # orch: worker 2
            "aggregate answer",         # orch: aggregator
            _FAKE_SCORE_PASS,           # orch: internal evaluator judge
            "baseline answer 2",        # case 2 baseline
            _FAKE_PLAN,                 # case 2 orch: planner
            "worker1-2",                # case 2 orch: worker 1
            "worker2-2",                # case 2 orch: worker 2
            "aggregate answer 2",       # case 2 orch: aggregator
            _FAKE_SCORE_PASS,           # case 2 orch: internal evaluator judge
        ],
        usage=[(10, 5, 0)] * 12,
    )
    # 2 judge responses per case (baseline + orch) = 4 total
    judge_client = FakeClient(
        [_FAKE_SCORE_PASS, _FAKE_SCORE_PASS,
         _FAKE_SCORE_PASS, _FAKE_SCORE_PASS],
        usage=[(5, 2, 0)] * 4,
    )
    results = run_compare(_config(), _domain(), _cases(), client, judge_client=judge_client)
    assert len(results) == 2

    r0 = results[0]
    assert r0.case.id == "se-001"
    assert r0.baseline.llm_calls == 1
    assert r0.baseline.answer == "baseline answer"
    assert r0.orchestrated.llm_calls == 5
    assert r0.orchestrated.answer == "aggregate answer"
    assert r0.baseline.quality is not None
    assert r0.orchestrated.quality is not None
    assert r0.quality_gain is not None
    assert r0.additional_tokens >= 0
    assert r0.cost_efficiency is not None


def test_compare_quality_gain():
    from agent.evaluation.compare import run_compare

    client = FakeClient([
        "baseline answer",
        _FAKE_PLAN,
        "worker1", "worker2", "aggregate answer",
        _FAKE_SCORE_PASS,  # internal judge passes
    ])
    # baseline score gives avg 3.0, orch score gives avg 4.0
    judge_client = FakeClient([
        '{"correctness": 3, "relevance": 3, "completeness": 3, '
        '"technical_depth": 3, "practical_usefulness": 3, "hallucination": 3}',
        '{"correctness": 4, "relevance": 4, "completeness": 4, '
        '"technical_depth": 4, "practical_usefulness": 4, "hallucination": 4}',
    ])
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    r0 = results[0]
    assert r0.baseline.quality == 3.0
    assert r0.orchestrated.quality == 4.0
    assert r0.quality_gain == 1.0
    assert r0.quality_gain_pct == 33.33


def test_compare_additional_tokens():
    from agent.evaluation.compare import run_compare

    client = FakeClient(
        ["baseline answer", _FAKE_PLAN, "w1", "w2", "agg", _FAKE_SCORE_PASS],
        usage=[(10, 5, 2), (20, 10, 5), (30, 15, 8), (30, 15, 8), (10, 5, 2), (5, 2, 1)],
    )
    # baseline: 1 call = 15 tokens (10+5)
    # orchestrated: 5 calls = 20+30+30+10+5 = 95 tokens + 15+15+15+5+2 completions = 52
    judge_client = FakeClient([_FAKE_SCORE_PASS, _FAKE_SCORE_PASS],
                              usage=[(5, 2, 1), (5, 2, 1)])
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    r0 = results[0]
    assert r0.baseline.total_tokens == 15
    # orch: 5 calls, prompt = 20+30+30+10+5 = 95, completion = 10+15+15+5+2 = 47, total = 142
    assert r0.orchestrated.total_tokens == 95 + 47
    assert r0.additional_tokens == r0.orchestrated.total_tokens - r0.baseline.total_tokens


def test_compare_cost_efficiency():
    from agent.evaluation.compare import run_compare

    client = FakeClient(
        ["baseline answer", _FAKE_PLAN, "w1", "w2", "agg", _FAKE_SCORE_PASS],
        usage=[(10, 5, 0), (20, 10, 0), (30, 15, 0), (30, 15, 0), (10, 5, 0), (5, 2, 0)],
    )
    judge_client = FakeClient(
        ['{"correctness": 3, "relevance": 3, "completeness": 3, '
         '"technical_depth": 3, "practical_usefulness": 3, "hallucination": 3}',
         '{"correctness": 4, "relevance": 4, "completeness": 4, '
         '"technical_depth": 4, "practical_usefulness": 4, "hallucination": 4}'],
        usage=[(5, 2, 0), (5, 2, 0)],
    )
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    r0 = results[0]
    # quality_gain = 1.0, additional_tokens > 0, cost_efficiency rounded to 6dp
    expected = round(1.0 / r0.additional_tokens, 6)
    assert r0.cost_efficiency == expected
    assert r0.cost_efficiency > 0


def test_compare_quality_pct():
    from agent.evaluation.compare import run_compare

    client = FakeClient([
        "baseline answer",
        _FAKE_PLAN, "w1", "w2", "agg", _FAKE_SCORE_PASS,
    ])
    # baseline avg = 2.0, orch avg = 4.0, gain = 2.0, gain_pct = 100.0
    judge_client = FakeClient([
        '{"correctness": 2, "relevance": 2, "completeness": 2, '
        '"technical_depth": 2, "practical_usefulness": 2, "hallucination": 2}',
        '{"correctness": 4, "relevance": 4, "completeness": 4, '
        '"technical_depth": 4, "practical_usefulness": 4, "hallucination": 4}',
    ])
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    r0 = results[0]
    assert r0.quality_gain_pct == 100.0


def test_compare_token_pct():
    from agent.evaluation.compare import run_compare

    client = FakeClient(
        ["baseline answer", _FAKE_PLAN, "w1", "w2", "agg", _FAKE_SCORE_PASS],
        usage=[(10, 5, 0), (20, 10, 0), (30, 15, 0), (30, 15, 0), (10, 5, 0), (5, 2, 0)],
    )
    judge_client = FakeClient([_FAKE_SCORE_PASS, _FAKE_SCORE_PASS],
                              usage=[(5, 2, 0), (5, 2, 0)])
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    r0 = results[0]
    # baseline = 15, orch = 20+30+30+10+5 = 95 + 10+15+15+5+2 = 47 = 142
    additional = r0.additional_tokens
    expected_pct = round((additional / r0.baseline.total_tokens) * 100, 2)
    assert r0.token_increase_pct == expected_pct


def test_compare_error_per_mode():
    from agent.evaluation.compare import run_compare

    class FailClient:
        def __init__(self):
            self.calls = []
        def chat_completion(self, *args, **kwargs):
            self.calls.append(kwargs)
            raise LLMError("baseline boom")

    client = FailClient()
    judge_client = FakeClient([_FAKE_SCORE_PASS, _FAKE_SCORE_PASS])
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    r0 = results[0]
    assert r0.baseline.error == "LLMError: baseline boom"
    assert r0.baseline.quality is None
    assert r0.quality_gain is None


def test_compare_both_modes_fail():
    from agent.evaluation.compare import run_compare

    class FailClient:
        def __init__(self):
            self.calls = []
        def chat_completion(self, *args, **kwargs):
            self.calls.append(kwargs)
            raise LLMError("all boom")

    client = FailClient()
    judge_client = FakeClient([_FAKE_SCORE_PASS, _FAKE_SCORE_PASS])
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    r0 = results[0]
    assert r0.baseline.error is not None
    assert r0.orchestrated.error is not None
    assert r0.quality_gain is None


def test_compare_quality_gain_pct():
    from agent.evaluation.compare import run_compare

    client = FakeClient([
        "baseline answer",
        _FAKE_PLAN, "w1", "w2", "agg", _FAKE_SCORE_PASS,
    ])
    # baseline quality = 1.0 (all dims = 1), orch quality = 4.0
    judge_client = FakeClient([
        '{"correctness": 1, "relevance": 1, "completeness": 1, '
        '"technical_depth": 1, "practical_usefulness": 1, "hallucination": 1}',
        '{"correctness": 4, "relevance": 4, "completeness": 4, '
        '"technical_depth": 4, "practical_usefulness": 4, "hallucination": 4}',
    ])
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    r0 = results[0]
    assert r0.baseline.quality == 1.0
    assert r0.quality_gain == 3.0
    assert r0.quality_gain_pct == 300.0


def test_compare_zero_tokens_guard():
    from agent.evaluation.compare import run_compare

    client = FakeClient(
        ["baseline answer", _FAKE_PLAN, "w1", "w2", "agg", _FAKE_SCORE_PASS],
        usage=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
    )
    judge_client = FakeClient([_FAKE_SCORE_PASS, _FAKE_SCORE_PASS],
                              usage=[(0, 0, 0), (0, 0, 0)])
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    r0 = results[0]
    assert r0.baseline.total_tokens == 0
    assert r0.orchestrated.total_tokens == 0
    assert r0.additional_tokens == 0
    assert r0.cost_efficiency is None  # division by zero guard


def test_compare_aggregate_by_intent():
    from agent.evaluation.compare import run_compare, _compute_aggregates

    client = FakeClient([
        "base1", _FAKE_PLAN, "w1", "w2", "agg1", _FAKE_SCORE_PASS,
        "base2", _FAKE_PLAN, "w1-2", "w2-2", "agg2", _FAKE_SCORE_PASS,
    ])
    judge_client = FakeClient([
        _FAKE_SCORE_PASS, _FAKE_SCORE_PASS,   # case 1: baseline + orch
        _FAKE_SCORE_PASS, _FAKE_SCORE_PASS,   # case 2: baseline + orch
    ])
    results = run_compare(_config(), _domain(), _cases(), client, judge_client=judge_client)
    agg = _compute_aggregates(results)
    assert "overall" in agg
    assert agg["overall"]["n_compared"] == 2
    assert agg["overall"]["mean_quality_gain"] >= 0
    assert agg["overall"]["sum_additional_tokens"] >= 0
    assert "by_intent" in agg
    assert "troubleshooting" in agg["by_intent"]
    assert agg["by_intent"]["troubleshooting"]["n_compared"] == 2


def test_compare_aggregate_by_complexity():
    from agent.evaluation.compare import run_compare, _compute_aggregates

    client = FakeClient([
        "base1", _FAKE_PLAN, "w1", "w2", "agg1", _FAKE_SCORE_PASS,
        "base2", _FAKE_PLAN, "w1-2", "w2-2", "agg2", _FAKE_SCORE_PASS,
    ])
    judge_client = FakeClient([
        _FAKE_SCORE_PASS, _FAKE_SCORE_PASS,
        _FAKE_SCORE_PASS, _FAKE_SCORE_PASS,
    ])
    results = run_compare(_config(), _domain(), _cases(), client, judge_client=judge_client)
    agg = _compute_aggregates(results)
    assert "by_complexity" in agg
    assert "complex" in agg["by_complexity"]
    assert agg["by_complexity"]["complex"]["n_compared"] == 2


def test_compare_aggregate_excludes_none_quality_gain():
    from agent.evaluation.compare import run_compare, _compute_aggregates

    class HalfFailClient:
        def __init__(self):
            self.call_count = 0
        def chat_completion(self, *args, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                raise LLMError("baseline boom")
            return ChatResult(text="answer", model="m", prompt_tokens=10, completion_tokens=5, total_tokens=15, cache_tokens=0)

    client = HalfFailClient()
    judge_client = FakeClient([_FAKE_SCORE_PASS, _FAKE_SCORE_PASS])
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    agg = _compute_aggregates(results)
    assert agg["overall"]["n_compared"] == 0  # quality_gain was None
    assert agg["overall"]["mean_quality_gain"] is None


def test_compare_serialize_roundtrip():
    from agent.evaluation.compare import run_compare, serialize_compare_result

    client = FakeClient([
        "base1", _FAKE_PLAN, "w1", "w2", "agg1", _FAKE_SCORE_PASS,
    ])
    judge_client = FakeClient([_FAKE_SCORE_PASS, _FAKE_SCORE_PASS])
    results = run_compare(_config(), _domain(), _cases()[:1], client, judge_client=judge_client)
    record = serialize_compare_result(
        results, domain="sw", label="test", model="m", judge_model="judge-a",
    )
    assert record["kind"] == "compare"
    assert record["n_cases"] == 1
    assert record["n_compared"] == 1
    assert len(record["cases"]) == 1
    assert "baseline" in record["cases"][0]
    assert "orchestrated" in record["cases"][0]
    assert "quality_gain" in record["cases"][0]
    assert "aggregates" in record
    assert "overall" in record["aggregates"]


def test_compare_zero_quality_guard():
    from agent.evaluation.compare import _compute_deltas, ModeRun

    baseline = ModeRun(quality=0.0, total_tokens=100)
    orchestrated = ModeRun(quality=4.0, total_tokens=500)
    deltas = _compute_deltas(baseline, orchestrated)
    assert deltas["quality_gain"] == 4.0
    assert deltas["quality_gain_pct"] is None
    assert deltas["additional_tokens"] == 400
    assert deltas["cost_efficiency"] == 0.01


def test_compare_aggregate_overall():
    from agent.evaluation.compare import _compute_aggregates, CompareCaseResult, ModeRun
    from agent.evaluation.dataset import EvalCase

    case1 = EvalCase(id="se-001", question="q1", expected_domain="se",
                     expected_intent="troubleshooting", expected_complexity="complex",
                     expected_strategy="debugging", expected_orchestrate=True,
                     tier="full_expert", reference=None)
    case2 = EvalCase(id="se-002", question="q2", expected_domain="se",
                     expected_intent="code_task", expected_complexity="complex",
                     expected_strategy="direct", expected_orchestrate=True,
                     tier="full_expert", reference=None)

    r1 = CompareCaseResult(case=case1,
        baseline=ModeRun(quality=3.0, total_tokens=100),
        orchestrated=ModeRun(quality=4.0, total_tokens=500),
        quality_gain=1.0, additional_tokens=400, cost_efficiency=0.0025)
    r2 = CompareCaseResult(case=case2,
        baseline=ModeRun(quality=2.0, total_tokens=200),
        orchestrated=ModeRun(quality=3.0, total_tokens=600),
        quality_gain=1.0, additional_tokens=400, cost_efficiency=0.0025)

    agg = _compute_aggregates([r1, r2])
    assert agg["overall"]["n_compared"] == 2
    assert agg["overall"]["mean_quality_gain"] == 1.0
    assert agg["overall"]["sum_additional_tokens"] == 800
    assert agg["overall"]["cost_efficiency"] is not None


def test_compare_cli_help_exits_zero():
    from agent.evaluation.__main__ import main
    try:
        main(["compare", "--help"])
    except SystemExit as e:
        assert e.code == 0


def test_compare_records_worker_decisions_via_observability(tmp_path):
    from agent.config import ObservabilityConfig
    from agent.evaluation.compare import run_compare
    from agent.observability import patch as patch_mod
    from agent.observability import install
    from agent.observability.report_data import build_timeline, group_stages
    from agent.observability.tracing import read_events

    cfg = _config()
    cfg.observability = ObservabilityConfig(enabled=True, data_dir=str(tmp_path / "obs"))

    client = FakeClient(
        [
            "baseline answer",          # baseline
            _FAKE_PLAN,                 # planner
            "worker1",                  # worker 1
            "worker2",                  # worker 2
            "aggregate answer",         # aggregate
            _FAKE_SCORE_PASS,           # internal evaluator
        ],
        usage=[(10, 5, 0)] * 6,
    )
    judge_client = FakeClient([_FAKE_SCORE_PASS, _FAKE_SCORE_PASS], usage=[(5, 2, 0)] * 2)
    try:
        traced, _plugin = install(client, cfg, _domain())
        run_compare(cfg, _domain(), _cases()[:1], traced, judge_client=judge_client)

        events, bad = read_events(tmp_path / "obs")
        assert bad == 0
        decisions = [e for e in events if e["type"] == "decision"]
        worker_decisions = [e for e in decisions if e["phase"].startswith("orchestration.worker")]
        assert len(worker_decisions) == 2
        assert {e["data"]["task"] for e in worker_decisions} == {"t1", "t2"}
        assert {e["data"]["role"] for e in worker_decisions} == {"R1", "R2"}
        planner_decisions = [e for e in decisions if e["phase"] == "orchestration.planner"]
        assert planner_decisions and len(planner_decisions[0]["data"]["tasks"]) == 2

        stages = group_stages(build_timeline(events))
        worker_stages = [st for stages_list in stages.values() for st in stages_list
                         if st.workers]
        assert worker_stages, "expected a worker stage with grouped workers"
        assert sorted((w.number, w.task_title) for w in worker_stages[0].workers) == [(1, "t1"), (2, "t2")]
    finally:
        patch_mod._ACTIVE = None