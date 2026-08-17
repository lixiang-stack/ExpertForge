from agent.evaluation.dataset import Suite, EvalCase
from agent.evaluation.metrics import _accuracy, compute_metrics
from agent.evaluation.runner import CaseResult


def _case(cid, domain="software_engineering", intent="faq", complexity="simple",
          strategy="direct", orchestrate=False):
    return EvalCase(
        id=cid, question=f"q {cid}",
        expected_domain=domain, expected_intent=intent,
        expected_complexity=complexity, expected_strategy=strategy,
        expected_orchestrate=orchestrate, answer_quality=True, reference=None,
    )


def _result(case, *, in_domain=True, intent=None, complexity=None, strategy=None,
            orchestrate=False, actual_model=None, expected_model=None, scorecard=None,
            in_tokens=10, out_tokens=5, cache_tokens=1, latency=10.0, error=None):
    return CaseResult(
        case=case, in_domain=in_domain,
        intent=case.expected_intent if intent is None else intent,
        complexity=case.expected_complexity if complexity is None else complexity,
        strategy=case.expected_strategy if strategy is None else strategy,
        orchestrate=orchestrate, answer="a", actual_model=actual_model,
        expected_model=expected_model, scorecard=scorecard,
        llm_calls=2, in_tokens=in_tokens, out_tokens=out_tokens,
        total_tokens=in_tokens + out_tokens, cache_tokens=cache_tokens, latency_ms=latency,
        error=error,
    )


def _m(cases, results):
    return compute_metrics(Suite(name="direct", domain="software_engineering", cases=cases), results)


def test_accuracy_none_when_empty():
    assert _accuracy(0, 0) is None
    assert _accuracy(3, 4) == 0.75


def test_per_complexity_accuracy():
    cases = [_case("a", complexity="simple"), _case("b", complexity="medium"),
             _case("c", complexity="complex")]
    results = [
        _result(cases[0], complexity="simple"),
        _result(cases[1], complexity="complex"),  # wrong
        _result(cases[2], complexity="complex"),
    ]
    m = _m(cases, results)
    pc = m["classification"]["per_complexity"]
    assert pc["simple"] == 1.0
    assert pc["medium"] == 0.0
    assert pc["complex"] == 1.0
    assert list(pc) == ["simple", "medium", "complex"]


def test_perfect_classification_and_routing():
    cases = [_case("a"), _case("b")]
    results = [_result(cases[0]), _result(cases[1])]
    m = _m(cases, results)
    assert m["n_cases"] == 2
    assert m["classification"]["domain_accuracy"] == 1.0
    assert m["classification"]["intent_accuracy"] == 1.0
    assert m["classification"]["complexity_accuracy"] == 1.0
    assert m["routing"]["strategy_accuracy"] == 1.0
    assert m["routing"]["orchestration_accuracy"] == 1.0
    assert m["routing"]["model_routing_accuracy"] is None  # no actual/expected models


def test_wrong_intent_counts_intent_only():
    cases = [_case("a", intent="faq"), _case("b", intent="faq")]
    results = [
        _result(cases[0], intent="faq"),
        _result(cases[1], intent="concept_explain"),
    ]
    m = _m(cases, results)
    assert m["classification"]["intent_accuracy"] == 0.5
    assert m["classification"]["domain_accuracy"] == 1.0  # both in-domain
    assert m["classification"]["complexity_accuracy"] == 1.0
    assert m["classification"]["per_intent"]["faq"] == 0.5


def test_out_of_domain_affects_domain_and_strategy_only():
    cases = [_case("a"), _case("ood", domain="other", intent=None, complexity=None,
                      strategy="reject")]
    results = [
        _result(cases[0]),
        _result(cases[1], in_domain=False, intent=None, complexity=None, strategy="reject"),
    ]
    m = _m(cases, results)
    assert m["classification"]["domain_accuracy"] == 1.0
    # intent/complexity exclude the out-of-domain case
    assert m["classification"]["intent_accuracy"] == 1.0
    assert m["classification"]["complexity_accuracy"] == 1.0
    assert m["routing"]["strategy_accuracy"] == 1.0
    # by_path excludes out-of-domain
    assert "simple" in m["cost"]["by_path"]
    assert set(m["cost"]["by_path"]) == {"simple"}


def test_out_of_domain_wrong_in_domain_prediction():
    cases = [_case("ood", domain="other", intent=None, complexity=None, strategy="reject")]
    results = [_result(cases[0], in_domain=True, strategy="direct")]
    m = _m(cases, results)
    assert m["classification"]["domain_accuracy"] == 0.0
    assert m["routing"]["strategy_accuracy"] == 0.0


def test_orchestration_accuracy():
    cases = [_case("a", orchestrate=True), _case("b", orchestrate=False)]
    results = [_result(cases[0], orchestrate=True), _result(cases[1], orchestrate=False)]
    m = _m(cases, results)
    assert m["routing"]["orchestration_accuracy"] == 1.0


def test_model_routing_accuracy():
    cases = [_case("a"), _case("b")]
    results = [
        _result(cases[0], actual_model="low-a", expected_model="low-a"),
        _result(cases[1], actual_model="high-a", expected_model="low-a"),
    ]
    m = _m(cases, results)
    assert m["routing"]["model_routing_accuracy"] == 0.5


def test_answer_quality_means():
    cases = [_case("a"), _case("b")]
    results = [
        _result(cases[0], scorecard={"correctness": 4, "relevance": 5, "completeness": 3,
                                     "technical_depth": 4, "practical_usefulness": 5,
                                     "hallucination": 4}),
        _result(cases[1], scorecard=None),
    ]
    m = _m(cases, results)
    assert m["answer_quality"]["correctness"] == 4.0
    assert m["answer_quality"]["relevance"] == 5.0


def test_cost_aggregates_and_by_path():
    cases = [
        _case("s", complexity="simple"),
        _case("m", complexity="medium"),
        _case("c", complexity="complex"),
    ]
    results = [_result(cases[0], in_tokens=10, out_tokens=5, cache_tokens=1, latency=10.0),
               _result(cases[1], in_tokens=20, out_tokens=8, cache_tokens=2, latency=20.0),
               _result(cases[2], in_tokens=30, out_tokens=12, cache_tokens=3, latency=30.0)]
    m = _m(cases, results)
    cost = m["cost"]
    assert cost["llm_calls"] == 6
    assert cost["in_tokens"] == 60
    assert cost["out_tokens"] == 25
    assert cost["total_tokens"] == 85
    assert cost["cache_tokens"] == 6
    assert set(cost["by_path"]) == {"simple", "medium", "complex"}
    assert cost["by_path"]["simple"]["in_tokens"] == 10
    assert cost["by_path"]["complex"]["total_tokens"] == 42


def test_n_failed_counts_error_cases():
    cases = [_case("a"), _case("b")]
    results = [_result(cases[0]), _result(cases[1], error="LLMError: boom")]
    m = _m(cases, results)
    assert m["n_failed"] == 1


def test_n_failed_zero_without_errors():
    cases = [_case("a")]
    m = _m(cases, [_result(cases[0])])
    assert m["n_failed"] == 0
