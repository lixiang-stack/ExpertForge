import json

from agent.evaluation.dataset import Suite, EvalCase
from agent.evaluation.metrics import compute_metrics, compute_metrics_by_tier, failed_cases
from agent.evaluation.report import format_summary, serialize_results, write_result
from agent.evaluation.runner import CaseResult


def _case(cid):
    return EvalCase(
        id=cid, question=f"q {cid}",
        expected_domain="software_engineering", expected_intent="faq",
        expected_complexity="simple", expected_strategy="direct",
        expected_orchestrate=False, tier="classification", reference=None,
    )


def _result(case):
    return CaseResult(
        case=case, in_domain=True, intent="faq", complexity="simple",
        strategy="direct", orchestrate=False, answer="the answer",
        actual_model="low-a", expected_model="low-a",
        scorecard={"correctness": 4, "relevance": 5, "completeness": 4,
                   "technical_depth": 4, "practical_usefulness": 5, "hallucination": 5},
        suite="direct", tier="classification", llm_calls=2, in_tokens=10,
        out_tokens=5, total_tokens=15, cache_tokens=1, latency_ms=10.0,
    )


def _record():
    cases = [_case("a")]
    results = [_result(cases[0])]
    suite = Suite(name="direct", domain="software_engineering", cases=cases)
    m = compute_metrics(suite, results)
    by_tier = compute_metrics_by_tier(cases, results, domain="software_engineering")
    return serialize_results(
        results, m, by_tier, domain="software_engineering", label="run1",
        model="m", judge_model="judge-a", tiers=["classification"], smoke_only=True,
        dataset_path="evaluation/datasets/software_engineering",
        failed_cases=[],
    )


def test_serialize_results_contains_expected_keys():
    rec = _record()
    assert rec["domain"] == "software_engineering"
    assert rec["label"] == "run1"
    assert rec["model"] == "m"
    assert rec["judge_model"] == "judge-a"
    assert rec["smoke_only"] is True
    assert rec["tiers"] == ["classification"]
    assert rec["metrics"]["n_cases"] == 1
    case = rec["cases"][0]
    assert case["id"] == "a"
    assert case["question"] == "q a"
    assert case["intent"] == "faq"
    assert case["complexity"] == "simple"
    assert case["strategy"] == "direct"
    assert case["scorecard"]["correctness"] == 4
    assert case["llm_calls"] == 2
    assert case["in_tokens"] == 10


def test_write_result_creates_json(tmp_path):
    rec = _record()
    path = write_result(str(tmp_path / "results"), rec, label="run1")
    assert path.endswith("run1.json")
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["label"] == "run1"
    assert loaded["metrics"]["n_cases"] == 1


def test_format_summary_contains_key_sections():
    text = format_summary(_record())
    assert "run1" in text
    assert "classification" in text.lower()
    assert "routing" in text.lower()
    assert "answer quality" in text.lower()
    assert "cost" in text.lower()
    assert "domain_accuracy" in text
    assert "intent_accuracy" in text
    assert "strategy_accuracy" in text
    assert "model_routing_accuracy" in text
    assert "correctness" in text
    assert "total=" in text
    assert "simple" in text


def test_serialize_results_has_tiers_and_metrics_by_tier():
    rec = _record()
    assert rec["tiers"] == ["classification"]
    assert rec["metrics_by_tier"]["classification"]["n_cases"] == 1
    assert rec["failed_cases"] == []
    case = rec["cases"][0]
    assert case["suite"] == "direct"
    assert case["tier"] == "classification"


def test_format_summary_has_per_tier_section():
    text = format_summary(_record())
    assert "Per-tier" in text
    assert "classification" in text


def test_format_summary_has_per_complexity():
    text = format_summary(_record())
    assert "per_complexity" in text
    assert "simple:" in text


def test_case_record_includes_error():
    case = _case("a")
    r = _result(case)
    r.error = "LLMError: boom"
    record = serialize_results(
        [r], {}, {}, domain="software_engineering", label="run1",
        model="m", judge_model="judge-a", tiers=[], smoke_only=True,
        dataset_path="evaluation/datasets/software_engineering", failed_cases=[],
    )
    assert record["cases"][0]["error"] == "LLMError: boom"


def test_case_record_error_none_by_default():
    case = _case("a")
    record = serialize_results(
        [_result(case)], {}, {}, domain="software_engineering", label="run1",
        model="m", judge_model="judge-a", tiers=[], smoke_only=True,
        dataset_path="evaluation/datasets/software_engineering", failed_cases=[],
    )
    assert record["cases"][0]["error"] is None


def test_format_summary_shows_failed_cases():
    case = _case("a")
    r = _result(case)
    r.error = "LLMError: boom"
    suite = Suite(name="direct", domain="software_engineering", cases=[case])
    m = compute_metrics(suite, [r])
    by_tier = compute_metrics_by_tier([case], [r], domain="software_engineering")
    failed = failed_cases([r], "software_engineering")
    record = serialize_results(
        [r], m, by_tier, domain="software_engineering", label="run1",
        model="m", judge_model="judge-a", tiers=["classification"], smoke_only=True,
        dataset_path="evaluation/datasets/software_engineering", failed_cases=failed,
    )
    text = format_summary(record)
    assert "Failed cases: 1" in text
    assert "LLMError: boom" in text
