import json

from agent.evaluation.diff import diff_runs, load_result


def _run(label, domain_acc, intent_acc, correctness, total_tokens):
    return {
        "label": label,
        "model": "m",
        "metrics": {
            "n_cases": 2,
            "classification": {
                "domain_accuracy": domain_acc,
                "intent_accuracy": intent_acc,
                "complexity_accuracy": 1.0,
                "per_intent": {"faq": 1.0},
            },
            "routing": {
                "strategy_accuracy": 1.0,
                "orchestration_accuracy": 1.0,
                "model_routing_accuracy": 1.0,
            },
            "answer_quality": {"correctness": correctness},
            "cost": {
                "llm_calls": 4, "in_tokens": 100, "out_tokens": 50,
                "total_tokens": total_tokens, "cache_tokens": 10,
                "latency_ms": 500.0, "by_path": {},
            },
        },
    }


def test_diff_shows_deltas():
    a = _run("a", 1.0, 0.5, 4.0, 100)
    b = _run("b", 0.75, 0.75, 4.5, 120)
    text = diff_runs(a, b)
    assert "domain_accuracy" in text
    assert "0.75" in text
    assert "intent_accuracy" in text
    assert "correctness" in text
    assert "total_tokens" in text


def test_diff_handles_none_accuracy():
    a = _run("a", None, 0.5, 4.0, 100)
    b = _run("b", 0.75, None, None, 120)
    text = diff_runs(a, b)
    assert "n/a" in text


def test_diff_shows_per_complexity():
    a = _run("a", 1.0, 0.5, 4.0, 100)
    a["metrics"]["classification"]["per_complexity"] = {"simple": 1.0, "medium": 0.5}
    b = _run("b", 1.0, 0.5, 4.0, 100)
    b["metrics"]["classification"]["per_complexity"] = {"simple": 1.0, "medium": 1.0}
    text = diff_runs(a, b)
    assert "per_complexity" in text
    assert "medium" in text


def test_load_result_roundtrip(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_run("x", 1.0, 1.0, 5.0, 10)), encoding="utf-8")
    rec = load_result(str(path))
    assert rec["label"] == "x"


def test_load_result_bad_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ not json", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        load_result(str(path))
