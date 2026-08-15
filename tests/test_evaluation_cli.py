import json

import pytest

from agent.evaluation import __main__ as eval_main


def test_main_run_prints_summary_and_writes_file(tmp_path, monkeypatch):
    domain_dir = tmp_path / "software_engineering"
    domain_dir.mkdir()
    (domain_dir / "domain.json").write_text(
        '{"name": "sw", "description": "d", "out_of_domain_reply": "Out."}',
        encoding="utf-8",
    )
    (domain_dir / "intents.yaml").write_text("- id: faq\n  description: quick\n", encoding="utf-8")
    (domain_dir / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (domain_dir / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (domain_dir / "prompts").mkdir()
    (domain_dir / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (domain_dir / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")

    dataset_dir = tmp_path / "evaluation" / "datasets"
    dataset_dir.mkdir(parents=True)
    suite_dir = dataset_dir / "software_engineering"
    suite_dir.mkdir()
    (suite_dir / "direct.yaml").write_text(
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    answer_quality: false\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        f'{{"base_url": "https://x", "model": "m", "domain_dir": "{domain_dir}"}}',
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._usage_local = __import__("threading").local()

        def chat_completion(self, messages, model=None, temperature=0.3,
                            disable_thinking=False, json_mode=False, json_schema=None):
            self._usage_local.usage = None
            return '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}'

        def chat_completion_stream(self, messages, **kwargs):
            return iter([])

    monkeypatch.setattr(eval_main, "LLMClient", FakeClient)
    monkeypatch.setenv("AGENT_API_KEY", "k")

    import io
    import sys

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main([
        "run",
        "--config", str(config_path),
        "--dataset", str(suite_dir),
        "--label", "x",
        "--results-dir", str(results_dir),
        "--skip-quality",
    ])
    assert rc == 0
    text = out.getvalue()
    assert "domain_accuracy" in text
    assert "x.json" in text


def _suite_cli_env(tmp_path):
    domain_dir = tmp_path / "software_engineering"
    domain_dir.mkdir()
    (domain_dir / "domain.json").write_text(
        '{"name": "sw", "description": "d", "out_of_domain_reply": "Out."}',
        encoding="utf-8",
    )
    (domain_dir / "intents.yaml").write_text("- id: faq\n  description: quick\n", encoding="utf-8")
    (domain_dir / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (domain_dir / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (domain_dir / "prompts").mkdir()
    (domain_dir / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (domain_dir / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")

    dataset_dir = tmp_path / "evaluation" / "datasets"
    dataset_dir.mkdir(parents=True)
    suite_dir = dataset_dir / "software_engineering"
    suite_dir.mkdir()
    (suite_dir / "direct.yaml").write_text(
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    answer_quality: false\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n'
        '  - id: a2\n'
        '    question: "q2"\n'
        '    answer_quality: false\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )
    (suite_dir / "teaching.yaml").write_text(
        'cases:\n'
        '  - id: b\n'
        '    question: "q"\n'
        '    answer_quality: false\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        f'{{"base_url": "https://x", "model": "m", "domain_dir": "{domain_dir}"}}',
        encoding="utf-8",
    )
    return config_path, suite_dir


def _run_with_fake(monkeypatch, argv):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._usage_local = __import__("threading").local()

        def chat_completion(self, messages, model=None, temperature=0.3,
                            disable_thinking=False, json_mode=False, json_schema=None):
            self._usage_local.usage = None
            return '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}'

        def chat_completion_stream(self, messages, **kwargs):
            return iter([])

    monkeypatch.setattr(eval_main, "LLMClient", FakeClient)
    import io
    import sys
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main(argv)
    assert rc == 0
    return out.getvalue()


def test_main_run_suite_selection(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    out = _run_with_fake(monkeypatch, [
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--suite", "direct", "--label", "sel", "--results-dir", str(tmp_path / "r"),
        "--skip-quality",
    ])
    assert "Per-suite" in out
    assert "direct" in out
    assert "teaching" not in out


def test_main_run_max_per_suite(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    out = _run_with_fake(monkeypatch, [
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--max-per-suite", "1", "--label", "mx", "--results-dir", str(tmp_path / "r"),
        "--skip-quality",
    ])
    assert "Per-suite" in out
    assert "direct: n=1" in out  # 2-case direct suite truncated to 1
    assert "teaching: n=1" in out


def test_main_max_per_suite_lt_1_returns_1(monkeypatch, capsys):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    rc = eval_main.main(["run", "--max-per-suite", "0"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--max-per-suite must be >= 1" in err


def test_main_diff(tmp_path, monkeypatch):
    import json

    results_dir = tmp_path / "r"
    results_dir.mkdir()
    a = results_dir / "a.json"
    b = results_dir / "b.json"
    record = {
        "label": "x", "model": "m",
        "metrics": {
            "classification": {"domain_accuracy": 1.0, "intent_accuracy": 1.0,
                               "complexity_accuracy": 1.0, "per_intent": {}},
            "routing": {"strategy_accuracy": 1.0, "orchestration_accuracy": 1.0,
                        "model_routing_accuracy": 1.0},
            "answer_quality": {"correctness": 4.0},
            "cost": {"llm_calls": 2, "in_tokens": 10, "out_tokens": 5,
                     "total_tokens": 15, "cache_tokens": 0, "latency_ms": 20.0,
                     "by_path": {}},
        },
    }
    a.write_text(json.dumps(record), encoding="utf-8")
    record["metrics"]["classification"]["domain_accuracy"] = 0.5
    b.write_text(json.dumps(record), encoding="utf-8")

    import io
    import sys

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main(["diff", str(a), str(b)])
    assert rc == 0
    assert "domain_accuracy" in out.getvalue()


def _result_record(domain_accuracy):
    return {
        "domain": "software_engineering",
        "label": "run",
        "model": "m",
        "judge_model": None,
        "skip_quality": True,
        "dataset": "evaluation/datasets/software_engineering",
        "suites": ["direct", "teaching"],
        "metrics": {
            "n_cases": 2,
            "classification": {"domain_accuracy": domain_accuracy, "intent_accuracy": 1.0,
                               "complexity_accuracy": 1.0, "per_intent": {}},
            "routing": {"strategy_accuracy": 1.0, "orchestration_accuracy": 1.0,
                        "model_routing_accuracy": 1.0},
            "answer_quality": {},
            "cost": {"llm_calls": 2, "in_tokens": 10, "out_tokens": 5,
                     "total_tokens": 15, "cache_tokens": 0, "latency_ms": 20.0,
                     "by_path": {}},
        },
        "metrics_by_suite": {
            "direct": {"n_cases": 1, "classification": {"domain_accuracy": domain_accuracy},
                       "routing": {}, "cost": {}},
            "teaching": {"n_cases": 1, "classification": {"domain_accuracy": domain_accuracy},
                         "routing": {}, "cost": {}},
        },
        "cases": [{"id": "a", "suite": "direct"}, {"id": "b", "suite": "teaching"}],
    }


def test_main_baseline_writes_slim_file(tmp_path, monkeypatch, capsys):
    import io
    import sys

    src = tmp_path / "src.json"
    src.write_text(json.dumps(_result_record(1.0)), encoding="utf-8")
    out_path = tmp_path / "baseline.json"
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main(["baseline", str(src), "--out", str(out_path)])
    assert rc == 0
    baseline = json.loads(out_path.read_text(encoding="utf-8"))
    assert "cases" not in baseline
    assert baseline["metrics"]["classification"]["domain_accuracy"] == 1.0
    assert baseline["metrics_by_suite"]["direct"]["n_cases"] == 1
    assert "Baseline written" in out.getvalue()


def test_main_baseline_prints_delta_when_previous_exists(tmp_path, monkeypatch, capsys):
    import io
    import sys

    out_path = tmp_path / "baseline.json"
    src = tmp_path / "src.json"
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    src.write_text(json.dumps(_result_record(1.0)), encoding="utf-8")
    assert eval_main.main(["baseline", str(src), "--out", str(out_path)]) == 0
    src.write_text(json.dumps(_result_record(0.5)), encoding="utf-8")
    rc = eval_main.main(["baseline", str(src), "--out", str(out_path)])
    assert rc == 0
    text = out.getvalue()
    assert "domain_accuracy" in text
    assert "-0.50" in text


def test_main_baseline_missing_file_returns_1(tmp_path, monkeypatch, capsys):
    rc = eval_main.main(["baseline", str(tmp_path / "nope.json"),
                         "--out", str(tmp_path / "baseline.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Baseline error" in err


def test_main_missing_config_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    rc = eval_main.main(["run", "--config", "/nonexistent/config.json"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Config error" in err


def test_main_bad_dataset_returns_1(tmp_path, monkeypatch, capsys):
    domain_dir = tmp_path / "software_engineering"
    domain_dir.mkdir()
    (domain_dir / "domain.json").write_text(
        '{"name": "sw", "description": "d", "out_of_domain_reply": "Out."}',
        encoding="utf-8",
    )
    (domain_dir / "intents.yaml").write_text("- id: faq\n  description: quick\n", encoding="utf-8")
    (domain_dir / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (domain_dir / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (domain_dir / "prompts").mkdir()
    (domain_dir / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (domain_dir / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        f'{{"base_url": "https://x", "model": "m", "domain_dir": "{domain_dir}"}}',
        encoding="utf-8",
    )
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    rc = eval_main.main(["run", "--config", str(config_path),
                         "--dataset", "/nonexistent/dataset.yaml"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Dataset error" in err
