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
    (dataset_dir / "software_engineering.yaml").write_text(
        'domain: software_engineering\n'
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    category: knowledge\n'
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
        "--dataset", str(dataset_dir / "software_engineering.yaml"),
        "--label", "x",
        "--results-dir", str(results_dir),
        "--skip-quality",
    ])
    assert rc == 0
    text = out.getvalue()
    assert "domain_accuracy" in text
    assert "x.json" in text


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
