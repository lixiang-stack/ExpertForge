import json

from agent.evaluation import __main__ as eval_main
from agent.llm import ChatResult


def test_main_run_prints_summary_and_writes_file(tmp_path, monkeypatch):
    domain_dir = tmp_path / "software_engineering"
    domain_dir.mkdir()
    (domain_dir / "domain.json").write_text(
        '{"name": "sw", "description": "d", "out_of_domain_reply": "Out."}',
        encoding="utf-8",
    )
    (domain_dir / "intents.yaml").write_text("- id: faq\n  description: quick\n", encoding="utf-8")
    (domain_dir / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (domain_dir / "prompts").mkdir()
    (domain_dir / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: complex\nintents:\n  - faq\n",
        encoding="utf-8",
    )
    (domain_dir / "prompts" / "direct.md").write_text("d", encoding="utf-8")

    dataset_dir = tmp_path / "evaluation" / "datasets"
    dataset_dir.mkdir(parents=True)
    suite_dir = dataset_dir / "software_engineering"
    suite_dir.mkdir()
    (suite_dir / "faq.yaml").write_text(
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    tier: classification\n'
        '    smoke: true\n'
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
        f'{{"base_url": "https://x", "model": "m", "domain_dir": "{domain_dir}", '
        f'"provider": "test", "provider_capabilities": {{}}}}',
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat_completion(self, messages, model=None, temperature=0.3,
                            disable_thinking=False, json_mode=False, json_schema=None):
            return ChatResult(
                text='{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
                model=model or "m",
            )

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
    (domain_dir / "prompts").mkdir()
    (domain_dir / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: complex\nintents:\n  - faq\n",
        encoding="utf-8",
    )
    (domain_dir / "prompts" / "direct.md").write_text("d", encoding="utf-8")

    dataset_dir = tmp_path / "evaluation" / "datasets"
    dataset_dir.mkdir(parents=True)
    suite_dir = dataset_dir / "software_engineering"
    suite_dir.mkdir()
    (suite_dir / "faq.yaml").write_text(
        'cases:\n'
        '  - id: a\n'
        '    question: "q"\n'
        '    tier: classification\n'
        '    smoke: true\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n'
        '  - id: a2\n'
        '    question: "q2"\n'
        '    tier: routing\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: faq\n'
        '      complexity: simple\n'
        '      strategy: direct\n',
        encoding="utf-8",
    )
    (suite_dir / "concept_explain.yaml").write_text(
        'cases:\n'
        '  - id: b\n'
        '    question: "q"\n'
        '    tier: classification\n'
        '    expected:\n'
        '      domain: software_engineering\n'
        '      intent: concept_explain\n'
        '      complexity: simple\n'
        '      strategy: teaching\n',
        encoding="utf-8",
    )

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        '{"base_url": "https://x", "model": "m", "domain_dir": "%s", '
        '"provider": "test", "provider_capabilities": {}}'
        % domain_dir,
        encoding="utf-8",
    )
    return config_path, suite_dir


def _run_with_fake(monkeypatch, argv):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat_completion(self, messages, model=None, temperature=0.3,
                            disable_thinking=False, json_mode=False, json_schema=None):
            return ChatResult(
                text='{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
                model=model or "m",
            )

    monkeypatch.setattr(eval_main, "LLMClient", FakeClient)
    import io
    import sys
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main(argv)
    assert rc == 0
    return out.getvalue()


def test_main_run_default_is_smoke(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    out = _run_with_fake(monkeypatch, [
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--label", "smoke", "--results-dir", str(tmp_path / "r"),
    ])
    assert "cases=1" in out  # only the smoke case runs
    assert "selection=smoke" in out
    assert "Per-tier" in out
    assert "classification: n=1" in out


def test_main_run_tier_selection(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    out = _run_with_fake(monkeypatch, [
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--tier", "classification", "--label", "cls", "--results-dir", str(tmp_path / "r"),
    ])
    assert "cases=2" in out  # a + b
    assert "selection=tiers: classification" in out


def test_main_run_tier_all(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    out = _run_with_fake(monkeypatch, [
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--tier", "all", "--label", "all", "--results-dir", str(tmp_path / "r"),
    ])
    assert "cases=3" in out


def test_main_run_no_matching_tier_returns_1(tmp_path, monkeypatch, capsys):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    rc = eval_main.main([
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--tier", "full_expert", "--results-dir", str(tmp_path / "r"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "No cases match the selection" in err


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


def test_main_passes_provider_and_capability_overrides_from_config(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    data["provider"] = "gemini"
    data["provider_capabilities"] = {"supports_json_schema": True}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    monkeypatch.setenv("AGENT_API_KEY", "k")

    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def chat_completion(self, messages, model=None, temperature=0.3,
                            disable_thinking=False, json_mode=False, json_schema=None):
            return ChatResult(
                text='{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
                model=model or "m",
            )

    monkeypatch.setattr(eval_main, "LLMClient", FakeClient)
    import io
    import sys

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main([
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--label", "pc", "--results-dir", str(tmp_path / "r"),
    ])
    assert rc == 0
    assert captured["provider"] == "gemini"
    assert captured["capability_overrides"] == {"supports_json_schema": True}


def test_main_judge_client_gets_capabilities_from_judge_config(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    data["evaluation"] = {
        "judge": {
            "base_url": "https://j",
            "model": "judge-a",
            "provider": "judge-prov",
            "provider_capabilities": {"supports_thinking_toggle": True,
                                      "supports_json_schema": True},
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.setenv("AGENT_JUDGE_API_KEY", "judge-env")

    captured = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)

        def chat_completion(self, messages, model=None, temperature=0.3,
                            disable_thinking=False, json_mode=False, json_schema=None):
            return ChatResult(
                text='{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
                model=model or "m",
            )

    monkeypatch.setattr(eval_main, "LLMClient", FakeClient)
    import io
    import sys

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main([
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--label", "jc", "--results-dir", str(tmp_path / "r"),
    ])
    assert rc == 0
    assert len(captured) == 2  # main client + judge client
    main_client, judge_client = captured
    assert main_client["provider"] == "test"
    assert judge_client["base_url"] == "https://j"
    assert judge_client["model"] == "judge-a"
    assert judge_client["provider"] == "judge-prov"
    assert judge_client["api_key"] == "judge-env"
    assert judge_client["capability_overrides"] == {
        "supports_json_schema": True,
        "supports_thinking_toggle": True,
        "supports_tool_call": False,
    }


def test_main_judge_client_capabilities_fall_back_to_top_level(tmp_path, monkeypatch):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    data["provider_capabilities"] = {"supports_thinking_toggle": True}
    data["evaluation"] = {"judge": {"base_url": "https://j", "model": "judge-a"}}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.setenv("AGENT_JUDGE_API_KEY", "judge-env")

    captured = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)

        def chat_completion(self, messages, model=None, temperature=0.3,
                            disable_thinking=False, json_mode=False, json_schema=None):
            return ChatResult(
                text='{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
                model=model or "m",
            )

    monkeypatch.setattr(eval_main, "LLMClient", FakeClient)
    import io
    import sys

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    rc = eval_main.main([
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--label", "jc", "--results-dir", str(tmp_path / "r"),
    ])
    assert rc == 0
    assert len(captured) == 2
    main_client, judge_client = captured
    assert main_client["provider"] == "test"
    assert main_client["api_key"] == "k"
    assert judge_client["model"] == "judge-a"
    assert judge_client["provider"] == "test"  # judge provider falls back too
    assert judge_client["api_key"] == "judge-env"  # own key, not the main client's
    assert judge_client["capability_overrides"] == {"supports_thinking_toggle": True}


def test_main_judge_client_missing_key_returns_1(tmp_path, monkeypatch, capsys):
    config_path, suite_dir = _suite_cli_env(tmp_path)
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    data["evaluation"] = {"judge": {"base_url": "https://j", "model": "judge-a"}}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_JUDGE_API_KEY", raising=False)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat_completion(self, messages, model=None, temperature=0.3,
                            disable_thinking=False, json_mode=False, json_schema=None):
            return ChatResult(
                text='{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
                model=model or "m",
            )

    monkeypatch.setattr(eval_main, "LLMClient", FakeClient)
    rc = eval_main.main([
        "run", "--config", str(config_path), "--dataset", str(suite_dir),
        "--label", "jc", "--results-dir", str(tmp_path / "r"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Config error:" in err
    assert "AGENT_JUDGE_API_KEY" in err


def _result_record(domain_accuracy):
    return {
        "domain": "software_engineering",
        "label": "run",
        "model": "m",
        "judge_model": None,
        "smoke_only": False,
        "dataset": "evaluation/datasets/software_engineering",
        "tiers": ["classification", "routing"],
        "metrics": {
            "n_cases": 2,
            "classification": {"domain_accuracy": domain_accuracy, "intent_accuracy": 1.0,
                               "complexity_accuracy": 1.0, "per_intent": {}},
            "routing": {"strategy_accuracy": 1.0, "per_strategy": {},
                        "orchestration_accuracy": 1.0, "model_routing_accuracy": 1.0},
            "answer_quality": {},
            "cost": {"llm_calls": 2, "in_tokens": 10, "out_tokens": 5,
                     "total_tokens": 15, "cache_tokens": 0, "latency_ms": 20.0,
                     "by_path": {}},
        },
        "metrics_by_tier": {
            "classification": {"n_cases": 1, "classification": {"domain_accuracy": domain_accuracy},
                               "routing": {}, "cost": {}},
            "routing": {"n_cases": 1, "classification": {"domain_accuracy": domain_accuracy},
                        "routing": {}, "cost": {}},
            "full_expert": {"n_cases": 0, "classification": {}, "routing": {}, "cost": {}},
        },
        "failed_cases": [],
        "cases": [{"id": "a", "suite": "faq", "tier": "classification"},
                  {"id": "a2", "suite": "faq", "tier": "routing"}],
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
    assert baseline["metrics_by_tier"]["classification"]["n_cases"] == 1
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
    (domain_dir / "prompts").mkdir()
    (domain_dir / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: complex\nintents:\n  - faq\n",
        encoding="utf-8",
    )
    (domain_dir / "prompts" / "direct.md").write_text("d", encoding="utf-8")

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        f'{{"base_url": "https://x", "model": "m", "domain_dir": "{domain_dir}", '
        f'"provider": "test", "provider_capabilities": {{}}}}',
        encoding="utf-8",
    )
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    rc = eval_main.main(["run", "--config", str(config_path),
                         "--dataset", "/nonexistent/dataset.yaml"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Dataset error" in err
