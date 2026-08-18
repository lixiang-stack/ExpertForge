import json

from agent import agent_cli
from agent.llm import ChatResult


def _write_root_config(tmp_path, domain_dir):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://x/v1",
        "model": "m",
        "domain_dir": domain_dir,
        "provider": "test",
        "provider_capabilities": {},
    }), encoding="utf-8")
    return str(path)


def _write_domain(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(json.dumps({
        "name": "软件工程", "description": "d", "out_of_domain_reply": "Out.",
    }, ensure_ascii=False), encoding="utf-8")
    (base / "intents.yaml").write_text("- id: faq\n  description: quick\n", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text(
        "Direct self-contained", encoding="utf-8"
    )
    (base / "prompts" / "unsupported_complex.md").write_text("unsupported", encoding="utf-8")
    return str(base)


def test_main_missing_config_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    assert agent_cli.main([str(tmp_path / "no-such.json")]) == 1
    assert "Config error" in capsys.readouterr().err


def test_main_missing_api_key_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_path = _write_root_config(tmp_path, _write_domain(tmp_path))
    assert agent_cli.main([str(config_path)]) == 1
    assert "AGENT_API_KEY" in capsys.readouterr().err


def test_main_help_exits_0(capsys):
    assert agent_cli.main(["-h"]) == 0
    assert "Usage" in capsys.readouterr().out


def test_main_runs_repl(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_path = _write_root_config(tmp_path, _write_domain(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt="": iter(["exit"]).__next__())

    class FakeClient:
        def chat_completion(
            self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
        ):
            return ChatResult(
                text='{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
                model=model or "m",
            )

    monkeypatch.setattr(agent_cli, "LLMClient", lambda *a, **k: FakeClient())
    assert agent_cli.main([str(config_path)]) == 0
    assert "ExpertForge" in capsys.readouterr().out


def test_main_ask_prints_answer(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_path = _write_root_config(tmp_path, _write_domain(tmp_path))

    class FakeClient:
        def __init__(self, *a, **k):
            self.responses = [
                '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
                "one-shot answer",
            ]

        def chat_completion(
            self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
        ):
            return ChatResult(text=self.responses.pop(0), model=model or "m")

    monkeypatch.setattr(agent_cli, "LLMClient", lambda *a, **k: FakeClient())
    assert agent_cli.main([str(config_path), "--ask", "what is defer"]) == 0
    assert "one-shot answer" in capsys.readouterr().out


def test_main_passes_provider_and_capability_overrides_from_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_path = _write_root_config(tmp_path, _write_domain(tmp_path))
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    data["provider"] = "gemini"
    data["provider_capabilities"] = {"supports_json_schema": True}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    captured = {}

    class FakeClient:
        def chat_completion(
            self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
        ):
            return ChatResult(
                text='{"in_domain": false, "intent": null, "complexity": null, "reason": "x"}',
                model=model or "m",
            )

    monkeypatch.setattr(agent_cli, "LLMClient",
                        lambda *a, **k: captured.update(k) or FakeClient())
    assert agent_cli.main([str(config_path), "--ask", "hi"]) == 0
    assert captured["provider"] == "gemini"
    assert captured["capability_overrides"] == {"supports_json_schema": True}
