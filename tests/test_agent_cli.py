from agent import agent_cli


def test_main_missing_config_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    code = agent_cli.main([str(tmp_path / "no-such.json")])
    assert code == 1
    err = capsys.readouterr().err
    assert "Config error" in err


def test_main_missing_api_key_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"base_url": "https://x/v1", "model": "m", '
        '"domain": {"description": "d"}}',
        encoding="utf-8",
    )
    code = agent_cli.main([str(config_file)])
    assert code == 1
    err = capsys.readouterr().err
    assert "AGENT_API_KEY" in err


def test_main_help_exits_0(capsys):
    assert agent_cli.main(["-h"]) == 0
    out = capsys.readouterr().out
    assert "Usage" in out


def test_main_runs_repl(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"base_url": "https://x/v1", "model": "m", '
        '"domain": {"name": "软件工程", "description": "d", '
        '"out_of_domain_reply": "Out of domain."}}',
        encoding="utf-8",
    )

    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    class FakeClient:
        def chat_completion(self, messages, model=None, disable_thinking=False):
            return '{"in_domain": true, "reason": "ok"}'

        def chat_completion_stream(self, messages, model=None):
            return iter(["你好"])

    monkeypatch.setattr(agent_cli, "LLMClient", lambda *a, **k: FakeClient())
    assert agent_cli.main([str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "ExpertForge" in out
