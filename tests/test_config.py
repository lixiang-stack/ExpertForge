import json

import pytest

from agent.config import (
    AgentConfig,
    ConfigError,
    DomainConfig,
    IntentDef,
    StrategyDef,
    get_api_key,
    load_config,
    load_domain_config,
)


def _write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_load_config_basic(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert isinstance(cfg, AgentConfig)
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.model == "model-a"
    assert cfg.classifier_model == "model-a"
    assert cfg.domain_dir == "domain/software_engineering"


def test_load_config_model_tiers(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "model_low": "low-a",
        "model_high": "high-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.model_low == "low-a"
    assert cfg.model_high == "high-a"


def test_model_tiers_empty_string_become_none(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "model_low": "",
        "model_high": "",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.model_low is None
    assert cfg.model_high is None


def test_model_tiers_absent_become_none(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.model_low is None
    assert cfg.model_high is None


def test_classifier_model_derives_from_model_low(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "model_low": "low-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.classifier_model == "low-a"


def test_legacy_classifier_model_entry_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "classifier_model": "legacy-a",
        "model_low": "low-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.classifier_model == "low-a"  # legacy key ignored


def test_legacy_classifier_model_without_model_low_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "classifier_model": "legacy-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.classifier_model == "model-a"  # legacy key ignored, derives from model


def test_classifier_model_falls_back_to_model(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.classifier_model == "model-a"


def test_env_base_url_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BASE_URL", "https://env.example.com/v1")
    path = _write_config(tmp_path, {
        "base_url": "https://file.example.com/v1",
        "model": "m",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.base_url == "https://env.example.com/v1"


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/path/config.json")


def test_default_config_falls_back_to_example(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.example.json").write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    }), encoding="utf-8")
    cfg = load_config()
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.domain_dir == "domain/software_engineering"


def test_default_config_no_fallback_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError):
        load_config()


def test_explicit_missing_path_no_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.example.json").write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "d",
    }), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "config.json"))


def test_invalid_json_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = tmp_path / "config.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_missing_base_url_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {"model": "m", "domain_dir": "d"})
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_model_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {"base_url": "https://x/v1", "domain_dir": "d"})
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_domain_dir_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {"base_url": "https://x/v1", "model": "m"})
    with pytest.raises(ConfigError):
        load_config(path)


def test_get_api_key(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "secret")
    assert get_api_key() == "secret"


def test_get_api_key_missing(monkeypatch):
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        get_api_key()


def _write_domain(tmp_path, **overrides):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(json.dumps({
        "name": "软件工程",
        "description": "software engineering",
        "out_of_domain_reply": "Out of domain.",
    }, ensure_ascii=False), encoding="utf-8")
    (base / "intents.yaml").write_text(
        "- id: concept_explain\n  description: explain a concept\n"
        "- id: faq\n  description: quick question\n",
        encoding="utf-8",
    )
    (base / "intent_mapping.yaml").write_text(
        "concept_explain: teaching\nfaq: direct\n", encoding="utf-8"
    )
    (base / "strategies.yaml").write_text(
        "teaching:\n  complexity_gate: true\ndirect:\n  model: model-direct\n  default: true\n",
        encoding="utf-8",
    )
    (base / "prompts" / "teaching.md").write_text(
        "teach self-contained", encoding="utf-8"
    )
    (base / "prompts" / "direct.md").write_text(
        "direct self-contained", encoding="utf-8"
    )
    (base / "prompts" / "unsupported_complex.md").write_text("unsupported", encoding="utf-8")
    return str(base)


def test_load_domain_config_basic(tmp_path):
    domain = load_domain_config(_write_domain(tmp_path))
    assert isinstance(domain, DomainConfig)
    assert domain.name == "软件工程"
    assert domain.description == "software engineering"
    assert domain.out_of_domain_reply == "Out of domain."
    assert set(domain.intents) == {"concept_explain", "faq"}
    assert domain.intent_mapping == {"concept_explain": "teaching", "faq": "direct"}
    assert domain.strategies["teaching"].complexity_gate is True
    assert domain.strategies["direct"].model == "model-direct"
    assert domain.strategies["direct"].complexity_gate is False
    assert domain.default_strategy == "direct"
    assert "teach self-contained" in domain.prompts["teaching"]
    assert "unsupported" in domain.prompts["unsupported_complex"]


def test_load_domain_config_out_of_domain_reply_default(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "软件工程", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    domain = load_domain_config(str(base))
    assert domain.out_of_domain_reply == (
        "This question falls outside my expert domain (软件工程) "
        "and I cannot provide a professional answer."
    )


def test_load_domain_config_missing_domain_json(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(str(tmp_path / "no-such-dir"))


def test_load_domain_config_bad_yaml(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(":: not: [valid", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text("", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_mapping_unknown_intent(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("- id: faq\n  description: q\n", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("bogus_intent: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_missing_prompt(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("- id: faq\n  description: q\n", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_domain_config(str(base))
    assert "unsupported_complex.md" in str(exc_info.value)


def _write_domain_with_default(tmp_path, strategies_yaml):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(json.dumps({
        "name": "x", "description": "d",
    }), encoding="utf-8")
    (base / "intents.yaml").write_text("", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text(strategies_yaml, encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    return str(base)


def test_load_domain_config_resolves_default_strategy(tmp_path):
    domain = load_domain_config(_write_domain_with_default(
        tmp_path, "direct:\n  default: true\n"))
    assert domain.default_strategy == "direct"
    assert domain.strategies["direct"].default is True


def test_load_domain_config_zero_defaults_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(_write_domain_with_default(tmp_path, "direct:\n"))


def test_load_domain_config_multiple_defaults_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(_write_domain_with_default(
            tmp_path, "direct:\n  default: true\nteaching:\n  default: true\n"))
