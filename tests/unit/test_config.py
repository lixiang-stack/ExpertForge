import json

import pytest

from agent.config import (
    AgentConfig,
    ConfigError,
    DomainConfig,
    IntentDef,
    get_api_key,
    get_judge_api_key,
    load_config,
)
from agent.domain_config import load_domain_config

ORCHESTRATION_YAML = (
    "enabled: true\n"
    "min_complexity: complex\n"
    "intents:\n"
    "  - faq\n"
    "max_workers: 4\n"
    "evaluator:\n"
    "  enabled: true\n"
    "  min_dimension_score: 3\n"
    "  max_rounds: 1\n"
)


def _write_config(tmp_path, data):
    data = {
        "provider": "test",
        "provider_capabilities": {},
        **data,
    }
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
        "provider": "test",
        "provider_capabilities": {},
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
        "provider": "test",
        "provider_capabilities": {},
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
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
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
    (base / "prompts" / "teaching.md").write_text(
        "teach self-contained", encoding="utf-8"
    )
    (base / "prompts" / "direct.md").write_text(
        "direct self-contained", encoding="utf-8"
    )
    return str(base)


def test_load_domain_config_basic(tmp_path):
    domain = load_domain_config(_write_domain(tmp_path))
    assert isinstance(domain, DomainConfig)
    assert domain.name == "软件工程"
    assert domain.description == "software engineering"
    assert domain.out_of_domain_reply == "Out of domain."
    assert set(domain.intents) == {"concept_explain", "faq"}
    assert domain.intent_mapping == {"concept_explain": "teaching", "faq": "direct"}
    assert domain.strategies == ["direct", "teaching"]
    assert "teach self-contained" in domain.prompts["teaching"]
    assert domain.orchestration is not None


def test_load_domain_config_out_of_domain_reply_default(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "软件工程", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
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
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


from agent.config import ComplexityPolicy


def test_load_domain_config_complexity_policy(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "complexity.yaml").write_text(
        "- level: simple\n"
        "  description: single concept\n"
        "  dimensions:\n"
        "    - 'Reasoning: single step'\n"
        "    - 'Scope: single concept'\n"
        "  positive_examples:\n"
        "    - 'What is dependency injection?'\n"
        "  negative_examples:\n"
        "    - 'Design a distributed rate limiter'\n"
        "  boundaries:\n"
        "    - 'Prefer medium when multiple concepts'\n"
        "- level: medium\n"
        "  description: multiple concepts\n"
        "- level: complex\n"
        "  description: multiple subsystems\n",
        encoding="utf-8",
    )
    domain = load_domain_config(str(base))
    assert isinstance(domain.complexity, ComplexityPolicy)
    assert [l.level for l in domain.complexity.levels] == ["simple", "medium", "complex"]
    level = domain.complexity.levels[0]
    assert level.level == "simple"
    assert level.description == "single concept"
    assert level.dimensions == ["Reasoning: single step", "Scope: single concept"]
    assert level.positive_examples == ["What is dependency injection?"]
    assert level.negative_examples == ["Design a distributed rate limiter"]
    assert level.boundaries == ["Prefer medium when multiple concepts"]


def test_load_domain_config_complexity_missing_is_none(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    domain = load_domain_config(str(base))
    assert domain.complexity is None


def test_load_domain_config_complexity_invalid_level(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "complexity.yaml").write_text("- level: bogus\n  description: d\n",
                                          encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_complexity_incomplete_levels_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "complexity.yaml").write_text(
        "- level: simple\n  description: d\n"
        "- level: medium\n  description: d\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_complexity_reordered_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "complexity.yaml").write_text(
        "- level: complex\n  description: d\n"
        "- level: medium\n  description: d\n"
        "- level: simple\n  description: d\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_complexity_non_list_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "complexity.yaml").write_text("not_a_list: true\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_complexity_missing_level_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "complexity.yaml").write_text("- description: d\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_expert_policy(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "expert_policy.md").write_text(
        "You are a Senior Software Engineering Expert.", encoding="utf-8"
    )
    domain = load_domain_config(str(base))
    assert domain.expert_policy == "You are a Senior Software Engineering Expert."


def test_load_domain_config_expert_policy_missing_is_empty(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    domain = load_domain_config(str(base))
    assert domain.expert_policy == ""


def test_load_domain_config_expert_policy_empty_is_empty(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "expert_policy.md").write_text("", encoding="utf-8")
    domain = load_domain_config(str(base))
    assert domain.expert_policy == ""


def test_load_domain_config_mapping_unknown_intent(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("- id: faq\n  description: q\n", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("bogus_intent: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_mapping_unknown_strategy_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("- id: faq\n  description: q\n", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "teaching.md").write_text("t", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_domain_config(str(base))
    assert "references unknown strategy" in str(exc_info.value)
    assert "direct" in str(exc_info.value)


def test_load_domain_config_strategies_derived_from_prompts(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "teaching.md").write_text("t", encoding="utf-8")
    domain = load_domain_config(str(base))
    assert domain.strategies == ["direct", "teaching"]


def test_load_domain_config_unmapped_intent_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n"
        "- id: tutorial\n  description: walkthrough\n",
        encoding="utf-8",
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_config_without_observability(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.observability is None


def test_load_config_observability_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "observability": {"enabled": False, "data_dir": "obs/"},
    })
    cfg = load_config(path)
    assert cfg.observability is not None
    assert cfg.observability.enabled is False
    assert cfg.observability.data_dir == "obs/"


def test_load_config_observability_enabled_with_phase_map(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "observability": {
            "enabled": True,
            "phase_map": {"Orchestrator._worker": "work"},
        },
    })
    cfg = load_config(path)
    assert cfg.observability.enabled is True
    assert cfg.observability.data_dir == ".observability"          # default
    assert cfg.observability.phase_map == {"Orchestrator._worker": "work"}


def test_load_config_observability_ignores_non_dict(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "observability": "nope",
    })
    cfg = load_config(path)
    assert cfg.observability is None


def test_load_config_evaluation_judge_block_and_results_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "evaluation": {
            "results_dir": "eval/results",
            "judge": {
                "base_url": "https://judge.example.com/v1",
                "model": "judge-a",
                "provider": "judge-provider",
                "provider_capabilities": {"supports_tool_call": True},
                "timeout": 30,
            },
        },
    })
    cfg = load_config(path)
    assert cfg.evaluation is not None
    assert cfg.evaluation.results_dir == "eval/results"
    judge = cfg.evaluation.judge
    assert judge is not None
    assert judge.base_url == "https://judge.example.com/v1"
    assert judge.model == "judge-a"
    assert judge.provider == "judge-provider"
    assert judge.provider_capabilities is not None
    assert judge.provider_capabilities.supports_tool_call is True
    assert judge.timeout == 30


def test_get_judge_api_key(monkeypatch):
    monkeypatch.setenv("AGENT_JUDGE_API_KEY", "judge-secret")
    assert get_judge_api_key() == "judge-secret"


def test_get_judge_api_key_missing(monkeypatch):
    monkeypatch.delenv("AGENT_JUDGE_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="AGENT_JUDGE_API_KEY"):
        get_judge_api_key()


def test_load_config_evaluation_judge_falls_back_to_top_level(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "evaluation": {
            "results_dir": "eval/results",
            "judge": {"timeout": 45},
        },
    })
    cfg = load_config(path)
    judge = cfg.evaluation.judge
    assert judge is not None
    assert judge.base_url == "https://api.example.com/v1"
    assert judge.model == "model-a"
    assert judge.provider == "test"
    assert judge.provider_capabilities is None
    assert judge.timeout == 45


def test_load_config_evaluation_judge_default_none(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "evaluation": {"results_dir": "eval/results"},
    })
    cfg = load_config(path)
    assert cfg.evaluation is not None
    assert cfg.evaluation.results_dir == "eval/results"
    assert cfg.evaluation.judge is None


def test_load_config_evaluation_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.evaluation is not None
    assert cfg.evaluation.results_dir == "evaluation/results"
    assert cfg.evaluation.judge is None


def test_load_config_evaluation_ignores_non_dict(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "evaluation": "nope",
    })
    cfg = load_config(path)
    assert cfg.evaluation is not None
    assert cfg.evaluation.results_dir == "evaluation/results"
    assert cfg.evaluation.judge is None  # non-dict evaluation treated as an empty block


def test_load_config_evaluation_results_dir_non_string_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "evaluation": {"results_dir": 123},
    })
    cfg = load_config(path)
    assert cfg.evaluation is not None
    assert cfg.evaluation.results_dir == "evaluation/results"


def test_load_config_judge_unknown_capability_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "evaluation": {"judge": {"provider_capabilities": {"supports_magic": True}}},
    })
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_judge_non_boolean_capability_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "evaluation": {"judge": {"provider_capabilities": {"supports_json_schema": "yes"}}},
    })
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_judge_invalid_timeout_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "evaluation": {"judge": {"timeout": "soon"}},
    })
    cfg = load_config(path)
    assert cfg.evaluation.judge is not None
    assert cfg.evaluation.judge.timeout == 60


def test_load_config_logging_block_parsed(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "logging": {"enabled": True, "level": "DEBUG", "file": "logs/out.jsonl"},
    })
    cfg = load_config(path)
    assert cfg.logging is not None
    assert cfg.logging.enabled is True
    assert cfg.logging.level == "DEBUG"
    assert cfg.logging.file == "logs/out.jsonl"


def test_load_config_logging_unknown_level_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "logging": {"enabled": True, "level": "VERBOSE"},
    })
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_logging_non_dict_disabled_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "logging": "nope",
    })
    cfg = load_config(path)
    assert cfg.logging is not None
    assert cfg.logging.enabled is False
    assert cfg.logging.level == "INFO"
    assert cfg.logging.file == "logs/agent.jsonl"


def test_load_config_logging_string_enabled_coerced_false(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "logging": {"enabled": "false"},
    })
    cfg = load_config(path)
    assert cfg.logging is not None
    assert cfg.logging.enabled is False


def test_load_domain_config_intent_definition_fields(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: concept_explain\n"
        "  description: explain a concept\n"
        "  positive_examples:\n"
        "    - Why does DI reduce coupling?\n"
        "  negative_examples:\n"
        "    - My app crashes.\n"
        "  boundaries:\n"
        "    - Prefer concept_explain over faq when the user wants understanding.\n"
        "- id: faq\n"
        "  description: quick question\n",
        encoding="utf-8",
    )
    (base / "intent_mapping.yaml").write_text(
        "concept_explain: direct\nfaq: direct\n", encoding="utf-8"
    )
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    domain = load_domain_config(str(base))
    intent = domain.intents["concept_explain"]
    assert intent.positive_examples == ["Why does DI reduce coupling?"]
    assert intent.negative_examples == ["My app crashes."]
    assert intent.boundaries == [
        "Prefer concept_explain over faq when the user wants understanding."
    ]


def test_load_domain_config_intent_fields_default_empty(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(ORCHESTRATION_YAML, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    domain = load_domain_config(str(base))
    intent = domain.intents["faq"]
    assert intent.positive_examples == []
    assert intent.negative_examples == []
    assert intent.boundaries == []


def test_load_config_timeout_default_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.timeout is None  # unspecified -> SDK default


def test_load_config_timeout_parsed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "timeout": 240,
    })
    cfg = load_config(path)
    assert cfg.timeout == 240.0


def test_load_config_timeout_invalid_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "timeout": "soon",
    })
    cfg = load_config(path)
    assert cfg.timeout is None


def test_load_config_provider_and_capabilities_parsed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider": "gemini",
        "provider_capabilities": {"supports_json_schema": True},
    })
    cfg = load_config(path)
    assert cfg.provider == "gemini"
    assert cfg.provider_capabilities == {
        "supports_json_schema": True,
    }


def test_load_config_missing_provider_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider_capabilities": {},
    }), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_load_config_missing_capabilities_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider": "gemini",
    }), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_load_config_unknown_capability_key_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider": "gemini",
        "provider_capabilities": {"supports_magic": True},
    }), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_load_config_non_boolean_capability_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "provider": "gemini",
        "provider_capabilities": {"supports_json_schema": "yes"},
    }), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_load_domain_config_orchestration_policy(tmp_path):
    domain = load_domain_config(_write_domain(tmp_path))
    oc = domain.orchestration
    assert oc is not None
    assert oc.enabled is True
    assert oc.min_complexity == "complex"
    assert oc.intents == ["faq"]
    assert oc.max_workers == 4
    assert oc.evaluator.enabled is True
    assert oc.evaluator.min_dimension_score == 3
    assert oc.evaluator.max_rounds == 1


def test_load_domain_config_orchestration_missing_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_orchestration_empty_intents_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: complex\nintents: []\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_orchestration_unknown_intent_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: complex\nintents:\n  - bogus\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_orchestration_bad_min_complexity_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: impossible\nintents:\n  - faq\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_orchestration_bad_evaluator_raises(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: faq\n  description: quick question\n", encoding="utf-8"
    )
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "orchestration.yaml").write_text(
        "enabled: true\nmin_complexity: complex\nintents:\n  - faq\n"
        "evaluator:\n  enabled: true\n  min_dimension_score: 9\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))


def test_load_domain_config_topology_defaults_to_map_reduce(tmp_path):
    domain = load_domain_config(_write_domain(tmp_path))
    assert domain.orchestration.topology == "map_reduce"


def test_load_domain_config_topology_critique(tmp_path):
    base = tmp_path / "domain"
    _write_domain(tmp_path)
    (base / "orchestration.yaml").write_text(
        ORCHESTRATION_YAML + "topology: critique\n", encoding="utf-8"
    )
    domain = load_domain_config(str(base))
    assert domain.orchestration.topology == "critique"


def test_load_domain_config_topology_invalid_raises(tmp_path):
    base = tmp_path / "domain"
    _write_domain(tmp_path)
    (base / "orchestration.yaml").write_text(
        ORCHESTRATION_YAML + "topology: bogus\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))
