from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


DEFAULT_CONFIG_PATH = "config.json"
DEFAULT_EXAMPLE_CONFIG_PATH = "config.example.json"


@dataclass
class ObservabilityConfig:
    enabled: bool = False
    data_dir: str = ".observability"
    phase_map: dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    judge_model: str | None = None
    results_dir: str = "evaluation/results"


@dataclass
class AgentConfig:
    base_url: str
    model: str
    classifier_model: str
    domain_dir: str
    model_low: str | None = None
    model_high: str | None = None
    observability: ObservabilityConfig | None = None
    evaluation: EvaluationConfig | None = None


def _read_json_file(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}.")
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid config JSON: {e}")
    if not isinstance(raw, dict):
        raise ConfigError("Config top-level must be a JSON object.")
    return raw


def _load_config_dict(path: str | None, default_path: str) -> dict:
    if path:
        return _read_json_file(path)
    if os.path.isfile(default_path):
        return _read_json_file(default_path)
    if os.path.isfile(DEFAULT_EXAMPLE_CONFIG_PATH):
        return _read_json_file(DEFAULT_EXAMPLE_CONFIG_PATH)
    raise ConfigError(
        f"Config file not found: {default_path}. "
        "Create one by copying config.example.json."
    )


def load_config(path: str | None = None) -> AgentConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    raw = _load_config_dict(path, config_path)

    base_url = os.environ.get("AGENT_BASE_URL") or raw.get("base_url")
    if not base_url:
        raise ConfigError("Missing 'base_url' in config or AGENT_BASE_URL env var.")

    model = raw.get("model")
    if not model:
        raise ConfigError("Missing 'model' in config.")

    model_low = raw.get("model_low")
    model_high = raw.get("model_high")
    model_low = model_low if isinstance(model_low, str) and model_low else None
    model_high = model_high if isinstance(model_high, str) and model_high else None

    classifier_model = model_low or model

    domain_dir = raw.get("domain_dir")
    if not isinstance(domain_dir, str) or not domain_dir:
        raise ConfigError("Missing 'domain_dir' in config.")

    raw_obs = raw.get("observability")
    observability = None
    if isinstance(raw_obs, dict):
        data_dir = raw_obs.get("data_dir") or ".observability"
        phase_map = raw_obs.get("phase_map")
        observability = ObservabilityConfig(
            enabled=bool(raw_obs.get("enabled")),
            data_dir=data_dir if isinstance(data_dir, str) else ".observability",
            phase_map=phase_map if isinstance(phase_map, dict) else {},
        )

    raw_eval = raw.get("evaluation")
    evaluation = None
    if isinstance(raw_eval, dict):
        judge_model = raw_eval.get("judge_model")
        judge_model = judge_model if isinstance(judge_model, str) and judge_model else None
        results_dir = raw_eval.get("results_dir") or "evaluation/results"
        evaluation = EvaluationConfig(
            judge_model=judge_model,
            results_dir=results_dir if isinstance(results_dir, str) else "evaluation/results",
        )

    return AgentConfig(
        base_url=base_url,
        model=model,
        classifier_model=classifier_model,
        domain_dir=domain_dir,
        model_low=model_low,
        model_high=model_high,
        observability=observability,
        evaluation=evaluation,
    )


def get_api_key() -> str:
    api_key = os.environ.get("AGENT_API_KEY")
    if not api_key:
        raise ConfigError(
            "AGENT_API_KEY environment variable is not set. "
            "Set it with: export AGENT_API_KEY=your_key"
        )
    return api_key


@dataclass
class IntentDef:
    id: str
    description: str


@dataclass
class StrategyDef:
    id: str
    model: str | None = None
    complexity_gate: bool = False
    default: bool = False


@dataclass
class DomainConfig:
    name: str
    description: str
    out_of_domain_reply: str
    intents: dict[str, IntentDef]
    intent_mapping: dict[str, str]
    strategies: dict[str, StrategyDef]
    default_strategy: str
    prompts: dict[str, str]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Domain config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid domain config JSON: {path}: {e}")
    if not isinstance(data, dict):
        raise ConfigError(f"Domain config must be a JSON object: {path}")
    return data


def _read_yaml(path: Path) -> object:
    if not path.is_file():
        raise ConfigError(f"Domain config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid domain config YAML: {path}: {e}")


def _read_prompt(path: Path) -> str:
    if not path.is_file():
        raise ConfigError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_domain_config(domain_dir: str) -> DomainConfig:
    base = Path(domain_dir)
    meta = _read_json(base / "domain.json")
    name = meta.get("name") or ""
    description = meta.get("description")
    if not description:
        raise ConfigError(f"Missing 'description' in {base / 'domain.json'}")
    out_of_domain_reply = meta.get("out_of_domain_reply") or (
        f"This question falls outside my expert domain ({name}) "
        "and I cannot provide a professional answer."
    )

    intents: dict[str, IntentDef] = {}
    intents_data = _read_yaml(base / "intents.yaml")
    if intents_data is None:
        intents_data = []
    if not isinstance(intents_data, list):
        raise ConfigError(f"intents.yaml must contain a list: {base / 'intents.yaml'}")
    for item in intents_data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ConfigError(f"Invalid intent entry in {base / 'intents.yaml'}: {item}")
        iid = item["id"]
        intents[iid] = IntentDef(
            id=iid,
            description=item.get("description") or "",
        )

    mapping_data = _read_yaml(base / "intent_mapping.yaml")
    if mapping_data is None:
        mapping_data = {}
    if not isinstance(mapping_data, dict):
        raise ConfigError(
            f"intent_mapping.yaml must contain a mapping: {base / 'intent_mapping.yaml'}"
        )
    intent_mapping: dict[str, str] = {}
    for intent_id, strategy_id in mapping_data.items():
        if not isinstance(strategy_id, str):
            raise ConfigError(f"Invalid mapping for intent '{intent_id}'")
        if intent_id not in intents:
            raise ConfigError(
                f"Mapping references unknown intent '{intent_id}' in {base / 'intent_mapping.yaml'}"
            )
        intent_mapping[intent_id] = strategy_id

    strategies_data = _read_yaml(base / "strategies.yaml")
    if strategies_data is None:
        strategies_data = {}
    if not isinstance(strategies_data, dict):
        raise ConfigError(f"strategies.yaml must contain a mapping: {base / 'strategies.yaml'}")
    strategies: dict[str, StrategyDef] = {}
    configured_default = None
    for sid, item in strategies_data.items():
        if isinstance(item, dict):
            model = item.get("model")
            strategies[sid] = StrategyDef(
                id=sid,
                model=model if isinstance(model, str) and model else None,
                complexity_gate=bool(item.get("complexity_gate", False)),
                default=bool(item.get("default", False)),
            )
            if item.get("default"):
                configured_default = sid
        else:
            strategies[sid] = StrategyDef(id=sid)

    for intent_id, strategy_id in intent_mapping.items():
        if strategy_id not in strategies:
            raise ConfigError(
                f"Mapping for intent '{intent_id}' references unknown strategy "
                f"'{strategy_id}' in {base / 'intent_mapping.yaml'}"
            )

    if configured_default is None:
        raise ConfigError(
            f"Exactly one strategy in {base / 'strategies.yaml'} must have default: true"
        )
    if sum(1 for sdef in strategies.values() if sdef.default) != 1:
        raise ConfigError(
            f"Only one strategy in {base / 'strategies.yaml'} may have default: true"
        )

    prompts: dict[str, str] = {}
    prompt_dir = base / "prompts"
    for sid in strategies:
        prompts[sid] = _read_prompt(prompt_dir / f"{sid}.md")
    prompts["unsupported_complex"] = _read_prompt(prompt_dir / "unsupported_complex.md")

    return DomainConfig(
        name=name,
        description=description,
        out_of_domain_reply=out_of_domain_reply,
        intents=intents,
        intent_mapping=intent_mapping,
        strategies=strategies,
        default_strategy=configured_default,
        prompts=prompts,
    )
