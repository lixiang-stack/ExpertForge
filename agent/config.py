from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .capabilities import KNOWN_CAPABILITY_KEYS, ProviderCapabilities


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


DEFAULT_CONFIG_PATH = "config.json"
DEFAULT_EXAMPLE_CONFIG_PATH = "config.example.json"

COMPLEXITY_LEVELS = ("simple", "medium", "complex")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass
class ObservabilityConfig:
    enabled: bool = False
    data_dir: str = ".observability"
    phase_map: dict[str, str] = field(default_factory=dict)


@dataclass
class JudgeConfig:
    base_url: str
    model: str
    provider: str
    provider_capabilities: ProviderCapabilities | None = None
    timeout: int = 60


@dataclass
class EvaluationConfig:
    results_dir: str = "evaluation/results"
    judge: JudgeConfig | None = None


@dataclass
class LoggingConfig:
    enabled: bool = False
    level: str = "INFO"
    file: str = "logs/agent.jsonl"


@dataclass
class AgentConfig:
    base_url: str
    model: str
    classifier_model: str
    domain_dir: str
    model_low: str | None = None
    model_high: str | None = None
    timeout: float | None = None
    provider: str = ""
    provider_capabilities: dict[str, bool] = field(default_factory=dict)
    observability: ObservabilityConfig | None = None
    evaluation: EvaluationConfig | None = None
    logging: LoggingConfig | None = None


def resolve_judge_model(config: AgentConfig) -> str:
    """Return the judge model, falling back to the main model when no judge is configured."""
    judge = config.evaluation.judge if config.evaluation is not None else None
    return judge.model if judge is not None else config.model


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


def _validate_capabilities(caps: dict, section: str) -> None:
    """Validate capability keys and value types for a config section."""
    for key, value in caps.items():
        if key not in KNOWN_CAPABILITY_KEYS:
            raise ConfigError(
                f"Unknown capability '{key}' in {section}. "
                f"Known capabilities: {', '.join(KNOWN_CAPABILITY_KEYS)}."
            )
        if not isinstance(value, bool):
            raise ConfigError(f"Capability '{key}' must be a boolean in {section}.")


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

    timeout = raw.get("timeout")
    timeout = timeout if isinstance(timeout, (int, float)) and timeout > 0 else None

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

    provider = raw.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ConfigError("Missing 'provider' in config (e.g. 'deepseek' or 'gemini').")

    raw_caps = raw.get("provider_capabilities")
    if not isinstance(raw_caps, dict):
        raise ConfigError(
            "Missing 'provider_capabilities' in config; declare the provider's capabilities."
        )
    _validate_capabilities(raw_caps, "provider_capabilities")
    provider_capabilities = dict(raw_caps)

    eval_block = raw.get("evaluation")
    if not isinstance(eval_block, dict):
        eval_block = {}
    eval_results_dir = eval_block.get("results_dir")
    if not isinstance(eval_results_dir, str):
        eval_results_dir = "evaluation/results"

    judge_block = eval_block.get("judge")
    if not isinstance(judge_block, dict):
        judge_block = {}
    judge_base_url = judge_block.get("base_url") or base_url
    judge_name = judge_block.get("model") or model
    judge_provider = judge_block.get("provider") or provider
    judge_caps = judge_block.get("provider_capabilities")
    if isinstance(judge_caps, dict):
        _validate_capabilities(judge_caps, "evaluation.judge.provider_capabilities")
    judge_caps_obj = (
        ProviderCapabilities(provider=judge_provider, **judge_caps)
        if isinstance(judge_caps, dict)
        else None
    )
    judge_timeout = judge_block.get("timeout", 60)
    judge_timeout = (
        judge_timeout if isinstance(judge_timeout, (int, float)) and judge_timeout > 0 else 60
    )
    judge_config = (
        JudgeConfig(
            base_url=judge_base_url,
            model=judge_name,
            provider=judge_provider,
            provider_capabilities=judge_caps_obj,
            timeout=judge_timeout,
        )
        if judge_block
        else None
    )

    evaluation = EvaluationConfig(results_dir=eval_results_dir, judge=judge_config)

    logging_block = raw.get("logging")
    if not isinstance(logging_block, dict):
        logging_block = {}
    logging_level = logging_block.get("level", "INFO")
    if logging_level not in LOG_LEVELS:
        raise ConfigError(f"Unknown log level '{logging_level}'. Valid: {', '.join(LOG_LEVELS)}")
    logging_enabled = logging_block.get("enabled", False)
    logging_config = LoggingConfig(
        enabled=logging_enabled if isinstance(logging_enabled, bool) else False,
        level=logging_level,
        file=logging_block.get("file", "logs/agent.jsonl"),
    )

    return AgentConfig(
        base_url=base_url,
        model=model,
        classifier_model=classifier_model,
        domain_dir=domain_dir,
        model_low=model_low,
        model_high=model_high,
        timeout=timeout,
        provider=provider,
        provider_capabilities=provider_capabilities,
        observability=observability,
        evaluation=evaluation,
        logging=logging_config,
    )


def get_api_key() -> str:
    api_key = os.environ.get("AGENT_API_KEY")
    if not api_key:
        raise ConfigError(
            "AGENT_API_KEY environment variable is not set. "
            "Set it with: export AGENT_API_KEY=your_key"
        )
    return api_key


def get_judge_api_key() -> str:
    api_key = os.environ.get("AGENT_JUDGE_API_KEY")
    if not api_key:
        raise ConfigError(
            "AGENT_JUDGE_API_KEY environment variable is not set. "
            "Set it with: export AGENT_JUDGE_API_KEY=your_key"
        )
    return api_key


@dataclass
class IntentDef:
    id: str
    description: str
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)


@dataclass
class ComplexityLevelDef:
    level: str
    description: str
    dimensions: list[str] = field(default_factory=list)
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)


@dataclass
class ComplexityPolicy:
    levels: list[ComplexityLevelDef]


@dataclass
class EvaluatorPolicy:
    enabled: bool = True
    min_dimension_score: int = 3
    max_rounds: int = 1


@dataclass
class OrchestrationPolicy:
    enabled: bool = True
    min_complexity: str = "complex"
    intents: list[str] = field(default_factory=list)
    max_workers: int = 4
    evaluator: EvaluatorPolicy = field(default_factory=EvaluatorPolicy)


@dataclass
class DomainConfig:
    name: str
    description: str
    out_of_domain_reply: str
    intents: dict[str, IntentDef]
    intent_mapping: dict[str, str]
    strategies: list[str]
    prompts: dict[str, str]
    complexity: ComplexityPolicy | None = None
    expert_policy: str = ""
    orchestration: OrchestrationPolicy | None = None
