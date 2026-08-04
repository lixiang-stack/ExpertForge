from __future__ import annotations

import json
import os
from dataclasses import dataclass


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


DEFAULT_CONFIG_PATH = "config.json"


@dataclass
class AgentConfig:
    base_url: str
    model: str
    classifier_model: str
    domain_name: str
    domain_description: str
    out_of_domain_reply: str


def load_config(path: str | None = None) -> AgentConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ConfigError(
            f"Config file not found: {config_path}. "
            "Create one by copying config.example.json."
        )
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid config JSON: {e}")

    if not isinstance(raw, dict):
        raise ConfigError("Config top-level must be a JSON object.")

    base_url = os.environ.get("AGENT_BASE_URL") or raw.get("base_url")
    if not base_url:
        raise ConfigError("Missing 'base_url' in config or AGENT_BASE_URL env var.")

    model = raw.get("model")
    if not model:
        raise ConfigError("Missing 'model' in config.")

    classifier_model = raw.get("classifier_model") or model

    domain = raw.get("domain")
    if not isinstance(domain, dict):
        raise ConfigError("Missing 'domain' in config.")

    domain_name = domain.get("name") or ""
    domain_description = domain.get("description")
    if not domain_description:
        raise ConfigError("Missing 'domain.description' in config.")

    out_of_domain_reply = domain.get("out_of_domain_reply") or (
        f"This question falls outside my expert domain ({domain_name}) "
        "and I cannot provide a professional answer."
    )

    return AgentConfig(
        base_url=base_url,
        model=model,
        classifier_model=classifier_model,
        domain_name=domain_name,
        domain_description=domain_description,
        out_of_domain_reply=out_of_domain_reply,
    )


def get_api_key() -> str:
    api_key = os.environ.get("AGENT_API_KEY")
    if not api_key:
        raise ConfigError(
            "AGENT_API_KEY environment variable is not set. "
            "Set it with: export AGENT_API_KEY=your_key"
        )
    return api_key
