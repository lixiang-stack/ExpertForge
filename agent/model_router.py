from __future__ import annotations

from .config import AgentConfig, DomainConfig
from .router import RouteResult


def resolve_model(
    config: AgentConfig,
    domain: DomainConfig,
    route: RouteResult,
    default: str,
) -> str:
    if route.complexity == "simple":
        return config.model_low or default
    return config.model_high or default
