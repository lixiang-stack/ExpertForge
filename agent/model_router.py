from __future__ import annotations

from .config import AgentConfig, DomainConfig
from .router import RouteResult


def resolve_model(
    config: AgentConfig,
    domain: DomainConfig,
    route: RouteResult,
    default: str,
) -> str:
    strategy_def = domain.strategies.get(route.strategy)
    if strategy_def and strategy_def.model:
        return strategy_def.model
    if route.complexity == "simple":
        return config.model_low or default
    return config.model_high or default