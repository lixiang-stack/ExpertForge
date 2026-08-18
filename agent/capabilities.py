from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str
    supports_json_schema: bool = False
    supports_thinking_toggle: bool = False
    supports_tool_call: bool = False


KNOWN_CAPABILITY_KEYS = (
    "supports_json_schema",
    "supports_thinking_toggle",
    "supports_tool_call",
)