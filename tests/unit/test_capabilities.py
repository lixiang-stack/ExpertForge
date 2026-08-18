from agent.capabilities import KNOWN_CAPABILITY_KEYS, ProviderCapabilities


def test_defaults():
    caps = ProviderCapabilities(provider="deepseek")
    assert caps.provider == "deepseek"
    assert caps.supports_json_schema is False
    assert caps.supports_thinking_toggle is False
    assert caps.supports_tool_call is False


def test_explicit_flags():
    caps = ProviderCapabilities(
        provider="gemini",
        supports_json_schema=True,
        supports_thinking_toggle=True,
    )
    assert caps.supports_json_schema is True
    assert caps.supports_thinking_toggle is True
    assert caps.supports_tool_call is False  # untouched default


def test_known_capability_keys():
    assert set(KNOWN_CAPABILITY_KEYS) == {
        "supports_json_schema",
        "supports_thinking_toggle",
        "supports_tool_call",
    }