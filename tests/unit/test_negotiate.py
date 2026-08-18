from agent.capabilities import ProviderCapabilities
from agent.negotiate import negotiate_structured_output

deepseek = ProviderCapabilities(provider="deepseek")
gemini = ProviderCapabilities(provider="gemini", supports_json_schema=True)


def test_schema_supported_uses_schema():
    assert negotiate_structured_output(gemini, json_mode=True, json_schema={"type": "object"}) == "json_schema"


def test_schema_unsupported_degrades_to_json_object():
    assert negotiate_structured_output(deepseek, json_mode=True, json_schema={"type": "object"}) == "json_object"


def test_json_mode_uses_json_object():
    assert negotiate_structured_output(deepseek, json_mode=True, json_schema=None) == "json_object"


def test_no_request_returns_none():
    assert negotiate_structured_output(gemini, json_mode=False, json_schema=None) is None