from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from agent.llm import LLMClient, LLMError


@patch("agent.llm.OpenAI")
def test_constructor_configures_openai(mock_openai):
    LLMClient("https://api.example.com/v1", "key", "model-a")
    mock_openai.assert_called_once_with(
        api_key="key", base_url="https://api.example.com/v1", timeout=60
    )


@patch("agent.llm.OpenAI")
def test_chat_completion_returns_content(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "你好"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])

    assert result.text == "你好"
    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "model-a"
    assert kwargs["stream"] is False
    assert "extra_body" not in kwargs


@patch("agent.llm.OpenAI")
def test_chat_completion_disable_thinking_passes_extra_body(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}], disable_thinking=True)

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@patch("agent.llm.OpenAI")
def test_chat_completion_stream_yields_content(mock_openai):
    chunk1 = MagicMock()
    chunk1.choices[0].delta.content = "世"
    chunk2 = MagicMock()
    chunk2.choices[0].delta.content = "界"
    empty = MagicMock()
    empty.choices[0].delta.content = None
    mock_openai.return_value.chat.completions.create.return_value = iter(
        [chunk1, empty, chunk2]
    )

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    out = list(client.chat_completion_stream([{"role": "user", "content": "hi"}]))

    assert out == ["世", "界"]
    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True


@patch("agent.llm.OpenAI")
def test_chat_completion_none_content_returns_empty_string(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = None
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.text == ""


@patch("agent.llm.OpenAI")
def test_sdk_error_wrapped_in_llm_error(mock_openai):
    mock_openai.return_value.chat.completions.create.side_effect = OpenAIError("boom")
    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    with pytest.raises(LLMError):
        client.chat_completion([{"role": "user", "content": "hi"}])


@patch("agent.llm.OpenAI")
def test_chat_completion_json_mode_passes_response_format(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "{}"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}], json_mode=True)

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@patch("agent.llm.OpenAI")
def test_chat_completion_json_mode_with_disable_thinking(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "{}"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion(
        [{"role": "user", "content": "hi"}], json_mode=True, disable_thinking=True
    )

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@patch("agent.llm.OpenAI")
def test_chat_completion_json_mode_off_by_default(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}])

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs


@patch("agent.llm.OpenAI")
def test_chat_completion_json_schema_passes_response_format(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "{}"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}], json_schema=schema)

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "classification_result", "schema": schema, "strict": False},
    }


@patch("agent.llm.OpenAI")
def test_chat_completion_json_schema_wins_over_json_mode(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "{}"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion(
        [{"role": "user", "content": "hi"}], json_mode=True, json_schema={"type": "object"}
    )

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"


def _usage(prompt, completion):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = prompt + completion
    return u


@patch("agent.llm.OpenAI")
def test_chat_completion_model_falls_back_to_requested(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = None
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}], model="low-a")
    assert result.model == "low-a"


@patch("agent.llm.OpenAI")
def test_chat_completion_returns_text_unaffected(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "你好"
    resp.model = "model-a"
    resp.usage = _usage(3, 4)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.text == "你好"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 4


def _usage_with_cache(prompt, completion, cached=0):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = prompt + completion
    details = MagicMock()
    details.cached_tokens = cached
    u.prompt_tokens_details = details
    return u


@patch("agent.llm.OpenAI")
def test_chat_completion_records_cache_tokens(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = _usage_with_cache(10, 5, cached=7)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.cache_tokens == 7


@patch("agent.llm.OpenAI")
def test_chat_completion_cache_tokens_zero_when_absent(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    u = _usage(10, 5)
    u.prompt_tokens_details = None  # real "details absent" path
    resp.usage = u
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.cache_tokens == 0


@patch("agent.llm.OpenAI")
def test_chat_completion_returns_chat_result(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "你好"
    resp.model = "model-a"
    resp.usage = _usage(10, 5)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])

    assert result.text == "你好"
    assert result.model == "model-a"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15
    assert result.cache_tokens == 0


@patch("agent.llm.OpenAI")
def test_chat_completion_model_falls_back_to_client_default(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = None
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.model == "model-a"


@patch("agent.llm.OpenAI")
def test_chat_completion_zero_tokens_when_usage_absent(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0
