import threading
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
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    text = client.chat_completion([{"role": "user", "content": "hi"}])

    assert text == "你好"
    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "model-a"
    assert kwargs["stream"] is False
    assert "extra_body" not in kwargs


@patch("agent.llm.OpenAI")
def test_chat_completion_disable_thinking_passes_extra_body(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
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
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    text = client.chat_completion([{"role": "user", "content": "hi"}])

    assert text == ""


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
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}], json_mode=True)

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@patch("agent.llm.OpenAI")
def test_chat_completion_json_mode_with_disable_thinking(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "{}"
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
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}])

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs


@patch("agent.llm.OpenAI")
def test_chat_completion_json_schema_passes_response_format(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "{}"
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
def test_chat_completion_records_thread_local_usage(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.usage = _usage(10, 5)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}])

    assert client._usage_local.usage.prompt_tokens == 10
    assert client._usage_local.usage.completion_tokens == 5


@patch("agent.llm.OpenAI")
def test_usage_isolated_across_threads(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.usage = _usage(10, 5)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}])
    assert client._usage_local.usage.prompt_tokens == 10

    seen = {}

    def read_in_thread():
        seen["fresh_has_usage"] = hasattr(client._usage_local, "usage")
        client._usage_local.usage = _usage(99, 1)

    t = threading.Thread(target=read_in_thread)
    t.start()
    t.join()

    # A fresh thread has its own thread-local slot: it sees no usage set by the
    # main thread, and writes in the worker thread never leak back.
    assert seen["fresh_has_usage"] is False
    assert client._usage_local.usage.prompt_tokens == 10


@patch("agent.llm.OpenAI")
def test_chat_completion_returns_text_unaffected(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "你好"
    resp.usage = _usage(3, 4)
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    text = client.chat_completion([{"role": "user", "content": "hi"}])

    assert text == "你好"
