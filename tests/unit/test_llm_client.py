from unittest.mock import Mock

import httpx2
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import SecretStr

from cvhub_agent.config.settings import LLMConfig
from cvhub_agent.llm.client import LLMClient


def make_config(**overrides) -> LLMConfig:
    values = {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "api_key": "unit-test-key",
        "temperature": 0.1,
        "max_output_tokens": 4096,
        "timeout_seconds": 120,
        "max_retries": 1,
        "enable_thinking": False,
    }
    values.update(overrides)
    return LLMConfig.model_validate(values)


@pytest.mark.parametrize("provider", ["deepseek", "local"])
def test_model_initialization(monkeypatch, provider):
    factory = Mock()
    monkeypatch.setattr("cvhub_agent.llm.client.ChatOpenAI", factory)

    config = make_config(provider=provider)
    client = LLMClient(config)

    factory.assert_called_once()
    kwargs = factory.call_args.kwargs

    assert client.model is factory.return_value
    assert kwargs["model"] == config.model
    assert kwargs["base_url"] == config.base_url
    assert isinstance(kwargs["api_key"], SecretStr)
    assert kwargs["max_retries"] == 1
    assert kwargs["timeout"] == 120
    assert kwargs["extra_body"]["max_tokens"] == 4096


def test_deepseek_parameters(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("cvhub_agent.llm.client.ChatOpenAI", factory)

    LLMClient(make_config())

    kwargs = factory.call_args.kwargs

    assert kwargs["temperature"] == 0.1
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert "chat_template_kwargs" not in kwargs["extra_body"]


def test_deepseek_thinking_mode(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("cvhub_agent.llm.client.ChatOpenAI", factory)

    LLMClient(make_config(enable_thinking=True))

    kwargs = factory.call_args.kwargs

    assert kwargs["temperature"] is None
    assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}


def test_local_thinking_parameters(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("cvhub_agent.llm.client.ChatOpenAI", factory)

    LLMClient(make_config(provider="local"))

    kwargs = factory.call_args.kwargs

    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking" not in kwargs["extra_body"]


def test_chat_success(monkeypatch):
    model = Mock()
    model.invoke.return_value = AIMessage(content="Hello!")

    monkeypatch.setattr(
        "cvhub_agent.llm.client.ChatOpenAI",
        lambda **kwargs: model,
    )

    client = LLMClient(make_config())
    messages: list[BaseMessage] = [HumanMessage(content="Hi")]

    response = client.chat(messages)

    model.invoke.assert_called_once_with(messages)
    assert isinstance(response, AIMessage)
    assert response.content == "Hello!"


def test_chat_invalid_response(monkeypatch):
    model = Mock()
    model.invoke.return_value = HumanMessage(content="Invalid")

    monkeypatch.setattr(
        "cvhub_agent.llm.client.ChatOpenAI",
        lambda **kwargs: model,
    )

    client = LLMClient(make_config())

    with pytest.raises(TypeError, match="Expected AIMessage"):
        client.chat([HumanMessage(content="Hi")])


@pytest.mark.parametrize(
    "error_type",
    ["timeout", "connection", "rate_limit", "authentication"],
)
def test_api_errors_are_propagated(monkeypatch, error_type):
    request = httpx2.Request(
        "POST",
        "https://api.deepseek.com/chat/completions",
    )

    if error_type == "timeout":
        error = APITimeoutError(request=request)
    elif error_type == "connection":
        error = APIConnectionError(request=request)
    else:
        status = 429 if error_type == "rate_limit" else 401
        response = httpx2.Response(status, request=request)

        error_class = (
            RateLimitError if error_type == "rate_limit" else AuthenticationError
        )
        error = error_class(
            "Simulated API error",
            response=response,
            body=None,
        )

    model = Mock()
    model.invoke.side_effect = error

    monkeypatch.setattr(
        "cvhub_agent.llm.client.ChatOpenAI",
        lambda **kwargs: model,
    )

    client = LLMClient(make_config())

    with pytest.raises(type(error)):
        client.chat([HumanMessage(content="Hi")])
