from collections.abc import Sequence
from typing import Any, Self

import pytest
from langchain.agents.middleware.tool_call_limit import (
    ToolCallLimitExceededError,
)
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool, tool

from cvhub_agent.agent.factory import AgentFactory
from cvhub_agent.agent.middleware import build_agent_middleware
from cvhub_agent.agent.prompts import DEFAULT_SYSTEM_PROMPT


class ScriptedChatModel(FakeMessagesListChatModel):
    """Fake chat model with an explicit invocation counter."""

    call_count: int = 0

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> Self:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1

        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


def make_tool_call(
    name: str,
    call_id: str,
    args: dict[str, Any] | None = None,
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args or {},
                "id": call_id,
            }
        ],
    )


def build_test_agent(
    responses: list[AIMessage],
    tools: Sequence[BaseTool],
    **middleware_kwargs: Any,
):
    model = ScriptedChatModel(responses=responses)  # type: ignore

    agent = AgentFactory(model).build(
        tools=tools,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        middleware=build_agent_middleware(**middleware_kwargs),
    )

    return agent, model


def test_model_call_limit():
    calls = {"count": 0}

    @tool
    def counter_tool() -> str:
        """Increment a counter and return the result."""
        calls["count"] += 1
        return str(calls["count"])

    agent, model = build_test_agent(
        responses=[
            make_tool_call("counter_tool", "call-1"),
            make_tool_call("counter_tool", "call-2"),
            AIMessage(content="This response must not be used."),
        ],
        tools=[counter_tool],
        max_model_calls=2,
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Keep calling the counter tool.")]}
    )

    # Exactly two model calls were allowed.
    assert model.call_count == 2

    # Both allowed tool calls were executed.
    assert calls["count"] == 2

    # Agent stopped without making another model call.
    assert isinstance(result["messages"][-1], AIMessage)
    assert not result["messages"][-1].tool_calls


def test_tool_call_limit():
    calls = {"count": 0}

    @tool
    def counter_tool() -> str:
        """Increment a counter and return the result."""
        calls["count"] += 1
        return str(calls["count"])

    agent, model = build_test_agent(
        responses=[
            make_tool_call("counter_tool", "call-1"),
            make_tool_call("counter_tool", "call-2"),
            AIMessage(content="Finished"),
        ],
        tools=[counter_tool],
        max_tool_calls=1,
    )

    with pytest.raises(ToolCallLimitExceededError):
        agent.invoke({"messages": [HumanMessage(content="Call the counter twice.")]})

    # First tool call executes, second is blocked.
    assert calls["count"] == 1

    # The model produced two tool requests.
    assert model.call_count == 2


def test_tool_retry_success():
    attempts = {"count": 0}

    @tool
    def flaky_tool() -> str:
        """Simulate a temporary network failure."""
        attempts["count"] += 1

        if attempts["count"] == 1:
            raise TimeoutError("Temporary timeout")

        return "Recovered successfully"

    agent, _ = build_test_agent(
        responses=[
            make_tool_call("flaky_tool", "call-1"),
            AIMessage(content="The tool recovered."),
        ],
        tools=[flaky_tool],
        retryable_tools=["flaky_tool"],
        max_tool_retries=1,
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Execute the flaky tool.")]}
    )

    # Initial attempt + one retry.
    assert attempts["count"] == 2

    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]

    assert len(tool_messages) == 1
    assert tool_messages[0].status == "success"
    assert tool_messages[0].content == "Recovered successfully"

    assert sum(isinstance(message, AIMessage) for message in result["messages"]) == 2


def test_tool_retry_exhausted():
    attempts = {"count": 0}

    @tool
    def failing_tool() -> str:
        """Simulate a persistent connection failure."""
        attempts["count"] += 1
        raise ConnectionError("Service unavailable")

    agent, _ = build_test_agent(
        responses=[
            make_tool_call("failing_tool", "call-1"),
            AIMessage(content="This should not be reached."),
        ],
        tools=[failing_tool],
        retryable_tools=["failing_tool"],
        max_tool_retries=1,
    )

    with pytest.raises(
        ConnectionError,
        match="Service unavailable",
    ):
        agent.invoke({"messages": [HumanMessage(content="Execute the failing tool.")]})

    # Initial attempt + one failed retry.
    assert attempts["count"] == 2


def test_retry_only_selected_tools():
    attempts = {"count": 0}

    @tool
    def allowed_tool() -> str:
        """A tool that is allowed to retry."""
        return "Success"

    @tool
    def unlisted_tool() -> str:
        """A tool that is not allowed to retry."""
        attempts["count"] += 1
        raise TimeoutError("Unlisted tool timeout")

    agent, _ = build_test_agent(
        responses=[
            make_tool_call("unlisted_tool", "call-1"),
            AIMessage(content="This should not be reached."),
        ],
        tools=[allowed_tool, unlisted_tool],
        retryable_tools=["allowed_tool"],
        max_tool_retries=1,
    )

    with pytest.raises(
        TimeoutError,
        match="Unlisted tool timeout",
    ):
        agent.invoke({"messages": [HumanMessage(content="Execute the unlisted tool.")]})

    # Retry middleware must not retry unlisted tools.
    assert attempts["count"] == 1


def test_non_retryable_error():
    attempts = {"count": 0}

    @tool
    def invalid_tool() -> str:
        """Simulate a non-retryable execution error."""
        attempts["count"] += 1
        raise RuntimeError("Internal execution failure")

    agent, _ = build_test_agent(
        responses=[
            make_tool_call("invalid_tool", "call-1"),
            AIMessage(content="This should not be reached."),
        ],
        tools=[invalid_tool],
        retryable_tools=["invalid_tool"],
        max_tool_retries=2,
    )

    with pytest.raises(
        RuntimeError,
        match="Internal execution failure",
    ):
        agent.invoke({"messages": [HumanMessage(content="Execute the invalid tool.")]})

    # RuntimeError is not configured as retryable.
    assert attempts["count"] == 1
