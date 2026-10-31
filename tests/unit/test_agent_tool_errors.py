from collections.abc import Sequence
from typing import Any, Self

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from cvhub_agent.agent.factory import AgentFactory
from cvhub_agent.agent.middleware import build_agent_middleware
from cvhub_agent.agent.prompts import DEFAULT_SYSTEM_PROMPT
from cvhub_agent.tools.native_tools import calculate


class ScriptedChatModel(FakeMessagesListChatModel):
    """Fake chat model that supports tool binding."""

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> Self:
        return self


def test_expected_tool_error():
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {
                            "a": 10,
                            "b": 0,
                            "operation": "divide",
                        },
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="Division by zero is not allowed."),
        ]
    )

    agent = AgentFactory(model).build(
        tools=[calculate],
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        middleware=build_agent_middleware(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Calculate 10 divided by zero.")]}
    )

    messages = result["messages"]

    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]

    assert len(tool_messages) == 1

    tool_result = tool_messages[0]

    assert tool_result.tool_call_id == "call-1"
    assert tool_result.status == "error"
    assert "Division by zero" in str(tool_result.content)

    assert isinstance(messages[-1], AIMessage)
    assert not messages[-1].tool_calls

    print("\n=== Expected Tool Error ===")
    for msg in messages:
        print(f"{type(msg).__name__}: {msg.content}")


def test_unexpected_tool_error_propagates():
    @tool
    def crash_tool() -> str:
        """Simulate an unexpected internal tool failure."""
        raise RuntimeError("Unexpected internal failure")

    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "crash_tool",
                        "args": {},
                        "id": "call-2",
                    }
                ],
            ),
            AIMessage(content="This should not be reached."),
        ]
    )

    agent = AgentFactory(model).build(
        tools=[crash_tool],
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        middleware=build_agent_middleware(),
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected internal failure",
    ):
        agent.invoke({"messages": [HumanMessage(content="Execute the crash tool.")]})
