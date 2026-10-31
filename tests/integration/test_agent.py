import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cvhub_agent.agent.factory import AgentFactory
from cvhub_agent.agent.middleware import build_agent_middleware
from cvhub_agent.agent.prompts import DEFAULT_SYSTEM_PROMPT
from cvhub_agent.config.settings import load_llm_config
from cvhub_agent.llm.client import LLMClient
from cvhub_agent.tools.loader import load_tools

if os.getenv("RUN_DEEPSEEK_TESTS") != "1":
    pytest.skip(
        "Live DeepSeek tests are disabled",
        allow_module_level=True,
    )


def test_agent_tool_execution():
    # Initialize LLM
    config = load_llm_config()

    assert config.provider == "deepseek"
    assert not config.enable_thinking

    llm_client = LLMClient(config)

    # Build agent
    factory = AgentFactory(model=llm_client.model)

    agent = factory.build(
        tools=load_tools(),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        middleware=build_agent_middleware(),
    )

    # Execute agent
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use the calculate tool to multiply "
                        "125 by 36. Return the result."
                    )
                )
            ]
        }
    )

    messages = result["messages"]

    # Verify actual tool calls
    tool_calls = [
        call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "calculate"
    ]

    assert tool_calls, "Agent did not request calculate tool"

    # Verify that the tool was executed
    call_ids = {call["id"] for call in tool_calls}

    tool_results = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id in call_ids
    ]

    assert tool_results, "Agent did not execute calculate tool"

    assert any(
        float(message.content) == 4500.0
        for message in tool_results
        if isinstance(message.content, str) and message.status == "success"
    )

    # Verify final answer
    final_message = messages[-1]

    assert isinstance(final_message, AIMessage)
    assert not final_message.tool_calls
    assert "4500" in str(final_message.content)

    # Print execution trace
    print("\n=== Agent Execution ===")

    for message in messages:
        print(f"{type(message).__name__}: {message.content}")

        if isinstance(message, AIMessage) and message.tool_calls:
            print(f"Tool calls: {message.tool_calls}")


def test_agent_multi_step_tool_execution():
    config = load_llm_config()

    assert config.provider == "deepseek"
    assert not config.enable_thinking

    client = LLMClient(config)
    factory = AgentFactory(model=client.model)

    agent = factory.build(
        tools=load_tools(),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        middleware=build_agent_middleware(),
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use the calculate tool to multiply 125 by 36. "
                        "Wait for the tool result. Then use the calculate "
                        "tool again to add 250 to that result. "
                        "Do not combine the two operations into one call."
                    )
                )
            ]
        }
    )

    messages = result["messages"]

    # Collect tool calls with their message positions
    calls = [
        (index, call)
        for index, message in enumerate(messages)
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "calculate"
    ]

    multiply_calls = [
        (index, call)
        for index, call in calls
        if call["args"]["operation"] == "multiply"
    ]

    add_calls = [
        (index, call) for index, call in calls if call["args"]["operation"] == "add"
    ]

    assert len(multiply_calls) == 1
    assert len(add_calls) == 1

    multiply_index, multiply_call = multiply_calls[0]
    add_index, add_call = add_calls[0]

    # Match tool results to their corresponding tool calls
    multiply_results = [
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, ToolMessage)
        and message.tool_call_id == multiply_call["id"]
    ]

    add_results = [
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, ToolMessage) and message.tool_call_id == add_call["id"]
    ]

    assert len(multiply_results) == 1
    assert len(add_results) == 1

    multiply_result_index, multiply_result = multiply_results[0]
    add_result_index, add_result = add_results[0]

    # Verify sequential execution
    assert multiply_index < multiply_result_index < add_index < add_result_index

    # Verify tool outputs
    assert multiply_result.status == "success"
    assert add_result.status == "success"

    assert float(str(multiply_result.content)) == 4500.0
    assert float(str(add_result.content)) == 4750.0

    # Verify final answer
    final_message = messages[-1]

    assert isinstance(final_message, AIMessage)
    assert not final_message.tool_calls
    assert "4750" in str(final_message.content).replace(",", "")

    # Print execution trace
    print("\n=== Multi-Step Agent Execution ===")

    for message in messages:
        print(f"{type(message).__name__}: {message.content}")

        if isinstance(message, AIMessage) and message.tool_calls:
            print(f"Tool calls: {message.tool_calls}")
