import os

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from cvhub_agent.agent.factory import AgentFactory
from cvhub_agent.agent.middleware import build_agent_middleware
from cvhub_agent.agent.prompts import DEFAULT_SYSTEM_PROMPT
from cvhub_agent.config.settings import load_llm_config
from cvhub_agent.llm.client import LLMClient
from cvhub_agent.mcp.client import load_mcp_tools
from cvhub_agent.tools.loader import load_tools

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DEEPSEEK_TESTS") != "1" or os.getenv("RUN_MCP_TESTS") != "1",
    reason="Live DeepSeek and MCP tests are disabled",
)


@pytest.mark.asyncio
async def test_agent_mcp_integration():
    # Initialize model
    config = load_llm_config()
    assert config.provider == "deepseek"
    assert not config.enable_thinking

    client = LLMClient(config)

    # Discover MCP tools
    mcp_tools = await load_mcp_tools(
        {
            "visual": "http://127.0.0.1:8001/mcp",
            "rag": "http://127.0.0.1:8002/mcp",
        }
    )

    # Create agent with Native and MCP tools
    agent = AgentFactory(client.model).build(
        tools=load_tools(external_tools=mcp_tools),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        middleware=build_agent_middleware(),
    )

    test_cases = [
        {
            "name": "Visual OCR",
            "prompt": (
                "Use the available OCR capability to read "
                "sample image sample-001. "
                "What is the total amount on the receipt? "
                "Do not guess without using the tool."
            ),
            "tool": "visual_mock_ocr",
            "expected": "12.50",
        },
        {
            "name": "RAG Search",
            "prompt": (
                "Search the available CVHub documentation "
                "for hybrid retrieval. "
                "Which retrieval methods does CVHub RAG combine? "
                "Base your answer on the search results."
            ),
            "tool": "rag_mock_search_documents",
            "expected": "BM25",
        },
    ]

    for case in test_cases:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=case["prompt"])]}
        )

        messages = result["messages"]

        # Verify the model requested the expected tool
        calls = [
            call
            for msg in messages
            if isinstance(msg, AIMessage)
            for call in msg.tool_calls
            if call["name"] == case["tool"]
        ]

        assert calls, f"Expected tool not called: {case['tool']}"

        # Verify remote tool execution
        call_ids = {call["id"] for call in calls}

        tool_results = [
            msg
            for msg in messages
            if isinstance(msg, ToolMessage) and msg.tool_call_id in call_ids
        ]

        assert tool_results, "No matching ToolMessage received"

        assert any(
            msg.status == "success" and case["expected"] in str(msg.content)
            for msg in tool_results
        )

        # Verify final response
        final_message = messages[-1]

        assert isinstance(final_message, AIMessage)
        assert not final_message.tool_calls
        assert str(final_message.content).strip()

        print(f"\n=== {case['name']} ===")

        for msg in messages:
            print(f"{type(msg).__name__}: {msg.content}")

            if isinstance(msg, AIMessage) and msg.tool_calls:
                print(f"Tool calls: {msg.tool_calls}")

        print(f"\nFinal answer: {final_message.content}")
