import os

import pytest

from cvhub_agent.mcp.client import load_mcp_tools
from cvhub_agent.tools.loader import load_tools

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MCP_TESTS") != "1",
    reason="Live MCP tests are disabled",
)


@pytest.mark.asyncio
async def test_mcp_multi_server_integration():
    servers = {
        "visual": "http://127.0.0.1:8001/mcp",
        "rag": "http://127.0.0.1:8002/mcp",
    }

    # Discover MCP tools
    mcp_tools = await load_mcp_tools(servers)
    tool_map = {tool.name: tool for tool in mcp_tools}

    expected_tools = {
        "visual_mock_ocr",
        "rag_mock_search_documents",
        "rag_mock_get_document",
    }

    assert expected_tools.issubset(tool_map)
    assert len(tool_map) == len(mcp_tools)

    # Execute remote Visual Tool
    ocr_result = await tool_map["visual_mock_ocr"].ainvoke({"image_id": "sample-001"})

    assert "CVHub Demo Receipt" in str(ocr_result)

    # Execute remote RAG Search Tool
    search_result = await tool_map["rag_mock_search_documents"].ainvoke(
        {"query": "hybrid retrieval", "top_k": 2}
    )

    assert "doc-001" in str(search_result)

    # Execute remote RAG Document Tool
    document_result = await tool_map["rag_mock_get_document"].ainvoke(
        {"document_id": "doc-001"}
    )

    assert "BM25" in str(document_result)

    # Merge Native and MCP Tools
    all_tools = load_tools(external_tools=mcp_tools)
    all_names = [tool.name for tool in all_tools]

    assert len(all_names) == len(set(all_names))
    assert expected_tools.issubset(all_names)
    assert {"calculate", "get_current_date"}.issubset(all_names)

    print("\n=== Discovered Agent Tools ===")
    for name in all_names:
        print(f"- {name}")

    print("\nMCP integration test passed.")
