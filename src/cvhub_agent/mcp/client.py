from collections.abc import Mapping

from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool


async def load_mcp_tools(
    servers: Mapping[str, str],
) -> list[BaseTool]:
    """Discover LangChain tools from multiple MCP servers."""

    if not servers:
        return []

    config = {"mcpServers": {name: {"url": url} for name, url in servers.items()}}

    async with MCPAdapter(config) as adapter:
        tools = await adapter.list_tools()

    return tools
