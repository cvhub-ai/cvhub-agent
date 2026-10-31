import argparse
import asyncio

from mcp.server import MCPServer

mcp = MCPServer("CVHub Mock Fault Server")


@mcp.tool()
def mock_ping() -> str:
    """Check whether the mock MCP server is available."""
    return "pong"


@mcp.tool()
def mock_fail() -> str:
    """Simulate an expected MCP tool execution failure."""
    raise ValueError("Simulated MCP tool failure")


@mcp.tool()
async def mock_slow(delay_seconds: float = 3.0) -> str:
    """Simulate a slow remote tool execution."""
    if delay_seconds < 0:
        raise ValueError("delay_seconds must not be negative")

    await asyncio.sleep(delay_seconds)
    return "Slow operation completed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=args.port,
        stateless_http=True,
        json_response=True,
    )
