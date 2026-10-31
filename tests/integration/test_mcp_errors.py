import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from cvhub_agent.mcp.client import load_mcp_tools

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MCP_TESTS") != "1",
    reason="Live MCP tests are disabled",
)


def _get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def fault_server():
    """Start and stop an isolated MCP server."""

    port = _get_free_port()

    server_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "mcp_servers"
        / "fault_server.py"
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(server_path),
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        deadline = time.monotonic() + 10

        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("Fault MCP Server exited unexpectedly")

            try:
                with socket.create_connection(
                    ("127.0.0.1", port),
                    timeout=0.1,
                ):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("Fault MCP Server startup timed out")

        yield (
            f"http://127.0.0.1:{port}/mcp",
            process,
        )

    finally:
        if process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


@pytest.mark.asyncio
async def test_offline_server_does_not_block_healthy_server(
    fault_server,
):
    url, _ = fault_server

    # Reserve an unused port without listening.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        offline_port = reserved.getsockname()[1]

        tools = await asyncio.wait_for(
            load_mcp_tools(
                {
                    "offline": (f"http://127.0.0.1:{offline_port}/mcp"),
                    "fault": url,
                }
            ),
            timeout=10,
        )

    tool_map = {tool.name: tool for tool in tools}

    # Healthy server must remain available.
    assert "fault_mock_ping" in tool_map

    # Offline server must not expose tools.
    assert not any(name.startswith("offline_") for name in tool_map)

    response = await tool_map["fault_mock_ping"].ainvoke({})

    assert "pong" in str(response)


@pytest.mark.asyncio
async def test_mcp_tool_execution_error(fault_server):
    url, _ = fault_server

    tools = await load_mcp_tools({"fault": url})
    tool_map = {tool.name: tool for tool in tools}

    tool = tool_map["mock_fail"]

    # Invoke the tool with a LangChain tool-call envelope.
    response = await tool.ainvoke(
        {
            "type": "tool_call",
            "name": tool.name,
            "args": {},
            "id": "call-error-1",
        }
    )

    assert isinstance(response, ToolMessage)
    assert response.tool_call_id == "call-error-1"
    assert response.status == "error"
    assert f"Error executing tool {tool.name}" in str(response.content)


@pytest.mark.asyncio
async def test_mcp_server_disconnect_after_discovery(
    fault_server,
):
    url, process = fault_server

    # Discover tools while the server is running.
    tools = await load_mcp_tools({"fault": url})
    tool_map = {tool.name: tool for tool in tools}

    ping_tool = tool_map["mock_ping"]

    # Verify the tool works before disconnecting.
    response = await ping_tool.ainvoke({})
    assert "pong" in str(response)

    # Stop the server after tool discovery.
    process.terminate()
    await asyncio.to_thread(process.wait, 3)

    # A disconnected server must not return success.
    with pytest.raises(Exception) as error:
        await asyncio.wait_for(
            ping_tool.ainvoke({}),
            timeout=5,
        )

    # The failure should come from the connection,
    # not from the outer test timeout.
    assert not isinstance(error.value, TimeoutError)


@pytest.mark.asyncio
async def test_mcp_tool_request_timeout(fault_server):
    url, _ = fault_server

    tools = await load_mcp_tools({"fault": url})
    tool_map = {tool.name: tool for tool in tools}

    # Verify the server is available.
    ping_response = await tool_map["mock_ping"].ainvoke({})
    assert "pong" in str(ping_response)

    slow_tool = tool_map["mock_slow"]

    # Remote execution takes 3 seconds.
    # The client deadline is only 0.5 seconds.
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            slow_tool.ainvoke({"delay_seconds": 3.0}),
            timeout=0.5,
        )
