from mcp.server import MCPServer

mcp = MCPServer("CVHub Mock Visual Server")


MOCK_OCR_RESULTS = {
    "sample-001": ("CVHub Demo Receipt\nItem: Notebook\nQuantity: 2\nTotal: 12.50 EUR"),
}


@mcp.tool()
def mock_ocr(image_id: str) -> dict[str, str | bool]:
    """Return simulated OCR text for a sample image.

    Use this tool to test OCR tool invocation over MCP.
    This tool does not process real images.
    """
    text = MOCK_OCR_RESULTS.get(image_id)

    if text is None:
        raise ValueError(f"Unknown sample image ID: {image_id}")

    return {
        "image_id": image_id,
        "text": text,
        "is_mock": True,
    }


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8001,
        stateless_http=True,
        json_response=True,
    )
