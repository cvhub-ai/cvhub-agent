from typing import Any

from mcp.server import MCPServer

mcp = MCPServer("CVHub Mock RAG Server")


MOCK_DOCUMENTS = {
    "doc-001": {
        "title": "CVHub RAG Overview",
        "content": (
            "CVHub RAG uses hybrid retrieval combining "
            "BM25 and vector search. Retrieved candidates "
            "are fused with RRF and processed by a reranker."
        ),
    },
    "doc-002": {
        "title": "CVHub OCR Service",
        "content": (
            "CVHub OCR is an independent computer vision "
            "microservice. It provides text recognition "
            "capabilities through a gRPC interface."
        ),
    },
}


@mcp.tool()
def mock_search_documents(
    query: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Search simulated CVHub documents.

    Use this tool to retrieve relevant documents.
    This tool returns mock data, not real RAG results.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")

    results = []

    for document_id, document in MOCK_DOCUMENTS.items():
        searchable_text = (document["title"] + " " + document["content"]).lower()

        if query.lower() in searchable_text:
            results.append(
                {
                    "document_id": document_id,
                    "title": document["title"],
                    "snippet": document["content"][:150],
                }
            )

    return {
        "query": query,
        "results": results[:top_k],
        "is_mock": True,
    }


@mcp.tool()
def mock_get_document(
    document_id: str,
) -> dict[str, str | bool]:
    """Retrieve a simulated document by its ID.

    Use this tool to obtain full document content
    after finding a document through search.
    """
    document = MOCK_DOCUMENTS.get(document_id)

    if document is None:
        raise ValueError(f"Unknown document ID: {document_id}")

    return {
        "document_id": document_id,
        "title": document["title"],
        "content": document["content"],
        "is_mock": True,
    }


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8002,
        stateless_http=True,
        json_response=True,
    )
