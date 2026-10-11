
"""MCP server exposing local product search tools.

Run from the repository root with:
    uv run python -m prod_assistant.mcp_servers.product_search_server
"""

from mcp.server.fastmcp import FastMCP

from prod_assistant.retriever.retrieval import Retriever

mcp = FastMCP("ecommerce-product-assistant")
retriever = Retriever()


@mcp.tool()
def get_product_info(query: str, limit: int = 4) -> str:
    """Search the local product catalog and return matching product details."""
    try:
        return retriever.format_results(query, limit=limit)
    except ValueError as exc:
        return f"Invalid request: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
