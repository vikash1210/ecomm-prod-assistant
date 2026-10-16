
"""Reusable MCP client for product search."""

import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPProductSearchClient:
    """Discover and invoke the product search MCP tool."""

    async def _search_product(self, query: str) -> str:
        client = MultiServerMCPClient(
            {
                "products": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "prod_assistant.mcp_servers.product_search_server",
                    ],
                }
            }
        )

        tools = await client.get_tools()
        product_tool = next(
            (
                tool
                for tool in tools
                if tool.name == "get_product_info"
            ),
            None,
        )

        if product_tool is None:
            raise RuntimeError(
                "MCP tool 'get_product_info' was not discovered"
            )

        result = await product_tool.ainvoke({"query": query})
        return str(result)

    def search_product(self, query: str) -> str:
        """Run product search from synchronous application code."""
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        return asyncio.run(self._search_product(query.strip()))
