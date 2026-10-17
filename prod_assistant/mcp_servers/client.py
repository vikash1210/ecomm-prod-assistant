
"""Reusable MCP client for product search."""

import asyncio
import logging
import sys
from time import perf_counter

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


class MCPProductSearchClient:
    """Discover and invoke the product search MCP tool."""

    async def _search_product(self, query: str) -> str:
        total_started = perf_counter()

        try:
            # Measure MCP client initialization.
            started = perf_counter()
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
            logger.info(
                "MCP client initialization duration: %.3f seconds",
                perf_counter() - started,
            )

            # Measure tool discovery.
            started = perf_counter()
            tools = await client.get_tools()
            logger.info(
                "MCP get_tools duration: %.3f seconds",
                perf_counter() - started,
            )

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

            # Measure the actual product search tool call.
            started = perf_counter()
            result = await product_tool.ainvoke({"query": query})
            logger.info(
                "MCP product tool invocation duration: %.3f seconds",
                perf_counter() - started,
            )

            return str(result)

        except Exception:
            logger.exception("MCP product search failed")
            raise

        finally:
            logger.info(
                "MCP total async search duration: %.3f seconds",
                perf_counter() - total_started,
            )

    def search_product(self, query: str) -> str:
        """Run product search from synchronous application code."""
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        started = perf_counter()

        try:
            return asyncio.run(self._search_product(query.strip()))
        finally:
            logger.info(
                "MCP total search_product duration: %.3f seconds",
                perf_counter() - started,
            )
