
"""Reusable MCP client for product search."""

import asyncio
import logging
import sys
from contextlib import AsyncExitStack
from time import perf_counter

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPProductSearchClient:
    """Keep one MCP stdio session alive and reuse its discovered tools."""

    def __init__(self) -> None:
        self._loop = None
        self._stack = None
        self._session = None
        self._tools = None
        self._lock = None

    async def _ensure_initialized(self) -> None:
        """Initialize the MCP server/session once on the persistent loop."""
        if self._tools is not None:
            return

        started = perf_counter()
        self._stack = AsyncExitStack()

        try:
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m",
                    "prod_assistant.mcp_servers.product_search_server",
                ],
            )

            read, write = await self._stack.enter_async_context(
                stdio_client(server_params)
            )

            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )

            await self._session.initialize()

            self._tools = await load_mcp_tools(self._session)

            if not any(
                tool.name == "get_product_info" for tool in self._tools
            ):
                raise RuntimeError(
                    "MCP tool 'get_product_info' was not discovered"
                )

            logger.info(
                "MCP initialization and tool discovery duration: %.3f seconds",
                perf_counter() - started,
            )

        except Exception:
            logger.exception("MCP initialization failed")
            await self._stack.aclose()
            self._stack = None
            self._session = None
            self._tools = None
            raise

    async def _search_product(self, query: str) -> str:
        """Search products using the persistent MCP session."""
        total_started = perf_counter()

        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            await self._ensure_initialized()

            product_tool = next(
                tool
                for tool in self._tools
                if tool.name == "get_product_info"
            )

            started = perf_counter()
            result = await product_tool.ainvoke({"query": query})
            logger.info(
                "MCP product tool invocation duration: %.3f seconds",
                perf_counter() - started,
            )

        logger.info(
            "MCP total async search duration: %.3f seconds",
            perf_counter() - total_started,
        )
        return str(result)

    def search_product(self, query: str) -> str:
        """Run product search on a persistent event loop."""
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        started = perf_counter()

        try:
            # Create a dedicated event loop once and reuse it.
            if self._loop is None:
                self._loop = asyncio.new_event_loop()

            asyncio.set_event_loop(self._loop)
            return self._loop.run_until_complete(
                self._search_product(query.strip())
            )
        except Exception:
            logger.exception("MCP product search failed")
            raise
        finally:
            logger.info(
                "MCP total search_product duration: %.3f seconds",
                perf_counter() - started,
            )
