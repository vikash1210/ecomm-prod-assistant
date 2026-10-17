
"""Reusable MCP client for product search with a persistent session."""

import asyncio
import atexit
import logging
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack
from time import perf_counter

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPProductSearchClient:
    """Reuse one MCP session on a dedicated background event loop."""

    def __init__(self) -> None:
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._loop_lock = threading.Lock()
        self._init_lock = None
        self._search_lock = None
        self._stack = None
        self._session = None
        self._tools = None
        self._closed = False

        atexit.register(self.close)

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Run the persistent event loop on its dedicated thread."""
        asyncio.set_event_loop(loop)
        self._ready.set()

        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)

            for task in pending:
                task.cancel()

            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            loop.close()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Create the background event loop if it does not exist."""
        with self._loop_lock:
            if self._closed:
                raise RuntimeError("MCP client is closed")

            if (
                self._loop is not None
                and self._thread is not None
                and self._thread.is_alive()
                and self._loop.is_running()
            ):
                return self._loop

            self._ready.clear()
            loop = asyncio.new_event_loop()

            thread = threading.Thread(
                target=self._run_loop,
                args=(loop,),
                name="mcp-product-search",
                daemon=True,
            )

            self._loop = loop
            self._thread = thread
            thread.start()

            if not self._ready.wait(timeout=10):
                raise RuntimeError("MCP event loop failed to start")

            if not loop.is_running():
                raise RuntimeError("MCP event loop is not running")

            return loop

    async def _ensure_initialized(self) -> None:
        """Start the MCP server, initialize its session, and load tools."""
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()

        async with self._init_lock:
            if self._tools is not None:
                return

            started = perf_counter()
            stack = AsyncExitStack()

            try:
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "prod_assistant.mcp_servers.product_search_server",
                    ],
                )

                read, write = await stack.enter_async_context(
                    stdio_client(params)
                )

                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )

                await session.initialize()
                tools = await load_mcp_tools(session)

                if not any(
                    tool.name == "get_product_info" for tool in tools
                ):
                    raise RuntimeError(
                        "MCP tool 'get_product_info' was not discovered"
                    )

                self._stack = stack
                self._session = session
                self._tools = tools
                self._search_lock = asyncio.Lock()

                logger.info(
                    "MCP initialization and tool discovery duration: %.3f seconds",
                    perf_counter() - started,
                )

            except Exception:
                logger.exception("MCP initialization failed")
                await stack.aclose()
                raise

    def initialize(self) -> None:
        """Initialize MCP before the first product request."""
        loop = self._ensure_loop()

        future = asyncio.run_coroutine_threadsafe(
            self._ensure_initialized(),
            loop,
        )

        try:
            future.result(timeout=120)
        except FutureTimeoutError as exc:
            future.cancel()
            logger.error("MCP initialization timed out")
            raise TimeoutError(
                "MCP initialization exceeded 120 seconds"
            ) from exc

    async def _search_product(self, query: str) -> str:
        """Search the product catalog using the persistent MCP session."""
        total_started = perf_counter()

        await self._ensure_initialized()

        async with self._search_lock:
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
        """Synchronously search using the persistent MCP session."""
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        started = perf_counter()
        future = None

        try:
            loop = self._ensure_loop()

            future = asyncio.run_coroutine_threadsafe(
                self._search_product(query.strip()),
                loop,
            )

            return future.result(timeout=120)

        except FutureTimeoutError as exc:
            if future is not None:
                future.cancel()

            logger.error("MCP product search timed out")

            raise TimeoutError(
                "MCP product search exceeded 120 seconds"
            ) from exc

        except Exception:
            logger.exception("MCP product search failed")
            raise

        finally:
            logger.info(
                "MCP total search_product duration: %.3f seconds",
                perf_counter() - started,
            )

    async def _close_async(self) -> None:
        """Release MCP resources on the event-loop thread."""
        stack = self._stack

        self._stack = None
        self._session = None
        self._tools = None
        self._search_lock = None
        self._init_lock = None

        if stack is not None:
            await stack.aclose()

    def close(self) -> None:
        """Close the MCP session and stop the background event loop."""
        with self._loop_lock:
            if self._closed:
                return

            self._closed = True
            loop = self._loop
            thread = self._thread

        if (
            loop is None
            or thread is None
            or not thread.is_alive()
        ):
            return

        try:
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._close_async(),
                    loop,
                )
                future.result(timeout=10)

        except Exception:
            logger.exception("Error while closing MCP client")

        finally:
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)

            if thread.is_alive():
                thread.join(timeout=10)
