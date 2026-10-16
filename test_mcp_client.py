import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    client = MultiServerMCPClient(
        {
            "products": {
                "transport": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "python",
                    "-m",
                    "prod_assistant.mcp_servers.product_search_server",
                ],
            }
        }
    )

    tools = await client.get_tools()
    print("Available MCP tools:", [tool.name for tool in tools])

    product_tool = next(
        (
            item
            for item in tools
            if item.name == "get_product_info"
        ),
        None,
    )

    if product_tool is None:
        raise RuntimeError("get_product_info was not discovered")

    result = await product_tool.ainvoke(
        {"query": "iPhone 15"}
    )

    print("\nProduct search result:\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())