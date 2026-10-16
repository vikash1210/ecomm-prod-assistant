# E-commerce Product Assistant

An LLM-powered e-commerce product assistant built with Python, FastAPI, LangGraph, LangChain, and Retrieval-Augmented Generation (RAG). It answers product questions using a product catalog and a configurable language model.

The application includes an MCP (Model Context Protocol) server for product search, an Agentic RAG workflow, automated tests, linting, and a deployed API.

## Features

- **Product catalog:** Product data is stored in `data/flipkart_products.csv`.
- **Product search API:** Search products through the `/products` endpoint.
- **Conversational API:** Ask product questions through `POST /chat`.
- **Agentic RAG:** Retrieve product context, evaluate relevance, rewrite a search query when needed, and generate an answer grounded in retrieved information.
- **MCP product search:** Expose local catalog search through the `get_product_info` MCP tool.
- **Configurable LLM:** Select a supported model provider through project configuration and environment variables.
- **Fallback behavior:** Return a fallback response when useful product context is unavailable.
- **Automated quality checks:** Run tests with Pytest and lint the code with Ruff.
- **Deployment:** The application is deployed on Render.

## Tech Stack

- Python 3.12
- FastAPI
- LangChain
- LangGraph
- Model Context Protocol (MCP)
- LangChain MCP Adapters
- Astra DB / Astra Vector Search (where configured by the project)
- Pandas
- Pydantic
- Pytest
- Ruff
- GitHub Actions
- Render

## Architecture

```text
Client
  |
  v
FastAPI (/chat)
  |
  v
Agentic RAG workflow
  |-- classify simple greetings
  |-- retrieve product context through MCP
  |-- assess context relevance with the LLM
  |-- rewrite the search query when needed
  |-- generate an answer grounded in the context
  |
  v
MCP client
  |
  v
Product Search MCP Server
  |
  v
Product Retriever / Product Catalog
```

The project also has a `/products` endpoint for product searches. Depending on the configured retriever, catalog retrieval may use the project's configured vector database or local product data. Product prices and availability are only as current as the catalog data; they are not live retailer verification.

## Project Structure

```text
ecomm-prod-assistant/
├── data/
│   └── flipkart_products.csv
├── prod_assistant/
│   ├── api/
│   │   └── main.py
│   ├── config/
│   ├── mcp_servers/
│   │   ├── client.py
│   │   └── product_search_server.py
│   ├── retriever/
│   │   └── retrieval.py
│   ├── utils/
│   │   └── model_loader.py
│   └── workflow/
│       └── agentic_rag_workflow.py
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Prerequisites

- Windows, macOS, or Linux
- Python `>=3.12,<3.13`
- Git
- `uv`
- API credentials for the selected LLM provider
- Any external database credentials required by your chosen configuration

## Installation

Clone the repository:

```cmd
git clone https://github.com/vikash1210/ecomm-prod-assistant.git
cd ecomm-prod-assistant
```

Install the project dependencies with `uv`:

```cmd
uv python install 3.12
uv sync --python 3.12
```

Run project commands through `uv run` so they use the project's managed environment.

## Configuration

1. Create a local `.env` file using the environment variable names expected by the application.
2. Add the API key for the LLM provider selected in the project configuration.
3. Review `prod_assistant/config/config.yaml` for model and retrieval settings.
4. Add any required Astra DB credentials and collection configuration if your selected retriever uses Astra DB.
5. Never commit `.env` files, API keys, or other secrets.

The repository has configuration for model providers including Google, Groq, and OpenAI. Which provider works depends on the configured model and the credentials available in your environment.

## Run the API Locally

Start the development server:

```cmd
uv run uvicorn prod_assistant.api.main:app --reload
```

Open the interactive API documentation:

- http://127.0.0.1:8000/docs

## API Endpoints

### Search products

```http
GET /products
GET /products?q=iPhone%2015
```

The endpoint returns products from the configured catalog/retrieval implementation.

### Ask a product question

Send a `POST` request to `/chat` with a JSON body:

```json
{
  "query": "What is the price of iPhone 15?"
}
```

Example using Windows CMD:

```cmd
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"query\":\"What is the price of iPhone 15?\"}"
```

The response format is:

```json
{
  "answer": "..."
}
```

The exact answer depends on the product catalog and configured LLM. For example, the project was tested with catalog data returning an iPhone 15 price of `INR 67999.00`; this is sample catalog data, not a guarantee of the current retail price.

## MCP Product Search

The MCP server is located at:

```text
prod_assistant/mcp_servers/product_search_server.py
```

It exposes the `get_product_info` tool, which searches the product catalog.

To run the MCP server directly from the repository root:

```cmd
uv run python -m prod_assistant.mcp_servers.product_search_server
```

The server uses the `stdio` transport, so it is intended to be launched by an MCP-compatible client rather than opened as a standalone HTTP endpoint.

A reusable client is provided in:

```text
prod_assistant/mcp_servers/client.py
```

Quick local MCP-client check:

```cmd
uv run python -c "from prod_assistant.mcp_servers.client import MCPProductSearchClient; print(MCPProductSearchClient().search_product('iPhone 15'))"
```

## Run Tests and Lint

Run the full test suite:

```cmd
uv run python -m pytest -q
```

Run Ruff:

```cmd
uv run ruff check prod_assistant tests
```

The project has most recently passed **50 tests** and the Ruff check in the development environment. Run these commands again after making changes to confirm the current state.

## CI and Deployment

- **Source control:** GitHub repository `vikash1210/ecomm-prod-assistant`, branch `master`.
- **CI:** GitHub Actions runs project quality checks, including Ruff and tests.
- **Hosting:** Render web service.
- **Production URL:** https://ecomm-prod-assistant.onrender.com

Test the deployed API with:

```cmd
curl -X POST https://ecomm-prod-assistant.onrender.com/chat -H "Content-Type: application/json" -d "{\"query\":\"What is the price of iPhone 15?\"}"
```

Render's free instance may spin down when inactive, which can make the first request take longer. A successful deployment does not replace checking the application logs and testing the production endpoints.

## Security and Limitations

- Keep API credentials in environment variables or a local `.env` file.
- Do not commit secrets or credentials.
- Treat catalog values as dataset values, not verified real-time retailer prices or inventory.
- Configure production logging, error handling, resource limits, and secret management appropriately before expanding usage.
- Review database access permissions and network settings for your deployment.
- Add a license file and update this section before distributing the project under an open-source license.

## License

No license is specified here yet. Add a `LICENSE` file if you intend to distribute the project under a particular license.
