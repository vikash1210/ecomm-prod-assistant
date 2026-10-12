# E-commerce Product Assistant

An LLM-powered e-commerce product assistant built with Python, FastAPI, LangGraph, and Retrieval-Augmented Generation (RAG).

The application searches a local product catalog and answers product-related questions using retrieved product information and a configurable language model.

## Features

- **Product Catalog:** Load product information from a CSV file.
- **Product Search API:** Search catalog items by query.
- **Conversational API:** Ask questions about product prices, ratings, and details.
- **Agentic RAG Workflow:** Retrieve relevant information, assess relevance, rewrite queries, and generate grounded answers.
- **LLM Configuration:** Configure language models through project settings and environment variables.
- **Fallback Responses:** Avoid inventing product information when the retrieved context does not contain the answer.
- **Automated Tests:** Validate API behavior and configuration.

## Tech Stack

- Python
- FastAPI
- LangChain
- LangGraph
- Pandas
- Pydantic
- Pytest
- Ruff

## Project Structure

```text
ecomm-prod-assistant/
├── data/
│   └── flipkart_products.csv
├── prod_assistant/
│   ├── api/
│   ├── config/
│   ├── retriever/
│   ├── utils/
│   └── workflow/
├── tests/
├── README.md
└── pyproject.toml
```

## Prerequisites

- Python version supported by the project dependencies
- Git
- API credentials for the language model provider you configure

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ecomm-prod-assistant
```

Create and activate a virtual environment on Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Install the project dependencies using the package manager and instructions defined in `pyproject.toml`.

If the project is configured for `uv`, use:

```cmd
uv sync
```

## Configuration

1. Create or update the local `.env` file using the environment-variable names expected by the application.
2. Add the API key for your selected language model provider.
3. Review `prod_assistant/config/config.yaml` to configure the model and retriever settings.
4. Never commit `.env` files or API secrets to Git.

The application configuration currently defines Google, Groq, and OpenAI model settings. Availability depends on your credentials, installed integrations, and selected provider.

## Product Catalog

The application reads product information from:

```text
data/flipkart_products.csv
```

The CSV contains product fields such as title, price, rating, product URL, and source.

When adding product titles containing commas, enclose the title in double quotes to preserve valid CSV formatting.

## Run the API

Activate the virtual environment and start the development server:

```cmd
uvicorn prod_assistant.api.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

### Search Products

```http
GET /products?q=iPhone%2015
```

Returns matching products from the catalog.

To retrieve the catalog without a search query:

```http
GET /products
```

### Ask a Product Question

Use the `/chat` endpoint in the interactive API documentation to submit a product-related question.

Example question:

```text
What is the price of Apple iPhone 15?
```

The response depends on the information available in the product catalog and the configured language model.

## Run Tests

Run the automated test suite:

```cmd
pytest -q
```

Run code quality checks:

```cmd
ruff check .
```

## Current Validation

The project has passed the following local checks:

- 50 automated tests passed.
- Ruff code quality checks passed.

These results reflect the local development environment and may change as the project evolves.

## Security

- Keep API keys in environment variables or a local `.env` file.
- Do not commit secrets or credentials.
- Use a production-ready server configuration before deploying publicly.

## License

Add your chosen open-source license before distributing this project.
