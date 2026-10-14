
"""FastAPI application entry point."""

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from prod_assistant.config.settings import settings
from prod_assistant.retriever.catalog import ProductCatalog
from prod_assistant.utils.logger import get_logger
from prod_assistant.workflow.agentic_rag_workflow import AgenticRAG


logger = get_logger(__name__)

# Resolve the project root directory.
# main.py is located inside prod_assistant/api/.
BASE_DIR = Path(__file__).resolve().parents[2]

# Configure the HTML templates directory.
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="An LLM-powered e-commerce product assistant.",
)

# Load the product catalog from the project root.
catalog = ProductCatalog.from_csv(
    BASE_DIR / "data" / "flipkart_products.csv"
)

print("Loaded product count:", len(catalog.list_products()))


class ChatRequest(BaseModel):
    """Request body for product questions."""

    query: str = Field(
        min_length=1,
        description="User's product question",
    )


class ChatResponse(BaseModel):
    """Response returned by the chat endpoint."""

    answer: str


@lru_cache(maxsize=1)
def get_agent() -> AgenticRAG:
    """Create the RAG agent once and reuse it."""

    return AgenticRAG()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Render the ShopSmart homepage."""

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application health status."""

    logger.info("Health check requested")

    return {
        "status": "healthy",
        "app": settings.app_name,
    }


@app.get("/products")
def get_products(q: str = "") -> list[dict]:
    """Return all products or search the catalog."""

    products = catalog.search_products(query=q)

    return [
        product.model_dump()
        for product in products
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a product question through the Agentic RAG workflow."""

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty.",
        )

    try:
        answer = get_agent().run(query)

        return ChatResponse(answer=answer)

    except ValueError as exc:
        logger.exception("Chat configuration or input error")

        raise HTTPException(
            status_code=500,
            detail=(
                "Chat configuration or input error. "
                "Check server logs."
            ),
        ) from exc

    except Exception as exc:
        logger.exception("Chat request failed")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the question right now.",
        ) from exc
