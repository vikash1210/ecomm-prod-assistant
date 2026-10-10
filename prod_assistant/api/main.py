
"""FastAPI application entry point."""

from fastapi import FastAPI, Query

from prod_assistant.config.settings import settings
from prod_assistant.retriever.catalog import ProductCatalog
from prod_assistant.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="An LLM-powered e-commerce product assistant.",
)

catalog = ProductCatalog.from_csv()


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application health status."""
    logger.info("Health check requested")
    return {"status": "healthy", "app": settings.app_name}


@app.get("/products")
def get_products(
    q: str = Query(default="", description="Search products by name"),
) -> list[dict]:
    """Return all products or search by product name."""
    products = catalog.search_products(query=q)
    return [product.model_dump() for product in products]
