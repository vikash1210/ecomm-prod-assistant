"""FastAPI application entry point."""

from fastapi import FastAPI

from prod_assistant.config.settings import settings
from prod_assistant.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="An LLM-powered e-commerce product assistant.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application health status."""
    logger.info("Health check requested")
    return {"status": "healthy", "app": settings.app_name}
