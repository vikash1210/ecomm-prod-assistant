"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient

from prod_assistant.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_healthy_status() -> None:
    """Verify the health endpoint returns a successful response."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["app"] == "E-commerce Product Assistant"
