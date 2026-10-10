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


def test_products_endpoint_returns_all_products() -> None:
    """Verify the products endpoint returns catalog products."""
    response = client.get("/products")

    assert response.status_code == 200
    assert len(response.json()) == 6


def test_products_endpoint_searches_by_query() -> None:
    """Verify the products endpoint filters by query."""
    response = client.get("/products", params={"q": "iPhone"})

    assert response.status_code == 200
    products = response.json()
    assert len(products) == 2
    assert all("iphone" in product["name"].casefold() for product in products)
