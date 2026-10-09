"""Tests for the product catalog service."""

from prod_assistant.retriever.catalog import ProductCatalog
from prod_assistant.retriever.product import Product


def create_sample_catalog() -> ProductCatalog:
    products = [
        Product(
            id="P001",
            name="Wireless Mouse",
            description="Ergonomic wireless mouse",
            category="Electronics",
            price=799.0,
        ),
        Product(
            id="P002",
            name="Mechanical Keyboard",
            description="RGB gaming keyboard",
            category="Electronics",
            price=2499.0,
        ),
        Product(
            id="P003",
            name="Cotton T-Shirt",
            description="Comfortable cotton shirt",
            category="Clothing",
            price=499.0,
        ),
    ]
    return ProductCatalog(products)


def test_list_products_returns_all_products() -> None:
    catalog = create_sample_catalog()
    assert len(catalog.list_products()) == 3


def test_get_product_by_id() -> None:
    catalog = create_sample_catalog()
    product = catalog.get_product("P002")

    assert product is not None
    assert product.name == "Mechanical Keyboard"


def test_get_unknown_product_returns_none() -> None:
    catalog = create_sample_catalog()
    assert catalog.get_product("UNKNOWN") is None


def test_search_products_by_name() -> None:
    catalog = create_sample_catalog()
    results = catalog.search_products(query="mouse")

    assert len(results) == 1
    assert results[0].id == "P001"


def test_search_products_by_category() -> None:
    catalog = create_sample_catalog()
    results = catalog.search_products(category="Electronics")

    assert len(results) == 2


def test_search_products_by_query_and_category() -> None:
    catalog = create_sample_catalog()
    results = catalog.search_products(
        query="gaming",
        category="Electronics",
    )

    assert len(results) == 1
    assert results[0].id == "P002"
