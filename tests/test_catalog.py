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

def test_load_catalog_from_csv(tmp_path) -> None:
    csv_file = tmp_path / "products.csv"
    csv_file.write_text(
        "product_title,price,rating,product_url,source\n"
        "Test Phone,500,4.5,https://dummyjson.com/products/1,dummyjson\n"
        "Test Laptop,1200,4.2,https://www.flipkart.com/test-product,flipkart\n",
        encoding="utf-8",
    )

    catalog = ProductCatalog.from_csv(csv_file)
    products = catalog.list_products()

    assert len(products) == 2
    assert products[0].name == "Test Phone"
    assert products[0].price == 500.0
    assert products[0].currency == "USD"
    assert products[1].name == "Test Laptop"
    assert products[1].price == 1200.0
    assert products[1].currency == "INR"
