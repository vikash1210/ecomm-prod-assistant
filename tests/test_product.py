"""Tests for the product data model."""

import pytest
from pydantic import ValidationError

from prod_assistant.retriever.product import Product


def test_product_accepts_valid_data() -> None:
    product = Product(
        id="P001",
        name="Wireless Mouse",
        description="Ergonomic wireless mouse",
        category="Electronics",
        price=799.0,
    )

    assert product.id == "P001"
    assert product.name == "Wireless Mouse"
    assert product.price == 799.0
    assert product.currency == "INR"
    assert product.in_stock is True


def test_product_rejects_non_positive_price() -> None:
    with pytest.raises(ValidationError):
        Product(
            id="P002",
            name="Keyboard",
            category="Electronics",
            price=0,
        )


def test_product_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        Product(
            id="P003",
            name="",
            category="Electronics",
            price=499.0,
        )
