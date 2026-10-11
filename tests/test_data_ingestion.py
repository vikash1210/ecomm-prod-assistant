
"""Unit tests for product data ingestion."""

import pandas as pd
import pytest
from langchain_core.documents import Document

from prod_assistant.etl.data_ingestion import DataIngestion


@pytest.fixture
def ingestion():
    """Create a DataIngestion instance for unit tests."""
    return DataIngestion()


def sample_dataframe():
    """Return representative product rows."""
    return pd.DataFrame(
        [
            {
                "product_title": "Test Phone",
                "price": 500.0,
                "rating": 4.5,
                "product_url": "https://dummyjson.com/products/1",
                "source": "dummyjson",
            },
            {
                "product_title": "Test Laptop",
                "price": 1200.0,
                "rating": 4.2,
                "product_url": "https://www.flipkart.com/test-laptop",
                "source": "flipkart",
            },
        ]
    )


def test_load_data_reads_valid_csv(ingestion, tmp_path):
    csv_path = tmp_path / "products.csv"
    sample_dataframe().to_csv(csv_path, index=False)

    df = ingestion.load_data(csv_path)

    assert len(df) == 2
    assert "product_title" in df.columns
    assert "price" in df.columns
    assert "product_url" in df.columns


def test_load_data_raises_for_missing_file(ingestion, tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Product CSV not found"):
        ingestion.load_data(missing_path)


def test_load_data_rejects_missing_required_columns(ingestion, tmp_path):
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame(
        [{"product_title": "Test Phone", "rating": 4.5}]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Product CSV is missing columns"):
        ingestion.load_data(csv_path)


def test_create_documents_preserves_product_details(ingestion):
    documents = ingestion.create_documents(sample_dataframe())

    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)

    phone = documents[0]
    assert "Test Phone" in phone.page_content
    assert "USD 500.00" in phone.page_content
    assert phone.metadata["product_title"] == "Test Phone"
    assert phone.metadata["price"] == 500.0
    assert phone.metadata["currency"] == "USD"
    assert phone.metadata["rating"] == 4.5
    assert phone.metadata["source"] == "dummyjson"
    assert (
        phone.metadata["product_url"]
        == "https://dummyjson.com/products/1"
    )

    laptop = documents[1]
    assert laptop.metadata["currency"] == "INR"
    assert laptop.metadata["source"] == "flipkart"


def test_create_documents_skips_invalid_rows(ingestion):
    df = pd.DataFrame(
        [
            {
                "product_title": "Valid Product",
                "price": 100.0,
                "rating": 4.0,
                "product_url": "https://example.com/valid",
                "source": "flipkart",
            },
            {
                "product_title": "Missing URL",
                "price": 100.0,
                "rating": 4.0,
                "product_url": None,
                "source": "flipkart",
            },
            {
                "product_title": "Invalid Price",
                "price": "not-a-price",
                "rating": 4.0,
                "product_url": "https://example.com/invalid-price",
                "source": "flipkart",
            },
            {
                "product_title": "Zero Price",
                "price": 0,
                "rating": 4.0,
                "product_url": "https://example.com/zero-price",
                "source": "flipkart",
            },
            {
                "product_title": None,
                "price": 100.0,
                "rating": 4.0,
                "product_url": "https://example.com/no-title",
                "source": "flipkart",
            },
        ]
    )

    documents = ingestion.create_documents(df)

    assert len(documents) == 1
    assert documents[0].metadata["product_title"] == "Valid Product"


def test_create_documents_handles_missing_or_invalid_rating(ingestion):
    df = pd.DataFrame(
        [
            {
                "product_title": "No Rating",
                "price": 100.0,
                "rating": None,
                "product_url": "https://example.com/no-rating",
                "source": "flipkart",
            },
            {
                "product_title": "Invalid Rating",
                "price": 200.0,
                "rating": 8.0,
                "product_url": "https://example.com/invalid-rating",
                "source": "flipkart",
            },
        ]
    )

    documents = ingestion.create_documents(df)

    assert len(documents) == 2
    assert documents[0].metadata["rating"] == -1.0
    assert documents[1].metadata["rating"] == -1.0
    assert "Not rated" in documents[0].page_content
    assert "Not rated" in documents[1].page_content


def test_ingest_data_requires_astra_configuration(
    ingestion,
    monkeypatch,
):
    for name in (
        "ASTRA_DB_API_ENDPOINT",
        "ASTRA_DB_APPLICATION_TOKEN",
        "ASTRA_DB_KEYSPACE",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(
        RuntimeError,
        match="Set ASTRA_DB_API_ENDPOINT",
    ):
        ingestion.ingest_data()
