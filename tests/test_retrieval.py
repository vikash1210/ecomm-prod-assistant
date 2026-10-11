
"""Unit tests for local and AstraDB-backed product retrieval."""
import pytest
from langchain_core.documents import Document

from prod_assistant.retriever.catalog import ProductCatalog
from prod_assistant.retriever.product import Product
from prod_assistant.retriever.retrieval import Retriever


def make_catalog() -> ProductCatalog:
    """Build a small in-memory catalog for testing."""
    return ProductCatalog(
        [
            Product(
                id="P001",
                name="Wireless Mouse",
                description="Ergonomic wireless mouse",
                category="Electronics",
                price=799.0,
                currency="INR",
                rating=4.5,
                product_url="https://example.com/mouse",
                source="test",
            ),
            Product(
                id="P002",
                name="Mechanical Keyboard",
                description="RGB gaming keyboard",
                category="Electronics",
                price=2499.0,
                currency="INR",
                rating=4.2,
                product_url="https://example.com/keyboard",
                source="test",
            ),
            Product(
                id="P003",
                name="Cotton T-Shirt",
                description="Comfortable cotton shirt",
                category="Clothing",
                price=499.0,
                currency="INR",
                rating=None,
                product_url="https://example.com/shirt",
                source="test",
            ),
        ]
    )


@pytest.fixture
def clear_astra_environment(monkeypatch):
    """Ensure retrieval tests do not use real AstraDB credentials."""
    for name in Retriever.REQUIRED_ASTRA_VARS:
        monkeypatch.delenv(name, raising=False)


def test_invoke_uses_local_catalog_without_remote_configuration(
    clear_astra_environment,
):
    retriever = Retriever(catalog=make_catalog())

    results = retriever.invoke("mouse")

    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert "Wireless Mouse" in results[0].page_content
    assert results[0].metadata["product_title"] == "Wireless Mouse"
    assert results[0].metadata["price"] == 799.0
    assert results[0].metadata["currency"] == "INR"
    assert results[0].metadata["product_url"] == "https://example.com/mouse"


def test_invoke_respects_result_limit(clear_astra_environment):
    retriever = Retriever(catalog=make_catalog())

    results = retriever.invoke("", limit=2)

    assert len(results) == 2


def test_invoke_rejects_zero_limit(clear_astra_environment):
    retriever = Retriever(catalog=make_catalog())

    with pytest.raises(ValueError, match="limit must be at least 1"):
        retriever.invoke("mouse", limit=0)


def test_invoke_rejects_negative_limit(clear_astra_environment):
    retriever = Retriever(catalog=make_catalog())

    with pytest.raises(ValueError, match="limit must be at least 1"):
        retriever.invoke("mouse", limit=-1)


def test_product_to_document_preserves_product_metadata():
    product = make_catalog().get_product("P003")
    assert product is not None

    document = Retriever._product_to_document(product)

    assert "Cotton T-Shirt" in document.page_content
    assert "Not rated" in document.page_content
    assert document.metadata["product_title"] == "Cotton T-Shirt"
    assert document.metadata["rating"] == -1.0
    assert document.metadata["product_url"] == "https://example.com/shirt"


def test_format_results_returns_local_results(clear_astra_environment):
    retriever = Retriever(catalog=make_catalog())

    result = retriever.format_results("keyboard")

    assert "Mechanical Keyboard" in result
    assert "2499.00" in result
    assert "https://example.com/keyboard" in result


def test_format_results_handles_no_matches(clear_astra_environment):
    retriever = Retriever(catalog=make_catalog())

    result = retriever.format_results("product-that-does-not-exist")

    assert result == "No local results found."


def test_invoke_uses_configured_remote_retriever(
    monkeypatch,
    clear_astra_environment,
):
    """Verify remote retrieval is selected without making network calls."""
    for name in Retriever.REQUIRED_ASTRA_VARS:
        monkeypatch.setenv(name, "test-value")

    expected_documents = [
        Document(
            page_content="Product: Remote Phone",
            metadata={"product_title": "Remote Phone"},
        )
    ]

    class FakeRemoteRetriever:
        def invoke(self, query):
            assert query == "remote phone"
            return expected_documents

    retriever = Retriever(catalog=make_catalog())
    monkeypatch.setattr(
        retriever,
        "load_retriever",
        lambda: FakeRemoteRetriever(),
    )

    results = retriever.invoke("remote phone", limit=1)

    assert results == expected_documents


def test_remote_retrieval_respects_limit(
    monkeypatch,
    clear_astra_environment,
):
    """Limit remote results without contacting AstraDB."""
    for name in Retriever.REQUIRED_ASTRA_VARS:
        monkeypatch.setenv(name, "test-value")

    documents = [
        Document(page_content=f"Product {index}", metadata={})
        for index in range(3)
    ]

    class FakeRemoteRetriever:
        def invoke(self, query):
            return documents

    retriever = Retriever(catalog=make_catalog())
    monkeypatch.setattr(
        retriever,
        "load_retriever",
        lambda: FakeRemoteRetriever(),
    )

    results = retriever.invoke("phone", limit=2)

    assert len(results) == 2


def test_load_retriever_reports_missing_configuration(
    clear_astra_environment,
):
    retriever = Retriever(catalog=make_catalog())

    with pytest.raises(RuntimeError, match="AstraDB retrieval requires"):
        retriever.load_retriever()
