
"""Local and AstraDB-backed retrieval for the product assistant."""

import os

from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document

from prod_assistant.retriever.catalog import ProductCatalog
from prod_assistant.retriever.product import Product
from prod_assistant.utils.config_loader import load_config
from prod_assistant.utils.model_loader import ModelLoader


class Retriever:
    """Retrieve products locally or through AstraDB."""

    REQUIRED_ASTRA_VARS = (
        "ASTRA_DB_API_ENDPOINT",
        "ASTRA_DB_APPLICATION_TOKEN",
        "ASTRA_DB_KEYSPACE",
        "GOOGLE_API_KEY",
    )

    def __init__(
        self,
        catalog: ProductCatalog | None = None,
    ) -> None:
        load_dotenv()
        self.catalog = (
            catalog if catalog is not None else ProductCatalog.from_csv()
        )
        self.config = load_config()
        self.model_loader = ModelLoader()
        self.store = None
        self.retriever_instance = None

    @staticmethod
    def _product_to_document(
        product: Product,
        catalog: ProductCatalog | None = None,
    ) -> Document:
        """Convert a product and optional extra attributes into a document."""
        extra_attributes = (
            catalog.get_extra_attributes(product.id)
            if catalog is not None
            else {}
        )

        fields = {
            "Product": product.name,
            "Brand": product.brand or "",
            "Description": product.description or "",
            "Category": product.category,
            "Price": f"{product.currency} {product.price:.2f}",
            "Availability": (
                "In stock" if product.in_stock else "Out of stock"
            ),
            "Rating": (
                str(product.rating)
                if product.rating is not None
                else "Not rated"
            ),
            "Source": product.source or "unknown",
            "Product URL": product.product_url or "",
        }

        fields.update(
            {
                key.replace("_", " ").title(): value
                for key, value in extra_attributes.items()
                if value
            }
        )

        content = "\n".join(
            f"{key}: {value}"
            for key, value in fields.items()
            if value
        )

        metadata = {
            "product_id": product.id,
            "product_title": product.name,
            "brand": product.brand or "",
            "description": product.description or "",
            "category": product.category,
            "price": product.price,
            "currency": product.currency,
            "in_stock": product.in_stock,
            "rating": (
                product.rating
                if product.rating is not None
                else -1.0
            ),
            "source": product.source or "unknown",
            "product_url": product.product_url or "",
        }

        metadata.update(
            {
                f"extra_{key}": value
                for key, value in extra_attributes.items()
                if value
            }
        )

        return Document(
            page_content=content,
            metadata=metadata,
        )

    def _has_remote_configuration(self) -> bool:
        """Return whether all remote-service environment variables exist."""
        return all(
            os.getenv(name)
            for name in self.REQUIRED_ASTRA_VARS
        )

    def load_retriever(self):
        """Create and cache the AstraDB MMR and compression retriever."""
        if not self._has_remote_configuration():
            raise RuntimeError(
                "AstraDB retrieval requires these environment variables: "
                f"{', '.join(self.REQUIRED_ASTRA_VARS)}"
            )

        if self.retriever_instance is not None:
            return self.retriever_instance

        if self.store is None:
            self.store = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=self.config["astra_db"]["collection_name"],
                api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
                token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
                namespace=os.environ["ASTRA_DB_KEYSPACE"],
            )

        top_k = int(
            self.config.get("retriever", {}).get("top_k", 4)
        )

        mmr_retriever = self.store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            },
        )

        compressor = LLMChainFilter.from_llm(
            self.model_loader.load_llm()
        )

        self.retriever_instance = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=mmr_retriever,
        )

        return self.retriever_instance

    def invoke(
        self,
        query: str,
        *,
        limit: int = 4,
    ) -> list[Document]:
        """Search AstraDB when configured; otherwise search the local catalog."""
        if limit < 1:
            raise ValueError("limit must be at least 1")

        if self._has_remote_configuration():
            return self.load_retriever().invoke(query)[:limit]

        products = self.catalog.search_products(query=query)

        return [
            self._product_to_document(product, self.catalog)
            for product in products[:limit]
        ]

    def call_retriever(self, query: str) -> list[Document]:
        """Compatibility method for the existing RAG workflow."""
        return self.invoke(query)

    def format_results(
        self,
        query: str,
        *,
        limit: int = 4,
    ) -> str:
        """Format retrieved documents for a tool or language model."""
        documents = self.invoke(query, limit=limit)

        if not documents:
            return "No local results found."

        return "\n\n---\n\n".join(
            document.page_content
            for document in documents
        )
