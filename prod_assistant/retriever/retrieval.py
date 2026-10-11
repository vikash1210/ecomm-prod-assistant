
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
    """Retrieve product documents locally or through AstraDB."""

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
        self.vstore = None
        self.retriever_instance = None

    @staticmethod
    def _product_to_document(product: Product) -> Document:
        """Convert a catalog product into a LangChain document."""
        content = "\n".join(
            [
                f"Product: {product.name}",
                f"Description: {product.description or 'Not provided'}",
                f"Category: {product.category}",
                f"Price: {product.currency} {product.price:.2f}",
                (
                    "Rating: "
                    f"{product.rating if product.rating is not None else 'Not rated'}"
                ),
                f"Source: {product.source or 'unknown'}",
                f"Product URL: {product.product_url or 'Not provided'}",
            ]
        )

        return Document(
            page_content=content,
            metadata={
                "product_id": product.id,
                "product_title": product.name,
                "category": product.category,
                "price": product.price,
                "currency": product.currency,
                "rating": (
                    product.rating if product.rating is not None else -1.0
                ),
                "source": product.source or "unknown",
                "product_url": product.product_url or "",
            },
        )

    def _has_remote_configuration(self) -> bool:
        """Return whether all required remote-service variables are set."""
        return all(os.getenv(name) for name in self.REQUIRED_ASTRA_VARS)

    def load_retriever(self):
        """Create and cache the AstraDB MMR and compression retriever."""
        if not self._has_remote_configuration():
            raise RuntimeError(
                "AstraDB retrieval requires these environment variables: "
                f"{', '.join(self.REQUIRED_ASTRA_VARS)}"
            )

        if self.retriever_instance is not None:
            return self.retriever_instance

        if self.vstore is None:
            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=self.config["astra_db"]["collection_name"],
                api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
                token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
                namespace=os.environ["ASTRA_DB_KEYSPACE"],
            )

        top_k = int(self.config.get("retriever", {}).get("top_k", 4))

        mmr_retriever = self.vstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            },
        )

        llm = self.model_loader.load_llm()
        compressor = LLMChainFilter.from_llm(llm)

        self.retriever_instance = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=mmr_retriever,
        )

        return self.retriever_instance

    def invoke(self, query: str, *, limit: int = 4) -> list[Document]:
        """Search AstraDB when configured; otherwise search the local catalog."""
        if limit < 1:
            raise ValueError("limit must be at least 1")

        if self._has_remote_configuration():
            retriever = self.load_retriever()
            return retriever.invoke(query)[:limit]

        products = self.catalog.search_products(query=query)
        return [
            self._product_to_document(product)
            for product in products[:limit]
        ]

    def call_retriever(self, query: str) -> list[Document]:
        """Compatibility method for existing RAG workflow code."""
        return self.invoke(query)

    def format_results(self, query: str, *, limit: int = 4) -> str:
        """Format retrieved documents for an MCP tool or language model."""
        documents = self.invoke(query, limit=limit)

        if not documents:
            return "No local results found."

        return "\n\n---\n\n".join(
            document.page_content for document in documents
        )
