
"""Local and AstraDB-backed retrieval for the product assistant."""

import os
import re

from dotenv import load_dotenv
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
            catalog
            if catalog is not None
            else ProductCatalog.from_csv()
        )
        self.config = load_config()
        self.model_loader = ModelLoader()

        # Cache the vector store and retriever for reuse.
        self.store: AstraDBVectorStore | None = None
        self.retriever_instance = None

    @staticmethod
    def _product_to_document(
        product: Product,
        catalog: ProductCatalog | None = None,
    ) -> Document:
        """Convert a product and its extra attributes into a document."""

        extra_attributes = (
            catalog.get_extra_attributes(product.id)
            if catalog is not None
            else {}
        )

        fields = {
            "Product": product.name,
            "Brand": product.brand or "",
            "Description": product.description or "",
            "Category": product.category or "",
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
                if value is not None and value != ""
            }
        )

        content = "\n".join(
            f"{key}: {value}"
            for key, value in fields.items()
            if value is not None and value != ""
        )

        metadata = {
            "product_id": product.id,
            "product_title": product.name,
            "brand": product.brand or "",
            "description": product.description or "",
            "category": product.category or "",
            "price": product.price,
            "currency": product.currency or "",
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
                if value is not None and value != ""
            }
        )

        return Document(
            page_content=content,
            metadata=metadata,
        )

    @staticmethod
    def _is_highest_rated_query(query: str) -> bool:
        """Recognize queries asking for the highest-rated in-stock product."""

        normalized_query = query.casefold()
        rating_terms = (
            "highest rated",
            "highest-rated",
            "top rated",
            "top-rated",
            "best rated",
            "best-rated",
            "highest rating",
            "maximum rating",
            "max rating",
        )
        availability_terms = (
            "in stock",
            "available",
            "availability",
        )

        asks_for_rating = any(
            term in normalized_query for term in rating_terms
        )
        asks_for_stock = any(
            term in normalized_query for term in availability_terms
        )

        return asks_for_rating and asks_for_stock

    def _get_highest_rated_documents(self) -> list[Document]:
        """Rank the full local catalog instead of vector-search results."""

        products = self.catalog.get_highest_rated_in_stock()

        return [
            self._product_to_document(product, self.catalog)
            for product in products
        ]

    def _has_remote_configuration(self) -> bool:
        """Check whether all required AstraDB environment variables exist."""

        return all(
            os.getenv(name)
            for name in self.REQUIRED_ASTRA_VARS
        )

    def load_retriever(self):
        """Create and cache the AstraDB similarity retriever."""

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

        if top_k < 1:
            raise ValueError("retriever.top_k must be at least 1")

        self.retriever_instance = self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        return self.retriever_instance

    
    def invoke(
        self,
        query: str,
        *,
        limit: int = 4,
    ) -> list[Document]:
        """Retrieve products and apply budget, stock, and phone filters."""

        if limit < 1:
            raise ValueError("limit must be at least 1")

        normalized_query = query.casefold()

        # Catalog-wide ranking must not depend on vector similarity.
        if self._is_highest_rated_query(query):
            return self._get_highest_rated_documents()[:limit]

        # Retrieve candidate documents using the existing backend.
        if self._has_remote_configuration():
            retriever = self.load_retriever()
            documents = retriever.invoke(query)
        else:
            products = self.catalog.search_products(query=query)
            documents = [
                self._product_to_document(product, self.catalog)
                for product in products
            ]

        # Detect an explicit price budget, e.g. "under 70000".
        budget_match = re.search(
            r"\b(?:under|below|less than|up to|upto|within|"
            r"cheaper than)\s*₹?\s*([\d,]+)",
            normalized_query,
        )
        max_price = (
            float(budget_match.group(1).replace(",", ""))
            if budget_match
            else None
        )

        phone_terms = {
            "phone", "phones", "mobile", "mobiles",
            "smartphone", "smartphones", "iphone",
        }
        query_words = set(re.findall(r"[^\W_]+", normalized_query))
        phone_intent = bool(query_words & phone_terms)

        accessory_terms = {
            "charger", "case", "cover", "airpods", "earphones",
            "earbuds", "headphones", "battery", "cable",
            "adapter", "protector", "powerbank",
        }

        filtered_documents = []

        for document in documents:
            metadata = document.metadata
            title = str(
                metadata.get("product_title")
                or document.page_content
            ).casefold()

            # Apply an explicit budget only to INR-priced products.
            if max_price is not None:
                currency = str(metadata.get("currency", "")).upper()
                price = metadata.get("price")

                if currency != "INR" or price is None:
                    continue

                try:
                    if float(price) >= max_price:
                        continue
                except (TypeError, ValueError):
                    continue

            # For phone requests, exclude common accessories and
            # require a recognizable phone product name.
            if phone_intent:
                if any(term in title for term in accessory_terms):
                    continue

                phone_name_terms = (
                    "iphone", "pixel", "samsung", "oneplus",
                    "redmi", "realme", "motorola", "vivo",
                    "oppo", "phone", "smartphone", "mobile",
                )
                if not any(term in title for term in phone_name_terms):
                    continue

            filtered_documents.append(document)

        return filtered_documents[:limit]


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
            return "No results found."

        return "\n\n---\n\n".join(
            document.page_content
            for document in documents
        )
