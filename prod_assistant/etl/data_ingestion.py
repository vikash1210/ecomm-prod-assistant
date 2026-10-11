
"""Ingest normalized product records from CSV into AstraDB."""

import os
from pathlib import Path
from typing import ClassVar

import pandas as pd
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document

from prod_assistant.retriever.product_schema import (
    normalize_dataframe,
    row_to_product_fields,
)
from prod_assistant.utils.config_loader import load_config
from prod_assistant.utils.model_loader import ModelLoader


class DataIngestion:
    """Load product records and ingest them into AstraDB."""

    REQUIRED_COLUMN_ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "product_title": (
            "product_title",
            "title",
            "product_name",
            "name",
            "item_name",
            "product",
        ),
        "price": (
            "price",
            "cost",
            "sale_price",
            "selling_price",
            "current_price",
            "amount",
            "product_price",
        ),
    }

    def __init__(self) -> None:
        self.config = load_config()
        self.model_loader = ModelLoader()

    def load_data(
        self,
        csv_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Load and normalize the product CSV."""
        if csv_path is None:
            csv_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "flipkart_products.csv"
            )

        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Product CSV not found: {path}")

        df = normalize_dataframe(pd.read_csv(path))

        missing = {
            field
            for field in self.REQUIRED_COLUMN_ALIASES
            if field not in df.columns
        }
        if missing:
            raise ValueError(
                "Product CSV is missing columns: "
                f"{sorted(missing)}. Supported title aliases include "
                "title, product_name and name; supported price aliases "
                "include cost and sale_price."
            )

        return df

    def create_documents(self, df: pd.DataFrame) -> list[Document]:
        """Convert valid product rows into searchable documents."""
        normalized_df = normalize_dataframe(df)
        documents: list[Document] = []

        for _, row in normalized_df.iterrows():
            fields = row_to_product_fields(row.to_dict())
            title = fields["product_title"]
            price = fields["price"]
            url = fields["product_url"]

            # Preserve existing behavior: title, positive price and URL
            # are required before a product becomes a searchable document.
            if not title or price is None or not url:
                continue

            extra_attributes = fields["extra_attributes"]

            content_fields = {
                "Product": title,
                "Brand": fields["brand"] or "",
                "Category": fields["category"],
                "Description": fields["description"],
                "Price": f"{fields['currency']} {price:.2f}",
                "Rating": (
                    str(fields["rating"])
                    if fields["rating"] is not None
                    else "Not rated"
                ),
                "Availability": (
                    "In stock" if fields["in_stock"] else "Out of stock"
                ),
                "Source": fields["source"],
                "Product URL": url,
            }

            content_lines = [
                f"{key}: {value}"
                for key, value in content_fields.items()
                if value
            ]
            content_lines.extend(
                f"{key.replace('_', ' ').title()}: {value}"
                for key, value in extra_attributes.items()
                if value
            )

            metadata = {
                "product_title": title,
                "price": float(price),
                "currency": fields["currency"],
                "rating": (
                    float(fields["rating"])
                    if fields["rating"] is not None
                    else -1.0
                ),
                "source": fields["source"],
                "product_url": url,
                "brand": fields["brand"] or "",
                "category": fields["category"],
                "description": fields["description"],
                "in_stock": fields["in_stock"],
            }
            metadata.update(
                {
                    f"extra_{key}": value
                    for key, value in extra_attributes.items()
                    if value
                }
            )

            documents.append(
                Document(
                    page_content="\n".join(content_lines),
                    metadata=metadata,
                )
            )

        return documents

    def ingest_data(
        self,
        csv_path: str | Path | None = None,
    ) -> int:
        """Embed product documents and add them to AstraDB."""
        endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        keyspace = os.getenv("ASTRA_DB_KEYSPACE")

        if not endpoint or not token or not keyspace:
            raise RuntimeError(
                "Set ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, "
                "and ASTRA_DB_KEYSPACE before ingesting data."
            )

        df = self.load_data(csv_path)
        documents = self.create_documents(df)

        if not documents:
            raise ValueError("No valid products found in the CSV.")

        embeddings = self.model_loader.load_embeddings()
        vector_store = AstraDBVectorStore(
            collection_name=self.config["astra_db"]["collection_name"],
            embedding=embeddings,
            api_endpoint=endpoint,
            token=token,
            namespace=keyspace,
        )

        vector_store.add_documents(documents)
        return len(documents)
