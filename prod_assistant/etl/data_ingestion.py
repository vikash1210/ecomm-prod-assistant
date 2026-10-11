
"""Ingest products from the local CSV catalog into AstraDB."""

import os
from pathlib import Path
from typing import ClassVar

import pandas as pd
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document

from prod_assistant.utils.config_loader import load_config
from prod_assistant.utils.model_loader import ModelLoader


class DataIngestion:
    """Load local product data and ingest it into AstraDB."""

    REQUIRED_COLUMNS: ClassVar[set[str]] = {
        "product_title",
        "price",
        "product_url",
    }

    def __init__(self) -> None:
        self.config = load_config()
        self.model_loader = ModelLoader()

    def load_data(
        self,
        csv_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Load and validate the existing product CSV."""
        if csv_path is None:
            csv_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "flipkart_products.csv"
            )

        path = Path(csv_path)

        if not path.is_file():
            raise FileNotFoundError(f"Product CSV not found: {path}")

        df = pd.read_csv(path)
        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                "Product CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )

        return df

    def create_documents(self, df: pd.DataFrame) -> list[Document]:
        """Convert product rows into documents with searchable metadata."""
        documents: list[Document] = []

        for _, row in df.iterrows():
            title = str(row["product_title"]).strip()
            product_url = str(row["product_url"]).strip()
            price = pd.to_numeric(row["price"], errors="coerce")

            if (
                not title
                or title.casefold() == "nan"
                or not product_url
                or product_url.casefold() == "nan"
                or pd.isna(price)
                or price <= 0
            ):
                continue

            rating_value = pd.to_numeric(
                row.get("rating"),
                errors="coerce",
            )
            rating = (
                float(rating_value)
                if pd.notna(rating_value)
                and 0 <= rating_value <= 5
                else None
            )

            source_value = row.get("source")
            source = (
                str(source_value).strip()
                if pd.notna(source_value) and str(source_value).strip()
                else "unknown"
            )

            currency = "USD" if source == "dummyjson" else "INR"

            content = "\n".join(
                [
                    f"Product: {title}",
                    f"Price: {currency} {float(price):.2f}",
                    f"Rating: {rating if rating is not None else 'Not rated'}",
                    f"Source: {source}",
                    f"Product URL: {product_url}",
                ]
            )

            metadata = {
                "product_title": title,
                "price": float(price),
                "currency": currency,
                "rating": rating if rating is not None else -1.0,
                "source": source,
                "product_url": product_url,
            }

            documents.append(
                Document(page_content=content, metadata=metadata)
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