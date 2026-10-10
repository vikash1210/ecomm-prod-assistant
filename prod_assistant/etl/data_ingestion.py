
import os
from pathlib import Path
from typing import ClassVar

import pandas as pd
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document

from prod_assistant.utils.config_loader import load_config
from prod_assistant.utils.model_loader import ModelLoader


class DataIngestion:
    """Load product reviews from CSV and ingest them into AstraDB."""

    REQUIRED_COLUMNS: ClassVar[set[str]] =  {
        "product_id",
        "product_title",
        "rating",
        "total_reviews",
        "price",
        "top_reviews",
    }

    def __init__(self) -> None:
        self.config = load_config()
        self.model_loader = ModelLoader()

    def load_data(self, csv_path: str | Path | None = None) -> pd.DataFrame:
        if csv_path is None:
            csv_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "product_reviews.csv"
            )

        csv_path = Path(csv_path)

        if not csv_path.is_file():
            raise FileNotFoundError(f"Product data CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Product data CSV is missing columns: {sorted(missing_columns)}"
            )

        return df

    def create_documents(self, df: pd.DataFrame) -> list[Document]:
        documents = []

        for _, row in df.iterrows():
            review_text = str(row["top_reviews"])

            metadata = {
                "product_id": str(row["product_id"]),
                "product_title": str(row["product_title"]),
                "rating": str(row["rating"]),
                "total_reviews": str(row["total_reviews"]),
                "price": str(row["price"]),
            }

            documents.append(
                Document(page_content=review_text, metadata=metadata)
            )

        return documents

    def ingest_data(self, csv_path: str | Path | None = None):
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
