
"""Load the local product catalog into Astra DB."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore

from prod_assistant.utils.config_loader import load_config
from prod_assistant.retriever.catalog import ProductCatalog
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "flipkart_products.csv"

REQUIRED_ENV_VARS = (
    "ASTRA_DB_API_ENDPOINT",
    "ASTRA_DB_APPLICATION_TOKEN",
    "ASTRA_DB_KEYSPACE",
    "GOOGLE_API_KEY",
)


def main() -> None:
    """Upsert CSV products into the configured Astra DB collection."""
    load_dotenv(PROJECT_ROOT / ".env")

    missing = [
        name for name in REQUIRED_ENV_VARS
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    config = load_config()
    catalog = ProductCatalog.from_csv(CSV_PATH)
    products = catalog.list_products()

    if not products:
        raise RuntimeError(f"No products found in {CSV_PATH}")

    retriever = Retriever(catalog=catalog)
    documents = [
        retriever._product_to_document(product, catalog)
        for product in products
    ]
    document_ids = [product.id for product in products]

    model_loader = ModelLoader()
    vector_store = AstraDBVectorStore(
        embedding=model_loader.load_embeddings(),
        collection_name=config["astra_db"]["collection_name"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        namespace=os.environ["ASTRA_DB_KEYSPACE"],
    )

    vector_store.add_documents(
        documents=documents,
        ids=document_ids,
    )

    print(f"CSV products loaded: {len(products)}")
    print(f"Astra DB collection: {config['astra_db']['collection_name']}")
    print("Product ingestion completed.")


if __name__ == "__main__":
    main()
