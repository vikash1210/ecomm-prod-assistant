
"""In-memory product catalog and search operations."""

import hashlib
from pathlib import Path

import pandas as pd

from prod_assistant.retriever.product import Product


class ProductCatalog:
    """Store products and provide basic search functionality."""

    def __init__(self, products: list[Product] | None = None) -> None:
        self._products = products or []

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path = "data/flipkart_products.csv",
    ) -> "ProductCatalog":
        """Load products from a CSV file into an in-memory catalog."""
        path = Path(csv_path)

        if not path.is_file():
            raise FileNotFoundError(f"Product CSV not found: {path}")

        df = pd.read_csv(path)

        required_columns = {
            "product_title",
            "price",
            "product_url",
        }
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Product CSV is missing columns: {sorted(missing_columns)}"
            )

        products = []

        for _, row in df.iterrows():
            title = row.get("product_title")
            url = row.get("product_url")
            price_value = pd.to_numeric(row.get("price"), errors="coerce")

            if pd.isna(title) or not str(title).strip():
                continue

            if pd.isna(url) or not str(url).strip():
                continue

            if pd.isna(price_value) or price_value <= 0:
                continue

            product_url = str(url).strip()
            product_id = hashlib.sha256(
                product_url.encode("utf-8")
            ).hexdigest()[:16]

            rating_value = pd.to_numeric(
                row.get("rating"),
                errors="coerce",
            )
            rating = (
                float(rating_value)
                if pd.notna(rating_value) and 0 <= rating_value <= 5
                else None
            )

            source_value = row.get("source")
            if pd.notna(source_value) and str(source_value).strip():
                source = str(source_value).strip()
            elif "dummyjson.com" in product_url:
                source = "dummyjson"
            elif "flipkart.com" in product_url:
                source = "flipkart"
            else:
                source = "unknown"

            currency = (
                "USD"
                if source == "dummyjson"
                else "INR"
            )

            products.append(
                Product(
                    id=product_id,
                    name=str(title).strip(),
                    description="",
                    category="Uncategorized",
                    price=float(price_value),
                    currency=currency,
                    rating=rating,
                    product_url=product_url,
                    source=source,
                )
            )

        return cls(products)

    def list_products(self) -> list[Product]:
        """Return all products."""
        return list(self._products)

    def get_product(self, product_id: str) -> Product | None:
        """Find a product by its ID."""
        for product in self._products:
            if product.id == product_id:
                return product
        return None

    def search_products(
        self,
        query: str = "",
        category: str | None = None,
    ) -> list[Product]:
        """Search product names and descriptions, optionally by category."""
        normalized_query = query.strip().casefold()
        normalized_category = (
            category.strip().casefold() if category else None
        )

        results = []

        for product in self._products:
            matches_query = (
                not normalized_query
                or normalized_query in product.name.casefold()
                or normalized_query in product.description.casefold()
            )

            matches_category = (
                not normalized_category
                or product.category.casefold() == normalized_category
            )

            if matches_query and matches_category:
                results.append(product)

        return results
