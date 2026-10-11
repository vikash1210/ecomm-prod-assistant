
"""In-memory product catalog and search operations."""

import hashlib
import re
from pathlib import Path

import pandas as pd

from prod_assistant.retriever.product import Product
from prod_assistant.retriever.product_schema import (
    normalize_dataframe,
    row_to_product_fields,
)


class ProductCatalog:
    """Store products and provide basic search functionality."""

    def __init__(
        self,
        products: list[Product] | None = None,
    ) -> None:
        self._products = products or []
        self._extra_attributes: dict[str, dict[str, str]] = {}

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path = "data/flipkart_products.csv",
    ) -> "ProductCatalog":
        """Load products from CSV using the shared schema normalizer."""
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Product CSV not found: {path}")

        df = normalize_dataframe(pd.read_csv(path))

        if "product_title" not in df.columns or "price" not in df.columns:
            raise ValueError(
                "Product CSV must include a product title and price. "
                "Accepted title aliases include title, product_name and name; "
                "accepted price aliases include cost and sale_price."
            )

        products: list[Product] = []
        extra_attributes: dict[str, dict[str, str]] = {}

        for _, row in df.iterrows():
            fields = row_to_product_fields(row.to_dict())
            title = fields["product_title"]
            price = fields["price"]

            if not title or price is None:
                continue

            product_url = fields["product_url"]
            identity = product_url or f"{title.casefold()}|{price}"

            product_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:16]

            product = Product(
                id=product_id,
                name=title,
                brand=fields["brand"],
                description=fields["description"],
                category=fields["category"],
                price=price,
                currency=fields["currency"],
                in_stock=fields["in_stock"],
                rating=fields["rating"],
                product_url=product_url,
                source=fields["source"],
            )

            products.append(product)
            extra_attributes[product_id] = fields["extra_attributes"]

        catalog = cls(products)
        catalog._extra_attributes = extra_attributes
        return catalog

    def list_products(self) -> list[Product]:
        """Return all products."""
        return list(self._products)

    def get_product(self, product_id: str) -> Product | None:
        """Find a product by its ID."""
        return next(
            (
                product
                for product in self._products
                if product.id == product_id
            ),
            None,
        )

    def get_extra_attributes(self, product_id: str) -> dict[str, str]:
        """Return extra CSV attributes for a product."""
        return dict(self._extra_attributes.get(product_id, {}))

    def search_products(
        self,
        query: str = "",
        category: str | None = None,
    ) -> list[Product]:
        """Search products using keyword ranking across standard and extra fields."""
        normalized_query = query.casefold().strip()
        normalized_category = (
            category.casefold().strip()
            if category and category.strip()
            else None
        )

        stop_words = {
            "a", "an", "and", "are", "best", "can", "do", "find",
            "for", "give", "good", "i", "in", "is", "me", "of",
            "on", "please", "recommend", "show", "suggest", "the",
            "to", "want", "what", "which", "with",
            "मुझे", "के", "का", "की", "है", "हैं", "और", "एक",
            "बताओ", "दिखाओ",
        }

        def tokenize(text: str) -> list[str]:
            return re.findall(
                r"[^\W_]+",
                text.casefold(),
                flags=re.UNICODE,
            )

        query_words = {
            word
            for word in tokenize(normalized_query)
            if word not in stop_words
        }

        ranked: list[tuple[int, int, Product]] = []

        for product in self._products:
            if (
                normalized_category
                and product.category.casefold() != normalized_category
            ):
                continue

            if not normalized_query:
                ranked.append((0, 0, product))
                continue

            if not query_words:
                continue

            extra_text = " ".join(
                self._extra_attributes.get(product.id, {}).values()
            )

            name_words = set(tokenize(product.name))
            brand_words = set(tokenize(product.brand or ""))
            description_words = set(tokenize(product.description))
            category_words = set(tokenize(product.category))
            extra_words = set(tokenize(extra_text))

            matched_name = query_words & name_words
            matched_brand = query_words & brand_words
            matched_description = query_words & description_words
            matched_category = query_words & category_words
            matched_extra = query_words & extra_words

            score = (
                3 * len(matched_name)
                + 3 * len(matched_brand)
                + 2 * len(matched_description)
                + len(matched_category)
                + len(matched_extra)
            )

            if normalized_query in product.name.casefold():
                score += 2

            if score > 0:
                ranked.append(
                    (
                        score,
                        len(matched_name) + len(matched_brand),
                        product,
                    )
                )

        ranked.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        return [product for _, _, product in ranked]
