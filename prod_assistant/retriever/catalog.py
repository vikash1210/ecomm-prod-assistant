
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

    PHONE_QUERY_TERMS = {
        "phone",
        "phones",
        "mobile",
        "mobiles",
        "smartphone",
        "smartphones",
    }

    PHONE_BRAND_TERMS = {
        "iphone",
        "pixel",
        "samsung",
        "oneplus",
        "redmi",
        "realme",
        "motorola",
        "vivo",
        "oppo",
    }

    ACCESSORY_TERMS = {
        "charger",
        "chargers",
        "case",
        "cover",
        "covers",
        "airpods",
        "earphones",
        "earbuds",
        "headphones",
        "battery",
        "cable",
        "adapter",
        "screen",
        "protector",
        "powerbank",
        "power",
        "bank",
    }

    STOP_WORDS = {
        "a", "an", "and", "are", "best", "can", "do", "find",
        "for", "give", "good", "i", "in", "is", "me", "of",
        "on", "please", "recommend", "show", "suggest", "the",
        "to", "want", "what", "which", "with", "under", "below",
        "less", "than", "up", "upto", "within", "cheaper",
        "phone", "phones", "mobile", "mobiles", "smartphone",
        "smartphones",
    }

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
            raise FileNotFoundError(
                f"Product CSV not found: {path}"
            )

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

    def get_highest_rated_in_stock(self) -> list[Product]:
        """Return all in-stock products tied for the highest rating."""
        rated_products = [
            product
            for product in self._products
            if product.in_stock and product.rating is not None
        ]

        if not rated_products:
            return []

        highest_rating = max(
            product.rating for product in rated_products
        )

        return [
            product
            for product in rated_products
            if product.rating == highest_rating
        ]

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

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase words."""
        return re.findall(
            r"[^\W_]+",
            text.casefold(),
            flags=re.UNICODE,
        )

    def _is_phone_product(self, product: Product) -> bool:
        """Identify phones and exclude common phone accessories."""
        text = " ".join(
            [
                product.name or "",
                product.brand or "",
                product.description or "",
                product.category or "",
            ]
        )
        words = set(self._tokenize(text))

        if words.intersection(self.ACCESSORY_TERMS):
            return False

        # Recognize common phone brands and product naming conventions.
        return bool(
            words.intersection(self.PHONE_BRAND_TERMS)
        ) or bool(
            {"phone", "smartphone", "mobile"}.intersection(words)
        )

    def search_products(
        self,
        query: str = "",
        category: str | None = None,
    ) -> list[Product]:
        """Search by keywords, phone intent, and optional budget."""
        normalized_query = query.casefold().strip()

        normalized_category = (
            category.casefold().strip()
            if category and category.strip()
            else None
        )

        query_tokens = self._tokenize(normalized_query)

        # Support queries such as "under ₹20,000" and "below 70000".
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

        phone_intent = bool(
            self.PHONE_QUERY_TERMS.intersection(query_tokens)
        )

        query_words = {
            word
            for word in query_tokens
            if word not in self.STOP_WORDS and not word.isdigit()
        }

        # A generic phone query should match common phone brands.
        if phone_intent:
            query_words.update(self.PHONE_BRAND_TERMS)

        ranked: list[tuple[int, int, Product]] = []

        for product in self._products:
            if (
                normalized_category
                and (product.category or "").casefold()
                != normalized_category
            ):
                continue

            # Only compare INR products with an INR-style numeric budget.
            if max_price is not None:
                if (product.currency or "").upper() != "INR":
                    continue

                if product.price >= max_price:
                    continue

            # A generic phone query must not return accessories.
            if phone_intent and not self._is_phone_product(product):
                continue

            if not normalized_query:
                ranked.append((0, 0, product))
                continue

            if not query_words:
                continue

            extra_text = " ".join(
                self._extra_attributes.get(product.id, {}).values()
            )

            name_words = set(
                self._tokenize(product.name or "")
            )
            brand_words = set(
                self._tokenize(product.brand or "")
            )
            description_words = set(
                self._tokenize(product.description or "")
            )
            category_words = set(
                self._tokenize(product.category or "")
            )
            extra_words = set(self._tokenize(extra_text))

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

            if normalized_query in (product.name or "").casefold():
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

        return [
            product
            for _, _, product in ranked
        ]
