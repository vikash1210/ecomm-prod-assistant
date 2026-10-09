"""In-memory product catalog and search operations."""

from prod_assistant.retriever.product import Product


class ProductCatalog:
    """Store products and provide basic search functionality."""

    def __init__(self, products: list[Product] | None = None) -> None:
        self._products = products or []

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
        normalized_category = category.strip().casefold() if category else None

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
