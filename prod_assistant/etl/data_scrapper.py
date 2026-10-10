
import csv
import json
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote_plus, urljoin

import httpx
from bs4 import BeautifulSoup


class FlipkartScraper:
    """Extract product data from Flipkart or sample data from DummyJSON."""

    BASE_URL = "https://www.flipkart.com"
    SEARCH_URL = f"{BASE_URL}/search?q="
    DUMMYJSON_URL = "https://dummyjson.com/products/search"

    FIELDNAMES: ClassVar[list[str]] = [
        "product_title",
        "price",
        "rating",
        "product_url",
        "source",
    ]

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self.driver = None

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-IN,en;q=0.9",
        }

    def _extract_product_data(
        self,
        html: str,
        product_url: str,
    ) -> dict[str, str | None]:
        """Extract product details from JSON-LD or page elements."""
        soup = BeautifulSoup(html, "html.parser")

        result: dict[str, str | None] = {
            "product_title": None,
            "price": None,
            "rating": None,
            "product_url": product_url,
            "source": "flipkart",
        }

        for script in soup.select('script[type="application/ld+json"]'):
            try:
                data = json.loads(
                    script.string or script.get_text()
                )
            except (json.JSONDecodeError, TypeError):
                continue

            pending = data if isinstance(data, list) else [data]

            while pending:
                obj = pending.pop(0)

                if not isinstance(obj, dict):
                    continue

                graph = obj.get("@graph", [])
                if isinstance(graph, list):
                    pending.extend(graph)

                obj_type = obj.get("@type", [])
                if isinstance(obj_type, str):
                    obj_type = [obj_type]

                if "Product" not in obj_type:
                    continue

                result["product_title"] = (
                    result["product_title"] or obj.get("name")
                )

                offers = obj.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}

                if isinstance(offers, dict):
                    price = offers.get("price")
                    if price is None:
                        price = offers.get("lowPrice")

                    if price is not None and result["price"] is None:
                        result["price"] = str(price)

                aggregate = obj.get("aggregateRating", {})
                if isinstance(aggregate, dict):
                    rating = aggregate.get("ratingValue")
                    if rating is not None and result["rating"] is None:
                        result["rating"] = str(rating)

        def element_value(element) -> str | None:
            if element is None:
                return None

            value = element.get("content")
            if value is not None:
                return str(value).strip() or None

            return element.get_text(" ", strip=True) or None

        title_element = (
            soup.select_one("span.VU-ZEz")
            or soup.select_one("span.B_NuCI")
            or soup.select_one("h1")
            or soup.select_one('meta[property="og:title"]')
        )

        price_element = (
            soup.select_one("div.Nx9bqj")
            or soup.select_one("div._30jeq3")
            or soup.select_one('[itemprop="price"]')
            or soup.select_one('meta[property="product:price:amount"]')
        )

        rating_element = (
            soup.select_one("div.XQDdHH")
            or soup.select_one("div._3LWZlK")
            or soup.select_one('[itemprop="ratingValue"]')
        )

        result["product_title"] = (
            result["product_title"] or element_value(title_element)
        )
        result["price"] = (
            result["price"] or element_value(price_element)
        )
        result["rating"] = (
            result["rating"] or element_value(rating_element)
        )

        return result

    def parse_search_results(
        self,
        html: str,
        max_products: int = 10,
    ) -> list[dict[str, str | None]]:
        """Parse product cards from Flipkart search-result HTML."""
        if max_products < 1:
            raise ValueError("max_products must be at least 1")

        soup = BeautifulSoup(html, "html.parser")
        products = []
        seen_urls = set()

        for link in soup.select('a[href*="/p/"]'):
            href = link.get("href")

            if not isinstance(href, str):
                continue

            url = urljoin(
                self.BASE_URL,
                href.split("?")[0],
            )

            if url in seen_urls:
                continue

            card = link.find_parent("div", attrs={"data-id": True})
            if card is None:
                card = link

            title_element = (
                card.select_one("div.KzDlHZ")
                or card.select_one("a[title]")
                or card.select_one("div[title]")
            )

            title = None
            if title_element:
                title = (
                    title_element.get("title")
                    or title_element.get_text(" ", strip=True)
                )

            if not title:
                title = link.get("title") or link.get_text(
                    " ",
                    strip=True,
                )

            if not title:
                continue

            price_element = (
                card.select_one("div.Nx9bqj")
                or card.select_one("div._30jeq3")
            )

            rating_element = (
                card.select_one("div.XQDdHH")
                or card.select_one("div._3LWZlK")
            )

            products.append({
                "product_title": title,
                "price": (
                    price_element.get_text(" ", strip=True)
                    if price_element
                    else None
                ),
                "rating": (
                    rating_element.get_text(" ", strip=True)
                    if rating_element
                    else None
                ),
                "product_url": url,
                "source": "flipkart",
            })

            seen_urls.add(url)

            if len(products) >= max_products:
                break

        return products

    def search_products(
        self,
        query: str,
        max_products: int = 10,
    ) -> list[dict[str, str | None]]:
        """Search Flipkart and return parsed products."""
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        if max_products < 1:
            raise ValueError("max_products must be at least 1")

        with httpx.Client(
            headers=self.headers,
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:
            response = client.get(
                self.SEARCH_URL + quote_plus(query)
            )
            response.raise_for_status()

        html = response.text
        lowered_html = html.lower()

        if "captcha" in lowered_html or "robot check" in lowered_html:
            raise RuntimeError(
                "Flipkart returned a CAPTCHA or access restriction."
            )

        products = self.parse_search_results(html, max_products)

        if not products:
            raise RuntimeError(
                "No products found. The page structure may have changed "
                "or access may be restricted."
            )

        return products

    def search_multiple_queries(
        self,
        queries: list[str],
        max_products_per_query: int = 5,
        output_path: str = "data/flipkart_products.csv",
    ) -> list[dict[str, str | None]]:
        """Search multiple Flipkart queries and save unique products."""
        if not queries:
            raise ValueError("At least one search query is required")

        if max_products_per_query < 1:
            raise ValueError("max_products_per_query must be at least 1")

        all_products = []
        seen_urls = set()

        for query in queries:
            print(f"\nSearching: {query}")

            products = self.search_products(
                query,
                max_products=max_products_per_query,
            )

            for product in products:
                url = product.get("product_url")

                if url and url not in seen_urls:
                    all_products.append(product)
                    seen_urls.add(url)

        self.save_to_csv(all_products, output_path)
        return all_products

    def fetch_dummyjson_products(
        self,
        query: str = "phone",
        limit: int = 10,
        output_path: str = "data/flipkart_products.csv",
    ) -> list[dict[str, str | None]]:
        """Fetch sample products from DummyJSON and save them to CSV."""
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        if limit < 1:
            raise ValueError("limit must be at least 1")

        response = httpx.get(
            self.DUMMYJSON_URL,
            params={"q": query, "limit": limit},
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        products = []

        for item in data.get("products", []):
            product_id = item.get("id")
            title = item.get("title")

            if product_id is None or not title:
                continue

            products.append({
                "product_title": str(title),
                "price": (
                    str(item["price"])
                    if item.get("price") is not None
                    else None
                ),
                "rating": (
                    str(item["rating"])
                    if item.get("rating") is not None
                    else None
                ),
                "product_url": (
                    f"https://dummyjson.com/products/{product_id}"
                ),
                "source": "dummyjson",
            })

        self.save_to_csv(products, output_path)
        return products

    def open_product_in_browser(
        self,
        product_url: str,
        output_path: str = "data/flipkart_products.csv",
    ) -> dict[str, str | None]:
        """Open a Flipkart product in Chrome, extract it, and save it."""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        if not product_url.startswith("https://www.flipkart.com/"):
            raise ValueError("Please provide a valid Flipkart product URL.")

        options = Options()
        options.add_argument("--start-maximized")
        self.driver = webdriver.Chrome(options=options)

        try:
            self.driver.get(product_url)

            print("\nFlipkart page opened in Chrome.")
            input(
                "When the actual product page is visible, "
                "return here and press Enter..."
            )

            current_url = self.driver.current_url
            html = self.driver.page_source
            result = self._extract_product_data(html, current_url)

            print("\nExtracted product data:")
            for key, value in result.items():
                print(f"{key}: {value}")

            if not result["product_title"]:
                raise RuntimeError(
                    "Product title could not be extracted. "
                    "Check that Chrome is showing the actual product page."
                )

            self.save_to_csv([result], output_path)
            print(f"\nCSV saved to: {output_path}")

            if not result["price"] or not result["rating"]:
                print("Warning: price or rating is missing from the page.")

            return result

        finally:
            # Keep Chrome open for manual inspection.
            pass

    def save_to_csv(
        self,
        products: list[dict[str, str | None]],
        output_path: str = "data/flipkart_products.csv",
    ) -> None:
        """Preserve existing rows and append products with unique URLs."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing_rows = []
        existing_urls = set()

        if path.exists() and path.stat().st_size > 0:
            with path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as csv_file:
                reader = csv.DictReader(csv_file)
                existing_rows = list(reader)

            for row in existing_rows:
                url = row.get("product_url")

                if url:
                    existing_urls.add(url)

                if not row.get("source"):
                    if url and "dummyjson.com" in url:
                        row["source"] = "dummyjson"
                    elif url and "flipkart.com" in url:
                        row["source"] = "flipkart"
                    else:
                        row["source"] = "unknown"

        new_products = []

        for product in products:
            url = product.get("product_url")

            if not url or url in existing_urls:
                continue

            new_product = {
                field: product.get(field)
                for field in self.FIELDNAMES
            }

            if not new_product.get("source"):
                if "dummyjson.com" in url:
                    new_product["source"] = "dummyjson"
                elif "flipkart.com" in url:
                    new_product["source"] = "flipkart"
                else:
                    new_product["source"] = "unknown"

            new_products.append(new_product)
            existing_urls.add(url)

        all_rows = existing_rows + new_products

        with path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=self.FIELDNAMES,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"Saved {len(new_products)} new product(s) to {path}.")
