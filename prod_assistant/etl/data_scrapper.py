
"""Scrape product data from Flipkart and DummyJSON into a CSV catalog."""

import csv
import json
import re
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote_plus, urljoin

import httpx
from bs4 import BeautifulSoup


class FlipkartScraper:
    """Extract product data from Flipkart or DummyJSON."""

    BASE_URL = "https://www.flipkart.com"
    SEARCH_URL = f"{BASE_URL}/search?q="
    DUMMYJSON_URL = "https://dummyjson.com/products/search"

    FIELDNAMES: ClassVar[list[str]] = [
        "product_title",
        "price",
        "rating",
        "product_url",
        "source",
        "brand",
        "category",
        "description",
        "currency",
        "in_stock",
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

    @staticmethod
    def _clean_text(value: object) -> str | None:
        """Normalize optional text values."""
        if value is None:
            return None

        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none", "null"}:
            return None

        return text

    @staticmethod
    def _number(value: object) -> str | None:
        """Extract a numeric price or rating from displayed text."""
        if value is None:
            return None

        text = str(value).strip().replace(",", "")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return match.group(0) if match else None

    @staticmethod
    def _currency_from_value(value: object, source: str) -> str:
        """Determine currency from a displayed price or known source."""
        text = str(value or "")

        if "₹" in text or "Rs." in text or "INR" in text.upper():
            return "INR"
        if "$" in text or "USD" in text.upper():
            return "USD"
        if "€" in text:
            return "EUR"
        if "£" in text:
            return "GBP"

        return "USD" if source.casefold() == "dummyjson" else "INR"

    @staticmethod
    def _availability(value: object) -> str:
        """Normalize a stock value to a CSV-friendly boolean string."""
        if isinstance(value, bool):
            return str(value).lower()

        if value is None:
            return "true"

        if isinstance(value, (int, float)):
            return str(value > 0).lower()

        normalized = str(value).strip().casefold()
        if normalized in {
            "false",
            "0",
            "no",
            "unavailable",
            "out of stock",
            "outofstock",
            "sold out",
            "discontinued",
        }:
            return "false"

        return "true"

    @staticmethod
    def _brand_from_title(title: str | None) -> str | None:
        """Infer a brand from the first title token."""
        if not title:
            return None

        words = title.strip().split()
        if not words:
            return None

        first = words[0].strip(" -|,:")
        return first or None

    def _extract_product_data(
        self,
        html: str,
        product_url: str,
    ) -> dict[str, str | None]:
        """Extract available product metadata from JSON-LD and HTML."""
        soup = BeautifulSoup(html, "html.parser")

        result: dict[str, str | None] = {
            "product_title": None,
            "price": None,
            "rating": None,
            "product_url": product_url,
            "source": "flipkart",
            "brand": None,
            "category": None,
            "description": None,
            "currency": None,
            "in_stock": None,
        }

        scripts = soup.select('script[type="application/ld+json"]')
        pending: list[object] = []

        for script in scripts:
            try:
                parsed = json.loads(script.string or script.get_text())
                pending.extend(parsed if isinstance(parsed, list) else [parsed])
            except (json.JSONDecodeError, TypeError):
                continue

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
                result["product_title"] or self._clean_text(obj.get("name"))
            )
            result["description"] = (
                result["description"]
                or self._clean_text(obj.get("description"))
            )
            result["category"] = (
                result["category"] or self._clean_text(obj.get("category"))
            )

            brand = obj.get("brand")
            if isinstance(brand, dict):
                brand = brand.get("name")
            result["brand"] = result["brand"] or self._clean_text(brand)

            offers = obj.get("offers", {})
            if isinstance(offers, list):
                offers = offers[0] if offers else {}

            if isinstance(offers, dict):
                price = offers.get("price", offers.get("lowPrice"))
                if price is not None and result["price"] is None:
                    result["price"] = self._number(price)

                result["currency"] = (
                    result["currency"]
                    or self._clean_text(offers.get("priceCurrency"))
                )

                availability = str(
                    offers.get("availability", "")
                ).casefold()
                if availability:
                    result["in_stock"] = str(
                        "outofstock" not in availability
                        and "soldout" not in availability
                        and "discontinued" not in availability
                    ).lower()

            aggregate = obj.get("aggregateRating", {})
            if isinstance(aggregate, dict):
                rating = aggregate.get("ratingValue")
                if rating is not None and result["rating"] is None:
                    result["rating"] = self._number(rating)

        def element_value(element) -> str | None:
            if element is None:
                return None

            content = element.get("content")
            if content is not None:
                return self._clean_text(content)

            return self._clean_text(element.get_text(" ", strip=True))

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
        description_element = (
            soup.select_one('meta[name="description"]')
            or soup.select_one('meta[property="og:description"]')
        )

        result["product_title"] = (
            result["product_title"] or element_value(title_element)
        )

        displayed_price = element_value(price_element)
        result["price"] = (
            result["price"] or self._number(displayed_price)
        )
        result["rating"] = (
            result["rating"]
            or self._number(element_value(rating_element))
        )
        result["description"] = (
            result["description"] or element_value(description_element)
        )
        result["brand"] = (
            result["brand"]
            or self._clean_text(
                element_value(soup.select_one('[itemprop="brand"]'))
            )
        )
        result["currency"] = (
            result["currency"]
            or self._currency_from_value(displayed_price, "flipkart")
        )
        result["brand"] = (
            result["brand"]
            or self._brand_from_title(result["product_title"])
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
        products: list[dict[str, str | None]] = []
        seen_urls: set[str] = set()

        for link in soup.select('a[href*="/p/"]'):
            href = link.get("href")
            if not isinstance(href, str):
                continue

            url = urljoin(self.BASE_URL, href.split("?")[0])
            if url in seen_urls:
                continue

            card = link.find_parent("div", attrs={"data-id": True}) or link
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

            title = title or link.get("title") or link.get_text(" ", strip=True)
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

            displayed_price = (
                price_element.get_text(" ", strip=True)
                if price_element
                else None
            )

            products.append(
                {
                    "product_title": title,
                    "price": displayed_price,
                    "rating": (
                        self._number(rating_element.get_text(" ", strip=True))
                        if rating_element
                        else None
                    ),
                    "product_url": url,
                    "source": "flipkart",
                    "brand": self._brand_from_title(title),
                    "category": None,
                    "description": None,
                    "currency": self._currency_from_value(
                        displayed_price, "flipkart"
                    ),
                    "in_stock": "true",
                }
            )
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
            response = client.get(self.SEARCH_URL + quote_plus(query))
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
        """Search multiple queries and save unique products."""
        if not queries:
            raise ValueError("At least one search query is required")
        if max_products_per_query < 1:
            raise ValueError("max_products_per_query must be at least 1")

        all_products: list[dict[str, str | None]] = []
        seen_urls: set[str] = set()

        for query in queries:
            print(f"\nSearching: {query}")
            for product in self.search_products(
                query,
                max_products=max_products_per_query,
            ):
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
        """Fetch product records and metadata from DummyJSON."""
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

        products: list[dict[str, str | None]] = []

        for item in response.json().get("products", []):
            product_id = item.get("id")
            title = self._clean_text(item.get("title"))
            if product_id is None or not title:
                continue

            products.append(
                {
                    "product_title": title,
                    "price": self._number(item.get("price")),
                    "rating": self._number(item.get("rating")),
                    "product_url": (
                        f"https://dummyjson.com/products/{product_id}"
                    ),
                    "source": "dummyjson",
                    "brand": self._clean_text(item.get("brand")),
                    "category": self._clean_text(item.get("category")),
                    "description": self._clean_text(item.get("description")),
                    "currency": "USD",
                    "in_stock": self._availability(
                        item.get("availabilityStatus", item.get("stock"))
                    ),
                }
            )

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

            result = self._extract_product_data(
                self.driver.page_source,
                self.driver.current_url,
            )

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
        """Append unique products while preserving existing and dynamic fields."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing_rows: list[dict[str, str | None]] = []
        existing_urls: set[str] = set()
        existing_fields: list[str] = []

        if path.exists() and path.stat().st_size > 0:
            with path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as csv_file:
                reader = csv.DictReader(csv_file)
                existing_fields = list(reader.fieldnames or [])
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

        fieldnames = list(
            dict.fromkeys(
                existing_fields
                + self.FIELDNAMES
                + [
                    key
                    for product in products
                    for key in product
                ]
            )
        )

        new_products: list[dict[str, str | None]] = []

        for product in products:
            url = self._clean_text(product.get("product_url"))

            if not url or url in existing_urls:
                continue

            new_product = {
                field: self._clean_text(product.get(field))
                for field in fieldnames
            }
            new_product["product_url"] = url

            if not new_product.get("source"):
                if "dummyjson.com" in url:
                    new_product["source"] = "dummyjson"
                elif "flipkart.com" in url:
                    new_product["source"] = "flipkart"
                else:
                    new_product["source"] = "unknown"

            if not new_product.get("currency"):
                new_product["currency"] = (
                    "USD"
                    if new_product["source"] == "dummyjson"
                    else "INR"
                )

            if not new_product.get("in_stock"):
                new_product["in_stock"] = "true"

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
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"Saved {len(new_products)} new product(s) to {path}.")
