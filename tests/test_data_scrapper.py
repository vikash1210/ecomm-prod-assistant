
import csv

import httpx
import pytest

from prod_assistant.etl.data_scrapper import FlipkartScraper

PRODUCT_HTML = """
<html>
<head>
    <meta property="og:title"
          content="Google Pixel 11 Pro (Canyon, 256 GB)">
    <meta property="product:price:amount" content="119999">
</head>
<body>
    <div class="XQDdHH">4.4</div>
</body>
</html>
"""


def test_extract_product_data():
    scraper = FlipkartScraper()

    result = scraper._extract_product_data(
        PRODUCT_HTML,
        "https://www.flipkart.com/test-product/p/123",
    )

    assert result["product_title"] == (
        "Google Pixel 11 Pro (Canyon, 256 GB)"
    )
    assert result["price"] == "119999"
    assert result["rating"] == "4.4"
    assert result["product_url"] == (
        "https://www.flipkart.com/test-product/p/123"
    )


def test_parse_multiple_search_results():
    scraper = FlipkartScraper()

    html = """
    <html>
    <body>
        <div data-id="PHONE1">
            <a href="/google-pixel/p/phone1">
                <div class="KzDlHZ">Google Pixel Phone</div>
            </a>
            <div class="Nx9bqj">₹50,000</div>
            <div class="XQDdHH">4.5</div>
        </div>

        <div data-id="PHONE2">
            <a href="/samsung-galaxy/p/phone2">
                <div class="KzDlHZ">Samsung Galaxy Phone</div>
            </a>
            <div class="Nx9bqj">₹40,000</div>
            <div class="XQDdHH">4.3</div>
        </div>
    </body>
    </html>
    """

    products = scraper.parse_search_results(html, max_products=5)

    assert len(products) == 2
    assert products[0]["product_title"] == "Google Pixel Phone"
    assert products[0]["price"] == "₹50,000"
    assert products[0]["rating"] == "4.5"
    assert products[1]["product_title"] == "Samsung Galaxy Phone"
    assert products[1]["price"] == "₹40,000"
    assert products[1]["rating"] == "4.3"


def test_parse_search_results_rejects_invalid_limit():
    scraper = FlipkartScraper()

    with pytest.raises(
        ValueError,
        match="max_products must be at least 1",
    ):
        scraper.parse_search_results("<html></html>", max_products=0)


def test_save_to_csv(tmp_path):
    scraper = FlipkartScraper()
    output_file = tmp_path / "products.csv"

    products = [
        {
            "product_title": "Google Pixel 11 Pro",
            "price": "119999",
            "rating": "4.4",
            "product_url": (
                "https://www.flipkart.com/test-product/p/123"
            ),
        }
    ]

    scraper.save_to_csv(products, str(output_file))

    with output_file.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["product_title"] == "Google Pixel 11 Pro"
    assert rows[0]["price"] == "119999"
    assert rows[0]["rating"] == "4.4"
    assert rows[0]["product_url"] == (
        "https://www.flipkart.com/test-product/p/123"
    )


def test_fetch_dummyjson_products(monkeypatch, tmp_path):
    scraper = FlipkartScraper()
    output_file = tmp_path / "products.csv"

    def mock_get(url, params, timeout):
        assert url == "https://dummyjson.com/products/search"
        assert params == {"q": "phone", "limit": 1}

        return httpx.Response(
            200,
            json={
                "products": [
                    {
                        "id": 1,
                        "title": "Test Phone",
                        "price": 500,
                        "rating": 4.5,
                    }
                ]
            },
            request=httpx.Request(
                "GET",
                "https://dummyjson.com/products/search",
            ),
        )

    monkeypatch.setattr(httpx, "get", mock_get)

    products = scraper.fetch_dummyjson_products(
        query="phone",
        limit=1,
        output_path=str(output_file),
    )

    assert len(products) == 1
    assert products[0]["product_title"] == "Test Phone"
    assert products[0]["price"] == "500"
    assert products[0]["rating"] == "4.5"
    assert products[0]["product_url"] == (
        "https://dummyjson.com/products/1"
    )

    with output_file.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["product_title"] == "Test Phone"
