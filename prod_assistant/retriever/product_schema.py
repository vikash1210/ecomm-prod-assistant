
"""Shared CSV schema normalization for product catalog and ingestion."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "product_title": (
        "product_title", "title", "product_name", "name", "item_name", "product"
    ),
    "price": (
        "price", "cost", "sale_price", "selling_price",
        "current_price", "amount", "product_price"
    ),
    "product_url": (
        "product_url", "url", "product_link", "link", "item_url"
    ),
    "brand": ("brand", "manufacturer", "make"),
    "category": ("category", "product_category", "department", "type"),
    "description": ("description", "details", "product_description", "summary"),
    "rating": ("rating", "stars", "product_rating", "average_rating"),
    "source": ("source", "provider", "marketplace", "platform"),
    "currency": ("currency", "currency_code", "price_currency"),
    "in_stock": (
        "in_stock", "availability", "stock_status",
        "available", "availability_status"
    ),
}

CORE_FIELDS = set(COLUMN_ALIASES)


def normalize_column_name(value: object) -> str:
    """Normalize a column name for alias matching."""
    return re.sub(
        r"[^a-z0-9]+", "_", str(value).strip().casefold()
    ).strip("_")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize recognized aliases while retaining additional columns."""
    result = df.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]

    for canonical, aliases in COLUMN_ALIASES.items():
        present = [alias for alias in aliases if alias in result.columns]
        if not present:
            continue

        if canonical not in result.columns:
            result[canonical] = result[present[0]]

        for alias in present:
            if alias == canonical:
                continue

            current = result[canonical]
            missing = current.isna() | current.astype(str).str.strip().eq("")
            result[canonical] = current.where(~missing, result[alias])

    return result


def optional_text(value: object) -> str:
    """Convert an optional CSV value to clean text."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null", "n/a", "na"} else text


def parse_price(value: object) -> float | None:
    """Parse numeric prices containing common currency symbols."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if number > 0 else None

    text = str(value).strip()
    cleaned = re.sub(r"[^\d.\-]", "", text.replace(",", ""))

    try:
        number = float(cleaned)
    except (TypeError, ValueError):
        return None

    return number if number > 0 else None


def infer_currency(
    value: object,
    supplied_currency: object = "",
    source: object = "",
) -> str:
    """Use explicit currency, symbols, or the project's source defaults."""
    explicit = optional_text(supplied_currency).upper()
    if explicit:
        return explicit

    text = optional_text(value)
    if "₹" in text:
        return "INR"
    if "$" in text:
        return "USD"
    if "€" in text:
        return "EUR"
    if "£" in text:
        return "GBP"

    source_name = optional_text(source).casefold()
    if source_name == "dummyjson":
        return "USD"
    if source_name:
        return "INR"

    return "UNKNOWN"


def parse_rating(value: object) -> float | None:
    """Return a valid rating between zero and five."""
    if value is None or pd.isna(value):
        return None

    try:
        rating = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    return rating if 0 <= rating <= 5 else None


def parse_stock(value: object) -> bool:
    """Parse common product availability values."""
    text = optional_text(value).casefold()
    if not text:
        return True

    if text in {"true", "1", "yes", "available", "in stock", "instock"}:
        return True

    return text not in {
        "false",
        "0",
        "no",
        "unavailable",
        "out of stock",
        "outofstock",
        "sold out",
        "discontinued",
    }


def row_to_product_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a normalized row into canonical fields and extra attributes."""
    values = {normalize_column_name(key): value for key, value in row.items()}

    title = optional_text(values.get("product_title"))
    raw_price = values.get("price")
    price = parse_price(raw_price)
    url = optional_text(values.get("product_url"))
    source = optional_text(values.get("source")) or "unknown"

    recognized_columns = {
        alias
        for aliases in COLUMN_ALIASES.values()
        for alias in aliases
    }

    extra_attributes = {
        key: optional_text(value)
        for key, value in values.items()
        if key not in recognized_columns and optional_text(value)
    }


    return {
        "product_title": title,
        "price": price,
        "product_url": url or None,
        "brand": optional_text(values.get("brand")) or None,
        "category": optional_text(values.get("category")) or "Uncategorized",
        "description": optional_text(values.get("description")),
        "rating": parse_rating(values.get("rating")),
        "source": source,
        "currency": infer_currency(
            raw_price, values.get("currency"), source
        ),
        "in_stock": parse_stock(values.get("in_stock")),
        "extra_attributes": extra_attributes,
    }
