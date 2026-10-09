"""Product data model for the e-commerce assistant."""

from pydantic import BaseModel, Field


class Product(BaseModel):
    """Represent a product in the catalog."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    category: str = Field(min_length=1)
    price: float = Field(gt=0)
    currency: str = "INR"
    in_stock: bool = True
