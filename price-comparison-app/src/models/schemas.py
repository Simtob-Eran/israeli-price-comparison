"""Pydantic models and schemas for the price comparison application.

This module defines all data structures used throughout the application,
including product information, price data, search results, and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ProductCategory(str, Enum):
    """Product category enumeration."""

    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    HOME_GARDEN = "home_garden"
    SPORTS = "sports"
    TOYS = "toys"
    HEALTH_BEAUTY = "health_beauty"
    AUTOMOTIVE = "automotive"
    FOOD_GROCERY = "food_grocery"
    OTHER = "other"


class AvailabilityStatus(str, Enum):
    """Product availability status."""

    IN_STOCK = "in_stock"
    OUT_OF_STOCK = "out_of_stock"
    LIMITED = "limited"
    PREORDER = "preorder"
    UNKNOWN = "unknown"


class CurrencyInfo(BaseModel):
    """Currency information for price normalization."""

    symbol: str = Field(..., description="Currency symbol (e.g., ₪, $, €)")
    code: str = Field(..., description="ISO currency code (e.g., ILS, USD)")
    name: str = Field(..., description="Full currency name")


class ProductInfo(BaseModel):
    """Structured product information extracted by the Product Understanding Agent.

    This model contains all relevant information about a product that was
    extracted from user input (either a product name or URL).
    """

    original_input: str = Field(
        ..., description="Original user input (product name or URL)"
    )
    product_name: str = Field(
        ..., description="Normalized product name"
    )
    brand: Optional[str] = Field(
        None, description="Product brand/manufacturer"
    )
    model: Optional[str] = Field(
        None, description="Product model number or identifier"
    )
    category: ProductCategory = Field(
        default=ProductCategory.OTHER, description="Product category"
    )
    key_specs: dict[str, Any] = Field(
        default_factory=dict, description="Key product specifications"
    )
    search_queries: list[str] = Field(
        default_factory=list, description="Optimized search queries for different strategies"
    )
    is_url_input: bool = Field(
        default=False, description="Whether the original input was a URL"
    )
    source_url: Optional[HttpUrl] = Field(
        None, description="Source URL if input was a URL"
    )

    @field_validator("search_queries", mode="before")
    @classmethod
    def ensure_search_queries(cls, v: list[str], info: Any) -> list[str]:
        """Ensure at least one search query exists."""
        if not v:
            # Generate default query from product name
            product_name = info.data.get("product_name", "")
            if product_name:
                return [product_name]
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "original_input": "iPhone 15 Pro Max 256GB",
                "product_name": "Apple iPhone 15 Pro Max 256GB",
                "brand": "Apple",
                "model": "iPhone 15 Pro Max",
                "category": "electronics",
                "key_specs": {
                    "storage": "256GB",
                    "color": "Natural Titanium",
                    "display": "6.7 inch",
                },
                "search_queries": [
                    "iPhone 15 Pro Max 256GB",
                    "Apple iPhone 15 Pro Max",
                    "iPhone 15 Pro Max price",
                ],
                "is_url_input": False,
                "source_url": None,
            }
        }
    }


class PriceData(BaseModel):
    """Price information from a single source.

    This model represents pricing data extracted from a single retailer
    or e-commerce website.
    """

    url: HttpUrl = Field(..., description="Product page URL")
    store_name: str = Field(..., description="Retailer/store name")
    price: float = Field(..., ge=0, description="Product price")
    currency: str = Field(default="ILS", description="Currency code (ISO 4217)")
    currency_symbol: str = Field(default="₪", description="Currency symbol")
    shipping_cost: Optional[float] = Field(
        None, ge=0, description="Shipping cost (if available)"
    )
    availability: AvailabilityStatus = Field(
        default=AvailabilityStatus.UNKNOWN, description="Product availability"
    )
    seller_rating: Optional[float] = Field(
        None, ge=0, le=5, description="Seller rating (0-5 scale)"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow, description="Extraction timestamp"
    )
    relevance_score: Optional[float] = Field(
        None, ge=0, le=100, description="Relevance score (0-100)"
    )
    product_title: Optional[str] = Field(
        None, description="Product title as shown on the page"
    )
    image_url: Optional[HttpUrl] = Field(
        None, description="Product image URL"
    )
    raw_price_text: Optional[str] = Field(
        None, description="Raw price text before parsing"
    )

    @property
    def total_cost(self) -> float:
        """Calculate total cost including shipping."""
        return self.price + (self.shipping_cost or 0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://www.example.com/product/123",
                "store_name": "Example Store",
                "price": 4999.00,
                "currency": "ILS",
                "currency_symbol": "₪",
                "shipping_cost": 29.00,
                "availability": "in_stock",
                "seller_rating": 4.5,
                "extracted_at": "2024-01-15T10:30:00Z",
                "relevance_score": 95.0,
                "product_title": "Apple iPhone 15 Pro Max 256GB",
                "image_url": "https://www.example.com/images/product.jpg",
            }
        }
    }


class SearchResult(BaseModel):
    """Search result from web search agent."""

    url: HttpUrl = Field(..., description="Result URL")
    title: str = Field(..., description="Result title")
    snippet: Optional[str] = Field(None, description="Result snippet/description")
    domain: str = Field(..., description="Domain name")
    is_ecommerce: bool = Field(
        default=False, description="Whether the site is an e-commerce platform"
    )
    position: int = Field(..., ge=1, description="Position in search results")

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://www.ksp.co.il/product/123",
                "title": "iPhone 15 Pro Max 256GB - KSP",
                "snippet": "Buy iPhone 15 Pro Max 256GB at the best price...",
                "domain": "ksp.co.il",
                "is_ecommerce": True,
                "position": 1,
            }
        }
    }


class ValidationResult(BaseModel):
    """Result of price data validation."""

    price_data: PriceData = Field(..., description="Original price data")
    is_valid: bool = Field(..., description="Whether the price data is valid")
    relevance_score: float = Field(
        ..., ge=0, le=100, description="Relevance score (0-100)"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence level (0-1)"
    )
    issues: list[str] = Field(
        default_factory=list, description="List of validation issues"
    )
    matched_specs: dict[str, bool] = Field(
        default_factory=dict, description="Specification match results"
    )
    is_suspicious_price: bool = Field(
        default=False, description="Whether the price seems suspicious"
    )
    suspicious_reason: Optional[str] = Field(
        None, description="Reason for suspicious price flag"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "is_valid": True,
                "relevance_score": 92.5,
                "confidence": 0.95,
                "issues": [],
                "matched_specs": {
                    "brand": True,
                    "model": True,
                    "storage": True,
                },
                "is_suspicious_price": False,
            }
        }
    }


class RankedResult(BaseModel):
    """A single ranked price result with analysis."""

    rank: int = Field(..., ge=1, description="Rank position")
    price_data: PriceData = Field(..., description="Price data for this result")
    total_cost: float = Field(..., ge=0, description="Total cost including shipping")
    savings_vs_average: Optional[float] = Field(
        None, description="Savings compared to average price"
    )
    savings_percentage: Optional[float] = Field(
        None, ge=-100, le=100, description="Savings percentage vs average"
    )
    is_best_deal: bool = Field(
        default=False, description="Whether this is the best deal"
    )
    deal_score: float = Field(
        default=0, ge=0, le=100, description="Overall deal score (0-100)"
    )
    recommendation: Optional[str] = Field(
        None, description="Purchase recommendation"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "rank": 1,
                "total_cost": 4999.00,
                "savings_vs_average": 500.00,
                "savings_percentage": 9.1,
                "is_best_deal": True,
                "deal_score": 95.0,
                "recommendation": "Best price with reliable seller",
            }
        }
    }


class RankedResults(BaseModel):
    """Complete ranked results from price comparison."""

    product_info: ProductInfo = Field(..., description="Product information")
    results: list[RankedResult] = Field(
        default_factory=list, description="Ranked results"
    )
    average_price: Optional[float] = Field(
        None, ge=0, description="Average market price"
    )
    lowest_price: Optional[float] = Field(
        None, ge=0, description="Lowest found price"
    )
    highest_price: Optional[float] = Field(
        None, ge=0, description="Highest found price"
    )
    price_range: Optional[float] = Field(
        None, ge=0, description="Price range (max - min)"
    )
    total_results: int = Field(
        default=0, ge=0, description="Total number of results"
    )
    search_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Search timestamp"
    )
    confidence_score: float = Field(
        default=0, ge=0, le=100, description="Overall confidence in results"
    )

    @property
    def best_deal(self) -> Optional[RankedResult]:
        """Get the best deal from results."""
        for result in self.results:
            if result.is_best_deal:
                return result
        return self.results[0] if self.results else None

    model_config = {
        "json_schema_extra": {
            "example": {
                "average_price": 5500.00,
                "lowest_price": 4999.00,
                "highest_price": 6200.00,
                "price_range": 1201.00,
                "total_results": 15,
                "confidence_score": 87.5,
            }
        }
    }


class AgentError(BaseModel):
    """Error information from an agent."""

    agent_name: str = Field(..., description="Name of the agent that encountered the error")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )
    recoverable: bool = Field(
        default=True, description="Whether the error is recoverable"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "agent_name": "WebSearchAgent",
                "error_type": "SearchError",
                "error_message": "Failed to fetch search results",
                "recoverable": True,
                "context": {"query": "iPhone 15 Pro Max", "attempt": 1},
            }
        }
    }


class MCPToolCall(BaseModel):
    """MCP tool call request."""

    tool_name: str = Field(..., description="Name of the MCP tool to call")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )
    timeout: Optional[float] = Field(
        None, ge=0, description="Timeout in seconds"
    )


class MCPToolResult(BaseModel):
    """MCP tool call result."""

    tool_name: str = Field(..., description="Name of the tool that was called")
    success: bool = Field(..., description="Whether the call was successful")
    result: Any = Field(None, description="Tool result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    duration_ms: float = Field(..., ge=0, description="Call duration in milliseconds")


class FinalReport(BaseModel):
    """Final report generated by the Reporting Agent."""

    product_info: ProductInfo = Field(..., description="Product information")
    ranked_results: RankedResults = Field(..., description="Ranked price results")
    summary: str = Field(..., description="Executive summary")
    recommendation: str = Field(..., description="Purchase recommendation")
    comparison_table: str = Field(..., description="Markdown comparison table")
    best_time_to_buy: Optional[str] = Field(
        None, description="Suggested best time to purchase"
    )
    price_history_context: Optional[str] = Field(
        None, description="Historical price context"
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Report generation timestamp"
    )
    search_duration_seconds: float = Field(
        default=0, ge=0, description="Total search duration"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "summary": "Found 15 prices for iPhone 15 Pro Max 256GB...",
                "recommendation": "Best deal at KSP for ₪4,999...",
                "comparison_table": "| Store | Price | Shipping | Total |...",
                "best_time_to_buy": "Prices typically drop during sales...",
            }
        }
    }
