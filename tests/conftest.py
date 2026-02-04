"""Pytest configuration and fixtures for the price comparison tests."""

import os
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.schemas import (
    AvailabilityStatus,
    PriceData,
    ProductCategory,
    ProductInfo,
    RankedResult,
    RankedResults,
    SearchResult,
)
from src.utils.config import Settings


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    # Set test environment variables
    os.environ.setdefault("OPENAI_API_KEY", "test-api-key")
    os.environ.setdefault("SERPER_API_KEY", "test-serper-key")

    return Settings.from_yaml()


@pytest.fixture
def sample_product_info() -> ProductInfo:
    """Create sample ProductInfo for testing."""
    return ProductInfo(
        original_input="iPhone 15 Pro Max 256GB",
        product_name="Apple iPhone 15 Pro Max 256GB",
        brand="Apple",
        model="iPhone 15 Pro Max",
        category=ProductCategory.ELECTRONICS,
        key_specs={
            "storage": "256GB",
            "color": "Natural Titanium",
            "display": "6.7 inch",
        },
        search_queries=[
            "iPhone 15 Pro Max 256GB",
            "Apple iPhone 15 Pro Max price",
            "iPhone 15 Pro Max buy",
        ],
        is_url_input=False,
        source_url=None,
    )


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create sample search results for testing."""
    return [
        SearchResult(
            url="https://www.ksp.co.il/product/12345",
            title="iPhone 15 Pro Max 256GB - KSP",
            snippet="Buy Apple iPhone 15 Pro Max 256GB at the best price",
            domain="ksp.co.il",
            is_ecommerce=True,
            position=1,
        ),
        SearchResult(
            url="https://www.ivory.co.il/product/67890",
            title="Apple iPhone 15 Pro Max 256GB - Ivory",
            snippet="iPhone 15 Pro Max with free shipping",
            domain="ivory.co.il",
            is_ecommerce=True,
            position=2,
        ),
        SearchResult(
            url="https://www.bug.co.il/product/11111",
            title="iPhone 15 Pro Max 256GB Natural Titanium",
            snippet="Best deal on iPhone 15 Pro Max",
            domain="bug.co.il",
            is_ecommerce=True,
            position=3,
        ),
    ]


@pytest.fixture
def sample_price_data() -> list[PriceData]:
    """Create sample price data for testing."""
    return [
        PriceData(
            url="https://www.ksp.co.il/product/12345",
            store_name="KSP",
            price=4999.0,
            currency="ILS",
            currency_symbol="₪",
            shipping_cost=0,
            availability=AvailabilityStatus.IN_STOCK,
            seller_rating=4.5,
            extracted_at=datetime.utcnow(),
            relevance_score=95.0,
            product_title="Apple iPhone 15 Pro Max 256GB Natural Titanium",
        ),
        PriceData(
            url="https://www.ivory.co.il/product/67890",
            store_name="Ivory",
            price=5199.0,
            currency="ILS",
            currency_symbol="₪",
            shipping_cost=29.0,
            availability=AvailabilityStatus.IN_STOCK,
            seller_rating=4.3,
            extracted_at=datetime.utcnow(),
            relevance_score=92.0,
            product_title="iPhone 15 Pro Max 256GB",
        ),
        PriceData(
            url="https://www.bug.co.il/product/11111",
            store_name="Bug",
            price=5099.0,
            currency="ILS",
            currency_symbol="₪",
            shipping_cost=0,
            availability=AvailabilityStatus.LIMITED,
            seller_rating=4.2,
            extracted_at=datetime.utcnow(),
            relevance_score=88.0,
            product_title="Apple iPhone 15 Pro Max 256GB",
        ),
    ]


@pytest.fixture
def sample_ranked_results(
    sample_product_info: ProductInfo,
    sample_price_data: list[PriceData],
) -> RankedResults:
    """Create sample ranked results for testing."""
    ranked = []
    sorted_prices = sorted(sample_price_data, key=lambda p: p.total_cost)

    for rank, price_data in enumerate(sorted_prices, start=1):
        ranked.append(
            RankedResult(
                rank=rank,
                price_data=price_data,
                total_cost=price_data.total_cost,
                savings_vs_average=200.0 if rank == 1 else 100.0,
                savings_percentage=3.8 if rank == 1 else 1.9,
                is_best_deal=(rank == 1),
                deal_score=95.0 if rank == 1 else 85.0,
                recommendation="Best deal!" if rank == 1 else "Good option",
            )
        )

    return RankedResults(
        product_info=sample_product_info,
        results=ranked,
        average_price=5099.0,
        lowest_price=4999.0,
        highest_price=5228.0,
        price_range=229.0,
        total_results=3,
        confidence_score=87.5,
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing agents without API calls."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock()
    return mock


@pytest.fixture
def mock_mcp_client() -> MagicMock:
    """Create a mock MCP client for testing."""
    mock = MagicMock()
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.call_tool = AsyncMock()
    mock._connected = True
    mock._tools = {
        "serper_search": MagicMock(),
        "serper_shopping": MagicMock(),
        "fetch_page_content": MagicMock(),
    }
    return mock


@pytest.fixture
def mock_search_response() -> dict[str, Any]:
    """Create mock search API response."""
    return {
        "organic": [
            {
                "title": "iPhone 15 Pro Max 256GB - KSP",
                "link": "https://www.ksp.co.il/product/12345",
                "snippet": "Best price for iPhone 15 Pro Max",
            },
            {
                "title": "Apple iPhone 15 Pro Max - Ivory",
                "link": "https://www.ivory.co.il/product/67890",
                "snippet": "Free shipping on iPhone 15 Pro Max",
            },
        ],
    }


@pytest.fixture
def mock_fetch_response() -> dict[str, Any]:
    """Create mock page fetch response."""
    return {
        "content": """
        <html>
        <head>
            <script type="application/ld+json">
            {
                "@type": "Product",
                "name": "Apple iPhone 15 Pro Max 256GB",
                "offers": {
                    "@type": "Offer",
                    "price": "4999",
                    "priceCurrency": "ILS",
                    "availability": "https://schema.org/InStock"
                }
            }
            </script>
        </head>
        <body>
            <h1>iPhone 15 Pro Max 256GB</h1>
            <span class="price">₪4,999</span>
        </body>
        </html>
        """,
        "status_code": 200,
    }
