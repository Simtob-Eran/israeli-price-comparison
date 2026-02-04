"""Price Extraction Agent.

This agent scrapes product pages and extracts pricing information
using MCP web scraping tools.
"""

import asyncio
import re
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import HttpUrl
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.mcp_client.client import MCPClient
from src.models.schemas import AvailabilityStatus, PriceData, SearchResult
from src.utils.config import Settings, get_settings
from src.utils.logger import AgentLogger

# System prompt for price extraction
EXTRACTION_PROMPT = """You are an expert at extracting product pricing information from HTML content.

Given the HTML content of a product page, extract the following information:
1. **Price**: The main product price (look for price elements, schema.org markup, JSON-LD)
2. **Currency**: Detect the currency (₪, $, €, £) and code (ILS, USD, EUR, GBP)
3. **Shipping Cost**: If available, extract shipping cost
4. **Availability**: In stock, out of stock, limited, preorder
5. **Store Name**: The retailer/store name
6. **Product Title**: The product title as shown on the page
7. **Seller Rating**: If shown (1-5 scale)

**Important Notes**:
- Israeli Shekel (₪) prices may be shown as "₪X,XXX" or "X,XXX ₪"
- Handle price formats: 4,999, 4999, 4.999,00
- Look for JSON-LD or Schema.org Product markup first (most reliable)
- Watch for "was/now" pricing - extract the current price
- Ignore prices that are clearly for accessories or related products

**Output Format (JSON)**:
{
    "price": 4999.00,
    "currency": "ILS",
    "currency_symbol": "₪",
    "shipping_cost": 29.00,
    "availability": "in_stock",
    "store_name": "Store Name",
    "product_title": "Product Title",
    "seller_rating": 4.5,
    "raw_price_text": "₪4,999"
}

Return null for any field you cannot determine with confidence."""


class PriceExtractionAgent:
    """Agent for extracting pricing information from web pages.

    This agent fetches product pages and uses LLM-powered extraction
    to pull out structured pricing data.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[ChatOpenAI] = None,
        mcp_client: Optional[MCPClient] = None,
    ) -> None:
        """Initialize the Price Extraction Agent.

        Args:
            settings: Application settings.
            llm: LangChain LLM instance.
            mcp_client: Connected MCP client for web scraping tools.
        """
        self.settings = settings or get_settings()
        self.logger = AgentLogger("PriceExtraction")
        self.mcp_client = mcp_client

        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model=self.settings.openai.model,
                temperature=0,  # Use 0 for consistent extraction
                api_key=self.settings.openai.api_key,
                request_timeout=self.settings.openai.request_timeout,
            )

        # Configuration
        self.concurrent_requests = (
            self.settings.agents.price_extraction.concurrent_requests or 5
        )
        self.rate_limit_delay = self.settings.scraping.rate_limit_delay
        self.timeout = self.settings.scraping.timeout

        # Currency patterns for regex extraction
        self.currency_patterns = {
            "ILS": (r"₪\s*([\d,]+(?:\.\d{2})?)", "₪"),
            "USD": (r"\$\s*([\d,]+(?:\.\d{2})?)", "$"),
            "EUR": (r"€\s*([\d,]+(?:\.\d{2})?)", "€"),
            "GBP": (r"£\s*([\d,]+(?:\.\d{2})?)", "£"),
        }

    async def extract_prices(
        self, search_results: list[SearchResult]
    ) -> list[PriceData]:
        """Extract prices from multiple URLs concurrently.

        Args:
            search_results: List of search results with URLs to scrape.

        Returns:
            List of PriceData objects with extracted pricing.
        """
        self.logger.start(
            "extract_prices",
            url_count=len(search_results),
            concurrent_requests=self.concurrent_requests,
        )

        all_prices: list[PriceData] = []

        try:
            # Process URLs in batches to respect rate limits
            semaphore = asyncio.Semaphore(self.concurrent_requests)

            async def extract_with_semaphore(result: SearchResult) -> Optional[PriceData]:
                async with semaphore:
                    price = await self._extract_single_price(result)
                    await asyncio.sleep(self.rate_limit_delay)
                    return price

            # Create extraction tasks
            tasks = [extract_with_semaphore(result) for result in search_results]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful extractions
            for result in results:
                if isinstance(result, Exception):
                    self.logger.debug(f"Extraction failed: {result}")
                    continue
                if result is not None:
                    all_prices.append(result)

            self.logger.complete(
                "extract_prices",
                duration_ms=0,
                total_urls=len(search_results),
                successful_extractions=len(all_prices),
            )

            return all_prices

        except Exception as e:
            self.logger.error("extract_prices", e)
            raise

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _extract_single_price(
        self, search_result: SearchResult
    ) -> Optional[PriceData]:
        """Extract price from a single URL.

        Args:
            search_result: Search result containing the URL.

        Returns:
            PriceData if extraction successful, None otherwise.
        """
        url = str(search_result.url)

        try:
            # Fetch page content
            html_content = await self._fetch_page_content(url)
            if not html_content:
                return None

            # First, try to extract structured data (JSON-LD, Schema.org)
            structured_data = await self._extract_structured_data(html_content)

            if structured_data:
                return self._parse_structured_data(structured_data, search_result)

            # Fallback: Use LLM to extract from HTML
            return await self._extract_with_llm(html_content, search_result)

        except Exception as e:
            self.logger.debug(
                f"Failed to extract price from {url}: {e}",
                url=url,
            )
            return None

    async def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch page content using MCP tool or fallback.

        Args:
            url: URL to fetch.

        Returns:
            HTML content or None if fetch failed.
        """
        if self.mcp_client:
            try:
                result = await self.mcp_client.call_tool(
                    tool_name="fetch_page_content",
                    arguments={"url": url},
                    timeout=self.timeout,
                )
                return result.get("content") if isinstance(result, dict) else None
            except Exception as e:
                self.logger.debug(f"MCP fetch failed for {url}: {e}")
                return None
        else:
            # Fallback: use httpx directly (if MCP not available)
            return await self._fetch_with_httpx(url)

    async def _fetch_with_httpx(self, url: str) -> Optional[str]:
        """Fetch page content using httpx (fallback).

        Args:
            url: URL to fetch.

        Returns:
            HTML content or None.
        """
        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.settings.scraping.user_agent},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Check content size
                if len(response.content) > self.settings.scraping.max_page_size:
                    self.logger.warning(f"Page too large: {url}")
                    return None

                return response.text

        except Exception as e:
            self.logger.debug(f"httpx fetch failed for {url}: {e}")
            return None

    async def _extract_structured_data(
        self, html_content: str
    ) -> Optional[dict[str, Any]]:
        """Extract structured data (JSON-LD, Schema.org) from HTML.

        Args:
            html_content: HTML page content.

        Returns:
            Extracted structured data or None.
        """
        import json

        # Look for JSON-LD script tags
        json_ld_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
        matches = re.findall(json_ld_pattern, html_content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                data = json.loads(match.strip())

                # Handle array of objects
                if isinstance(data, list):
                    for item in data:
                        if self._is_product_schema(item):
                            return item
                elif self._is_product_schema(data):
                    return data

            except json.JSONDecodeError:
                continue

        return None

    def _is_product_schema(self, data: dict[str, Any]) -> bool:
        """Check if JSON-LD data represents a Product.

        Args:
            data: JSON-LD data.

        Returns:
            True if this is product data.
        """
        schema_type = data.get("@type", "")
        if isinstance(schema_type, list):
            return "Product" in schema_type
        return schema_type == "Product"

    def _parse_structured_data(
        self, data: dict[str, Any], search_result: SearchResult
    ) -> Optional[PriceData]:
        """Parse Schema.org Product data into PriceData.

        Args:
            data: Schema.org Product data.
            search_result: Original search result.

        Returns:
            PriceData if parsing successful.
        """
        try:
            # Extract offers
            offers = data.get("offers", {})
            if isinstance(offers, list):
                offers = offers[0] if offers else {}

            # Get price
            price_str = offers.get("price") or offers.get("lowPrice")
            if not price_str:
                return None

            price = float(str(price_str).replace(",", ""))

            # Get currency
            currency = offers.get("priceCurrency", "ILS")
            currency_symbol = self._get_currency_symbol(currency)

            # Get availability
            availability_str = offers.get("availability", "")
            availability = self._parse_availability(availability_str)

            # Get seller/store name
            seller = offers.get("seller", {})
            store_name = (
                seller.get("name")
                if isinstance(seller, dict)
                else search_result.domain
            )

            return PriceData(
                url=search_result.url,
                store_name=store_name or search_result.domain,
                price=price,
                currency=currency,
                currency_symbol=currency_symbol,
                shipping_cost=self._extract_shipping_cost(offers),
                availability=availability,
                extracted_at=datetime.utcnow(),
                product_title=data.get("name"),
                image_url=data.get("image"),
            )

        except Exception as e:
            self.logger.debug(f"Failed to parse structured data: {e}")
            return None

    async def _extract_with_llm(
        self, html_content: str, search_result: SearchResult
    ) -> Optional[PriceData]:
        """Use LLM to extract price from HTML content.

        Args:
            html_content: HTML page content.
            search_result: Original search result.

        Returns:
            PriceData if extraction successful.
        """
        # Truncate HTML to avoid token limits
        max_chars = 15000
        if len(html_content) > max_chars:
            # Try to keep relevant parts
            html_content = self._smart_truncate(html_content, max_chars)

        messages = [
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(
                content=f"Extract pricing information from this product page:\n\n"
                f"URL: {search_result.url}\n"
                f"Domain: {search_result.domain}\n\n"
                f"HTML Content:\n{html_content}"
            ),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return self._parse_llm_response(response.content, search_result)
        except Exception as e:
            self.logger.debug(f"LLM extraction failed: {e}")
            return None

    def _parse_llm_response(
        self, response_content: str, search_result: SearchResult
    ) -> Optional[PriceData]:
        """Parse LLM extraction response into PriceData.

        Args:
            response_content: LLM response.
            search_result: Original search result.

        Returns:
            PriceData if parsing successful.
        """
        import json

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response_content)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group(0))

            # Skip if no price found
            if not data.get("price"):
                return None

            price = float(data["price"])
            if price <= 0:
                return None

            return PriceData(
                url=search_result.url,
                store_name=data.get("store_name") or search_result.domain,
                price=price,
                currency=data.get("currency", "ILS"),
                currency_symbol=data.get("currency_symbol", "₪"),
                shipping_cost=data.get("shipping_cost"),
                availability=self._parse_availability(data.get("availability", "")),
                seller_rating=data.get("seller_rating"),
                extracted_at=datetime.utcnow(),
                product_title=data.get("product_title"),
                raw_price_text=data.get("raw_price_text"),
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.debug(f"Failed to parse LLM response: {e}")
            return None

    def _smart_truncate(self, html: str, max_chars: int) -> str:
        """Intelligently truncate HTML keeping relevant sections.

        Args:
            html: Full HTML content.
            max_chars: Maximum character limit.

        Returns:
            Truncated HTML with relevant sections.
        """
        # Remove scripts and styles
        html = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
        html = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", html, flags=re.IGNORECASE)

        # Keep price-related sections
        price_keywords = ["price", "מחיר", "cost", "₪", "$", "€", "£", "add to cart", "buy"]
        lines = html.split("\n")

        relevant_lines: list[str] = []
        context_lines = 10

        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in price_keywords):
                # Add surrounding context
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines)
                relevant_lines.extend(lines[start:end])

        result = "\n".join(relevant_lines) if relevant_lines else html
        return result[:max_chars]

    def _get_currency_symbol(self, currency_code: str) -> str:
        """Get currency symbol from code.

        Args:
            currency_code: ISO currency code.

        Returns:
            Currency symbol.
        """
        symbols = {
            "ILS": "₪",
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
        }
        return symbols.get(currency_code.upper(), "₪")

    def _parse_availability(self, availability_str: str) -> AvailabilityStatus:
        """Parse availability string to enum.

        Args:
            availability_str: Availability text or URL.

        Returns:
            AvailabilityStatus enum value.
        """
        availability_lower = availability_str.lower()

        if any(x in availability_lower for x in ["instock", "in_stock", "in stock", "במלאי"]):
            return AvailabilityStatus.IN_STOCK
        elif any(x in availability_lower for x in ["outofstock", "out_of_stock", "out of stock", "אזל"]):
            return AvailabilityStatus.OUT_OF_STOCK
        elif any(x in availability_lower for x in ["limited", "low", "מוגבל"]):
            return AvailabilityStatus.LIMITED
        elif any(x in availability_lower for x in ["preorder", "pre-order", "הזמנה מראש"]):
            return AvailabilityStatus.PREORDER

        return AvailabilityStatus.UNKNOWN

    def _extract_shipping_cost(self, offers: dict[str, Any]) -> Optional[float]:
        """Extract shipping cost from offers data.

        Args:
            offers: Schema.org Offer data.

        Returns:
            Shipping cost or None.
        """
        shipping = offers.get("shippingDetails", {})
        if isinstance(shipping, dict):
            rate = shipping.get("shippingRate", {})
            if isinstance(rate, dict):
                value = rate.get("value")
                if value:
                    try:
                        return float(value)
                    except ValueError:
                        pass

        return None


# Factory function
def create_price_extraction_agent(
    settings: Optional[Settings] = None,
    mcp_client: Optional[MCPClient] = None,
) -> PriceExtractionAgent:
    """Create a Price Extraction Agent instance.

    Args:
        settings: Optional application settings.
        mcp_client: Optional MCP client.

    Returns:
        Configured PriceExtractionAgent instance.
    """
    return PriceExtractionAgent(settings=settings, mcp_client=mcp_client)
