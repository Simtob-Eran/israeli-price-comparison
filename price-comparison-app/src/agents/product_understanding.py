"""Product Understanding Agent.

This agent analyzes user input (product name or URL) and extracts
structured product information for search optimization.
"""

import re
from typing import Any, Optional
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import HttpUrl

from src.models.schemas import ProductCategory, ProductInfo
from src.utils.config import Settings, get_settings
from src.utils.logger import AgentLogger

# System prompt for the Product Understanding Agent
SYSTEM_PROMPT = """You are an expert product analyst specializing in consumer electronics,
home goods, and retail products. Your task is to analyze product information and extract
structured data that will be used for price comparison searches.

When given a product name or URL, you must:

1. **Identify the Product**: Extract the exact product name, brand, and model number
2. **Categorize**: Determine the product category (electronics, clothing, books, etc.)
3. **Extract Specifications**: Identify key technical specifications that differentiate
   this product from similar variants (storage size, color, screen size, etc.)
4. **Generate Search Queries**: Create 3-5 optimized search queries that will help find
   this exact product across different e-commerce sites. Include:
   - Exact product name with brand
   - Model number variations
   - Key specifications
   - Price-focused queries

**Important Guidelines**:
- Be precise with model numbers and variants (e.g., iPhone 15 Pro vs iPhone 15 Pro Max)
- Include storage/memory/size variations when relevant
- Consider regional naming differences
- Generate queries in both Hebrew and English if the product is commonly sold in Israel

**Output Format**:
Provide your analysis in the following JSON structure:
{
    "product_name": "Full normalized product name",
    "brand": "Brand name",
    "model": "Model number/name",
    "category": "One of: electronics, clothing, books, home_garden, sports, toys, health_beauty, automotive, food_grocery, other",
    "key_specs": {
        "spec_name": "spec_value"
    },
    "search_queries": [
        "query1",
        "query2",
        "query3"
    ]
}

Ensure your response is valid JSON that can be parsed."""


class ProductUnderstandingAgent:
    """Agent for analyzing product input and extracting structured information.

    This agent uses LLM reasoning to understand product information from
    various input formats (product names, URLs) and generates optimized
    search queries for finding the best prices.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[ChatOpenAI] = None,
    ) -> None:
        """Initialize the Product Understanding Agent.

        Args:
            settings: Application settings. If not provided, uses default settings.
            llm: LangChain LLM instance. If not provided, creates one from settings.
        """
        self.settings = settings or get_settings()
        self.logger = AgentLogger("ProductUnderstanding")

        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model=self.settings.openai.model,
                temperature=self.settings.openai.temperature,
                api_key=self.settings.openai.api_key,
                request_timeout=self.settings.openai.request_timeout,
            )

    async def analyze(self, user_input: str) -> ProductInfo:
        """Analyze user input and extract structured product information.

        Args:
            user_input: Product name or URL provided by the user.

        Returns:
            ProductInfo containing structured product data.

        Raises:
            ValueError: If the input cannot be processed.
        """
        self.logger.start("analyze_product", input_length=len(user_input))

        try:
            # Detect if input is a URL
            is_url = self._is_url(user_input)

            # Prepare the analysis prompt
            if is_url:
                prompt = self._prepare_url_prompt(user_input)
            else:
                prompt = self._prepare_product_name_prompt(user_input)

            # Get LLM analysis
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = await self.llm.ainvoke(messages)
            result = self._parse_response(response.content, user_input, is_url)

            self.logger.complete(
                "analyze_product",
                duration_ms=0,  # Would calculate actual duration
                product_name=result.product_name,
                query_count=len(result.search_queries),
            )

            return result

        except Exception as e:
            self.logger.error("analyze_product", e, input=user_input[:100])
            raise

    def _is_url(self, text: str) -> bool:
        """Check if the input is a URL.

        Args:
            text: Input text to check.

        Returns:
            True if the input appears to be a URL.
        """
        try:
            result = urlparse(text)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False

    def _prepare_url_prompt(self, url: str) -> str:
        """Prepare analysis prompt for URL input.

        Args:
            url: Product URL.

        Returns:
            Formatted prompt for LLM.
        """
        return f"""Analyze this product URL and extract product information:

URL: {url}

Based on the URL structure and any product identifiers visible in it,
determine the product details. Common patterns include:
- Amazon: /dp/ASIN or /gp/product/ASIN
- eBay: /itm/item-number
- Israeli sites (ksp.co.il, ivory.co.il): /web/item/SKU

Extract what you can from the URL and use your knowledge to fill in
likely product details. Generate search queries that will find this
same product on other sites."""

    def _prepare_product_name_prompt(self, product_name: str) -> str:
        """Prepare analysis prompt for product name input.

        Args:
            product_name: Product name or description.

        Returns:
            Formatted prompt for LLM.
        """
        return f"""Analyze this product and extract structured information:

Product: {product_name}

Identify the exact product, normalize the name, extract key specifications,
and generate optimized search queries for price comparison. Consider:
- Is this a specific variant? (storage size, color, etc.)
- Are there regional naming differences?
- What key specs differentiate this from similar products?

Generate search queries in both English and Hebrew if this product
is commonly sold in Israel."""

    def _parse_response(
        self, response_content: str, original_input: str, is_url: bool
    ) -> ProductInfo:
        """Parse LLM response into ProductInfo schema.

        Args:
            response_content: Raw LLM response content.
            original_input: Original user input.
            is_url: Whether the input was a URL.

        Returns:
            Parsed ProductInfo object.

        Raises:
            ValueError: If parsing fails.
        """
        import json

        # Extract JSON from response (may be wrapped in markdown code blocks)
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r"\{[\s\S]*\}", response_content)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError(f"Could not extract JSON from response: {response_content[:200]}")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        # Map category string to enum
        category_str = data.get("category", "other").lower().replace(" ", "_")
        try:
            category = ProductCategory(category_str)
        except ValueError:
            category = ProductCategory.OTHER

        # Build ProductInfo
        return ProductInfo(
            original_input=original_input,
            product_name=data.get("product_name", original_input),
            brand=data.get("brand"),
            model=data.get("model"),
            category=category,
            key_specs=data.get("key_specs", {}),
            search_queries=data.get("search_queries", [original_input]),
            is_url_input=is_url,
            source_url=HttpUrl(original_input) if is_url else None,
        )

    def _extract_from_url(self, url: str) -> dict[str, Any]:
        """Extract basic product info from URL patterns.

        Args:
            url: Product URL.

        Returns:
            Dictionary with extracted information.
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path

        info: dict[str, Any] = {
            "domain": domain,
            "potential_ids": [],
        }

        # Amazon ASIN extraction
        if "amazon" in domain:
            asin_match = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})", path)
            if asin_match:
                info["potential_ids"].append(("ASIN", asin_match.group(1)))

        # eBay item number
        elif "ebay" in domain:
            item_match = re.search(r"/itm/(\d+)", path)
            if item_match:
                info["potential_ids"].append(("eBay Item", item_match.group(1)))

        # Israeli sites
        elif any(site in domain for site in ["ksp.co.il", "ivory.co.il", "bug.co.il"]):
            sku_match = re.search(r"/(?:web/)?(?:item|product)/(\d+)", path)
            if sku_match:
                info["potential_ids"].append(("SKU", sku_match.group(1)))

        return info


# Factory function for creating the agent
def create_product_understanding_agent(
    settings: Optional[Settings] = None,
) -> ProductUnderstandingAgent:
    """Create a Product Understanding Agent instance.

    Args:
        settings: Optional application settings.

    Returns:
        Configured ProductUnderstandingAgent instance.
    """
    return ProductUnderstandingAgent(settings=settings)
