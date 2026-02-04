"""Web Search Agent.

This agent executes intelligent web searches to find product listings
across multiple e-commerce platforms using MCP tools.
"""

import asyncio
from typing import Any, Optional
from urllib.parse import urlparse

from langchain.agents import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.mcp_client.client import MCPClient, create_mcp_tools
from src.models.schemas import ProductInfo, SearchResult
from src.utils.config import Settings, get_settings
from src.utils.logger import AgentLogger

# System prompt for the Web Search Agent
SYSTEM_PROMPT = """You are an expert web researcher specializing in e-commerce and price comparison.
Your task is to search the web to find product listings for price comparison.

**Your Responsibilities**:
1. Execute multiple search strategies to maximize coverage
2. Use both general web search and shopping-specific search
3. Filter results to e-commerce sites only
4. Collect diverse sources (Israeli and international retailers)
5. Handle search failures gracefully with alternative queries

**Search Strategies to Use**:
1. **Exact Match**: Search for the exact product name with brand
2. **Model Number**: Search using model number/SKU
3. **Shopping Search**: Use shopping-specific search for price comparisons
4. **Hebrew Search**: For Israeli market, search in Hebrew
5. **Price Focus**: Include "price" or "מחיר" in queries

**E-commerce Sites to Prioritize (Israel)**:
- zap.co.il (price comparison)
- ksp.co.il
- ivory.co.il
- bug.co.il
- eilat.co.il
- lastprice.co.il

**International Sites**:
- amazon.com
- ebay.com
- walmart.com
- aliexpress.com
- newegg.com

**Output Requirements**:
- Return at least 10 URLs, maximum 30
- Include direct product page links (not search result pages)
- Avoid duplicate URLs
- Prefer sites known for competitive pricing

When using tools:
- serper_search: For general web searches
- serper_shopping: For shopping-specific results

Always explain your search strategy and summarize the results found."""


class SearchQuery(BaseModel):
    """Model for a search query."""

    query: str = Field(..., description="The search query string")
    search_type: str = Field(
        default="web", description="Type of search: web or shopping"
    )
    language: str = Field(default="en", description="Search language")


class WebSearchAgent:
    """Agent for executing intelligent web searches for product listings.

    This agent uses MCP tools (Serper API) to search the web and find
    product listings across multiple e-commerce platforms.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[ChatOpenAI] = None,
        mcp_client: Optional[MCPClient] = None,
    ) -> None:
        """Initialize the Web Search Agent.

        Args:
            settings: Application settings.
            llm: LangChain LLM instance.
            mcp_client: Connected MCP client for tool access.
        """
        self.settings = settings or get_settings()
        self.logger = AgentLogger("WebSearch")
        self.mcp_client = mcp_client

        # Known e-commerce domains for filtering
        self.ecommerce_domains = set(self.settings.get_all_ecommerce_domains())

        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model=self.settings.openai.model,
                temperature=self.settings.openai.temperature,
                api_key=self.settings.openai.api_key,
                request_timeout=self.settings.openai.request_timeout,
            )

        # Configuration
        self.min_urls = self.settings.agents.web_search.min_urls or 10
        self.max_urls = self.settings.agents.web_search.max_urls or 30

    async def search(self, product_info: ProductInfo) -> list[SearchResult]:
        """Search the web for product listings.

        Args:
            product_info: Structured product information from the understanding agent.

        Returns:
            List of SearchResult objects with candidate URLs.
        """
        self.logger.start(
            "web_search",
            product=product_info.product_name,
            query_count=len(product_info.search_queries),
        )

        all_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        try:
            # Execute searches concurrently
            search_tasks = []

            for query in product_info.search_queries:
                # Web search
                search_tasks.append(self._execute_search(query, "web"))
                # Shopping search
                search_tasks.append(self._execute_search(query, "shopping"))

            # Add Hebrew searches if product name suggests it's relevant
            hebrew_query = self._generate_hebrew_query(product_info)
            if hebrew_query:
                search_tasks.append(self._execute_search(hebrew_query, "web"))
                search_tasks.append(self._execute_search(hebrew_query, "shopping"))

            # Execute all searches concurrently
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.warning(
                        "Search task failed",
                        error=str(result),
                    )
                    continue

                for search_result in result:
                    if search_result.url not in seen_urls:
                        seen_urls.add(str(search_result.url))
                        all_results.append(search_result)

                        if len(all_results) >= self.max_urls:
                            break

            # Filter to e-commerce sites
            filtered_results = self._filter_ecommerce_results(all_results)

            self.logger.complete(
                "web_search",
                duration_ms=0,
                total_results=len(all_results),
                filtered_results=len(filtered_results),
            )

            return filtered_results[:self.max_urls]

        except Exception as e:
            self.logger.error("web_search", e, product=product_info.product_name)
            raise

    async def _execute_search(
        self, query: str, search_type: str
    ) -> list[SearchResult]:
        """Execute a single search query.

        Args:
            query: Search query string.
            search_type: Type of search (web or shopping).

        Returns:
            List of SearchResult objects.
        """
        results: list[SearchResult] = []

        try:
            if self.mcp_client:
                # Use MCP tools for search
                tool_name = "serper_shopping" if search_type == "shopping" else "serper_search"
                response = await self.mcp_client.call_tool(
                    tool_name=tool_name,
                    arguments={"query": query, "num_results": 20},
                )
                results = self._parse_search_response(response, search_type)
            else:
                # Fallback: use LLM to simulate search (for testing)
                results = await self._simulate_search(query, search_type)

            self.logger.debug(
                f"Search completed",
                query=query,
                type=search_type,
                result_count=len(results),
            )

        except Exception as e:
            self.logger.warning(
                f"Search failed",
                query=query,
                type=search_type,
                error=str(e),
            )

        return results

    def _parse_search_response(
        self, response: dict[str, Any], search_type: str
    ) -> list[SearchResult]:
        """Parse search API response into SearchResult objects.

        Args:
            response: Raw API response.
            search_type: Type of search performed.

        Returns:
            List of SearchResult objects.
        """
        results: list[SearchResult] = []

        if search_type == "shopping":
            items = response.get("shopping", [])
        else:
            items = response.get("organic", [])

        for idx, item in enumerate(items, start=1):
            try:
                url = item.get("link", "")
                if not url:
                    continue

                domain = urlparse(url).netloc.lower()
                domain = domain.replace("www.", "")

                result = SearchResult(
                    url=url,
                    title=item.get("title", "Unknown"),
                    snippet=item.get("snippet") or item.get("description"),
                    domain=domain,
                    is_ecommerce=self._is_ecommerce_domain(domain),
                    position=idx,
                )
                results.append(result)

            except Exception as e:
                self.logger.debug(f"Failed to parse search item: {e}")
                continue

        return results

    def _is_ecommerce_domain(self, domain: str) -> bool:
        """Check if a domain is a known e-commerce site.

        Args:
            domain: Domain name to check.

        Returns:
            True if the domain is e-commerce.
        """
        # Remove www. prefix
        domain = domain.replace("www.", "")

        # Check against known domains
        if domain in self.ecommerce_domains:
            return True

        # Check for common e-commerce patterns
        ecommerce_patterns = [
            "shop", "store", "buy", "price", "deal",
            "amazon", "ebay", "walmart", "alibaba", "aliexpress",
        ]
        return any(pattern in domain for pattern in ecommerce_patterns)

    def _filter_ecommerce_results(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Filter results to only include e-commerce sites.

        Args:
            results: All search results.

        Returns:
            Filtered list of e-commerce results.
        """
        # First, prioritize known e-commerce domains
        ecommerce_results = [r for r in results if r.is_ecommerce]

        # If not enough e-commerce results, include others that might be relevant
        if len(ecommerce_results) < self.min_urls:
            # Add results from domains that might be e-commerce
            potential_ecommerce = [
                r for r in results
                if not r.is_ecommerce and self._might_be_product_page(r)
            ]
            ecommerce_results.extend(potential_ecommerce)

        return ecommerce_results

    def _might_be_product_page(self, result: SearchResult) -> bool:
        """Heuristic check if a result might be a product page.

        Args:
            result: Search result to check.

        Returns:
            True if the result might be a product page.
        """
        # Check URL patterns
        url_lower = str(result.url).lower()
        product_patterns = [
            "/product/", "/item/", "/p/", "/dp/",
            "/goods/", "/buy/", "sku=", "pid=",
        ]
        if any(pattern in url_lower for pattern in product_patterns):
            return True

        # Check title patterns
        title_lower = result.title.lower() if result.title else ""
        price_indicators = ["₪", "$", "€", "price", "buy", "מחיר", "קנה"]
        if any(indicator in title_lower for indicator in price_indicators):
            return True

        return False

    def _generate_hebrew_query(self, product_info: ProductInfo) -> Optional[str]:
        """Generate Hebrew search query if appropriate.

        Args:
            product_info: Product information.

        Returns:
            Hebrew query string or None.
        """
        # For now, add "מחיר" (price) prefix to product name
        # In production, this could use translation
        product_name = product_info.product_name

        # Check if any query already contains Hebrew
        for query in product_info.search_queries:
            if any("\u0590" <= c <= "\u05FF" for c in query):
                return None  # Already has Hebrew query

        return f"מחיר {product_name}"

    async def _simulate_search(
        self, query: str, search_type: str
    ) -> list[SearchResult]:
        """Simulate search results for testing without MCP.

        Args:
            query: Search query.
            search_type: Type of search.

        Returns:
            Simulated search results.
        """
        # This is a fallback for testing - returns empty results
        self.logger.warning(
            "Using simulated search (no MCP client)",
            query=query,
        )
        return []


# Factory function
def create_web_search_agent(
    settings: Optional[Settings] = None,
    mcp_client: Optional[MCPClient] = None,
) -> WebSearchAgent:
    """Create a Web Search Agent instance.

    Args:
        settings: Optional application settings.
        mcp_client: Optional MCP client for tool access.

    Returns:
        Configured WebSearchAgent instance.
    """
    return WebSearchAgent(settings=settings, mcp_client=mcp_client)
