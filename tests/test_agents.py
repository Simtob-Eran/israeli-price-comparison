"""Unit tests for price comparison agents."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.data_validation import DataValidationAgent
from src.agents.price_comparison import PriceComparisonAgent
from src.agents.price_extraction import PriceExtractionAgent
from src.agents.product_understanding import ProductUnderstandingAgent
from src.agents.reporting import ReportingAgent
from src.agents.web_search import WebSearchAgent
from src.models.schemas import (
    AvailabilityStatus,
    PriceData,
    ProductCategory,
    ProductInfo,
    SearchResult,
)


class TestProductUnderstandingAgent:
    """Tests for ProductUnderstandingAgent."""

    @pytest.fixture
    def agent(self, mock_settings, mock_llm):
        """Create agent with mocked dependencies."""
        return ProductUnderstandingAgent(settings=mock_settings, llm=mock_llm)

    def test_is_url_detection(self, agent):
        """Test URL detection."""
        assert agent._is_url("https://www.amazon.com/dp/B0123")
        assert agent._is_url("http://ksp.co.il/product/123")
        assert not agent._is_url("iPhone 15 Pro Max")
        assert not agent._is_url("MacBook Pro M3 2024")

    @pytest.mark.asyncio
    async def test_analyze_product_name(self, agent, mock_llm):
        """Test analyzing product name input."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "product_name": "Apple iPhone 15 Pro Max 256GB",
            "brand": "Apple",
            "model": "iPhone 15 Pro Max",
            "category": "electronics",
            "key_specs": {"storage": "256GB"},
            "search_queries": ["iPhone 15 Pro Max 256GB"],
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await agent.analyze("iPhone 15 Pro Max 256GB")

        assert result.product_name == "Apple iPhone 15 Pro Max 256GB"
        assert result.brand == "Apple"
        assert result.category == ProductCategory.ELECTRONICS
        assert len(result.search_queries) >= 1

    @pytest.mark.asyncio
    async def test_analyze_url_input(self, agent, mock_llm):
        """Test analyzing URL input."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "product_name": "Test Product",
            "brand": "Test Brand",
            "model": "Test Model",
            "category": "electronics",
            "key_specs": {},
            "search_queries": ["Test Product"],
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await agent.analyze("https://www.amazon.com/dp/B0123456")

        assert result.is_url_input
        assert result.source_url is not None


class TestWebSearchAgent:
    """Tests for WebSearchAgent."""

    @pytest.fixture
    def agent(self, mock_settings, mock_mcp_client):
        """Create agent with mocked dependencies."""
        return WebSearchAgent(settings=mock_settings, mcp_client=mock_mcp_client)

    def test_is_ecommerce_domain(self, agent):
        """Test e-commerce domain detection."""
        assert agent._is_ecommerce_domain("ksp.co.il")
        assert agent._is_ecommerce_domain("amazon.com")
        assert agent._is_ecommerce_domain("shop.example.com")
        # Non-ecommerce should still potentially be included
        assert not agent._is_ecommerce_domain("wikipedia.org")

    @pytest.mark.asyncio
    async def test_search_with_product_info(
        self, agent, sample_product_info, mock_mcp_client, mock_search_response
    ):
        """Test search execution with product info."""
        mock_mcp_client.call_tool = AsyncMock(return_value=mock_search_response)

        results = await agent.search(sample_product_info)

        assert isinstance(results, list)
        # Should have called MCP tools
        assert mock_mcp_client.call_tool.called

    def test_parse_search_response(self, agent, mock_search_response):
        """Test parsing of search response."""
        results = agent._parse_search_response(mock_search_response, "web")

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].domain == "ksp.co.il"

    def test_might_be_product_page(self, agent):
        """Test product page detection heuristics."""
        product_result = SearchResult(
            url="https://store.com/product/123",
            title="iPhone 15 Pro - Buy Now",
            snippet=None,
            domain="store.com",
            is_ecommerce=True,
            position=1,
        )
        assert agent._might_be_product_page(product_result)

        blog_result = SearchResult(
            url="https://blog.com/article/review",
            title="Review of smartphones",
            snippet=None,
            domain="blog.com",
            is_ecommerce=False,
            position=2,
        )
        assert not agent._might_be_product_page(blog_result)


class TestPriceExtractionAgent:
    """Tests for PriceExtractionAgent."""

    @pytest.fixture
    def agent(self, mock_settings, mock_mcp_client):
        """Create agent with mocked dependencies."""
        return PriceExtractionAgent(settings=mock_settings, mcp_client=mock_mcp_client)

    def test_parse_availability(self, agent):
        """Test availability status parsing."""
        assert agent._parse_availability("InStock") == AvailabilityStatus.IN_STOCK
        assert agent._parse_availability("OutOfStock") == AvailabilityStatus.OUT_OF_STOCK
        assert agent._parse_availability("Limited") == AvailabilityStatus.LIMITED
        assert agent._parse_availability("unknown") == AvailabilityStatus.UNKNOWN

    def test_get_currency_symbol(self, agent):
        """Test currency symbol mapping."""
        assert agent._get_currency_symbol("ILS") == "₪"
        assert agent._get_currency_symbol("USD") == "$"
        assert agent._get_currency_symbol("EUR") == "€"
        assert agent._get_currency_symbol("GBP") == "£"

    def test_smart_truncate(self, agent):
        """Test HTML truncation with price preservation."""
        html = """
        <html>
        <script>console.log('test');</script>
        <style>.class { color: red; }</style>
        <body>
        <div>Some content</div>
        <span class="price">₪4,999</span>
        <div>More content that should be preserved</div>
        </body>
        </html>
        """
        truncated = agent._smart_truncate(html, 500)

        # Should keep price-related content
        assert "₪4,999" in truncated or "price" in truncated.lower()
        # Should remove scripts
        assert "console.log" not in truncated

    def test_is_product_schema(self, agent):
        """Test Schema.org product detection."""
        product_data = {"@type": "Product", "name": "Test"}
        assert agent._is_product_schema(product_data)

        list_data = {"@type": ["Product", "Thing"], "name": "Test"}
        assert agent._is_product_schema(list_data)

        non_product = {"@type": "Organization", "name": "Test"}
        assert not agent._is_product_schema(non_product)


class TestDataValidationAgent:
    """Tests for DataValidationAgent."""

    @pytest.fixture
    def agent(self, mock_settings, mock_llm):
        """Create agent with mocked dependencies."""
        return DataValidationAgent(settings=mock_settings, llm=mock_llm)

    def test_name_similarity_score(self, agent):
        """Test name similarity calculation."""
        # Identical names
        score = agent._name_similarity_score(
            "iPhone 15 Pro Max", "iPhone 15 Pro Max"
        )
        assert score == 1.0

        # Similar names
        score = agent._name_similarity_score(
            "iPhone 15 Pro Max 256GB", "iPhone 15 Pro Max"
        )
        assert 0.5 < score < 1.0

        # Different names
        score = agent._name_similarity_score(
            "iPhone 15 Pro Max", "Samsung Galaxy S24"
        )
        assert score < 0.3

    def test_statistical_validation(self, agent, sample_price_data):
        """Test statistical outlier detection."""
        # Add an outlier
        outlier = PriceData(
            url="https://scam.com/product",
            store_name="Scam Store",
            price=999.0,  # Suspiciously low
            currency="ILS",
            currency_symbol="₪",
            availability=AvailabilityStatus.IN_STOCK,
            extracted_at=datetime.utcnow(),
            relevance_score=90.0,
        )
        prices_with_outlier = sample_price_data + [outlier]

        validated = agent._statistical_validation(prices_with_outlier)

        # Outlier should have reduced relevance score
        outlier_validated = next(
            (p for p in validated if p.store_name == "Scam Store"), None
        )
        assert outlier_validated is not None
        assert outlier_validated.relevance_score < 90.0

    @pytest.mark.asyncio
    async def test_validate_prices(
        self, agent, sample_price_data, sample_product_info, mock_llm
    ):
        """Test price validation flow."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "is_valid": True,
            "relevance_score": 90,
            "confidence": 0.9,
            "issues": [],
            "matched_specs": {"brand": True, "model": True},
            "is_suspicious_price": False,
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        validated = await agent.validate(sample_price_data, sample_product_info)

        assert len(validated) > 0
        assert all(p.relevance_score is not None for p in validated)


class TestPriceComparisonAgent:
    """Tests for PriceComparisonAgent."""

    @pytest.fixture
    def agent(self, mock_settings):
        """Create agent."""
        return PriceComparisonAgent(settings=mock_settings)

    def test_calculate_statistics(self, agent, sample_price_data):
        """Test price statistics calculation."""
        stats = agent._calculate_statistics(sample_price_data)

        assert stats["min"] is not None
        assert stats["max"] is not None
        assert stats["average"] is not None
        assert stats["min"] <= stats["average"] <= stats["max"]

    def test_calculate_deal_score(self, agent, sample_price_data):
        """Test deal score calculation."""
        stats = agent._calculate_statistics(sample_price_data)

        # Best price should have higher score
        best = sample_price_data[0]  # Lowest price
        worst = sample_price_data[1]  # Higher price

        best_score = agent._calculate_deal_score(best, stats, 1, 3)
        worst_score = agent._calculate_deal_score(worst, stats, 2, 3)

        assert best_score >= worst_score

    @pytest.mark.asyncio
    async def test_compare_prices(
        self, agent, sample_price_data, sample_product_info
    ):
        """Test price comparison."""
        result = await agent.compare(sample_price_data, sample_product_info)

        assert result.total_results == len(sample_price_data)
        assert result.lowest_price <= result.highest_price
        assert len(result.results) == len(sample_price_data)
        # Results should be sorted by total cost
        for i in range(len(result.results) - 1):
            assert result.results[i].total_cost <= result.results[i + 1].total_cost

    def test_generate_recommendation(self, agent, sample_price_data):
        """Test recommendation generation."""
        recommendation = agent._generate_recommendation(
            sample_price_data[0], 95.0, 1
        )

        assert len(recommendation) > 0
        assert "deal" in recommendation.lower() or "price" in recommendation.lower()


class TestReportingAgent:
    """Tests for ReportingAgent."""

    @pytest.fixture
    def agent(self, mock_settings, mock_llm):
        """Create agent with mocked dependencies."""
        return ReportingAgent(settings=mock_settings, llm=mock_llm)

    def test_generate_comparison_table(self, agent, sample_ranked_results):
        """Test comparison table generation."""
        table = agent._generate_comparison_table(sample_ranked_results)

        assert "|" in table  # Has table structure
        assert "Rank" in table or "rank" in table.lower()
        assert "₪" in table or "Price" in table

    @pytest.mark.asyncio
    async def test_generate_report(self, agent, sample_ranked_results, mock_llm):
        """Test full report generation."""
        mock_response = MagicMock()
        mock_response.content = "Summary: Found great deals. Recommendation: Buy from KSP."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        report = await agent.generate_report(sample_ranked_results)

        assert report.product_info is not None
        assert len(report.summary) > 0
        assert len(report.comparison_table) > 0

    def test_export_csv(self, agent, sample_ranked_results):
        """Test CSV export."""
        # First generate report
        from src.models.schemas import FinalReport

        report = FinalReport(
            product_info=sample_ranked_results.product_info,
            ranked_results=sample_ranked_results,
            summary="Test summary",
            recommendation="Test recommendation",
            comparison_table="| test |",
        )

        csv = agent.export_csv(report)

        assert "rank,store,price" in csv.lower()
        assert "KSP" in csv or "ksp" in csv.lower()

    def test_format_for_cli(self, agent, sample_ranked_results):
        """Test CLI formatting."""
        from src.models.schemas import FinalReport

        report = FinalReport(
            product_info=sample_ranked_results.product_info,
            ranked_results=sample_ranked_results,
            summary="Test summary",
            recommendation="Test recommendation",
            comparison_table="| test |",
        )

        cli_output = agent.format_for_cli(report)

        assert "PRICE COMPARISON REPORT" in cli_output
        assert "SUMMARY" in cli_output
        assert "RECOMMENDATION" in cli_output
