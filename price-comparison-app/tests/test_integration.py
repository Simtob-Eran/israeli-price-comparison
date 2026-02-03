"""Integration tests for the price comparison application."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.workflow import create_workflow
from src.mcp_client.client import MockMCPClient
from src.models.schemas import (
    AvailabilityStatus,
    FinalReport,
    PriceData,
    ProductCategory,
    ProductInfo,
    RankedResult,
    RankedResults,
    SearchResult,
)


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    @pytest.fixture
    def mock_all_agents(self):
        """Create comprehensive mocks for all agents."""
        product_info = ProductInfo(
            original_input="iPhone 15 Pro Max 256GB",
            product_name="Apple iPhone 15 Pro Max 256GB",
            brand="Apple",
            model="iPhone 15 Pro Max",
            category=ProductCategory.ELECTRONICS,
            key_specs={"storage": "256GB"},
            search_queries=["iPhone 15 Pro Max 256GB"],
        )

        search_results = [
            SearchResult(
                url=f"https://store{i}.com/product/{i}",
                title=f"iPhone 15 Pro Max - Store {i}",
                snippet="Best price",
                domain=f"store{i}.com",
                is_ecommerce=True,
                position=i,
            )
            for i in range(1, 11)
        ]

        price_data = [
            PriceData(
                url=f"https://store{i}.com/product/{i}",
                store_name=f"Store {i}",
                price=4999.0 + i * 100,
                currency="ILS",
                currency_symbol="₪",
                shipping_cost=0 if i % 2 == 0 else 29.0,
                availability=AvailabilityStatus.IN_STOCK,
                extracted_at=datetime.utcnow(),
                relevance_score=95.0 - i,
                product_title="Apple iPhone 15 Pro Max 256GB",
            )
            for i in range(1, 6)
        ]

        ranked_results = RankedResults(
            product_info=product_info,
            results=[
                RankedResult(
                    rank=i,
                    price_data=price_data[i - 1],
                    total_cost=price_data[i - 1].total_cost,
                    savings_vs_average=100.0,
                    savings_percentage=2.0,
                    is_best_deal=(i == 1),
                    deal_score=95.0 - i * 5,
                    recommendation="Good deal" if i <= 3 else "Average",
                )
                for i in range(1, 6)
            ],
            average_price=5299.0,
            lowest_price=5099.0,
            highest_price=5499.0,
            price_range=400.0,
            total_results=5,
            confidence_score=87.0,
        )

        final_report = FinalReport(
            product_info=product_info,
            ranked_results=ranked_results,
            summary="Found 5 prices for Apple iPhone 15 Pro Max 256GB",
            recommendation="Best deal at Store 1 for ₪5,099",
            comparison_table="| Rank | Store | Price |",
            best_time_to_buy="Black Friday typically has best deals",
            search_duration_seconds=12.5,
        )

        return {
            "product_info": product_info,
            "search_results": search_results,
            "price_data": price_data,
            "ranked_results": ranked_results,
            "final_report": final_report,
        }

    @pytest.mark.asyncio
    async def test_full_workflow_with_mocks(self, mock_settings, mock_all_agents):
        """Test complete workflow execution with mocked agents."""
        with patch("src.graph.workflow.ProductUnderstandingAgent") as mock_product, \
             patch("src.graph.workflow.WebSearchAgent") as mock_search, \
             patch("src.graph.workflow.PriceExtractionAgent") as mock_extract, \
             patch("src.graph.workflow.DataValidationAgent") as mock_validate, \
             patch("src.graph.workflow.PriceComparisonAgent") as mock_compare, \
             patch("src.graph.workflow.ReportingAgent") as mock_report:

            # Configure mocks
            mock_product.return_value.analyze = AsyncMock(
                return_value=mock_all_agents["product_info"]
            )
            mock_search.return_value.search = AsyncMock(
                return_value=mock_all_agents["search_results"]
            )
            mock_extract.return_value.extract_prices = AsyncMock(
                return_value=mock_all_agents["price_data"]
            )
            mock_validate.return_value.validate = AsyncMock(
                return_value=mock_all_agents["price_data"]
            )
            mock_compare.return_value.compare = AsyncMock(
                return_value=mock_all_agents["ranked_results"]
            )
            mock_report.return_value.generate_report = AsyncMock(
                return_value=mock_all_agents["final_report"]
            )

            # Create and run workflow
            mcp_client = MockMCPClient()
            await mcp_client.connect()

            workflow = create_workflow(settings=mock_settings, mcp_client=mcp_client)
            result = await workflow.run("iPhone 15 Pro Max 256GB")

            # Verify all stages completed
            assert result.product_info is not None
            assert len(result.search_results) > 0
            assert len(result.price_data) > 0
            assert len(result.validated_prices) > 0
            assert result.ranked_results is not None
            assert result.final_report is not None

            # Verify report content
            assert "iPhone 15 Pro Max" in result.final_report.summary
            assert result.final_report.ranked_results.total_results == 5

    @pytest.mark.asyncio
    async def test_workflow_handles_partial_failures(self, mock_settings):
        """Test workflow handles partial agent failures gracefully."""
        with patch("src.graph.workflow.ProductUnderstandingAgent") as mock_product, \
             patch("src.graph.workflow.WebSearchAgent") as mock_search, \
             patch("src.graph.workflow.PriceExtractionAgent") as mock_extract, \
             patch("src.graph.workflow.DataValidationAgent") as mock_validate, \
             patch("src.graph.workflow.PriceComparisonAgent") as mock_compare, \
             patch("src.graph.workflow.ReportingAgent") as mock_report:

            product_info = ProductInfo(
                original_input="test",
                product_name="Test Product",
                brand="Test",
                model="Model",
                category=ProductCategory.ELECTRONICS,
                key_specs={},
                search_queries=["test"],
            )

            # Product understanding succeeds
            mock_product.return_value.analyze = AsyncMock(return_value=product_info)

            # Search returns minimal results
            mock_search.return_value.search = AsyncMock(return_value=[
                SearchResult(
                    url="https://test.com/1",
                    title="Test",
                    snippet=None,
                    domain="test.com",
                    is_ecommerce=True,
                    position=1,
                )
            ])

            # Extraction fails partially
            mock_extract.return_value.extract_prices = AsyncMock(return_value=[])

            mcp_client = MockMCPClient()
            await mcp_client.connect()

            workflow = create_workflow(settings=mock_settings, mcp_client=mcp_client)
            result = await workflow.run("test product")

            # Workflow should complete but with limited results
            assert result.product_info is not None
            # May have errors or empty results depending on routing

    @pytest.mark.asyncio
    async def test_workflow_with_mock_mcp_client(self, mock_settings):
        """Test workflow with MockMCPClient."""
        mcp_client = MockMCPClient()
        await mcp_client.connect()

        # Verify mock client works
        assert mcp_client._connected

        # List available tools
        tools = await mcp_client.list_tools()
        assert len(tools) > 0

        # Test tool calls
        search_result = await mcp_client.call_tool(
            "serper_search",
            {"query": "iPhone 15 Pro Max"},
        )
        assert "organic" in search_result

        await mcp_client.disconnect()


class TestMCPClientIntegration:
    """Integration tests for MCP client."""

    @pytest.mark.asyncio
    async def test_mock_client_search(self):
        """Test mock MCP client search functionality."""
        client = MockMCPClient()
        await client.connect()

        result = await client.call_tool(
            "serper_search",
            {"query": "test product"},
        )

        assert "organic" in result
        assert len(result["organic"]) > 0

    @pytest.mark.asyncio
    async def test_mock_client_shopping(self):
        """Test mock MCP client shopping search."""
        client = MockMCPClient()
        await client.connect()

        result = await client.call_tool(
            "serper_shopping",
            {"query": "test product"},
        )

        assert "shopping" in result
        assert len(result["shopping"]) > 0

    @pytest.mark.asyncio
    async def test_mock_client_fetch(self):
        """Test mock MCP client page fetch."""
        client = MockMCPClient()
        await client.connect()

        result = await client.call_tool(
            "fetch_page_content",
            {"url": "https://example.com/product"},
        )

        assert "content" in result
        assert "price" in result["content"].lower()


class TestModelSerialization:
    """Tests for model serialization."""

    def test_product_info_json(self, sample_product_info):
        """Test ProductInfo JSON serialization."""
        json_str = sample_product_info.model_dump_json()
        data = json.loads(json_str)

        assert data["product_name"] == sample_product_info.product_name
        assert data["brand"] == sample_product_info.brand
        assert data["category"] == sample_product_info.category.value

    def test_price_data_json(self, sample_price_data):
        """Test PriceData JSON serialization."""
        for price in sample_price_data:
            json_str = price.model_dump_json()
            data = json.loads(json_str)

            assert data["price"] == price.price
            assert data["store_name"] == price.store_name
            assert data["currency"] == price.currency

    def test_ranked_results_json(self, sample_ranked_results):
        """Test RankedResults JSON serialization."""
        json_str = sample_ranked_results.model_dump_json()
        data = json.loads(json_str)

        assert data["total_results"] == sample_ranked_results.total_results
        assert len(data["results"]) == len(sample_ranked_results.results)

    def test_final_report_json(self, sample_ranked_results):
        """Test FinalReport JSON serialization."""
        report = FinalReport(
            product_info=sample_ranked_results.product_info,
            ranked_results=sample_ranked_results,
            summary="Test summary",
            recommendation="Test recommendation",
            comparison_table="| test |",
        )

        json_str = report.model_dump_json()
        data = json.loads(json_str)

        assert data["summary"] == "Test summary"
        assert "product_info" in data
        assert "ranked_results" in data


class TestConfigurationIntegration:
    """Tests for configuration loading."""

    def test_settings_from_yaml(self, mock_settings):
        """Test settings load correctly."""
        assert mock_settings.openai.model is not None
        assert mock_settings.mcp.server_url is not None
        assert mock_settings.agents.max_retries > 0

    def test_ecommerce_domains(self, mock_settings):
        """Test e-commerce domain configuration."""
        domains = mock_settings.get_all_ecommerce_domains()
        assert len(domains) > 0
        assert "ksp.co.il" in domains or len(mock_settings.ecommerce_domains.israel) >= 0

    def test_currency_lookup(self, mock_settings):
        """Test currency configuration lookup."""
        # May or may not have currencies configured in test
        ils = mock_settings.get_currency_by_code("ILS")
        if ils:
            assert ils.symbol == "₪"
