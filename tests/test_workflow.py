"""Unit tests for the LangGraph workflow."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.state import GraphState, create_initial_state
from src.graph.workflow import PriceComparisonWorkflow, create_workflow
from src.models.schemas import (
    AgentError,
    AvailabilityStatus,
    PriceData,
    ProductCategory,
    ProductInfo,
    RankedResult,
    RankedResults,
    SearchResult,
)


class TestGraphState:
    """Tests for GraphState."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state("iPhone 15 Pro Max")

        assert state.input == "iPhone 15 Pro Max"
        assert state.product_info is None
        assert state.search_results == []
        assert state.price_data == []
        assert state.errors == []
        assert state.retry_count == 0

    def test_state_properties(self, sample_product_info, sample_search_results, sample_price_data):
        """Test state property calculations."""
        state = GraphState(
            input="test",
            product_info=sample_product_info,
            search_results=sample_search_results,
            price_data=sample_price_data,
            validated_prices=sample_price_data[:2],
        )

        assert state.search_result_count == 3
        assert state.price_count == 3
        assert state.validated_count == 2
        assert not state.has_errors

    def test_state_with_errors(self):
        """Test state error handling."""
        error = AgentError(
            agent_name="TestAgent",
            error_type="TestError",
            error_message="Test error message",
            recoverable=True,
        )

        state = GraphState(
            input="test",
            errors=[error],
        )

        assert state.has_errors
        assert len(state.errors) == 1

    def test_state_summary(self, sample_product_info):
        """Test state summary generation."""
        state = GraphState(
            input="iPhone 15 Pro Max 256GB test query",
            product_info=sample_product_info,
        )

        summary = state.to_summary()

        assert "input" in summary
        assert summary["product_identified"]
        assert summary["error_count"] == 0


class TestPriceComparisonWorkflow:
    """Tests for PriceComparisonWorkflow."""

    @pytest.fixture
    def mock_workflow(self, mock_settings, mock_mcp_client):
        """Create workflow with mocked dependencies."""
        with patch("src.graph.workflow.ProductUnderstandingAgent") as mock_product, \
             patch("src.graph.workflow.WebSearchAgent") as mock_search, \
             patch("src.graph.workflow.PriceExtractionAgent") as mock_extract, \
             patch("src.graph.workflow.DataValidationAgent") as mock_validate, \
             patch("src.graph.workflow.PriceComparisonAgent") as mock_compare, \
             patch("src.graph.workflow.ReportingAgent") as mock_report:

            # Setup mock agents
            mock_product.return_value.analyze = AsyncMock()
            mock_search.return_value.search = AsyncMock()
            mock_extract.return_value.extract_prices = AsyncMock()
            mock_validate.return_value.validate = AsyncMock()
            mock_compare.return_value.compare = AsyncMock()
            mock_report.return_value.generate_report = AsyncMock()

            workflow = PriceComparisonWorkflow(
                settings=mock_settings,
                mcp_client=mock_mcp_client,
            )

            yield workflow, {
                "product": mock_product.return_value,
                "search": mock_search.return_value,
                "extract": mock_extract.return_value,
                "validate": mock_validate.return_value,
                "compare": mock_compare.return_value,
                "report": mock_report.return_value,
            }

    def test_workflow_initialization(self, mock_settings, mock_mcp_client):
        """Test workflow initializes correctly."""
        workflow = create_workflow(settings=mock_settings, mcp_client=mock_mcp_client)

        assert workflow is not None
        assert workflow.graph is not None

    def test_routing_after_search_success(self, mock_settings, mock_mcp_client):
        """Test routing after successful search."""
        workflow = create_workflow(settings=mock_settings, mcp_client=mock_mcp_client)

        state = GraphState(
            input="test",
            search_results=[
                SearchResult(
                    url=f"https://test{i}.com",
                    title=f"Test {i}",
                    snippet=None,
                    domain=f"test{i}.com",
                    is_ecommerce=True,
                    position=i,
                )
                for i in range(10)
            ],
        )

        result = workflow._after_search_routing(state)
        assert result == "continue"

    def test_routing_after_search_retry(self, mock_settings, mock_mcp_client):
        """Test routing triggers retry when not enough results."""
        workflow = create_workflow(settings=mock_settings, mcp_client=mock_mcp_client)

        state = GraphState(
            input="test",
            search_results=[
                SearchResult(
                    url="https://test.com",
                    title="Test",
                    snippet=None,
                    domain="test.com",
                    is_ecommerce=True,
                    position=1,
                )
            ],
            retry_count=0,
        )

        result = workflow._after_search_routing(state)
        assert result == "retry"

    def test_routing_after_search_end(self, mock_settings, mock_mcp_client):
        """Test routing ends when no results and max retries."""
        workflow = create_workflow(settings=mock_settings, mcp_client=mock_mcp_client)

        state = GraphState(
            input="test",
            search_results=[],
            retry_count=3,  # Max retries exceeded
        )

        result = workflow._after_search_routing(state)
        assert result == "end"

    def test_routing_after_extraction_continue(self, mock_settings, mock_mcp_client):
        """Test routing continues after extraction with enough prices."""
        workflow = create_workflow(settings=mock_settings, mcp_client=mock_mcp_client)

        state = GraphState(
            input="test",
            price_data=[
                PriceData(
                    url=f"https://test{i}.com",
                    store_name=f"Store {i}",
                    price=1000.0 + i * 100,
                    currency="ILS",
                    currency_symbol="₪",
                    availability=AvailabilityStatus.IN_STOCK,
                    extracted_at=datetime.utcnow(),
                )
                for i in range(5)
            ],
        )

        result = workflow._after_extraction_routing(state)
        assert result == "continue"

    def test_routing_after_validation_continue(self, mock_settings, mock_mcp_client):
        """Test routing continues after validation with results."""
        workflow = create_workflow(settings=mock_settings, mcp_client=mock_mcp_client)

        state = GraphState(
            input="test",
            validated_prices=[
                PriceData(
                    url="https://test.com",
                    store_name="Store",
                    price=1000.0,
                    currency="ILS",
                    currency_symbol="₪",
                    availability=AvailabilityStatus.IN_STOCK,
                    extracted_at=datetime.utcnow(),
                )
            ],
        )

        result = workflow._after_validation_routing(state)
        assert result == "continue"

    def test_routing_handles_unrecoverable_errors(self, mock_settings, mock_mcp_client):
        """Test routing ends on unrecoverable errors."""
        workflow = create_workflow(settings=mock_settings, mcp_client=mock_mcp_client)

        state = GraphState(
            input="test",
            errors=[
                AgentError(
                    agent_name="TestAgent",
                    error_type="Fatal",
                    error_message="Fatal error",
                    recoverable=False,
                )
            ],
        )

        # Should end regardless of other state
        assert workflow._after_search_routing(state) == "end"
        assert workflow._after_extraction_routing(state) == "end"
        assert workflow._after_validation_routing(state) == "end"


class TestWorkflowNodes:
    """Tests for individual workflow nodes."""

    @pytest.fixture
    def workflow(self, mock_settings, mock_mcp_client):
        """Create workflow for testing nodes."""
        return PriceComparisonWorkflow(
            settings=mock_settings,
            mcp_client=mock_mcp_client,
        )

    @pytest.mark.asyncio
    async def test_product_understanding_node_success(
        self, workflow, sample_product_info
    ):
        """Test product understanding node success."""
        workflow.product_agent.analyze = AsyncMock(return_value=sample_product_info)

        state = GraphState(input="iPhone 15 Pro Max")
        update = await workflow._product_understanding_node(state)

        assert "product_info" in update
        assert update["product_info"].product_name == sample_product_info.product_name

    @pytest.mark.asyncio
    async def test_product_understanding_node_error(self, workflow):
        """Test product understanding node error handling."""
        workflow.product_agent.analyze = AsyncMock(
            side_effect=Exception("API Error")
        )

        state = GraphState(input="test")
        update = await workflow._product_understanding_node(state)

        assert "errors" in update
        assert len(update["errors"]) == 1

    @pytest.mark.asyncio
    async def test_web_search_node_success(
        self, workflow, sample_product_info, sample_search_results
    ):
        """Test web search node success."""
        workflow.search_agent.search = AsyncMock(return_value=sample_search_results)

        state = GraphState(input="test", product_info=sample_product_info)
        update = await workflow._web_search_node(state)

        assert "search_results" in update
        assert len(update["search_results"]) == 3

    @pytest.mark.asyncio
    async def test_web_search_node_missing_input(self, workflow):
        """Test web search node with missing product info."""
        state = GraphState(input="test", product_info=None)
        update = await workflow._web_search_node(state)

        assert "errors" in update

    @pytest.mark.asyncio
    async def test_price_extraction_node_success(
        self, workflow, sample_search_results, sample_price_data
    ):
        """Test price extraction node success."""
        workflow.extraction_agent.extract_prices = AsyncMock(
            return_value=sample_price_data
        )

        state = GraphState(input="test", search_results=sample_search_results)
        update = await workflow._price_extraction_node(state)

        assert "price_data" in update
        assert len(update["price_data"]) == 3

    @pytest.mark.asyncio
    async def test_data_validation_node_success(
        self, workflow, sample_product_info, sample_price_data
    ):
        """Test data validation node success."""
        validated = sample_price_data[:2]
        workflow.validation_agent.validate = AsyncMock(return_value=validated)

        state = GraphState(
            input="test",
            product_info=sample_product_info,
            price_data=sample_price_data,
        )
        update = await workflow._data_validation_node(state)

        assert "validated_prices" in update
        assert len(update["validated_prices"]) == 2

    @pytest.mark.asyncio
    async def test_price_comparison_node_success(
        self, workflow, sample_product_info, sample_price_data, sample_ranked_results
    ):
        """Test price comparison node success."""
        workflow.comparison_agent.compare = AsyncMock(
            return_value=sample_ranked_results
        )

        state = GraphState(
            input="test",
            product_info=sample_product_info,
            validated_prices=sample_price_data,
        )
        update = await workflow._price_comparison_node(state)

        assert "ranked_results" in update
        assert update["ranked_results"].total_results == 3

    @pytest.mark.asyncio
    async def test_reporting_node_success(
        self, workflow, sample_ranked_results
    ):
        """Test reporting node success."""
        from src.models.schemas import FinalReport

        mock_report = FinalReport(
            product_info=sample_ranked_results.product_info,
            ranked_results=sample_ranked_results,
            summary="Test summary",
            recommendation="Test recommendation",
            comparison_table="| test |",
        )
        workflow.reporting_agent.generate_report = AsyncMock(return_value=mock_report)

        state = GraphState(
            input="test",
            ranked_results=sample_ranked_results,
        )
        update = await workflow._reporting_node(state)

        assert "final_report" in update
        assert update["final_report"].summary == "Test summary"
