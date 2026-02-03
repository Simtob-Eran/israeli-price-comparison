"""LangGraph workflow implementation for price comparison.

This module defines the workflow graph that orchestrates all agents
to perform end-to-end price comparison searches.
"""

import time
from datetime import datetime
from typing import Any, Literal, Optional

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.data_validation import DataValidationAgent
from src.agents.price_comparison import PriceComparisonAgent
from src.agents.price_extraction import PriceExtractionAgent
from src.agents.product_understanding import ProductUnderstandingAgent
from src.agents.reporting import ReportingAgent
from src.agents.web_search import WebSearchAgent
from src.graph.state import GraphState, StateUpdate, create_initial_state
from src.mcp_client.client import MCPClient
from src.models.schemas import AgentError
from src.utils.config import Settings, get_settings
from src.utils.logger import WorkflowLogger


class PriceComparisonWorkflow:
    """Orchestrates the price comparison workflow using LangGraph.

    This class builds and manages the LangGraph workflow that coordinates
    all agents to perform end-to-end price comparison.

    Architecture Diagram:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Price Comparison Workflow                     │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  ┌───────────────────┐                                          │
    │  │  User Input       │                                          │
    │  │  (Name/URL)       │                                          │
    │  └─────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │  ┌───────────────────┐                                          │
    │  │ Product           │                                          │
    │  │ Understanding     │ ─── Extract product info, generate       │
    │  │ Agent             │     optimized search queries             │
    │  └─────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │  ┌───────────────────┐     ┌──────────────────┐                 │
    │  │ Web Search        │ ◄───│ Retry if         │                 │
    │  │ Agent             │     │ < 5 URLs found   │                 │
    │  └─────────┬─────────┘     └──────────────────┘                 │
    │            │                                                     │
    │            ▼                                                     │
    │  ┌───────────────────┐     ┌──────────────────┐                 │
    │  │ Price Extraction  │ ◄───│ Expand search if │                 │
    │  │ Agent             │     │ < 3 prices       │                 │
    │  └─────────┬─────────┘     └──────────────────┘                 │
    │            │                                                     │
    │            ▼                                                     │
    │  ┌───────────────────┐     ┌──────────────────┐                 │
    │  │ Data Validation   │ ◄───│ Retry search if  │                 │
    │  │ Agent             │     │ no valid results │                 │
    │  └─────────┬─────────┘     └──────────────────┘                 │
    │            │                                                     │
    │            ▼                                                     │
    │  ┌───────────────────┐                                          │
    │  │ Price Comparison  │ ─── Rank, analyze, score deals           │
    │  │ Agent             │                                          │
    │  └─────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │  ┌───────────────────┐                                          │
    │  │ Reporting         │ ─── Generate final report with           │
    │  │ Agent             │     recommendations                       │
    │  └─────────┬─────────┘                                          │
    │            │                                                     │
    │            ▼                                                     │
    │  ┌───────────────────┐                                          │
    │  │  Final Report     │                                          │
    │  └───────────────────┘                                          │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        mcp_client: Optional[MCPClient] = None,
    ) -> None:
        """Initialize the workflow.

        Args:
            settings: Application settings.
            mcp_client: Connected MCP client for tool access.
        """
        self.settings = settings or get_settings()
        self.mcp_client = mcp_client
        self.logger = WorkflowLogger("price_comparison")

        # Initialize agents
        self.product_agent = ProductUnderstandingAgent(settings=self.settings)
        self.search_agent = WebSearchAgent(
            settings=self.settings, mcp_client=mcp_client
        )
        self.extraction_agent = PriceExtractionAgent(
            settings=self.settings, mcp_client=mcp_client
        )
        self.validation_agent = DataValidationAgent(settings=self.settings)
        self.comparison_agent = PriceComparisonAgent(settings=self.settings)
        self.reporting_agent = ReportingAgent(settings=self.settings)

        # Workflow configuration
        self.min_urls = self.settings.workflow.min_urls_for_extraction
        self.min_prices = self.settings.workflow.min_prices_for_validation
        self.max_retries = self.settings.workflow.max_search_retries

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph.

        Returns:
            Compiled StateGraph.
        """
        # Create the state graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("product_understanding", self._product_understanding_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("price_extraction", self._price_extraction_node)
        workflow.add_node("data_validation", self._data_validation_node)
        workflow.add_node("price_comparison", self._price_comparison_node)
        workflow.add_node("reporting", self._reporting_node)

        # Set entry point
        workflow.set_entry_point("product_understanding")

        # Add edges with conditional routing
        workflow.add_edge("product_understanding", "web_search")
        workflow.add_conditional_edges(
            "web_search",
            self._after_search_routing,
            {
                "continue": "price_extraction",
                "retry": "web_search",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "price_extraction",
            self._after_extraction_routing,
            {
                "continue": "data_validation",
                "expand": "web_search",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "data_validation",
            self._after_validation_routing,
            {
                "continue": "price_comparison",
                "retry": "web_search",
                "end": END,
            },
        )
        workflow.add_edge("price_comparison", "reporting")
        workflow.add_edge("reporting", END)

        # Compile with checkpointing
        checkpointer = MemorySaver() if self.settings.workflow.checkpoint_enabled else None
        return workflow.compile(checkpointer=checkpointer)

    async def _product_understanding_node(self, state: GraphState) -> StateUpdate:
        """Product understanding node - analyzes user input.

        Args:
            state: Current graph state.

        Returns:
            State update with product information.
        """
        self.logger.node_enter("product_understanding")
        start_time = time.time()

        try:
            product_info = await self.product_agent.analyze(state.input)

            duration = (time.time() - start_time) * 1000
            self.logger.node_exit("product_understanding", duration)

            return {
                "product_info": product_info,
                "current_node": "product_understanding",
            }

        except Exception as e:
            error = AgentError(
                agent_name="ProductUnderstanding",
                error_type=type(e).__name__,
                error_message=str(e),
                recoverable=False,
            )
            return {
                "errors": [error],
                "current_node": "product_understanding",
            }

    async def _web_search_node(self, state: GraphState) -> StateUpdate:
        """Web search node - finds product listings.

        Args:
            state: Current graph state.

        Returns:
            State update with search results.
        """
        self.logger.node_enter("web_search")
        start_time = time.time()

        if not state.product_info:
            error = AgentError(
                agent_name="WebSearch",
                error_type="MissingInput",
                error_message="No product information available",
                recoverable=False,
            )
            return {"errors": [error]}

        try:
            # Use alternative query on retry
            search_results = await self.search_agent.search(state.product_info)

            duration = (time.time() - start_time) * 1000
            self.logger.node_exit("web_search", duration, result_count=len(search_results))

            return {
                "search_results": search_results,
                "current_node": "web_search",
            }

        except Exception as e:
            error = AgentError(
                agent_name="WebSearch",
                error_type=type(e).__name__,
                error_message=str(e),
                recoverable=True,
            )
            return {
                "errors": [error],
                "retry_count": state.retry_count + 1,
            }

    async def _price_extraction_node(self, state: GraphState) -> StateUpdate:
        """Price extraction node - scrapes product pages.

        Args:
            state: Current graph state.

        Returns:
            State update with extracted prices.
        """
        self.logger.node_enter("price_extraction")
        start_time = time.time()

        if not state.search_results:
            error = AgentError(
                agent_name="PriceExtraction",
                error_type="MissingInput",
                error_message="No search results to process",
                recoverable=False,
            )
            return {"errors": [error]}

        try:
            price_data = await self.extraction_agent.extract_prices(state.search_results)

            duration = (time.time() - start_time) * 1000
            self.logger.node_exit("price_extraction", duration, extracted_count=len(price_data))

            return {
                "price_data": price_data,
                "current_node": "price_extraction",
            }

        except Exception as e:
            error = AgentError(
                agent_name="PriceExtraction",
                error_type=type(e).__name__,
                error_message=str(e),
                recoverable=True,
            )
            return {"errors": [error]}

    async def _data_validation_node(self, state: GraphState) -> StateUpdate:
        """Data validation node - validates extracted prices.

        Args:
            state: Current graph state.

        Returns:
            State update with validated prices.
        """
        self.logger.node_enter("data_validation")
        start_time = time.time()

        if not state.price_data or not state.product_info:
            error = AgentError(
                agent_name="DataValidation",
                error_type="MissingInput",
                error_message="No price data or product info to validate",
                recoverable=False,
            )
            return {"errors": [error]}

        try:
            validated = await self.validation_agent.validate(
                state.price_data, state.product_info
            )

            duration = (time.time() - start_time) * 1000
            self.logger.node_exit("data_validation", duration, validated_count=len(validated))

            return {
                "validated_prices": validated,
                "current_node": "data_validation",
            }

        except Exception as e:
            error = AgentError(
                agent_name="DataValidation",
                error_type=type(e).__name__,
                error_message=str(e),
                recoverable=True,
            )
            return {"errors": [error]}

    async def _price_comparison_node(self, state: GraphState) -> StateUpdate:
        """Price comparison node - ranks and analyzes prices.

        Args:
            state: Current graph state.

        Returns:
            State update with ranked results.
        """
        self.logger.node_enter("price_comparison")
        start_time = time.time()

        if not state.product_info:
            error = AgentError(
                agent_name="PriceComparison",
                error_type="MissingInput",
                error_message="No product info available",
                recoverable=False,
            )
            return {"errors": [error]}

        try:
            ranked = await self.comparison_agent.compare(
                state.validated_prices, state.product_info
            )

            duration = (time.time() - start_time) * 1000
            self.logger.node_exit("price_comparison", duration)

            return {
                "ranked_results": ranked,
                "current_node": "price_comparison",
            }

        except Exception as e:
            error = AgentError(
                agent_name="PriceComparison",
                error_type=type(e).__name__,
                error_message=str(e),
                recoverable=False,
            )
            return {"errors": [error]}

    async def _reporting_node(self, state: GraphState) -> StateUpdate:
        """Reporting node - generates final report.

        Args:
            state: Current graph state.

        Returns:
            State update with final report.
        """
        self.logger.node_enter("reporting")
        start_time = time.time()

        if not state.ranked_results:
            error = AgentError(
                agent_name="Reporting",
                error_type="MissingInput",
                error_message="No ranked results to report",
                recoverable=False,
            )
            return {"errors": [error]}

        try:
            report = await self.reporting_agent.generate_report(
                state.ranked_results,
                search_duration_seconds=state.elapsed_seconds,
            )

            duration = (time.time() - start_time) * 1000
            self.logger.node_exit("reporting", duration)

            return {
                "final_report": report,
                "current_node": "reporting",
            }

        except Exception as e:
            error = AgentError(
                agent_name="Reporting",
                error_type=type(e).__name__,
                error_message=str(e),
                recoverable=False,
            )
            return {"errors": [error]}

    def _after_search_routing(
        self, state: GraphState
    ) -> Literal["continue", "retry", "end"]:
        """Routing logic after web search.

        Args:
            state: Current graph state.

        Returns:
            Next node to execute.
        """
        # Check for unrecoverable errors
        unrecoverable = [e for e in state.errors if not e.recoverable]
        if unrecoverable:
            return "end"

        # Check if we have enough URLs
        if state.search_result_count >= self.min_urls:
            self.logger.transition("web_search", "price_extraction", "enough_urls")
            return "continue"

        # Retry if under limit
        if state.retry_count < self.max_retries:
            self.logger.transition("web_search", "web_search", "retry_search")
            return "retry"

        # Proceed with what we have
        if state.search_result_count > 0:
            self.logger.transition("web_search", "price_extraction", "partial_results")
            return "continue"

        return "end"

    def _after_extraction_routing(
        self, state: GraphState
    ) -> Literal["continue", "expand", "end"]:
        """Routing logic after price extraction.

        Args:
            state: Current graph state.

        Returns:
            Next node to execute.
        """
        # Check for unrecoverable errors
        unrecoverable = [e for e in state.errors if not e.recoverable]
        if unrecoverable:
            return "end"

        # Check if we have enough prices
        if state.price_count >= self.min_prices:
            self.logger.transition("price_extraction", "data_validation", "enough_prices")
            return "continue"

        # Try to expand search
        if state.retry_count < self.max_retries:
            self.logger.transition("price_extraction", "web_search", "expand_search")
            return "expand"

        # Proceed with what we have
        if state.price_count > 0:
            self.logger.transition("price_extraction", "data_validation", "partial_prices")
            return "continue"

        return "end"

    def _after_validation_routing(
        self, state: GraphState
    ) -> Literal["continue", "retry", "end"]:
        """Routing logic after data validation.

        Args:
            state: Current graph state.

        Returns:
            Next node to execute.
        """
        # Check for unrecoverable errors
        unrecoverable = [e for e in state.errors if not e.recoverable]
        if unrecoverable:
            return "end"

        # Check if we have valid results
        if state.validated_count > 0:
            self.logger.transition("data_validation", "price_comparison", "has_valid_prices")
            return "continue"

        # Retry search with different queries
        if state.retry_count < self.max_retries:
            self.logger.transition("data_validation", "web_search", "no_valid_prices_retry")
            return "retry"

        return "end"

    async def run(
        self,
        user_input: str,
        config: Optional[dict[str, Any]] = None,
    ) -> GraphState:
        """Execute the price comparison workflow.

        Args:
            user_input: User's product search input.
            config: Optional LangGraph configuration.

        Returns:
            Final graph state with results.
        """
        self.logger.start(user_input)
        start_time = time.time()

        try:
            # Create initial state
            initial_state = create_initial_state(user_input)

            # Run the graph
            final_state = await self.graph.ainvoke(
                initial_state.model_dump(),
                config=config or {},
            )

            # Convert back to GraphState
            result = GraphState(**final_state)

            duration = (time.time() - start_time) * 1000
            self.logger.complete(
                duration,
                success=result.final_report is not None,
                result_count=result.validated_count,
            )

            return result

        except Exception as e:
            self.logger.error(e)
            raise


def create_workflow(
    settings: Optional[Settings] = None,
    mcp_client: Optional[MCPClient] = None,
) -> PriceComparisonWorkflow:
    """Create a PriceComparisonWorkflow instance.

    Args:
        settings: Optional application settings.
        mcp_client: Optional connected MCP client.

    Returns:
        Configured workflow instance.
    """
    return PriceComparisonWorkflow(settings=settings, mcp_client=mcp_client)
