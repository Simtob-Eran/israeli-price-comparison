"""LangGraph state definitions for the price comparison workflow.

This module defines the state schema used throughout the LangGraph workflow,
including all data that flows between agent nodes.
"""

import operator
from datetime import datetime
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field

from src.models.schemas import (
    AgentError,
    FinalReport,
    PriceData,
    ProductInfo,
    RankedResults,
    SearchResult,
)


class GraphState(BaseModel):
    """LangGraph state for the price comparison workflow.

    This state object is passed between all nodes in the workflow graph,
    accumulating data as each agent processes it.

    Attributes:
        input: Original user input (product name or URL).
        product_info: Structured product information from understanding agent.
        search_results: Web search results with candidate URLs.
        price_data: Extracted price information from all sources.
        validated_prices: Prices that passed validation.
        ranked_results: Final ranked and compared prices.
        final_report: Generated report for user display.
        errors: List of errors encountered during processing.
        retry_count: Number of retry attempts made.
        current_node: Name of the currently executing node.
        start_time: Workflow start timestamp.
        metadata: Additional workflow metadata.
    """

    # Input
    input: str = Field(..., description="Original user input (product name or URL)")

    # Processing state
    product_info: Optional[ProductInfo] = Field(
        None, description="Structured product information"
    )
    search_results: Annotated[list[SearchResult], operator.add] = Field(
        default_factory=list, description="Web search results"
    )
    price_data: Annotated[list[PriceData], operator.add] = Field(
        default_factory=list, description="Extracted price data"
    )
    validated_prices: list[PriceData] = Field(
        default_factory=list, description="Validated price data"
    )
    ranked_results: Optional[RankedResults] = Field(
        None, description="Ranked comparison results"
    )
    final_report: Optional[FinalReport] = Field(
        None, description="Final generated report"
    )

    # Error tracking
    errors: Annotated[list[AgentError], operator.add] = Field(
        default_factory=list, description="Errors encountered"
    )

    # Workflow control
    retry_count: int = Field(default=0, description="Retry attempt count")
    current_node: str = Field(default="", description="Current node name")
    start_time: datetime = Field(
        default_factory=datetime.utcnow, description="Workflow start time"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time since workflow start."""
        return (datetime.utcnow() - self.start_time).total_seconds()

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0

    @property
    def search_result_count(self) -> int:
        """Get count of search results."""
        return len(self.search_results)

    @property
    def price_count(self) -> int:
        """Get count of extracted prices."""
        return len(self.price_data)

    @property
    def validated_count(self) -> int:
        """Get count of validated prices."""
        return len(self.validated_prices)

    def add_error(
        self,
        agent_name: str,
        error_type: str,
        message: str,
        recoverable: bool = True,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add an error to the state.

        Args:
            agent_name: Name of the agent that encountered the error.
            error_type: Type of error.
            message: Error message.
            recoverable: Whether the workflow can continue.
            context: Additional error context.
        """
        error = AgentError(
            agent_name=agent_name,
            error_type=error_type,
            error_message=message,
            recoverable=recoverable,
            context=context or {},
        )
        # Note: Can't use append with Annotated list, this is for documentation
        # In actual workflow, return {"errors": [error]} to use the reducer

    def to_summary(self) -> dict[str, Any]:
        """Generate a summary of the current state.

        Returns:
            Dictionary with state summary.
        """
        return {
            "input": self.input[:50] + "..." if len(self.input) > 50 else self.input,
            "product_identified": self.product_info is not None,
            "search_results": self.search_result_count,
            "prices_extracted": self.price_count,
            "prices_validated": self.validated_count,
            "has_report": self.final_report is not None,
            "error_count": len(self.errors),
            "retry_count": self.retry_count,
            "elapsed_seconds": self.elapsed_seconds,
        }


# Type alias for state updates returned by nodes
StateUpdate = dict[str, Any]


def create_initial_state(user_input: str) -> GraphState:
    """Create initial graph state from user input.

    Args:
        user_input: User's product search input.

    Returns:
        Initialized GraphState.
    """
    return GraphState(
        input=user_input,
        start_time=datetime.utcnow(),
        metadata={
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0.0",
        },
    )


# Reducer functions for state aggregation
def add_search_results(
    current: list[SearchResult], new: list[SearchResult]
) -> list[SearchResult]:
    """Reducer to combine search results without duplicates.

    Args:
        current: Current search results.
        new: New search results to add.

    Returns:
        Combined list without URL duplicates.
    """
    seen_urls = {str(r.url) for r in current}
    combined = list(current)

    for result in new:
        if str(result.url) not in seen_urls:
            seen_urls.add(str(result.url))
            combined.append(result)

    return combined


def add_price_data(
    current: list[PriceData], new: list[PriceData]
) -> list[PriceData]:
    """Reducer to combine price data without duplicates.

    Args:
        current: Current price data.
        new: New price data to add.

    Returns:
        Combined list without URL duplicates.
    """
    seen_urls = {str(p.url) for p in current}
    combined = list(current)

    for price in new:
        if str(price.url) not in seen_urls:
            seen_urls.add(str(price.url))
            combined.append(price)

    return combined


def add_errors(
    current: list[AgentError], new: list[AgentError]
) -> list[AgentError]:
    """Reducer to combine error lists.

    Args:
        current: Current errors.
        new: New errors to add.

    Returns:
        Combined error list.
    """
    return current + new
