"""Price Comparison Agent.

This agent analyzes validated prices and ranks them to identify
the best deals for the user.
"""

from typing import Optional

from src.models.schemas import (
    PriceData,
    ProductInfo,
    RankedResult,
    RankedResults,
)
from src.utils.config import Settings, get_settings
from src.utils.logger import AgentLogger


class PriceComparisonAgent:
    """Agent for comparing and ranking validated prices.

    This agent analyzes all validated prices, calculates total costs,
    identifies the best deals, and generates confidence scores.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the Price Comparison Agent.

        Args:
            settings: Application settings.
        """
        self.settings = settings or get_settings()
        self.logger = AgentLogger("PriceComparison")

    async def compare(
        self,
        validated_prices: list[PriceData],
        product_info: ProductInfo,
    ) -> RankedResults:
        """Compare and rank validated prices.

        Args:
            validated_prices: List of validated price data.
            product_info: Original product information.

        Returns:
            RankedResults with sorted and analyzed prices.
        """
        self.logger.start(
            "compare_prices",
            price_count=len(validated_prices),
            product=product_info.product_name,
        )

        try:
            if not validated_prices:
                return RankedResults(
                    product_info=product_info,
                    results=[],
                    total_results=0,
                    confidence_score=0,
                )

            # Calculate statistics
            stats = self._calculate_statistics(validated_prices)

            # Sort prices by total cost
            sorted_prices = sorted(validated_prices, key=lambda p: p.total_cost)

            # Create ranked results
            ranked_results: list[RankedResult] = []

            for rank, price_data in enumerate(sorted_prices, start=1):
                total_cost = price_data.total_cost
                savings = stats["average"] - total_cost if stats["average"] else None
                savings_pct = (savings / stats["average"] * 100) if savings and stats["average"] else None

                # Calculate deal score
                deal_score = self._calculate_deal_score(
                    price_data, stats, rank, len(sorted_prices)
                )

                # Generate recommendation
                recommendation = self._generate_recommendation(
                    price_data, deal_score, rank
                )

                ranked_result = RankedResult(
                    rank=rank,
                    price_data=price_data,
                    total_cost=total_cost,
                    savings_vs_average=savings,
                    savings_percentage=savings_pct,
                    is_best_deal=(rank == 1 and deal_score >= 80),
                    deal_score=deal_score,
                    recommendation=recommendation,
                )
                ranked_results.append(ranked_result)

            # Calculate overall confidence
            confidence_score = self._calculate_confidence(validated_prices, stats)

            result = RankedResults(
                product_info=product_info,
                results=ranked_results,
                average_price=stats["average"],
                lowest_price=stats["min"],
                highest_price=stats["max"],
                price_range=stats["range"],
                total_results=len(validated_prices),
                confidence_score=confidence_score,
            )

            self.logger.complete(
                "compare_prices",
                duration_ms=0,
                best_price=stats["min"],
                average_price=stats["average"],
                result_count=len(ranked_results),
            )

            return result

        except Exception as e:
            self.logger.error("compare_prices", e)
            raise

    def _calculate_statistics(
        self, prices: list[PriceData]
    ) -> dict[str, Optional[float]]:
        """Calculate price statistics.

        Args:
            prices: List of price data.

        Returns:
            Dictionary with statistical measures.
        """
        if not prices:
            return {
                "min": None,
                "max": None,
                "average": None,
                "median": None,
                "range": None,
                "std_dev": None,
            }

        total_costs = [p.total_cost for p in prices]
        total_costs.sort()

        n = len(total_costs)
        total_sum = sum(total_costs)
        average = total_sum / n

        # Calculate standard deviation
        variance = sum((c - average) ** 2 for c in total_costs) / n
        std_dev = variance ** 0.5

        # Calculate median
        if n % 2 == 0:
            median = (total_costs[n // 2 - 1] + total_costs[n // 2]) / 2
        else:
            median = total_costs[n // 2]

        return {
            "min": total_costs[0],
            "max": total_costs[-1],
            "average": average,
            "median": median,
            "range": total_costs[-1] - total_costs[0],
            "std_dev": std_dev,
        }

    def _calculate_deal_score(
        self,
        price_data: PriceData,
        stats: dict[str, Optional[float]],
        rank: int,
        total_count: int,
    ) -> float:
        """Calculate deal score for a price.

        Score is based on:
        - Price relative to average (40%)
        - Price relative to minimum (20%)
        - Relevance score (20%)
        - Availability (10%)
        - Seller rating (10%)

        Args:
            price_data: Price data to score.
            stats: Price statistics.
            rank: Rank position.
            total_count: Total number of results.

        Returns:
            Deal score between 0 and 100.
        """
        score = 0.0

        # Price vs average (40 points max)
        if stats["average"]:
            price_ratio = price_data.total_cost / stats["average"]
            if price_ratio <= 0.8:
                score += 40
            elif price_ratio <= 0.9:
                score += 35
            elif price_ratio <= 1.0:
                score += 30
            elif price_ratio <= 1.1:
                score += 20
            else:
                score += max(0, 15 - (price_ratio - 1.1) * 50)

        # Price vs minimum (20 points max)
        if stats["min"] and stats["range"]:
            if stats["range"] > 0:
                position_ratio = (price_data.total_cost - stats["min"]) / stats["range"]
                score += max(0, 20 * (1 - position_ratio))
            else:
                score += 20  # All prices are the same

        # Relevance score (20 points max)
        if price_data.relevance_score:
            score += (price_data.relevance_score / 100) * 20

        # Availability (10 points max)
        from src.models.schemas import AvailabilityStatus
        availability_scores = {
            AvailabilityStatus.IN_STOCK: 10,
            AvailabilityStatus.LIMITED: 7,
            AvailabilityStatus.PREORDER: 5,
            AvailabilityStatus.UNKNOWN: 3,
            AvailabilityStatus.OUT_OF_STOCK: 0,
        }
        score += availability_scores.get(price_data.availability, 3)

        # Seller rating (10 points max)
        if price_data.seller_rating:
            score += (price_data.seller_rating / 5) * 10

        return min(100, max(0, score))

    def _generate_recommendation(
        self, price_data: PriceData, deal_score: float, rank: int
    ) -> str:
        """Generate purchase recommendation text.

        Args:
            price_data: Price data.
            deal_score: Calculated deal score.
            rank: Rank position.

        Returns:
            Recommendation text.
        """
        from src.models.schemas import AvailabilityStatus

        parts: list[str] = []

        # Deal quality
        if deal_score >= 90:
            parts.append("Excellent deal!")
        elif deal_score >= 80:
            parts.append("Very good deal")
        elif deal_score >= 70:
            parts.append("Good deal")
        elif deal_score >= 60:
            parts.append("Fair price")
        else:
            parts.append("Average price")

        # Availability note
        if price_data.availability == AvailabilityStatus.IN_STOCK:
            parts.append("In stock")
        elif price_data.availability == AvailabilityStatus.LIMITED:
            parts.append("Limited availability")
        elif price_data.availability == AvailabilityStatus.OUT_OF_STOCK:
            parts.append("Currently out of stock")

        # Seller rating note
        if price_data.seller_rating:
            if price_data.seller_rating >= 4.5:
                parts.append(f"Highly rated seller ({price_data.seller_rating:.1f}/5)")
            elif price_data.seller_rating >= 4.0:
                parts.append(f"Well rated seller ({price_data.seller_rating:.1f}/5)")

        # Shipping note
        if price_data.shipping_cost:
            if price_data.shipping_cost == 0:
                parts.append("Free shipping")
            else:
                parts.append(f"Shipping: {price_data.currency_symbol}{price_data.shipping_cost:.0f}")

        return " • ".join(parts)

    def _calculate_confidence(
        self, prices: list[PriceData], stats: dict[str, Optional[float]]
    ) -> float:
        """Calculate overall confidence score for results.

        Confidence is based on:
        - Number of results (more = better)
        - Price consistency (lower std dev = better)
        - Average relevance scores

        Args:
            prices: List of validated prices.
            stats: Price statistics.

        Returns:
            Confidence score between 0 and 100.
        """
        confidence = 0.0

        # Result count factor (30 points max)
        result_count = len(prices)
        if result_count >= 10:
            confidence += 30
        elif result_count >= 5:
            confidence += 25
        elif result_count >= 3:
            confidence += 20
        else:
            confidence += result_count * 5

        # Price consistency factor (30 points max)
        if stats["std_dev"] and stats["average"]:
            coefficient_of_variation = stats["std_dev"] / stats["average"]
            if coefficient_of_variation <= 0.1:
                confidence += 30
            elif coefficient_of_variation <= 0.2:
                confidence += 25
            elif coefficient_of_variation <= 0.3:
                confidence += 20
            else:
                confidence += max(0, 15 - coefficient_of_variation * 30)

        # Average relevance factor (40 points max)
        relevance_scores = [
            p.relevance_score for p in prices if p.relevance_score is not None
        ]
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            confidence += (avg_relevance / 100) * 40

        return min(100, max(0, confidence))


# Factory function
def create_price_comparison_agent(
    settings: Optional[Settings] = None,
) -> PriceComparisonAgent:
    """Create a Price Comparison Agent instance.

    Args:
        settings: Optional application settings.

    Returns:
        Configured PriceComparisonAgent instance.
    """
    return PriceComparisonAgent(settings=settings)
