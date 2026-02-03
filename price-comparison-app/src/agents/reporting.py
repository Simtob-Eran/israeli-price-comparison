"""Reporting Agent.

This agent generates user-friendly reports from price comparison results,
including formatted tables, recommendations, and insights.
"""

from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models.schemas import FinalReport, RankedResults
from src.utils.config import Settings, get_settings
from src.utils.logger import AgentLogger

# System prompt for report generation
REPORT_PROMPT = """You are a consumer advisor helping users make informed purchasing decisions.
Generate a clear, helpful report based on price comparison results.

**Report Guidelines**:
1. Start with a brief executive summary (2-3 sentences)
2. Highlight the best deal with specific savings
3. Provide actionable recommendations
4. Be honest about data confidence levels
5. Mention any caveats (limited results, price variations, etc.)

**Tone**: Professional but friendly, focused on helping the user save money.

**Language**: Write in English. Include prices in their original currency format."""


class ReportingAgent:
    """Agent for generating user-friendly price comparison reports.

    This agent creates formatted reports with comparison tables,
    recommendations, and insights based on ranked price results.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[ChatOpenAI] = None,
    ) -> None:
        """Initialize the Reporting Agent.

        Args:
            settings: Application settings.
            llm: LangChain LLM instance.
        """
        self.settings = settings or get_settings()
        self.logger = AgentLogger("Reporting")

        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model=self.settings.openai.model,
                temperature=0.3,  # Slightly creative for better writing
                api_key=self.settings.openai.api_key,
                request_timeout=self.settings.openai.request_timeout,
            )

    async def generate_report(
        self,
        ranked_results: RankedResults,
        search_duration_seconds: float = 0,
    ) -> FinalReport:
        """Generate a comprehensive price comparison report.

        Args:
            ranked_results: Ranked price comparison results.
            search_duration_seconds: Total search duration.

        Returns:
            FinalReport with formatted content.
        """
        self.logger.start(
            "generate_report",
            product=ranked_results.product_info.product_name,
            result_count=ranked_results.total_results,
        )

        try:
            # Generate comparison table
            comparison_table = self._generate_comparison_table(ranked_results)

            # Generate summary and recommendation
            summary, recommendation = await self._generate_insights(ranked_results)

            # Generate best time to buy suggestion
            best_time = self._suggest_best_time(ranked_results)

            report = FinalReport(
                product_info=ranked_results.product_info,
                ranked_results=ranked_results,
                summary=summary,
                recommendation=recommendation,
                comparison_table=comparison_table,
                best_time_to_buy=best_time,
                price_history_context=None,  # Would come from storage MCP
                generated_at=datetime.utcnow(),
                search_duration_seconds=search_duration_seconds,
            )

            self.logger.complete(
                "generate_report",
                duration_ms=0,
                has_recommendation=bool(recommendation),
            )

            return report

        except Exception as e:
            self.logger.error("generate_report", e)
            raise

    def _generate_comparison_table(self, ranked_results: RankedResults) -> str:
        """Generate a Markdown comparison table.

        Args:
            ranked_results: Ranked price results.

        Returns:
            Markdown formatted table.
        """
        if not ranked_results.results:
            return "No results found."

        # Table header
        lines = [
            "| Rank | Store | Price | Shipping | Total | Availability | Score |",
            "|------|-------|-------|----------|-------|--------------|-------|",
        ]

        # Table rows
        for result in ranked_results.results[:10]:  # Top 10 only
            price_data = result.price_data
            shipping = (
                f"{price_data.currency_symbol}{price_data.shipping_cost:.0f}"
                if price_data.shipping_cost
                else "Free"
            )
            availability = price_data.availability.value.replace("_", " ").title()

            # Highlight best deal
            rank_str = f"**{result.rank}**" if result.is_best_deal else str(result.rank)

            lines.append(
                f"| {rank_str} | "
                f"[{price_data.store_name}]({price_data.url}) | "
                f"{price_data.currency_symbol}{price_data.price:,.0f} | "
                f"{shipping} | "
                f"{price_data.currency_symbol}{result.total_cost:,.0f} | "
                f"{availability} | "
                f"{result.deal_score:.0f}/100 |"
            )

        return "\n".join(lines)

    async def _generate_insights(
        self, ranked_results: RankedResults
    ) -> tuple[str, str]:
        """Generate summary and recommendation using LLM.

        Args:
            ranked_results: Ranked price results.

        Returns:
            Tuple of (summary, recommendation).
        """
        if not ranked_results.results:
            return (
                "No valid prices found for this product.",
                "Try searching with different keywords or check back later.",
            )

        best = ranked_results.best_deal
        product = ranked_results.product_info

        # Build context for LLM
        context = f"""Product: {product.product_name}
Brand: {product.brand or "Unknown"}
Category: {product.category.value}

Results Summary:
- Total results: {ranked_results.total_results}
- Lowest price: {best.price_data.currency_symbol}{ranked_results.lowest_price:,.0f}
- Highest price: {best.price_data.currency_symbol}{ranked_results.highest_price:,.0f}
- Average price: {best.price_data.currency_symbol}{ranked_results.average_price:,.0f}
- Price range: {best.price_data.currency_symbol}{ranked_results.price_range:,.0f}
- Confidence score: {ranked_results.confidence_score:.0f}/100

Best Deal:
- Store: {best.price_data.store_name}
- Price: {best.price_data.currency_symbol}{best.price_data.price:,.0f}
- Total cost: {best.price_data.currency_symbol}{best.total_cost:,.0f}
- Savings vs average: {best.price_data.currency_symbol}{best.savings_vs_average:,.0f} ({best.savings_percentage:.1f}%)
- Deal score: {best.deal_score:.0f}/100
- Availability: {best.price_data.availability.value}"""

        messages = [
            SystemMessage(content=REPORT_PROMPT),
            HumanMessage(
                content=f"Generate a summary and recommendation for these results:\n\n{context}"
            ),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            content = response.content

            # Split into summary and recommendation
            # Try to find sections or use heuristics
            if "Recommendation:" in content:
                parts = content.split("Recommendation:", 1)
                summary = parts[0].replace("Summary:", "").strip()
                recommendation = parts[1].strip()
            elif "recommendation" in content.lower():
                lines = content.split("\n")
                summary_lines = []
                rec_lines = []
                in_rec = False
                for line in lines:
                    if "recommend" in line.lower():
                        in_rec = True
                    if in_rec:
                        rec_lines.append(line)
                    else:
                        summary_lines.append(line)
                summary = "\n".join(summary_lines).strip()
                recommendation = "\n".join(rec_lines).strip()
            else:
                # Use first half as summary, second as recommendation
                mid = len(content) // 2
                summary = content[:mid].strip()
                recommendation = content[mid:].strip()

            return summary, recommendation

        except Exception as e:
            self.logger.debug(f"LLM insight generation failed: {e}")
            # Fallback to template-based insights
            return self._template_insights(ranked_results)

    def _template_insights(
        self, ranked_results: RankedResults
    ) -> tuple[str, str]:
        """Generate template-based insights as fallback.

        Args:
            ranked_results: Ranked price results.

        Returns:
            Tuple of (summary, recommendation).
        """
        best = ranked_results.best_deal
        if not best:
            return ("No results found.", "Try different search terms.")

        product = ranked_results.product_info

        # Summary
        summary = (
            f"Found {ranked_results.total_results} prices for {product.product_name}. "
            f"Prices range from {best.price_data.currency_symbol}{ranked_results.lowest_price:,.0f} "
            f"to {best.price_data.currency_symbol}{ranked_results.highest_price:,.0f}, "
            f"with an average of {best.price_data.currency_symbol}{ranked_results.average_price:,.0f}."
        )

        # Recommendation
        savings = best.savings_vs_average or 0
        savings_pct = best.savings_percentage or 0

        if savings > 0:
            recommendation = (
                f"Best deal at {best.price_data.store_name} for "
                f"{best.price_data.currency_symbol}{best.total_cost:,.0f} "
                f"(save {best.price_data.currency_symbol}{savings:,.0f} / {savings_pct:.1f}% vs average). "
            )
        else:
            recommendation = (
                f"Best price at {best.price_data.store_name} for "
                f"{best.price_data.currency_symbol}{best.total_cost:,.0f}. "
            )

        # Add availability note
        if best.price_data.availability.value == "in_stock":
            recommendation += "Item is in stock."
        elif best.price_data.availability.value == "limited":
            recommendation += "Limited availability - act fast!"

        return summary, recommendation

    def _suggest_best_time(self, ranked_results: RankedResults) -> Optional[str]:
        """Suggest the best time to buy based on category.

        Args:
            ranked_results: Ranked results with product info.

        Returns:
            Best time suggestion or None.
        """
        from src.models.schemas import ProductCategory

        category = ranked_results.product_info.category

        suggestions = {
            ProductCategory.ELECTRONICS: (
                "Electronics often see price drops during Black Friday (November), "
                "Amazon Prime Day (July), and after new model releases."
            ),
            ProductCategory.CLOTHING: (
                "Best deals on clothing during end-of-season sales, "
                "Black Friday, and post-holiday clearances (January)."
            ),
            ProductCategory.TOYS: (
                "Toy prices typically drop after the holiday season "
                "(January-February) and during summer sales."
            ),
            ProductCategory.HOME_GARDEN: (
                "Home goods see best prices during holiday weekends "
                "(Memorial Day, Labor Day) and January white sales."
            ),
            ProductCategory.SPORTS: (
                "Sports equipment often discounted at season end "
                "and during major sporting events."
            ),
        }

        return suggestions.get(category)

    def format_for_cli(self, report: FinalReport) -> str:
        """Format report for command-line display.

        Args:
            report: Final report to format.

        Returns:
            CLI-friendly formatted string.
        """
        lines = [
            "=" * 70,
            f"PRICE COMPARISON REPORT",
            f"Product: {report.product_info.product_name}",
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "SUMMARY",
            "-" * 40,
            report.summary,
            "",
            "RECOMMENDATION",
            "-" * 40,
            report.recommendation,
            "",
            "PRICE COMPARISON",
            "-" * 40,
            report.comparison_table,
            "",
        ]

        if report.best_time_to_buy:
            lines.extend([
                "BEST TIME TO BUY",
                "-" * 40,
                report.best_time_to_buy,
                "",
            ])

        # Statistics
        results = report.ranked_results
        lines.extend([
            "STATISTICS",
            "-" * 40,
            f"Total results: {results.total_results}",
            f"Average price: {results.results[0].price_data.currency_symbol if results.results else '₪'}{results.average_price:,.0f}" if results.average_price else "N/A",
            f"Price range: {results.results[0].price_data.currency_symbol if results.results else '₪'}{results.price_range:,.0f}" if results.price_range else "N/A",
            f"Confidence: {results.confidence_score:.0f}/100",
            f"Search time: {report.search_duration_seconds:.1f}s",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def export_json(self, report: FinalReport) -> str:
        """Export report as JSON.

        Args:
            report: Final report to export.

        Returns:
            JSON string.
        """
        return report.model_dump_json(indent=2)

    def export_csv(self, report: FinalReport) -> str:
        """Export price data as CSV.

        Args:
            report: Final report to export.

        Returns:
            CSV string.
        """
        lines = [
            "rank,store,price,currency,shipping,total,availability,score,url"
        ]

        for result in report.ranked_results.results:
            pd = result.price_data
            lines.append(
                f"{result.rank},"
                f'"{pd.store_name}",'
                f"{pd.price},"
                f"{pd.currency},"
                f"{pd.shipping_cost or 0},"
                f"{result.total_cost},"
                f"{pd.availability.value},"
                f"{result.deal_score},"
                f'"{pd.url}"'
            )

        return "\n".join(lines)


# Factory function
def create_reporting_agent(
    settings: Optional[Settings] = None,
) -> ReportingAgent:
    """Create a Reporting Agent instance.

    Args:
        settings: Optional application settings.

    Returns:
        Configured ReportingAgent instance.
    """
    return ReportingAgent(settings=settings)
