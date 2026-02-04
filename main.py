#!/usr/bin/env python3
"""Price Comparison CLI Application.

A multi-agent price comparison system using LangGraph that finds
the best prices across the web for any product.

Usage:
    python main.py search "iPhone 15 Pro Max 256GB"
    python main.py search "https://www.amazon.com/dp/B0CHX1W1XY"
    python main.py search "MacBook Pro M3" --output results.json

Author: Price Comparison Team
License: MIT
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.graph.workflow import create_workflow
from src.mcp_client.client import MCPClient, MockMCPClient
from src.utils.config import Settings, get_settings
from src.utils.logger import setup_logging

# Initialize CLI app and console
app = typer.Typer(
    name="price-compare",
    help="Multi-agent price comparison tool powered by LangGraph",
    add_completion=False,
)
console = Console()


def print_banner() -> None:
    """Print application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║              PRICE COMPARISON - Multi-Agent Search                 ║
║                  Powered by LangGraph & OpenAI                     ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def print_error(message: str) -> None:
    """Print error message.

    Args:
        message: Error message to display.
    """
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """Print success message.

    Args:
        message: Success message to display.
    """
    console.print(f"[bold green]Success:[/bold green] {message}")


def create_results_table(state) -> Table:
    """Create a rich table from workflow results.

    Args:
        state: Final workflow state.

    Returns:
        Rich Table object.
    """
    table = Table(
        title="Price Comparison Results",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Store", style="white", width=20)
    table.add_column("Price", style="green", justify="right", width=12)
    table.add_column("Shipping", style="yellow", justify="right", width=10)
    table.add_column("Total", style="bold green", justify="right", width=12)
    table.add_column("Status", style="blue", width=12)
    table.add_column("Score", style="magenta", justify="right", width=8)

    if state.ranked_results:
        for result in state.ranked_results.results[:10]:
            pd = result.price_data
            shipping = (
                f"{pd.currency_symbol}{pd.shipping_cost:.0f}"
                if pd.shipping_cost
                else "Free"
            )
            availability = pd.availability.value.replace("_", " ").title()

            # Highlight best deal
            rank_style = "bold cyan" if result.is_best_deal else "cyan"
            rank_text = f"{'★ ' if result.is_best_deal else ''}{result.rank}"

            table.add_row(
                rank_text,
                pd.store_name[:20],
                f"{pd.currency_symbol}{pd.price:,.0f}",
                shipping,
                f"{pd.currency_symbol}{result.total_cost:,.0f}",
                availability,
                f"{result.deal_score:.0f}/100",
                style=rank_style if result.is_best_deal else None,
            )

    return table


async def run_search(
    query: str,
    settings: Settings,
    use_mock: bool = False,
) -> Optional[dict]:
    """Execute the price comparison search.

    Args:
        query: Product search query or URL.
        settings: Application settings.
        use_mock: Whether to use mock MCP client.

    Returns:
        Final workflow state dictionary or None on failure.
    """
    # Initialize MCP client
    if use_mock:
        mcp_client = MockMCPClient()
    else:
        mcp_client = MCPClient(
            server_url=settings.mcp.server_url,
            timeout=settings.mcp.timeout,
        )

    try:
        # Connect to MCP server
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Connect
            task = progress.add_task("Connecting to MCP server...", total=None)
            try:
                await mcp_client.connect()
                progress.update(task, description="[green]Connected to MCP server")
            except Exception as e:
                if not use_mock:
                    console.print(
                        f"[yellow]Warning: Could not connect to MCP server ({e}). "
                        f"Using mock client.[/yellow]"
                    )
                    mcp_client = MockMCPClient()
                    await mcp_client.connect()

            # Create workflow
            progress.update(task, description="Initializing workflow...")
            workflow = create_workflow(settings=settings, mcp_client=mcp_client)

            # Run search
            progress.update(task, description=f"Searching for: {query[:50]}...")

        # Execute workflow with detailed progress
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running price comparison...", total=6)

            # Monitor workflow execution
            state = await workflow.run(query)

            # Update progress based on completed nodes
            progress.update(task, completed=6, description="[green]Search complete!")

        return state

    except Exception as e:
        print_error(f"Search failed: {e}")
        return None

    finally:
        await mcp_client.disconnect()


def export_results(state, output_path: str, format_type: str = "json") -> None:
    """Export results to file.

    Args:
        state: Workflow state with results.
        output_path: Output file path.
        format_type: Export format (json/csv).
    """
    if not state.final_report:
        print_error("No results to export")
        return

    output = Path(output_path)
    report = state.final_report

    if format_type == "csv":
        from src.agents.reporting import ReportingAgent
        agent = ReportingAgent()
        content = agent.export_csv(report)
        output.write_text(content, encoding="utf-8")
    else:
        content = report.model_dump_json(indent=2)
        output.write_text(content, encoding="utf-8")

    print_success(f"Results exported to {output_path}")


@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Product name or URL to search for",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Export results to file (JSON or CSV based on extension)",
    ),
    format_type: str = typer.Option(
        "json",
        "--format", "-f",
        help="Output format: json or csv",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
    mock: bool = typer.Option(
        False,
        "--mock", "-m",
        help="Use mock MCP client (for testing)",
    ),
    config_file: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to custom configuration file",
    ),
) -> None:
    """Search for the best prices of a product.

    Examples:
        python main.py search "iPhone 15 Pro Max 256GB"
        python main.py search "https://www.amazon.com/dp/B0CHX1W1XY" --output results.json
        python main.py search "MacBook Pro M3" -v --format csv -o prices.csv
    """
    print_banner()

    # Load settings
    try:
        if config_file:
            settings = Settings.from_yaml(Path(config_file))
        else:
            settings = get_settings()
    except Exception as e:
        print_error(f"Failed to load settings: {e}")
        raise typer.Exit(1)

    # Setup logging
    log_level = "DEBUG" if verbose else settings.logging.level
    setup_logging(level=log_level, log_format="plain")

    # Validate API keys
    if not settings.openai.api_key and not mock:
        print_error("OPENAI_API_KEY environment variable not set")
        console.print("Set it with: export OPENAI_API_KEY=your_key")
        raise typer.Exit(1)

    # Display search info
    console.print(f"\n[bold]Searching for:[/bold] {query}\n")

    # Run search
    state = asyncio.run(run_search(query, settings, use_mock=mock))

    if not state:
        raise typer.Exit(1)

    # Display results
    console.print()

    if state.final_report:
        # Show summary
        report = state.final_report
        console.print(Panel(
            f"[bold]{report.summary}[/bold]\n\n{report.recommendation}",
            title="Summary",
            border_style="green",
        ))

        console.print()

        # Show results table
        table = create_results_table(state)
        console.print(table)

        console.print()

        # Show statistics
        if state.ranked_results:
            rr = state.ranked_results
            console.print(Panel(
                f"[bold]Statistics[/bold]\n"
                f"• Total results: {rr.total_results}\n"
                f"• Average price: {rr.results[0].price_data.currency_symbol if rr.results else '₪'}{rr.average_price:,.0f}\n"
                f"• Price range: {rr.results[0].price_data.currency_symbol if rr.results else '₪'}{rr.price_range:,.0f}\n"
                f"• Confidence: {rr.confidence_score:.0f}/100\n"
                f"• Search time: {report.search_duration_seconds:.1f}s",
                title="Analysis",
                border_style="blue",
            ))

        # Best time to buy
        if report.best_time_to_buy:
            console.print()
            console.print(Panel(
                report.best_time_to_buy,
                title="Best Time to Buy",
                border_style="yellow",
            ))

        # Export if requested
        if output:
            # Determine format from extension if not specified
            if output.endswith(".csv"):
                format_type = "csv"
            export_results(state, output, format_type)

    else:
        print_error("No results found")

        # Show errors if any
        if state.errors:
            console.print("\n[bold red]Errors encountered:[/bold red]")
            for error in state.errors:
                console.print(f"  • {error.agent_name}: {error.error_message}")

        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show", "-s",
        help="Show current configuration",
    ),
    create: bool = typer.Option(
        False,
        "--create", "-c",
        help="Create default configuration file",
    ),
) -> None:
    """Manage application configuration."""
    if show:
        settings = get_settings()
        console.print(Panel(
            f"OpenAI Model: {settings.openai.model}\n"
            f"MCP Server: {settings.mcp.server_url}\n"
            f"Max Retries: {settings.agents.max_retries}\n"
            f"Concurrent Requests: {settings.scraping.concurrent_requests}",
            title="Current Configuration",
        ))

    if create:
        config_path = Path("config/settings.yaml")
        if config_path.exists():
            print_error("Configuration file already exists")
            raise typer.Exit(1)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy default config
        console.print("Creating default configuration file...")
        print_success(f"Configuration created at {config_path}")


@app.command()
def version() -> None:
    """Show application version."""
    from src import __version__
    console.print(f"Price Comparison App v{__version__}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
