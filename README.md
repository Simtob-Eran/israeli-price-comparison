# Price Comparison App

A production-ready multi-agent price comparison system built with LangGraph and LangChain that finds the best prices across the web for any product.

## Features

- **Multi-Agent Architecture**: Six specialized agents working together to find, extract, validate, and compare prices
- **LangGraph Workflow**: Robust state machine with conditional routing and error recovery
- **MCP Integration**: Connects to Model Context Protocol servers for web search and scraping tools
- **Smart Validation**: AI-powered product matching to ensure price accuracy
- **Rich CLI Interface**: Beautiful terminal output with progress indicators and formatted tables
- **Export Options**: Export results to JSON or CSV format
- **Israeli Market Focus**: Optimized for Israeli e-commerce sites with ILS currency support

## Architecture

```
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
│  │ Product           │ ─── Extract product info, generate       │
│  │ Understanding     │     optimized search queries             │
│  │ Agent             │                                          │
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
```

## Agent Descriptions

### 1. Product Understanding Agent
Analyzes user input (product name or URL) and extracts structured product information including brand, model, specifications, and optimized search queries.

### 2. Web Search Agent
Executes multiple search strategies using MCP tools to find product listings across Israeli and international e-commerce sites.

### 3. Price Extraction Agent
Scrapes product pages concurrently to extract pricing information, handling various formats including Schema.org structured data and Hebrew price formats.

### 4. Data Validation Agent
Validates extracted prices against original product specifications using AI-powered matching to ensure accuracy and filter irrelevant results.

### 5. Price Comparison Agent
Ranks validated prices, calculates total costs including shipping, and generates deal scores based on multiple factors.

### 6. Reporting Agent
Generates comprehensive reports with comparison tables, recommendations, and insights about the best time to buy.

## Installation

### Prerequisites
- Python 3.11+
- OpenAI API key
- MCP server for tool integration (web search, scraping)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/price-comparison-app.git
cd price-comparison-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export MCP_SERVER_URL="http://localhost:8000/mcp"
```

5. (Optional) Configure settings:
```bash
cp config/settings.yaml config/settings.local.yaml
# Edit config/settings.local.yaml with your preferences
```

## Usage

### Basic Search

Search by product name:
```bash
python main.py search "iPhone 15 Pro Max 256GB"
```

Search by product URL:
```bash
python main.py search "https://www.amazon.com/dp/B0CHX1W1XY"
```

### Export Results

Export to JSON:
```bash
python main.py search "MacBook Pro M3" --output results.json
```

Export to CSV:
```bash
python main.py search "MacBook Pro M3" --output results.csv --format csv
```

### Options

```bash
python main.py search --help

Usage: main.py search [OPTIONS] QUERY

Arguments:
  QUERY  Product name or URL to search for [required]

Options:
  -o, --output TEXT   Export results to file (JSON or CSV)
  -f, --format TEXT   Output format: json or csv [default: json]
  -v, --verbose       Enable verbose output
  -m, --mock          Use mock MCP client (for testing)
  -c, --config TEXT   Path to custom configuration file
  --help              Show this message and exit
```

### Configuration Management

View current configuration:
```bash
python main.py config --show
```

## Example Output

```
╔═══════════════════════════════════════════════════════════════════╗
║              PRICE COMPARISON - Multi-Agent Search                 ║
║                  Powered by LangGraph & OpenAI                     ║
╚═══════════════════════════════════════════════════════════════════╝

Searching for: iPhone 15 Pro Max 256GB

╭─────────────────────────────────── Summary ───────────────────────────────────╮
│ Found 15 prices for Apple iPhone 15 Pro Max 256GB. Prices range from ₪4,999  │
│ to ₪5,699 with an average of ₪5,299.                                         │
│                                                                               │
│ Best deal at KSP for ₪4,999 (save ₪300 / 5.7% vs average). In stock.        │
╰───────────────────────────────────────────────────────────────────────────────╯

┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Rank ┃ Store              ┃ Price      ┃ Shipping ┃ Total      ┃ Status      ┃ Score  ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩
│ ★ 1  │ KSP                │ ₪4,999     │ Free     │ ₪4,999     │ In Stock    │ 95/100 │
│ 2    │ Ivory              │ ₪5,099     │ ₪29      │ ₪5,128     │ In Stock    │ 88/100 │
│ 3    │ Bug                │ ₪5,149     │ Free     │ ₪5,149     │ Limited     │ 85/100 │
│ 4    │ Eilat              │ ₪5,199     │ Free     │ ₪5,199     │ In Stock    │ 82/100 │
│ 5    │ Amazon IL          │ ₪5,299     │ ₪49      │ ₪5,348     │ In Stock    │ 78/100 │
└──────┴────────────────────┴────────────┴──────────┴────────────┴─────────────┴────────┘

╭──────────────────────────────── Best Time to Buy ─────────────────────────────╮
│ Electronics often see price drops during Black Friday (November), Amazon      │
│ Prime Day (July), and after new model releases.                              │
╰───────────────────────────────────────────────────────────────────────────────╯
```

## Project Structure

```
price-comparison-app/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── product_understanding.py  # Product analysis agent
│   │   ├── web_search.py             # Web search agent
│   │   ├── price_extraction.py       # Price scraping agent
│   │   ├── data_validation.py        # Validation agent
│   │   ├── price_comparison.py       # Comparison agent
│   │   └── reporting.py              # Report generation agent
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py                  # LangGraph state schema
│   │   └── workflow.py               # Workflow definition
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                # Pydantic models
│   ├── mcp_client/
│   │   ├── __init__.py
│   │   └── client.py                 # MCP client implementation
│   └── utils/
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       └── logger.py                 # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Test fixtures
│   ├── test_agents.py                # Agent unit tests
│   ├── test_workflow.py              # Workflow tests
│   └── test_integration.py           # Integration tests
├── config/
│   └── settings.yaml                 # Default configuration
├── main.py                           # CLI entry point
├── pyproject.toml                    # Project metadata
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## Configuration

The application can be configured via `config/settings.yaml` or environment variables:

```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4-turbo-preview
  temperature: 0

mcp:
  server_url: http://localhost:8000/mcp
  timeout: 30

agents:
  max_retries: 3
  timeout: 120

scraping:
  concurrent_requests: 5
  rate_limit_delay: 1.0
```

## Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_agents.py -v
```

## Development

### Code Style

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

Run formatting and linting:
```bash
black src tests
ruff check src tests
mypy src
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

## API Reference

### GraphState

The workflow state contains:
- `input`: Original user input
- `product_info`: Extracted product information
- `search_results`: Web search results
- `price_data`: Extracted prices
- `validated_prices`: Validated prices
- `ranked_results`: Ranked comparison results
- `final_report`: Generated report

### PriceData

Price information model:
- `url`: Product page URL
- `store_name`: Retailer name
- `price`: Product price
- `currency`: Currency code (ILS, USD, etc.)
- `shipping_cost`: Shipping cost
- `availability`: Stock status
- `relevance_score`: Validation score (0-100)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests.

## Support

- Report issues: [GitHub Issues](https://github.com/your-org/price-comparison-app/issues)
- Documentation: [Wiki](https://github.com/your-org/price-comparison-app/wiki)
