"""Configuration management for the price comparison application.

This module handles loading and validating configuration from environment
variables (.env file) and YAML files using Pydantic settings.

Configuration Priority (highest to lowest):
1. Environment variables
2. .env file
3. config/settings.yaml
4. Default values
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Load .env file from project root
def _load_env_file() -> None:
    """Load environment variables from .env file.

    Searches for .env file in the following locations:
    1. Current working directory
    2. Project root (relative to this module)
    """
    # Try current working directory first
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        return

    # Try project root (relative to this module)
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


# Load .env on module import
_load_env_file()


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key",
    )
    model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
        description="Model to use",
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0")),
        ge=0,
        le=2,
        description="Temperature",
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
        ge=1,
        description="Max tokens",
    )
    request_timeout: int = Field(
        default_factory=lambda: int(os.getenv("OPENAI_REQUEST_TIMEOUT", "60")),
        ge=1,
        description="Request timeout in seconds",
    )


class MCPConfig(BaseModel):
    """MCP server configuration."""

    server_url: str = Field(
        default_factory=lambda: os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp"),
        description="MCP server URL",
    )
    timeout: int = Field(
        default_factory=lambda: int(os.getenv("MCP_TIMEOUT", "30")),
        ge=1,
        description="Request timeout in seconds",
    )
    retry_attempts: int = Field(
        default_factory=lambda: int(os.getenv("MCP_RETRY_ATTEMPTS", "3")),
        ge=1,
        description="Number of retry attempts",
    )
    retry_delay: float = Field(
        default_factory=lambda: float(os.getenv("MCP_RETRY_DELAY", "1.0")),
        ge=0,
        description="Delay between retries",
    )


class AgentConfig(BaseModel):
    """Individual agent configuration."""

    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    min_urls: Optional[int] = Field(None, description="Minimum URLs (web search)")
    max_urls: Optional[int] = Field(None, description="Maximum URLs (web search)")
    concurrent_requests: Optional[int] = Field(None, description="Concurrent requests")
    min_relevance_score: Optional[int] = Field(None, description="Min relevance score")


class AgentsConfig(BaseModel):
    """Agents configuration."""

    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_RETRIES", "3")),
        ge=1,
        description="Maximum retries",
    )
    timeout: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_TIMEOUT", "120")),
        ge=1,
        description="Agent timeout in seconds",
    )
    product_understanding: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            name="Product Understanding Agent",
            description="Analyzes product input and extracts structured information",
        )
    )
    web_search: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            name="Web Search Agent",
            description="Searches the web for product listings",
            min_urls=10,
            max_urls=30,
        )
    )
    price_extraction: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            name="Price Extraction Agent",
            description="Extracts pricing information from web pages",
            concurrent_requests=int(os.getenv("SCRAPING_CONCURRENT_REQUESTS", "5")),
        )
    )
    data_validation: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            name="Data Validation Agent",
            description="Validates extracted prices against product specifications",
            min_relevance_score=int(os.getenv("VALIDATION_MIN_RELEVANCE_SCORE", "70")),
        )
    )
    price_comparison: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            name="Price Comparison Agent",
            description="Analyzes and ranks validated prices",
        )
    )
    reporting: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            name="Reporting Agent",
            description="Generates final user-friendly reports",
        )
    )


class ScrapingConfig(BaseModel):
    """Web scraping configuration."""

    concurrent_requests: int = Field(
        default_factory=lambda: int(os.getenv("SCRAPING_CONCURRENT_REQUESTS", "5")),
        ge=1,
        description="Concurrent requests",
    )
    rate_limit_delay: float = Field(
        default_factory=lambda: float(os.getenv("SCRAPING_RATE_LIMIT_DELAY", "1.0")),
        ge=0,
        description="Rate limit delay",
    )
    user_agent: str = Field(
        default_factory=lambda: os.getenv(
            "SCRAPING_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
        description="User agent string",
    )
    timeout: int = Field(
        default_factory=lambda: int(os.getenv("SCRAPING_TIMEOUT", "15")),
        ge=1,
        description="Request timeout in seconds",
    )
    max_page_size: int = Field(
        default_factory=lambda: int(os.getenv("SCRAPING_MAX_PAGE_SIZE", "5242880")),
        ge=1,
        description="Max page size in bytes",
    )


class WorkflowConfig(BaseModel):
    """Workflow configuration."""

    min_urls_for_extraction: int = Field(
        default_factory=lambda: int(os.getenv("WORKFLOW_MIN_URLS", "5")),
        ge=1,
        description="Min URLs for extraction",
    )
    min_prices_for_validation: int = Field(
        default_factory=lambda: int(os.getenv("WORKFLOW_MIN_PRICES", "3")),
        ge=1,
        description="Min prices for validation",
    )
    max_search_retries: int = Field(
        default_factory=lambda: int(os.getenv("WORKFLOW_MAX_SEARCH_RETRIES", "2")),
        ge=0,
        description="Max search retries",
    )
    checkpoint_enabled: bool = Field(
        default_factory=lambda: os.getenv("WORKFLOW_CHECKPOINT_ENABLED", "true").lower() == "true",
        description="Enable checkpointing",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"),
        description="Log level",
    )
    format: str = Field(
        default_factory=lambda: os.getenv("LOG_FORMAT", "structured"),
        description="Log format (structured/plain)",
    )
    file: Optional[str] = Field(
        default_factory=lambda: os.getenv("LOG_FILE") or None,
        description="Log file path",
    )


class CurrencyConfig(BaseModel):
    """Currency configuration."""

    symbol: str = Field(..., description="Currency symbol")
    code: str = Field(..., description="Currency code")
    name: str = Field(..., description="Currency name")


class EcommerceDomains(BaseModel):
    """E-commerce domain configuration."""

    israel: list[str] = Field(default_factory=list, description="Israeli domains")
    international: list[str] = Field(default_factory=list, description="International domains")


class Settings(BaseSettings):
    """Application settings loaded from environment and config files.

    Settings are loaded in the following order of precedence:
    1. Environment variables (highest priority)
    2. .env file
    3. Configuration file (settings.yaml)
    4. Default values (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    currencies: list[CurrencyConfig] = Field(default_factory=list)
    ecommerce_domains: EcommerceDomains = Field(default_factory=EcommerceDomains)

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from a YAML configuration file.

        Environment variables and .env file values take precedence over
        YAML configuration.

        Args:
            config_path: Path to the configuration file. If not provided,
                        looks for config/settings.yaml relative to the project root.

        Returns:
            Settings instance with loaded configuration.
        """
        if config_path is None:
            # Find the config file relative to this module
            module_dir = Path(__file__).parent.parent.parent
            config_path = module_dir / "config" / "settings.yaml"

        config_data: dict[str, Any] = {}

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}

            # Process environment variable substitutions in YAML
            config_data = _substitute_env_vars(raw_config)

        # Create settings - environment variables take precedence
        return cls(**config_data)

    def get_all_ecommerce_domains(self) -> list[str]:
        """Get all e-commerce domains (Israeli and international)."""
        return self.ecommerce_domains.israel + self.ecommerce_domains.international

    def get_currency_by_code(self, code: str) -> Optional[CurrencyConfig]:
        """Get currency configuration by code."""
        for currency in self.currencies:
            if currency.code.upper() == code.upper():
                return currency
        return None

    def get_currency_by_symbol(self, symbol: str) -> Optional[CurrencyConfig]:
        """Get currency configuration by symbol."""
        for currency in self.currencies:
            if currency.symbol == symbol:
                return currency
        return None

    def print_config_summary(self) -> str:
        """Generate a summary of current configuration for debugging.

        Returns:
            Formatted string with configuration summary.
        """
        lines = [
            "Configuration Summary:",
            "-" * 40,
            f"OpenAI Model: {self.openai.model}",
            f"OpenAI API Key: {'[SET]' if self.openai.api_key else '[NOT SET]'}",
            f"MCP Server URL: {self.mcp.server_url}",
            f"Log Level: {self.logging.level}",
            f"Agent Max Retries: {self.agents.max_retries}",
            f"Scraping Concurrent Requests: {self.scraping.concurrent_requests}",
            "-" * 40,
        ]
        return "\n".join(lines)


def _substitute_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively substitute environment variables in configuration.

    Supports ${VAR_NAME} syntax for environment variable substitution.

    Args:
        config: Configuration dictionary to process.

    Returns:
        Configuration with environment variables substituted.
    """
    result: dict[str, Any] = {}

    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _substitute_env_vars(value)
        elif isinstance(value, list):
            result[key] = [
                _substitute_env_vars(item) if isinstance(item, dict)
                else _substitute_single_value(item)
                for item in value
            ]
        elif isinstance(value, str):
            result[key] = _substitute_single_value(value)
        else:
            result[key] = value

    return result


def _substitute_single_value(value: Any) -> Any:
    """Substitute environment variables in a single value.

    Args:
        value: Value to process.

    Returns:
        Value with environment variables substituted.
    """
    if not isinstance(value, str):
        return value

    # Handle ${VAR_NAME} syntax
    pattern = r"\$\{([^}]+)\}"

    def replace_env_var(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return re.sub(pattern, replace_env_var, value)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    This function uses LRU cache to ensure settings are only loaded once
    and reused throughout the application lifetime.

    Returns:
        Cached Settings instance.
    """
    return Settings.from_yaml()


def reload_settings() -> Settings:
    """Reload settings, clearing the cache.

    Use this when configuration needs to be refreshed (e.g., after
    environment variable changes).

    Returns:
        Fresh Settings instance.
    """
    get_settings.cache_clear()
    _load_env_file()  # Reload .env file
    return get_settings()
