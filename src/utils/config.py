"""Configuration management for the price comparison application.

This module handles loading and validating configuration from YAML files
and environment variables using Pydantic settings.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(default="gpt-4-turbo-preview", description="Model to use")
    temperature: float = Field(default=0, ge=0, le=2, description="Temperature")
    max_tokens: int = Field(default=4096, ge=1, description="Max tokens")
    request_timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")


class MCPConfig(BaseModel):
    """MCP server configuration."""

    server_url: str = Field(
        default="http://localhost:8000/mcp", description="MCP server URL"
    )
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, ge=1, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, ge=0, description="Delay between retries")


class SerperConfig(BaseModel):
    """Serper API configuration."""

    api_key: str = Field(default="", description="Serper API key")
    results_per_search: int = Field(default=20, ge=1, le=100, description="Results per search")
    country: str = Field(default="il", description="Country code")
    language: str = Field(default="he", description="Language code")


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

    max_retries: int = Field(default=3, ge=1, description="Maximum retries")
    timeout: int = Field(default=120, ge=1, description="Agent timeout in seconds")
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
            concurrent_requests=5,
        )
    )
    data_validation: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            name="Data Validation Agent",
            description="Validates extracted prices against product specifications",
            min_relevance_score=70,
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

    concurrent_requests: int = Field(default=5, ge=1, description="Concurrent requests")
    rate_limit_delay: float = Field(default=1.0, ge=0, description="Rate limit delay")
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        description="User agent string",
    )
    timeout: int = Field(default=15, ge=1, description="Request timeout in seconds")
    max_page_size: int = Field(default=5242880, ge=1, description="Max page size in bytes")


class WorkflowConfig(BaseModel):
    """Workflow configuration."""

    min_urls_for_extraction: int = Field(default=5, ge=1, description="Min URLs for extraction")
    min_prices_for_validation: int = Field(default=3, ge=1, description="Min prices for validation")
    max_search_retries: int = Field(default=2, ge=0, description="Max search retries")
    checkpoint_enabled: bool = Field(default=True, description="Enable checkpointing")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="structured", description="Log format (structured/plain)")
    file: Optional[str] = Field(None, description="Log file path")


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
    2. Configuration file (settings.yaml)
    3. Default values (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_prefix="PRICE_COMPARE_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    serper: SerperConfig = Field(default_factory=SerperConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    currencies: list[CurrencyConfig] = Field(default_factory=list)
    ecommerce_domains: EcommerceDomains = Field(default_factory=EcommerceDomains)

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from a YAML configuration file.

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

            # Process environment variable substitutions
            config_data = _substitute_env_vars(raw_config)

        # Merge with environment variables (env vars take precedence)
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
    import re
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
    return get_settings()
