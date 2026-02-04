"""Logging configuration for the price comparison application.

This module provides structured logging using structlog with support for
both development (colored, readable) and production (JSON) output formats.
"""

import logging
import sys
from functools import lru_cache
from typing import Any, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_format: str = "structured",
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format - "structured" for JSON, "plain" for readable.
        log_file: Optional file path for logging to file.
    """
    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard logging handlers
    handlers: list[logging.Handler] = []

    if log_format == "plain":
        # Rich handler for beautiful console output
        console = Console(stderr=True)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        rich_handler.setLevel(log_level)
        handlers.append(rich_handler)
    else:
        # Standard stream handler for structured output
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(log_level)
        handlers.append(stream_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )

    # Configure structlog
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "structured":
        # JSON output for production
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Colored, readable output for development
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(
                    colors=True,
                    exception_formatter=structlog.dev.rich_traceback,
                ),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


@lru_cache(maxsize=100)
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance for the specified module.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured structlog bound logger.
    """
    return structlog.get_logger(name)


class AgentLogger:
    """Specialized logger for agent operations.

    Provides convenient methods for logging agent-specific events
    with consistent structure.
    """

    def __init__(self, agent_name: str) -> None:
        """Initialize agent logger.

        Args:
            agent_name: Name of the agent.
        """
        self.agent_name = agent_name
        self._logger = get_logger(f"agent.{agent_name}")

    def start(self, task: str, **context: Any) -> None:
        """Log agent task start.

        Args:
            task: Description of the task.
            **context: Additional context to log.
        """
        self._logger.info(
            "Agent starting task",
            agent=self.agent_name,
            task=task,
            **context,
        )

    def complete(self, task: str, duration_ms: float, **context: Any) -> None:
        """Log agent task completion.

        Args:
            task: Description of the task.
            duration_ms: Task duration in milliseconds.
            **context: Additional context to log.
        """
        self._logger.info(
            "Agent completed task",
            agent=self.agent_name,
            task=task,
            duration_ms=duration_ms,
            **context,
        )

    def error(self, task: str, error: Exception, **context: Any) -> None:
        """Log agent error.

        Args:
            task: Description of the task that failed.
            error: Exception that occurred.
            **context: Additional context to log.
        """
        self._logger.error(
            "Agent error",
            agent=self.agent_name,
            task=task,
            error_type=type(error).__name__,
            error_message=str(error),
            **context,
        )

    def retry(self, task: str, attempt: int, max_attempts: int, **context: Any) -> None:
        """Log agent retry attempt.

        Args:
            task: Description of the task being retried.
            attempt: Current attempt number.
            max_attempts: Maximum number of attempts.
            **context: Additional context to log.
        """
        self._logger.warning(
            "Agent retrying task",
            agent=self.agent_name,
            task=task,
            attempt=attempt,
            max_attempts=max_attempts,
            **context,
        )

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message.

        Args:
            message: Debug message.
            **context: Additional context to log.
        """
        self._logger.debug(
            message,
            agent=self.agent_name,
            **context,
        )

    def info(self, message: str, **context: Any) -> None:
        """Log info message.

        Args:
            message: Info message.
            **context: Additional context to log.
        """
        self._logger.info(
            message,
            agent=self.agent_name,
            **context,
        )

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message.

        Args:
            message: Warning message.
            **context: Additional context to log.
        """
        self._logger.warning(
            message,
            agent=self.agent_name,
            **context,
        )


class WorkflowLogger:
    """Specialized logger for workflow operations.

    Provides convenient methods for logging workflow-specific events
    with consistent structure.
    """

    def __init__(self, workflow_name: str = "price_comparison") -> None:
        """Initialize workflow logger.

        Args:
            workflow_name: Name of the workflow.
        """
        self.workflow_name = workflow_name
        self._logger = get_logger(f"workflow.{workflow_name}")

    def start(self, input_data: str, **context: Any) -> None:
        """Log workflow start.

        Args:
            input_data: User input that started the workflow.
            **context: Additional context to log.
        """
        self._logger.info(
            "Workflow started",
            workflow=self.workflow_name,
            input=input_data[:100] + "..." if len(input_data) > 100 else input_data,
            **context,
        )

    def node_enter(self, node_name: str, **context: Any) -> None:
        """Log entering a workflow node.

        Args:
            node_name: Name of the node being entered.
            **context: Additional context to log.
        """
        self._logger.info(
            "Entering workflow node",
            workflow=self.workflow_name,
            node=node_name,
            **context,
        )

    def node_exit(self, node_name: str, duration_ms: float, **context: Any) -> None:
        """Log exiting a workflow node.

        Args:
            node_name: Name of the node being exited.
            duration_ms: Node execution duration in milliseconds.
            **context: Additional context to log.
        """
        self._logger.info(
            "Exiting workflow node",
            workflow=self.workflow_name,
            node=node_name,
            duration_ms=duration_ms,
            **context,
        )

    def transition(self, from_node: str, to_node: str, condition: Optional[str] = None) -> None:
        """Log workflow transition.

        Args:
            from_node: Source node name.
            to_node: Destination node name.
            condition: Optional condition that triggered the transition.
        """
        self._logger.debug(
            "Workflow transition",
            workflow=self.workflow_name,
            from_node=from_node,
            to_node=to_node,
            condition=condition,
        )

    def complete(self, duration_ms: float, success: bool, **context: Any) -> None:
        """Log workflow completion.

        Args:
            duration_ms: Total workflow duration in milliseconds.
            success: Whether the workflow completed successfully.
            **context: Additional context to log.
        """
        level = "info" if success else "error"
        getattr(self._logger, level)(
            "Workflow completed",
            workflow=self.workflow_name,
            duration_ms=duration_ms,
            success=success,
            **context,
        )

    def error(self, error: Exception, node: Optional[str] = None, **context: Any) -> None:
        """Log workflow error.

        Args:
            error: Exception that occurred.
            node: Optional node where the error occurred.
            **context: Additional context to log.
        """
        self._logger.error(
            "Workflow error",
            workflow=self.workflow_name,
            node=node,
            error_type=type(error).__name__,
            error_message=str(error),
            **context,
        )
