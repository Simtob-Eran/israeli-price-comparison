"""MCP (Model Context Protocol) HTTP Streamable Client.

This module provides an async client for connecting to MCP servers
over HTTP with Server-Sent Events (SSE) support.
"""

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Callable, Optional

import httpx
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCPToolDefinition(BaseModel):
    """Definition of an MCP tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for tool input"
    )


class MCPRequest(BaseModel):
    """MCP JSON-RPC request."""

    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request ID")
    method: str = Field(..., description="Method name")
    params: dict[str, Any] = Field(default_factory=dict, description="Method parameters")


class MCPResponse(BaseModel):
    """MCP JSON-RPC response."""

    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str = Field(..., description="Request ID")
    result: Optional[Any] = Field(None, description="Result data")
    error: Optional[dict[str, Any]] = Field(None, description="Error information")


class MCPError(Exception):
    """MCP-specific error."""

    def __init__(self, code: int, message: str, data: Optional[Any] = None) -> None:
        """Initialize MCP error.

        Args:
            code: Error code.
            message: Error message.
            data: Additional error data.
        """
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP Error {code}: {message}")


class MCPConnectionError(Exception):
    """Error connecting to MCP server."""

    pass


class MCPClient:
    """Async client for MCP HTTP streamable server.

    This client implements the MCP protocol for connecting to MCP servers
    over HTTP with Server-Sent Events (SSE) support for streaming responses.
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize MCP client.

        Args:
            server_url: URL of the MCP server.
            timeout: Request timeout in seconds.
            retry_attempts: Number of retry attempts.
            retry_delay: Base delay between retries in seconds.
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._client: Optional[httpx.AsyncClient] = None
        self._tools: dict[str, MCPToolDefinition] = {}
        self._connected = False
        self._session_id: Optional[str] = None

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Establish connection to MCP server.

        Raises:
            MCPConnectionError: If connection fails.
        """
        if self._connected:
            return

        try:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )

            # Initialize session with the server
            response = await self._send_request(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "sampling": {},
                    },
                    "clientInfo": {
                        "name": "price-comparison-client",
                        "version": "1.0.0",
                    },
                },
            )

            if response.error:
                raise MCPConnectionError(
                    f"Failed to initialize: {response.error.get('message', 'Unknown error')}"
                )

            self._session_id = response.result.get("sessionId") if response.result else None
            self._connected = True

            # Fetch available tools
            await self._refresh_tools()

            logger.info(
                "Connected to MCP server",
                server_url=self.server_url,
                session_id=self._session_id,
                tool_count=len(self._tools),
            )

        except httpx.RequestError as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        self._session_id = None
        self._tools.clear()
        logger.info("Disconnected from MCP server")

    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _send_request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
    ) -> MCPResponse:
        """Send JSON-RPC request to MCP server.

        Args:
            method: Method name.
            params: Method parameters.

        Returns:
            MCP response.

        Raises:
            MCPConnectionError: If not connected.
            httpx.RequestError: If request fails.
        """
        if not self._client:
            raise MCPConnectionError("Not connected to MCP server")

        request = MCPRequest(method=method, params=params or {})
        request_data = request.model_dump()

        logger.debug(
            "Sending MCP request",
            method=method,
            request_id=request.id,
        )

        start_time = time.time()
        response = await self._client.post(
            f"{self.server_url}",
            json=request_data,
        )
        duration_ms = (time.time() - start_time) * 1000

        response.raise_for_status()

        # Handle SSE response
        if "text/event-stream" in response.headers.get("content-type", ""):
            result = await self._parse_sse_response(response)
        else:
            result = response.json()

        logger.debug(
            "Received MCP response",
            method=method,
            request_id=request.id,
            duration_ms=duration_ms,
            has_error=result.get("error") is not None,
        )

        return MCPResponse(**result)

    async def _parse_sse_response(self, response: httpx.Response) -> dict[str, Any]:
        """Parse Server-Sent Events response.

        Args:
            response: HTTP response with SSE content.

        Returns:
            Parsed JSON-RPC response.
        """
        result: dict[str, Any] = {}
        accumulated_data = ""

        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break
                try:
                    accumulated_data += data
                    # Try to parse as complete JSON
                    result = json.loads(accumulated_data)
                    accumulated_data = ""
                except json.JSONDecodeError:
                    # Incomplete JSON, continue accumulating
                    continue
            elif line.startswith("event: "):
                # Handle event types if needed
                pass

        return result

    async def _refresh_tools(self) -> None:
        """Refresh the list of available tools from the server."""
        response = await self._send_request(method="tools/list")

        if response.error:
            logger.warning(
                "Failed to fetch tools",
                error=response.error,
            )
            return

        tools = response.result.get("tools", []) if response.result else []
        self._tools = {
            tool["name"]: MCPToolDefinition(
                name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
            )
            for tool in tools
        }

        logger.info(
            "Loaded MCP tools",
            tool_count=len(self._tools),
            tools=list(self._tools.keys()),
        )

    async def list_tools(self) -> list[MCPToolDefinition]:
        """Get list of available tools from server.

        Returns:
            List of tool definitions.

        Raises:
            MCPConnectionError: If not connected.
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        return list(self._tools.values())

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute MCP tool and return results.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments.
            timeout: Optional custom timeout.

        Returns:
            Tool execution result.

        Raises:
            MCPConnectionError: If not connected.
            MCPError: If tool execution fails.
            ValueError: If tool not found.
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}")

        logger.info(
            "Calling MCP tool",
            tool_name=tool_name,
            arguments=arguments,
        )

        start_time = time.time()

        # Use custom timeout if provided
        if timeout and self._client:
            original_timeout = self._client.timeout
            self._client.timeout = httpx.Timeout(timeout)

        try:
            response = await self._send_request(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments,
                },
            )
        finally:
            # Restore original timeout
            if timeout and self._client:
                self._client.timeout = original_timeout

        duration_ms = (time.time() - start_time) * 1000

        if response.error:
            error = response.error
            raise MCPError(
                code=error.get("code", -1),
                message=error.get("message", "Unknown error"),
                data=error.get("data"),
            )

        logger.info(
            "MCP tool completed",
            tool_name=tool_name,
            duration_ms=duration_ms,
        )

        return response.result

    async def stream_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Execute MCP tool and stream results.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments.

        Yields:
            Tool execution results as they stream in.

        Raises:
            MCPConnectionError: If not connected.
            MCPError: If tool execution fails.
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        if not self._client:
            raise MCPConnectionError("Client not initialized")

        request = MCPRequest(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments,
            },
        )

        async with self._client.stream(
            "POST",
            self.server_url,
            json=request.model_dump(),
            headers={"Accept": "text/event-stream"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        result = json.loads(data)
                        if "error" in result:
                            raise MCPError(
                                code=result["error"].get("code", -1),
                                message=result["error"].get("message", "Unknown error"),
                            )
                        if "result" in result:
                            yield result["result"]
                    except json.JSONDecodeError:
                        continue


class MCPToolWrapper(BaseTool):
    """LangChain tool wrapper for MCP tools.

    This wrapper allows MCP tools to be used with LangChain agents
    seamlessly.
    """

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    mcp_client: MCPClient = Field(..., description="MCP client instance")
    tool_name: str = Field(..., description="MCP tool name")
    args_schema: Optional[type[BaseModel]] = Field(None, description="Arguments schema")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous run - not supported for MCP tools."""
        raise NotImplementedError("MCP tools only support async execution")

    async def _arun(self, **kwargs: Any) -> Any:
        """Execute the MCP tool asynchronously.

        Args:
            **kwargs: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            ToolException: If tool execution fails.
        """
        try:
            result = await self.mcp_client.call_tool(
                tool_name=self.tool_name,
                arguments=kwargs,
            )
            return result
        except MCPError as e:
            raise ToolException(f"MCP tool error: {e.message}") from e
        except Exception as e:
            raise ToolException(f"Tool execution failed: {e}") from e


def create_mcp_tools(
    mcp_client: MCPClient,
    tool_filter: Optional[Callable[[str], bool]] = None,
) -> list[BaseTool]:
    """Create LangChain tools from MCP tool definitions.

    Args:
        mcp_client: Connected MCP client.
        tool_filter: Optional function to filter tools by name.

    Returns:
        List of LangChain tools.
    """
    tools: list[BaseTool] = []

    for tool_def in mcp_client._tools.values():
        # Apply filter if provided
        if tool_filter and not tool_filter(tool_def.name):
            continue

        # Create dynamic args schema from input_schema
        args_schema = None
        if tool_def.input_schema:
            # Convert JSON schema to Pydantic model
            args_schema = _create_args_schema(tool_def.name, tool_def.input_schema)

        tool = MCPToolWrapper(
            name=tool_def.name,
            description=tool_def.description or f"MCP tool: {tool_def.name}",
            mcp_client=mcp_client,
            tool_name=tool_def.name,
            args_schema=args_schema,
        )
        tools.append(tool)

    return tools


def _create_args_schema(tool_name: str, json_schema: dict[str, Any]) -> Optional[type[BaseModel]]:
    """Create a Pydantic model from JSON schema.

    Args:
        tool_name: Tool name for the model.
        json_schema: JSON schema definition.

    Returns:
        Pydantic model class or None.
    """
    if not json_schema.get("properties"):
        return None

    # Map JSON schema types to Python types
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    fields: dict[str, Any] = {}
    required = set(json_schema.get("required", []))

    for prop_name, prop_schema in json_schema.get("properties", {}).items():
        prop_type = type_mapping.get(prop_schema.get("type", "string"), Any)
        description = prop_schema.get("description", "")

        if prop_name in required:
            fields[prop_name] = (prop_type, Field(..., description=description))
        else:
            fields[prop_name] = (Optional[prop_type], Field(None, description=description))

    # Create dynamic Pydantic model
    model_name = f"{tool_name.replace('-', '_').title()}Args"
    return type(model_name, (BaseModel,), {"__annotations__": {k: v[0] for k, v in fields.items()}, **{k: v[1] for k, v in fields.items()}})


# Mock MCP client for testing and development
class MockMCPClient(MCPClient):
    """Mock MCP client for testing without a real server.

    This client provides mock responses for common MCP tools,
    allowing development and testing without a running MCP server.
    """

    def __init__(self) -> None:
        """Initialize mock client."""
        super().__init__(server_url="http://mock-server")
        self._mock_responses: dict[str, Callable[..., Any]] = {}
        self._setup_default_mocks()

    def _setup_default_mocks(self) -> None:
        """Set up default mock responses."""
        # Mock tools
        self._tools = {
            "web_search": MCPToolDefinition(
                name="web_search",
                description="Search the web",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "num_results": {"type": "integer", "description": "Number of results"},
                    },
                    "required": ["query"],
                },
            ),
            "shopping_search": MCPToolDefinition(
                name="shopping_search",
                description="Search for shopping results",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
            "fetch_page_content": MCPToolDefinition(
                name="fetch_page_content",
                description="Fetch content from a web page",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                    },
                    "required": ["url"],
                },
            ),
            "extract_structured_data": MCPToolDefinition(
                name="extract_structured_data",
                description="Extract structured data from HTML",
                input_schema={
                    "type": "object",
                    "properties": {
                        "html": {"type": "string", "description": "HTML content"},
                        "schema_type": {"type": "string", "description": "Schema type"},
                    },
                    "required": ["html"],
                },
            ),
        }

        # Mock responses
        self._mock_responses = {
            "web_search": self._mock_search,
            "shopping_search": self._mock_shopping,
            "fetch_page_content": self._mock_fetch,
            "extract_structured_data": self._mock_extract,
        }

    async def connect(self) -> None:
        """Mock connect - always succeeds."""
        self._connected = True
        logger.info("Connected to mock MCP server")

    async def disconnect(self) -> None:
        """Mock disconnect."""
        self._connected = False

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute mock tool.

        Args:
            tool_name: Tool name.
            arguments: Tool arguments.
            timeout: Optional timeout.

        Returns:
            Mock response.
        """
        if tool_name not in self._mock_responses:
            raise ValueError(f"Unknown mock tool: {tool_name}")

        return self._mock_responses[tool_name](arguments)

    def _mock_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Generate mock search results."""
        query = args.get("query", "")
        return {
            "organic": [
                {
                    "title": f"Mock Result 1 for {query}",
                    "link": "https://www.ksp.co.il/product/1234",
                    "snippet": f"Great price for {query}...",
                },
                {
                    "title": f"Mock Result 2 for {query}",
                    "link": "https://www.ivory.co.il/product/5678",
                    "snippet": f"Best deal on {query}...",
                },
            ],
        }

    def _mock_shopping(self, args: dict[str, Any]) -> dict[str, Any]:
        """Generate mock shopping results."""
        query = args.get("query", "")
        return {
            "shopping": [
                {
                    "title": f"{query} - Store A",
                    "link": "https://www.example.com/product/1",
                    "price": "₪4,999",
                    "source": "Store A",
                },
                {
                    "title": f"{query} - Store B",
                    "link": "https://www.example.com/product/2",
                    "price": "₪5,199",
                    "source": "Store B",
                },
            ],
        }

    def _mock_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        """Generate mock page content."""
        return {
            "content": "<html><body><h1>Mock Product Page</h1><span class='price'>₪4,999</span></body></html>",
            "status_code": 200,
        }

    def _mock_extract(self, args: dict[str, Any]) -> dict[str, Any]:
        """Generate mock extracted data."""
        return {
            "product": {
                "name": "Mock Product",
                "price": 4999,
                "currency": "ILS",
                "availability": "InStock",
            },
        }
