"""Async stdio MCP client used by the eval suite.

This thin wrapper around the official ``mcp`` Python SDK keeps the rest of the
codebase free of low-level session plumbing. Spin one up with:

    async with MCPClient(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as client:
        tools = await client.list_tools()
        result = await client.call_tool("list_directory", {"path": "/tmp"})
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ToolDef:
    """One tool advertised by an MCP server via ``tools/list``."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    """Outcome of a single ``tools/call`` request."""

    is_error: bool
    content: list[dict[str, Any]]


class MCPClient:
    """Async context manager that owns a stdio MCP session."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._params = StdioServerParameters(
            command=command,
            args=list(args or []),
            env=dict(env) if env else None,
        )
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> MCPClient:
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            read, write = await stack.enter_async_context(stdio_client(self._params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is None:
            return
        try:
            await self._stack.__aexit__(exc_type, exc, tb)
        finally:
            self._stack = None
            self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected. Use 'async with MCPClient(...)'.")
        return self._session

    async def list_tools(self) -> list[ToolDef]:
        session = self._require_session()
        result = await session.list_tools()
        out: list[ToolDef] = []
        for tool in result.tools:
            schema = tool.inputSchema or {}
            out.append(
                ToolDef(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=dict(schema),
                )
            )
        return out

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCallResult:
        session = self._require_session()
        result = await session.call_tool(name, arguments or {})
        content: list[dict[str, Any]] = []
        for block in result.content:
            if hasattr(block, "model_dump"):
                content.append(block.model_dump())
            else:
                content.append({"value": str(block)})
        return ToolCallResult(is_error=bool(result.isError), content=content)
