"""MCPManager — manages MCP server connections, tool discovery, and call routing."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from blipshell.mcp.schema import mcp_tool_to_definition
from blipshell.models.tools import ToolDefinition

if TYPE_CHECKING:
    from blipshell.models.config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages MCP server connections and tool routing.

    Each MCP server runs as a subprocess (stdio transport). The manager
    handles lifecycle, tool discovery, and call routing.
    """

    def __init__(self):
        self._sessions: dict[str, Any] = {}  # name -> ClientSession
        self._exit_stacks: dict[str, Any] = {}  # name -> AsyncExitStack
        self._server_configs: dict[str, MCPServerConfig] = {}
        self._server_tools: dict[str, dict[str, Any]] = {}  # name -> {tool_name: mcp_tool}

    async def connect_server(self, config: MCPServerConfig) -> list[ToolDefinition]:
        """Connect to an MCP server and discover its tools.

        Uses stdio transport: starts the server as a subprocess and communicates
        via stdin/stdout using JSON-RPC 2.0.

        Returns list of BlipShell ToolDefinitions (with prefixed names).
        """
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = dict(os.environ)
        for key, value in config.env.items():
            env[key] = _resolve_env(value)

        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=env if config.env else None,
        )

        # Use AsyncExitStack to manage the nested context managers
        stack = AsyncExitStack()
        try:
            transport = await stack.enter_async_context(stdio_client(params))
            read_stream, write_stream = transport

            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()

            # Discover tools
            tools_result = await session.list_tools()
            server_tools: dict[str, Any] = {}
            definitions: list[ToolDefinition] = []

            for mcp_tool in tools_result.tools:
                server_tools[mcp_tool.name] = mcp_tool
                defn = mcp_tool_to_definition(mcp_tool, config.name)
                definitions.append(defn)

            self._sessions[config.name] = session
            self._exit_stacks[config.name] = stack
            self._server_configs[config.name] = config
            self._server_tools[config.name] = server_tools

            logger.info(
                "MCP server '%s' connected: %d tools discovered",
                config.name, len(definitions),
            )
            return definitions

        except Exception:
            await stack.aclose()
            raise

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict,
    ) -> Any:
        """Call a tool on a specific MCP server.

        Args:
            server_name: Name of the connected server.
            tool_name: Original (unprefixed) tool name.
            arguments: Tool arguments as a dict.

        Returns:
            MCP CallToolResult.
        """
        session = self._sessions.get(server_name)
        if not session:
            raise RuntimeError(f"MCP server '{server_name}' not connected")

        return await session.call_tool(tool_name, arguments)

    async def disconnect_server(self, name: str):
        """Disconnect from an MCP server and clean up its subprocess."""
        stack = self._exit_stacks.pop(name, None)
        self._sessions.pop(name, None)
        self._server_configs.pop(name, None)
        self._server_tools.pop(name, None)

        if stack:
            try:
                await stack.aclose()
            except Exception as e:
                logger.warning("Error disconnecting MCP server '%s': %s", name, e)

        logger.info("MCP server '%s' disconnected", name)

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        names = list(self._sessions.keys())
        for name in names:
            await self.disconnect_server(name)

    def get_connected_servers(self) -> list[str]:
        """Get names of connected servers."""
        return list(self._sessions.keys())

    def get_server_tool_count(self, name: str) -> int:
        """Get number of tools for a connected server."""
        return len(self._server_tools.get(name, {}))

    def get_server_tool_names(self, name: str) -> list[str]:
        """Get original (unprefixed) tool names for a server."""
        return list(self._server_tools.get(name, {}).keys())


def _resolve_env(value: str) -> str:
    """Expand ${ENV_VAR} syntax in a string."""
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value
