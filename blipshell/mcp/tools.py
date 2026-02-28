"""MCPTool — wraps a single MCP server tool as a BlipShell Tool."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition

if TYPE_CHECKING:
    from blipshell.mcp.manager import MCPManager

logger = logging.getLogger(__name__)


class MCPTool(Tool):
    """Wraps an MCP tool so it can be registered in BlipShell's ToolRegistry.

    Tool calls are routed through MCPManager to the correct server session.
    """

    def __init__(
        self,
        tool_def: ToolDefinition,
        server_name: str,
        original_name: str,
        manager: MCPManager,
        timeout: int = 30,
    ):
        self._definition = tool_def
        self.server_name = server_name
        self.original_name = original_name
        self.manager = manager
        self.timeout = timeout

    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> str:
        try:
            result = await asyncio.wait_for(
                self.manager.call_tool(self.server_name, self.original_name, kwargs),
                timeout=self.timeout,
            )
            return _extract_text(result)
        except asyncio.TimeoutError:
            return (
                f"Error: MCP tool '{self.original_name}' on server '{self.server_name}' "
                f"timed out after {self.timeout}s. The server may be unresponsive."
            )
        except Exception as e:
            return (
                f"Error calling MCP tool '{self.original_name}' on '{self.server_name}': {e}. "
                "Check that the MCP server is running and the arguments are correct."
            )


def _extract_text(result) -> str:
    """Extract text from an MCP CallToolResult."""
    content = getattr(result, "content", None)
    if content is None:
        return str(result)

    texts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
        else:
            mime = getattr(block, "mimeType", "unknown")
            texts.append(f"[Binary data: {mime}]")

    return "\n".join(texts) if texts else "(no output)"
