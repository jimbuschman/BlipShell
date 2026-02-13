"""Tool base class and registry for native Ollama tool calling."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from blipshell.models.tools import ToolCall, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base class for tools.

    Subclasses define their tool schema and implement execute().
    """

    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool definition for Ollama."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool with given arguments. Returns result string."""
        ...

    def to_ollama_tool(self) -> dict:
        """Convert to Ollama's native tool format."""
        return self.definition().to_ollama_tool()


class ToolRegistry:
    """Registry for dynamic tool registration and execution."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        defn = tool.definition()
        self._tools[defn.name] = tool
        logger.debug("Registered tool: %s", defn.name)

    def unregister(self, name: str):
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_ollama_tools(self) -> list[dict]:
        """Get all tools in Ollama format for the tools parameter."""
        return [tool.to_ollama_tool() for tool in self._tools.values()]

    def get_tool_names(self) -> list[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        tool = self._tools.get(tool_call.name)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=f"Error: Unknown tool '{tool_call.name}'",
                success=False,
            )

        start = time.monotonic()
        try:
            result_str = await tool.execute(**tool_call.arguments)
            elapsed = (time.monotonic() - start) * 1000

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=result_str,
                success=True,
                execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Tool %s failed: %s", tool_call.name, e)

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=f"Error executing {tool_call.name}: {e}",
                success=False,
                execution_time_ms=elapsed,
            )
