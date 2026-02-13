"""Tool base class and registry for native Ollama tool calling."""

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from blipshell.models.tools import ToolCall, ToolDefinition, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Keyword patterns for detecting which tool groups a message needs.
# If none match, no tools are passed (pure conversation = fast).
TOOL_GROUP_PATTERNS: dict[str, re.Pattern] = {
    "filesystem": re.compile(
        r"\b(read|write|edit|create|open|save|file|folder|directory|list\s+dir|ls\b|"
        r"cat\b|path|\.py\b|\.js\b|\.ts\b|\.txt\b|\.json\b|\.yaml\b|\.md\b|\.csv\b)",
        re.IGNORECASE,
    ),
    "shell": re.compile(
        r"\b(run|execute|command|shell|terminal|pip\b|git\b|npm\b|python\b|"
        r"install|compile|build|make\b|cargo\b|cmake\b)",
        re.IGNORECASE,
    ),
    "web": re.compile(
        r"\b(search|google|look\s*up|web|fetch|url|http|browse|website|online)",
        re.IGNORECASE,
    ),
    "memory": re.compile(
        r"\b(remember|recall|forgot|memory|memories|session|previous|last\s+time|"
        r"we\s+talked|you\s+said|i\s+told\s+you|do\s+you\s+know|save\s+this)",
        re.IGNORECASE,
    ),
}


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


def detect_tool_groups(message: str) -> set[str]:
    """Detect which tool groups a message might need based on keywords.

    Returns a set of group names like {"filesystem", "shell"}, or empty set
    if the message is pure conversation (no tools needed).
    """
    groups = set()
    for group, pattern in TOOL_GROUP_PATTERNS.items():
        if pattern.search(message):
            groups.add(group)
    return groups


class ToolRegistry:
    """Registry for dynamic tool registration and execution."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._tool_groups: dict[str, str] = {}  # tool_name -> group

    def register(self, tool: Tool, group: str = "general"):
        """Register a tool with a group name."""
        defn = tool.definition()
        self._tools[defn.name] = tool
        self._tool_groups[defn.name] = group
        logger.debug("Registered tool: %s (group: %s)", defn.name, group)

    def unregister(self, name: str):
        """Unregister a tool by name."""
        self._tools.pop(name, None)
        self._tool_groups.pop(name, None)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_ollama_tools(self) -> list[dict]:
        """Get all tools in Ollama format for the tools parameter."""
        return [tool.to_ollama_tool() for tool in self._tools.values()]

    def get_tools_for_groups(self, groups: set[str]) -> list[dict]:
        """Get only tools matching the given groups in Ollama format."""
        if not groups:
            return []
        return [
            tool.to_ollama_tool()
            for name, tool in self._tools.items()
            if self._tool_groups.get(name) in groups
        ]

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
