"""Tool base class and registry for native Ollama tool calling."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable, Optional

from blipshell.models.tools import ToolCall, ToolDefinition, ToolParameter, ToolParameterType, ToolResult

# Type for the approval callback: (tool_name, arguments) -> approved?
ApprovalCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]

logger = logging.getLogger(__name__)

class Tool(ABC):
    """Abstract base class for tools.

    Subclasses define their tool schema and implement execute().
    """

    read_only: bool = False  # Override to True for tools safe in plan mode

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
        self._tool_groups: dict[str, str] = {}  # tool_name -> group
        self._approval_callback: Optional[ApprovalCallback] = None
        self._tools_requiring_approval: set[str] = set()
        self._plan_mode: bool = False

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

    def set_approval_callback(
        self,
        callback: ApprovalCallback,
        tools_requiring_approval: set[str],
    ):
        """Configure tool approval.

        Args:
            callback: Async function(tool_name, arguments) -> bool.
                      Called before executing tools in the approval set.
                      Returns True to allow, False to deny.
            tools_requiring_approval: Set of tool names that require approval.
        """
        self._approval_callback = callback
        self._tools_requiring_approval = tools_requiring_approval

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

    @property
    def in_plan_mode(self) -> bool:
        """Whether the registry is in plan (read-only) mode."""
        return self._plan_mode

    def get_plan_mode_tools(self) -> list[dict]:
        """Get only read-only tools + exit_plan_mode in Ollama format."""
        return [
            tool.to_ollama_tool()
            for name, tool in self._tools.items()
            if tool.read_only or name == "exit_plan_mode"
        ]

    @staticmethod
    def _coerce_arguments(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
        """Coerce argument types based on the tool's parameter definitions.

        LLMs sometimes send integers as strings (e.g., max_lines="600").
        This converts them to the declared type before calling execute().
        """
        defn = tool.definition()
        param_types = {p.name: p.type for p in defn.parameters}
        coerced = {}
        for key, value in arguments.items():
            expected = param_types.get(key)
            if expected and value is not None:
                try:
                    if expected == ToolParameterType.INTEGER and not isinstance(value, int):
                        coerced[key] = int(value)
                        continue
                    if expected == ToolParameterType.NUMBER and not isinstance(value, (int, float)):
                        coerced[key] = float(value)
                        continue
                    if expected == ToolParameterType.BOOLEAN and not isinstance(value, bool):
                        coerced[key] = str(value).lower() in ("true", "1", "yes")
                        continue
                except (ValueError, TypeError):
                    pass  # Keep original value if conversion fails
            coerced[key] = value
        return coerced

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result.

        If the tool requires approval and a callback is set, the callback
        is invoked first. The tool is only executed if approved.
        """
        tool = self._tools.get(tool_call.name)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=f"Error: Unknown tool '{tool_call.name}'",
                success=False,
            )

        # Check approval for dangerous tools
        if (
            self._approval_callback
            and tool_call.name in self._tools_requiring_approval
        ):
            try:
                approved = await self._approval_callback(
                    tool_call.name, tool_call.arguments,
                )
            except Exception as e:
                logger.error("Approval callback error: %s", e)
                approved = False

            if not approved:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=f"Tool '{tool_call.name}' was denied by the user.",
                    success=False,
                )

        # Coerce argument types — LLMs sometimes send ints as strings
        coerced_args = self._coerce_arguments(tool, tool_call.arguments)

        start = time.monotonic()
        try:
            result_str = await tool.execute(**coerced_args)
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
