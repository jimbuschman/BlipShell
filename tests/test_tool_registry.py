"""Tests for the tool registry (core/tools/base.py)."""

import pytest
from unittest.mock import AsyncMock

from blipshell.core.tools.base import Tool, ToolRegistry
from blipshell.models.tools import ToolCall, ToolDefinition, ToolParameter, ToolParameterType


class DummyTool(Tool):
    """A simple tool for testing."""

    def __init__(self, name="dummy_tool", fail=False):
        self._name = name
        self._fail = fail

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="A dummy tool for testing",
            parameters=[
                ToolParameter(name="arg1", type=ToolParameterType.STRING,
                              description="First argument"),
            ],
        )

    async def execute(self, arg1: str = "", **kwargs) -> str:
        if self._fail:
            raise RuntimeError("Tool execution failed")
        return f"Result: {arg1}"


class TestToolRegistry:
    def test_register_and_list(self):
        registry = ToolRegistry()
        registry.register(DummyTool("tool_a"), group="test")
        registry.register(DummyTool("tool_b"), group="test")
        names = registry.get_tool_names()
        assert "tool_a" in names
        assert "tool_b" in names

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(DummyTool("removable"), group="test")
        assert "removable" in registry.get_tool_names()
        registry.unregister("removable")
        assert "removable" not in registry.get_tool_names()

    def test_get_tool(self):
        registry = ToolRegistry()
        tool = DummyTool("findable")
        registry.register(tool, group="test")
        assert registry.get_tool("findable") is tool
        assert registry.get_tool("nonexistent") is None

    def test_get_all_ollama_tools(self):
        registry = ToolRegistry()
        registry.register(DummyTool("tool_1"), group="test")
        tools = registry.get_all_ollama_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "tool_1"

    def test_get_tools_for_groups(self):
        registry = ToolRegistry()
        registry.register(DummyTool("fs_tool"), group="filesystem")
        registry.register(DummyTool("web_tool"), group="web")
        fs_tools = registry.get_tools_for_groups({"filesystem"})
        assert len(fs_tools) == 1
        assert fs_tools[0]["function"]["name"] == "fs_tool"

    def test_get_tools_for_empty_groups(self):
        registry = ToolRegistry()
        registry.register(DummyTool("tool"), group="test")
        assert registry.get_tools_for_groups(set()) == []

    async def test_execute_tool_call_success(self):
        registry = ToolRegistry()
        registry.register(DummyTool("test_tool"), group="test")
        call = ToolCall(name="test_tool", arguments={"arg1": "hello"})
        result = await registry.execute_tool_call(call)
        assert result.success
        assert "hello" in result.result

    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        call = ToolCall(name="unknown_tool", arguments={})
        result = await registry.execute_tool_call(call)
        assert not result.success
        assert "Unknown tool" in result.result

    async def test_execute_tool_failure(self):
        registry = ToolRegistry()
        registry.register(DummyTool("failing_tool", fail=True), group="test")
        call = ToolCall(name="failing_tool", arguments={"arg1": "test"})
        result = await registry.execute_tool_call(call)
        assert not result.success
        assert "Error" in result.result

    async def test_execution_time_tracked(self):
        registry = ToolRegistry()
        registry.register(DummyTool("timed_tool"), group="test")
        call = ToolCall(name="timed_tool", arguments={"arg1": "test"})
        result = await registry.execute_tool_call(call)
        assert result.execution_time_ms >= 0




class ErrorStringTool(Tool):
    """A tool that reports failure the way real tools do: by RETURNING an
    error string rather than raising."""

    def __init__(self, name="error_tool", result="Error: something broke"):
        self._name = name
        self._result = result

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self._name, description="returns an error string")

    async def execute(self, **kwargs) -> str:
        return self._result


class TestErrorStringsAreFailures:
    """No tool raises to signal failure — ~44 sites return "Error: ...".
    The registry must translate that into success=False, or a failed write
    poisons the executor's file cache and satisfies the completion audit
    (deep-dive 2026-08-04)."""

    async def test_error_prefixed_result_is_failure(self):
        registry = ToolRegistry()
        registry.register(ErrorStringTool(result="Error: 'x.py' does not exist."))
        result = await registry.execute_tool_call(
            ToolCall(id="1", name="error_tool", arguments={})
        )
        assert result.success is False
        assert "does not exist" in result.result

    async def test_error_executing_prefix_is_failure(self):
        registry = ToolRegistry()
        registry.register(ErrorStringTool(result="Error executing command: boom"))
        result = await registry.execute_tool_call(
            ToolCall(id="1", name="error_tool", arguments={})
        )
        assert result.success is False

    async def test_leading_whitespace_still_detected(self):
        registry = ToolRegistry()
        registry.register(ErrorStringTool(result="\n  Error: late newline"))
        result = await registry.execute_tool_call(
            ToolCall(id="1", name="error_tool", arguments={})
        )
        assert result.success is False

    async def test_output_merely_containing_error_is_success(self):
        """The false-positive guard that makes prefix-matching safe:
        run_command relaying stderr and read_file returning a log file both
        contain the word error and are NOT tool failures."""
        registry = ToolRegistry()
        registry.register(ErrorStringTool(
            name="run_command",
            result="ran 12 tests\nFAILED test_x.py - Error: assertion failed\n1 failed",
        ))
        result = await registry.execute_tool_call(
            ToolCall(id="1", name="run_command", arguments={})
        )
        assert result.success is True, (
            "tool output that merely mentions an error must stay a success"
        )

    async def test_empty_result_is_success(self):
        registry = ToolRegistry()
        registry.register(ErrorStringTool(result=""))
        result = await registry.execute_tool_call(
            ToolCall(id="1", name="error_tool", arguments={})
        )
        assert result.success is True

    async def test_normal_result_unaffected(self):
        registry = ToolRegistry()
        registry.register(ErrorStringTool(result="Wrote 42 lines to x.py"))
        result = await registry.execute_tool_call(
            ToolCall(id="1", name="error_tool", arguments={})
        )
        assert result.success is True
