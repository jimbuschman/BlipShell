"""Tests for the tool registry (core/tools/base.py)."""

import pytest
from unittest.mock import AsyncMock

from blipshell.core.tools.base import Tool, ToolRegistry, detect_tool_groups
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


class TestDetectToolGroups:
    def test_filesystem_keywords(self):
        groups = detect_tool_groups("read the file at path/to/file.py")
        assert "filesystem" in groups

    def test_shell_keywords(self):
        groups = detect_tool_groups("run the pip install command")
        assert "shell" in groups

    def test_web_keywords(self):
        groups = detect_tool_groups("search for python tutorials online")
        assert "web" in groups

    def test_memory_keywords(self):
        groups = detect_tool_groups("do you remember what we talked about")
        assert "memory" in groups

    def test_no_match(self):
        groups = detect_tool_groups("what is the meaning of life")
        assert len(groups) == 0

    def test_multiple_groups(self):
        groups = detect_tool_groups("read the file and search online for documentation")
        assert "filesystem" in groups
        assert "web" in groups
