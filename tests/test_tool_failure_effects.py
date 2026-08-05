"""What a FAILED tool call must not do to downstream state.

The registry now reports "Error: ..." returns as success=False (see
test_tool_registry.TestErrorStringsAreFailures). These tests pin the
consequences that made the bug worth fixing: before, a failed write_file
still cached the content it never wrote, so a later read_file was served
phantom text from cache and the model "verified" work that didn't exist.

Real ToolRegistry, real TaskExecutor callback, no LLM.
"""

from unittest.mock import MagicMock

import pytest

from blipshell.core.executor import TaskExecutor
from blipshell.core.tools.base import Tool, ToolRegistry
from blipshell.models.config import PlannerConfig
from blipshell.models.tools import ToolCall, ToolDefinition


class _ScriptedTool(Tool):
    def __init__(self, name: str, result: str):
        self._name = name
        self._result = result

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self._name, description="scripted")

    async def execute(self, **kwargs) -> str:
        return self._result


def _executor(registry: ToolRegistry) -> TaskExecutor:
    return TaskExecutor(
        router=MagicMock(),
        sqlite=MagicMock(),
        tool_registry=registry,
        config=PlannerConfig(),
    )


async def _run(registry: ToolRegistry, executor: TaskExecutor, name: str, args: dict):
    """Execute through the registry, then feed the result to the executor
    callback exactly as ChatLoop does."""
    result = await registry.execute_tool_call(
        ToolCall(id="1", name=name, arguments=args)
    )
    executor._on_tool_executed(name, args, result)
    return result


class TestFailedWriteDoesNotPoisonCache:
    async def test_failed_write_content_not_cached(self):
        registry = ToolRegistry()
        registry.register(_ScriptedTool(
            "write_file", "Error: Permission denied for '/etc/hosts'."
        ))
        executor = _executor(registry)

        result = await _run(
            registry, executor, "write_file",
            {"path": "/etc/hosts", "content": "content that was never written"},
        )

        assert result.success is False
        assert "/etc/hosts" not in executor._file_cache, (
            "failed write cached content that was never written — a later "
            "read_file would be served this phantom text"
        )
        assert "/etc/hosts" not in executor._step_files_created

    async def test_successful_write_still_caches(self):
        """The fix must not break the working case."""
        registry = ToolRegistry()
        registry.register(_ScriptedTool("write_file", "Wrote 3 lines to a.py"))
        executor = _executor(registry)

        await _run(
            registry, executor, "write_file",
            {"path": "a.py", "content": "print(1)"},
        )

        assert executor._file_cache.get("a.py") == "print(1)"
        assert "a.py" in executor._step_files_created


class TestFailedReadAndEdit:
    async def test_failed_read_not_marked_as_read(self):
        """files_read drives the look-before-review completion gate — a
        failed read must not satisfy it."""
        registry = ToolRegistry()
        registry.register(_ScriptedTool(
            "read_file", "Error: 'missing.py' does not exist. Use list_directory."
        ))
        executor = _executor(registry)

        await _run(registry, executor, "read_file", {"path": "missing.py"})

        assert "missing.py" not in executor.files_read

    async def test_failed_edit_leaves_cache_intact(self):
        """A failed edit must not evict the cached content, because the file
        on disk is unchanged — evicting it would force a needless re-read and
        (worse) imply the file had changed."""
        registry = ToolRegistry()
        registry.register(_ScriptedTool(
            "edit_file", "Error: old_text not found in 'a.py'."
        ))
        executor = _executor(registry)
        executor._file_cache["a.py"] = "original content"
        executor.files_read.add("a.py")

        await _run(
            registry, executor, "edit_file",
            {"path": "a.py", "old_text": "nope", "new_text": "x"},
        )

        assert executor._file_cache.get("a.py") == "original content"
        assert "a.py" not in executor._stale_files, (
            "a failed edit marked the file stale — nothing changed on disk"
        )
