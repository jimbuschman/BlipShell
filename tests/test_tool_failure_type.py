"""ToolFailure: tools say "this failed" with a type, not a wording.

The chokepoint used to guess failure from two string prefixes. Web search
and fetch ("Search error:", "Fetch error:"), the workflow tool
("Workflow 'x' failed:") and note persistence ("saved in memory but failed
to persist") never matched, so those failures counted as SUCCESS — the
completion audit accepted turns whose only action had failed. These tests
pin the type at the chokepoint and through the real tools that were wrong.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.core.tools.base import Tool, ToolFailure, ToolRegistry, result_reports_failure
from blipshell.models.tools import ToolCall, ToolDefinition


class _Returns(Tool):
    def __init__(self, result):
        self._result = result

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="t", description="scripted")

    async def execute(self, **kwargs) -> str:
        return self._result


async def _through_registry(tool: Tool, **args):
    reg = ToolRegistry()
    reg.register(tool)
    return await reg.execute_tool_call(
        ToolCall(id="1", name=tool.definition().name, arguments=args)
    )


# --- the type itself -------------------------------------------------------------


def test_toolfailure_is_a_plain_string_to_every_direct_caller():
    f = ToolFailure("Search error: boom")
    assert isinstance(f, str)
    assert f == "Search error: boom"
    assert f.startswith("Search error")
    assert "boom" in f


def test_result_reports_failure_honours_the_type_before_the_prefix():
    assert result_reports_failure(ToolFailure("Search error: boom")) is True
    assert result_reports_failure(ToolFailure("anything at all")) is True
    # legacy backstop still works for un-converted tools
    assert result_reports_failure("Error: legacy") is True
    # and stays narrow: this wording is why ToolFailure exists
    assert result_reports_failure("Search error: boom") is False


def test_rebuilding_the_string_drops_the_marker():
    """Documented sharp edge: wrap at the point of return, not upstream."""
    f = ToolFailure("Error: x")
    assert not isinstance(f"{f}", ToolFailure)
    assert not isinstance(f + "!", ToolFailure)


# --- the chokepoint ---------------------------------------------------------------


async def test_toolfailure_without_error_prefix_is_a_failed_result():
    result = await _through_registry(_Returns(ToolFailure("Search error: no backend")))
    assert result.success is False
    assert result.result == "Search error: no backend"  # the model's text is untouched


async def test_same_wording_as_plain_str_is_still_success():
    """The prefix backstop cannot see this — that is the gap the type closes."""
    result = await _through_registry(_Returns("Search error: no backend"))
    assert result.success is True


async def test_plain_success_string_is_unaffected():
    result = await _through_registry(_Returns("3 files found"))
    assert result.success is True


# --- the real tools that were mis-scored ---------------------------------------


async def test_web_fetch_ssrf_block_is_a_failure():
    from blipshell.core.tools.web import WebFetchTool
    result = await _through_registry(WebFetchTool(timeout=1), url="http://127.0.0.1/admin")
    assert result.success is False
    assert "localhost" in result.result


async def test_web_fetch_unreachable_host_is_a_failure():
    """'Fetch error:' — real I/O: .invalid is a reserved TLD that never resolves."""
    from blipshell.core.tools.web import WebFetchTool
    result = await _through_registry(WebFetchTool(timeout=2), url="http://blipshell-test.invalid/")
    assert result.success is False
    assert result.result.startswith("Fetch error:")


async def test_workflow_not_found_and_failed_are_failures():
    from blipshell.core.tools.task_tools import RunWorkflowTool
    missing = MagicMock()
    missing.run_workflow = AsyncMock(side_effect=KeyError("nope"))
    r = await _through_registry(RunWorkflowTool(missing), workflow_name="nope")
    assert r.success is False and "not found" in r.result

    broken = MagicMock()
    broken.run_workflow = AsyncMock(side_effect=RuntimeError("step 2 died"))
    r = await _through_registry(RunWorkflowTool(broken), workflow_name="w")
    assert r.success is False and "step 2 died" in r.result

    r = await _through_registry(RunWorkflowTool(broken), workflow_name="w", params="{not json")
    assert r.success is False and "Invalid params JSON" in r.result


async def test_note_that_failed_to_persist_is_a_failure():
    """Saved in the in-memory dict but not in the DB: gone next session. The
    model must be told, and the turn must not count it as done."""
    from blipshell.core.tools.note_tools import SaveNoteTool
    from blipshell.models.config import NotesConfig
    sqlite = MagicMock()
    sqlite.save_session_notes = AsyncMock(side_effect=RuntimeError("disk full"))
    notes: dict = {}
    r = await _through_registry(
        SaveNoteTool(sqlite, session_id=1, notes_config=NotesConfig(), notes=notes),
        name="plan", content="ship it",
    )
    assert r.success is False
    assert "failed to persist" in r.result and "disk full" in r.result
    assert notes == {"plan": "ship it"}  # in-memory save still happened


async def test_filesystem_helper_failures_keep_the_type_through_the_caller(tmp_path):
    """_validate_within_root / _check_symlink return the failure; the tool
    passes it straight back — the type must survive that hop."""
    from blipshell.core.tools.filesystem import WriteFileTool
    tool = WriteFileTool(root_path=str(tmp_path))
    r = await _through_registry(tool, path="../../outside.txt", content="x")
    assert r.success is False
    assert "escapes the project root" in r.result or "blocked" in r.result.lower()


async def test_memory_fs_directory_failures_are_failures():
    """'Cannot edit a directory' never matched the prefix backstop either."""
    from blipshell.core.tools.memory_fs import MemoryStrReplaceTool
    tool = MemoryStrReplaceTool(MagicMock(), MagicMock())
    r = await _through_registry(tool, path="/memories/core/", old_text="a", new_text="b")
    assert r.success is False
    assert r.result.startswith("Cannot edit a directory")
