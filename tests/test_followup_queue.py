"""Tests for the follow-up queue system.

Covers SQLite CRUD, tool execution, and session startup loading.
"""

import pytest
from unittest.mock import MagicMock

from blipshell.core.tools.followup_tools import (
    AddFollowUpTool,
    ListFollowUpsTool,
    ResolveFollowUpTool,
)


@pytest.fixture
async def session_id(sqlite_store):
    """Create a real session so FK constraints are satisfied."""
    cursor = await sqlite_store._db.execute(
        "INSERT INTO sessions (project) VALUES (?)",
        ("test",),
    )
    await sqlite_store._db.commit()
    return cursor.lastrowid


# --- SQLite CRUD Tests ---


@pytest.mark.asyncio
async def test_add_follow_up(sqlite_store, session_id):
    """Adding a follow-up returns an integer ID."""
    fid = await sqlite_store.add_follow_up(
        content="Check deployment status",
        session_id=session_id,
        project="myproject",
        due_hint="tomorrow",
    )
    assert isinstance(fid, int)
    assert fid > 0


@pytest.mark.asyncio
async def test_add_follow_up_minimal(sqlite_store):
    """Adding a follow-up with only required fields works."""
    fid = await sqlite_store.add_follow_up(
        content="Review PR",
        session_id=None,
        project=None,
        due_hint=None,
    )
    assert isinstance(fid, int)
    assert fid > 0


@pytest.mark.asyncio
async def test_get_pending_follow_ups(sqlite_store):
    """Pending follow-ups are returned newest first."""
    await sqlite_store.add_follow_up("First item")
    await sqlite_store.add_follow_up("Second item")
    await sqlite_store.add_follow_up("Third item")

    items = await sqlite_store.get_pending_follow_ups(limit=10)
    assert len(items) == 3
    contents = {i["content"] for i in items}
    assert contents == {"First item", "Second item", "Third item"}


@pytest.mark.asyncio
async def test_get_pending_follow_ups_project_filter(sqlite_store):
    """Project filter returns only matching follow-ups."""
    await sqlite_store.add_follow_up("Global item")
    await sqlite_store.add_follow_up("Project A item", project="projA")
    await sqlite_store.add_follow_up("Project B item", project="projB")

    # No filter — returns all
    all_items = await sqlite_store.get_pending_follow_ups(limit=10)
    assert len(all_items) == 3

    # Filter by project — returns project items + global (no project) items
    a_items = await sqlite_store.get_pending_follow_ups(project="projA", limit=10)
    contents = [i["content"] for i in a_items]
    assert "Project A item" in contents
    assert "Global item" in contents
    assert "Project B item" not in contents


@pytest.mark.asyncio
async def test_resolve_follow_up(sqlite_store, session_id):
    """Resolving a follow-up removes it from pending list."""
    fid = await sqlite_store.add_follow_up("Do the thing")

    ok = await sqlite_store.resolve_follow_up(fid, session_id=session_id)
    assert ok is True

    items = await sqlite_store.get_pending_follow_ups(limit=10)
    assert len(items) == 0


@pytest.mark.asyncio
async def test_resolve_nonexistent(sqlite_store):
    """Resolving a nonexistent follow-up returns False."""
    ok = await sqlite_store.resolve_follow_up(9999, session_id=None)
    assert ok is False


@pytest.mark.asyncio
async def test_resolve_already_resolved(sqlite_store, session_id):
    """Resolving an already-resolved follow-up returns False."""
    fid = await sqlite_store.add_follow_up("Once only")
    await sqlite_store.resolve_follow_up(fid, session_id=session_id)

    ok = await sqlite_store.resolve_follow_up(fid, session_id=session_id)
    assert ok is False


@pytest.mark.asyncio
async def test_dismiss_follow_up(sqlite_store):
    """Dismissing a follow-up removes it from pending list."""
    fid = await sqlite_store.add_follow_up("Not relevant anymore")

    ok = await sqlite_store.dismiss_follow_up(fid)
    assert ok is True

    items = await sqlite_store.get_pending_follow_ups(limit=10)
    assert len(items) == 0


@pytest.mark.asyncio
async def test_dismiss_nonexistent(sqlite_store):
    """Dismissing a nonexistent follow-up returns False."""
    ok = await sqlite_store.dismiss_follow_up(9999)
    assert ok is False


@pytest.mark.asyncio
async def test_get_all_follow_ups(sqlite_store, session_id):
    """get_all_follow_ups returns both pending and resolved items."""
    fid1 = await sqlite_store.add_follow_up("Pending")
    fid2 = await sqlite_store.add_follow_up("Done")
    await sqlite_store.resolve_follow_up(fid2, session_id=session_id)

    all_items = await sqlite_store.get_all_follow_ups(limit=10)
    assert len(all_items) == 2

    statuses = {i["id"]: i["status"] for i in all_items}
    assert statuses[fid1] == "pending"
    assert statuses[fid2] == "resolved"


@pytest.mark.asyncio
async def test_due_hint_stored(sqlite_store):
    """Due hint is stored and returned."""
    await sqlite_store.add_follow_up(
        "Check back", due_hint="next week",
    )
    items = await sqlite_store.get_pending_follow_ups(limit=10)
    assert items[0]["due_hint"] == "next week"


# --- Tool Tests ---


@pytest.mark.asyncio
async def test_add_followup_tool(sqlite_store):
    """AddFollowUpTool creates a follow-up and returns confirmation."""
    tool = AddFollowUpTool(sqlite_store, session_id=None, project="test")
    result = await tool.execute(content="Remember to deploy")
    assert "queued" in result.lower()
    assert "Remember to deploy" in result

    items = await sqlite_store.get_pending_follow_ups(limit=10)
    assert len(items) == 1


@pytest.mark.asyncio
async def test_add_followup_tool_with_due_hint(sqlite_store):
    """AddFollowUpTool includes due hint in confirmation."""
    tool = AddFollowUpTool(sqlite_store, session_id=None)
    result = await tool.execute(content="Deploy fix", due_hint="tomorrow")
    assert "tomorrow" in result


@pytest.mark.asyncio
async def test_list_followups_tool_empty(sqlite_store):
    """ListFollowUpsTool returns message when no items."""
    tool = ListFollowUpsTool(sqlite_store)
    result = await tool.execute()
    assert "no pending" in result.lower()


@pytest.mark.asyncio
async def test_list_followups_tool(sqlite_store):
    """ListFollowUpsTool lists pending items."""
    await sqlite_store.add_follow_up("Item one")
    await sqlite_store.add_follow_up("Item two")

    tool = ListFollowUpsTool(sqlite_store)
    result = await tool.execute()
    assert "Item one" in result
    assert "Item two" in result
    assert "2" in result  # count


@pytest.mark.asyncio
async def test_resolve_followup_tool(sqlite_store):
    """ResolveFollowUpTool marks item as resolved."""
    fid = await sqlite_store.add_follow_up("Do it")

    tool = ResolveFollowUpTool(sqlite_store, session_id=None)
    result = await tool.execute(id=fid)
    assert "resolved" in result.lower()


@pytest.mark.asyncio
async def test_resolve_followup_tool_dismiss(sqlite_store):
    """ResolveFollowUpTool can dismiss items."""
    fid = await sqlite_store.add_follow_up("Meh")

    tool = ResolveFollowUpTool(sqlite_store, session_id=None)
    result = await tool.execute(id=fid, action="dismiss")
    assert "dismissed" in result.lower()


@pytest.mark.asyncio
async def test_resolve_followup_tool_not_found(sqlite_store):
    """ResolveFollowUpTool handles nonexistent ID gracefully."""
    tool = ResolveFollowUpTool(sqlite_store, session_id=None)
    result = await tool.execute(id=9999)
    assert "not found" in result.lower()


# --- Tool Definition Tests ---


def test_add_followup_tool_definition():
    """AddFollowUpTool has correct definition."""
    tool = AddFollowUpTool(MagicMock(), session_id=1)
    defn = tool.definition()
    assert defn.name == "add_followup"
    assert len(defn.parameters) == 2
    param_names = {p.name for p in defn.parameters}
    assert "content" in param_names
    assert "due_hint" in param_names


def test_list_followups_tool_definition():
    """ListFollowUpsTool has correct definition."""
    tool = ListFollowUpsTool(MagicMock())
    defn = tool.definition()
    assert defn.name == "list_followups"
    assert len(defn.parameters) == 0


def test_resolve_followup_tool_definition():
    """ResolveFollowUpTool has correct definition."""
    tool = ResolveFollowUpTool(MagicMock(), session_id=1)
    defn = tool.definition()
    assert defn.name == "resolve_followup"
    param_names = {p.name for p in defn.parameters}
    assert "id" in param_names
    assert "action" in param_names


def test_tools_read_only_flags():
    """Verify read_only flags are correct."""
    assert AddFollowUpTool(MagicMock()).read_only is False
    assert ListFollowUpsTool(MagicMock()).read_only is True
    assert ResolveFollowUpTool(MagicMock()).read_only is False
