"""Tests for the friction detection system.

Covers: prompt generation, response parsing, SQLite CRUD, nightly job integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from blipshell.llm.prompts import analyze_session_friction, idle_friction_probe
from blipshell.memory.processor import MemoryProcessor


# --- Prompt Tests ---


class TestAnalyzeSessionFrictionPrompt:
    def test_returns_system_and_user(self):
        system, user = analyze_session_friction("summary", "conversation")
        assert "friction" in system.lower()
        assert "summary" in user
        assert "conversation" in user

    def test_includes_categories(self):
        system, _ = analyze_session_friction("s", "c")
        assert "TOOL_FAILURE" in system
        assert "REPEATED_RETRY" in system
        assert "MISSING_CAPABILITY" in system
        assert "WORKFLOW_FRICTION" in system
        assert "CONTEXT_ISSUE" in system

    def test_includes_project_context(self):
        _, user = analyze_session_friction("s", "c", project="myproj")
        assert "myproj" in user

    def test_no_project_context(self):
        _, user = analyze_session_friction("s", "c")
        assert "project" not in user.lower()

    def test_none_instruction(self):
        system, _ = analyze_session_friction("s", "c")
        assert "NONE" in system


class TestIdleFrictionProbePrompt:
    def test_returns_system_and_user(self):
        system, user = idle_friction_probe("conversation text here")
        assert "friction" in system.lower() or "self-assess" in system.lower()
        assert "conversation text here" in user

    def test_includes_categories(self):
        system, _ = idle_friction_probe("c")
        assert "TOOL_ISSUE" in system
        assert "MISSING_FEATURE" in system
        assert "CONTEXT_PROBLEM" in system
        assert "WORKFLOW_ISSUE" in system

    def test_honest_instruction(self):
        system, _ = idle_friction_probe("c")
        assert "brutally honest" in system.lower()

    def test_none_instruction(self):
        system, _ = idle_friction_probe("c")
        assert "NONE" in system


# --- Parser Tests ---


class TestParseFrictionResponse:
    def test_none_response(self):
        items = MemoryProcessor._parse_friction_response("NONE", 1, "nightly")
        assert items == []

    def test_empty_response(self):
        items = MemoryProcessor._parse_friction_response("", 1, "nightly")
        assert items == []

    def test_single_item(self):
        raw = "- TOOL_FAILURE: grep_files returned 0 results on valid pattern"
        items = MemoryProcessor._parse_friction_response(raw, 42, "nightly")
        assert len(items) == 1
        assert items[0]["category"] == "TOOL_FAILURE"
        assert items[0]["description"] == "grep_files returned 0 results on valid pattern"
        assert items[0]["session_id"] == 42
        assert items[0]["source"] == "nightly"

    def test_multiple_items(self):
        raw = (
            "- TOOL_FAILURE: read_file failed on binary file\n"
            "- REPEATED_RETRY: shell command retried 3 times before succeeding\n"
            "- MISSING_CAPABILITY: no way to search memory by date range"
        )
        items = MemoryProcessor._parse_friction_response(raw, 10, "nightly")
        assert len(items) == 3
        assert items[0]["category"] == "TOOL_FAILURE"
        assert items[1]["category"] == "REPEATED_RETRY"
        assert items[2]["category"] == "MISSING_CAPABILITY"

    def test_idle_probe_categories(self):
        raw = (
            "- TOOL_ISSUE: web_fetch keeps timing out\n"
            "- MISSING_FEATURE: can't run Python code without writing a file first"
        )
        items = MemoryProcessor._parse_friction_response(raw, 5, "idle_probe")
        assert len(items) == 2
        assert items[0]["category"] == "TOOL_ISSUE"
        assert items[1]["category"] == "MISSING_FEATURE"
        assert all(i["source"] == "idle_probe" for i in items)

    def test_invalid_category_skipped(self):
        raw = (
            "- TOOL_FAILURE: valid item\n"
            "- RANDOM_THING: should be skipped\n"
            "- CONTEXT_ISSUE: another valid item"
        )
        items = MemoryProcessor._parse_friction_response(raw, 1, "nightly")
        assert len(items) == 2
        assert items[0]["category"] == "TOOL_FAILURE"
        assert items[1]["category"] == "CONTEXT_ISSUE"

    def test_no_description_skipped(self):
        raw = "- TOOL_FAILURE:"
        items = MemoryProcessor._parse_friction_response(raw, 1, "nightly")
        assert items == []

    def test_without_dash_prefix(self):
        raw = "TOOL_FAILURE: grep returned nothing"
        items = MemoryProcessor._parse_friction_response(raw, 1, "nightly")
        assert len(items) == 1

    def test_none_session_id(self):
        raw = "- TOOL_ISSUE: something broke"
        items = MemoryProcessor._parse_friction_response(raw, None, "idle_probe")
        assert len(items) == 1
        assert items[0]["session_id"] is None


# --- SQLite CRUD Tests ---


@pytest.mark.asyncio
async def test_add_friction_entry(sqlite_store):
    fid = await sqlite_store.add_friction_entry(
        session_id=None, source="nightly",
        category="TOOL_FAILURE", description="test failure",
    )
    assert isinstance(fid, int)
    assert fid > 0


@pytest.mark.asyncio
async def test_get_friction_entries(sqlite_store):
    await sqlite_store.add_friction_entry(None, "nightly", "TOOL_FAILURE", "fail 1")
    await sqlite_store.add_friction_entry(None, "idle_probe", "MISSING_FEATURE", "need X")

    items = await sqlite_store.get_friction_entries(limit=10)
    assert len(items) == 2
    categories = {i["category"] for i in items}
    assert categories == {"TOOL_FAILURE", "MISSING_FEATURE"}


@pytest.mark.asyncio
async def test_get_friction_entries_unreviewed_only(sqlite_store):
    fid1 = await sqlite_store.add_friction_entry(None, "nightly", "TOOL_FAILURE", "old")
    await sqlite_store.add_friction_entry(None, "nightly", "CONTEXT_ISSUE", "new")

    # Mark first as reviewed
    await sqlite_store.mark_friction_reviewed([fid1])

    items = await sqlite_store.get_friction_entries(unreviewed_only=True, limit=10)
    assert len(items) == 1
    assert items[0]["category"] == "CONTEXT_ISSUE"


@pytest.mark.asyncio
async def test_mark_friction_reviewed(sqlite_store):
    fid1 = await sqlite_store.add_friction_entry(None, "nightly", "A", "desc1")
    fid2 = await sqlite_store.add_friction_entry(None, "nightly", "B", "desc2")

    count = await sqlite_store.mark_friction_reviewed([fid1, fid2])
    assert count == 2

    items = await sqlite_store.get_friction_entries(unreviewed_only=True, limit=10)
    assert len(items) == 0


@pytest.mark.asyncio
async def test_mark_friction_reviewed_empty(sqlite_store):
    count = await sqlite_store.mark_friction_reviewed([])
    assert count == 0


@pytest.mark.asyncio
async def test_get_sessions_missing_friction_analysis(sqlite_store):
    """Sessions with reflections but no friction analysis show up."""
    # Create a session
    cursor = await sqlite_store._db.execute(
        "INSERT INTO sessions (project, summary, message_count) VALUES (?, ?, ?)",
        ("test", "Did some coding", 10),
    )
    await sqlite_store._db.commit()
    sid = cursor.lastrowid

    # Add a reflection (not SKIP)
    await sqlite_store._db.execute(
        "INSERT INTO session_reflections (session_id, reflection_text) VALUES (?, ?)",
        (sid, "EFFECTIVENESS: effective\nWHAT_WORKED:\n- stuff"),
    )
    await sqlite_store._db.commit()

    # Should appear in missing list
    missing = await sqlite_store.get_sessions_missing_friction_analysis(limit=10)
    assert len(missing) == 1
    assert missing[0]["id"] == sid

    # Add friction entry for this session
    await sqlite_store.add_friction_entry(sid, "nightly", "NONE", "No friction detected")

    # Should no longer appear
    missing = await sqlite_store.get_sessions_missing_friction_analysis(limit=10)
    assert len(missing) == 0


# --- Processor Integration Tests ---


@pytest.mark.asyncio
async def test_analyze_session_friction_none(sqlite_store, mock_chroma, memory_config):
    """When LLM returns NONE, no items are produced."""
    router = MagicMock()
    router.generate = AsyncMock(return_value="NONE")
    processor = MemoryProcessor(sqlite_store, mock_chroma, router, memory_config)

    items = await processor.analyze_session_friction(
        session_id=1, session_summary="Normal session",
        conversation_text="user: hi\nassistant: hello",
    )
    assert items == []


@pytest.mark.asyncio
async def test_analyze_session_friction_with_items(sqlite_store, mock_chroma, memory_config):
    """When LLM returns friction items, they are parsed correctly."""
    router = MagicMock()
    router.generate = AsyncMock(return_value=(
        "- TOOL_FAILURE: grep_files returned empty on valid query\n"
        "- WORKFLOW_FRICTION: had to read 5 files to find the right one"
    ))
    processor = MemoryProcessor(sqlite_store, mock_chroma, router, memory_config)

    items = await processor.analyze_session_friction(
        session_id=1, session_summary="Debugging session",
        conversation_text="user: find the bug\nassistant: let me look...",
    )
    assert len(items) == 2
    assert items[0]["source"] == "nightly"


@pytest.mark.asyncio
async def test_analyze_idle_friction(sqlite_store, mock_chroma, memory_config):
    """Idle friction probe returns parsed items."""
    router = MagicMock()
    router.generate = AsyncMock(return_value=(
        "- TOOL_ISSUE: web_fetch timing out on large pages"
    ))
    processor = MemoryProcessor(sqlite_store, mock_chroma, router, memory_config)

    items = await processor.analyze_idle_friction(
        session_id=5,
        conversation_text="user: fetch that page\nassistant: trying...",
    )
    assert len(items) == 1
    assert items[0]["source"] == "idle_probe"
    assert items[0]["category"] == "TOOL_ISSUE"


@pytest.mark.asyncio
async def test_analyze_session_friction_routes_to_session_review(
    sqlite_store, mock_chroma, memory_config,
):
    """Friction must run on SESSION_REVIEW (fast, large-context), not REASONING
    (local-only qwen3:14b). Routing it to REASONING fed kimi-sized chunks to a
    slow local model and blew past the per-session timeout — heavy sessions were
    skipped forever. Pin the task type so this can't silently regress."""
    from blipshell.llm.router import TaskType

    router = MagicMock()
    router.generate = AsyncMock(return_value="NONE")
    processor = MemoryProcessor(sqlite_store, mock_chroma, router, memory_config)

    await processor.analyze_session_friction(
        session_id=1, session_summary="s", conversation_text="c",
    )

    router.generate.assert_awaited_once()
    assert router.generate.await_args.args[0] == TaskType.SESSION_REVIEW


@pytest.mark.asyncio
async def test_analyze_friction_llm_error(sqlite_store, mock_chroma, memory_config):
    """LLM error returns empty list, doesn't crash."""
    router = MagicMock()
    router.generate = AsyncMock(side_effect=Exception("model timeout"))
    processor = MemoryProcessor(sqlite_store, mock_chroma, router, memory_config)

    items = await processor.analyze_session_friction(
        session_id=1, session_summary="s", conversation_text="c",
    )
    assert items == []
