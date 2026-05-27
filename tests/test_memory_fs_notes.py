"""Unit tests for NotesBackend — filesystem view over session_notes.

Verifies the notes tier shares the same dict + persistence as the
save_note/get_notes tools, so the two interfaces never drift apart.
"""

import pytest

from blipshell.memory.fs_backend import FSError
from blipshell.memory.fs_notes import NotesBackend


@pytest.fixture
async def session_id(sqlite_store):
    return await sqlite_store.create_session(title="Test session")


@pytest.fixture
def shared_notes():
    """The dict the agent shares between note tools and the notes backend."""
    return {}


@pytest.fixture
async def notes(sqlite_store, shared_notes, session_id):
    return NotesBackend(sqlite_store, shared_notes, lambda: session_id, max_notes=3)


class TestWriteRead:
    async def test_write_then_read(self, notes):
        await notes.write("plan", "Refactor the auth module.")
        assert notes.read("plan") == "Refactor the auth module."

    async def test_write_updates_shared_dict(self, notes, shared_notes):
        await notes.write("plan", "step 1")
        assert shared_notes["plan"] == "step 1"

    async def test_write_persists_to_sqlite(self, notes, sqlite_store, session_id):
        await notes.write("plan", "persisted content")
        stored = await sqlite_store.get_session_notes(session_id)
        assert stored.get("plan") == "persisted content"

    async def test_read_missing(self, notes):
        with pytest.raises(FSError, match="No note 'ghost'"):
            notes.read("ghost")

    async def test_empty_content_rejected(self, notes):
        with pytest.raises(FSError, match="empty"):
            await notes.write("x", "   ")


class TestSharedWithNoteTools:
    async def test_external_dict_change_visible(self, notes, shared_notes):
        # Simulate save_note writing directly to the shared dict.
        shared_notes["task"] = "set by save_note tool"
        assert notes.read("task") == "set by save_note tool"

    async def test_notes_write_visible_to_dict(self, notes, shared_notes):
        await notes.write("decision", "use Node sidecar")
        # A GetNotesTool reading the same dict would see it.
        assert shared_notes["decision"] == "use Node sidecar"


class TestQuota:
    async def test_max_notes_enforced(self, notes):
        await notes.write("a", "1")
        await notes.write("b", "2")
        await notes.write("c", "3")
        with pytest.raises(FSError, match="Max notes"):
            await notes.write("d", "4")

    async def test_update_existing_at_quota_ok(self, notes):
        await notes.write("a", "1")
        await notes.write("b", "2")
        await notes.write("c", "3")
        # Updating an existing note at quota is allowed.
        await notes.write("a", "updated")
        assert notes.read("a") == "updated"


class TestListing:
    async def test_list(self, notes):
        await notes.write("plan", "the plan")
        await notes.write("task", "the task")
        entries = notes.list()
        names = {e.path for e in entries}
        assert names == {"/memories/notes/plan.md", "/memories/notes/task.md"}


class TestReplaceText:
    async def test_simple(self, notes):
        await notes.write("plan", "use Redis")
        count = await notes.replace_text("plan", "Redis", "Memcached")
        assert count == 1
        assert notes.read("plan") == "use Memcached"

    async def test_not_found(self, notes):
        await notes.write("plan", "abc")
        with pytest.raises(FSError, match="not found"):
            await notes.replace_text("plan", "xyz", "q")

    async def test_ambiguous(self, notes):
        await notes.write("plan", "ab ab")
        with pytest.raises(FSError, match="matches 2 times"):
            await notes.replace_text("plan", "ab", "x")


class TestDelete:
    async def test_delete(self, notes, shared_notes):
        await notes.write("plan", "x")
        await notes.delete("plan")
        assert "plan" not in shared_notes
        with pytest.raises(FSError):
            notes.read("plan")

    async def test_delete_missing(self, notes):
        with pytest.raises(FSError, match="No note 'ghost'"):
            await notes.delete("ghost")

    async def test_delete_persists(self, notes, sqlite_store, session_id):
        await notes.write("plan", "x")
        await notes.delete("plan")
        stored = await sqlite_store.get_session_notes(session_id)
        assert "plan" not in stored


class TestNoSession:
    async def test_persist_without_session_raises(self, sqlite_store, shared_notes):
        nb = NotesBackend(sqlite_store, shared_notes, lambda: None, max_notes=5)
        with pytest.raises(FSError, match="No active session"):
            await nb.write("x", "content")
