"""Unit tests for memory_fs tools (view, create, str_replace, delete).

Revised design: lessons read-only; core writable (approval-gated, vector-synced);
notes writable (free, backed by session_notes).
"""

from unittest.mock import MagicMock

import pytest

from blipshell.core.tools.memory_fs import (
    MemoryCreateTool,
    MemoryDeleteTool,
    MemoryStrReplaceTool,
    MemoryViewTool,
)
from blipshell.memory.fs_backend import MemoryFSBackend
from blipshell.memory.fs_notes import NotesBackend
from blipshell.models.memory import Lesson


@pytest.fixture
def mock_vectors():
    v = MagicMock()
    v.add_core_memory = MagicMock()
    v.delete_core_memory = MagicMock()
    return v


@pytest.fixture
async def session_id(sqlite_store):
    return await sqlite_store.create_session(title="Test")


@pytest.fixture
async def backend(sqlite_store, mock_vectors):
    return MemoryFSBackend(sqlite_store, mock_vectors)


@pytest.fixture
async def notes(sqlite_store, session_id):
    return NotesBackend(sqlite_store, {}, lambda: session_id, max_notes=10)


@pytest.fixture
def approval_yes():
    async def cb(name, args):
        return True
    return cb


@pytest.fixture
def approval_no():
    async def cb(name, args):
        return False
    return cb


# -------- view --------


class TestView:
    async def test_root(self, backend, notes):
        tool = MemoryViewTool(backend, notes)
        result = await tool.execute(path="/memories")
        assert "/memories/lessons/" in result
        assert "/memories/notes/" in result
        assert "/memories/core/" in result

    async def test_invalid_path(self, backend, notes):
        tool = MemoryViewTool(backend, notes)
        assert "Invalid path" in await tool.execute(path="not a path")

    async def test_read_lesson(self, backend, notes, sqlite_store):
        lid = await sqlite_store.create_lesson(Lesson(content="A lesson body.", project="p"))
        tool = MemoryViewTool(backend, notes)
        body = await tool.execute(path=f"/memories/lessons/p/{lid}-x.md")
        assert body == "A lesson body."

    async def test_note_view(self, backend, notes):
        create = MemoryCreateTool(backend, notes)
        await create.execute(path="/memories/notes/plan.md", content="my plan")
        view = MemoryViewTool(backend, notes)
        assert await view.execute(path="/memories/notes/plan.md") == "my plan"


# -------- create --------


class TestCreate:
    async def test_lesson_create_blocked(self, backend, notes):
        tool = MemoryCreateTool(backend, notes)
        result = await tool.execute(path="/memories/lessons/p/", content="x")
        assert "read-only" in result.lower()

    async def test_core_requires_approval_succeeds(
        self, backend, notes, approval_yes, mock_vectors
    ):
        tool = MemoryCreateTool(backend, notes, approval_yes)
        result = await tool.execute(path="/memories/core/", content="core content")
        assert result.startswith("Created /memories/core/")
        mock_vectors.add_core_memory.assert_called_once()

    async def test_core_denied(self, backend, notes, approval_no, mock_vectors):
        tool = MemoryCreateTool(backend, notes, approval_no)
        result = await tool.execute(path="/memories/core/", content="x")
        assert "denied" in result.lower()
        mock_vectors.add_core_memory.assert_not_called()

    async def test_core_no_callback_denies(self, backend, notes):
        tool = MemoryCreateTool(backend, notes, approval_callback=None)
        result = await tool.execute(path="/memories/core/", content="x")
        assert "no approval callback" in result.lower()

    async def test_note_create_free(self, backend, notes):
        tool = MemoryCreateTool(backend, notes)
        result = await tool.execute(path="/memories/notes/scratchpad.md", content="hi")
        assert "Created /memories/notes/scratchpad.md" in result
        assert notes.read("scratchpad") == "hi"

    async def test_digest_create_blocked(self, backend, notes):
        tool = MemoryCreateTool(backend, notes)
        result = await tool.execute(path="/memories/digests/blipshell.md", content="x")
        assert "read-only" in result.lower()


# -------- str_replace --------


class TestStrReplace:
    async def test_lesson_edit_blocked(self, backend, notes, sqlite_store):
        lid = await sqlite_store.create_lesson(Lesson(content="abc xyz", project="p"))
        tool = MemoryStrReplaceTool(backend, notes)
        result = await tool.execute(
            path=f"/memories/lessons/p/{lid}-x.md", old_text="xyz", new_text="123"
        )
        assert "read-only" in result.lower()

    async def test_core_edit_requires_approval(self, backend, notes, approval_no):
        async def yes(name, args):
            return True
        create = MemoryCreateTool(backend, notes, yes)
        canonical = (await create.execute(
            path="/memories/core/", content="hello world"
        )).removeprefix("Created ").strip()

        edit = MemoryStrReplaceTool(backend, notes, approval_no)
        out = await edit.execute(path=canonical, old_text="world", new_text="there")
        assert "denied" in out.lower()

    async def test_note_edit_free(self, backend, notes):
        create = MemoryCreateTool(backend, notes)
        await create.execute(path="/memories/notes/n.md", content="foo bar")
        edit = MemoryStrReplaceTool(backend, notes)
        out = await edit.execute(path="/memories/notes/n.md", old_text="bar", new_text="baz")
        assert "Replaced" in out
        assert notes.read("n") == "foo baz"

    async def test_digest_edit_blocked(self, backend, notes):
        edit = MemoryStrReplaceTool(backend, notes)
        out = await edit.execute(
            path="/memories/digests/blipshell.md", old_text="x", new_text="y"
        )
        assert "read-only" in out.lower()


# -------- delete --------


class TestDelete:
    async def test_lesson_delete_blocked(self, backend, notes, sqlite_store):
        lid = await sqlite_store.create_lesson(Lesson(content="x", project="p"))
        delete = MemoryDeleteTool(backend, notes)
        out = await delete.execute(path=f"/memories/lessons/p/{lid}-x.md")
        assert "read-only" in out.lower()

    async def test_core_requires_approval(self, backend, notes, approval_no):
        async def yes(name, args):
            return True
        create = MemoryCreateTool(backend, notes, yes)
        canonical = (await create.execute(
            path="/memories/core/", content="x"
        )).removeprefix("Created ").strip()

        delete = MemoryDeleteTool(backend, notes, approval_no)
        out = await delete.execute(path=canonical)
        assert "denied" in out.lower()

    async def test_core_delete_unembeds(self, backend, notes, mock_vectors):
        async def yes(name, args):
            return True
        create = MemoryCreateTool(backend, notes, yes)
        canonical = (await create.execute(
            path="/memories/core/", content="x"
        )).removeprefix("Created ").strip()

        delete = MemoryDeleteTool(backend, notes, yes)
        out = await delete.execute(path=canonical)
        assert out.startswith("Deleted ")
        mock_vectors.delete_core_memory.assert_called_once()

    async def test_note_delete_free(self, backend, notes):
        create = MemoryCreateTool(backend, notes)
        await create.execute(path="/memories/notes/n.md", content="x")
        delete = MemoryDeleteTool(backend, notes)
        out = await delete.execute(path="/memories/notes/n.md")
        assert "Deleted" in out
        assert notes.list() == []

    async def test_session_delete_blocked(self, backend, notes):
        delete = MemoryDeleteTool(backend, notes)
        out = await delete.execute(path="/memories/sessions/1-x.md")
        assert "read-only" in out.lower()
