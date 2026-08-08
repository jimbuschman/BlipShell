"""Digest export (V2_PLAN 5.3): the digest + lessons rendered into the
target repo where other tools can read them.

The safety properties matter more than the rendering: never raises into a
session close, never touches a repo it can't find, never rewrites an
identical file (dirtying the repo for a no-op).
"""

import json

import pytest

from blipshell.memory.digest_export import (
    EXPORT_DIRNAME, EXPORT_FILENAME, export_digest, render_markdown,
)
from blipshell.models.memory import Lesson


def _lesson(content, importance=0.5):
    return Lesson(content=content, importance=importance)


class TestRender:
    def test_digest_and_lessons_render(self):
        md = render_markdown("blipshell", "Memory-first assistant.",
                             [_lesson("test before commit", 0.9),
                              _lesson("trace the call chain", 0.7)])
        assert "# blipshell — BlipShell project digest" in md
        assert "Memory-first assistant." in md
        assert md.index("test before commit") < md.index("trace the call chain"), (
            "lessons should render highest-importance first"
        )

    def test_generated_header_warns_against_hand_edits(self):
        md = render_markdown("p", "d", [])
        assert "do not hand-edit" in md

    def test_empty_project_says_so(self):
        md = render_markdown("p", None, [])
        assert "No digest or lessons recorded yet" in md


@pytest.fixture
async def project(sqlite_store, tmp_path):
    """A project whose root_path is a real directory, with digest + lesson."""
    repo = tmp_path / "repo"
    repo.mkdir()
    await sqlite_store._db.execute(
        "INSERT INTO projects (name, root_path, metadata_json) VALUES (?, ?, ?)",
        ("proj", str(repo),
         json.dumps({"digest": "What this project is.",
                     "digest_updated_at": "2026-08-07"})),
    )
    await sqlite_store._db.execute(
        "INSERT INTO lessons (content, importance, project) VALUES (?, ?, ?)",
        ("always run the suite", 0.8, "proj"),
    )
    await sqlite_store._db.commit()
    return sqlite_store, repo


class TestExport:
    async def test_writes_into_the_repo(self, project):
        sqlite, repo = project

        path = await export_digest(sqlite, "proj")

        assert path == repo / EXPORT_DIRNAME / EXPORT_FILENAME
        body = path.read_text(encoding="utf-8")
        assert "What this project is." in body
        assert "always run the suite" in body

    async def test_identical_rewrite_is_skipped(self, project):
        """Two exports back-to-back can land inside one mtime tick on
        Windows, so comparing natural mtimes proved nothing (this test's
        first version survived its own mutant). Plant a sentinel mtime: a
        rewrite destroys it, a skip preserves it."""
        import os

        sqlite, repo = project
        path = await export_digest(sqlite, "proj")
        sentinel = 946684800  # 2000-01-01: unmistakably not "now"
        os.utime(path, (sentinel, sentinel))

        again = await export_digest(sqlite, "proj")

        assert again == path
        assert int(path.stat().st_mtime) == sentinel, (
            "an identical export rewrote the file — dirties the repo for a no-op"
        )

    async def test_missing_root_returns_none(self, sqlite_store):
        await sqlite_store._db.execute(
            "INSERT INTO projects (name, root_path, metadata_json) VALUES (?, ?, ?)",
            ("ghost", r"C:\definitely\not\here",
             json.dumps({"digest": "x"})),
        )
        await sqlite_store._db.commit()

        assert await export_digest(sqlite_store, "ghost") is None

    async def test_no_root_path_returns_none(self, sqlite_store):
        await sqlite_store._db.execute(
            "INSERT INTO projects (name, metadata_json) VALUES (?, ?)",
            ("rootless", json.dumps({"digest": "x"})),
        )
        await sqlite_store._db.commit()

        assert await export_digest(sqlite_store, "rootless") is None

    async def test_unknown_project_returns_none(self, sqlite_store):
        assert await export_digest(sqlite_store, "nope") is None

    async def test_broken_store_never_raises(self):
        """Session close calls this — a failure must degrade, not propagate."""
        class Broken:
            async def get_project(self, name):
                raise RuntimeError("db is toast")

        assert await export_digest(Broken(), "proj") is None
