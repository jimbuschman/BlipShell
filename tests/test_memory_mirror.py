"""Memory mirror export (memory/mirror.py + the nightly export_mirror job).

The mirror is EXPORT-ONLY transparency: what the assistant believes, as
Markdown a person can read and diff. These tests pin that every layer lands,
that the export-only contract is stated in the files themselves, and that the
job is registered and never gated on Ollama (it's read-only against the DB).
"""

from datetime import datetime, timezone

from blipshell.core.nightly import _OLLAMA_JOBS, JOB_ORDER
from blipshell.memory.mirror import export_mirror
from blipshell.models.memory import CoreMemory, Lesson


async def _populate(sqlite_store):
    await sqlite_store.set_metadata("user_model", "- (high) Prefers measured evidence over vibes")
    await sqlite_store.set_metadata("user_model_updated_at", "2026-09-01T00:00:00+00:00")
    await sqlite_store.create_core_memory(CoreMemory(
        content="Works on BlipShell and Wisp", category="fact", importance=0.9,
    ))
    await sqlite_store.create_core_memory(CoreMemory(
        content="Likes herons", category="preference", importance=0.6,
    ))
    await sqlite_store.create_lesson(Lesson(
        content="Trace the full call chain before fixing anything.",
        summary="Trace the full call chain before fixing.",
        rank=4, importance=0.8,
    ))
    await sqlite_store.add_self_thought(
        "I keep returning to how the garden changes when nobody watches.",
        datetime.now(timezone.utc).isoformat(),
    )


async def test_export_writes_all_four_files(sqlite_store, temp_db_path, tmp_path):
    await _populate(sqlite_store)
    stats = await export_mirror(sqlite_store, temp_db_path)

    out = tmp_path / "mirror"
    for name in ("USER_MODEL.md", "CORE_MEMORIES.md", "LESSONS.md",
                 "SELF_THOUGHTS.md"):
        assert (out / name).exists(), name
        # The contract lives in the file, not just the docs.
        assert "EXPORT-ONLY" in (out / name).read_text(encoding="utf-8"), name

    assert stats["core_memories"] == 2
    assert stats["lessons"] == 1
    assert stats["self_thoughts"] == 1
    assert stats["user_model_lines"] == 1

    user_model = (out / "USER_MODEL.md").read_text(encoding="utf-8")
    assert "Prefers measured evidence" in user_model
    assert "Last revised: 2026-09-01" in user_model

    cores = (out / "CORE_MEMORIES.md").read_text(encoding="utf-8")
    assert "## fact" in cores and "## preference" in cores
    assert "(0.90) Works on BlipShell and Wisp" in cores

    lessons = (out / "LESSONS.md").read_text(encoding="utf-8")
    assert "(rank 4, imp 0.80) Trace the full call chain before fixing." in lessons

    thoughts = (out / "SELF_THOUGHTS.md").read_text(encoding="utf-8")
    assert "garden changes" in thoughts
    assert "echoes 0" in thoughts


async def test_export_on_empty_store_writes_placeholders(sqlite_store, temp_db_path, tmp_path):
    stats = await export_mirror(sqlite_store, temp_db_path)
    assert stats["core_memories"] == 0
    content = (tmp_path / "mirror" / "CORE_MEMORIES.md").read_text(encoding="utf-8")
    assert "(none)" in content
    assert "(no user model yet)" in (
        tmp_path / "mirror" / "USER_MODEL.md"
    ).read_text(encoding="utf-8")


async def test_export_regenerates_wholesale(sqlite_store, temp_db_path, tmp_path):
    await _populate(sqlite_store)
    await export_mirror(sqlite_store, temp_db_path)
    # Simulate a hand-edit, then re-export: the edit must be gone — the DB is
    # the source of truth and the mirror never merges.
    target = tmp_path / "mirror" / "USER_MODEL.md"
    target.write_text("HAND EDIT", encoding="utf-8")
    await export_mirror(sqlite_store, temp_db_path)
    text = target.read_text(encoding="utf-8")
    assert "HAND EDIT" not in text
    assert "Prefers measured evidence" in text


def test_job_registered_and_not_ollama_gated():
    assert "export_mirror" in JOB_ORDER
    # Runs after the user-model revision so tonight's revision is mirrored.
    assert JOB_ORDER.index("export_mirror") > JOB_ORDER.index("update_user_model")
    # Read-only against the DB — must not be skipped when Ollama is down.
    assert "export_mirror" not in _OLLAMA_JOBS
