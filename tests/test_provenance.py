"""Provenance on the derived layers (V3 Stage B4).

Raw memories carry `role`; core memories and lessons - the layers built from
them - carried nothing, so once distilled, "Jim said this" and "the model
concluded this" were the same kind of row and rendered the same way. Now
every creation site stamps source_type, the verification state defaults
from it (stated / inferred / verified), a contradicted core memory says so,
and a model's conclusion renders with an [inferred] tag.
"""

from __future__ import annotations

import sqlite3

import pytest

from blipshell.models.memory import (
    CoreMemory, Lesson, default_verification, provenance_tag,
)


class TestModel:
    def test_defaults_follow_the_source_type(self):
        assert default_verification("user_statement") == "stated"
        assert default_verification("import") == "stated"
        assert default_verification("tool_observation") == "verified"
        assert default_verification("assistant_inference") == "inferred"
        assert default_verification("reflection") == "inferred"
        assert default_verification("unknown") == "unknown"

    def test_tag_only_for_inferred_or_contradicted(self):
        assert provenance_tag("user_statement", "stated") == ""
        assert provenance_tag("unknown", "unknown") == "", "pre-B4 rows get no invented label"
        assert provenance_tag("assistant_inference", "inferred") == "[inferred] "
        assert provenance_tag("reflection", "inferred") == "[inferred] "
        assert provenance_tag("assistant_inference", "contradicted") == "[contradicted] "


class TestSchema:
    async def test_fresh_store_has_the_columns(self, sqlite_store):
        for table in ("core_memories", "lessons"):
            cur = await sqlite_store._db.execute(f"PRAGMA table_info({table})")
            cols = {r[1] for r in await cur.fetchall()}
            assert {"source_type", "verification_state"} <= cols, table

    async def test_existing_store_is_migrated(self, tmp_path):
        """A pre-B4 database (tables without the columns) gains them on open.
        Built by creating a current store and DROPPING the two columns, so
        the rest of the schema is exactly what an old install has."""
        from blipshell.memory.sqlite_store import SQLiteStore
        db = tmp_path / "old.db"
        store = SQLiteStore(str(db))
        await store.initialize()
        await store.create_core_memory(CoreMemory(content="old fact"))
        await store.create_lesson(Lesson(content="old lesson"))
        await store.close()

        con = sqlite3.connect(db)
        for table in ("core_memories", "lessons"):
            con.execute(f"ALTER TABLE {table} DROP COLUMN source_type")
            con.execute(f"ALTER TABLE {table} DROP COLUMN verification_state")
        con.commit()
        cols = {r[1] for r in con.execute("PRAGMA table_info(lessons)")}
        assert "source_type" not in cols
        con.close()

        store = SQLiteStore(str(db))
        await store.initialize()
        try:
            cores = await store.get_active_core_memories()
            assert cores[0].source_type == "unknown" and cores[0].verification_state == "unknown"
            lessons = await store.get_all_lessons()
            assert lessons[0].source_type == "unknown" and lessons[0].verification_state == "unknown"
            assert provenance_tag(cores[0].source_type, cores[0].verification_state) == ""
        finally:
            await store.close()


class TestCreationSites:
    async def test_core_memory_state_derives_from_type(self, sqlite_store):
        a = await sqlite_store.create_core_memory(CoreMemory(content="I use Neovim", source_type="user_statement"))
        b = await sqlite_store.create_core_memory(CoreMemory(content="probably likes Rust", source_type="assistant_inference"))
        prov = await sqlite_store.get_provenance("core_memories", [a, b])
        assert prov[a] == ("user_statement", "stated")
        assert prov[b] == ("assistant_inference", "inferred")

    async def test_explicit_state_wins(self, sqlite_store):
        c = await sqlite_store.create_core_memory(CoreMemory(content="pi at .77", source_type="tool_observation",
                                                             verification_state="verified"))
        assert (await sqlite_store.get_provenance("core_memories", [c]))[c] == ("tool_observation", "verified")

    async def test_contradiction_deactivation_records_the_state(self, sqlite_store):
        cid = await sqlite_store.create_core_memory(CoreMemory(content="lives in Ohio", source_type="user_statement"))
        await sqlite_store.deactivate_core_memory(cid)
        assert (await sqlite_store.get_provenance("core_memories", [cid]))[cid][1] == "contradicted"
        assert all(c.id != cid for c in await sqlite_store.get_active_core_memories())

    async def test_processor_default_is_assistant_inference(self, memory_processor, sqlite_store):
        cid = await memory_processor.process_core_memory("the user seems to like dark mode")
        assert (await sqlite_store.get_provenance("core_memories", [cid]))[cid] == ("assistant_inference", "inferred")

    async def test_promotion_follows_the_source_role(self, memory_processor, sqlite_store):
        from blipshell.core.tools.memory_tools import PromoteToCoreMemoryTool
        from blipshell.models.memory import Memory
        sid = await sqlite_store.create_session(title="s")
        user_mid = await sqlite_store.create_memory(Memory(session_id=sid, role="user", content="I take my coffee black.", summary="coffee black"))
        asst_mid = await sqlite_store.create_memory(Memory(session_id=sid, role="assistant", content="You might prefer espresso.", summary="espresso guess"))
        tool = PromoteToCoreMemoryTool(sqlite_store, memory_processor, session_id=sid)
        out_u = await tool.execute(source_type="memory", source_id=user_mid)
        out_a = await tool.execute(source_type="memory", source_id=asst_mid)
        cores = {c.content: c for c in await sqlite_store.get_active_core_memories()}
        assert cores["coffee black"].source_type == "user_statement"
        assert cores["coffee black"].verification_state == "stated"
        assert cores["espresso guess"].source_type == "assistant_inference"
        assert "Promoted" in out_u and "Promoted" in out_a

    async def test_reflection_lessons_are_inferred(self, memory_processor, sqlite_store):
        lid = await memory_processor.process_lesson("user: profile first\nassistant: agreed, numbers over vibes", session_id=None)
        assert lid is not None
        lesson = await sqlite_store.get_lesson(lid)
        assert lesson.source_type == "reflection" and lesson.verification_state == "inferred"
        assert lesson.added_by == "session_review"

    async def test_user_feedback_lessons_are_stated(self, sqlite_store):
        lid = await sqlite_store.create_lesson(Lesson(content="User feedback: stop apologising", source_type="user_statement", added_by="user"))
        lesson = await sqlite_store.get_lesson(lid)
        assert (lesson.source_type, lesson.verification_state, lesson.added_by) == ("user_statement", "stated", "user")

    async def test_get_provenance_ignores_unknown_tables(self, sqlite_store):
        assert await sqlite_store.get_provenance("memories", [1]) == {}
        assert await sqlite_store.get_provenance("lessons", []) == {}


class TestRendering:
    async def test_core_pool_tags_inferred_but_not_stated(self, tmp_path):
        from blipshell.benchmark import continuity
        agent, client = await continuity.bootstrap_headless_agent(tmp_path / "prov.db")
        try:
            await agent.sqlite.create_core_memory(CoreMemory(content="I take my coffee black", source_type="user_statement"))
            await agent.sqlite.create_core_memory(CoreMemory(content="probably prefers Python", source_type="assistant_inference"))
            await agent.sqlite.create_lesson(Lesson(content="Profile before optimising", source_type="reflection"))
            await agent.sqlite.create_lesson(Lesson(content="User feedback: be terse", source_type="user_statement", added_by="user"))
            from blipshell.memory.user_model import DOC_KEY
            await agent.sqlite.set_metadata(DOC_KEY, "- (medium) prefers short answers")
            await agent.start_session()
            core = [i.text for i in agent.memory_manager.get_pool("Core")._items]
            lessons = [i.text for i in agent.memory_manager.get_pool("Lessons")._items]
            assert "I take my coffee black" in core
            assert "[inferred] probably prefers Python" in core
            assert "[inferred] Profile before optimising" in lessons
            assert "User feedback: be terse" in lessons
            assert any("inferred nightly" in t for t in core), "the user model header says it is inferred"
        finally:
            await agent.session_manager.flush_pending_persists()
            await agent.sqlite.close()
            agent.vectors.close()
