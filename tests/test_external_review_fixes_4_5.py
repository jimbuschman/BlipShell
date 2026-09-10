"""Desired-behaviour tests for review findings 4 and 5 (2026-09-10)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory import decisions, dossier, supersession as sup
from blipshell.memory.processor import MemoryProcessor
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import CoreMemory


class TestFinding4ReopenVsRestore:
    async def test_reopen_for_discussion_leaves_the_replacement_governing(self, sqlite_store):
        s = sqlite_store
        await s.create_project("p")
        old = await decisions.record_decision(s, None, decision="Use SQLite", project="p")
        new = await decisions.revise_decision(s, None, old.id, decision="Use Postgres")
        back = await decisions.reopen_decision(s, old.id, reason="replacement questioned")
        assert back.status == "reopened" and back.superseded_by == new.id
        d = await dossier.build(s, "p")
        assert [x.decision for x in d.decisions_active] == ["Use Postgres"]
        assert [x.decision for x in d.decisions_reopened] == ["Use SQLite"]
        md = dossier.render(d)
        assert "## Reopened for discussion (NOT in force)" in md
        assert "- #%d Use SQLite - reopened because replacement questioned - currently replaced by #%d" % (old.id, new.id) in md
        # the supersession that retired it is still active
        assert old.id in await sup.superseded(s, "memory", [old.id])
        # and the dossier carries both ids for de-duplication
        dec_ids, _ = dossier.listed_ids(d)
        assert {old.id, new.id} <= dec_ids

    async def test_restore_retires_the_replacement_and_keeps_history(self, sqlite_store):
        s = sqlite_store
        await s.create_project("p")
        old = await decisions.record_decision(s, None, decision="Use SQLite", project="p")
        new = await decisions.revise_decision(s, None, old.id, decision="Use Postgres")
        back = await decisions.reopen_decision(s, old.id, reason="Postgres rejected", restore=True)
        assert back.status == "active" and back.superseded_by is None and back.reopened_reason == "Postgres rejected"
        repl = await decisions.get_decision(s, new.id)
        assert repl.status == "superseded" and repl.superseded_by == old.id
        d = await dossier.build(s, "p")
        assert [x.decision for x in d.decisions_active] == ["Use SQLite"]
        assert [x.decision for x, _ in d.decisions_superseded] == ["Use Postgres"]
        # history: old->new undone, new->old active
        hist = await sup.history_of(s, "memory", old.id)
        assert any(h.old_id == old.id and h.new_id == new.id and h.undone_at for h in hist)
        assert any(h.old_id == new.id and h.new_id == old.id and h.undone_at is None for h in hist)
        assert await sup.superseded(s, "memory", [old.id]) == {}
        assert new.id in await sup.superseded(s, "memory", [new.id])

    async def test_tool_defaults_to_discussion(self, sqlite_store):
        from blipshell.core.tools.decision_tools import ReopenDecisionTool
        await sqlite_store.create_project("p")
        old = await decisions.record_decision(sqlite_store, None, decision="A", project="p")
        new = await decisions.revise_decision(sqlite_store, None, old.id, decision="B")
        out = await ReopenDecisionTool(sqlite_store).execute(decision_id=old.id, reason="why")
        assert out.startswith("Reopened for discussion")
        assert (await decisions.get_decision(sqlite_store, new.id)).status == "active"
        out = await ReopenDecisionTool(sqlite_store).execute(decision_id=old.id, reason="user said so", restore=True)
        assert out.startswith("Restored as the governing decision")
        assert (await decisions.get_decision(sqlite_store, new.id)).status == "superseded"


class TestFinding5CoreUndoRestoresState:
    async def test_undo_reactivates_and_reembeds(self, sqlite_store, mock_chroma):
        s = sqlite_store
        old = await s.create_core_memory(CoreMemory(content="Old core fact", source_type="user_statement"))
        new = await s.create_core_memory(CoreMemory(content="New core fact"))
        mock_chroma.search_core_memories.return_value = [{"id": old, "document": "Old core fact", "similarity": .99}]
        router = MagicMock()
        router.generate = AsyncMock(return_value="YES")
        proc = MemoryProcessor(sqlite=s, vectors=mock_chroma, router=router, config=MemoryConfig())
        await proc._check_core_memory_contradictions(new, "New core fact")
        assert old not in {x.id for x in await s.get_active_core_memories()}
        rec = (await sup.superseded(s, "core_memory", [old]))[old]

        assert await sup.undo(s, rec.id, vectors=mock_chroma) is True
        active = {x.id: x for x in await s.get_active_core_memories()}
        assert old in active and active[old].verification_state == "stated"
        mock_chroma.add_core_memory.assert_called_with(old, "Old core fact")
        assert await sup.superseded(s, "core_memory", [old]) == {}
        assert await sup.undo(s, rec.id, vectors=mock_chroma) is False, "already undone"

    async def test_undo_keeps_a_core_memory_retired_while_another_record_still_applies(self, sqlite_store, mock_chroma):
        s = sqlite_store
        old = await s.create_core_memory(CoreMemory(content="Old"))
        n1 = await s.create_core_memory(CoreMemory(content="New one"))
        n2 = await s.create_core_memory(CoreMemory(content="New two"))
        await s.deactivate_core_memory(old)
        r1 = await sup.record(s, old_kind="core_memory", old_id=old, new_kind="core_memory", new_id=n1,
                              scope=None, relation="contradicts", detected_by="core_contradiction")
        await sup.record(s, old_kind="core_memory", old_id=old, new_kind="core_memory", new_id=n2,
                         scope=None, relation="contradicts", detected_by="core_contradiction")
        await sup.undo(s, r1, vectors=mock_chroma)
        assert old not in {x.id for x in await s.get_active_core_memories()}
        mock_chroma.add_core_memory.assert_not_called()

    async def test_memory_undo_is_unchanged(self, sqlite_store):
        from blipshell.models.memory import Memory
        a = await sqlite_store.create_memory(Memory(role="user", content="a", summary="a"))
        b = await sqlite_store.create_memory(Memory(role="user", content="b", summary="b"))
        rec_id = await sup.record(sqlite_store, old_kind="memory", old_id=a, new_kind="memory", new_id=b,
                                  scope=None, relation="contradicts", detected_by="user")
        assert await sup.undo(sqlite_store, rec_id) is True
        assert await sup.superseded(sqlite_store, "memory", [a]) == {}
