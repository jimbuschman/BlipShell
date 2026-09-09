"""Explicit, scoped supersession with provenance (V3 Stage E1).

Old records are preserved. A newer fact replacing an older one is a ROW in
`supersessions` (old -> new, scope, relation, detector, evidence,
provenance), written by the production paths that detect it - the dedup
verdict and the core-memory contradiction check - and read by search, which
hides superseded records for current-state questions and labels them for
historical ones. Scope keeps unrelated projects apart.

Real SQLite; the vector store is the shared mock where candidates must be
controlled, the real store with the deterministic embedder where search is
under test (headless agent from the continuity harness).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory import supersession as sup
from blipshell.memory.processor import MemoryProcessor
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import CoreMemory, Memory


# ---------------------------------------------------------------- module

class TestRecords:
    async def test_record_is_idempotent_and_readable(self, sqlite_store):
        a = await sqlite_store.create_memory(Memory(role="user", content="old", summary="old"))
        b = await sqlite_store.create_memory(Memory(role="user", content="new", summary="new"))
        sid = await sup.record(sqlite_store, old_kind="memory", old_id=a, new_kind="memory", new_id=b,
                               scope="blipshell", relation="contradicts", detected_by="dedup_verdict",
                               evidence="DELETE 1", source_type="user_statement")
        again = await sup.record(sqlite_store, old_kind="memory", old_id=a, new_kind="memory", new_id=b,
                                 scope="blipshell", relation="contradicts", detected_by="dedup_verdict")
        assert again == sid, "a repeated detection must not duplicate the row"
        got = await sup.superseded(sqlite_store, "memory", [a, b])
        assert set(got) == {a}
        rec = got[a]
        assert (rec.new_id, rec.scope, rec.relation, rec.detected_by, rec.evidence, rec.source_type) == \
            (b, "blipshell", "contradicts", "dedup_verdict", "DELETE 1", "user_statement")
        assert rec.label().startswith("[superseded ") and f"by memory {b}]" in rec.label()

    async def test_undo_preserves_the_row_and_restores_currency(self, sqlite_store):
        a = await sqlite_store.create_memory(Memory(role="user", content="old", summary="old"))
        b = await sqlite_store.create_memory(Memory(role="user", content="new", summary="new"))
        sid = await sup.record(sqlite_store, old_kind="memory", old_id=a, new_kind="memory", new_id=b,
                               scope=None, relation="refines", detected_by="user")
        assert await sup.undo(sqlite_store, sid) is True
        assert await sup.superseded(sqlite_store, "memory", [a]) == {}
        hist = await sup.history_of(sqlite_store, "memory", a)
        assert len(hist) == 1 and hist[0].undone_at is not None, "undone, not deleted"
        assert await sup.undo(sqlite_store, sid) is False

    async def test_newest_supersession_wins(self, sqlite_store):
        a, b, c = [await sqlite_store.create_memory(Memory(role="user", content=t, summary=t)) for t in "abc"]
        await sup.record(sqlite_store, old_kind="memory", old_id=a, new_kind="memory", new_id=b,
                         scope=None, relation="refines", detected_by="user")
        await sup.record(sqlite_store, old_kind="memory", old_id=a, new_kind="memory", new_id=c,
                         scope=None, relation="contradicts", detected_by="user")
        assert (await sup.superseded(sqlite_store, "memory", [a]))[a].new_id == c

    async def test_bad_vocabulary_is_rejected(self, sqlite_store):
        with pytest.raises(ValueError):
            await sup.record(sqlite_store, old_kind="memory", old_id=1, new_kind="memory", new_id=2,
                             scope=None, relation="replaces", detected_by="user")
        with pytest.raises(ValueError):
            await sup.record(sqlite_store, old_kind="memory", old_id=1, new_kind="memory", new_id=2,
                             scope=None, relation="refines", detected_by="magic")

    def test_scope_rules(self):
        assert sup.same_scope(None, None) and sup.same_scope("a", None) and sup.same_scope(None, "a")
        assert sup.same_scope("a", "a")
        assert not sup.same_scope("blipshell", "wisp")
        assert sup.scope_of(None) == "global" and sup.scope_of("wisp") == "wisp"

    @pytest.mark.parametrize("q,expected", [
        ("How did my indentation preference change over time?", True),
        ("What did I use before I switched to Neovim?", True),
        ("what was my editor originally?", True),
        ("history of the vector store decision", True),
        ("I used to prefer tabs, right?", True),
        ("Do I prefer tabs or spaces?", False),
        ("What vector store does BlipShell use?", False),
        ("What is my cat's name?", False),
        ("", False),
    ])
    def test_historical_question_detection(self, q, expected):
        assert sup.is_historical_question(q) is expected


# ---------------------------------------------------------------- write path: dedup verdict

def _scripted(*replies):
    r = MagicMock()
    r.generate = AsyncMock(side_effect=list(replies))
    return r


async def _mem(sqlite_store, text, project=None, role="user"):
    sid = await sqlite_store.create_session(title="s", project=project)
    mid = await sqlite_store.create_memory(Memory(session_id=sid, role=role, content=text, summary=text))
    return mid


def _candidates(vectors, pairs):
    vectors.search_memories.return_value = [{"id": i, "document": t, "similarity": 0.9} for i, t in pairs]


class TestDedupWritesSupersession:

    async def test_delete_verdict_supersedes_instead_of_archiving(self, sqlite_store, mock_chroma):
        old = await _mem(sqlite_store, "I prefer tabs for indentation.", project="blipshell")
        new = await _mem(sqlite_store, "Correction: I switched to spaces for indentation.", project="blipshell")
        _candidates(mock_chroma, [(old, "I prefer tabs for indentation.")])
        proc = MemoryProcessor(sqlite=sqlite_store, vectors=mock_chroma, router=_scripted("DELETE 1"), config=MemoryConfig())

        assert await proc._decide_and_apply_action(new, "switched to spaces") == "DELETE"

        m = await sqlite_store.get_memory(old)
        assert not m.is_archived, "the old memory must be preserved"
        mock_chroma.delete_memory.assert_not_called()
        rec = (await sup.superseded(sqlite_store, "memory", [old]))[old]
        assert rec.new_id == new and rec.relation == "contradicts" and rec.detected_by == "dedup_verdict"
        assert rec.scope == "blipshell" and rec.evidence == "DELETE 1" and rec.source_type == "user_statement"
        meta = json.loads(m.metadata_json)
        assert meta["superseded_by"] == new and meta["dedup"]["action"] == "DELETE"

    async def test_update_verdict_is_a_refinement(self, sqlite_store, mock_chroma):
        old = await _mem(sqlite_store, "User has a dog named Max.")
        new = await _mem(sqlite_store, "User's dog Max is a 5-year-old golden retriever.", role="assistant")
        _candidates(mock_chroma, [(old, "User has a dog named Max.")])
        proc = MemoryProcessor(sqlite=sqlite_store, vectors=mock_chroma, router=_scripted("UPDATE 1"), config=MemoryConfig())
        assert await proc._decide_and_apply_action(new, "dog Max golden") == "UPDATE"
        rec = (await sup.superseded(sqlite_store, "memory", [old]))[old]
        assert rec.relation == "refines" and rec.scope == "global"
        assert rec.source_type == "assistant_inference", "provenance follows the NEW memory's speaker"

    async def test_other_project_candidates_are_out_of_scope(self, sqlite_store, mock_chroma):
        """A correction in blipshell must not supersede Wisp's similar fact."""
        wisp = await _mem(sqlite_store, "The vector store is ChromaDB.", project="wisp")
        new = await _mem(sqlite_store, "We replaced ChromaDB: the vector store is now sqlite-vec.", project="blipshell")
        _candidates(mock_chroma, [(wisp, "The vector store is ChromaDB.")])
        router = _scripted("DELETE 1")
        proc = MemoryProcessor(sqlite=sqlite_store, vectors=mock_chroma, router=router, config=MemoryConfig())

        assert await proc._decide_and_apply_action(new, "vector store sqlite-vec") == "ADD"

        router.generate.assert_not_awaited()  # nothing left to judge
        assert await sup.superseded(sqlite_store, "memory", [wisp]) == {}
        assert not (await sqlite_store.get_memory(wisp)).is_archived

    async def test_global_candidate_is_in_scope_for_a_project_memory(self, sqlite_store, mock_chroma):
        glob = await _mem(sqlite_store, "User lives in Ohio.")
        new = await _mem(sqlite_store, "User moved to Texas.", project="blipshell")
        _candidates(mock_chroma, [(glob, "User lives in Ohio.")])
        proc = MemoryProcessor(sqlite=sqlite_store, vectors=mock_chroma, router=_scripted("DELETE 1"), config=MemoryConfig())
        assert await proc._decide_and_apply_action(new, "moved to Texas") == "DELETE"
        assert (await sup.superseded(sqlite_store, "memory", [glob]))[glob].scope == "blipshell"


# ---------------------------------------------------------------- write path: core contradiction

class TestCoreContradictionWritesSupersession:
    async def test_yes_verdict_records_core_memory_supersession(self, sqlite_store, mock_chroma):
        old = await sqlite_store.create_core_memory(CoreMemory(content="User lives in Ohio", source_type="user_statement"))
        mock_chroma.search_core_memories.return_value = [{"id": old, "document": "User lives in Ohio", "similarity": 0.9}]
        router = MagicMock()

        async def gen(task_type, prompt="", system=None, **kw):
            return "YES" if "contradict" in (system or "").lower() else "3 0.5 fact"
        router.generate = AsyncMock(side_effect=gen)
        proc = MemoryProcessor(sqlite=sqlite_store, vectors=mock_chroma, router=router, config=MemoryConfig())

        new = await proc.process_core_memory("User moved to Texas", source_type="user_statement")

        rec = (await sup.superseded(sqlite_store, "core_memory", [old]))[old]
        assert rec.new_id == new and rec.detected_by == "core_contradiction" and rec.relation == "contradicts"
        assert rec.evidence.startswith("YES") and rec.source_type == "user_statement"
        prov = await sqlite_store.get_provenance("core_memories", [old])
        assert prov[old][1] == "contradicted"


# ---------------------------------------------------------------- read path: search

async def _agent(tmp_path):
    from blipshell.benchmark import continuity
    return await continuity.bootstrap_headless_agent(tmp_path / "sup.db")


async def _close(agent):
    await agent.sqlite.close()
    agent.vectors.close()


async def _plant(agent, text, project=None, role="user"):
    sid = await agent.sqlite.create_session(title="s", project=project)
    mid = await agent.sqlite.create_memory(Memory(session_id=sid, role=role, content=text, summary=text,
                                                  rank=3, importance=0.6))
    agent.vectors.add_memory(mid, text, {"session_id": str(sid), "role": role})
    return mid


class TestSearchReadsSupersession:
    async def test_current_question_hides_superseded_history_question_labels_it(self, tmp_path):
        agent, _ = await _agent(tmp_path)
        try:
            old = await _plant(agent, "I prefer tabs for indentation in all my code.")
            new = await _plant(agent, "Correction: I switched to spaces, four wide, for indentation.")
            await sup.record(agent.sqlite, old_kind="memory", old_id=old, new_kind="memory", new_id=new,
                             scope=None, relation="contradicts", detected_by="dedup_verdict", evidence="DELETE 1")

            current = await agent.search.search("tabs or spaces indentation", include_superseded=False)
            assert new in {r.memory_id for r in current}
            assert old not in {r.memory_id for r in current}

            history = await agent.search.search("tabs or spaces indentation", include_superseded=True)
            by_id = {r.memory_id: r for r in history}
            assert old in by_id and new in by_id
            assert by_id[old].superseded is not None and by_id[old].superseded.new_id == new
            assert by_id[new].superseded is None
        finally:
            await _close(agent)

    async def test_recall_rendering_labels_superseded_for_historical_questions(self, tmp_path):
        agent, client = await _agent(tmp_path)
        try:
            old = await _plant(agent, "I prefer tabs for indentation in all my code.")
            new = await _plant(agent, "Correction: I switched to spaces, four wide, for indentation.")
            await sup.record(agent.sqlite, old_kind="memory", old_id=old, new_kind="memory", new_id=new,
                             scope=None, relation="contradicts", detected_by="dedup_verdict")
            await agent.start_session()

            await agent.chat("Do I prefer tabs or spaces for indentation?")
            sent = "\n".join(m["content"] for m in client.sent[-1] if m["role"] == "system")
            assert "spaces, four wide" in sent and "I prefer tabs" not in sent

            await agent.chat("How did my indentation preference change over time?")
            sent = "\n".join(m["content"] for m in client.sent[-1] if m["role"] == "system")
            assert "spaces, four wide" in sent
            lines = [ln for ln in sent.splitlines() if "I prefer tabs" in ln]
            assert lines and all("superseded" in ln for ln in lines), lines
        finally:
            await agent.session_manager.flush_pending_persists()
            await _close(agent)
