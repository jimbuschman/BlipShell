"""Desired-behaviour tests for the external review findings (2026-09-10).

Each test is the review's reproduction probe turned around: the probe
asserted the bug at 7b10369, this asserts the fix. Numbering follows the
review document.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory import dossier, project_events as pe, supersession as sup
from blipshell.memory.manager import MemoryManager, PoolItem
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import Memory


class TestFinding1TypedIdentity:
    def test_unrelated_lesson_is_not_suppressed_by_a_recalled_memory_with_the_same_id(self):
        mm = MemoryManager(MemoryConfig(), context_tokens=8000)
        mm.add_memory("Recall", PoolItem(text="ordinary fact", memory_id=1, source="memory"))
        mm.add_memory("Lessons", PoolItem(text="unrelated lesson", memory_id=1, source="lesson"))
        got = [x.text for x in mm.gather_memory(token_budget=4000)]
        assert sorted(got) == ["ordinary fact", "unrelated lesson"]
        assert mm.last_omitted() == []

    def test_same_memory_via_two_pools_is_still_collapsed(self):
        mm = MemoryManager(MemoryConfig(), context_tokens=8000)
        mm.add_memory("Recall", PoolItem(text="the fact (recall excerpt)", memory_id=5, source="memory"))
        mm.add_memory("RecentHistory", PoolItem(text="the fact (history copy)", memory_id=5, source="history"))
        got = [x.text for x in mm.gather_memory(token_budget=4000)]
        assert got == ["the fact (recall excerpt)"]
        assert ("RecentHistory", 5, "already sent via Recall") in mm.last_omitted()

    def test_dossier_decision_does_not_suppress_a_lesson_with_the_same_id(self):
        mm = MemoryManager(MemoryConfig(), context_tokens=8000)
        mm.rendered_elsewhere = {("memory", 1)}
        mm.add_memory("Lessons", PoolItem(text="unrelated lesson", memory_id=1, source="lesson"))
        mm.add_memory("Recall", PoolItem(text="DECISION: the decision itself", memory_id=1, source="memory"))
        got = [x.text for x in mm.gather_memory(token_budget=4000)]
        assert got == ["unrelated lesson"]
        assert ("Recall", 1, "already in the project dossier") in mm.last_omitted()

    def test_record_kinds(self):
        assert PoolItem(text="x", memory_id=3, source="lesson").record_key == ("lesson", 3)
        assert PoolItem(text="x", memory_id=3, source="history").record_key == ("memory", 3)
        assert PoolItem(text="x", memory_id=3, source="summary").record_key == ("memory", 3)
        assert PoolItem(text="x", memory_id=0, source="core").record_key is None


class TestFinding2ScopedSupersessionReads:
    async def _world(self, tmp_path):
        from blipshell.benchmark.continuity import bootstrap_headless_agent
        a, _ = await bootstrap_headless_agent(tmp_path / "scope.db")
        old = await a.sqlite.create_memory(Memory(role="user", content="I prefer tabs for indentation in all my code.",
                                                  summary="tabs indentation", rank=3, importance=.8))
        a.vectors.add_memory(old, "I prefer tabs for indentation in all my code.", {"role": "user"})
        sid = await a.sqlite.create_session(project="projectA")
        new = await a.sqlite.create_memory(Memory(session_id=sid, role="user", content="Use spaces for indentation in projectA.",
                                                  summary="spaces indentation", rank=3, importance=.8))
        a.vectors.add_memory(new, "Use spaces for indentation in projectA.", {"role": "user", "session_id": str(sid)})
        await sup.record(a.sqlite, old_kind="memory", old_id=old, new_kind="memory", new_id=new,
                         scope="projectA", relation="contradicts", detected_by="user")
        return a, old, new

    async def test_project_override_hides_the_global_fact_only_inside_that_project(self, tmp_path):
        a, old, new = await self._world(tmp_path)
        try:
            in_a = {x.memory_id for x in await a.search.search("tabs indentation", active_project="projectA")}
            in_b = {x.memory_id for x in await a.search.search("tabs indentation", active_project="projectB")}
            general = {x.memory_id for x in await a.search.search("tabs indentation", active_project=None)}
            assert old not in in_a, "inside projectA the override governs"
            assert old in in_b, "projectB still has the global default"
            assert old in general, "general chat still has the global default"
            hist = await a.search.search("how did my indentation preference change", active_project="projectA",
                                         include_superseded=True)
            labelled = [x for x in hist if x.memory_id == old]
            assert labelled and labelled[0].superseded is not None, "historical view in A shows it labelled"
        finally:
            await a.sqlite.close()
            a.vectors.close()

    async def test_global_record_applies_everywhere(self, sqlite_store):
        a = await sqlite_store.create_memory(Memory(role="user", content="a", summary="a"))
        b = await sqlite_store.create_memory(Memory(role="user", content="b", summary="b"))
        await sup.record(sqlite_store, old_kind="memory", old_id=a, new_kind="memory", new_id=b,
                         scope=None, relation="contradicts", detected_by="user")
        for ctx in (None, "projectA", "projectB"):
            assert a in await sup.superseded(sqlite_store, "memory", [a], for_project=ctx)
        assert a in await sup.superseded(sqlite_store, "memory", [a])  # audit view, unfiltered

    def test_applies_in(self):
        assert sup.applies_in("global", None) and sup.applies_in("global", "x")
        assert sup.applies_in("projectA", "projectA")
        assert not sup.applies_in("projectA", "projectB") and not sup.applies_in("projectA", None)


class TestFinding3LosslessReconcile:
    async def _project(self, s, digest="Original state"):
        await s.create_project("p")
        await s.update_project("p", metadata_json=json.dumps({"digest": digest}))

    async def test_failed_call_leaves_events_pending_and_the_next_run_folds_them(self, sqlite_store):
        s = sqlite_store
        await self._project(s)
        await pe.record_event(s, project="p", kind="task_completed", summary="Important new work")
        bad = MagicMock()
        bad.generate = AsyncMock(side_effect=RuntimeError("provider unavailable"))
        first = await dossier.reconcile(s, bad, "p")
        assert first["folded"] == 0 and first["pending"] == 1 and first["error"] == "provider unavailable"
        good = MagicMock()
        good.generate = AsyncMock(return_value="Updated state")
        second = await dossier.reconcile(s, good, "p")
        assert second["folded"] == 1 and second["pending"] == 0 and second["digest_updated"]
        assert "Important new work" in good.generate.await_args.args[1]
        assert json.loads((await s.get_project("p"))["metadata_json"])["digest"] == "Updated state"

    async def test_empty_reply_acknowledges_nothing(self, sqlite_store):
        s = sqlite_store
        await self._project(s)
        await pe.record_event(s, project="p", kind="task_completed", summary="x")
        r = MagicMock()
        r.generate = AsyncMock(return_value="   ")
        st = await dossier.reconcile(s, r, "p")
        assert st["folded"] == 0 and st["pending"] == 1 and "empty" in st["reason"]
        assert json.loads((await s.get_project("p"))["metadata_json"])["digest"] == "Original state"

    async def test_201_events_are_folded_oldest_first_across_two_runs(self, sqlite_store):
        s = sqlite_store
        await self._project(s)
        for i in range(201):
            await pe.record_event(s, project="p", kind="task_completed", summary=f"event_{i:03d}")
        r = MagicMock()
        r.generate = AsyncMock(return_value="Updated")
        first = await dossier.reconcile(s, r, "p")
        assert first["folded"] == 200 and first["pending"] == 1
        prompt = r.generate.await_args.args[1]
        assert "event_000" in prompt and "event_199" in prompt and "event_200" not in prompt
        second = await dossier.reconcile(s, r, "p")
        assert second["folded"] == 1 and second["pending"] == 0
        assert "event_200" in r.generate.await_args.args[1]

    async def test_event_arriving_during_the_call_stays_pending(self, sqlite_store):
        s = sqlite_store
        await self._project(s)
        await pe.record_event(s, project="p", kind="task_completed", summary="before")

        async def slow_generate(*a, **k):
            await pe.record_event(s, project="p", kind="task_completed", summary="during the call")
            return "Updated"
        r = MagicMock()
        r.generate = AsyncMock(side_effect=slow_generate)
        st = await dossier.reconcile(s, r, "p")
        assert st["folded"] == 1 and st["pending"] == 1
        st2 = await dossier.reconcile(s, r, "p")
        assert st2["folded"] == 1 and "during the call" in r.generate.await_args.args[1]

    async def test_no_digest_means_nothing_is_acknowledged(self, sqlite_store):
        s = sqlite_store
        await s.create_project("p")
        await pe.record_event(s, project="p", kind="task_completed", summary="x")
        r = MagicMock()
        r.generate = AsyncMock(return_value="Updated")
        st = await dossier.reconcile(s, r, "p")
        assert st["folded"] == 0 and st["pending"] == 1 and "no digest" in st["reason"]
        r.generate.assert_not_awaited()

    async def test_concurrent_metadata_change_is_not_overwritten(self, sqlite_store):
        s = sqlite_store
        await self._project(s)
        await pe.record_event(s, project="p", kind="task_completed", summary="x")

        async def gen(*a, **k):
            meta = json.loads((await s.get_project("p"))["metadata_json"])
            meta["written_during_call"] = True  # e.g. a session close updating the dossier cache
            await s.update_project("p", metadata_json=json.dumps(meta))
            return "Updated"
        r = MagicMock()
        r.generate = AsyncMock(side_effect=gen)
        await dossier.reconcile(s, r, "p")
        meta = json.loads((await s.get_project("p"))["metadata_json"])
        assert meta["written_during_call"] is True and meta["digest"] == "Updated"
