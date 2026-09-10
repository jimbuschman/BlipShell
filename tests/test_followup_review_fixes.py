"""Follow-up review (BLIPSHELL_FIX_FOLLOWUP_REVIEW.md, 2026-09-10): the four
findings reproduced at bc52456, each probe inverted into the desired behaviour.

F1 tool schemas marked omitted were still transmitted.
F2 trimming cut the dossier first while its records stayed excluded elsewhere.
F3 restoring A in A -> B -> C retired B, leaving A and C both in force.
F4 the reconcile's fresh read still overwrote a digest changed during the call.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory import decisions, dossier, project_events, supersession as sup
from blipshell.memory.manager import PoolItem
from blipshell.memory.project_digest import ProjectDigestManager as ProjectDigest
from blipshell.models.session import MessageRole


async def _agent(tmp_path, name):
    from blipshell.benchmark.continuity import bootstrap_headless_agent
    return await bootstrap_headless_agent(tmp_path / name)


def _capture_tools(client):
    captured = []
    original = client.chat_stream

    async def capture(messages, model, tools=None, **kwargs):
        captured.append(tools)
        async for chunk in original(messages, model, tools=tools, **kwargs):
            yield chunk
    client.chat_stream = capture
    return captured


class TestF1OmittedToolsNotTransmitted:
    async def test_tiny_window_sends_no_tools_to_the_provider(self, tmp_path):
        a, c = await _agent(tmp_path, "tools.db")
        captured = _capture_tools(c)
        try:
            await a.start_session()
            a.endpoint_manager._endpoints[0].context_tokens = 2000
            await a.chat("hello")
            assert a._last_context_stats["tools_omitted"] is True
            assert captured and not captured[0], f"provider received tools despite the omission: {captured[0]!r}"
            assert a._last_tools_sent is None
        finally:
            await a.force_cleanup()

    async def test_normal_window_still_sends_tools(self, tmp_path):
        a, c = await _agent(tmp_path, "tools2.db")
        captured = _capture_tools(c)
        try:
            await a.start_session()
            await a.chat("hello")
            assert a._last_context_stats["tools_omitted"] is False
            assert captured and captured[0], "tools are sent when they fit"
            assert a._last_tools_sent == captured[0]
        finally:
            await a.force_cleanup()


class TestF2DossierAndTrimming:
    async def _world(self, tmp_path, name, scan_lines):
        a, c = await _agent(tmp_path, name)
        await a.sqlite.create_project("p", root_path=str(tmp_path))
        d = await decisions.record_decision(a.sqlite, a.vectors, decision="UNIQUE_DECISION_739 use SQLite",
                                            reason="one database", project="p")
        fid = await a.sqlite.add_follow_up("UNIQUE_FOLLOWUP_739 wire the scheduler hook", project="p")
        await a.start_session()
        await a.activate_project("p")
        dossier_block = a._project_context[a._project_context.index("\n=== Project Dossier"):]
        a._project_context = ("repo line of text\n" * scan_lines) + dossier_block
        a.memory_manager.add_memory("Recall", PoolItem(text="UNIQUE_DECISION_739 use SQLite", source="memory",
                                                       memory_id=d.id, priority_score=10))
        return a, c, d, fid

    async def test_the_scan_is_cut_before_the_dossier_and_exclusions_stand(self, tmp_path):
        a, c, d, fid = await self._world(tmp_path, "trim.db", 3000)
        try:
            # room for the tool schemas (~7k tokens) and the dossier, not for a 3000-line scan
            a.endpoint_manager._endpoints[0].context_tokens = 12000
            a.session_manager.add_message(MessageRole.USER, "What database do we use?")
            msgs = a._build_messages("What database do we use?")
            text = "\n".join(x.get("content", "") for x in msgs)
            s = a._last_context_stats
            assert any(o.startswith("project context truncated") for o in s["omitted_fixed"]), s["omitted_fixed"]
            assert s["dossier_trimmed"] is False
            assert "UNIQUE_DECISION_739 use SQLite - because one database" in text, "the dossier survived the cut"
            assert "UNIQUE_FOLLOWUP_739" in text
            # the dossier still carries it, so Recall's copy is still the duplicate
            assert ("Recall", d.id, "already in the project dossier") in a.memory_manager.last_omitted()
            assert text.count("UNIQUE_DECISION_739") == 1
        finally:
            await a.session_manager.flush_pending_persists()
            await a.force_cleanup()

    async def test_when_the_dossier_is_cut_its_records_reach_the_request_another_way(self, tmp_path):
        a, c, d, fid = await self._world(tmp_path, "trim2.db", 0)
        try:
            # make the dossier itself too big for the room by padding the digest section
            a._project_context = a._project_context.replace(
                "## Objective and current state", "## Objective and current state\n" + ("digest prose line\n" * 800))
            a.config.agent.system_prompt = "word " * 200
            a.endpoint_manager._endpoints[0].context_tokens = 2500
            a.session_manager.add_message(MessageRole.USER, "What database do we use?")
            msgs = a._build_messages("What database do we use?")
            text = "\n".join(x.get("content", "") for x in msgs)
            s = a._last_context_stats
            assert s["dossier_trimmed"] is True, s
            assert any(o.startswith("dossier truncated") for o in s["omitted_fixed"]), s["omitted_fixed"]
            # exclusions lifted THIS turn: Recall carries the decision, the block carries the follow-up
            assert ("Recall", d.id, "already in the project dossier") not in a.memory_manager.last_omitted()
            assert "UNIQUE_DECISION_739" in text
            assert "UNIQUE_FOLLOWUP_739" in text
            assert s["request_tokens_estimate"] + s["response_reserve"] <= s["context_limit"]
            # and it does not leak: a later roomy turn is back to normal
            a.endpoint_manager._endpoints[0].context_tokens = 65536
            a._build_messages("and now?")
            assert a._last_context_stats["dossier_trimmed"] is False
        finally:
            await a.session_manager.flush_pending_persists()
            await a.force_cleanup()


class TestF3RestoreAcrossAChain:
    async def test_restore_a_retires_the_governing_c(self, sqlite_store):
        s = sqlite_store
        await s.create_project("p")
        a = await decisions.record_decision(s, None, decision="Use A", project="p")
        b = await decisions.revise_decision(s, None, a.id, decision="Use B")
        c = await decisions.revise_decision(s, None, b.id, decision="Use C")
        back = await decisions.reopen_decision(s, a.id, restore=True, reason="return to A")
        assert back.status == "active" and back.superseded_by is None
        current = await dossier.build(s, "p")
        assert {x.id for x in current.decisions_active} == {a.id}
        cc = await decisions.get_decision(s, c.id)
        assert cc.status == "superseded" and cc.superseded_by == a.id
        bb = await decisions.get_decision(s, b.id)
        assert bb.status == "superseded" and bb.superseded_by == c.id, "the intermediate keeps its history"
        assert await sup.superseded(s, "memory", [a.id]) == {}
        assert c.id in await sup.superseded(s, "memory", [c.id]) and b.id in await sup.superseded(s, "memory", [b.id])

    async def test_two_node_restore_still_works(self, sqlite_store):
        s = sqlite_store
        await s.create_project("p")
        a = await decisions.record_decision(s, None, decision="A", project="p")
        b = await decisions.revise_decision(s, None, a.id, decision="B")
        await decisions.reopen_decision(s, a.id, restore=True)
        assert {x.id for x in (await dossier.build(s, "p")).decisions_active} == {a.id}
        assert (await decisions.get_decision(s, b.id)).superseded_by == a.id

    async def test_looping_history_is_refused_not_guessed(self, sqlite_store):
        s = sqlite_store
        await s.create_project("p")
        a = await decisions.record_decision(s, None, decision="A", project="p")
        b = await decisions.revise_decision(s, None, a.id, decision="B")
        await decisions._update_meta(s, b.id, superseded_by=a.id)  # corrupt: B points back to A
        with pytest.raises(ValueError):
            await decisions.reopen_decision(s, a.id, restore=True)


class TestF4ConcurrentDigest:
    async def test_digest_changed_during_the_call_is_not_overwritten_and_events_stay_pending(self, sqlite_store):
        s = sqlite_store
        await s.create_project("p")
        await s.update_project("p", metadata_json=json.dumps({"digest": "Original digest"}))
        await project_events.record_event(s, project="p", kind="task_completed", summary="event")

        async def generate(*args, **kwargs):
            meta = json.loads((await s.get_project("p"))["metadata_json"])
            meta["digest"] = "New session-close information"
            await s.update_project("p", metadata_json=json.dumps(meta))
            return "Original digest plus old event"
        r = MagicMock()
        r.generate = AsyncMock(side_effect=generate)
        st = await dossier.reconcile(s, r, "p")
        after = json.loads((await s.get_project("p"))["metadata_json"])
        assert after["digest"] == "New session-close information"
        assert st["folded"] == 0 and st["pending"] == 1 and "changed during the call" in st["reason"]
        assert "dossier_reconciled_event_id" not in after
        # the next run folds the event into the NEW digest
        r2 = MagicMock()
        r2.generate = AsyncMock(return_value="New session-close information plus event")
        st2 = await dossier.reconcile(s, r2, "p")
        assert st2["folded"] == 1
        assert "New session-close information" in r2.generate.await_args.args[1]

    async def test_field_level_metadata_writes_do_not_clobber_each_other(self, sqlite_store):
        s = sqlite_store
        await s.create_project("p")
        await dossier.refresh(s, "p")
        pd = ProjectDigest(sqlite=s, router=MagicMock())
        await pd._save_digest("p", "new digest", [1, 2])
        await s.set_project_metadata("p", side_key={"nested": [1]})
        meta = json.loads((await s.get_project("p"))["metadata_json"])
        assert meta["digest"] == "new digest" and meta["digest_session_ids"] == [1, 2]
        assert meta.get("dossier_md") and meta["side_key"] == {"nested": [1]}

    async def test_refresh_does_not_clobber_a_key_written_between_its_read_and_write(self, sqlite_store, monkeypatch):
        s = sqlite_store
        await s.create_project("p")
        real_build = dossier.build

        async def build_then_write(sqlite, project):
            d = await real_build(sqlite, project)
            await s.set_project_metadata("p", written_during=True)
            return d
        monkeypatch.setattr(dossier, "build", build_then_write)
        await dossier.refresh(s, "p")
        meta = json.loads((await s.get_project("p"))["metadata_json"])
        assert meta["written_during"] is True and meta["dossier_md"]

    async def test_key_validation(self, sqlite_store):
        await sqlite_store.create_project("p")
        with pytest.raises(ValueError):
            await sqlite_store.set_project_metadata("p", **{"bad key": 1})
