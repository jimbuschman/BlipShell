"""Project dossier assembled from records, updated by events (V3 Stage E2).

The digest stays prose and is labelled inferred; decisions, follow-ups,
completed work and the last session are RECORDS and render deterministically.
Every writer appends a project event and marks the dossier stale; the next
read re-renders. The nightly reconcile folds events into the prose (one
canned call here) for ACTIVE projects only.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.core.tools.followup_tools import AddFollowUpTool, ResolveFollowUpTool
from blipshell.core.tools.interaction_tools import TaskCompleteTool
from blipshell.memory import decisions, dossier, project_events as pe


async def _project(sqlite_store, name="blip", root=None):
    await sqlite_store.create_project(name, description="the assistant", root_path=str(root) if root else None)
    return name


class TestEvents:
    async def test_record_and_query(self, sqlite_store):
        await _project(sqlite_store)
        eid = await pe.record_event(sqlite_store, project="blip", kind="task_completed", summary="wrote the parser",
                                    session_id=None)
        assert eid
        evs = await pe.events(sqlite_store, "blip")
        assert evs[0]["kind"] == "task_completed" and evs[0]["source_type"] == "assistant_inference"
        meta = json.loads((await sqlite_store.get_project("blip"))["metadata_json"])
        assert meta["dossier_stale"] == 1
        assert await pe.record_event(sqlite_store, project=None, kind="task_completed", summary="x") is None
        with pytest.raises(ValueError):
            await pe.record_event(sqlite_store, project="blip", kind="teleported", summary="x")

    async def test_active_projects_window(self, sqlite_store):
        await _project(sqlite_store, "hot")
        await _project(sqlite_store, "cold")
        await pe.record_event(sqlite_store, project="hot", kind="task_completed", summary="x")
        await sqlite_store._db.execute(
            "INSERT INTO project_events (project, kind, summary, at) VALUES ('cold', 'task_completed', 'old', '2020-01-01T00:00:00+00:00')")
        await sqlite_store._db.commit()
        assert await pe.active_projects(sqlite_store) == ["hot"]


class TestBuildAndRender:
    async def test_sections_and_labels(self, sqlite_store, mock_chroma):
        await _project(sqlite_store)
        await sqlite_store.update_project("blip", metadata_json=json.dumps(
            {"digest": "BlipShell is a memory-first assistant. Currently in Stage E2.",
             "digest_updated_at": "2026-09-08T10:00:00+00:00", "digest_session_ids": [3, 4]}))
        old = await decisions.record_decision(sqlite_store, mock_chroma, decision="Use ChromaDB", reason="only option",
                                              project="blip")
        new = await decisions.revise_decision(sqlite_store, mock_chroma, old.id, decision="Use sqlite-vec",
                                              reason="one database", revisit_when="sqlite-vec is abandoned")
        prop = await decisions.record_decision(sqlite_store, mock_chroma, decision="Try a reranker", reason="might help",
                                               project="blip", decided_by="assistant")
        f1 = await sqlite_store.add_follow_up("check the Pi's IP after the router swap", project="blip", due_hint="next week")
        f2 = await sqlite_store.add_follow_up("older open question", project="blip")
        await sqlite_store._db.execute("UPDATE follow_ups SET created_at = '2026-01-01 00:00:00' WHERE id = ?", (f2,))
        await sqlite_store._db.commit()
        f3 = await sqlite_store.add_follow_up("resolved one", project="blip")
        await sqlite_store.resolve_follow_up(f3)
        await pe.record_event(sqlite_store, project="blip", kind="task_completed", summary="added the excerpt module")
        await pe.record_event(sqlite_store, project="blip", kind="verification", summary="suite green 2268",
                              source_type="tool_observation")
        await pe.record_event(sqlite_store, project="blip", kind="session_closed", summary="worked on E2", source_type="reflection")

        d = await dossier.build(sqlite_store, "blip")
        assert [x.id for x in d.decisions_active] == sorted([new.id, prop.id], reverse=True) or {x.id for x in d.decisions_active} == {new.id, prop.id}
        assert [x.id for x, _ in d.decisions_superseded] == [old.id]
        assert [f["id"] for f in d.open_followups] == [f2, f1], "oldest first; resolved excluded"
        assert d.next_action["id"] == f2
        assert {e["kind"] for e in d.completed} == {"task_completed", "verification"}
        assert d.last_session["summary"] == "worked on E2"

        md = dossier.render(d)
        assert "## Objective and current state" in md and "[inferred by the assistant from session summaries (updated 2026-09-08)]" in md
        assert "Currently in Stage E2" in md
        assert "## Decisions in force" in md and "Use sqlite-vec - because one database - revisit when: sqlite-vec is abandoned" in md
        assert "[proposed by assistant] Try a reranker" in md
        assert "## Recently superseded decisions" in md and "[superseded " in md and "Use ChromaDB - was because only option" in md
        assert f"- #{f2} older open question" in md and f"- #{f1} check the Pi's IP after the router swap (due: next week)" in md
        assert "resolved one" not in md
        assert "claimed by assistant, not verified] added the excerpt module" in md
        assert "verified] suite green 2268" in md
        assert "## Last session" in md and "worked on E2" in md
        assert f"Follow-up #{f2} (the oldest open question above)" in md
        assert "## Sources" in md and "digest sessions [3, 4]" in md

    async def test_empty_project_renders_honest_placeholders(self, sqlite_store):
        await _project(sqlite_store)
        md = dossier.render(await dossier.build(sqlite_store, "blip"))
        assert "_No digest yet._" in md and "_None recorded._" in md and "_None open._" in md
        assert "Ask before assuming one" in md

    async def test_refresh_caches_and_stale_rerenders(self, sqlite_store, mock_chroma):
        await _project(sqlite_store)
        md1 = await dossier.get_dossier_md(sqlite_store, "blip")
        assert "_None recorded._" in md1
        meta = json.loads((await sqlite_store.get_project("blip"))["metadata_json"])
        assert meta["dossier_md"] == md1 and meta["dossier_stale"] == 0
        await decisions.record_decision(sqlite_store, mock_chroma, decision="Ship it", project="blip")  # marks stale
        md2 = await dossier.get_dossier_md(sqlite_store, "blip")
        assert "Ship it" in md2 and md2 != md1
        assert await dossier.get_dossier_md(sqlite_store, "nope") is None


class TestWriters:
    async def test_followup_tools_record_events(self, sqlite_store):
        await _project(sqlite_store)
        out = await AddFollowUpTool(sqlite_store, session_id=None, project="blip").execute(content="ask about the Pi", due_hint="friday")
        fid = int(out.split("#")[1].split(" ")[0])
        await ResolveFollowUpTool(sqlite_store, session_id=None).execute(id=fid)
        kinds = [e["kind"] for e in await pe.events(sqlite_store, "blip")]
        assert kinds == ["followup_resolved", "followup_added"]

    async def test_task_complete_callback(self):
        seen = []

        async def cb(summary, files, decisions_made):
            seen.append((summary, files, decisions_made))
        tool = TaskCompleteTool(on_complete=cb)
        await tool.execute(summary="done the thing", files_modified="a.py", decisions_made="kept X")
        assert seen == [("done the thing", "a.py", "kept X")]
        # no callback, no failure
        await TaskCompleteTool().execute(summary="ok")

    async def test_decisions_record_events(self, sqlite_store, mock_chroma):
        await _project(sqlite_store)
        d = await decisions.record_decision(sqlite_store, mock_chroma, decision="A", project="blip")
        n = await decisions.revise_decision(sqlite_store, mock_chroma, d.id, decision="B")
        await decisions.reopen_decision(sqlite_store, d.id, reason="B failed")
        kinds = [e["kind"] for e in await pe.events(sqlite_store, "blip")]
        # revise = a new decision recorded + the revision itself
        assert kinds == ["decision_reopened", "decision_revised", "decision_recorded", "decision_recorded"]


class TestReconcile:
    async def test_folds_events_into_the_digest_for_active_projects_only(self, sqlite_store, mock_chroma):
        await _project(sqlite_store, "hot")
        await _project(sqlite_store, "cold")
        for name in ("hot", "cold"):
            await sqlite_store.update_project(name, metadata_json=json.dumps({"digest": f"{name} digest v1"}))
        await pe.record_event(sqlite_store, project="hot", kind="task_completed", summary="built the dossier")
        router = MagicMock()
        router.generate = AsyncMock(return_value="hot digest v2 (folded events)")

        stats = await dossier.reconcile(sqlite_store, router, "hot")
        assert stats["folded"] == 1 and stats["digest_updated"] is True
        meta = json.loads((await sqlite_store.get_project("hot"))["metadata_json"])
        assert meta["digest"] == "hot digest v2 (folded events)" and meta["dossier_reconciled_event_id"]
        assert "hot digest v2" in meta["dossier_md"]
        prompt = router.generate.await_args.args[1]
        assert "built the dossier" in prompt and "hot digest v1" in prompt

        # nothing new -> no call
        router.generate.reset_mock()
        stats2 = await dossier.reconcile(sqlite_store, router, "hot")
        assert stats2["folded"] == 0
        router.generate.assert_not_awaited()
        # the cold project is not active and is left alone by the nightly selection
        assert await pe.active_projects(sqlite_store) == ["hot"]


class TestActivationContext:
    async def test_project_context_carries_the_dossier(self, tmp_path):
        from blipshell.benchmark import continuity
        agent, client = await continuity.bootstrap_headless_agent(tmp_path / "act.db")
        try:
            root = tmp_path / "repo"
            root.mkdir()
            await agent.sqlite.create_project("blip", root_path=str(root))
            await decisions.record_decision(agent.sqlite, agent.vectors, decision="Keep minimax on the free tier",
                                            reason="glm-5.2 is paid", revisit_when="a free tier appears", project="blip")
            await agent.sqlite.add_follow_up("re-run the Tailscale benchmark", project="blip")
            row = await agent.sqlite.get_project("blip")
            scan = await agent._scan_project_context(row)
            assert "Project Dossier" not in scan, "the scan is cached for an hour; the dossier must not ride in it"
            ctx = await agent._dossier_context(row)
            assert "=== Project Dossier (auto-maintained) ===" in ctx
            assert "Keep minimax on the free tier - because glm-5.2 is paid - revisit when: a free tier appears" in ctx
            assert "re-run the Tailscale benchmark" in ctx
        finally:
            await agent.sqlite.close()
            agent.vectors.close()

    async def test_activation_dedups_against_the_dossier_and_deactivation_restores(self, tmp_path):
        """What the dossier carries is not rendered twice: the pools skip the
        decision memories, the follow-ups block skips the listed items (a
        project-less follow-up still shows). Deactivation clears both."""
        from blipshell.benchmark import continuity
        from blipshell.memory.manager import PoolItem
        agent, client = await continuity.bootstrap_headless_agent(tmp_path / "dd.db")
        try:
            root = tmp_path / "repo"
            root.mkdir()
            await agent.sqlite.create_project("blip", root_path=str(root))
            d = await decisions.record_decision(agent.sqlite, agent.vectors, decision="Keep minimax on the free tier",
                                                project="blip")
            fid = await agent.sqlite.add_follow_up("re-run the Tailscale benchmark", project="blip")
            gid = await agent.sqlite.add_follow_up("renew the domain", project=None)
            await agent.start_session()
            assert agent.memory_manager.rendered_elsewhere == set()

            await agent.activate_project("blip")
            assert agent.memory_manager.rendered_elsewhere == {("memory", d.id)}
            assert agent._dossier_followup_ids == {fid}
            assert "renew the domain" in agent._pending_follow_ups
            assert "re-run the Tailscale benchmark" not in agent._pending_follow_ups
            assert "=== Project Dossier" in agent._project_context

            # a pool item carrying the decision's memory id is skipped in every pool
            for pool in ("Recall", "RecentHistory"):
                agent.memory_manager.add_memory(pool, PoolItem(
                    text="DECISION: Keep minimax on the free tier", session_role="system",
                    priority_score=9.0, memory_id=d.id, source="test"))
            agent.memory_manager.add_memory("Recall", PoolItem(
                text="an unrelated recalled memory", session_role="system", priority_score=1.0,
                memory_id=d.id + 1000, source="test"))
            texts = [i.text for i in agent.memory_manager.gather_memory()]
            assert "an unrelated recalled memory" in texts
            assert not any("Keep minimax" in t for t in texts)

            await agent.deactivate_project()
            assert agent.memory_manager.rendered_elsewhere == set() and agent._dossier_followup_ids == set()
            assert "re-run the Tailscale benchmark" in agent._pending_follow_ups
            texts = [i.text for i in agent.memory_manager.gather_memory()]
            assert any("Keep minimax" in t for t in texts)
        finally:
            await agent.sqlite.close()
            agent.vectors.close()

    async def test_export_writes_the_dossier(self, sqlite_store, mock_chroma, tmp_path):
        from blipshell.memory.digest_export import export_digest
        root = tmp_path / "repo"
        root.mkdir()
        await sqlite_store.create_project("blip", root_path=str(root))
        await sqlite_store.update_project("blip", metadata_json=json.dumps({"digest": "prose digest"}))
        await decisions.record_decision(sqlite_store, mock_chroma, decision="Use sqlite-vec", project="blip")
        path = await export_digest(sqlite_store, "blip")
        text = Path(path).read_text(encoding="utf-8")
        assert "## Decisions in force" in text and "Use sqlite-vec" in text and "prose digest" in text
