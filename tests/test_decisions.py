"""Decisions with conditions (V3 Stage E1): record / revise / reopen / list,
the tools over them, and how they surface in Recall."""

from __future__ import annotations

import json

from blipshell.core.tools.decision_tools import (
    ListDecisionsTool, RecordDecisionTool, ReopenDecisionTool, ReviseDecisionTool,
)
from blipshell.memory import decisions, supersession as sup
from blipshell.models.memory import MemoryType


class TestModule:
    async def test_record_and_read(self, sqlite_store, mock_chroma):
        d = await decisions.record_decision(
            sqlite_store, mock_chroma, decision="No cloud judge for lessons",
            reason="a model grading a model's summary of a model", revisit_when="a local judge scores >0.9 on the attribution set",
            project="blipshell", session_id=None, decided_by="user",
        )
        assert d.status == "active" and d.project == "blipshell" and d.decided_by == "user"
        mem = await sqlite_store.get_memory(d.id)
        assert mem.memory_type == MemoryType.DECISION and mem.role == "user"
        assert mem.content.startswith("DECISION: No cloud judge for lessons.")
        assert "BECAUSE:" in mem.content and "REVISIT WHEN:" in mem.content
        mock_chroma.add_memory.assert_called_once()
        assert json.loads(mem.metadata_json)["decision"]["revisit_when"].startswith("a local judge")

    async def test_revise_supersedes_and_keeps_the_old(self, sqlite_store, mock_chroma):
        old = await decisions.record_decision(sqlite_store, mock_chroma, decision="Use ChromaDB", reason="only option", project="blipshell")
        new = await decisions.revise_decision(sqlite_store, mock_chroma, old.id, decision="Use sqlite-vec",
                                              reason="single database", revisit_when="never", decided_by="user")
        assert new is not None and new.id != old.id
        old2 = await decisions.get_decision(sqlite_store, old.id)
        assert old2.status == "superseded" and old2.superseded_by == new.id
        assert not (await sqlite_store.get_memory(old.id)).is_archived
        rec = (await sup.superseded(sqlite_store, "memory", [old.id]))[old.id]
        assert rec.new_id == new.id and rec.relation == "revises" and rec.detected_by == "decision_tool"
        assert rec.scope == "blipshell" and rec.source_type == "user_statement"

    async def test_reopen_undoes_the_supersession(self, sqlite_store, mock_chroma):
        old = await decisions.record_decision(sqlite_store, mock_chroma, decision="A", reason="r")
        new = await decisions.revise_decision(sqlite_store, mock_chroma, old.id, decision="B", reason="r2")
        back = await decisions.reopen_decision(sqlite_store, old.id, reason="B did not work")
        assert back.status == "reopened" and back.reopened_reason == "B did not work"
        assert await sup.superseded(sqlite_store, "memory", [old.id]) == {}
        hist = await sup.history_of(sqlite_store, "memory", old.id)
        assert len(hist) == 1 and hist[0].undone_at is not None

    async def test_list_filters(self, sqlite_store, mock_chroma):
        a = await decisions.record_decision(sqlite_store, mock_chroma, decision="A", project="p1")
        b = await decisions.record_decision(sqlite_store, mock_chroma, decision="B", project="p2")
        await decisions.revise_decision(sqlite_store, mock_chroma, a.id, decision="A2")
        assert {d.decision for d in await decisions.list_decisions(sqlite_store, project="p1", status="active")} == {"A2"}
        assert {d.decision for d in await decisions.list_decisions(sqlite_store, project="p1", status="superseded")} == {"A"}
        assert {d.decision for d in await decisions.list_decisions(sqlite_store)} == {"A", "A2", "B"}

    async def test_non_decision_memory_is_not_a_decision(self, sqlite_store, mock_chroma):
        from blipshell.models.memory import Memory
        mid = await sqlite_store.create_memory(Memory(role="user", content="plain", summary="plain"))
        assert await decisions.get_decision(sqlite_store, mid) is None
        assert await decisions.revise_decision(sqlite_store, mock_chroma, mid, decision="x") is None


class TestTools:
    async def test_record_revise_reopen_list_round_trip(self, sqlite_store, mock_chroma):
        rec = RecordDecisionTool(sqlite_store, mock_chroma, session_id=None, project="blipshell")
        out = await rec.execute(decision="Keep minimax-m3 on the free tier", reason="glm-5.2 is paid",
                                revisit_when="glm-5.2 gets a free tier")
        assert out.startswith("Recorded. Decision #")
        did = int(out.split("#")[1].split(" ")[0])

        rev = ReviseDecisionTool(sqlite_store, mock_chroma)
        out2 = await rev.execute(decision_id=did, decision="Move to glm-5.2", reason="free tier appeared")
        assert "superseded" in out2
        new_id = int(out2.split("Decision #")[2].split(" ")[0])

        lst = ListDecisionsTool(sqlite_store, project="blipshell")
        active = await lst.execute()
        assert "Move to glm-5.2" in active and "Keep minimax" not in active
        assert "Keep minimax" in await lst.execute(status="superseded")

        reo = ReopenDecisionTool(sqlite_store)
        assert (await reo.execute(decision_id=did, reason="free tier was rate limited")).startswith("Reopened.")
        assert "Keep minimax" in await lst.execute(status="reopened")

    async def test_failures_are_typed(self, sqlite_store, mock_chroma):
        from blipshell.core.tools.base import ToolFailure
        assert isinstance(await RecordDecisionTool(sqlite_store, mock_chroma).execute(decision="  "), ToolFailure)
        assert isinstance(await ReviseDecisionTool(sqlite_store, mock_chroma).execute(decision_id=99999, decision="x"), ToolFailure)
        assert isinstance(await ReopenDecisionTool(sqlite_store).execute(decision_id=99999), ToolFailure)


class TestRecall:
    async def test_current_question_sees_only_the_current_decision(self, tmp_path):
        from blipshell.benchmark import continuity
        agent, client = await continuity.bootstrap_headless_agent(tmp_path / "dec.db")
        try:
            old = await decisions.record_decision(agent.sqlite, agent.vectors, decision="Use ChromaDB for the vector store",
                                                  reason="it was the only option in March", project="blipshell")
            await decisions.revise_decision(agent.sqlite, agent.vectors, old.id, decision="Use sqlite-vec for the vector store",
                                            reason="one database, no sync drift", revisit_when="sqlite-vec is abandoned")
            agent.active_project = {"name": "blipshell", "root_path": None}
            await agent.start_session()

            await agent.chat("What did we decide about the vector store?")
            sent = "\n".join(m["content"] for m in client.sent[-1] if m["role"] == "system")
            assert "Use sqlite-vec" in sent and "REVISIT WHEN: sqlite-vec is abandoned" in sent
            assert "Use ChromaDB" not in sent

            await agent.chat("How did our vector store decision change over time?")
            sent = "\n".join(m["content"] for m in client.sent[-1] if m["role"] == "system")
            assert "Use sqlite-vec" in sent
            old_lines = [ln for ln in sent.splitlines() if "Use ChromaDB" in ln]
            assert old_lines and all("superseded" in ln for ln in old_lines)
        finally:
            await agent.session_manager.flush_pending_persists()
            await agent.sqlite.close()
            agent.vectors.close()

    async def test_tools_are_registered_on_the_agent(self, tmp_path):
        from blipshell.benchmark import continuity
        agent, client = await continuity.bootstrap_headless_agent(tmp_path / "reg.db")
        try:
            await agent.start_session()
            names = {t.definition().name for t in agent.tool_registry._tools.values()}
            assert {"record_decision", "revise_decision", "reopen_decision", "list_decisions"} <= names
        finally:
            await agent.session_manager.flush_pending_persists()
            await agent.sqlite.close()
            agent.vectors.close()
