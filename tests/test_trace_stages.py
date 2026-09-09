"""The retrieval trace distinguishes retrieved / sent / omitted, and /why
reports what was SENT (V3 Stage B3).

The trace used to be written by _search_relevant_memories, BEFORE
gather_memory decided what fit the budget, under the key "injected"; /why
printed it as if it were what the model saw (review F6). Now _build_messages
stamps the same trace with the memory ids that actually reached the request
and the ones that did not, with the pool's reason.

Headless real agent from the continuity harness; no model.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from blipshell.benchmark import continuity
from blipshell.benchmark.harness import _load_dataset
from blipshell.models.memory import Memory, MemoryType


async def _seeded_agent(tmp_path, n_memories: int, long: bool = False):
    agent, client = await continuity.bootstrap_headless_agent(tmp_path / "trace.db")
    sid = await agent.sqlite.create_session(title="seed")
    for i in range(n_memories):
        body = f"Fact number {i}: the robot arm torque limit is {i} newton metres."
        if long:
            # Lexically DISTINCT padding per memory: identical filler would be
            # folded by search's Jaccard dedup before the budget ever applied,
            # and the test would measure dedup instead of packing.
            # ...and the padding rides INSIDE the matching sentence (no period),
            # otherwise the excerpt window keeps only the ~70-char fact and six
            # of them fit any budget - which is the excerpt working, not the
            # packing under test.
            pad = " ".join(f"topic{i}word{k}" for k in range(120))
            body = body.rstrip(".") + f", and for context {pad}"
        mid = await agent.sqlite.create_memory(Memory(session_id=sid, role="user", content=body,
                                                      summary=body[:100], rank=3, importance=0.6,
                                                      memory_type=MemoryType.CONVERSATION))
        agent.vectors.add_memory(mid, body, {"session_id": str(sid), "role": "user"})
    return agent, client


async def _close(agent):
    await agent.session_manager.flush_pending_persists()
    await agent.sqlite.close()
    agent.vectors.close()


async def test_trace_has_retrieved_sent_and_omitted_stages(tmp_path):
    agent, client = await _seeded_agent(tmp_path, 4)
    try:
        await agent.start_session()
        await agent.chat("what is the robot arm torque limit?")
        trace = agent._last_retrieval_trace
        for key in ("retrieved", "sent", "omitted"):
            assert key in trace, key
        retrieved_ids = {i["id"] for i in trace["retrieved"] if i.get("source") == "memory"}
        sent_ids = {i["id"] for i in trace["sent"] if i.get("source") == "memory"}
        omitted_ids = {i["id"] for i in trace["omitted"]}
        assert sent_ids, "nothing recorded as sent although memories were recalled"
        assert sent_ids <= retrieved_ids
        assert not (sent_ids & omitted_ids)
        assert sent_ids | omitted_ids == retrieved_ids
        # every id claimed as SENT is really in the request the client got
        system = "\n".join(m["content"] for m in client.sent[-1] if m["role"] == "system")
        for item in trace["sent"]:
            if item.get("source") == "memory":
                assert item["preview"][:40] in system, item
        # "injected" stays as an alias of retrieved for older readers/events
        assert trace["injected"] == trace["retrieved"]
    finally:
        await _close(agent)


async def test_budget_omissions_carry_a_reason(tmp_path):
    agent, client = await _seeded_agent(tmp_path, 6, long=True)
    try:
        # a Recall budget too small for all six long memories
        from blipshell.memory import query_profiles
        import blipshell.core.agent_chat as ac
        orig = query_profiles.compute_pool_budgets

        def tiny(profile, total, caps):
            b = orig(profile, total, caps)
            b["Recall"] = 400
            # the conversation's unused share rolls into Recall by design;
            # pin it so the cap under test is really 400-ish
            b["ActiveSession"] = 50
            return b

        import pytest
        mp = pytest.MonkeyPatch()
        mp.setattr(ac, "compute_pool_budgets", tiny)
        try:
            await agent.start_session()
            await agent.chat("what is the robot arm torque limit?")
        finally:
            mp.undo()
        trace = agent._last_retrieval_trace
        assert trace["omitted"], "with six long memories and a 400-token Recall budget something must be omitted"
        assert all(o.get("reason") for o in trace["omitted"])
        assert all(o["reason"] in ("over budget", "item cap", "already sent via Recall", "already in the project dossier", "not selected")
                   for o in trace["omitted"])
    finally:
        await _close(agent)


async def test_why_reports_sent_not_retrieved(tmp_path):
    from blipshell.ui.command_handlers import _why

    agent, client = await _seeded_agent(tmp_path, 6, long=True)
    try:
        from blipshell.memory import query_profiles
        import blipshell.core.agent_chat as ac
        import pytest
        orig = query_profiles.compute_pool_budgets
        mp = pytest.MonkeyPatch()
        mp.setattr(ac, "compute_pool_budgets",
                   lambda p, t, c: {**orig(p, t, c), "Recall": 400, "ActiveSession": 50})
        try:
            await agent.start_session()
            await agent.chat("what is the robot arm torque limit?")
        finally:
            mp.undo()
        console = MagicMock()
        await _why(SimpleNamespace(agent=agent, console=console))
        printed = "\n".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "sent" in printed.lower()
        assert "not sent" in printed.lower() or "omitted" in printed.lower()
        n_sent = len([i for i in agent._last_retrieval_trace["sent"] if i.get("source") == "memory"])
        n_omitted = len(agent._last_retrieval_trace["omitted"])
        assert str(n_omitted) in printed
        assert n_sent + n_omitted == len([i for i in agent._last_retrieval_trace["retrieved"] if i.get("source") == "memory"])
        # transmission is not proof of reliance - the caveat is printed
        assert "not proof" in printed.lower() or "does not prove" in printed.lower()
    finally:
        await _close(agent)
