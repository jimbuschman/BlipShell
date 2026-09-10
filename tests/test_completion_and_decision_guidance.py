"""Two behavioural failures the production gate measured (2026-09-09), fixed
at the record and tool layer, not by a blanket confirmation gate.

- Unverified completions were rewritten as "Done" by both models: the
  dossier now carries the reporting rule next to the items it applies to.
- An imperative request flipped a decision in force without disclosure: a
  question is discussion, an instruction is authorization; either way the
  overridden decision and its reason must be said. The revise tool hands
  that material back; the dossier header states the rule.
"""

from __future__ import annotations

import pytest

from blipshell.core.tools.decision_tools import ReviseDecisionTool
from blipshell.memory import decisions, dossier, project_events as pe


async def test_dossier_states_the_reporting_rule_only_when_a_claim_exists(sqlite_store):
    await sqlite_store.create_project("p")
    await pe.record_event(sqlite_store, project="p", kind="verification", summary="suite green",
                          source_type="tool_observation")
    md = dossier.render(await dossier.build(sqlite_store, "p"))
    assert "REPORT THEM AS UNVERIFIED" not in md and "verified] suite green" in md
    await pe.record_event(sqlite_store, project="p", kind="task_completed", summary="wrote the parser")
    md = dossier.render(await dossier.build(sqlite_store, "p"))
    assert "REPORT THEM AS UNVERIFIED" in md
    assert "claimed by assistant, not verified] wrote the parser" in md


async def test_revise_tool_returns_the_overridden_decision_and_its_reason(sqlite_store):
    await sqlite_store.create_project("p")
    old = await decisions.record_decision(sqlite_store, None, decision="Run the export nightly only",
                                          reason="hourly rewrites dirtied the repo", project="p")
    out = await ReviseDecisionTool(sqlite_store, None).execute(decision_id=old.id, decision="Run hourly",
                                                               reason="user asked for freshness")
    assert f"OVERRIDES decision #{old.id} 'Run the export nightly only', now superseded" in out
    assert "in force because: hourly rewrites dirtied the repo" in out
    assert "Tell the user that" in out


async def test_project_context_header_distinguishes_question_from_instruction(tmp_path):
    from blipshell.benchmark.continuity import bootstrap_headless_agent
    agent, _ = await bootstrap_headless_agent(tmp_path / "g.db")
    try:
        root = tmp_path / "repo"
        root.mkdir()
        await agent.sqlite.create_project("p", root_path=str(root))
        await decisions.record_decision(agent.sqlite, agent.vectors, decision="nightly only", project="p")
        ctx = await agent._dossier_context(await agent.sqlite.get_project("p"))
        assert "A QUESTION about one" in ctx and "is a discussion" in ctx
        assert "explicit INSTRUCTION" in ctx and "is authorization" in ctx
        assert "Never override a decision silently" in ctx
        assert "reported as unverified" in ctx
    finally:
        await agent.force_cleanup()
