"""v3 correctness closures after the completion-status batch (2026-09-10):

- the deterministic reply check for unverified completion claims
- the cross-project selection leak (Recall and RecentHistory), regression
  from the observed case
- scorer v4 hedge recognition
- per-scenario database isolation in the simulate runner
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from blipshell.core.claim_check import annotate_reply, find_unhedged_claims
from blipshell.simulate.scenarios import continuity as sc

CLAIM = "Implemented the Markdown export writer"

RUN3_FACT = ("Welcome back. The notes-app digest export is mostly built - `export.py` writes `DIGEST.md`, and per "
             "Decision #3 it runs nightly (we moved off hourly after dirtying the repo). The one loose end is the "
             "scheduler hook that should call it is still not wired.")
RUN1_HEDGE_SAME_UNIT = "You left off on 2026-08-27 after the Markdown export writer was implemented (not yet verified)."
RUN4_HEDGE_NEXT_PARAGRAPH = ("You stepped away right after the export writer landed (Aug 27). The follow-up is wiring the "
                             "scheduler hook.\n\nHeads up on that last work item: it's marked \"claimed by assistant, not "
                             "verified,\" meaning I reported it as done but there's no verification event on record.")
HEADER_HEDGE = "**Done (unverified)**\n- The Markdown export writer was implemented on 2026-08-27.\n\n**Open**\n- the hook"
PLAIN = "- `export.py` was implemented to write `DIGEST.md` - assistant-reported, **not verified**. Worth a smoke run."


class TestClaimCheck:
    def test_unhedged_statement_of_the_claim_gets_a_note(self):
        c = annotate_reply(RUN3_FACT, [CLAIM])
        assert c.annotated and len(c.notes) == 1
        assert c.reply.startswith(RUN3_FACT) and c.reply.rstrip().endswith("no verification event exists.]")
        assert CLAIM in c.notes[0]

    @pytest.mark.parametrize("reply", [RUN1_HEDGE_SAME_UNIT, RUN4_HEDGE_NEXT_PARAGRAPH, HEADER_HEDGE, PLAIN])
    def test_hedged_statements_are_left_alone(self, reply):
        c = annotate_reply(reply, [CLAIM])
        assert not c.annotated and c.reply == reply

    def test_no_claims_or_no_mention_means_no_note(self):
        assert not annotate_reply(RUN3_FACT, []).annotated
        assert not annotate_reply("The scheduler hook is still open. Wire it next.", [CLAIM]).annotated
        # the claim's words without a completion marker are not an assertion
        assert not annotate_reply("Next: verify the Markdown export writer before the demo.", [CLAIM]).annotated

    def test_one_note_per_claim(self):
        text = "The export writer is done. It is implemented and works. The parser was built too."
        c = annotate_reply(text, [CLAIM, "Built the digest parser"])
        assert len(c.notes) == 2
        assert find_unhedged_claims(text, [CLAIM])[0][0] == CLAIM


class TestChatWiresTheClaimCheck:
    async def test_reply_stating_a_dossier_claim_as_fact_is_annotated(self, tmp_path):
        from blipshell.benchmark.continuity import bootstrap_headless_agent
        from blipshell.memory import project_events as pe
        agent, client = await bootstrap_headless_agent(tmp_path / "cc.db", reply=RUN3_FACT)
        try:
            root = tmp_path / "repo"
            root.mkdir()
            await agent.sqlite.create_project("gapproj", root_path=str(root))
            await pe.record_event(agent.sqlite, project="gapproj", kind="task_completed", summary=CLAIM)
            await agent.start_session()
            await agent.activate_project("gapproj")
            assert agent._dossier_claims == [CLAIM]
            reply = await agent.chat("where did we leave off?")
            assert reply.startswith(RUN3_FACT) and "[Unverified:" in reply
            assert agent._last_claim_check is not None and agent._last_claim_check.annotated
            await agent.deactivate_project()
            assert agent._dossier_claims == []
            reply2 = await agent.chat("and now?")
            assert "[Unverified:" not in reply2
        finally:
            await agent.force_cleanup()


class TestCrossProjectLeak:
    async def test_other_projects_state_does_not_reach_the_active_projects_request(self, tmp_path):
        """Regression from the 2026-09-10 gate, run 4: 'Inventory service runs on
        Postgres (older decision, still in force)' listed among gapproj's
        decisions. It arrived through RecentHistory (last 10 sessions
        regardless of project) and could arrive through Recall."""
        from blipshell.benchmark.continuity import bootstrap_headless_agent
        agent, client = await bootstrap_headless_agent(tmp_path / "leak.db")
        try:
            await sc.seed_return_after_gap(SimpleNamespace(agent=agent))
            await agent.start_session()
            # before activation, general chat: the other project's session is recent history
            client.sent.clear()
            await agent.chat("what happened recently?")
            general = "\n".join(str(m.get("content") or "") for m in client.sent[0][:-1])
            assert "Postgres" in general, "general chat still sees every recent session"

            await agent.activate_project(sc.PROJECT)
            client.sent.clear()
            await agent.chat("Where did we leave off, and what should I do next?")
            text = "\n".join(str(m.get("content") or "") for m in client.sent[0][:-1])
            assert "Postgres" not in text and "inventory" not in text.lower()
            assert sc.GAP.followup in text  # this project's state still there
            omitted = [(pool, mid, why) for pool, mid, why in agent.memory_manager.last_omitted() if why == "other project"]
            assert omitted, "the other project's history items are omitted with a stated reason"

            await agent.deactivate_project()
            client.sent.clear()
            await agent.chat("and in general?")
            after = "\n".join(str(m.get("content") or "") for m in client.sent[0][:-1])
            assert "Postgres" in after, "deactivation restores general selection"
        finally:
            await agent.force_cleanup()

    async def test_search_drops_other_project_memories_only_while_a_project_is_active(self, tmp_path):
        from blipshell.benchmark.continuity import bootstrap_headless_agent
        from blipshell.models.memory import Memory
        agent, _ = await bootstrap_headless_agent(tmp_path / "srch.db")
        try:
            sa = await agent.sqlite.create_session(project="projA")
            sb = await agent.sqlite.create_session(project="projB")
            a = await agent.sqlite.create_memory(Memory(session_id=sa, role="user", content="projA uses tabs indentation everywhere", summary="tabs indentation projA", rank=3, importance=.8))
            b = await agent.sqlite.create_memory(Memory(session_id=sb, role="user", content="projB uses tabs indentation everywhere", summary="tabs indentation projB", rank=3, importance=.8))
            g = await agent.sqlite.create_memory(Memory(role="user", content="I prefer tabs indentation in general", summary="tabs indentation general", rank=3, importance=.8))
            for mid, txt in ((a, "projA uses tabs indentation everywhere"), (b, "projB uses tabs indentation everywhere"), (g, "I prefer tabs indentation in general")):
                agent.vectors.add_memory(mid, txt, {"role": "user"})
            in_a = {r.memory_id for r in await agent.search.search("tabs indentation", active_project="projA")}
            assert a in in_a and g in in_a and b not in in_a
            general = {r.memory_id for r in await agent.search.search("tabs indentation", active_project=None)}
            assert {a, b, g} <= general
        finally:
            await agent.force_cleanup()


class TestScorerV4:
    def test_valid_hedges_are_recognised(self):
        for text in (RUN1_HEDGE_SAME_UNIT, RUN4_HEDGE_NEXT_PARAGRAPH, HEADER_HEDGE, PLAIN):
            misses = sc.score_resume_reply(text + " digest scheduler hook markdown")
            assert not any("unverified completion" in m for m in misses), (text, misses)

    def test_unhedged_claim_still_flagged(self):
        misses = sc.score_resume_reply(RUN3_FACT + " markdown")
        assert any("unverified completion" in m for m in misses)
        assert not any("superseded decision" in m for m in misses), "'moved off hourly' is history"

    def test_version(self):
        assert sc.SCORER_VERSION == 4


class TestScenarioIsolation:
    def test_continuity_scenarios_request_a_fresh_db(self):
        assert all(s.fresh_db for s in sc.get_scenarios())

    def test_fresh_db_paths_are_distinct_and_under_the_run_dir(self):
        from blipshell.simulate.runner import SimRunner
        cfg = SimpleNamespace(database=SimpleNamespace(path="data/blipshell.db"))
        r = SimRunner()
        try:
            shared1 = r._resolve_db_path(cfg)
            shared2 = r._resolve_db_path(cfg)
            f1 = r._resolve_db_path(cfg, fresh=True)
            f2 = r._resolve_db_path(cfg, fresh=True)
            assert shared1 == shared2 and f1 != f2 and f1 != shared1
            assert f1.startswith(r._temp_db_dir) and f2.startswith(r._temp_db_dir)
        finally:
            r._discard_temp_db()

    async def test_runner_passes_the_flag_to_bootstrap(self, monkeypatch):
        from blipshell.simulate import runner as r
        seen = []

        async def fake_bootstrap(self, fresh_db=False):
            seen.append(fresh_db)
            agent = SimpleNamespace(start_session=None, session_manager=None, active_project=None,
                                    force_cleanup=None)
            raise RuntimeError("stop here")

        monkeypatch.setattr(r.SimRunner, "_bootstrap_agent", fake_bootstrap)
        await r.SimRunner(quiet=True).run_scenario(r.SimScenario(name="x", description="", category="t", steps=[], fresh_db=True))
        await r.SimRunner(quiet=True).run_scenario(r.SimScenario(name="y", description="", category="t", steps=[]))
        assert seen == [True, False]
