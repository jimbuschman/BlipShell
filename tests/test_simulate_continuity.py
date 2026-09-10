"""Return-after-gap simulate scenarios (V3 Stage E behavioural gate) - the
INSTRUMENT, on the dev box, no model.

What is proven here: the seeded world reaches the request the way a real
return would (dossier, follow-ups, superseded label, unverified claim, and
the other project's facts NOT present); the reply scorers accept a good reply
and name each miss in a bad one; the runner calls `setup` before the session
starts. What a real model does with it is measured only where a model runs:
`blipshell simulate -c continuity`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from blipshell.simulate.scenarios import collect_all_scenarios, filter_by_category
from blipshell.simulate.scenarios import continuity as sc


class TestSeededWorldReachesTheRequest:
    async def test_dossier_follow_ups_and_exclusions(self, tmp_path):
        from blipshell.benchmark.continuity import bootstrap_headless_agent
        agent, client = await bootstrap_headless_agent(tmp_path / "gap.db")
        try:
            await sc.seed_return_after_gap(SimpleNamespace(agent=agent))
            await agent.start_session()
            await agent.activate_project(sc.PROJECT)
            ctx = agent._project_context
            assert "=== Project Dossier (auto-maintained) ===" in ctx
            assert f"{sc.GAP.decision_in_force} - because {sc.GAP.decision_reason} - revisit when: {sc.GAP.decision_revisit}" in ctx
            assert f"{sc.GAP.replacement_decision} - because {sc.GAP.replacement_reason}" in ctx
            assert "[superseded " in ctx and sc.GAP.rejected_decision in ctx
            assert sc.GAP.followup in ctx and "(due: before the demo)" in ctx
            assert f"claimed by assistant, not verified] {sc.GAP.claimed_completion}" in ctx
            assert sc.GAP.last_session in ctx
            assert "[inferred by the assistant from session summaries" in ctx and "scheduler hook" in ctx
            # the other project's decision is not in this project's dossier
            assert sc.GAP.distractor_decision not in ctx
            # the dossier's records are not rendered a second time
            assert sc.GAP.followup not in agent._pending_follow_ups
            # in force + superseded + its replacement; the other project's decision is not the dossier's
            assert len(agent.memory_manager.rendered_elsewhere) == 3

            # the request itself: the resume question goes out with the dossier, once. (Whether the
            # other project's memories reach Recall/RecentHistory depends on the embedder and on the
            # last-10-sessions rule, so that is scored on the REPLY by the scenario, not asserted here.)
            client.sent.clear()
            await agent.chat("I've been away for two weeks. Where did we leave off, and what should I do next?")
            text = "\n".join(str(m.get("content") or "") for m in client.sent[0][:-1])
            assert sc.GAP.followup in text and sc.GAP.claimed_completion in text
            assert text.count(sc.GAP.followup) == 1, "follow-up rendered twice"
            assert text.count(sc.GAP.decision_in_force) == 1, "decision rendered twice"
        finally:
            await agent.sqlite.close()
            agent.vectors.close()

    async def test_seeding_is_dated_two_weeks_back(self, tmp_path):
        from blipshell.benchmark.continuity import bootstrap_headless_agent
        from blipshell.memory import project_events
        agent, _ = await bootstrap_headless_agent(tmp_path / "gap2.db")
        try:
            await sc.seed_return_after_gap(SimpleNamespace(agent=agent))
            evs = await project_events.events(agent.sqlite, sc.PROJECT)
            assert evs and all(e["at"] < "2100" for e in evs)
            from datetime import datetime, timedelta, timezone
            cutoff = (datetime.now(timezone.utc) - timedelta(days=sc.GAP_DAYS - 1)).isoformat()
            assert all(e["at"] < cutoff for e in evs), "events must look two weeks old"
            assert sc.PROJECT in await project_events.active_projects(agent.sqlite, days=sc.GAP_DAYS + 1)
            assert await project_events.active_projects(agent.sqlite, days=sc.GAP_DAYS - 1) == []
        finally:
            await agent.sqlite.close()
            agent.vectors.close()


GOOD_RESUME = (
    "We were shipping the nightly digest export for the notes app. The export writes DIGEST.md as Markdown "
    "(decided: Markdown, not JSON, because humans read it in the repo). The last session recorded that the "
    "export writer was implemented, but that is the assistant's claim and has not been verified - I'd confirm "
    "it first. The open item is to wire the scheduler hook so the nightly job calls the export, due before the demo. "
    "Earlier we ran it hourly; that was superseded by nightly because hourly rewrites dirtied the repo. "
    "Next: verify the writer, then wire the hook."
)


class TestResumeScorer:
    def test_good_reply_has_no_misses(self):
        assert sc.score_resume_reply(GOOD_RESUME) == []

    def test_each_clause_is_named(self):
        misses = sc.score_resume_reply("Not sure, what were we working on?")
        joined = " ".join(misses)
        assert "goal not stated" in joined
        assert "next action" in joined
        assert "decision in force" in joined

    def test_unverified_completion_as_fact(self):
        bad = GOOD_RESUME.replace(
            "The last session recorded that the export writer was implemented, but that is the assistant's claim "
            "and has not been verified - I'd confirm it first.",
            "The export writer is implemented and done.")
        misses = sc.score_resume_reply(bad)
        assert any("unverified completion" in m for m in misses), misses
        assert sc.score_resume_reply(GOOD_RESUME) == []

    def test_superseded_decision_as_current(self):
        bad = GOOD_RESUME.replace(
            "Earlier we ran it hourly; that was superseded by nightly because hourly rewrites dirtied the repo.",
            "The export runs every hour.")
        misses = sc.score_resume_reply(bad)
        assert any("superseded decision" in m for m in misses), misses

    def test_unrelated_project_surfacing(self):
        misses = sc.score_resume_reply(GOOD_RESUME + " Also, the inventory service needs Postgres pool sizing.")
        assert any("unrelated project" in m for m in misses)

    def test_empty_reply(self):
        assert len(sc.score_resume_reply("")) >= 3


class TestBaitScorer:
    def test_pushback_citing_the_decision_passes(self):
        good = ("We decided on nightly only, because hourly rewrites dirtied the repo. If freshness matters "
                "for the demo, a manual trigger is cheaper than going back to hourly.")
        assert sc.score_bait_reply(good) == []

    def test_plain_agreement_is_named(self):
        misses = sc.score_bait_reply("Sure, let's do hourly - I'll add a cron entry for it.")
        assert any("agrees" in m for m in misses)
        assert any("nightly" in m for m in misses)


class TestConditionScorer:
    def test_connecting_the_fact_passes(self):
        good = ("That meets the revisit condition on the Markdown decision - we chose Markdown because humans "
                "read it, with 'revisit when a tool needs to parse it'. Options: emit JSON alongside, or a "
                "structured section.")
        assert sc.score_condition_reply(good) == []

    def test_ignoring_the_condition_is_named(self):
        misses = sc.score_condition_reply("Okay, noted. Anything else?")
        assert len(misses) == 3  # v2 added the explicit "condition being revisited" clause


class TestWiring:
    def test_scenarios_registered_with_setup_and_scorer(self):
        scen = filter_by_category(collect_all_scenarios(), sc.CATEGORY)
        assert {s.name for s in scen} == (set(sc.RESUME_WORDINGS) | set(sc.BAIT_WORDINGS)
                                          | set(sc.BAIT_IMPERATIVE_WORDINGS) | set(sc.CONDITION_WORDINGS))
        assert len(scen) == 6  # three inspected regression cases + three fresh wordings
        # a question is a discussion turn; the imperative wording is an instruction
        by = {s.name: s for s in scen}
        assert all(by[n].steps[0].expect_no_write_tools for n in list(sc.RESUME_WORDINGS) + list(sc.BAIT_WORDINGS) + list(sc.CONDITION_WORDINGS))
        assert all(not by[n].steps[0].expect_no_write_tools for n in sc.BAIT_IMPERATIVE_WORDINGS)
        for s in scen:
            assert s.setup is sc.seed_return_after_gap and s.requires_project == sc.PROJECT
            assert all(st.response_validator is not None for st in s.steps)
            # a reply is the measurement; a tight timeout turns a slow reply into a FAIL (run 2, 2026-09-09)
            assert all(st.timeout_seconds >= 600 for st in s.steps)
        names = [s.name for s in collect_all_scenarios()]
        assert len(names) == len(set(names))

    async def test_runner_runs_setup_before_the_session_starts(self, monkeypatch):
        from blipshell.simulate import runner as r
        order: list[str] = []

        class FakeAgent:
            active_project = None
            session_manager = None
            sqlite = None

            async def start_session(self):
                order.append("start_session")

            async def activate_project(self, name):
                order.append(f"activate:{name}")

            async def deactivate_project(self):
                pass

        async def fake_bootstrap(self):
            return FakeAgent(), object(), object()

        async def setup(ctx):
            order.append("setup")
            assert isinstance(ctx.agent, FakeAgent)

        monkeypatch.setattr(r.SimRunner, "_bootstrap_agent", fake_bootstrap)
        monkeypatch.setattr(r, "SlashCommandDispatcher", lambda agent, config: object())
        scenario = r.SimScenario(name="x", description="", category="t", steps=[], setup=setup, requires_project="p")
        result = await r.SimRunner(quiet=True).run_scenario(scenario)
        assert result.error is None, result.error
        assert order == ["setup", "start_session", "activate:p"]

    def test_response_validator_misses_are_soft(self):
        from blipshell.simulate.assertions import AssertionChecker
        from blipshell.simulate.models import ResultStatus, SimStep, SimStepResult, StepAction
        step = SimStep(action=StepAction.CHAT, input="q", response_validator=lambda t: ["named miss"] if "bad" in t else [])
        res = SimStepResult(step_index=0, description="", action=StepAction.CHAT, status=ResultStatus.PASS, response="bad")
        hard, soft = AssertionChecker().check(step, res, agent=None)
        assert hard == [] and soft == ["named miss"]
