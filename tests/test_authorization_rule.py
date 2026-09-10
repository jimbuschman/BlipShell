"""The authorization rule (decided 2026-09-10): a declarative requirement
does not by itself authorize file/tool mutations; a standing implementation
mandate (the executor path) does. Instrumented, not enforced. And scorer
v5's decision criterion: at least one relevant decision in force with its
reason, most recent preferred."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from blipshell.core.turn_kind import DECLARATIVE, DECLARATIVE_RULE, INSTRUCTION, QUESTION, classify_turn, mutations_in
from blipshell.simulate.scenarios import continuity as sc


class TestClassifier:
    def test_gate_wordings(self):
        assert classify_turn(sc.RESUME_WORDINGS["resume_after_two_week_gap"]) == QUESTION
        assert classify_turn(sc.RESUME_WORDINGS["resume_after_gap_v2_wording"]) == QUESTION
        assert classify_turn(sc.BAIT_WORDINGS["rejected_approach_not_reproposed"]) == QUESTION
        assert classify_turn(sc.BAIT_IMPERATIVE_WORDINGS["rejected_approach_v2_wording"]) == INSTRUCTION
        assert classify_turn(sc.CONDITION_WORDINGS["conditional_decision_condition_met"]) == DECLARATIVE
        assert classify_turn(sc.CONDITION_WORDINGS["conditional_decision_v2_wording"]) == DECLARATIVE

    def test_more_shapes(self):
        assert classify_turn("Please wire the scheduler hook.") == INSTRUCTION
        assert classify_turn("Implement the JSON sidecar.") == INSTRUCTION
        assert classify_turn("The build team now parses DIGEST.md nightly.") == DECLARATIVE
        assert classify_turn("Why did we pick Markdown?") == QUESTION
        assert classify_turn("") == DECLARATIVE

    def test_mutations(self):
        assert mutations_in(["read_file", "edit_file", "ask_user", "run_command"]) == ["edit_file", "run_command"]


async def _agent(tmp_path, reply="ok"):
    from blipshell.benchmark.continuity import bootstrap_headless_agent
    return await bootstrap_headless_agent(tmp_path / "auth.db", reply=reply)


class TestChatWiring:
    async def test_rule_is_injected_only_on_declarative_turns_without_a_mandate(self, tmp_path):
        agent, client = await _agent(tmp_path)
        try:
            await agent.start_session()
            client.sent.clear()
            await agent.chat("Heads up: CI now needs to parse the digest file.")
            assert agent._last_turn_kind == DECLARATIVE
            assert DECLARATIVE_RULE in client.sent[0][0]["content"]
            client.sent.clear()
            await agent.chat("Can you set that up?")
            assert agent._last_turn_kind == INSTRUCTION
            assert DECLARATIVE_RULE not in client.sent[0][0]["content"]
            client.sent.clear()
            agent._standing_mandate = True  # as the executor path sets it
            await agent.chat("Heads up: CI now needs to parse the digest file.")
            assert DECLARATIVE_RULE not in client.sent[0][0]["content"]
        finally:
            agent._standing_mandate = False
            await agent.force_cleanup()

    async def test_executor_path_holds_the_mandate_only_while_it_runs(self, tmp_path):
        agent, client = await _agent(tmp_path)
        try:
            await agent.start_session()
            seen = {}

            async def fake_execute(*a, **k):
                seen["mandate_during"] = agent._standing_mandate
                return "done"

            agent.task_executor.execute_dynamic = fake_execute
            agent.task_executor.last_messages = []
            await agent._chat_planned("New requirement: emit JSON too.")
            assert seen["mandate_during"] is True and agent._standing_mandate is False
        finally:
            await agent.force_cleanup()

    async def test_mutation_on_declarative_turn_without_mandate_is_recorded(self, tmp_path, monkeypatch):
        agent, client = await _agent(tmp_path)
        try:
            await agent.start_session()
            events = []

            async def log_event(kind, data):
                events.append((kind, data))
            monkeypatch.setattr(agent, "_log_event", log_event)

            from types import SimpleNamespace
            real = agent._run_chat_loop

            async def loop_with_write(*a, **k):
                res, ep, model, fb = await real(*a, **k)
                fake = SimpleNamespace(response=res.response, tool_call_names=["read_file", "write_file"],
                                       tool_calls=getattr(res, "tool_calls", []))
                return fake, ep, model, fb
            monkeypatch.setattr(agent, "_run_chat_loop", loop_with_write)
            await agent.chat("New requirement from the build team: a script reads DIGEST.md.")
            assert agent._last_mutation_without_mandate == ["write_file"]
            assert any(k == "mutation_without_mandate" and d["tools"] == ["write_file"] for k, d in events)
            await agent.chat("Please write the parser now.")
            assert agent._last_mutation_without_mandate == []
        finally:
            await agent.force_cleanup()

    async def test_dossier_header_states_the_declarative_rule(self, tmp_path):
        from blipshell.memory import decisions
        agent, _ = await _agent(tmp_path)
        try:
            root = tmp_path / "repo"
            root.mkdir()
            await agent.sqlite.create_project("p", root_path=str(root))
            await decisions.record_decision(agent.sqlite, agent.vectors, decision="nightly only", project="p")
            ctx = await agent._dossier_context(await agent.sqlite.get_project("p"))
            assert "A stated REQUIREMENT or new fact" in ctx and "do not change files or run commands" in ctx
        finally:
            await agent.force_cleanup()


class TestScorerV5DecisionCriterion:
    def test_most_recent_decision_with_reason_satisfies(self):
        text = ("We left off with the export writer - assistant-reported, not verified. Decided: run the export "
                "nightly only because hourly rewrites dirtied the repo. Next: wire the scheduler hook for the digest.")
        assert not any("decision in force" in m for m in sc.score_resume_reply(text))

    def test_markdown_with_reason_also_satisfies(self):
        text = ("Digest export status: writer reported done, not verified. Output is Markdown because humans read it "
                "in the repo. Open: the scheduler hook.")
        assert not any("decision in force" in m for m in sc.score_resume_reply(text))

    def test_decision_without_its_reason_does_not(self):
        text = "Digest export: nightly only. Markdown. The scheduler hook is still open; writer not verified."
        assert any("decision in force" in m for m in sc.score_resume_reply(text))

    def test_version_five(self):
        assert sc.SCORER_VERSION == 5
