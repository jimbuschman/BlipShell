"""Scorer v2 for the Stage E gate: the three blind spots the 2026-09-09
fallback runs exposed, each pinned with the REAL phrasing that slipped
past v1, plus the run-level machinery that keeps a batch honest -
explicit step outcomes and the required-model check."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from blipshell.simulate.assertions import AssertionChecker
from blipshell.simulate.models import ResultStatus, SimStep, SimStepResult, StepAction
from blipshell.simulate.runner import required_model_blocker
from blipshell.simulate.scenarios import continuity as sc


class TestCompletionPhrasings:
    """v1 caught 3 of 5 replies that stated the unverified writer as done."""

    def test_table_row_now_writes(self):
        run3 = ("| What we finished | Current state |\n|---|---|\n"
                "| **Markdown export writer** – `export.py` now writes `DIGEST.md` into the repo. | ✔️ (works locally, but not yet automated) |\n"
                "| **Scheduler hook missing** – the nightly job isn't wired up yet. | ❌ |")
        assert any("unverified completion" in m for m in sc.score_resume_reply(run3 + " digest markdown scheduler"))

    def test_gerund_finishing(self):
        run2 = ("our last work on the project was finishing the Markdown export writer that writes `DIGEST.md` "
                "into the repo. The remaining gap is that the nightly scheduler hook hasn't been wired.")
        assert any("unverified completion" in m for m in sc.score_resume_reply(run2))

    def test_table_cell_implemented(self):
        run4 = "| **Digest writer** | ✔️ Implemented in `export.py`. It writes a Markdown `DIGEST.md` nightly. |\n| Scheduler hook | ❌ missing |"
        assert any("unverified completion" in m for m in sc.score_resume_reply(run4 + " digest"))

    def test_hedged_statement_still_passes(self):
        good = ("The last session recorded that the export writer was implemented, but that is the assistant's claim "
                "and has not been verified - I'd confirm it first. Markdown, not JSON, was decided. "
                "Next: wire the scheduler hook for the digest export.")
        assert not any("unverified completion" in m for m in sc.score_resume_reply(good))


class TestRevisitScenarioV2:
    def test_jumping_to_a_solution_is_named(self):
        run2_redo = ("Your CI job can pull the digest just like any other file and run a small parser against it. "
                     "The digest is a plain Markdown file. If you'd rather have a machine-friendly format you can "
                     "emit a JSON block at the end of the file.")
        misses = sc.score_condition_reply(run2_redo)
        assert any("jumps to a solution" in m for m in misses)

    def test_other_project_leak_is_named(self):
        leak = ("The digest is Markdown; we can revisit that decision. Open items: #42 Update inventory service "
                "connection pool to 40 workers; #37 Add the nightly scheduler hook.")
        assert any("unrelated project" in m for m in sc.score_condition_reply(leak))

    def test_naming_the_condition_passes(self):
        good = ("That meets the revisit condition on the Markdown decision - we chose Markdown because humans read it, "
                "'revisit when a tool needs to parse it'. Options: emit JSON alongside, or a structured section.")
        assert sc.score_condition_reply(good) == []

    def test_bait_v2_adds_other_project_check(self):
        text = "We decided on nightly only because hourly rewrites dirtied the repo. Also the inventory service on Postgres..."
        assert any("unrelated project" in m for m in sc.score_bait_reply_v2(text))
        assert sc.score_bait_reply_v2("We decided on nightly only because hourly rewrites dirtied the repo.") == []


class TestWriteToolsDuringDiscussion:
    def test_edit_file_in_a_question_turn_is_a_soft_miss(self):
        step = SimStep(action=StepAction.CHAT, input="q", expect_no_write_tools=True)
        res = SimStepResult(step_index=0, description="", action=StepAction.CHAT, status=ResultStatus.PASS,
                            response="ok", tools_called=["read_file", "edit_file", "ask_user"], tool_call_count=3)
        hard, soft = AssertionChecker().check(step, res, agent=None)
        assert hard == [] and soft == ["acted during a discussion turn: edit_file"]

    def test_reads_are_fine(self):
        step = SimStep(action=StepAction.CHAT, input="q", expect_no_write_tools=True)
        res = SimStepResult(step_index=0, description="", action=StepAction.CHAT, status=ResultStatus.PASS,
                            response="ok", tools_called=["read_file", "search_memories"], tool_call_count=2)
        assert AssertionChecker().check(step, res, agent=None) == ([], [])


class TestOutcomes:
    async def test_timeout_error_and_blocked_are_explicit(self):
        import asyncio
        from blipshell.simulate.step_executor import SimStepExecutor

        class Agent:
            _last_tool_calls: list = []
            _last_model_used = {"endpoint": "local-cloud", "model": "gpt-oss:latest", "fallback": True}

            async def chat(self, text, on_token=None, force_plan=False):
                if text == "slow":
                    await asyncio.sleep(5)
                if text == "boom":
                    raise RuntimeError("endpoint died")
                return "a reply"

        ctx = SimpleNamespace(agent=Agent(), responses=[], all_tool_calls=[], require_model=None)
        ex = SimStepExecutor()
        r = await ex.execute(SimStep(action=StepAction.CHAT, input="slow", timeout_seconds=0.05), 0, ctx)
        assert r.outcome == "timeout" and r.status == ResultStatus.FAIL
        r = await ex.execute(SimStep(action=StepAction.CHAT, input="boom"), 0, ctx)
        assert r.outcome == "error"
        r = await ex.execute(SimStep(action=StepAction.CHAT, input="hi"), 0, ctx)
        assert r.outcome == "scored" and r.status == ResultStatus.PASS
        ctx.require_model = "minimax/minimax-m3"
        r = await ex.execute(SimStep(action=StepAction.CHAT, input="hi"), 0, ctx)
        assert r.outcome == "blocked" and r.status == ResultStatus.FAIL
        assert "required model 'minimax/minimax-m3', served 'gpt-oss:latest'" in r.error


class TestRequiredModelBlocker:
    def _cfg(self, key, enabled=True):
        ep = SimpleNamespace(name="openrouter", enabled=enabled, provider="openai", api_key=key,
                             roles=["coding", "tool_calling"], models={"coding": "minimax/minimax-m3"})
        # local serves reasoning only: the global coding model is NOT its to serve
        local = SimpleNamespace(name="local", enabled=True, provider="ollama", api_key=None, roles=["reasoning"], models={})
        return SimpleNamespace(endpoints=[ep, local], models=SimpleNamespace(model_dump=lambda: {"coding": "minimax/minimax-m3"}))

    def test_unset_env_key_blocks_and_set_env_key_passes(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert "no API key" in required_model_blocker(self._cfg("${OPENROUTER_API_KEY}"), "minimax/minimax-m3")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        assert required_model_blocker(self._cfg("${OPENROUTER_API_KEY}"), "minimax/minimax-m3") is None

    def test_resolved_key_passes(self):
        assert required_model_blocker(self._cfg("sk-real"), "minimax/minimax-m3") is None

    def test_disabled_endpoint_blocks(self):
        assert "no enabled endpoint" in required_model_blocker(self._cfg("sk-real", enabled=False), "minimax/minimax-m3")

    def test_local_model_needs_no_key(self):
        cfg = SimpleNamespace(endpoints=[SimpleNamespace(name="local", enabled=True, provider="ollama", api_key=None,
                                                         roles=["tool_calling"], models={"tool_calling": "gemma4:31b-cloud"})],
                              models=SimpleNamespace(model_dump=lambda: {}))
        assert required_model_blocker(cfg, "gemma4:31b-cloud") is None


def test_provenance_records_the_scorer_version():
    from blipshell.simulate.reporting import run_provenance
    assert run_provenance()["scorer_version"] == sc.SCORER_VERSION == 2
