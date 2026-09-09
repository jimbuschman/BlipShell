"""Preserved simulate runs carry provenance and the reply text.

A behavioural result is only comparable to another if both say which commit,
which host and which models produced them; and a run can only be re-read for
miss PATTERNS if the reply itself was kept. Neither used to be exported.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from blipshell.simulate.models import ResultStatus, SimScenarioResult, SimStepResult, SimSuiteResult, StepAction
from blipshell.simulate.reporting import RESPONSE_EXCERPT_CHARS, export_json, run_provenance


def _suite():
    step = SimStepResult(step_index=0, description="q", action=StepAction.CHAT, status=ResultStatus.WARN,
                         response="x" * (RESPONSE_EXCERPT_CHARS + 50), soft_failures=["goal not stated"])
    return SimSuiteResult(scenario_results=[SimScenarioResult(name="s", category="continuity",
                                                              status=ResultStatus.WARN, step_results=[step])])


def test_export_carries_provenance_and_reply_excerpt():
    prov = {"kind": "behavioural", "git_sha": "abc1234", "host": "h", "run_ts": "20260909T000000"}
    data = json.loads(export_json(_suite(), provenance=prov))
    assert data["kind"] == "behavioural" and data["git_sha"] == "abc1234" and data["host"] == "h"
    step = data["scenarios"][0]["steps"][0]
    assert step["soft_failures"] == ["goal not stated"]
    assert len(step["response"]) == RESPONSE_EXCERPT_CHARS


def test_export_without_provenance_is_unchanged_shape():
    data = json.loads(export_json(_suite()))
    assert "git_sha" not in data and data["summary"]["warned"] == 1


def test_run_provenance_names_endpoints_and_models_but_never_urls():
    ep = SimpleNamespace(name="local-cloud", enabled=True, roles=["tool_calling"], url="http://100.1.2.3:11434",
                         models={"tool_calling": "gemma4:31b-cloud"})
    cfg = SimpleNamespace(endpoints=[ep], models=SimpleNamespace(tool_calling="x", reasoning="qwen3:14b"))
    prov = run_provenance(cfg)
    assert prov["kind"] == "behavioural" and prov["git_sha"] and prov["host"] and prov["run_ts"]
    assert prov["endpoints"] == [{"name": "local-cloud", "enabled": True, "roles": ["tool_calling"],
                                  "models": {"tool_calling": "gemma4:31b-cloud"}}]
    assert prov["models"] == {"tool_calling": "x", "reasoning": "qwen3:14b"}
    assert "100.1.2.3" not in json.dumps(prov)


def test_run_provenance_without_config():
    prov = run_provenance(None)
    assert "endpoints" not in prov and prov["kind"] == "behavioural"


async def test_chat_step_records_the_model_that_served_the_reply():
    """The configured primary may be unreachable; the reply's real producer is recorded per step."""
    from blipshell.simulate.models import SimStep
    from blipshell.simulate.step_executor import SimStepExecutor

    class FakeAgent:
        _last_tool_calls: list = []
        _last_model_used = {"endpoint": "local", "model": "gpt-oss:latest", "fallback": True}

        async def chat(self, text, on_token=None, force_plan=False):
            return "a reply"

    ctx = SimpleNamespace(agent=FakeAgent(), responses=[], all_tool_calls=[])
    res = await SimStepExecutor().execute(SimStep(action=StepAction.CHAT, input="q"), 0, ctx)
    assert res.model_used == {"endpoint": "local", "model": "gpt-oss:latest", "fallback": True}
    suite = SimSuiteResult(scenario_results=[SimScenarioResult(name="s", category="c", status=res.status,
                                                               step_results=[res])])
    assert json.loads(export_json(suite))["scenarios"][0]["steps"][0]["model_used"]["model"] == "gpt-oss:latest"
