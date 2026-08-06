"""Every executor path goes through the shared loop runner.

Two gaps this pins (deep-dive 2026-08-04):

1. `_execute_step` — the path behind run_workflow and /workflow — called
   ChatLoop directly instead of going through chat_loop_runner, so a workflow
   step got no endpoint fallback, no vision gating and no PII scrubbing. A
   whole execution path was missing machinery every other path has.

2. `execute_dynamic` appended the task instruction with no image refs, so a
   vision turn's image rode only on chat_history — which is truncated to the
   last 10 messages. Past that, the executor ran a vision task blind, with no
   error to say so.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.core.chat_loop import LoopResult
from blipshell.core.executor import TaskExecutor
from blipshell.core.tools.base import ToolRegistry
from blipshell.models.config import PlannerConfig
from blipshell.models.task import PlanStatus, StepStatus, TaskPlan, TaskStep


def _executor(**kw):
    ex = TaskExecutor(
        router=MagicMock(),
        sqlite=MagicMock(),
        tool_registry=ToolRegistry(),
        config=PlannerConfig(),
        **kw,
    )
    ex.sqlite.update_step = AsyncMock()
    return ex


class _RecordingRunner:
    """Stands in for Agent._run_chat_loop, capturing what it was handed."""

    def __init__(self, response="done"):
        self.calls: list[dict] = []
        self._response = response

    async def __call__(self, messages, config, on_token=None, on_tool_executed=None):
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "config": config,
        })
        return (
            LoopResult(response=self._response, messages=list(messages)),
            "endpoint-A", "model-A", False,
        )


class TestWorkflowStepUsesTheRunner:
    async def test_execute_step_goes_through_chat_loop_runner(self):
        """Without this, a workflow step has no endpoint fallback at all."""
        runner = _RecordingRunner(response="step done")
        ex = _executor()
        ex.chat_loop_runner = runner

        plan = TaskPlan(
            id=1, user_request="build the thing", status=PlanStatus.RUNNING,
            steps=[TaskStep(id=10, plan_id=1, step_number=1,
                            description="first step", status=StepStatus.PENDING)],
        )

        out = await ex._execute_step(
            plan=plan, step_number=1, step_description="first step",
            total_steps=1, completed_summaries=[],
        )

        assert out == "step done"
        assert len(runner.calls) == 1, "the step bypassed the shared runner"

    async def test_step_prompt_and_system_prompt_still_reach_the_model(self):
        runner = _RecordingRunner()
        ex = _executor(system_prompt="SYSTEM RULES")
        ex.chat_loop_runner = runner
        plan = TaskPlan(
            id=1, user_request="build the thing", status=PlanStatus.RUNNING,
            steps=[TaskStep(id=10, plan_id=1, step_number=1,
                            description="first step", status=StepStatus.PENDING)],
        )

        await ex._execute_step(
            plan=plan, step_number=1, step_description="first step",
            total_steps=2, completed_summaries=["earlier work"],
        )

        msgs = runner.calls[0]["messages"]
        assert msgs[0]["role"] == "system" and "SYSTEM RULES" in msgs[0]["content"]
        assert "first step" in msgs[1]["content"]

    async def test_no_endpoint_raises_rather_than_returning_none(self):
        async def _no_endpoint(messages, config, on_token=None, on_tool_executed=None):
            return None, "", "", False

        ex = _executor()
        ex.chat_loop_runner = _no_endpoint
        plan = TaskPlan(
            id=1, user_request="x", status=PlanStatus.RUNNING,
            steps=[TaskStep(id=10, plan_id=1, step_number=1,
                            description="s", status=StepStatus.PENDING)],
        )

        with pytest.raises(RuntimeError, match="No available LLM endpoint"):
            await ex._execute_step(
                plan=plan, step_number=1, step_description="s",
                total_steps=1, completed_summaries=[],
            )


class TestImagesRideOnTheTaskMessage:
    async def _run(self, images):
        runner = _RecordingRunner()
        ex = _executor()
        ex.chat_loop_runner = runner
        ex.router.get_model.return_value = "m"
        ex.router._endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=None)
        await ex.execute_dynamic("describe this screenshot", images=images)
        return runner.calls[0]["messages"]

    async def test_image_refs_land_on_the_task_instruction(self):
        refs = [{"path": "/tmp/shot.png", "orig_name": "shot.png"}]
        msgs = await self._run(refs)

        task_msg = msgs[-1]
        assert task_msg["role"] == "user"
        assert task_msg.get("_image_refs") == refs, (
            "the image rode only on history, which is truncated to 10 turns — "
            "past that the executor runs a vision task blind"
        )

    async def test_no_images_leaves_the_message_clean(self):
        msgs = await self._run(None)
        assert "_image_refs" not in msgs[-1]

    async def test_vision_gating_can_see_them(self):
        """has_image_refs is what _run_chat_loop uses to pick a vision-capable
        endpoint — it has to find them on the messages the runner receives."""
        from blipshell.core.vision import has_image_refs

        msgs = await self._run([{"path": "/tmp/a.png", "orig_name": "a.png"}])
        assert has_image_refs(msgs)


class TestDirectPathStillWorksWithoutAnAgent:
    """The benchmark harness builds a bare TaskExecutor with no runner. That
    path keeps working — it deliberately measures the model, not the fallback
    machinery — but it is now the single shared _run_loop, not a duplicate."""

    async def test_falls_back_to_a_single_endpoint(self):
        ex = _executor()
        assert ex.chat_loop_runner is None

        endpoint = MagicMock()
        endpoint.name = "solo"
        endpoint.models = {}
        endpoint.context_tokens = 4096
        endpoint.provider = "openai"
        ex.router._endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=endpoint)
        ex.router.get_model.return_value = "model-x"

        from blipshell.core import executor as executor_mod

        class _Loop:
            def __init__(self, *a, **k):
                pass

            async def run(self, **kwargs):
                return LoopResult(response="direct", messages=kwargs["messages"])

        original = executor_mod.ChatLoop
        executor_mod.ChatLoop = _Loop
        try:
            from blipshell.core.chat_loop import LoopConfig
            result, name, model, fb = await ex._run_loop(
                [{"role": "user", "content": "hi"}], LoopConfig(budget=1), None,
            )
        finally:
            executor_mod.ChatLoop = original

        assert result.response == "direct"
        assert name == "solo"
        assert model == "model-x"
        assert fb is False

    async def test_missing_endpoint_returns_none_not_raises(self):
        ex = _executor()
        ex.router._endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=None)
        from blipshell.core.chat_loop import LoopConfig

        result, name, model, fb = await ex._run_loop([], LoopConfig(budget=1), None)
        assert result is None
