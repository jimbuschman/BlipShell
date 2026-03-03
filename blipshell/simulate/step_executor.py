"""Executes individual simulation steps against a live Agent."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING

from blipshell.simulate.assertions import AssertionChecker
from blipshell.simulate.models import (
    ResultStatus,
    SimStep,
    SimStepResult,
    StepAction,
)

if TYPE_CHECKING:
    from blipshell.simulate.runner import SimContext

logger = logging.getLogger(__name__)
_checker = AssertionChecker()


class SimStepExecutor:
    """Executes one SimStep and collects results."""

    async def execute(
        self,
        step: SimStep,
        step_index: int,
        ctx: "SimContext",
    ) -> SimStepResult:
        """Execute a step and return its result."""
        t0 = time.monotonic()
        response = ""
        tools_called: list[str] = []
        tool_call_count = 0
        error: str | None = None

        try:
            result = await asyncio.wait_for(
                self._dispatch(step, ctx),
                timeout=step.timeout_seconds,
            )
            response = result.get("response", "")
            tools_called = result.get("tools_called", [])
            tool_call_count = result.get("tool_call_count", 0)
        except asyncio.TimeoutError:
            error = f"Step timed out after {step.timeout_seconds}s"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.exception("Step %d failed: %s", step_index, step.description)

        elapsed = time.monotonic() - t0

        # Build preliminary result (without assertions)
        step_result = SimStepResult(
            step_index=step_index,
            description=step.description,
            action=step.action,
            status=ResultStatus.PASS,  # will be updated below
            response=response,
            tools_called=tools_called,
            tool_call_count=tool_call_count,
            error=error,
            elapsed_seconds=round(elapsed, 2),
        )

        # Run assertions
        hard_failures, soft_failures = _checker.check(step, step_result, ctx.agent)
        step_result.hard_failures = hard_failures
        step_result.soft_failures = soft_failures

        # Determine status
        if error or hard_failures:
            step_result.status = ResultStatus.FAIL
        elif soft_failures:
            step_result.status = ResultStatus.WARN
        else:
            step_result.status = ResultStatus.PASS

        return step_result

    async def _dispatch(
        self,
        step: SimStep,
        ctx: "SimContext",
    ) -> dict:
        """Route step to the right handler. Returns {response, tools_called, tool_call_count}."""
        if step.action == StepAction.CHAT:
            return await self._exec_chat(step, ctx)
        elif step.action == StepAction.SLASH:
            return await self._exec_slash(step, ctx)
        elif step.action == StepAction.AGENT_METHOD:
            return await self._exec_agent_method(step, ctx)
        elif step.action == StepAction.ASSERT_STATE:
            return {"response": "", "tools_called": [], "tool_call_count": 0}
        elif step.action == StepAction.WAIT:
            await asyncio.sleep(step.wait_seconds)
            return {"response": "", "tools_called": [], "tool_call_count": 0}
        else:
            raise ValueError(f"Unknown action: {step.action}")

    async def _exec_chat(self, step: SimStep, ctx: "SimContext") -> dict:
        """Send a chat message through agent.chat() and capture results."""
        tokens: list[str] = []

        def on_token(chunk: str):
            tokens.append(chunk)

        response = await ctx.agent.chat(
            step.input,
            on_token=on_token,
            force_plan=step.force_plan,
        )

        # Primary: read tool calls from Agent's structured tracking
        # (set by _chat_simple/_chat_planned after each chat call)
        tools_called: list[str] = []
        agent_tools = getattr(ctx.agent, '_last_tool_calls', [])
        if agent_tools:
            tools_called = [t.get("name", "") for t in agent_tools if t.get("name")]

        # Fallback: parse streaming output (catches edge cases)
        if not tools_called:
            raw_output = "".join(tokens)
            # Match both formats: "▸ tool_name" and "Running N tools: tool1, tool2"
            single = re.findall(r"\u25b8 (\w+)", raw_output)
            multi = re.findall(r"Running \d+ tools: (.+?)(?:\x1b|$)", raw_output)
            if single:
                tools_called = single
            elif multi:
                for group in multi:
                    tools_called.extend(
                        name.strip() for name in group.split(",") if name.strip()
                    )

        tool_call_count = len(tools_called)

        ctx.responses.append(response)
        ctx.all_tool_calls.extend(
            {"name": t, "step": step.description} for t in tools_called
        )

        return {
            "response": response,
            "tools_called": tools_called,
            "tool_call_count": tool_call_count,
        }

    async def _exec_slash(self, step: SimStep, ctx: "SimContext") -> dict:
        """Execute a slash command through the dispatcher."""
        result = await ctx.slash_dispatcher.execute(step.input)

        return {
            "response": result.output,
            "tools_called": [],
            "tool_call_count": 0,
        }

    async def _exec_agent_method(self, step: SimStep, ctx: "SimContext") -> dict:
        """Call an agent method directly (for setup/teardown)."""
        agent = ctx.agent

        # Navigate dotted method paths like "tool_registry.in_plan_mode"
        parts = step.method.split(".")
        obj = agent
        for part in parts[:-1]:
            obj = getattr(obj, part)
        method_name = parts[-1]

        attr = getattr(obj, method_name)

        if callable(attr):
            if asyncio.iscoroutinefunction(attr):
                result = await attr(**step.method_args)
            else:
                result = attr(**step.method_args)
        else:
            # It's a property — set it if method_args has 'value', else just read
            if "value" in step.method_args:
                setattr(obj, method_name, step.method_args["value"])
                result = step.method_args["value"]
            else:
                result = attr

        return {
            "response": str(result) if result is not None else "",
            "tools_called": [],
            "tool_call_count": 0,
        }
