"""Assertion checker for simulation steps.

Evaluates step expectations against actual results.
Returns (hard_failures, soft_failures) — hard = deterministic, soft = LLM-dependent.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blipshell.simulate.models import SimStep, SimStepResult, StepAction


class AssertionChecker:
    """Evaluates expectations on a step result + agent state."""

    def check(
        self,
        step: SimStep,
        result: SimStepResult,
        agent,
    ) -> tuple[list[str], list[str]]:
        """Check all expectations.

        Returns (hard_failures, soft_failures).
        """
        hard: list[str] = []
        soft: list[str] = []

        # --- Tool call expectations (hard) ---
        if step.expect_tools:
            for tool_name in step.expect_tools:
                if tool_name not in result.tools_called:
                    hard.append(f"Expected tool not called: {tool_name}")

        if step.expect_no_tools and result.tool_call_count > 0:
            hard.append(
                f"Expected no tools but {result.tool_call_count} called: "
                f"{', '.join(result.tools_called)}"
            )

        if (
            step.expect_max_tool_calls is not None
            and result.tool_call_count > step.expect_max_tool_calls
        ):
            hard.append(
                f"Tool calls {result.tool_call_count} > max {step.expect_max_tool_calls}"
            )

        # --- Response content expectations (soft — LLM-dependent) ---
        if step.expect_response_contains:
            for substr in step.expect_response_contains:
                if substr.lower() not in result.response.lower():
                    soft.append(f"Response missing: '{substr}'")

        if step.expect_response_not_contains:
            for substr in step.expect_response_not_contains:
                if substr.lower() in result.response.lower():
                    soft.append(f"Response should not contain: '{substr}'")

        if step.expect_no_error and result.error:
            hard.append(f"Step error: {result.error}")

        if step.expect_no_error and "error:" in result.response.lower()[:100]:
            soft.append(f"Response starts with error: {result.response[:200]}")

        # --- State expectations (hard — deterministic) ---
        if step.expect_tools_registered:
            registered = set(agent.tool_registry.get_tool_names())
            for tool_name in step.expect_tools_registered:
                if tool_name not in registered:
                    close = difflib.get_close_matches(tool_name, registered, n=3, cutoff=0.6)
                    hint = f" (did you mean: {', '.join(close)}?)" if close else ""
                    hard.append(f"Tool not registered: {tool_name}{hint}")

        if step.expect_tools_not_registered:
            registered = set(agent.tool_registry.get_tool_names())
            for tool_name in step.expect_tools_not_registered:
                if tool_name in registered:
                    hard.append(f"Tool should not be registered: {tool_name}")

        if step.expect_project_active is not None:
            actual = agent.active_project.get("name") if agent.active_project else None
            if actual != step.expect_project_active:
                hard.append(
                    f"Expected project '{step.expect_project_active}', "
                    f"got '{actual}'"
                )

        if step.expect_project_inactive:
            if agent.active_project:
                hard.append(
                    f"Expected no active project, "
                    f"got '{agent.active_project.get('name')}'"
                )

        if step.expect_think_enabled is not None:
            if agent.think_enabled != step.expect_think_enabled:
                hard.append(
                    f"Expected think_enabled={step.expect_think_enabled}, "
                    f"got {agent.think_enabled}"
                )

        if step.expect_reflect_enabled is not None:
            if agent.reflect_enabled != step.expect_reflect_enabled:
                hard.append(
                    f"Expected reflect_enabled={step.expect_reflect_enabled}, "
                    f"got {agent.reflect_enabled}"
                )

        # --- File expectations (hard) ---
        if step.expect_files_exist:
            for fpath in step.expect_files_exist:
                if not Path(fpath).exists():
                    hard.append(f"File not found: {fpath}")

        if step.expect_files_not_exist:
            for fpath in step.expect_files_not_exist:
                if Path(fpath).exists():
                    hard.append(f"File should not exist: {fpath}")

        # --- Custom validator ---
        if step.custom_validator:
            try:
                custom_failures = step.custom_validator(agent)
                hard.extend(custom_failures)
            except Exception as e:
                hard.append(f"Custom validator raised: {e}")

        return hard, soft
