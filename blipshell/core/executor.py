"""Step-by-step task executor.

Loops through plan steps sequentially. For each step, builds a focused
prompt with accumulated context and runs the existing tool-calling loop.
"""

import logging
from typing import Callable, Optional

from blipshell.core.tools.base import ToolRegistry
from blipshell.llm.client import LLMClient
from blipshell.llm.prompts import execute_step, summarize_plan_results, UTILITY_SYSTEM_PROMPT
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import PlannerConfig
from blipshell.models.task import PlanStatus, StepStatus, TaskPlan
from blipshell.models.tools import ToolCall

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Executes a TaskPlan step-by-step, reusing the agent's tool-calling loop."""

    def __init__(
        self,
        router: LLMRouter,
        sqlite: SQLiteStore,
        tool_registry: ToolRegistry,
        config: PlannerConfig,
        system_prompt: str = "",
        max_tool_iterations: int = 5,
    ):
        self.router = router
        self.sqlite = sqlite
        self.tool_registry = tool_registry
        self.config = config
        self.system_prompt = system_prompt
        self.max_tool_iterations = max_tool_iterations

    async def execute_plan(
        self,
        plan: TaskPlan,
        on_step_start: Optional[Callable[[int, int, str], None]] = None,
        on_step_complete: Optional[Callable[[int, int, str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Execute all steps in a plan sequentially.

        Args:
            plan: The plan to execute
            on_step_start: Callback(step_num, total, description) when a step begins
            on_step_complete: Callback(step_num, total, result_summary) when a step finishes
            on_token: Token streaming callback

        Returns:
            Final summary of all step results
        """
        await self.sqlite.update_plan(plan.id, status=PlanStatus.RUNNING)

        completed_summaries: list[str] = []
        step_results: list[str] = []
        total_steps = len(plan.steps)

        for step in plan.steps:
            if on_step_start:
                on_step_start(step.step_number, total_steps, step.description)

            # Execute the step with retries
            success = False
            for attempt in range(self.config.max_retries_per_step + 1):
                try:
                    result = await self._execute_step(
                        plan=plan,
                        step_number=step.step_number,
                        step_description=step.description,
                        total_steps=total_steps,
                        completed_summaries=completed_summaries,
                        on_token=on_token,
                    )

                    # Update step as completed
                    await self.sqlite.update_step(
                        step.id,
                        status=StepStatus.COMPLETED,
                        output_result=result[:4000],  # cap storage
                        retry_count=attempt,
                    )

                    completed_summaries.append(
                        f"{step.description}: {result[:200]}"
                    )
                    step_results.append(result)
                    success = True

                    if on_step_complete:
                        on_step_complete(step.step_number, total_steps, result[:200])

                    break

                except Exception as e:
                    logger.error(
                        "Step %d/%d failed (attempt %d): %s",
                        step.step_number, total_steps, attempt + 1, e,
                    )
                    if attempt >= self.config.max_retries_per_step:
                        await self.sqlite.update_step(
                            step.id,
                            status=StepStatus.FAILED,
                            error_message=str(e),
                            retry_count=attempt + 1,
                        )

            if not success:
                # Mark remaining steps as skipped
                for remaining in plan.steps[step.step_number:]:
                    await self.sqlite.update_step(
                        remaining.id, status=StepStatus.SKIPPED,
                    )
                await self.sqlite.update_plan(plan.id, status=PlanStatus.FAILED)
                return f"Plan failed at step {step.step_number}: {step.description}"

        # All steps completed — generate summary
        if on_token:
            on_token("\n[Summarizing results...]\n\n")

        try:
            summary = await self._generate_summary(plan.user_request, step_results)
        except Exception as e:
            logger.error("Summary generation failed: %s", e)
            # Fallback: just return the last step result
            summary = step_results[-1] if step_results else "Plan completed but summary generation failed."

        await self.sqlite.update_plan(
            plan.id,
            status=PlanStatus.COMPLETED,
            result_summary=summary,
        )

        return summary

    async def _execute_step(
        self,
        plan: TaskPlan,
        step_number: int,
        step_description: str,
        total_steps: int,
        completed_summaries: list[str],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Execute a single step using the LLM + tool-calling loop."""
        # Mark step as running
        step = plan.steps[step_number - 1]
        await self.sqlite.update_step(step.id, status=StepStatus.RUNNING)

        # Build focused prompt for this step
        step_prompt = execute_step(
            user_request=plan.user_request,
            step_description=step_description,
            step_number=step_number,
            total_steps=total_steps,
            completed_summaries=completed_summaries,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": step_prompt},
        ]

        # Get model and client
        model = self.router.get_model(TaskType.REASONING)
        client = await self.router.get_client(TaskType.REASONING)
        if not client:
            raise RuntimeError("No available LLM endpoint")

        tools = self.tool_registry.get_all_ollama_tools() or None
        max_iterations = self.max_tool_iterations if tools else 0

        # Tool-calling loop (same pattern as Agent.chat)
        full_response = ""
        for iteration in range(max_iterations + 1):
            response = await client.chat(
                messages=messages,
                model=model,
                tools=tools if iteration < max_iterations else None,
            )

            content, tool_calls = self._extract_response(response)

            if tool_calls and iteration < max_iterations:
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })

                for tc in tool_calls:
                    name, arguments = self._extract_tool_call_info(tc)
                    tool_call = ToolCall(name=name, arguments=arguments)

                    if on_token:
                        on_token(f"\n  [Tool: {tool_call.name}]\n")

                    result = await self.tool_registry.execute_tool_call(tool_call)
                    messages.append(result.to_ollama_message())

                    if on_token:
                        on_token(f"  [Result: {result.result[:150]}]\n")

                continue
            else:
                full_response = content
                break

        return full_response

    async def _generate_summary(
        self, user_request: str, step_results: list[str],
    ) -> str:
        """Generate a final summary from all step results.

        Uses the reasoning model (not summarization) since the summary
        needs to synthesize tool results — not just compress text.
        """
        prompt = summarize_plan_results(user_request, step_results)
        return await self.router.generate(
            TaskType.REASONING, prompt, system=UTILITY_SYSTEM_PROMPT,
        )

    @staticmethod
    def _extract_response(response) -> tuple[str, list | None]:
        """Extract content and tool_calls from an Ollama response."""
        msg = getattr(response, "message", None)
        if msg is not None:
            content = getattr(msg, "content", "") or ""
            tool_calls = getattr(msg, "tool_calls", None)
            return content, tool_calls

        if isinstance(response, dict):
            msg = response.get("message", {})
            return msg.get("content", ""), msg.get("tool_calls", None)

        return "", None

    @staticmethod
    def _extract_tool_call_info(tc) -> tuple[str, dict]:
        """Extract name and arguments from a tool call."""
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", "") or ""
            args = getattr(fn, "arguments", {}) or {}
            return name, args

        if isinstance(tc, dict):
            fn = tc.get("function", {})
            return fn.get("name", ""), fn.get("arguments", {})

        return "", {}
