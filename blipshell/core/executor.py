"""Step-by-step task executor.

Loops through plan steps sequentially. For each step, builds a focused
prompt with accumulated context and runs the unified ChatLoop.
"""

import json
import logging
from typing import Callable, Optional

from blipshell.core.chat_loop import ChatLoop, LoopConfig, estimate_messages_tokens
from blipshell.core.intent_detection import detect_review_intent, REVIEW_GROUNDING_GUIDANCE
from blipshell.memory.manager import estimate_tokens
from blipshell.core.tools.base import ToolRegistry
from blipshell.llm.prompts import dynamic_execution_prompt, executor_system_prompt, execute_step, summarize_plan_results, UTILITY_SYSTEM_PROMPT
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import GuardrailsConfig, PlannerConfig
from blipshell.models.task import PlanStatus, StepStatus, TaskPlan

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from blipshell.memory.processor import MemoryProcessor

logger = logging.getLogger(__name__)


def build_executor_narrative(messages: list[dict]) -> str:
    """Build a clean, memory-friendly narrative from raw executor messages.

    Extracts the valuable content from an executor transcript:
    - User's original request
    - Assistant's reasoning and planning text
    - Compressed tool call summaries (action + file path)
    - Final TASK_COMPLETE summary

    Drops the noise:
    - System prompts
    - Raw tool results (file contents, directory listings, etc.)
    - Empty messages

    Returns a concise narrative suitable for the memory pipeline.
    """
    import re

    parts: list[str] = []
    files_read: list[str] = []
    files_written: list[str] = []
    files_edited: list[str] = []
    commands_run: list[str] = []
    searches: list[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        # Skip system messages and tool results entirely
        if role == "system":
            continue
        if role == "tool":
            continue

        # User messages — keep the original request (skip nudges)
        if role == "user":
            if content.startswith("Continue.") or content.startswith("Task:"):
                # Keep the task prompt but strip the RULES/APPROACH boilerplate
                task_match = re.match(r"Task:\s*(.+?)(?:\n\nAPPROACH:|$)", content, re.DOTALL)
                if task_match:
                    parts.append(f"Task: {task_match.group(1).strip()}")
                continue
            # Regular user message (from chat history injection)
            parts.append(f"User: {content}")
            continue

        # Assistant messages
        if role == "assistant":
            # Extract reasoning text (the valuable part)
            if content and content.strip():
                # Strip legacy TASK_COMPLETE prefix if present
                text = content.replace("TASK_COMPLETE", "").strip()
                if text:
                    parts.append(text)

            # Summarize tool calls as action log
            if tool_calls:
                for tc in tool_calls:
                    _summarize_tool_call(
                        tc, files_read, files_written, files_edited,
                        commands_run, searches,
                    )

    # Build the final narrative
    narrative_parts = []

    # Add the reasoning/planning text
    if parts:
        narrative_parts.append("\n".join(parts))

    # Add compact action summary
    actions = []
    if files_read:
        # Deduplicate
        unique = list(dict.fromkeys(files_read))
        actions.append(f"Read: {', '.join(unique)}")
    if files_written:
        actions.append(f"Created: {', '.join(files_written)}")
    if files_edited:
        unique = list(dict.fromkeys(files_edited))
        actions.append(f"Edited: {', '.join(unique)}")
    if searches:
        actions.append(f"Searched: {', '.join(searches[:5])}")
    if commands_run:
        actions.append(f"Ran: {', '.join(commands_run[:5])}")

    if actions:
        narrative_parts.append("\nActions:\n" + "\n".join(f"- {a}" for a in actions))

    return "\n\n".join(narrative_parts)


def _summarize_tool_call(
    tc,
    files_read: list[str],
    files_written: list[str],
    files_edited: list[str],
    commands_run: list[str],
    searches: list[str],
):
    """Extract action summary from a single tool call."""
    # Tool calls can be dicts or stringified objects depending on API
    if isinstance(tc, dict):
        func = tc.get("function", {})
        if isinstance(func, dict):
            name = func.get("name", "")
            args = func.get("arguments", {})
            # OpenAI returns arguments as JSON string, not dict
            if isinstance(args, str):
                try:
                    import json
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
        else:
            # Stringified — try to parse
            name = str(func)
            args = {}
    elif isinstance(tc, str):
        # Ollama format: "function=Function(name='read_file', arguments={...})"
        import re
        name_match = re.search(r"name='(\w+)'", tc)
        name = name_match.group(1) if name_match else ""
        path_match = re.search(r"'path':\s*'([^']+)'", tc)
        args = {"path": path_match.group(1)} if path_match else {}
        cmd_match = re.search(r"'command':\s*'([^']+)'", tc)
        if cmd_match:
            args["command"] = cmd_match.group(1)
        pattern_match = re.search(r"'pattern':\s*'([^']+)'", tc)
        if pattern_match:
            args["pattern"] = pattern_match.group(1)
    else:
        return

    path = args.get("path", "")

    if name == "read_file" and path:
        files_read.append(path)
    elif name == "write_file" and path:
        files_written.append(path)
    elif name == "edit_file" and path:
        files_edited.append(path)
    elif name == "list_directory" and path:
        files_read.append(f"{path}/")
    elif name == "run_command":
        cmd = args.get("command", "")[:60]
        if cmd:
            commands_run.append(cmd)
    elif name in ("grep_files", "glob_files"):
        pattern = args.get("pattern", "")
        if pattern:
            searches.append(pattern)


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
        processor: Optional["MemoryProcessor"] = None,
    ):
        self.router = router
        self.sqlite = sqlite
        self.tool_registry = tool_registry
        self.config = config
        self.system_prompt = system_prompt
        self.max_tool_iterations = max_tool_iterations
        self.processor = processor
        # Project-mode overrides (set by Agent when project is active)
        self.active_project: dict | None = None
        self.project_context: str = ""
        self.files_read: set[str] = set()  # shared with Agent to track across steps
        self._file_cache: dict[str, str] = {}  # path → content, for cross-step re-reads
        self._stale_files: set[str] = set()  # files modified since last read — need re-read
        # Per-step tracking for rich context summaries
        self._step_files_created: list[str] = []
        self._step_files_edited: list[str] = []
        # Last execute_dynamic messages — available for narrative building after execution
        self.last_messages: list[dict] = []
        # Interactive callbacks (wired from Agent)
        self.pause_check_callback: Optional[Callable] = None
        # Shared chat loop runner — set by Agent to reuse endpoint/fallback logic
        self.chat_loop_runner: Optional[Callable] = None
        # Guardrails configuration (set by Agent from config)
        self.guardrails_config: Optional[GuardrailsConfig] = None
        # Phase 2 test override flags (set by TestOverrides.apply() or benchmark configs)
        self._disable_winddown: bool = False
        self._disable_state_block: bool = False
        self._natural_completion_primary: bool = False
        # A/B benchmark overrides for LoopConfig (None = use defaults)
        self._override_enable_dedup: bool | None = None
        self._override_enable_compaction: bool | None = None
        # Compaction config (set by Agent from config)
        self._compaction_config = None

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

        # Clear file cache at plan start and wire it into ReadFileTool
        self._file_cache.clear()
        read_tool = self.tool_registry.get_tool("read_file")
        if read_tool is not None:
            read_tool.file_cache = self._file_cache

        for step in plan.steps:
            if on_step_start:
                on_step_start(step.step_number, total_steps, step.description)

            # Reset per-step file tracking
            self._step_files_created.clear()
            self._step_files_edited.clear()

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

                    # Build rich context summary for later steps
                    summary = self._build_step_summary(
                        step.description, result,
                    )
                    completed_summaries.append(summary)
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
                # Detach cache from ReadFileTool on failure exit
                if read_tool is not None:
                    read_tool.file_cache = None
                return f"Plan failed at step {step.step_number}: {step.description}"

        # All steps completed — generate summary
        if on_token:
            on_token("\n[Summarizing results...]\n\n")

        try:
            summary = await self._generate_summary(plan.user_request, step_results)
        except Exception as e:
            logger.error("Summary generation failed: %s", e)
            summary = ""

        # Fallback if summary is empty (LLM returned empty string)
        if not summary or not summary.strip():
            logger.warning("Summary generation returned empty, using last step result as fallback")
            summary = step_results[-1] if step_results else "Plan completed but summary generation failed."

        # Stream summary to user
        if on_token and summary:
            on_token(summary)

        await self.sqlite.update_plan(
            plan.id,
            status=PlanStatus.COMPLETED,
            result_summary=summary,
        )

        # Feed plan result through memory pipeline
        if self.processor and summary:
            try:
                await self.processor.process_message(
                    text=summary,
                    role="assistant",
                    session_id=plan.session_id or 0,
                )
            except Exception as e:
                logger.error("Plan result memory save failed: %s", e)

        # Detach cache from ReadFileTool on success exit
        if read_tool is not None:
            read_tool.file_cache = None

        return summary

    async def execute_dynamic(
        self,
        user_request: str,
        on_step_start: Optional[Callable[[int], None]] = None,
        on_step_complete: Optional[Callable[[int, str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_display: Optional[Callable] = None,
        max_tool_calls: int = 0,
        memory_context: str = "",
        chat_history: list[dict] | None = None,
        log_event: Optional[Callable] = None,
        capability_context: str = "",
    ) -> str:
        """Execute a task dynamically — single continuous conversation.

        Uses ChatLoop for the tool-calling loop. The LLM sees full conversation
        history. Stops when task_complete is called or budget is hit.

        Args:
            max_tool_calls: Tool call budget. 0 = use self.max_tool_iterations.
            memory_context: Relevant memories from past sessions (injected as system message).
            chat_history: Recent chat messages for design discussion context.
            log_event: Optional async callback(event_type, data) for flow observability.
        """
        # Ensure task_complete is available (may not be registered outside project mode)
        if not self.tool_registry.get_tool("task_complete"):
            from blipshell.core.tools.interaction_tools import TaskCompleteTool
            self.tool_registry.register(TaskCompleteTool(), group="general")

        # Set up guardrails engine if configured
        guardrails_engine = None
        if self.guardrails_config and self.guardrails_config.enabled:
            from blipshell.core.guardrails import GuardrailsEngine
            guardrails_engine = GuardrailsEngine(self.guardrails_config, self.router)
            guardrails_engine.original_request = user_request

            # Register confirm_plan tool if requirement checklist is enabled
            if self.guardrails_config.requirement_checklist:
                if not self.tool_registry.get_tool("confirm_plan"):
                    from blipshell.core.tools.interaction_tools import ConfirmPlanTool
                    # Reuse ask_user's callback for confirm_plan (same UX pattern)
                    ask_user_tool = self.tool_registry.get_tool("ask_user")
                    cb = ask_user_tool.callback if ask_user_tool else None
                    self.tool_registry.register(
                        ConfirmPlanTool(callback=cb, guardrails_engine=guardrails_engine),
                        group="general",
                    )

        # Wire file cache AND files_read into ReadFileTool so it can detect
        # re-reads and serve cached content instead of re-reading from disk.
        self._file_cache.clear()
        self.files_read.clear()
        self._stale_files.clear()
        read_tool = self.tool_registry.get_tool("read_file")
        if read_tool is not None:
            read_tool.file_cache = self._file_cache
            read_tool.files_read = self.files_read
            read_tool.stale_files = self._stale_files

        # Build system prompt — use executor-specific prompt with rules
        sys_prompt = executor_system_prompt()
        if self.active_project and self.project_context:
            sys_prompt += "\n\n" + self.project_context

        # Guardrails: add context pinning and checklist guidance to system prompt
        if guardrails_engine:
            guardrails_prompt = "\n\n# Guardrails\n"
            if self.guardrails_config.requirement_checklist:
                guardrails_prompt += (
                    "- For complex tasks (3+ files, ambiguous requirements), "
                    "call confirm_plan FIRST to show your plan and get approval.\n"
                )
            if self.guardrails_config.completion_audit:
                guardrails_prompt += (
                    "- Your task_complete will be validated against the original request. "
                    "Make sure you address every requirement before calling it.\n"
                )
            if self.guardrails_config.context_pinning:
                pinned = guardrails_engine.pinned_context
                if pinned:
                    guardrails_prompt += f"\n{pinned}\n"
            sys_prompt += guardrails_prompt

        # Look-before-review: ground review/critique requests in a real read/grep
        # before findings are stated. Guidance fires on the review_grounding flag
        # regardless of whether the full guardrails engine is enabled.
        if (self.guardrails_config and self.guardrails_config.review_grounding
                and detect_review_intent(user_request)):
            sys_prompt += REVIEW_GROUNDING_GUIDANCE

        tools = self.tool_registry.get_all_ollama_tools() or None

        # Tool call budget — bump for project mode
        budget = max_tool_calls or self.max_tool_iterations
        if self.active_project:
            budget = max(budget, 50)

        # Build initial messages — single system message (CC approach)
        task_prompt = dynamic_execution_prompt(user_request)
        # Derived capability block (e.g. vision availability) — kept in sync with
        # what's actually true this turn rather than a hand-written claim. Built by
        # the caller (agent_chat._build_capability_block) so derivation lives in one place.
        if capability_context:
            sys_prompt += f"\n\n{capability_context}"
        if memory_context:
            sys_prompt += f"\n\n--- RELEVANT MEMORIES ---\n{memory_context}"
        messages = [
            {"role": "system", "content": sys_prompt},
        ]

        # Inject recent chat history so executor has design discussion context
        if chat_history:
            messages.extend(chat_history)

        # The actual task instruction
        messages.append({"role": "user", "content": task_prompt})

        if on_step_start:
            on_step_start(1)

        # Get context limit from the endpoint that will handle this request
        task_type = "coding" if self.active_project else "tool_calling"
        try:
            ep = await self.router._endpoint_manager.get_endpoint_for_role(task_type)
            effective_context = ep.context_tokens if ep and ep.context_tokens else 65536
        except Exception:
            effective_context = 65536

        # Log executor context info for /flow observability
        if log_event:
            try:
                memory_count = memory_context.count("\n- ") if memory_context else 0
                msg_tokens = estimate_messages_tokens(messages)
                await log_event("context_built", {
                    "query_profile": "executor",
                    "context_limit": effective_context,
                    "available_tokens": effective_context - msg_tokens,
                    "total_context_items": memory_count + len(chat_history or []),
                    "pool_budgets": {"memory": memory_count, "chat_history": len(chat_history or [])},
                    "pool_usage": {
                        "memory": {"items": memory_count, "tokens": estimate_tokens(memory_context)},
                        "chat_history": {"items": len(chat_history or []), "tokens": sum(estimate_tokens(m.get("content", "") or "") for m in (chat_history or []))},
                    },
                })
            except Exception as e:
                logger.debug("Failed to log executor flow event: %s", e)

        # Dynamic tool provider — switches tools mid-loop when plan mode toggles
        def _get_current_tools():
            if self.tool_registry.in_plan_mode:
                return self.tool_registry.get_plan_mode_tools() or None
            return tools

        # Build executor-specific loop config (A/B overrides take precedence)
        compaction_cfg = self._compaction_config
        config = LoopConfig(
            budget=budget,
            enable_dedup=self._override_enable_dedup if self._override_enable_dedup is not None else True,
            enable_compaction=self._override_enable_compaction if self._override_enable_compaction is not None else True,
            compaction_threshold=compaction_cfg.compaction_threshold if compaction_cfg else 0.85,
            context_limit=effective_context,
            completion_tool="task_complete",
            capture_inline_text=True,
            tool_provider=_get_current_tools,
            on_pause_check=self.pause_check_callback,
            guardrails=guardrails_engine,
            on_tool_display=on_tool_display,
            compaction_config=compaction_cfg if (compaction_cfg and compaction_cfg.use_llm) else None,
            compaction_router=self.router if (compaction_cfg and compaction_cfg.use_llm) else None,
            compaction_files_read=self.files_read,
            compaction_file_cache=self._file_cache,
        )

        # Use shared chat loop runner (from Agent) if available — gives us
        # endpoint fallback, gate setup, and model selection in one place.
        # Falls back to direct execution if runner not wired (e.g. in tests).
        if self.chat_loop_runner:
            result, endpoint_name, model, using_fallback = await self.chat_loop_runner(
                messages=messages,
                config=config,
                on_token=on_token,
                on_tool_executed=self._on_tool_executed,
            )
            if result is None:
                self.last_messages = messages  # Preserve messages for narrative even on failure
                raise RuntimeError("No available LLM endpoint")
        else:
            # Legacy direct path (kept for backwards compatibility with tests)
            task_type = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
            endpoint = await self.router._endpoint_manager.get_endpoint_for_role(task_type)
            if not endpoint:
                raise RuntimeError("No available LLM endpoint")
            model = endpoint.models.get(task_type) or self.router.get_model(task_type)
            endpoint_name = endpoint.name

            chat_kwargs: dict = {}
            if endpoint.context_tokens:
                chat_kwargs["options"] = {"num_ctx": endpoint.context_tokens}
                config.context_limit = endpoint.context_tokens

            if endpoint.provider == "ollama":
                from blipshell.llm.ollama_gate import INTERACTIVE, get_gate
                config.ollama_gate = get_gate()
                config.gate_priority = INTERACTIVE

            loop = ChatLoop(self.tool_registry, on_token)
            endpoint.start_request()
            try:
                result = await loop.run(
                    client=endpoint.client,
                    messages=messages,
                    model=model,
                    tools=tools,
                    chat_kwargs=chat_kwargs,
                    config=config,
                    on_tool_executed=self._on_tool_executed,
                )
                endpoint.record_success(0)
            finally:
                endpoint.complete_request()
            using_fallback = False

        final_response = result.response

        # Store messages for narrative building
        self.last_messages = result.messages

        # Log executor completion for /flow observability
        if log_event:
            try:
                await log_event("llm_complete", {
                    "endpoint": endpoint_name,
                    "model": model,
                    "fallback": using_fallback,
                    "tool_calls": result.tool_call_names,
                    "response_length": len(final_response or ""),
                    "total_tool_calls": result.tool_call_count,
                    "budget": budget,
                    "files_read": len(self.files_read),
                    "files_created": len(self._step_files_created),
                    "files_edited": len(self._step_files_edited),
                })
            except Exception as e:
                logger.debug("Failed to log executor completion event: %s", e)

        # Save transcript for reference
        if self.active_project:
            try:
                from datetime import datetime
                from pathlib import Path
                transcript_dir = Path("data/project_transcripts")
                transcript_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                project_name = self.active_project.get("name", "unknown")
                filename = f"{project_name}__{timestamp}.json"
                transcript_path = transcript_dir / filename
                transcript_path.write_text(
                    json.dumps(result.messages, indent=2, default=str),
                    encoding="utf-8",
                )
                logger.info("Saved coding transcript to %s", transcript_path)
            except Exception as e:
                logger.error("Failed to save transcript: %s", e)

        # Detach cache
        if read_tool is not None:
            read_tool.file_cache = None

        if on_step_complete:
            on_step_complete(1, (final_response or "")[:200])

        if not final_response or not final_response.strip():
            final_response = "Task completed."

        return final_response

    async def _execute_step(
        self,
        plan: TaskPlan,
        step_number: int,
        step_description: str,
        total_steps: int,
        completed_summaries: list[str],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Execute a single step using the unified ChatLoop."""
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

        # Build system prompt with project context if available
        sys_prompt = self.system_prompt
        if self.active_project and self.project_context:
            sys_prompt += "\n\n" + self.project_context

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": step_prompt},
        ]

        # Route to coding model when project is active
        task_type = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        endpoint = await self.router._endpoint_manager.get_endpoint_for_role(task_type)
        if not endpoint:
            raise RuntimeError("No available LLM endpoint")
        model = endpoint.models.get(task_type) or self.router.get_model(task_type)
        client = endpoint.client

        # Pass context window size to Ollama
        chat_kwargs: dict = {}
        if endpoint.context_tokens:
            chat_kwargs["options"] = {"num_ctx": endpoint.context_tokens}

        tools = self.tool_registry.get_all_ollama_tools() or None
        max_iterations = self.max_tool_iterations if tools else 0
        if self.active_project and tools:
            max_iterations = max(max_iterations, 30)

        # Run the unified tool-calling loop
        loop = ChatLoop(self.tool_registry, on_token)
        config = LoopConfig(budget=max_iterations)
        # Gate local Ollama calls (cloud endpoints bypass)
        if endpoint.provider == "ollama":
            from blipshell.llm.ollama_gate import INTERACTIVE, get_gate
            config.ollama_gate = get_gate()
            config.gate_priority = INTERACTIVE
        result = await loop.run(
            client=client,
            messages=messages,
            model=model,
            tools=tools,
            chat_kwargs=chat_kwargs,
            config=config,
            on_tool_executed=self._on_tool_executed,
        )

        return result.response

    def _build_step_summary(self, description: str, result: str) -> str:
        """Build a rich context summary of a completed step for later steps.

        Instead of just 200 chars of the LLM response, includes:
        - What the step did (description)
        - Files created with their key content (class names, imports)
        - Files edited
        - Truncated LLM response for additional context
        """
        parts = [f"Step: {description}"]

        if self._step_files_created:
            parts.append("Files created:")
            for path in self._step_files_created:
                cached = self._file_cache.get(path, "")
                if cached:
                    # Extract key structural info: imports, class/function names
                    key_lines = self._extract_key_lines(cached)
                    parts.append(f"  - {path}")
                    if key_lines:
                        parts.append(f"    Key structure: {key_lines}")
                else:
                    parts.append(f"  - {path}")

        if self._step_files_edited:
            parts.append("Files edited: " + ", ".join(self._step_files_edited))

        # Include LLM's summary but more generous than 200 chars
        if result:
            parts.append(f"Result: {result[:500]}")

        return "\n".join(parts)

    @staticmethod
    def _extract_key_lines(content: str) -> str:
        """Extract key structural info from file content for step summaries.

        Pulls out imports, class definitions, and function signatures — enough
        for later steps to use correct names without re-reading the file.
        """
        key = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                key.append(stripped)
            elif stripped.startswith(("class ", "def ", "async def ")):
                key.append(stripped.rstrip(":"))
        # Cap to prevent bloating context
        if len(key) > 20:
            key = key[:20] + [f"... ({len(key) - 20} more)"]
        return "; ".join(key) if key else ""

    def _on_tool_executed(self, name: str, arguments: dict, result) -> None:
        """Callback for ChatLoop — tracks files read/created/edited and manages cache."""
        if result.success and name == "read_file":
            read_path = arguments.get("path", "")
            if read_path:
                self.files_read.add(read_path)
                self._stale_files.discard(read_path)  # just read — no longer stale
        if result.success and name == "write_file":
            file_path = arguments.get("path", "")
            if file_path:
                self._step_files_created.append(file_path)
                written = arguments.get("content", "")
                if written:
                    self._file_cache[file_path] = written
                # Mark stale if it was previously read (content changed)
                if file_path in self.files_read:
                    self._stale_files.add(file_path)
        if result.success and name == "edit_file":
            file_path = arguments.get("path", "")
            if file_path:
                self._step_files_edited.append(file_path)
                self._file_cache.pop(file_path, None)
                # Mark stale — cached/in-context content is now outdated
                if file_path in self.files_read:
                    self._stale_files.add(file_path)
        if result.success and name == "list_directory":
            read_path = arguments.get("path", "")
            if read_path:
                self.files_read.add(read_path)

    async def _generate_summary(
        self, user_request: str, step_results: list[str],
    ) -> str:
        """Generate a final summary from all step results.

        Uses the reasoning model (not summarization) since the summary
        needs to synthesize tool results — not just compress text.
        """
        task_type = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        prompt = summarize_plan_results(user_request, step_results)
        return await self.router.generate(
            task_type, prompt, system=UTILITY_SYSTEM_PROMPT,
        )

    def _build_state_block(
        self,
        tool_call_count: int,
        budget: int,
        tool_call_names: list[str],
    ) -> str:
        """Build a structured state block for injection before each LLM turn.

        Gives the model explicit awareness of what's been done, what it has,
        and how much budget remains — so it doesn't have to re-parse the
        full conversation history to figure out where it is.
        """
        parts = [f"[STATE] Tool calls: {tool_call_count}/{budget}"]

        # Files read (annotate stale files)
        if self.files_read:
            files = sorted(self.files_read)
            annotated = []
            for f in files:
                if f in self._stale_files:
                    annotated.append(f"{f} (STALE — modified, re-read before using)")
                else:
                    annotated.append(f)
            if len(annotated) > 10:
                shown = annotated[:10]
                parts.append(f"Files read ({len(files)}): {', '.join(shown)}, ... +{len(files) - 10} more")
            else:
                parts.append(f"Files read: {', '.join(annotated)}")

        # Files created/modified
        if self._step_files_created:
            parts.append(f"Files created: {', '.join(self._step_files_created)}")
        if self._step_files_edited:
            parts.append(f"Files edited: {', '.join(self._step_files_edited)}")

        # Recent actions (last 5 tool calls)
        if tool_call_names:
            recent = tool_call_names[-5:]
            parts.append(f"Recent actions: {' → '.join(recent)}")

        # Reminder
        parts.append("Do NOT re-read files listed above. When done, call task_complete.")

        return "\n".join(parts)
