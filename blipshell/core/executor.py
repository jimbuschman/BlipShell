"""Step-by-step task executor.

Loops through plan steps sequentially. For each step, builds a focused
prompt with accumulated context and runs the existing tool-calling loop.
"""

import json
import logging
from typing import Callable, Optional

from blipshell.core.tool_rules import ToolRuleEngine, create_coding_rules, create_default_rules
from blipshell.core.tools.base import ToolRegistry
from blipshell.llm.client import LLMClient
from blipshell.llm.prompts import dynamic_execution_prompt, executor_system_prompt, execute_step, summarize_plan_results, UTILITY_SYSTEM_PROMPT
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import PlannerConfig
from blipshell.models.task import PlanStatus, StepStatus, TaskPlan
from blipshell.models.tools import ToolCall

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from blipshell.memory.processor import MemoryProcessor

logger = logging.getLogger(__name__)


async def _stream_chat(
    client: LLMClient,
    messages: list[dict],
    model: str,
    tools: list[dict] | None,
    chat_kwargs: dict,
    on_token=None,
) -> tuple[str, list | None]:
    """Stream an LLM response, yielding text tokens to on_token as they arrive.

    Returns (content, tool_calls). Falls back to non-streaming on error.
    """
    try:
        content_parts: list[str] = []
        tool_calls = None

        async for chunk in client.chat_stream(
            messages=messages, model=model, tools=tools, **chat_kwargs,
        ):
            msg = getattr(chunk, "message", None)
            if msg is not None:
                chunk_content = getattr(msg, "content", "") or ""
                chunk_tool_calls = getattr(msg, "tool_calls", None)
            else:
                msg = chunk.get("message", {}) if isinstance(chunk, dict) else {}
                chunk_content = msg.get("content", "") or ""
                chunk_tool_calls = msg.get("tool_calls", None)

            if chunk_content:
                content_parts.append(chunk_content)
                if on_token:
                    on_token(chunk_content)

            if chunk_tool_calls:
                tool_calls = chunk_tool_calls

        return "".join(content_parts), tool_calls
    except Exception as e:
        logger.warning("Streaming failed in executor, falling back: %s", e)
        response = await client.chat(
            messages=messages, model=model, tools=tools, **chat_kwargs,
        )
        msg = getattr(response, "message", None)
        if msg is not None:
            content = getattr(msg, "content", "") or ""
            tool_calls = getattr(msg, "tool_calls", None)
        elif isinstance(response, dict):
            msg_d = response.get("message", {})
            content = msg_d.get("content", "")
            tool_calls = msg_d.get("tool_calls", None)
        else:
            content, tool_calls = "", None
        if on_token and content:
            on_token(content)
        return content, tool_calls


def _estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens in a message list (len/4 heuristic)."""
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total += len(content) // 4
        # Tool calls in assistant messages add tokens too
        if "tool_calls" in msg:
            try:
                total += len(json.dumps(msg["tool_calls"], default=str)) // 4
            except (TypeError, ValueError):
                # Fallback: estimate from string representation
                total += len(str(msg["tool_calls"])) // 4
    return total


def _compact_messages(messages: list[dict], keep_last_n: int = 5) -> list[dict]:
    """Compact older tool results in a message list to reduce context size.

    Keeps:
    - All system messages (prompt, memory context)
    - The first user message (task instruction)
    - The last N tool call/result pairs in full
    - Assistant reasoning text in full (planning, explanations)

    Compresses:
    - Older tool result messages → one-line summary
    """
    # Find boundaries: system messages, first user message, then conversation
    prefix = []  # system + first user message
    conversation = []  # everything after
    found_user = False
    for msg in messages:
        if not found_user:
            prefix.append(msg)
            if msg.get("role") == "user":
                found_user = True
        else:
            conversation.append(msg)

    if not conversation:
        return messages

    # Count tool result messages from the end to find the keep boundary
    tool_results_from_end = 0
    keep_from_idx = len(conversation)
    for i in range(len(conversation) - 1, -1, -1):
        msg = conversation[i]
        if msg.get("role") == "tool":
            tool_results_from_end += 1
            if tool_results_from_end >= keep_last_n:
                keep_from_idx = i
                break

    # Everything before keep_from_idx gets compacted
    compacted = []
    for msg in conversation[:keep_from_idx]:
        role = msg.get("role")
        if role == "tool":
            # Compress tool results to a one-liner
            content = msg.get("content", "")
            # Estimate original size for the summary
            orig_len = len(content)
            # Take first 100 chars as preview
            preview = content[:100].replace("\n", " ").strip()
            if orig_len > 100:
                preview += "..."
            compacted.append({
                "role": "tool",
                "content": f"[Compacted — {orig_len} chars] {preview}",
                **({"tool_call_id": msg["tool_call_id"]} if "tool_call_id" in msg else {}),
            })
        else:
            # Keep assistant messages (reasoning text) and system messages as-is
            compacted.append(msg)

    # Reassemble: prefix + compacted older + recent full messages
    return prefix + compacted + conversation[keep_from_idx:]


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
        self._tool_rules: ToolRuleEngine = create_default_rules()
        # Per-step tracking for rich context summaries
        self._step_files_created: list[str] = []
        self._step_files_edited: list[str] = []
        # Last execute_dynamic messages — available for narrative building after execution
        self.last_messages: list[dict] = []
        # Phase 2 test override flags (set by TestOverrides.apply())
        self._disable_winddown: bool = False
        self._disable_state_block: bool = False
        self._natural_completion_primary: bool = False

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

        # Switch to coding rules if project is active
        if self.active_project:
            self._tool_rules = create_coding_rules()

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
        max_tool_calls: int = 0,
        memory_context: str = "",
        chat_history: list[dict] | None = None,
        log_event: Optional[Callable] = None,
    ) -> str:
        """Execute a task dynamically — single continuous conversation.

        One long tool-calling loop where the LLM sees the full conversation
        history (all tool calls and results). Stops when the LLM signals
        TASK_COMPLETE or hits the tool call budget.

        Args:
            max_tool_calls: Tool call budget. 0 = use self.max_tool_iterations.
            memory_context: Relevant memories from past sessions (injected as system message).
            chat_history: Recent chat messages for design discussion context.
            log_event: Optional async callback(event_type, data) for flow observability.
        """
        # Switch to coding rules if project is active
        if self.active_project:
            self._tool_rules = create_coding_rules()

        # Ensure task_complete is available (may not be registered outside project mode)
        if not self.tool_registry.get_tool("task_complete"):
            from blipshell.core.tools.interaction_tools import TaskCompleteTool
            self.tool_registry.register(TaskCompleteTool(), group="general")

        # Wire file cache AND files_read into ReadFileTool so it can detect
        # re-reads and serve cached content instead of re-reading from disk.
        # Without this, the executor's files_read set and the tool's set are
        # disconnected — the tool never sees what the executor already read.
        self._file_cache.clear()
        self.files_read.clear()
        read_tool = self.tool_registry.get_tool("read_file")
        if read_tool is not None:
            read_tool.file_cache = self._file_cache
            read_tool.files_read = self.files_read

        # Build system prompt — use executor-specific prompt with rules,
        # not the generic agent system prompt (which is for chat)
        sys_prompt = executor_system_prompt()
        if self.active_project and self.project_context:
            sys_prompt += "\n\n" + self.project_context

        # Route to coding model when project is active
        task_type = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        endpoint = await self.router._endpoint_manager.get_endpoint_for_role(task_type)
        if not endpoint:
            raise RuntimeError("No available LLM endpoint")
        model = endpoint.models.get(task_type) or self.router.get_model(task_type)
        client = endpoint.client

        chat_kwargs: dict = {}
        if endpoint.context_tokens:
            chat_kwargs["options"] = {"num_ctx": endpoint.context_tokens}

        tools = self.tool_registry.get_all_ollama_tools() or None

        # Tool call budget — bump for project mode
        budget = max_tool_calls or self.max_tool_iterations
        if self.active_project:
            budget = max(budget, 50)

        # Build initial messages — one continuous conversation
        task_prompt = dynamic_execution_prompt(user_request)
        messages = [
            {"role": "system", "content": sys_prompt},
        ]

        # Inject memory context so executor has relevant past knowledge
        if memory_context:
            messages.append({"role": "system", "content": memory_context})

        # Inject recent chat history so executor has design discussion context
        if chat_history:
            messages.extend(chat_history)

        # The actual task instruction
        messages.append({"role": "user", "content": task_prompt})

        if on_step_start:
            on_step_start(1)

        # Log executor context info for /flow observability
        if log_event:
            try:
                memory_count = memory_context.count("\n- ") if memory_context else 0
                await log_event("context_built", {
                    "query_profile": "executor",
                    "context_limit": endpoint.context_tokens or 65536,
                    "available_tokens": (endpoint.context_tokens or 65536) - _estimate_messages_tokens(messages),
                    "total_context_items": memory_count + len(chat_history or []),
                    "pool_budgets": {"memory": memory_count, "chat_history": len(chat_history or [])},
                    "pool_usage": {
                        "memory": {"items": memory_count, "tokens": len(memory_context) // 4},
                        "chat_history": {"items": len(chat_history or []), "tokens": sum(len(m.get("content", "") or "") // 4 for m in (chat_history or []))},
                    },
                })
            except Exception:
                pass

        tool_call_count = 0
        tool_call_names: list[str] = []
        final_response = ""
        wind_down_injected = False
        self._force_complete_injected = False
        last_tool_call: tuple[str, str] = ("", "")  # (name, args_key) for dedup
        last_state_msg_idx: int = -1  # track state message for replacement

        # Single continuous loop — LLM keeps full conversation history
        max_rounds = budget + 10  # generous round limit (text-only responses don't cost tools)
        for _round in range(max_rounds):
            # Budget wind-down: inject guidance when ~80% of budget used
            if (not self._disable_winddown
                    and not wind_down_injected
                    and tool_call_count >= int(budget * 0.8)
                    and tool_call_count > 0):
                messages.append({
                    "role": "system",
                    "content": (
                        "You are running low on tool calls. Wrap up your current work "
                        "and call the task_complete tool with a summary of what you accomplished."
                    ),
                })
                wind_down_injected = True
                if on_token:
                    on_token("  [Budget warning injected]\n")

            # Context compaction: if messages exceed 70% of context window,
            # compress older tool results to prevent context overflow.
            context_limit = endpoint.context_tokens or 65536
            est_tokens = _estimate_messages_tokens(messages)
            if est_tokens > int(context_limit * 0.7):
                before = est_tokens
                messages = _compact_messages(messages, keep_last_n=5)
                after = _estimate_messages_tokens(messages)
                logger.info("Context compacted: %d → %d tokens (saved %d)",
                            before, after, before - after)
                if on_token:
                    on_token(f"  [Context compacted: {before} → {after} tokens]\n")

            # Forced completion at 95%: strip all tools except task_complete
            force_complete = (not self._disable_winddown
                              and tool_call_count >= int(budget * 0.95)
                              and tool_call_count < budget)
            if force_complete and not getattr(self, '_force_complete_injected', False):
                messages.append({
                    "role": "system",
                    "content": (
                        "You have almost no tool calls left. You MUST call task_complete "
                        "NOW with a summary of what you accomplished and what remains."
                    ),
                })
                self._force_complete_injected = True
                if on_token:
                    on_token("  [Forced completion — only task_complete available]\n")

            # Stop offering tools once budget is exhausted
            iter_tools = None
            if tools and tool_call_count < budget:
                if force_complete:
                    # Only offer task_complete
                    iter_tools = [t for t in tools if t.get("function", {}).get("name") == "task_complete"]
                else:
                    iter_tools = self._tool_rules.filter_tools(tools, tool_call_names)
                if not iter_tools:
                    iter_tools = None

            # State injection: give model explicit awareness of progress.
            # Replace previous state message (if any) to avoid accumulation.
            if not self._disable_state_block:
                state_block = self._build_state_block(
                    tool_call_count, budget, tool_call_names,
                )
                if last_state_msg_idx >= 0 and last_state_msg_idx < len(messages):
                    messages[last_state_msg_idx] = {
                        "role": "system", "content": state_block,
                    }
                else:
                    messages.append({"role": "system", "content": state_block})
                    last_state_msg_idx = len(messages) - 1

            content, tool_calls = await _stream_chat(
                client, messages, model, iter_tools, chat_kwargs, on_token,
            )

            if tool_calls and tool_call_count < budget:
                # Check if task_complete is among the tool calls
                task_complete_result = None
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })

                for tc in tool_calls:
                    name, arguments, tc_id = self._extract_tool_call_info(tc)
                    tool_call_names.append(name)
                    tool_call_count += 1
                    tool_call = ToolCall(id=tc_id, name=name, arguments=arguments)

                    if on_token:
                        on_token(f"  [Tool: {tool_call.name}]\n")

                    # Same-args dedup: if identical tool+args as last call, redirect
                    args_key = json.dumps(arguments, sort_keys=True, default=str)
                    current_call = (name, args_key)
                    if current_call == last_tool_call and name != "task_complete":
                        from blipshell.models.tools import ToolResult
                        result = ToolResult(
                            tool_call_id=tc_id,
                            name=name,
                            result=(
                                f"You just called {name} with the same arguments. "
                                "If the task is done, call task_complete. "
                                "If not, try a different approach."
                            ),
                            success=False,
                        )
                        if on_token:
                            on_token(f"  [Duplicate call blocked]\n")
                    else:
                        result = await self.tool_registry.execute_tool_call(tool_call)
                    result.tool_call_id = tc_id
                    last_tool_call = current_call

                    # Check for task_complete tool — this is the primary completion signal
                    if name == "task_complete":
                        task_complete_result = result.result
                        if on_token:
                            on_token(f"  [Task complete signal received]\n")

                    # Cache tracking — NOTE: do NOT cache result.result for read_file
                    # because it contains paginated output (line numbers, footer text).
                    # The ReadFileTool caches raw content internally via file_cache.
                    if result.success and name == "read_file":
                        read_path = arguments.get("path", "")
                        if read_path:
                            self.files_read.add(read_path)
                    if result.success and name == "write_file":
                        file_path = arguments.get("path", "")
                        if file_path:
                            self._step_files_created.append(file_path)
                            written = arguments.get("content", "")
                            if written:
                                self._file_cache[file_path] = written
                    if result.success and name == "edit_file":
                        file_path = arguments.get("path", "")
                        if file_path:
                            self._step_files_edited.append(file_path)
                            self._file_cache.pop(file_path, None)
                    if result.success and name == "list_directory":
                        read_path = arguments.get("path", "")
                        if read_path:
                            self.files_read.add(read_path)

                    messages.append(result.to_ollama_message())

                    if on_token:
                        on_token(f"  [Result: {result.result[:150]}]\n")

                # If task_complete was called, we're done
                if task_complete_result is not None:
                    final_response = task_complete_result
                    break

                continue

            # Text-only response (no tool calls) — model is done naturally
            # This is the Claude Code / Codex CLI pattern: no tool calls = done.
            if content:
                final_response = content
                if on_token:
                    on_token(f"  [No tool calls — treating as complete]\n")
                break

            # Empty response — model is stuck
            break
        else:
            final_response = "Task reached maximum tool call budget."

        # Store messages for narrative building
        self.last_messages = messages

        # Log executor completion for /flow observability
        if log_event:
            try:
                await log_event("llm_complete", {
                    "endpoint": endpoint.name,
                    "model": model,
                    "fallback": False,
                    "tool_calls": tool_call_names,
                    "response_length": len(final_response or ""),
                    "total_tool_calls": tool_call_count,
                    "budget": budget,
                    "files_read": len(self.files_read),
                    "files_created": len(self._step_files_created),
                    "files_edited": len(self._step_files_edited),
                })
            except Exception:
                pass

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
                    json.dumps(messages, indent=2, default=str),
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

        # Build system prompt with project context if available
        sys_prompt = self.system_prompt
        if self.active_project and self.project_context:
            sys_prompt += "\n\n" + self.project_context

        messages = [
            {"role": "system", "content": sys_prompt},
        ]

        messages.append({"role": "user", "content": step_prompt})

        # Route to coding model when project is active
        task_type = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        endpoint = await self.router._endpoint_manager.get_endpoint_for_role(task_type)
        if not endpoint:
            raise RuntimeError("No available LLM endpoint")
        model = endpoint.models.get(task_type) or self.router.get_model(task_type)
        client = endpoint.client

        # Pass context window size to Ollama (critical — without this, default is ~2K-4K)
        chat_kwargs: dict = {}
        if endpoint.context_tokens:
            chat_kwargs["options"] = {"num_ctx": endpoint.context_tokens}

        tools = self.tool_registry.get_all_ollama_tools() or None
        max_iterations = self.max_tool_iterations if tools else 0
        # Bump iteration limit for project mode
        if self.active_project and tools:
            max_iterations = max(max_iterations, 30)

        # Tool-calling loop with tool rules and file caching
        full_response = ""
        tool_call_names: list[str] = []
        for iteration in range(max_iterations + 1):
            # Apply tool rules to filter available tools
            iter_tools = None
            if tools and iteration < max_iterations:
                iter_tools = self._tool_rules.filter_tools(tools, tool_call_names)
                if not iter_tools:
                    iter_tools = None

            content, tool_calls = await _stream_chat(
                client, messages, model, iter_tools, chat_kwargs, on_token,
            )

            if tool_calls and iteration < max_iterations:
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })

                for tc in tool_calls:
                    name, arguments, tc_id = self._extract_tool_call_info(tc)
                    tool_call_names.append(name)
                    tool_call = ToolCall(id=tc_id, name=name, arguments=arguments)

                    if on_token:
                        on_token(f"\n  [Tool: {tool_call.name}]\n")

                    result = await self.tool_registry.execute_tool_call(tool_call)
                    result.tool_call_id = tc_id

                    # Track file reads — raw content is cached by ReadFileTool itself
                    if result.success and name == "read_file":
                        read_path = arguments.get("path", "")
                        if read_path:
                            self.files_read.add(read_path)

                    # Track file operations for rich step summaries
                    if result.success and name == "write_file":
                        file_path = arguments.get("path", "")
                        if file_path:
                            self._step_files_created.append(file_path)
                            # Cache the written content too
                            written = arguments.get("content", "")
                            if written:
                                self._file_cache[file_path] = written
                    if result.success and name == "edit_file":
                        file_path = arguments.get("path", "")
                        if file_path:
                            self._step_files_edited.append(file_path)
                            # Invalidate cache — file changed, next read gets fresh
                            self._file_cache.pop(file_path, None)

                    if result.success and name == "list_directory":
                        read_path = arguments.get("path", "")
                        if read_path:
                            self.files_read.add(read_path)

                    messages.append(result.to_ollama_message())

                    if on_token:
                        on_token(f"  [Result: {result.result[:150]}]\n")

                continue
            else:
                full_response = content
                break

        return full_response

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

        # Files read
        if self.files_read:
            files = sorted(self.files_read)
            if len(files) > 10:
                shown = files[:10]
                parts.append(f"Files read ({len(files)}): {', '.join(shown)}, ... +{len(files) - 10} more")
            else:
                parts.append(f"Files read: {', '.join(files)}")

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
    def _extract_tool_call_info(tc) -> tuple[str, dict, str]:
        """Extract name, arguments, and id from a tool call.

        Handles both Ollama (args as dict) and OpenAI-compatible APIs
        (args as JSON string). Returns (name, arguments, tool_call_id).
        """
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", "") or ""
            args = getattr(fn, "arguments", {}) or {}
            tc_id = getattr(tc, "id", "") or ""
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            return name, args, tc_id

        if isinstance(tc, dict):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            tc_id = tc.get("id", "")
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            return fn.get("name", ""), args, tc_id

        return "", {}, ""
