"""Unified LLM tool-calling loop.

Shared by Agent._chat_simple() and TaskExecutor.execute_dynamic().
Callers configure behavior via LoopConfig; results returned via LoopResult.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from blipshell.models.tools import ToolCall, ToolResult

if TYPE_CHECKING:
    from blipshell.core.tools.base import ToolRegistry
    from blipshell.llm.client import LLMClient

logger = logging.getLogger(__name__)


# ── Pause / Redirect ─────────────────────────────────────────────────────────


class PauseAction(enum.Enum):
    """Result from a pause check callback."""
    CONTINUE = "continue"
    REDIRECT = "redirect"
    STOP = "stop"


@dataclass
class PauseResult:
    """Details from a pause check."""
    action: PauseAction = PauseAction.CONTINUE
    message: str = ""  # redirect instructions from the user


# ── Configuration & Result ───────────────────────────────────────────────────


@dataclass
class LoopConfig:
    """Configuration for a single ChatLoop.run() invocation."""

    budget: int = 50
    """Maximum number of tool calls before stopping."""

    enable_dedup: bool = True
    """Block consecutive identical tool calls (same name + same args)."""

    enable_compaction: bool = False
    """Compact older tool results when context window fills up."""

    compaction_threshold: float = 0.85
    """Trigger compaction at this fraction of context_limit."""

    context_limit: int = 65536
    """Total context window size in tokens (for compaction calculation)."""

    completion_tool: str | None = None
    """Tool name that signals task completion (e.g. 'task_complete'). None = text-only."""

    capture_inline_text: bool = False
    """When True, capture substantial text (>=200 chars) returned alongside tool calls.
    Used as a fallback response if the model never calls the completion_tool and
    produces nothing on subsequent turns."""

    auto_continue_on_exhaustion: bool = False
    """When budget is hit with no text response, nudge model to summarize."""

    tool_provider: Optional[Callable[[], list[dict] | None]] = None
    """If set, called each iteration to get current tools (for plan mode).
    When None, the static ``tools`` passed to run() is used."""

    enable_parallel: bool = True
    """Execute multiple tool calls from the same LLM response concurrently."""

    max_parallel: int = 8
    """Maximum number of concurrent tool executions per batch."""

    on_pause_check: Optional[Callable[[], "asyncio.coroutine"]] = None
    """Async callback checked between tool batches. Returns PauseResult.
    If None, no pause checking occurs."""

    on_tool_display: Optional[Callable] = None
    """Callback for structured tool display: (calls, results) -> None.
    calls: list of (name, args) tuples for the batch.
    results: list of ToolResult objects (same order as calls).
    When set, Phase 4/7 tool display via on_token is suppressed."""

    guardrails: object | None = None
    """GuardrailsEngine instance for instruction adherence checks.
    None = no guardrails (default). Set by executor when guardrails are enabled."""

    ollama_gate: object | None = None
    """OllamaGate instance for serializing local Ollama calls. None = no gating."""

    gate_priority: int = 0
    """Priority for gate acquisition (0=INTERACTIVE, 2=BACKGROUND)."""


@dataclass
class LoopResult:
    """Result of a ChatLoop.run() invocation."""

    response: str = ""
    """Final text response from the LLM."""

    messages: list[dict] = field(default_factory=list)
    """Full conversation history (for narrative building, transcript saving)."""

    tool_call_names: list[str] = field(default_factory=list)
    """Ordered list of tool names called during the loop."""

    tool_call_count: int = 0
    """Total number of tool calls executed."""

    completion_method: str = "empty"
    """How the loop ended: 'text' | 'tool' | 'budget' | 'nudge' | 'empty'."""


# ── Utility functions (moved from executor.py) ───────────────────────────────


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens in a message list.

    Uses tiktoken via estimate_tokens() when available, falls back to len//4.
    """
    from blipshell.memory.manager import estimate_tokens

    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total += estimate_tokens(content)
        if "tool_calls" in msg:
            try:
                total += estimate_tokens(json.dumps(msg["tool_calls"], default=str))
            except (TypeError, ValueError):
                total += estimate_tokens(str(msg["tool_calls"]))
    return total


def compact_messages(messages: list[dict], keep_last_n: int = 5) -> list[dict]:
    """Compact older tool results in a message list to reduce context size.

    Keeps:
    - All system messages (prompt, memory context)
    - The first user message (task instruction)
    - The last N tool call/result pairs in full
    - Assistant reasoning text in full (planning, explanations)

    Compresses (per-tool-type strategy):
    - read_file → dropped entirely (re-readable from disk)
    - grep_files/glob_files → collapsed to file list
    - shell/run_command → first + last lines with exit hint
    - edit_file/write_file → kept short (already concise)
    - other → generic preview
    """
    prefix = []
    conversation = []
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

    # Build tool_call_id → tool_name mapping from assistant messages
    tc_id_to_name: dict[str, str] = {}
    for msg in conversation:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                try:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else tc
                    tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                    if tc_id and name:
                        tc_id_to_name[tc_id] = name
                except (AttributeError, TypeError):
                    pass

    tool_results_from_end = 0
    keep_from_idx = len(conversation)
    for i in range(len(conversation) - 1, -1, -1):
        msg = conversation[i]
        if msg.get("role") == "tool":
            tool_results_from_end += 1
            if tool_results_from_end >= keep_last_n:
                keep_from_idx = i
                break

    compacted = []
    for msg in conversation[:keep_from_idx]:
        role = msg.get("role")
        if role == "tool":
            compacted.append(_compact_tool_result(msg, tc_id_to_name))
        else:
            compacted.append(msg)

    return prefix + compacted + conversation[keep_from_idx:]


def _compact_tool_result(msg: dict, tc_id_to_name: dict[str, str]) -> dict:
    """Compact a single tool result message using a type-specific strategy."""
    content = msg.get("content", "")
    orig_len = len(content)
    tc_id = msg.get("tool_call_id", "")
    tool_name = tc_id_to_name.get(tc_id, "")

    if tool_name in ("read_file",):
        # File reads are re-readable — drop content entirely
        lines = content.split("\n")
        summary = f"[Compacted] read_file ({len(lines)} lines) — re-read from disk if needed"
    elif tool_name in ("grep_files", "glob_files"):
        # Collapse search results to just file paths
        lines = content.strip().split("\n")
        files = set()
        for line in lines:
            # Extract file paths (grep: "path:line:content", glob: just paths)
            if ":" in line:
                files.add(line.split(":")[0].strip())
            elif line.strip():
                files.add(line.strip())
        file_list = sorted(files)
        if len(file_list) > 5:
            shown = ", ".join(file_list[:5])
            summary = f"[Compacted] {tool_name}: {len(lines)} hits in {len(file_list)} files — {shown}, +{len(file_list) - 5} more"
        else:
            summary = f"[Compacted] {tool_name}: {len(lines)} hits in {', '.join(file_list) or 'no matches'}"
    elif tool_name in ("run_command", "shell"):
        # Shell output: keep first and last lines + exit status hint
        lines = content.strip().split("\n")
        if len(lines) <= 3:
            summary = f"[Compacted] {tool_name}: {content.strip()}"
        else:
            first = lines[0][:100]
            last = lines[-1][:100]
            summary = f"[Compacted] {tool_name} ({len(lines)} lines): {first} ... {last}"
    elif tool_name in ("edit_file", "write_file"):
        # Edit/write results are already concise — short preview
        preview = content[:150].replace("\n", " ").strip()
        if orig_len > 150:
            preview += "..."
        summary = f"[Compacted] {tool_name}: {preview}"
    else:
        # Generic fallback
        preview = content[:100].replace("\n", " ").strip()
        if orig_len > 100:
            preview += "..."
        summary = f"[Compacted — {orig_len} chars] {tool_name + ': ' if tool_name else ''}{preview}"

    return {
        "role": "tool",
        "content": summary,
        **({"tool_call_id": tc_id} if tc_id else {}),
    }


def extract_tool_call_info(tc) -> tuple[str, dict, str]:
    """Extract name, arguments, and id from a tool call object or dict.

    Returns (name, arguments, tool_call_id). Handles both Ollama (args
    as dict) and OpenAI-compatible APIs (args as JSON string).
    """
    fn = getattr(tc, "function", None)
    if fn is not None:
        name = getattr(fn, "name", "") or ""
        args = getattr(fn, "arguments", {}) or {}
        tc_id = getattr(tc, "id", "") or ""
        if isinstance(args, str):
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
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        return fn.get("name", ""), args, tc_id

    return "", {}, ""


def format_tool_arg_hint(name: str, arguments: dict) -> str:
    """Format a short argument hint for tool call display."""
    if not arguments:
        return ""
    if "pattern" in arguments:
        return f" {arguments['pattern'][:50]}"
    if "path" in arguments:
        return f" {arguments['path']}"
    if "command" in arguments:
        return f" {arguments['command'][:60]}"
    if "query" in arguments:
        return f" {arguments['query'][:50]}"
    if "message" in arguments:
        return f" {arguments['message'][:60]}"
    if "paths" in arguments:
        return f" {arguments['paths'][:60]}"
    return ""


# ── Streaming ────────────────────────────────────────────────────────────────


async def stream_chat(
    client,
    messages: list[dict],
    model: str,
    tools: list[dict] | None,
    chat_kwargs: dict,
    on_token: Callable[[str], None] | None = None,
    on_stream_done: Callable | None = None,
) -> tuple[str, list | None]:
    """Stream an LLM response, yielding text tokens to on_token as they arrive.

    Returns (content, tool_calls). Falls back to non-streaming on error.
    If on_stream_done is provided, called with the final chunk for token tracking.
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

            # Token usage tracking — final chunk has done=True
            is_done = getattr(chunk, "done", False)
            if not is_done and isinstance(chunk, dict):
                is_done = chunk.get("done", False)
            if is_done and on_stream_done:
                on_stream_done(chunk)

        return "".join(content_parts), tool_calls
    except Exception as e:
        logger.warning("Streaming failed, falling back to non-streaming: %s", e)
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
        if on_stream_done:
            on_stream_done(response)
        return content, tool_calls


# ── ChatLoop ─────────────────────────────────────────────────────────────────


class ChatLoop:
    """Unified LLM tool-calling loop.

    Core loop shared by Agent._chat_simple() and TaskExecutor.execute_dynamic().
    Callers configure behavior via LoopConfig; results returned via LoopResult.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        on_token: Callable[[str], None] | None = None,
    ):
        self.tool_registry = tool_registry
        self.on_token = on_token

    def _partition_for_parallel(
        self,
        parsed_calls: list[tuple[str, dict, str]],
        config: LoopConfig,
    ) -> tuple[list[int], list[int]]:
        """Partition tool call indices into sequential and parallel groups.

        Sequential: tools requiring approval or ask_user (interactive prompts).
        Parallel: everything else (reads, greps, globs, web tools, etc.).

        Returns (sequential_indices, parallel_indices).
        """
        sequential: list[int] = []
        parallel: list[int] = []
        approval_set = self.tool_registry._tools_requiring_approval
        has_approval_cb = self.tool_registry._approval_callback is not None

        for i, (name, _args, _tc_id) in enumerate(parsed_calls):
            needs_sequential = (
                (has_approval_cb and name in approval_set)
                or name == "ask_user"
            )
            if needs_sequential:
                sequential.append(i)
            else:
                parallel.append(i)

        return sequential, parallel

    async def run(
        self,
        client: LLMClient,
        messages: list[dict],
        model: str,
        tools: list[dict] | None,
        chat_kwargs: dict,
        config: LoopConfig,
        on_tool_executed: Callable[[str, dict, ToolResult], None] | None = None,
        on_stream_done: Callable | None = None,
    ) -> LoopResult:
        """Execute the tool-calling loop.

        Args:
            client: LLM client to use for chat.
            messages: Pre-built message list (system + history + user). Modified in-place.
            model: Model name to use.
            tools: Ollama tool definitions, or None for no tools.
            chat_kwargs: Extra kwargs (e.g. {"options": {"num_ctx": ...}}).
            config: Loop behavior configuration.
            on_tool_executed: Called after each tool execution with (name, arguments, result).
            on_stream_done: Called with final streaming chunk for token tracking.

        Returns:
            LoopResult with response, messages, tool call info, and completion method.
        """
        tool_call_count = 0
        tool_call_names: list[str] = []
        last_tool_call: tuple[str, str] | None = None  # (name, args_key) for dedup
        last_inline_text = ""  # substantial text returned alongside tool calls (fallback)
        final_response = ""
        completion_method = "empty"

        max_rounds = config.budget + 10  # generous: text-only rounds don't cost tools
        for _round in range(max_rounds):
            # ── Context compaction ──
            if config.enable_compaction:
                est_tokens = estimate_messages_tokens(messages)
                if est_tokens > int(config.context_limit * config.compaction_threshold):
                    before = est_tokens
                    messages[:] = compact_messages(messages, keep_last_n=5)
                    after = estimate_messages_tokens(messages)
                    logger.info("Context compacted: %d → %d tokens (saved %d)",
                                before, after, before - after)
                    if self.on_token:
                        self.on_token(f"  [Context compacted: {before} → {after} tokens]\n")

            # ── Tool availability ──
            if config.tool_provider is not None:
                iter_tools = config.tool_provider() if tool_call_count < config.budget else None
            else:
                iter_tools = tools if (tools and tool_call_count < config.budget) else None

            # ── LLM call (gated for local Ollama) ──
            if config.ollama_gate:
                async with config.ollama_gate.async_gate(config.gate_priority):
                    content, tool_calls = await stream_chat(
                        client, messages, model, iter_tools, chat_kwargs,
                        self.on_token, on_stream_done,
                    )
            else:
                content, tool_calls = await stream_chat(
                    client, messages, model, iter_tools, chat_kwargs,
                    self.on_token, on_stream_done,
                )

            # ── Tool calls ──
            if tool_calls and tool_call_count < config.budget:
                completion_tool_result = None
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })

                # Phase 1: Parse all tool calls up front
                parsed_calls: list[tuple[str, dict, str]] = []
                for tc in tool_calls:
                    parsed_calls.append(extract_tool_call_info(tc))

                # Phase 2: Budget — trim to remaining budget, reserve slots
                remaining = config.budget - tool_call_count
                parsed_calls = parsed_calls[:remaining]
                tool_call_count += len(parsed_calls)
                for name, _, _ in parsed_calls:
                    tool_call_names.append(name)

                # Phase 3: Batch dedup — across previous batch + within this batch
                dedup_blocked: set[int] = set()
                batch_seen: set[tuple[str, str]] = set()
                if config.enable_dedup:
                    for i, (name, arguments, _) in enumerate(parsed_calls):
                        args_key = json.dumps(arguments, sort_keys=True, default=str)
                        call_key = (name, args_key)
                        if call_key == last_tool_call and name != config.completion_tool:
                            dedup_blocked.add(i)
                        elif call_key in batch_seen and name != config.completion_tool:
                            dedup_blocked.add(i)
                        else:
                            batch_seen.add(call_key)

                # Phase 4: Display tool call summary (compact, pre-execution)
                if parsed_calls and self.on_token and not config.on_tool_display:
                    # Legacy on_token display (when no structured callback)
                    active_calls = [
                        (name, args) for i, (name, args, _) in enumerate(parsed_calls)
                        if i not in dedup_blocked
                    ]
                    if len(active_calls) == 1:
                        name, args = active_calls[0]
                        hint = format_tool_arg_hint(name, args)
                        self.on_token(f"\n\x1b[2m  \u25b8 {name}{hint}\x1b[0m\n")
                    elif active_calls:
                        names = ", ".join(n for n, _ in active_calls)
                        self.on_token(f"\n\x1b[2m  \u25b8 Running {len(active_calls)} tools: {names}\x1b[0m\n")
                elif parsed_calls and config.on_tool_display:
                    # Structured display: show tool names immediately (before execution)
                    active_calls = [
                        (name, args) for i, (name, args, _) in enumerate(parsed_calls)
                        if i not in dedup_blocked
                    ]
                    if active_calls and self.on_token:
                        if len(active_calls) == 1:
                            name, args = active_calls[0]
                            hint = format_tool_arg_hint(name, args)
                            self.on_token(f"\n\x1b[2m  {name}{hint} …\x1b[0m")
                        else:
                            names = ", ".join(n for n, _ in active_calls)
                            self.on_token(f"\n\x1b[2m  {len(active_calls)} tools: {names} …\x1b[0m")

                # Phase 5: Partition into sequential (approval/ask_user) and parallel
                results: list[ToolResult | None] = [None] * len(parsed_calls)
                use_parallel = (
                    config.enable_parallel
                    and len(parsed_calls) > 1
                )

                if use_parallel:
                    seq_indices, par_indices = self._partition_for_parallel(
                        parsed_calls, config,
                    )
                else:
                    seq_indices = list(range(len(parsed_calls)))
                    par_indices = []

                # Phase 6a: Execute sequential tools (approval, ask_user)
                for i in seq_indices:
                    name, arguments, tc_id = parsed_calls[i]
                    if i in dedup_blocked:
                        results[i] = ToolResult(
                            tool_call_id=tc_id, name=name,
                            result=(
                                f"You just called {name} with the same arguments. "
                                "Try a different approach or complete the task."
                            ),
                            success=False,
                        )
                    else:
                        tc_obj = ToolCall(id=tc_id, name=name, arguments=arguments)
                        results[i] = await self.tool_registry.execute_tool_call(tc_obj)
                        results[i].tool_call_id = tc_id

                # Phase 6b: Execute parallel tools concurrently
                if par_indices:
                    semaphore = asyncio.Semaphore(config.max_parallel)

                    async def _run_one(idx: int) -> None:
                        pname, pargs, ptc_id = parsed_calls[idx]
                        if idx in dedup_blocked:
                            results[idx] = ToolResult(
                                tool_call_id=ptc_id, name=pname,
                                result=(
                                    f"You just called {pname} with the same arguments. "
                                    "Try a different approach or complete the task."
                                ),
                                success=False,
                            )
                            return
                        try:
                            async with semaphore:
                                tc_obj = ToolCall(id=ptc_id, name=pname, arguments=pargs)
                                results[idx] = await self.tool_registry.execute_tool_call(tc_obj)
                                results[idx].tool_call_id = ptc_id
                        except Exception as e:
                            logger.error("Parallel tool %s[%d] failed: %s", pname, idx, e)
                            results[idx] = ToolResult(
                                tool_call_id=ptc_id, name=pname,
                                result=f"Error executing {pname}: {e}",
                                success=False,
                            )

                    await asyncio.gather(*[_run_one(i) for i in par_indices])

                # Phase 7: Append results in order, run callbacks, display
                for i, (name, arguments, tc_id) in enumerate(parsed_calls):
                    result = results[i]

                    # Completion tool detection
                    if config.completion_tool and name == config.completion_tool:
                        completion_tool_result = result.result
                        if self.on_token and not config.on_tool_display:
                            self.on_token("  [Task complete signal received]\n")

                    # Caller tracking callback (sync or async)
                    if on_tool_executed:
                        ret = on_tool_executed(name, arguments, result)
                        if asyncio.iscoroutine(ret):
                            await ret

                    messages.append(result.to_ollama_message())

                    # Display result preview (compact, one line per tool)
                    if self.on_token and not config.on_tool_display:
                        if i in dedup_blocked:
                            self.on_token(f"\x1b[2m    \u2502 {name}: [duplicate blocked]\x1b[0m\n")
                        elif not result.success:
                            err = result.result[:100].replace("\n", " ")
                            self.on_token(f"\x1b[31m    \u2718 {name}: {err}\x1b[0m\n")
                        elif name == "edit_file" and "\x1b[" in result.result:
                            # Show first line (success msg) + colored diff
                            lines = result.result.split("\n", 1)
                            self.on_token(f"\x1b[2m    \u2502 {lines[0]}\x1b[0m\n")
                            if len(lines) > 1 and lines[1].strip():
                                self.on_token(lines[1] + "\n")
                        else:
                            preview = result.result[:80].replace("\n", " ")
                            self.on_token(f"\x1b[2m    \u2502 {name}: {preview}\x1b[0m\n")

                # Structured tool display callback (Rich rendering)
                if config.on_tool_display and parsed_calls:
                    # Clear the pre-execution "⏳ ..." line before showing results
                    if self.on_token:
                        self.on_token("\r\x1b[2K")
                    batch_calls = [
                        (name, args) for name, args, _ in parsed_calls
                    ]
                    batch_results = [
                        (results[i], i in dedup_blocked)
                        for i in range(len(parsed_calls))
                    ]
                    config.on_tool_display(batch_calls, batch_results)

                # Blank line after tool results block
                if self.on_token and parsed_calls and not config.on_tool_display:
                    self.on_token("\n")

                # Update last_tool_call for next batch dedup
                if parsed_calls:
                    last_name, last_args, _ = parsed_calls[-1]
                    last_tool_call = (
                        last_name,
                        json.dumps(last_args, sort_keys=True, default=str),
                    )

                # ── Doom-loop detection ──
                # Cheap counter check — no LLM cost
                if config.guardrails and hasattr(config.guardrails, 'check_doom_loop'):
                    try:
                        batch_calls_for_doom = [
                            (name, args) for name, args, _ in parsed_calls
                        ]
                        doom_warning = config.guardrails.check_doom_loop(batch_calls_for_doom)
                        if doom_warning:
                            messages.append({
                                "role": "user",
                                "content": doom_warning,
                            })
                            if self.on_token:
                                self.on_token(
                                    "\x1b[33m  [Doom-loop pattern detected]\x1b[0m\n"
                                )
                    except Exception as e:
                        logger.debug("Doom-loop check error: %s", e)

                # ── Critique: edit review ──
                # After tool results are in, check if any edits need critique
                if (config.guardrails and hasattr(config.guardrails, 'critique_edit')
                        and completion_tool_result is None):
                    for i, (name, arguments, tc_id) in enumerate(parsed_calls):
                        if name == "edit_file" and results[i] and results[i].success:
                            try:
                                critique = await config.guardrails.critique_edit(
                                    file_path=arguments.get("path", ""),
                                    old_text=arguments.get("old_text", ""),
                                    new_text=arguments.get("new_text", ""),
                                )
                                if critique:
                                    messages.append({
                                        "role": "user",
                                        "content": critique,
                                    })
                                    if self.on_token:
                                        self.on_token(
                                            f"\x1b[33m  [Critique: edit issue found]\x1b[0m\n"
                                        )
                            except Exception as e:
                                logger.debug("Edit critique error: %s", e)

                # ── Completion tool fired ──
                if completion_tool_result is not None:
                    # Guardrails: critique completion — richer quality review (runs first)
                    if config.guardrails and hasattr(config.guardrails, 'critique_completion'):
                        try:
                            tc_files = ""
                            for _i3, (n3, a3, _) in enumerate(parsed_calls):
                                if n3 == config.completion_tool:
                                    tc_files = a3.get("files_modified", "")
                                    break
                            critique = await config.guardrails.critique_completion(
                                summary=completion_tool_result,
                                files_modified=tc_files,
                                tool_call_names=tool_call_names,
                            )
                            if critique:
                                if self.on_token:
                                    self.on_token(
                                        f"\x1b[33m  [Critique: completion issue found]\x1b[0m\n"
                                    )
                                messages.append({
                                    "role": "user",
                                    "content": critique,
                                })
                                completion_tool_result = None
                                continue  # Model should fix and retry
                        except Exception as e:
                            logger.debug("Completion critique error: %s", e)

                    # Guardrails: completion audit — validate against original request
                    if config.guardrails and hasattr(config.guardrails, 'validate_completion'):
                        try:
                            # Extract files_modified from the task_complete args
                            tc_files = ""
                            for _i2, (n2, a2, _) in enumerate(parsed_calls):
                                if n2 == config.completion_tool:
                                    tc_files = a2.get("files_modified", "")
                                    break
                            valid, feedback = await config.guardrails.validate_completion(
                                completion_tool_result, tc_files,
                            )
                            if not valid:
                                if self.on_token:
                                    self.on_token(
                                        f"\x1b[33m  [Completion audit: rejected "
                                        f"({config.guardrails.audit_retries}/"
                                        f"{config.guardrails.config.max_audit_retries})]\x1b[0m\n"
                                    )
                                messages.append({
                                    "role": "user",
                                    "content": feedback,
                                })
                                completion_tool_result = None
                                continue  # Continue the loop — model should fix and retry
                        except Exception as e:
                            logger.warning("Completion audit error: %s — accepting", e)

                    final_response = completion_tool_result
                    completion_method = "tool"
                    break

                # ── Capture substantial inline text as fallback ──
                if config.capture_inline_text and content and len(content) >= 200:
                    last_inline_text = content

                # ── Pause check between tool batches ──
                if config.on_pause_check:
                    try:
                        pause_result = await config.on_pause_check()
                        if pause_result and pause_result.action == PauseAction.STOP:
                            if self.on_token:
                                self.on_token("\x1b[33m  [Stopped by user]\x1b[0m\n")
                            completion_method = "stopped"
                            break
                        elif pause_result and pause_result.action == PauseAction.REDIRECT:
                            if self.on_token:
                                self.on_token(f"\x1b[33m  [Redirected: {pause_result.message[:60]}]\x1b[0m\n")
                            messages.append({
                                "role": "user",
                                "content": pause_result.message,
                            })
                    except Exception as e:
                        logger.debug("Pause check error: %s", e)

                # ── Guardrails: trajectory critique (heavier, LLM call) ──
                if config.guardrails and hasattr(config.guardrails, 'critique_trajectory'):
                    try:
                        critique = await config.guardrails.critique_trajectory(
                            tool_call_count, config.budget, tool_call_names,
                        )
                        if critique:
                            messages.append({
                                "role": "user",
                                "content": critique,
                            })
                            if self.on_token:
                                self.on_token(
                                    "\x1b[33m  [Critique: approach concern raised]\x1b[0m\n"
                                )
                    except Exception as e:
                        logger.debug("Trajectory critique error: %s", e)

                # ── Guardrails: trajectory monitor injection ──
                if config.guardrails and hasattr(config.guardrails, 'build_trajectory_injection'):
                    try:
                        injection = config.guardrails.build_trajectory_injection(
                            tool_call_count, config.budget, tool_call_names,
                        )
                        if injection:
                            messages.append({
                                "role": "user",
                                "content": injection,
                            })
                            if self.on_token:
                                self.on_token(
                                    "\x1b[2m  [Trajectory checkpoint injected]\x1b[0m\n"
                                )
                    except Exception as e:
                        logger.debug("Trajectory injection error: %s", e)

                continue

            # ── Text-only response (no tool calls) — model is done ──
            if content:
                final_response = content
                completion_method = "text"
                break

            # ── Empty response — model is stuck ──
            break
        else:
            # Loop exhausted without breaking
            completion_method = "budget"

        # ── Auto-continue nudge on budget exhaustion ──
        if (config.auto_continue_on_exhaustion
                and not final_response
                and tool_call_names):
            if self.on_token:
                self.on_token("\n\x1b[2m[Continuing...]\x1b[0m\n")
            messages.append({
                "role": "user",
                "content": (
                    "You hit the tool call limit. Summarize what you've done so far "
                    "and what remains. Do NOT call any more tools — just respond."
                ),
            })
            try:
                # Gate this call too — same as the main loop LLM calls
                if config.ollama_gate:
                    async with config.ollama_gate.async_gate(config.gate_priority):
                        content, _ = await stream_chat(
                            client, messages, model, None, chat_kwargs,
                            self.on_token, on_stream_done,
                        )
                else:
                    content, _ = await stream_chat(
                        client, messages, model, None, chat_kwargs,
                        self.on_token, on_stream_done,
                    )
                final_response = content or f"[Hit tool limit after {len(tool_call_names)} calls]"
                completion_method = "nudge"
            except Exception as e:
                logger.error("Auto-continue failed: %s", e)
                final_response = f"[Hit tool limit after {len(tool_call_names)} calls]"
                completion_method = "nudge"

        # ── Budget/empty exhaustion fallback ──
        if not final_response:
            # Use captured inline text if available (model answered alongside tool calls)
            if last_inline_text:
                final_response = last_inline_text
                completion_method = "text"
                if self.on_token:
                    self.on_token("  [Inline text used as completion]\n")
                logger.info(
                    "Using inline text fallback (%d chars) — model answered "
                    "alongside tool calls without calling task_complete",
                    len(last_inline_text),
                )
            elif tool_call_names:
                final_response = f"[Completed {len(tool_call_names)} tool calls but no summary generated]"
                if completion_method == "empty":
                    completion_method = "budget"
            else:
                final_response = "No response generated."
                if completion_method == "empty":
                    completion_method = "budget"

        # Safety: reset plan mode if loop exits while still planning
        if self.tool_registry._plan_mode:
            self.tool_registry._plan_mode = False

        return LoopResult(
            response=final_response,
            messages=messages,
            tool_call_names=tool_call_names,
            tool_call_count=tool_call_count,
            completion_method=completion_method,
        )
