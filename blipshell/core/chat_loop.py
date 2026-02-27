"""Unified LLM tool-calling loop.

Shared by Agent._chat_simple() and TaskExecutor.execute_dynamic().
Callers configure behavior via LoopConfig; results returned via LoopResult.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from blipshell.models.tools import ToolCall, ToolResult

if TYPE_CHECKING:
    from blipshell.core.tools.base import ToolRegistry
    from blipshell.llm.client import LLMClient

logger = logging.getLogger(__name__)


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

    auto_continue_on_exhaustion: bool = False
    """When budget is hit with no text response, nudge model to summarize."""


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
    """Estimate total tokens in a message list (len/4 heuristic)."""
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total += len(content) // 4
        if "tool_calls" in msg:
            try:
                total += len(json.dumps(msg["tool_calls"], default=str)) // 4
            except (TypeError, ValueError):
                total += len(str(msg["tool_calls"])) // 4
    return total


def compact_messages(messages: list[dict], keep_last_n: int = 5) -> list[dict]:
    """Compact older tool results in a message list to reduce context size.

    Keeps:
    - All system messages (prompt, memory context)
    - The first user message (task instruction)
    - The last N tool call/result pairs in full
    - Assistant reasoning text in full (planning, explanations)

    Compresses:
    - Older tool result messages → one-line summary
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
            content = msg.get("content", "")
            orig_len = len(content)
            preview = content[:100].replace("\n", " ").strip()
            if orig_len > 100:
                preview += "..."
            compacted.append({
                "role": "tool",
                "content": f"[Compacted — {orig_len} chars] {preview}",
                **({"tool_call_id": msg["tool_call_id"]} if "tool_call_id" in msg else {}),
            })
        else:
            compacted.append(msg)

    return prefix + compacted + conversation[keep_from_idx:]


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
            iter_tools = tools if (tools and tool_call_count < config.budget) else None

            # ── LLM call ──
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

                for tc in tool_calls:
                    name, arguments, tc_id = extract_tool_call_info(tc)
                    tool_call_names.append(name)
                    tool_call_count += 1
                    tool_call = ToolCall(id=tc_id, name=name, arguments=arguments)

                    if self.on_token:
                        arg_hint = format_tool_arg_hint(name, arguments)
                        self.on_token(f"\n\x1b[36m\x1b[1m[Tool: {name}{arg_hint}]\x1b[0m\n")

                    # ── Same-args dedup ──
                    if config.enable_dedup:
                        args_key = json.dumps(arguments, sort_keys=True, default=str)
                        current_call = (name, args_key)
                        if (current_call == last_tool_call
                                and name != config.completion_tool):
                            result = ToolResult(
                                tool_call_id=tc_id,
                                name=name,
                                result=(
                                    f"You just called {name} with the same arguments. "
                                    "Try a different approach or complete the task."
                                ),
                                success=False,
                            )
                            if self.on_token:
                                self.on_token("  [Duplicate call blocked]\n")
                        else:
                            result = await self.tool_registry.execute_tool_call(tool_call)
                        last_tool_call = current_call
                    else:
                        result = await self.tool_registry.execute_tool_call(tool_call)

                    result.tool_call_id = tc_id

                    # ── Completion tool detection ──
                    if config.completion_tool and name == config.completion_tool:
                        completion_tool_result = result.result
                        if self.on_token:
                            self.on_token("  [Task complete signal received]\n")

                    # ── Caller tracking callback (sync or async) ──
                    if on_tool_executed:
                        ret = on_tool_executed(name, arguments, result)
                        if asyncio.iscoroutine(ret):
                            await ret

                    messages.append(result.to_ollama_message())

                    if self.on_token:
                        preview = result.result[:120].replace("\n", " ")
                        if result.success:
                            self.on_token(f"\x1b[2m[{preview}]\x1b[0m\n\n")
                        else:
                            self.on_token(f"\x1b[31m[{preview}]\x1b[0m\n\n")

                # ── Completion tool fired ──
                if completion_tool_result is not None:
                    final_response = completion_tool_result
                    completion_method = "tool"
                    break

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
            if tool_call_names:
                final_response = f"[Completed {len(tool_call_names)} tool calls but no summary generated]"
            else:
                final_response = "No response generated."
            if completion_method == "empty":
                completion_method = "budget"

        return LoopResult(
            response=final_response,
            messages=messages,
            tool_call_names=tool_call_names,
            tool_call_count=tool_call_count,
            completion_method=completion_method,
        )
