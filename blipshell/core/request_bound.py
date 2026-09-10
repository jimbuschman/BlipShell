"""Hard bound for an executor's INITIAL request (review finding 6, executor
integration, 2026-09-10).

`_chat_simple` budgets its request as a whole and refuses what cannot fit.
The executor path (`!plan`, TaskExecutor.execute_dynamic) composed its first
request from the base prompt, memory context, continuity block, chat history
and the task, and relied on the loop's compaction - which trims OLD tool
results and messages, never the system message - so an oversized first
request went out over the limit. This is the same policy as the chat path,
as one pure function:

  trim, in order, the memory block -> the oldest chat-history turns -> the
  continuity block -> the tool schemas; record each cut; if the MANDATORY
  parts (base prompt, capability block, task) plus the response reserve
  still do not fit, raise ContextOverflowError rather than send.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from blipshell.llm.exceptions import ContextOverflowError
from blipshell.memory.manager import estimate_tokens

RESPONSE_RESERVE_MAX_TOKENS = 2048
MEMORY_HEADER = "\n\n--- RELEVANT MEMORIES ---\n"


@dataclass
class BoundedRequest:
    messages: list[dict]
    tools: list | None
    omissions: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    response_reserve: int = 0
    context_limit: int = 0


def response_reserve_for(context_limit: int) -> int:
    return min(RESPONSE_RESERVE_MAX_TOKENS, max(256, context_limit // 8))


def _tools_tokens(tools) -> int:
    return estimate_tokens(json.dumps(tools, default=str)) if tools else 0


def _messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content") or "")) + 4 for m in messages)


def bound_initial_request(*, base_prompt: str, capability_context: str, memory_context: str,
                          continuity_context: str, chat_history: list[dict] | None, task_message: dict,
                          tools: list | None, context_limit: int) -> BoundedRequest:
    """Compose the first request so that estimated tokens + reserve <= limit.
    Raises ContextOverflowError when the mandatory content alone cannot fit."""
    reserve = response_reserve_for(context_limit)
    history = list(chat_history or [])
    memory = memory_context or ""
    continuity = continuity_context or ""
    omissions: list[str] = []

    def compose():
        sys_prompt = base_prompt
        if capability_context:
            sys_prompt += f"\n\n{capability_context}"
        if memory:
            sys_prompt += f"{MEMORY_HEADER}{memory}"
        if continuity:
            sys_prompt += continuity
        return [{"role": "system", "content": sys_prompt}, *history, task_message]

    def total(msgs, tls):
        return _messages_tokens(msgs) + _tools_tokens(tls) + reserve

    msgs = compose()
    if total(msgs, tools) <= context_limit:
        return BoundedRequest(msgs, tools, omissions, _messages_tokens(msgs) + _tools_tokens(tools), reserve, context_limit)

    # 1. memory block: truncate to what fits, else drop
    if memory:
        over = total(msgs, tools) - context_limit
        mem_tokens = estimate_tokens(memory)
        keep_tokens = max(mem_tokens - over, 0)
        if keep_tokens > 0:
            keep_chars = int(len(memory) * keep_tokens / mem_tokens * 0.9)
            memory = memory[:keep_chars] + f"\n[memories truncated: ~{mem_tokens - keep_tokens} tokens omitted to fit the window]"
            omissions.append(f"memory block truncated (~{mem_tokens - keep_tokens} tokens)")
        else:
            omissions.append(f"memory block omitted ({mem_tokens} tokens)")
            memory = ""
        msgs = compose()
    # 2. chat history: drop oldest turns first
    dropped = 0
    while history and total(msgs, tools) > context_limit:
        history.pop(0)
        dropped += 1
        msgs = compose()
    if dropped:
        omissions.append(f"chat history: {dropped} oldest message(s) omitted")
    # 3. continuity block
    if continuity and total(msgs, tools) > context_limit:
        omissions.append(f"scratchpad/notes block omitted ({estimate_tokens(continuity)} tokens)")
        continuity = ""
        msgs = compose()
    # 4. tool schemas (the model cannot call tools this turn - said so)
    if tools and total(msgs, tools) > context_limit:
        omissions.append(f"tool schemas omitted ({_tools_tokens(tools)} tokens): window too small")
        tools = None
    if total(msgs, tools) > context_limit:
        raise ContextOverflowError(
            f"The task request does not fit the model's context window of {context_limit} tokens: "
            f"the mandatory parts (system prompt, task, response reserve {reserve}) estimate "
            f"{total(msgs, tools)} tokens even after {'; '.join(omissions) or 'nothing left to trim'}."
        )
    return BoundedRequest(msgs, tools, omissions, _messages_tokens(msgs) + _tools_tokens(tools), reserve, context_limit)
