"""Benchmark file: agent.py _chat_simple with KNOWN BUGS.

This file contains a version of the agent's chat loop with intentional
bugs for testing LLM code review capabilities. Use this to benchmark
different models on their ability to find real issues.

Known bugs (don't peek before testing!):
- Count: 3 bugs of varying severity
- Hint: focus on control flow and error handling

To test: point an LLM at this file and ask it to find bugs.
"""

import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Endpoint:
    """Simplified endpoint for context."""
    def __init__(self, name: str):
        self.name = name
        self.active_requests = 0
        self.failure_count = 0
        self.success_count = 0

    @property
    def can_accept_request(self) -> bool:
        return self.active_requests < 2

    def start_request(self):
        self.active_requests += 1

    def complete_request(self):
        self.active_requests = max(0, self.active_requests - 1)

    def record_success(self, response_time: float):
        self.failure_count = 0
        self.success_count += 1

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= 3:
            logger.warning("Endpoint %s disabled after %d failures", self.name, self.failure_count)


class ToolCall:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments


class ToolResult:
    def __init__(self, result: str):
        self.result = result

    def to_ollama_message(self) -> dict:
        return {"role": "tool", "content": self.result}


class Agent:
    """Simplified agent with the chat loop containing known bugs.

    This is extracted from the real agent for benchmarking LLM code review.
    """

    def __init__(self):
        self.endpoint_manager = None
        self.router = None
        self.tool_registry = None
        self.config = None
        self.session_manager = None
        self.memory_manager = None
        self._last_endpoint_used: Optional[str] = None

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
        """Extract name and arguments from a tool call object or dict."""
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", "") or ""
            args = getattr(fn, "arguments", {}) or {}
            return name, args

        if isinstance(tc, dict):
            fn = tc.get("function", {})
            return fn.get("name", ""), fn.get("arguments", {})

        return "", {}

    async def _chat_simple(
        self,
        user_message: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Simple chat path — flat tool-calling loop.

        Processes a user message by sending it to the LLM with available tools.
        Handles tool call responses in a loop up to max_iterations.
        Returns the final text response from the LLM.
        """
        # Search relevant memories for recall
        await self._search_relevant_memories(user_message)

        # Build message list
        messages = self._build_messages(user_message)

        # Get model and client
        model = self.router.get_model("reasoning")
        client = await self.router.get_client("reasoning")
        if not client:
            return "Error: No available LLM endpoint."

        # Always pass all tools — the model decides whether to use them
        tools = self.tool_registry.get_all_ollama_tools() or None
        max_iterations = self.config.agent.max_tool_iterations if tools else 0
        logger.info("Passing %d tools to model", len(tools) if tools else 0)
        full_response = ""

        for iteration in range(max_iterations + 1):
            endpoint = await self.endpoint_manager.get_endpoint_for_role("reasoning")
            if endpoint:
                endpoint.start_request()
                self._last_endpoint_used = endpoint.name

            try:
                response = await client.chat(
                    messages=messages,
                    model=model,
                    tools=tools if iteration < max_iterations else None,
                )

                content, tool_calls = self._extract_response(response)
                logger.info("LLM response: tool_calls=%s, content_len=%d",
                           bool(tool_calls), len(content))

                if tool_calls and iteration < max_iterations:
                    # Process tool calls
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                    for tc in tool_calls:
                        name, arguments = self._extract_tool_call_info(tc)
                        tool_call = ToolCall(name=name, arguments=arguments)

                        if on_token:
                            on_token(f"\n[Tool: {tool_call.name}]\n")

                        result = await self.tool_registry.execute_tool_call(tool_call)
                        messages.append(result.to_ollama_message())

                        if on_token:
                            on_token(f"[Result: {result.result[:200]}]\n\n")

                    continue  # Loop back for LLM to process tool results
                else:
                    # No tool calls — use the response directly (single call, no double hit)
                    full_response = content
                    if on_token and content:
                        on_token(content)
                    break

                if endpoint:
                    endpoint.record_success(0)
            except Exception as e:
                if endpoint:
                    endpoint.record_failure()
                logger.error("Chat error: %s", e)
                full_response = f"Error: {e}"
                break
            finally:
                if endpoint:
                    endpoint.complete_request()

        return full_response

    async def _search_relevant_memories(self, query: str):
        """Placeholder — searches memories for relevant context."""
        pass

    def _build_messages(self, user_message: str) -> list[dict]:
        """Placeholder — builds the message list with memory context."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]
