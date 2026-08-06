"""Endpoint fallback in _run_chat_loop must not replay a failed attempt.

ChatLoop mutates `messages` in place — it appends the assistant turn and every
tool result, and compaction replaces the contents wholesale. The retry loop
handed that same mutated list to the next endpoint, which meant the failed
attempt was replayed as conversation history, the tool budget effectively
multiplied by the number of endpoints, and a failure landing mid-tool-exchange
produced an assistant message whose tool_calls had no matching tool responses —
which OpenAI-compatible endpoints reject with a 400 (deep-dive 2026-08-04).

Drives the real ChatMixin._run_chat_loop against fake endpoints.
"""

from unittest.mock import MagicMock

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.core.chat_loop import LoopConfig
from blipshell.core.tools.base import Tool, ToolRegistry
from blipshell.models.tools import ToolDefinition


class _EchoTool(Tool):
    read_only = True

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="read_file", description="fake read")

    async def execute(self, **kwargs) -> str:
        return "file contents"


class _Client:
    """Streams one scripted turn per call, then plain text."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    async def chat_stream(self, messages, model, tools=None, **kwargs):
        turn = self._script[self._i] if self._i < len(self._script) else {"text": "done"}
        self._i += 1
        if isinstance(turn, Exception):
            raise turn
        tool_calls = None
        if turn.get("tools"):
            tool_calls = [
                {"function": {"name": n, "arguments": a}, "id": f"tc{i}"}
                for i, (n, a) in enumerate(turn["tools"])
            ]
        yield {
            "message": {"content": turn.get("text", ""), "tool_calls": tool_calls},
            "done": True,
        }


class _FailsAfterToolCall(_Client):
    """Emits a tool call (so ChatLoop appends the assistant turn AND the tool
    result), then dies on the follow-up turn. A client that fails on its first
    call never mutates `messages`, so it cannot exercise the rewind at all."""

    def __init__(self):
        super().__init__([])

    async def chat_stream(self, messages, model, tools=None, **kwargs):
        if self._i == 0:
            self._i += 1
            yield {
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "function": {"name": "read_file",
                                     "arguments": {"path": "config.yaml"}},
                        "id": "tc0",
                    }],
                },
                "done": True,
            }
            return
        raise RuntimeError("died after mutating the conversation")


class _Endpoint:
    def __init__(self, name, client, provider="openai"):
        self.name = name
        self.client = client
        self.provider = provider
        self.models = {}
        self.context_tokens = 8192
        self.should_sanitize_pii = False
        self.started = 0
        self.failures = 0

    def start_request(self):
        self.started += 1

    def complete_request(self):
        pass

    def record_success(self, _latency):
        pass

    def record_failure(self):
        self.failures += 1


class _EndpointManager:
    """Hands out endpoints in order, honoring `exclude`."""

    def __init__(self, endpoints):
        self._eps = endpoints

    async def get_endpoint_for_role(self, role, exclude=None, min_context_tokens=None):
        for ep in self._eps:
            if ep.name != exclude:
                return ep
        return None

    def get_context_tokens_for_role(self, role, default=65536):
        return default


class _Agent(ChatMixin):
    """Minimal host for _run_chat_loop."""

    def __init__(self, endpoints):
        self.active_project = None
        self.think_enabled = False
        self.endpoint_manager = _EndpointManager(endpoints)
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(_EchoTool())
        self.model_settings = MagicMock()
        self.model_settings.is_vision.return_value = False
        self.router = MagicMock()
        self.router.get_model.return_value = "model-a"
        self.router.get_fallback_model.return_value = None
        self.config = MagicMock()
        self.config.pii.enabled = False
        self._last_endpoint_used = None

    def _on_tool_executed(self, *a, **k):
        pass


def _messages():
    return [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "read config.yaml"},
    ]


class TestFailedAttemptIsRewound:
    async def test_second_endpoint_sees_the_original_messages(self):
        """The core fix: endpoint B must start from the same conversation
        endpoint A did, not from A's wreckage."""
        seen_by_b = {}

        bad = _FailsAfterToolCall()

        class _RecordingClient(_Client):
            async def chat_stream(self, messages, model, tools=None, **kwargs):
                seen_by_b["messages"] = [dict(m) for m in messages]
                async for chunk in super().chat_stream(messages, model, tools, **kwargs):
                    yield chunk

        ep_a = _Endpoint("A", bad)
        ep_b = _Endpoint("B", _RecordingClient([{"text": "all good"}]))
        agent = _Agent([ep_a, ep_b])

        messages = _messages()
        result, name, model, _ = await agent._run_chat_loop(
            messages=messages,
            config=LoopConfig(budget=5, enable_compaction=False),
        )

        assert name == "B"
        assert result.response == "all good"
        assert [m["content"] for m in seen_by_b["messages"]] == [
            "be helpful", "read config.yaml",
        ], "endpoint B was handed endpoint A's partial attempt"

    async def test_no_orphaned_tool_calls_reach_the_next_endpoint(self):
        """The 400-producing shape: an assistant message carrying tool_calls
        with no matching tool responses."""
        seen = {}

        class _Recorder(_Client):
            async def chat_stream(self, messages, model, tools=None, **kwargs):
                seen["messages"] = [dict(m) for m in messages]
                async for chunk in super().chat_stream(messages, model, tools, **kwargs):
                    yield chunk

        agent = _Agent([_Endpoint("A", _FailsAfterToolCall()),
                        _Endpoint("B", _Recorder([{"text": "ok"}]))])
        await agent._run_chat_loop(
            messages=_messages(),
            config=LoopConfig(budget=5, enable_compaction=False),
        )

        got = seen["messages"]
        assistant_with_calls = [m for m in got if m.get("tool_calls")]
        tool_replies = [m for m in got if m.get("role") == "tool"]
        assert not assistant_with_calls, (
            "next endpoint received an assistant tool_calls message from the "
            "failed attempt"
        )
        assert not tool_replies

    async def test_callers_list_is_restored_not_just_the_local(self):
        """_run_chat_loop is handed the caller's list; a failed attempt must
        not leave debris in it either."""
        agent = _Agent([
            _Endpoint("A", _FailsAfterToolCall()),
            _Endpoint("B", _Client([{"text": "fine"}])),
        ])
        messages = _messages()
        original = [dict(m) for m in messages]

        await agent._run_chat_loop(
            messages=messages,
            config=LoopConfig(budget=5, enable_compaction=False),
        )

        # B answers with plain text and appends nothing, so a clean run leaves
        # the list exactly as it started. Slicing a prefix here would be
        # useless — A's debris is APPENDED, so [:2] looks fine either way.
        assert messages == original, (
            f"failed attempt left debris in the caller's list: "
            f"{[m.get('role') for m in messages]}"
        )

    async def test_successful_first_attempt_is_unaffected(self):
        """Control: with no failure there is nothing to rewind. Passes with
        and without the fix by design — it guards the happy path."""
        agent = _Agent([_Endpoint("A", _Client([{"text": "straight through"}]))])
        result, name, _, _ = await agent._run_chat_loop(
            messages=_messages(),
            config=LoopConfig(budget=5, enable_compaction=False),
        )
        assert name == "A"
        assert result.response == "straight through"
