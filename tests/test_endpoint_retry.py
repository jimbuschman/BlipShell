"""Endpoint recovery happens at the MODEL-CALL boundary (V3_PLAN Stage A2).

History: ChatLoop mutates `messages` in place. The first fix (2026-08) had
_run_chat_loop snapshot the list before each endpoint attempt and rewind it
on failure, so the next endpoint would not see the failed attempt's debris
(and OpenAI-compatible endpoints would not 400 on an assistant tool_calls
message with no results). But rewinding the transcript does not rewind the
WORLD: a tool that had already run (a file write, a queued follow-up, a
command) ran again on the next endpoint, and ChatLoop.run restarted the tool
budget from zero per attempt. The external review reproduced two executions
under budget=1.

Now the loop itself switches endpoints when a model call fails, via
LoopConfig.on_model_call_error. The transcript at that point is consistent
(every announced call has its result), so nothing is rewound: completed
tool/result pairs stay, the budget is turn-wide, and the next endpoint
continues the same turn. A transcript repair is never a side-effect rollback.

Drives the real ChatMixin._run_chat_loop and the real ChatLoop against fake
endpoints; no model.
"""

from unittest.mock import MagicMock

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.core.chat_loop import ChatLoop, LoopConfig
from blipshell.core.tools.base import Tool, ToolRegistry
from blipshell.models.tools import ToolDefinition


class _CountingTool(Tool):
    read_only = True

    def __init__(self):
        self.calls: list[dict] = []

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="read_file", description="fake read")

    async def execute(self, **kwargs) -> str:
        self.calls.append(dict(kwargs))
        return "file contents"


class _Client:
    """Streams one scripted turn per call, then plain text. A turn may be an
    Exception (raised) instead of a dict."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.seen: list[list[dict]] = []
        self._last_exc: Exception | None = None

    async def chat_stream(self, messages, model, tools=None, **kwargs):
        self.seen.append([dict(m) for m in messages])
        turn = self._script[self._i] if self._i < len(self._script) else {"text": "done"}
        self._i += 1
        if isinstance(turn, Exception):
            self._last_exc = turn
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

    async def chat(self, messages, model, tools=None, **kwargs):
        # stream_chat falls back to non-streaming on a streaming error; the
        # fallback must fail the same way or the "failure" never surfaces.
        raise self._last_exc or RuntimeError("non-streaming fallback also down")


def _fails_after_tool():
    """Emits a tool call (loop appends assistant turn AND tool result), then
    dies on the follow-up model call - the only place recovery matters."""
    return _Client([{"tools": [("read_file", {"path": "config.yaml"})]},
                    RuntimeError("died after the tool ran")])


class _Endpoint:
    def __init__(self, name, client, provider="openai"):
        self.name = name
        self.client = client
        self.provider = provider
        self.models = {}
        self.context_tokens = 8192
        self.should_sanitize_pii = False
        self.started = 0
        self.completed = 0
        self.failures = 0
        self.successes = 0

    def start_request(self):
        self.started += 1

    def complete_request(self):
        self.completed += 1

    def record_success(self, _latency):
        self.successes += 1

    def record_failure(self):
        self.failures += 1


class _EndpointManager:
    """Hands out endpoints in order, honoring `exclude`."""

    def __init__(self, endpoints):
        self._eps = endpoints

    async def get_endpoint_for_role(self, role, exclude=None, min_context_tokens=None):
        # The real manager accepts a name or a set of names (router.generate
        # passes a set); the runner now excludes every endpoint that failed
        # this turn, so a dead endpoint is not re-tried after the next dies.
        excluded = set() if exclude is None else ({exclude} if isinstance(exclude, str) else set(exclude))
        for ep in self._eps:
            if ep.name not in excluded:
                return ep
        return None

    def get_context_tokens_for_role(self, role, default=65536):
        return default


class _Agent(ChatMixin):
    """Minimal host for _run_chat_loop."""

    def __init__(self, endpoints, fallback_model=None):
        self.active_project = None
        self.think_enabled = False
        self.endpoint_manager = _EndpointManager(endpoints)
        self.tool_registry = ToolRegistry()
        self.tool = _CountingTool()
        self.tool_registry.register(self.tool)
        self.model_settings = MagicMock()
        self.model_settings.is_vision.return_value = False
        self.router = MagicMock()
        self.router.get_model.return_value = "model-a"
        self.router.get_fallback_model.return_value = fallback_model
        self.config = MagicMock()
        self.config.pii.enabled = False
        self._last_endpoint_used = None
        self.tokens: list[str] = []

    def _on_tool_executed(self, *a, **k):
        pass


def _messages():
    return [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "read config.yaml"},
    ]


def _assert_paired(messages):
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            want = [tc["id"] for tc in m["tool_calls"]]
            got = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                got.append(messages[j].get("tool_call_id"))
                j += 1
            assert got == want, f"announced {want}, answered {got}"
            i = j
        else:
            i += 1


class TestRecoveryAtTheModelCallBoundary:

    async def test_tool_runs_once_and_the_next_endpoint_continues_the_turn(self):
        """THE fix. Endpoint B is handed the completed exchange, not the
        original conversation, and the tool is not executed again."""
        b_client = _Client([{"text": "all good"}])
        agent = _Agent([_Endpoint("A", _fails_after_tool()), _Endpoint("B", b_client)])

        messages = _messages()
        result, name, model, using_fallback = await agent._run_chat_loop(
            messages=messages, config=LoopConfig(budget=5, enable_compaction=False),
        )

        assert name == "B"
        assert result.response == "all good"
        assert using_fallback is False
        assert agent.tool.calls == [{"path": "config.yaml"}], "the tool ran more than once"
        roles = [m["role"] for m in b_client.seen[0]]
        assert roles == ["system", "user", "assistant", "tool"], (
            "B must see A's completed tool exchange, not a rewound conversation"
        )
        _assert_paired(b_client.seen[0])

    async def test_budget_is_turn_wide_across_endpoints(self):
        """budget=1: A spent it. B answers with another tool call, which must
        NOT run - the counter used to restart at zero per endpoint."""
        b_client = _Client([{"tools": [("read_file", {"path": "other.yaml"})]}, {"text": "x"}])
        agent = _Agent([_Endpoint("A", _fails_after_tool()), _Endpoint("B", b_client)])

        result, name, _, _ = await agent._run_chat_loop(
            messages=_messages(), config=LoopConfig(budget=1, enable_compaction=False),
        )

        assert name == "B"
        assert result is not None
        assert agent.tool.calls == [{"path": "config.yaml"}], (
            f"budget=1 but the tool ran {len(agent.tool.calls)} times across endpoints"
        )
        assert result.tool_call_count == 1

    async def test_failure_before_any_mutation_hands_b_the_untouched_conversation(self):
        b_client = _Client([{"text": "fine"}])
        agent = _Agent([_Endpoint("A", _Client([RuntimeError("cold")])), _Endpoint("B", b_client)])
        messages = _messages()
        result, name, _, _ = await agent._run_chat_loop(
            messages=messages, config=LoopConfig(budget=5, enable_compaction=False),
        )
        assert name == "B" and result.response == "fine"
        assert [m["content"] for m in b_client.seen[0]] == ["be helpful", "read config.yaml"]
        assert agent.tool.calls == []

    async def test_callers_list_holds_one_consistent_exchange_and_no_debris(self):
        agent = _Agent([_Endpoint("A", _fails_after_tool()), _Endpoint("B", _Client([{"text": "fine"}]))])
        messages = _messages()
        await agent._run_chat_loop(
            messages=messages, config=LoopConfig(budget=5, enable_compaction=False),
        )
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "tool"], roles
        _assert_paired(messages)
        assert sum(1 for m in messages if m.get("tool_calls")) == 1, "duplicate exchange"

    async def test_every_endpoint_dead_returns_none_without_reexecuting(self):
        agent = _Agent([_Endpoint("A", _fails_after_tool()),
                        _Endpoint("B", _Client([RuntimeError("also dead")]))])
        result, name, _, _ = await agent._run_chat_loop(
            messages=_messages(), config=LoopConfig(budget=5, enable_compaction=False),
        )
        assert result is None
        assert agent.tool.calls == [{"path": "config.yaml"}]

    async def test_fallback_model_is_tried_when_endpoints_are_exhausted(self):
        """Single endpoint, primary model fails; the router's fallback model is
        tried on the same endpoint, continuing the turn."""
        client = _Client([{"tools": [("read_file", {"path": "a"})]},
                          RuntimeError("primary model gone"),
                          {"text": "ok on fallback"}])
        agent = _Agent([_Endpoint("A", client)], fallback_model="model-b")
        result, name, model, using_fallback = await agent._run_chat_loop(
            messages=_messages(), config=LoopConfig(budget=5, enable_compaction=False),
        )
        assert result.response == "ok on fallback"
        assert using_fallback is True
        assert model == "model-b"
        assert agent.tool.calls == [{"path": "a"}]
        # the fallback call continued from the completed exchange
        assert [m["role"] for m in client.seen[-1]] == ["system", "user", "assistant", "tool"]

    async def test_endpoint_accounting_is_balanced(self):
        ep_a, ep_b = _Endpoint("A", _fails_after_tool()), _Endpoint("B", _Client([{"text": "ok"}]))
        agent = _Agent([ep_a, ep_b])
        await agent._run_chat_loop(messages=_messages(), config=LoopConfig(budget=5, enable_compaction=False))
        assert (ep_a.started, ep_a.completed, ep_a.failures, ep_a.successes) == (1, 1, 1, 0)
        assert (ep_b.started, ep_b.completed, ep_b.failures, ep_b.successes) == (1, 1, 0, 1)

    async def test_successful_first_attempt_is_unaffected(self):
        agent = _Agent([_Endpoint("A", _Client([{"text": "straight through"}]))])
        result, name, _, _ = await agent._run_chat_loop(
            messages=_messages(), config=LoopConfig(budget=5, enable_compaction=False),
        )
        assert name == "A"
        assert result.response == "straight through"


class TestLoopHookDirectly:
    """ChatLoop.on_model_call_error, without the agent around it."""

    async def test_hook_swaps_client_and_the_turn_continues(self):
        tool = _CountingTool()
        reg = ToolRegistry()
        reg.register(tool)
        a = _fails_after_tool()
        b = _Client([{"text": "from b"}])
        errors = []

        async def switch(exc):
            errors.append(exc)
            return b, "model-b", {"options": {"num_ctx": 1}}

        cfg = LoopConfig(budget=5, on_model_call_error=switch)
        messages = _messages()
        result = await ChatLoop(reg).run(client=a, messages=messages, model="model-a",
                                         tools=[{}], chat_kwargs={}, config=cfg)
        assert result.response == "from b"
        assert len(errors) == 1 and "died after the tool ran" in str(errors[0])
        assert tool.calls == [{"path": "config.yaml"}]
        assert [m["role"] for m in b.seen[0]] == ["system", "user", "assistant", "tool"]

    async def test_hook_returning_none_reraises(self):
        reg = ToolRegistry()

        async def give_up(exc):
            return None

        with pytest.raises(RuntimeError, match="cold"):
            await ChatLoop(reg).run(client=_Client([RuntimeError("cold")]), messages=_messages(),
                                    model="m", tools=None, chat_kwargs={},
                                    config=LoopConfig(budget=1, on_model_call_error=give_up))

    async def test_no_hook_reraises_as_before(self):
        with pytest.raises(RuntimeError, match="cold"):
            await ChatLoop(ToolRegistry()).run(client=_Client([RuntimeError("cold")]), messages=_messages(),
                                               model="m", tools=None, chat_kwargs={}, config=LoopConfig(budget=1))

    async def test_tool_exception_becomes_a_failure_result_not_an_orphan(self):
        """A tool that RAISES (rather than returning ToolFailure) used to
        escape the sequential path with the assistant tool_calls message
        appended and no result - the one shape the old rewind existed for."""
        class _Boom(Tool):
            read_only = True

            def definition(self):
                return ToolDefinition(name="boom", description="raises")

            async def execute(self, **kwargs):
                raise ValueError("kaboom")

        reg = ToolRegistry()
        reg.register(_Boom())
        client = _Client([{"tools": [("boom", {})]}, {"text": "recovered"}])
        messages = _messages()
        result = await ChatLoop(reg).run(client=client, messages=messages, model="m", tools=[{}],
                                         chat_kwargs={}, config=LoopConfig(budget=3, enable_parallel=False))
        assert result.response == "recovered"
        _assert_paired(messages)
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        assert "kaboom" in tool_msg["content"]
