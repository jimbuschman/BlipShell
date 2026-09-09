"""Every announced tool call gets a result (V3_PLAN Stage A3).

ChatLoop.run appended the assistant message with the FULL tool_calls list, then
sliced the parsed calls to the remaining budget - so calls past the budget were
announced and never answered. OpenAI-compatible endpoints reject the next
request (every tool_call id must have a tool message). The external review
reproduced it: two announced ids, one result, one unmatched.

Now: unexecuted calls get an explicit budget-denied tool result, executed-call
accounting is separate from declared-call accounting, and a pairing repair
runs before EVERY outbound request as a backstop (fail-open, logged at
WARNING, like the other guardrail internals).

Drives the real ChatLoop with the scripted client; no model.
"""

from __future__ import annotations

import logging

from blipshell.core.chat_loop import ChatLoop, LoopConfig, repair_tool_pairing
from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition
from tests.fakes import ScriptedLLMClient, make_registry


class CountingTool(Tool):
    read_only = True

    def __init__(self, name: str):
        self._name = name
        self.calls: list[dict] = []

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self._name, description=f"fake {self._name}")

    async def execute(self, **kwargs) -> str:
        self.calls.append(dict(kwargs))
        return f"{self._name} ran"


def announced_ids(messages: list[dict]) -> list[list[str]]:
    return [[tc["id"] for tc in m["tool_calls"]] for m in messages if m.get("tool_calls")]


def assert_paired(messages: list[dict]) -> None:
    """Every assistant tool_calls id is answered, in order, before any other message."""
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
            assert got == want, f"assistant announced {want}, tool replies were {got}"
            i = j
        else:
            i += 1


def _messages():
    return [{"role": "system", "content": "be helpful"},
            {"role": "user", "content": "do two things"}]


class TestBudgetDenied:

    async def test_every_announced_call_is_answered_when_budget_trims(self):
        read, write = CountingTool("read_file"), CountingTool("write_file")
        client = ScriptedLLMClient([
            {"tools": [("read_file", {"path": "a"}), ("write_file", {"path": "b"})]},
            {"text": "done"},
        ])
        loop = ChatLoop(make_registry(read, write))
        messages = _messages()
        result = await loop.run(client=client, messages=messages, model="m", tools=[{}],
                                chat_kwargs={}, config=LoopConfig(budget=1, enable_parallel=False))

        assert result.response == "done"
        assert_paired(messages)
        assert announced_ids(messages) == [["tc0", "tc1"]]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert tool_msgs[0]["tool_call_id"] == "tc0" and "read_file ran" in tool_msgs[0]["content"]
        assert tool_msgs[1]["tool_call_id"] == "tc1"
        assert "budget" in tool_msgs[1]["content"].lower()
        assert "not executed" in tool_msgs[1]["content"].lower()
        # executed-call accounting is separate from declared-call accounting
        assert result.tool_call_count == 1
        assert result.tool_call_names == ["read_file"]
        assert read.calls == [{"path": "a"}]
        assert write.calls == [], "a budget-denied call must not run"
        # and the NEXT request the model saw was well-formed
        assert_paired(client.sent_messages[1])

    async def test_parallel_path_also_pairs_denied_calls(self):
        a, b, c = (CountingTool(n) for n in ("ta", "tb", "tc"))
        client = ScriptedLLMClient([
            {"tools": [("ta", {}), ("tb", {}), ("tc", {})]},
            {"text": "ok"},
        ])
        loop = ChatLoop(make_registry(a, b, c))
        messages = _messages()
        result = await loop.run(client=client, messages=messages, model="m", tools=[{}],
                                chat_kwargs={}, config=LoopConfig(budget=2, enable_parallel=True))

        assert_paired(messages)
        assert result.tool_call_count == 2
        assert result.tool_call_names == ["ta", "tb"]
        assert c.calls == []
        denied = [m for m in messages if m.get("role") == "tool" and m["tool_call_id"] == "tc2"]
        assert len(denied) == 1 and "budget" in denied[0]["content"].lower()

    async def test_denied_calls_do_not_fire_the_executed_callback(self):
        a, b = CountingTool("ta"), CountingTool("tb")
        client = ScriptedLLMClient([{"tools": [("ta", {}), ("tb", {})]}, {"text": "ok"}])
        seen = []
        loop = ChatLoop(make_registry(a, b))
        await loop.run(client=client, messages=_messages(), model="m", tools=[{}], chat_kwargs={},
                       config=LoopConfig(budget=1, enable_parallel=False),
                       on_tool_executed=lambda name, args, res: seen.append(name))
        assert seen == ["ta"], "on_tool_executed is for executed tools only"

    async def test_no_trim_no_denial(self):
        a, b = CountingTool("ta"), CountingTool("tb")
        client = ScriptedLLMClient([{"tools": [("ta", {}), ("tb", {})]}, {"text": "ok"}])
        loop = ChatLoop(make_registry(a, b))
        messages = _messages()
        result = await loop.run(client=client, messages=messages, model="m", tools=[{}],
                                chat_kwargs={}, config=LoopConfig(budget=5, enable_parallel=False))
        assert_paired(messages)
        assert result.tool_call_count == 2
        assert all("budget" not in m["content"].lower() for m in messages if m.get("role") == "tool")


class TestPairingRepair:

    def test_inserts_missing_results_in_place(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc0", "function": {"name": "f", "arguments": {}}},
                {"id": "tc1", "function": {"name": "g", "arguments": {}}},
            ]},
            {"role": "tool", "tool_call_id": "tc0", "content": "f result"},
            {"role": "user", "content": "next"},
        ]
        fixed = repair_tool_pairing(messages)
        assert fixed == 1
        assert_paired(messages)
        inserted = messages[3]
        assert inserted["role"] == "tool" and inserted["tool_call_id"] == "tc1"
        assert "no result" in inserted["content"].lower()
        assert "unknown" in inserted["content"].lower(), "a missing result is not proof of non-execution"
        assert messages[4]["role"] == "user"

    def test_orphan_at_end_of_transcript(self):
        messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a1", "function": {"name": "f", "arguments": {}}},
                {"id": "a2", "function": {"name": "f", "arguments": {}}},
            ]},
        ]
        assert repair_tool_pairing(messages) == 2
        assert_paired(messages)

    def test_consistent_transcript_is_untouched(self):
        messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a1", "function": {"name": "f", "arguments": {}}},
            ]},
            {"role": "tool", "tool_call_id": "a1", "content": "r"},
            {"role": "assistant", "content": "done"},
        ]
        before = [dict(m) for m in messages]
        assert repair_tool_pairing(messages) == 0
        assert messages == before

    def test_ollama_style_calls_without_ids_are_left_alone(self):
        # Native Ollama tool calls may carry no id; there is nothing to pair by.
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "f", "arguments": {}}},
            ]},
            {"role": "tool", "content": "r"},
        ]
        before = [dict(m) for m in messages]
        assert repair_tool_pairing(messages) == 0
        assert messages == before

    async def test_repair_runs_before_every_outbound_request(self, caplog):
        """Backstop: a caller hands the loop a history with an orphan (e.g. a
        transcript restored from disk). The model must never see it."""
        client = ScriptedLLMClient([{"text": "ok"}])
        loop = ChatLoop(make_registry())
        messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "z9", "function": {"name": "f", "arguments": {}}},
            ]},
            {"role": "user", "content": "carry on"},
        ]
        with caplog.at_level(logging.WARNING, logger="blipshell.core.chat_loop"):
            await loop.run(client=client, messages=messages, model="m", tools=None,
                           chat_kwargs={}, config=LoopConfig(budget=1))
        assert_paired(client.sent_messages[0])
        assert any("pairing" in r.getMessage().lower() for r in caplog.records)


class TestCompletionToolIsNeverDenied:

    async def test_completion_call_survives_the_trim(self):
        """[read_file, task_complete] with one slot left: the model reached
        completion, so completion runs and the READ is the denied call.
        Slicing used to drop task_complete silently and end the turn on
        'budget' with no summary."""
        read = CountingTool("read_file")
        done = CountingTool("task_complete")
        client = ScriptedLLMClient([
            {"tools": [("read_file", {"path": "a"}), ("task_complete", {"summary": "all done"})]},
        ])
        loop = ChatLoop(make_registry(read, done))
        messages = _messages()
        result = await loop.run(client=client, messages=messages, model="m", tools=[{}], chat_kwargs={},
                                config=LoopConfig(budget=1, enable_parallel=False,
                                                  completion_tool="task_complete"))
        assert_paired(messages)
        assert result.completion_method == "tool"
        assert done.calls == [{"summary": "all done"}]
        assert read.calls == []
        assert result.tool_call_names == ["task_complete"]
        denied = [m for m in messages if m.get("role") == "tool" and m["tool_call_id"] == "tc0"]
        assert denied and "budget" in denied[0]["content"].lower()
