"""Only read-only tools run concurrently (V3_PLAN Stage A5).

_partition_for_parallel put a call in the sequential group only when it
needed an approval callback (and one was installed) or was ask_user. In
normal chat there is no approval callback, so every mutating tool in a batch
ran concurrently - two edits to one file, a write and a command that reads
it. Tools already declare `read_only` (used by plan mode); the partition now
honours it: writes run sequentially in announced order, reads in parallel.

Drives the real ChatLoop with the scripted client; no model.
"""

from __future__ import annotations

import asyncio

from blipshell.core.chat_loop import ChatLoop, LoopConfig
from blipshell.core.tools.base import Tool, ToolRegistry
from blipshell.models.tools import ToolDefinition
from tests.fakes import FakeTool, ScriptedLLMClient, make_registry


class TimelineTool(Tool):
    """Records start/end so concurrency can be asserted."""

    def __init__(self, name: str, read_only: bool, timeline: list, hold: float = 0.02):
        self._name = name
        self.read_only = read_only
        self._timeline = timeline
        self._hold = hold

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self._name, description=f"fake {self._name}")

    async def execute(self, **kwargs) -> str:
        self._timeline.append(("start", self._name))
        await asyncio.sleep(self._hold)
        self._timeline.append(("end", self._name))
        return f"{self._name} ran"


def _max_overlap(timeline, names):
    active, peak = 0, 0
    for ev, name in timeline:
        if name not in names:
            continue
        active += 1 if ev == "start" else -1
        peak = max(peak, active)
    return peak


def _messages():
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]


class TestPartition:

    def test_writes_are_sequential_reads_parallel_without_approval_callback(self):
        reg = make_registry(FakeTool("read_a", read_only=True), FakeTool("read_b", read_only=True),
                            FakeTool("write_a"), FakeTool("write_b"))
        loop = ChatLoop(reg)
        calls = [("read_a", {}, "0"), ("write_a", {}, "1"), ("read_b", {}, "2"), ("write_b", {}, "3")]
        seq, par = loop._partition_for_parallel(calls, LoopConfig())
        assert seq == [1, 3], "mutating tools must be sequential even with no approval callback"
        assert par == [0, 2]

    def test_unknown_tool_is_sequential(self):
        loop = ChatLoop(make_registry(FakeTool("read_a", read_only=True)))
        seq, par = loop._partition_for_parallel([("nope", {}, "0"), ("read_a", {}, "1")], LoopConfig())
        assert seq == [0] and par == [1]

    def test_ask_user_is_sequential_even_if_read_only(self):
        loop = ChatLoop(make_registry(FakeTool("ask_user", read_only=True)))
        seq, par = loop._partition_for_parallel([("ask_user", {}, "0")], LoopConfig())
        assert seq == [0] and par == []

    def test_approval_gated_read_is_sequential_when_a_callback_exists(self):
        reg = make_registry(FakeTool("read_a", read_only=True))

        async def approve(name, args, force=False):
            return True

        reg.set_approval_callback(approve, {"read_a"})
        loop = ChatLoop(reg)
        seq, par = loop._partition_for_parallel([("read_a", {}, "0")], LoopConfig())
        assert seq == [0]


class TestExecutionOrder:

    async def test_writes_never_overlap_and_keep_announced_order(self):
        timeline: list = []
        reg = make_registry(
            TimelineTool("read_a", True, timeline), TimelineTool("read_b", True, timeline),
            TimelineTool("write_a", False, timeline), TimelineTool("write_b", False, timeline),
        )
        client = ScriptedLLMClient([
            {"tools": [("read_a", {}), ("write_a", {}), ("read_b", {}), ("write_b", {})]},
            {"text": "ok"},
        ])
        messages = _messages()
        result = await ChatLoop(reg).run(client=client, messages=messages, model="m", tools=[{}],
                                         chat_kwargs={}, config=LoopConfig(budget=10, enable_parallel=True))
        assert result.tool_call_count == 4
        assert _max_overlap(timeline, {"write_a", "write_b"}) == 1, "two mutating tools ran concurrently"
        starts = [n for ev, n in timeline if ev == "start" and n.startswith("write")]
        assert starts == ["write_a", "write_b"], "writes must keep the announced order"
        # reads are allowed to overlap (and do, given the hold)
        assert _max_overlap(timeline, {"read_a", "read_b"}) == 2
        # results are appended in announced order regardless of execution order
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["tc0", "tc1", "tc2", "tc3"]

    async def test_all_reads_still_run_in_parallel(self):
        timeline: list = []
        reg = make_registry(TimelineTool("r1", True, timeline), TimelineTool("r2", True, timeline),
                            TimelineTool("r3", True, timeline))
        client = ScriptedLLMClient([{"tools": [("r1", {}), ("r2", {}), ("r3", {})]}, {"text": "ok"}])
        await ChatLoop(reg).run(client=client, messages=_messages(), model="m", tools=[{}],
                                chat_kwargs={}, config=LoopConfig(budget=10, enable_parallel=True))
        assert _max_overlap(timeline, {"r1", "r2", "r3"}) == 3
