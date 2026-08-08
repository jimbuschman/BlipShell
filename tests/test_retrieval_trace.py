"""/why provenance (V2_PLAN 5.4): the retrieval trace exists, carries what
was actually injected, and renders.

The search path itself is exercised via the real _search_relevant_memories
with faked search results — what's pinned is that every injected item shows
up in the trace with its score and source, because a trace that silently
drops items answers "why did you bring that up" wrong.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.core.agent_chat import ChatMixin


def _result(id, text, score):
    return SimpleNamespace(
        id=id, text=text, summary=None, timestamp=None, boosted_score=score,
    )


def _agent(memories=(), core=(), lessons=()):
    a = ChatMixin.__new__(ChatMixin)
    a.memory_manager = MagicMock()
    a.session_manager = MagicMock(session_id=1)
    a.sqlite = MagicMock()
    a.sqlite.log_turn_event = AsyncMock()
    a._turn_number = 1
    a.active_project = None
    a._relevance_injected_thoughts = set()
    a._pending_thought_fatigue = {}
    a.config = MagicMock()
    a.config.reflection.inject_enabled = False

    a.search = MagicMock()
    a.search.search = AsyncMock(return_value=list(memories))
    a.search.search_core_memories = AsyncMock(return_value=list(core))
    a.search.search_lessons = AsyncMock(return_value=list(lessons))
    a.search.last_search_stats = {"chroma_hits": len(memories), "fts_hits": 0,
                                  "entity_names": ["blipshell"]}
    return a


class TestTrace:
    async def test_every_injected_item_is_in_the_trace(self):
        a = _agent(
            memories=[_result(11, "we discussed the entity graph", 0.91)],
            core=[{"document": "user prefers thorough tests", "similarity": 0.8}],
            lessons=[{"document": "trace the call chain first", "similarity": 0.7}],
        )

        await a._search_relevant_memories("entity graph work")

        trace = a._last_retrieval_trace
        sources = sorted(i["source"] for i in trace["injected"])
        assert sources == ["core", "lesson", "memory"]
        by_source = {i["source"]: i for i in trace["injected"]}
        assert by_source["memory"]["id"] == 11
        assert by_source["memory"]["score"] == 0.91
        assert "entity graph" in by_source["memory"]["preview"]

    async def test_below_floor_results_do_not_appear_as_injected(self):
        """Core/lesson results under their floors never reach the pool — a
        trace claiming they were injected would be lying in the other
        direction."""
        a = _agent(core=[{"document": "weak match", "similarity": 0.2}])

        await a._search_relevant_memories("query")

        assert a._last_retrieval_trace["injected"] == []

    async def test_trace_is_persisted_to_turn_events(self):
        a = _agent(memories=[_result(5, "stored fact", 0.8)])

        await a._search_relevant_memories("query")

        call = next(
            c for c in a.sqlite.log_turn_event.await_args_list
            if c.args[2] == "search_complete"
        )
        assert call.args[3]["injected"][0]["id"] == 5

    async def test_stats_ride_along(self):
        a = _agent(memories=[_result(5, "x", 0.8)])

        await a._search_relevant_memories("query")

        assert a._last_retrieval_trace["stats"]["entity_names"] == ["blipshell"]
