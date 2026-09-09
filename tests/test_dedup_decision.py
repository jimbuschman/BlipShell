"""Dedup decision parsing + application (V3_PLAN Stage A1).

The old parser scanned the model's reply for action SUBSTRINGS and defaulted
a missing index to item 0, so "Do not DELETE anything; ADD this as distinct."
archived candidate #1. These tests pin the replacement:

- whole-response grammar (first/last line, last segment), RETRY on anything
  ambiguous, an explicit in-range index required for UPDATE/DELETE;
- an optional schema-constrained JSON path behind `memory.dedup.structured_output`;
- an undecided reply re-asks ONCE, then keeps the new memory (ADD) and archives
  nothing - ambiguity must never destroy a record;
- every archive stamps the archived row with who/why, so `blipshell repair
  --unarchive-memory` can explain and reverse it.

Logic/wiring only - no model. The structured path's schema-validity rate
against qwen3:14b is a benchmark job, not a unit test.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory import dedup_decision as dd
from blipshell.memory.processor import MemoryProcessor
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import Memory


# ---------------------------------------------------------------------------
# Text grammar
# ---------------------------------------------------------------------------

class TestParseActionText:

    @pytest.mark.parametrize("reply,expected", [
        ("ADD", ("ADD", None)),
        ("NONE", ("NONE", None)),
        ("none", ("NONE", None)),
        ("UPDATE 1", ("UPDATE", 0)),
        ("UPDATE 2", ("UPDATE", 1)),
        ("DELETE 1", ("DELETE", 0)),
        ("UPDATE 2.", ("UPDATE", 1)),          # trailing punctuation
        ("**DELETE 1**", ("DELETE", 0)),       # markdown
        ("`ADD`", ("ADD", None)),
        ("Action: UPDATE 3", ("UPDATE", 2)),   # label prefix
        ("UPDATE #2", ("UPDATE", 1)),
        # explanation then verdict: the verdict is the LAST segment
        ("This refines existing memory 2. UPDATE 2", ("UPDATE", 1)),
        ("The new memory adds unique information. ADD", ("ADD", None)),
        ("This is redundant. NONE", ("NONE", None)),
        ("Reasoning first.\n\nUPDATE 1", ("UPDATE", 0)),
        # verdict then explanation: the verdict is the FIRST line
        ("ADD\nThe new memory has details not present elsewhere.", ("ADD", None)),
    ])
    def test_accepts_well_formed(self, reply, expected):
        assert dd.parse_action_text(reply) == expected

    @pytest.mark.parametrize("reply", [
        # THE bug: a careful refusal parsed as DELETE 0
        "Do not DELETE anything; ADD this as distinct.",
        # bare UPDATE/DELETE: no default target, ever
        "UPDATE",
        "DELETE",
        "UPDATE 0",                      # indices are 1-based
        "UPDATE -1",
        "UPDATE 1 and 2",                # ambiguous target
        "garbage response",
        "",
        "   ",
        "ADD\n...\nNONE",                # two conflicting verdicts
        "I would DELETE 1 but UPDATE 2 also works",
        "The action should be taken.",   # prompt echo
    ])
    def test_rejects_ambiguous(self, reply):
        assert dd.parse_action_text(reply) == (dd.RETRY, None)

    def test_substring_of_a_sentence_is_not_a_verdict(self):
        # "ADD" appears as a word but the sentence is not a bare action
        assert dd.parse_action_text("Please ADD this memory.") == (dd.RETRY, None)


# ---------------------------------------------------------------------------
# JSON grammar
# ---------------------------------------------------------------------------

class TestParseActionJson:

    def test_valid_update(self):
        reply = json.dumps({"action": "UPDATE", "target_index": 2, "reason": "refines"})
        assert dd.parse_action_json(reply) == ("UPDATE", 1)

    def test_valid_add_ignores_index(self):
        reply = json.dumps({"action": "ADD", "target_index": None})
        assert dd.parse_action_json(reply) == ("ADD", None)

    def test_none(self):
        assert dd.parse_action_json('{"action": "NONE"}') == ("NONE", None)

    def test_fenced_block_tolerated(self):
        reply = '```json\n{"action": "DELETE", "target_index": 1}\n```'
        assert dd.parse_action_json(reply) == ("DELETE", 0)

    def test_lowercase_action_tolerated(self):
        assert dd.parse_action_json('{"action": "update", "target_index": 1}') == ("UPDATE", 0)

    @pytest.mark.parametrize("reply", [
        '{"action": "UPDATE"}',                          # missing index
        '{"action": "UPDATE", "target_index": 0}',       # 1-based
        '{"action": "DELETE", "target_index": "two"}',   # wrong type
        '{"action": "DELETE", "target_index": 1.5}',
        '{"action": "MERGE", "target_index": 1}',        # unknown action
        '{"target_index": 1}',                           # missing action
        'UPDATE 1',                                      # not JSON
        '[{"action": "ADD"}]',                           # wrong shape
        '',
        'null',
    ])
    def test_rejects_invalid(self, reply):
        assert dd.parse_action_json(reply) == (dd.RETRY, None)

    def test_bool_index_is_not_an_int(self):
        # json true is an int subclass in Python; must not pass as index 1
        assert dd.parse_action_json('{"action": "UPDATE", "target_index": true}') == (dd.RETRY, None)

    def test_schema_is_a_closed_object(self):
        schema = dd.MEMORY_ACTION_SCHEMA
        assert schema["type"] == "object"
        assert set(schema["required"]) >= {"action"}
        assert set(schema["properties"]["action"]["enum"]) == {"ADD", "UPDATE", "DELETE", "NONE"}


# ---------------------------------------------------------------------------
# Processor static method delegates (back-compat surface)
# ---------------------------------------------------------------------------

class TestProcessorParserSurface:

    def test_processor_parser_is_the_strict_one(self):
        assert MemoryProcessor._parse_memory_action(
            "Do not DELETE anything; ADD this as distinct."
        ) == (dd.RETRY, None)
        assert MemoryProcessor._parse_memory_action("UPDATE") == (dd.RETRY, None)
        assert MemoryProcessor._parse_memory_action("UPDATE 2") == ("UPDATE", 1)


# ---------------------------------------------------------------------------
# Applying decisions against a real SQLite store
# ---------------------------------------------------------------------------

async def _seed(sqlite_store, *texts):
    ids = []
    for t in texts:
        ids.append(await sqlite_store.create_memory(
            Memory(session_id=None, role="user", content=t, summary=t),
        ))
    return ids


def _scripted_router(*replies):
    router = MagicMock()
    router.generate = AsyncMock(side_effect=list(replies))
    return router


def _processor(sqlite_store, vectors, router, *, structured=False):
    cfg = MemoryConfig()
    cfg.dedup.structured_output = structured
    return MemoryProcessor(sqlite=sqlite_store, vectors=vectors, router=router, config=cfg)


def _candidates(vectors, ids, texts):
    vectors.search_memories.return_value = [
        {"id": i, "document": t, "similarity": 0.9} for i, t in zip(ids, texts)
    ]


async def _meta(sqlite_store, memory_id) -> dict:
    m = await sqlite_store.get_memory(memory_id)
    return json.loads(m.metadata_json) if m.metadata_json else {}


class TestDecideAndApply:

    async def test_ambiguous_reply_archives_nothing_and_keeps_new(self, sqlite_store, mock_chroma):
        old = await _seed(sqlite_store, "User prefers dark mode", "User uses VS Code")
        (new,) = await _seed(sqlite_store, "User prefers dark mode in the terminal too")
        _candidates(mock_chroma, old, ["User prefers dark mode", "User uses VS Code"])
        bad = "Do not DELETE anything; ADD this as distinct."
        router = _scripted_router(bad, bad)
        proc = _processor(sqlite_store, mock_chroma, router)

        action = await proc._decide_and_apply_action(new, "User prefers dark mode in the terminal too")

        assert action == "ADD"
        assert router.generate.await_count == 2, "undecided -> re-ask exactly once"
        for oid in old:
            m = await sqlite_store.get_memory(oid)
            assert not m.is_archived, f"candidate {oid} was archived on an ambiguous reply"
        mock_chroma.delete_memory.assert_not_called()
        meta = await _meta(sqlite_store, new)
        assert "dedup_undecided" in meta
        assert meta["dedup_undecided"]["candidates"] == old
        assert bad in meta["dedup_undecided"]["reply"]

    async def test_retry_reply_is_used_when_valid(self, sqlite_store, mock_chroma):
        old = await _seed(sqlite_store, "cand one", "cand two")
        (new,) = await _seed(sqlite_store, "refined cand two")
        _candidates(mock_chroma, old, ["cand one", "cand two"])
        router = _scripted_router("UPDATE", "UPDATE 2")
        proc = _processor(sqlite_store, mock_chroma, router)

        action = await proc._decide_and_apply_action(new, "refined cand two")

        assert action == "UPDATE"
        m1 = await sqlite_store.get_memory(old[0])
        m2 = await sqlite_store.get_memory(old[1])
        assert not m1.is_archived
        assert m2.is_archived
        mock_chroma.delete_memory.assert_called_once_with(old[1])
        # the second prompt tells the model its first reply was unusable
        second_prompt = router.generate.await_args_list[1].args[1]
        assert "could not be parsed" in second_prompt.lower() or "exactly one" in second_prompt.lower()

    async def test_out_of_range_index_is_undecided(self, sqlite_store, mock_chroma):
        old = await _seed(sqlite_store, "only candidate")
        (new,) = await _seed(sqlite_store, "new thing")
        _candidates(mock_chroma, old, ["only candidate"])
        router = _scripted_router("DELETE 5", "DELETE 5")
        proc = _processor(sqlite_store, mock_chroma, router)

        assert await proc._decide_and_apply_action(new, "new thing") == "ADD"
        assert not (await sqlite_store.get_memory(old[0])).is_archived
        mock_chroma.delete_memory.assert_not_called()

    async def test_archive_is_stamped_with_provenance(self, sqlite_store, mock_chroma):
        old = await _seed(sqlite_store, "User lives in Ohio")
        (new,) = await _seed(sqlite_store, "User moved to Texas")
        _candidates(mock_chroma, old, ["User lives in Ohio"])
        router = _scripted_router("DELETE 1")
        proc = _processor(sqlite_store, mock_chroma, router)

        assert await proc._decide_and_apply_action(new, "User moved to Texas") == "DELETE"

        m = await sqlite_store.get_memory(old[0])
        assert m.is_archived
        meta = json.loads(m.metadata_json)
        rec = meta["dedup"]
        assert rec["action"] == "DELETE"
        assert rec["by"] == new
        assert rec["candidates"] == old
        assert "DELETE 1" in rec["reply"]
        assert rec["at"]  # ISO timestamp present

    async def test_stamp_preserves_existing_metadata(self, sqlite_store, mock_chroma):
        (old,) = await _seed(sqlite_store, "x")
        await sqlite_store.update_memory(old, metadata_json=json.dumps({"origin": "import"}))
        (new,) = await _seed(sqlite_store, "x refined")
        _candidates(mock_chroma, [old], ["x"])
        proc = _processor(sqlite_store, mock_chroma, _scripted_router("UPDATE 1"))

        await proc._decide_and_apply_action(new, "x refined")

        meta = await _meta(sqlite_store, old)
        assert meta["origin"] == "import"
        assert meta["dedup"]["action"] == "UPDATE"

    async def test_none_is_returned_without_touching_candidates(self, sqlite_store, mock_chroma):
        old = await _seed(sqlite_store, "same fact")
        (new,) = await _seed(sqlite_store, "same fact")
        _candidates(mock_chroma, old, ["same fact"])
        proc = _processor(sqlite_store, mock_chroma, _scripted_router("NONE"))

        assert await proc._decide_and_apply_action(new, "same fact") == "NONE"
        assert not (await sqlite_store.get_memory(old[0])).is_archived

    async def test_no_candidates_means_add_without_llm(self, sqlite_store, mock_chroma):
        (new,) = await _seed(sqlite_store, "lonely")
        mock_chroma.search_memories.return_value = []
        router = _scripted_router()
        proc = _processor(sqlite_store, mock_chroma, router)
        assert await proc._decide_and_apply_action(new, "lonely") == "ADD"
        router.generate.assert_not_awaited()


class TestStructuredPath:

    async def test_structured_reply_applies_and_passes_schema(self, sqlite_store, mock_chroma):
        old = await _seed(sqlite_store, "User uses Windows 10")
        (new,) = await _seed(sqlite_store, "User switched to Linux")
        _candidates(mock_chroma, old, ["User uses Windows 10"])
        reply = json.dumps({"action": "DELETE", "target_index": 1, "reason": "stale OS"})
        router = _scripted_router(reply)
        proc = _processor(sqlite_store, mock_chroma, router, structured=True)

        assert await proc._decide_and_apply_action(new, "User switched to Linux") == "DELETE"
        assert (await sqlite_store.get_memory(old[0])).is_archived
        kwargs = router.generate.await_args.kwargs
        assert kwargs.get("response_format") == dd.MEMORY_ACTION_SCHEMA

    async def test_structured_invalid_twice_is_undecided(self, sqlite_store, mock_chroma):
        old = await _seed(sqlite_store, "a")
        (new,) = await _seed(sqlite_store, "b")
        _candidates(mock_chroma, old, ["a"])
        router = _scripted_router("DELETE 1", "not json either")  # text verdicts are NOT accepted on the JSON path
        proc = _processor(sqlite_store, mock_chroma, router, structured=True)

        assert await proc._decide_and_apply_action(new, "b") == "ADD"
        assert router.generate.await_count == 2
        assert not (await sqlite_store.get_memory(old[0])).is_archived
        assert "dedup_undecided" in await _meta(sqlite_store, new)

    def test_default_is_text_path(self):
        assert MemoryConfig().dedup.structured_output is False


# ---------------------------------------------------------------------------
# Full pipeline: an undecided dedup must not stall or loop the message
# ---------------------------------------------------------------------------

class TestPipelineUndecided:

    async def test_process_message_completes_and_marks_processed(self, sqlite_store, mock_chroma, memory_config):
        old = await _seed(sqlite_store, "User prefers dark mode")
        _candidates(mock_chroma, old, ["User prefers dark mode"])
        bad = "Do not DELETE anything; ADD this as distinct."

        def gen(task_type, prompt="", system=None, think=None, **kw):
            if task_type == "summarization":
                return "User also likes dark mode in the terminal."
            if task_type == "ranking_importance":
                return "3 0.5 preference"
            return bad  # every dedup ask is ambiguous

        router = MagicMock()
        router.generate = AsyncMock(side_effect=gen)
        proc = MemoryProcessor(sqlite=sqlite_store, vectors=mock_chroma, router=router, config=memory_config)

        # Long enough to clear the noise filter.
        mid = await proc.process_message(
            "I also really like dark mode in my terminal, not just the editor, it is easier on my eyes at night.",
            role="user", session_id=None,
        )

        assert mid is not None
        m = await sqlite_store.get_memory(mid)
        cur = await sqlite_store._db.execute("SELECT is_processed FROM memories WHERE id = ?", (mid,))
        (is_processed,) = await cur.fetchone()
        assert is_processed == 1, "an undecided dedup must not leave the row for the sweep to re-pick forever"
        assert not m.is_archived
        assert not (await sqlite_store.get_memory(old[0])).is_archived
        dedup_calls = [c for c in router.generate.await_args_list
                       if (c.kwargs.get("system") or "").startswith("You decide what to do with a new memory")]
        assert len(dedup_calls) == 2


# ---------------------------------------------------------------------------
# Repair: explain + reverse a dedup archive
# ---------------------------------------------------------------------------

class TestUnarchiveRepair:

    async def test_unarchive_restores_row_reembeds_and_keeps_history(self, sqlite_store, mock_chroma):
        old = await _seed(sqlite_store, "User lives in Ohio")
        (new,) = await _seed(sqlite_store, "User moved to Texas")
        _candidates(mock_chroma, old, ["User lives in Ohio"])
        proc = _processor(sqlite_store, mock_chroma, _scripted_router("DELETE 1"))
        await proc._decide_and_apply_action(new, "User moved to Texas")
        assert (await sqlite_store.get_memory(old[0])).is_archived

        report = await dd.unarchive_memory(sqlite_store, mock_chroma, old[0])

        assert report["restored"] is True
        assert report["dedup"]["by"] == new
        m = await sqlite_store.get_memory(old[0])
        assert not m.is_archived
        mock_chroma.add_memory.assert_called_once()
        args = mock_chroma.add_memory.call_args
        assert args.args[0] == old[0]
        assert "Ohio" in args.args[1]
        meta = json.loads(m.metadata_json)
        assert meta["dedup"]["action"] == "DELETE", "history is kept, not erased"
        assert meta["dedup"]["unarchived_at"]

    async def test_unarchive_dry_run_changes_nothing(self, sqlite_store, mock_chroma):
        (old,) = await _seed(sqlite_store, "x")
        await sqlite_store.update_memory(old, is_archived=True)
        report = await dd.unarchive_memory(sqlite_store, mock_chroma, old, dry_run=True)
        assert report["restored"] is False
        assert (await sqlite_store.get_memory(old)).is_archived
        mock_chroma.add_memory.assert_not_called()

    async def test_unarchive_missing_or_active(self, sqlite_store, mock_chroma):
        (active,) = await _seed(sqlite_store, "active")
        r = await dd.unarchive_memory(sqlite_store, mock_chroma, active)
        assert r["restored"] is False and r["reason"] == "not archived"
        r = await dd.unarchive_memory(sqlite_store, mock_chroma, 999999)
        assert r["restored"] is False and r["reason"] == "not found"


# ---------------------------------------------------------------------------
# Router: response_format reaches the client as `format`
# ---------------------------------------------------------------------------

class TestRouterResponseFormat:

    @pytest.fixture
    def router(self):
        from blipshell.llm.endpoints import EndpointManager
        from blipshell.llm.router import LLMRouter
        from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig
        cfg = [EndpointConfig(name="local", url="http://localhost:11434", provider="ollama",
                              roles=["reasoning"], priority=1, max_concurrent=1)]
        models = ModelsConfig(reasoning="qwen3:14b", embedding="e")
        r = LLMRouter(models, EndpointManager(cfg, LLMConfig()), pii_enabled=False)
        r._gated_generate = AsyncMock(return_value='{"action": "ADD"}')
        return r

    async def test_format_forwarded(self, router):
        out = await router.generate("reasoning", "p", system="s", think=False,
                                    response_format=dd.MEMORY_ACTION_SCHEMA)
        assert out == '{"action": "ADD"}'
        gen_kwargs = router._gated_generate.await_args.args[-1]
        assert gen_kwargs["format"] == dd.MEMORY_ACTION_SCHEMA
        assert gen_kwargs["think"] is False

    async def test_format_absent_by_default(self, router):
        await router.generate("reasoning", "p")
        gen_kwargs = router._gated_generate.await_args.args[-1]
        assert "format" not in gen_kwargs
