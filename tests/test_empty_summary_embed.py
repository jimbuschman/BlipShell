"""An empty summarization reply must never reach the embedder.

Incident 2026-09-11: the live log carried one line —

    [blipshell.memory.processor] ERROR: Dedup check failed (continuing):
    list index out of range

The chain: a summarization call returned "" (no exception — both clients
return "" for a reply with no content), the empty string was written as the
memory's summary, and the dedup step embedded it. Ollama DROPS an empty input
from `/api/embed` and answers `{"embeddings": []}` (measured against the live
daemon), so `response["embeddings"][0]` raised a bare IndexError.

These tests pin all three layers: the empty reply is treated as a failed
summarization, the embedder refuses blank input by name, and a dropped input
can no longer shift a batch's vectors onto the wrong rowids.
"""

import pytest

from blipshell.memory.processor import (
    find_blank_summaries,
    repair_blank_summaries,
    summary_or_raw,
)
from blipshell.memory.vector_store import VectorStore


class FakeEmbedder:
    """Mimics the live Ollama daemon: an empty input is DROPPED, not rejected.

    Measured 2026-09-11 against qwen3-embedding:0.6b —
    ""  -> {"embeddings": []}; " " and "hello" -> one vector each.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list = []

    def embed(self, model: str, input):  # noqa: A002 - ollama's kwarg name
        items = [input] if isinstance(input, str) else list(input)
        self.calls.append(items)
        # First element encodes the text so a caller can prove which vector
        # landed on which row.
        return {"embeddings": [[float(len(t))] * self.dim for t in items if t != ""]}


@pytest.fixture
def vectors(sqlite_store, temp_db_path):
    v = VectorStore(db_path=temp_db_path, embedding_model="fake",
                    ollama_url="http://localhost:1", embedding_dim=8)
    v.initialize()
    v._ollama_client = FakeEmbedder(dim=8)
    yield v
    v.close()


# --- Layer 1: the empty reply is a failed summarization -------------------

class TestSummaryOrRaw:
    def test_keeps_a_real_summary(self):
        assert summary_or_raw("A real summary.", "raw text") == "A real summary."

    def test_empty_reply_falls_back_to_raw_text(self):
        assert summary_or_raw("", "raw text") == "raw text"

    def test_whitespace_only_reply_falls_back_to_raw_text(self):
        assert summary_or_raw("   \n ", "raw text") == "raw text"

    def test_none_falls_back_to_raw_text(self):
        assert summary_or_raw(None, "raw text") == "raw text"


class TestPipelineWithEmptySummary:
    async def test_empty_summary_never_persisted_or_embedded(
        self, memory_processor, sqlite_store, mock_chroma, canned_router,
    ):
        """The exact production shape: summarization returns "", everything
        downstream must still see real text."""
        original = canned_router.generate.side_effect

        def empty_summary(task_type, prompt="", system=None, **kwargs):
            if task_type == "summarization":
                return ""
            return original(task_type, prompt, system, kwargs.get("think"))

        canned_router.generate.side_effect = empty_summary
        text = ("Python profiling with cProfile finds the bottleneck before "
                "you optimize anything, which is the whole point.")
        session_id = await sqlite_store.create_session("Test")

        mem_id = await memory_processor.process_message(
            text=text, role="user", session_id=session_id,
        )

        assert mem_id is not None
        memory = await sqlite_store.get_memory(mem_id)
        assert memory.summary == text          # not "" — FTS and Recall read this
        # Dedup embedded the summary; it must never be blank.
        queries = [c.args[0] for c in mock_chroma.search_memories.call_args_list]
        assert queries, "dedup did not run"
        assert all(q.strip() for q in queries)


# --- Layer 2: the embedder names its own failure --------------------------

class TestEmbedRefusesBlankInput:
    def test_empty_string_raises_value_error_not_index_error(self, vectors):
        with pytest.raises(ValueError, match="empty text"):
            vectors._embed("")

    def test_whitespace_only_raises_value_error(self, vectors):
        with pytest.raises(ValueError, match="empty text"):
            vectors._embed("   \n")

    def test_search_with_blank_query_is_not_an_index_error(self, vectors):
        with pytest.raises(ValueError, match="empty text"):
            vectors.search_memories("")

    def test_backend_returning_nothing_names_the_model(self, vectors):
        vectors._ollama_client.embed = lambda model, input: {"embeddings": []}
        with pytest.raises(RuntimeError, match="returned no vector"):
            vectors._embed("real text")

    def test_real_text_still_embeds(self, vectors):
        assert vectors._embed("hello") == [5.0] * 8


class TestEmbedBatchAlignment:
    def test_blank_member_refused_by_position(self, vectors):
        with pytest.raises(ValueError, match=r"position\(s\) \[1\]"):
            vectors._embed_batch(["a", "", "c"])

    def test_short_reply_refuses_to_misalign(self, vectors):
        vectors._ollama_client.embed = lambda model, input: {
            "embeddings": [[1.0] * 8 for _ in range(len(input) - 1)]
        }
        with pytest.raises(RuntimeError, match="refusing to misalign"):
            vectors._embed_batch(["aa", "bb", "cc"])

    def test_aligned_reply_passes_through(self, vectors):
        got = vectors._embed_batch(["a", "bb", "ccc"])
        assert [v[0] for v in got] == [1.0, 2.0, 3.0]


class TestBackfillSkipsBlankRows:
    async def test_blank_row_does_not_cross_wire_vectors(
        self, vectors, sqlite_store,
    ):
        """A blank text used to be dropped mid-batch, shifting every later
        vector onto the wrong memory id."""
        from blipshell.models.memory import Memory

        session_id = await sqlite_store.create_session("Test")
        ids = {}
        for label, content in (("blank", ""), ("short", "ab"), ("long", "abcdef")):
            mem = Memory(session_id=session_id, role="user",
                         content=content, summary="")
            ids[label] = await sqlite_store.create_memory(mem)

        stats = vectors.backfill_missing_vectors("memories", limit=50)

        # Alignment first: this is the claim. Pre-fix, Ollama dropped the
        # blank input and zip() gave `short` the vector for "abcdef".
        stored = vectors.get_embeddings_by_ids(list(ids.values()))
        assert stored.get(ids["short"], [None])[0] == 2.0   # its OWN text
        assert stored.get(ids["long"], [None])[0] == 6.0
        assert ids["blank"] not in stored
        assert stats["succeeded"] == 2
        assert stats["failed"] == 0

    async def test_blank_row_is_counted_not_silently_dropped(
        self, vectors, sqlite_store,
    ):
        from blipshell.models.memory import Memory

        session_id = await sqlite_store.create_session("Test")
        await sqlite_store.create_memory(
            Memory(session_id=session_id, role="user", content="", summary=""),
        )
        await sqlite_store.create_memory(
            Memory(session_id=session_id, role="user", content="real", summary=""),
        )
        stats = vectors.backfill_missing_vectors("memories", limit=50)
        assert stats["skipped_blank"] == 1
        assert stats["succeeded"] == 1

    async def test_only_blank_rows_reports_nothing_to_do(
        self, vectors, sqlite_store,
    ):
        """`drain` breaks on processed == 0 — a blank row must not spin it."""
        from blipshell.models.memory import Memory

        session_id = await sqlite_store.create_session("Test")
        await sqlite_store.create_memory(
            Memory(session_id=session_id, role="user", content="", summary=""),
        )
        stats = vectors.backfill_missing_vectors("memories", limit=50)
        assert stats["processed"] == 0
        assert stats["skipped_blank"] == 1


# --- Layer 3: the repair for rows written before the fix ------------------

class TestRepairBlankSummaries:
    """`blipshell repair --blank-summaries`. Three rows were found on the live
    corpus (32,099 active memories) — one of them the 2026-09-11 incident."""

    async def _blank_row(self, sqlite_store, session_id, content):
        from blipshell.models.memory import Memory
        return await sqlite_store.create_memory(
            Memory(session_id=session_id, role="assistant",
                   content=content, summary=""),
        )

    async def test_finds_only_blank_active_rows(self, sqlite_store):
        from blipshell.models.memory import Memory

        sid = await sqlite_store.create_session("Test")
        blank = await self._blank_row(sqlite_store, sid, "some real content")
        await sqlite_store.create_memory(
            Memory(session_id=sid, role="user", content="x", summary="has one"))
        archived = await self._blank_row(sqlite_store, sid, "archived content")
        await sqlite_store.update_memory(archived, is_archived=True)

        found = await find_blank_summaries(sqlite_store)
        assert [r["id"] for r in found] == [blank]

    async def test_dry_run_lists_without_calling_the_model(
        self, sqlite_store, canned_router,
    ):
        sid = await sqlite_store.create_session("Test")
        mid = await self._blank_row(sqlite_store, sid, "some real content")
        lines = []

        stats = await repair_blank_summaries(
            sqlite_store, canned_router, dry_run=True, on_status=lines.append)

        assert stats["found"] == 1
        canned_router.generate.assert_not_awaited()   # a shared GPU costs nothing to preview
        assert any(str(mid) in ln for ln in lines)
        assert (await sqlite_store.get_memory(mid)).summary == ""

    async def test_apply_writes_the_new_summary(self, sqlite_store, canned_router):
        sid = await sqlite_store.create_session("Test")
        mid = await self._blank_row(sqlite_store, sid, "content worth summarizing")

        stats = await repair_blank_summaries(
            sqlite_store, canned_router, dry_run=False)

        assert stats["resummarized"] == 1
        summary = (await sqlite_store.get_memory(mid)).summary
        assert summary and summary.strip()

    async def test_repaired_summary_is_searchable(self, sqlite_store, canned_router):
        """FTS indexes the summary through a trigger on UPDATE — prove it fired."""
        sid = await sqlite_store.create_session("Test")
        mid = await self._blank_row(sqlite_store, sid, "content worth summarizing")
        canned_router.generate.side_effect = None
        canned_router.generate.return_value = "Kangaroo telemetry notes."

        await repair_blank_summaries(sqlite_store, canned_router, dry_run=False)

        cursor = await sqlite_store._db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'kangaroo'")
        assert [r[0] for r in await cursor.fetchall()] == [mid]

    async def test_empty_reply_again_falls_back_to_content(
        self, sqlite_store, canned_router,
    ):
        sid = await sqlite_store.create_session("Test")
        mid = await self._blank_row(sqlite_store, sid, "the raw content")
        canned_router.generate.side_effect = None
        canned_router.generate.return_value = ""

        stats = await repair_blank_summaries(
            sqlite_store, canned_router, dry_run=False)

        assert stats["content_fallback"] == 1
        assert (await sqlite_store.get_memory(mid)).summary == "the raw content"

    async def test_skip_verdict_never_archives_the_row(
        self, sqlite_store, canned_router,
    ):
        """At write time SKIP filters a new message; a repair must not remove
        a record that has existed for months."""
        sid = await sqlite_store.create_session("Test")
        mid = await self._blank_row(sqlite_store, sid, "the raw content")
        canned_router.generate.side_effect = None
        canned_router.generate.return_value = "SKIP"

        stats = await repair_blank_summaries(
            sqlite_store, canned_router, dry_run=False)

        memory = await sqlite_store.get_memory(mid)
        assert stats["skip_verdict"] == 1
        assert memory.is_archived is False
        assert memory.summary == "the raw content"

    async def test_model_failure_leaves_the_row_for_a_retry(
        self, sqlite_store, canned_router,
    ):
        """An outage must not rewrite every row as a copy of its own content."""
        sid = await sqlite_store.create_session("Test")
        mid = await self._blank_row(sqlite_store, sid, "the raw content")
        canned_router.generate.side_effect = RuntimeError("endpoint down")

        stats = await repair_blank_summaries(
            sqlite_store, canned_router, dry_run=False)

        assert stats["failed"] == 1
        assert (await sqlite_store.get_memory(mid)).summary == ""
        # Re-runnable: the row is still on the worklist.
        assert [r["id"] for r in await find_blank_summaries(sqlite_store)] == [mid]

    async def test_row_without_content_is_reported_not_touched(
        self, sqlite_store, canned_router,
    ):
        sid = await sqlite_store.create_session("Test")
        mid = await self._blank_row(sqlite_store, sid, "")

        stats = await repair_blank_summaries(
            sqlite_store, canned_router, dry_run=False)

        assert stats == {"found": 1, "resummarized": 0, "content_fallback": 0,
                         "no_content": 1, "failed": 0, "skip_verdict": 0}
        assert (await sqlite_store.get_memory(mid)).summary == ""
