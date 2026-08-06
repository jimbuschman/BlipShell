"""Memory consolidation: throughput, and not destroying things while merging.

Consolidation managed ~20 memories a night against a 17K corpus because it
re-embedded every candidate through Ollama — one HTTP round trip per check.
It also hard-deleted losers, which cascaded their entity edges and mentions
away, violating the archive-never-delete mandate at the edge level
(deep-dive 2026-08-04).

Real SQLite, real sqlite-vec, stubbed embeddings.
"""

import struct
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from blipshell.memory.consolidation import MemoryConsolidator
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.vector_store import VectorStore
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import Memory

DIM = 8


def _vec(*first):
    v = list(first) + [0.0] * (DIM - len(first))
    return v


def _orthogonal(i):
    """A unit vector on axis i. Scaling one axis (0,0,1) vs (0,0,2) gives
    PARALLEL vectors — cosine 1.0 — not distinct ones."""
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


class _ExplodingEmbedder:
    """Any embedding call is a regression — consolidation must use the vectors
    already stored, never ask Ollama for another one."""

    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        raise AssertionError(
            "consolidation embedded text — it should query with the STORED vector"
        )


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "consolidate.db")
    sqlite = SQLiteStore(db_path)
    await sqlite.initialize()

    vectors = VectorStore(db_path, embedding_dim=DIM)
    vectors.initialize()
    vectors._embed = _ExplodingEmbedder()
    vectors._ollama_client = MagicMock()

    yield sqlite, vectors

    vectors.close()
    await sqlite.close()


async def _add(sqlite, vectors, session_id, *, content, summary, vec,
               importance=0.5, rank=3, age_days=0, access=0):
    mem = Memory(
        session_id=session_id, role="user", content=content, summary=summary,
        importance=importance, rank=rank,
        timestamp=datetime.now(timezone.utc) - timedelta(days=age_days),
    )
    mid = await sqlite.create_memory(mem)
    if access:
        await sqlite.update_memory(mid, access_count=access)
    with vectors._lock:
        vectors._conn.execute(
            "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
            [mid, struct.pack(f"{DIM}f", *vec)],
        )
        vectors._conn.commit()
    return mid


def _consolidator(sqlite, vectors, **overrides):
    cfg = MemoryConfig(
        consolidation_similarity=0.95,
        consolidation_batch_size=100,
        **overrides,
    )
    return MemoryConsolidator(sqlite, vectors, cfg)


class TestNoEmbeddingCalls:
    async def test_consolidation_never_embeds(self, store):
        """The whole throughput fix. The stub raises if touched."""
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        await _add(sqlite, vectors, sid, content="a", summary="a", vec=_vec(1.0))
        await _add(sqlite, vectors, sid, content="b", summary="b", vec=_vec(0.0, 1.0))

        stats = await _consolidator(sqlite, vectors).consolidate_batch()

        assert vectors._embed.calls == 0
        assert stats["checked"] == 2
        assert stats["errors"] == 0


class TestMerging:
    async def test_near_duplicates_are_merged(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        keep = await _add(sqlite, vectors, sid, content="dup A", summary="dup A",
                          vec=_vec(1.0), importance=0.9)
        drop = await _add(sqlite, vectors, sid, content="dup B", summary="dup B",
                          vec=_vec(1.0), importance=0.2)

        stats = await _consolidator(sqlite, vectors).consolidate_batch()

        assert stats["merged"] == 1
        assert (await sqlite.get_memory(drop)).is_archived is True
        assert (await sqlite.get_memory(keep)).is_archived is False

    async def test_distinct_memories_are_left_alone(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        a = await _add(sqlite, vectors, sid, content="a", summary="a", vec=_vec(1.0))
        b = await _add(sqlite, vectors, sid, content="b", summary="b", vec=_vec(0.0, 1.0))

        stats = await _consolidator(sqlite, vectors).consolidate_batch()

        assert stats["merged"] == 0
        assert (await sqlite.get_memory(a)).is_archived is False
        assert (await sqlite.get_memory(b)).is_archived is False

    async def test_higher_importance_wins(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        weak = await _add(sqlite, vectors, sid, content="x", summary="x",
                          vec=_vec(1.0), importance=0.1)
        strong = await _add(sqlite, vectors, sid, content="y", summary="y",
                            vec=_vec(1.0), importance=0.9)

        await _consolidator(sqlite, vectors).consolidate_batch()

        assert (await sqlite.get_memory(weak)).is_archived is True
        assert (await sqlite.get_memory(strong)).is_archived is False

    async def test_access_counts_are_summed_and_longer_summary_kept(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        winner = await _add(sqlite, vectors, sid, content="w", summary="short",
                            vec=_vec(1.0), importance=0.9, access=3)
        await _add(sqlite, vectors, sid, content="l",
                   summary="a considerably longer and more informative summary",
                   vec=_vec(1.0), importance=0.1, access=4)

        await _consolidator(sqlite, vectors).consolidate_batch()

        kept = await sqlite.get_memory(winner)
        assert kept.access_count == 7
        assert "considerably longer" in kept.summary


class TestArchiveNeverDelete:
    """The mandate, at the edge level."""

    async def test_loser_row_survives(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        await _add(sqlite, vectors, sid, content="a", summary="a",
                   vec=_vec(1.0), importance=0.9)
        drop = await _add(sqlite, vectors, sid, content="b", summary="b",
                          vec=_vec(1.0), importance=0.1)

        await _consolidator(sqlite, vectors).consolidate_batch()

        assert await sqlite.get_memory(drop) is not None, "loser was deleted"

    async def test_entity_edges_survive_a_merge(self, store):
        """Deleting cascaded entity_relationships away — a memory merge could
        destroy graph structure unrelated to the duplication."""
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        await _add(sqlite, vectors, sid, content="a", summary="a",
                   vec=_vec(1.0), importance=0.9)
        drop = await _add(sqlite, vectors, sid, content="b", summary="b",
                          vec=_vec(1.0), importance=0.1)

        subj = await sqlite.get_or_create_entity("user", "person")
        obj = await sqlite.get_or_create_entity("acme", "organization")
        rel = await sqlite.create_entity_relationship_temporal(
            subj, "works_at", obj, memory_id=drop,
        )
        await sqlite.create_entity_mention(subj, drop)
        assert rel is not None

        await _consolidator(sqlite, vectors).consolidate_batch()

        active = await sqlite.get_active_relationships_for_entity(subj)
        assert any(r["id"] == rel for r in active), (
            "merging a duplicate destroyed an entity edge"
        )

    async def test_archived_loser_is_invisible_to_search(self, store):
        """Archiving is only safe because search excludes archived rows."""
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        await _add(sqlite, vectors, sid, content="kangaroo facts",
                   summary="kangaroo facts", vec=_vec(1.0), importance=0.9)
        drop = await _add(sqlite, vectors, sid, content="kangaroo trivia",
                          summary="kangaroo trivia", vec=_vec(1.0), importance=0.1)

        await _consolidator(sqlite, vectors).consolidate_batch()

        hits = await sqlite.search_fts("kangaroo")
        assert all(h["id"] != drop for h in hits)


class TestResumability:
    async def test_time_budget_stops_early_and_reports_it(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        for i in range(5):
            await _add(sqlite, vectors, sid, content=f"m{i}", summary=f"m{i}",
                       vec=_orthogonal(i))

        c = _consolidator(sqlite, vectors)
        stats = await c.consolidate_batch(time_budget_seconds=-1)   # already expired

        assert stats["stopped_early"] is True
        assert stats["checked"] == 0

    async def test_unchecked_memories_are_picked_up_next_run(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        for i in range(4):
            await _add(sqlite, vectors, sid, content=f"m{i}", summary=f"m{i}",
                       vec=_orthogonal(i))

        c = _consolidator(sqlite, vectors)
        await c.consolidate_batch(time_budget_seconds=-1)      # does nothing
        remaining = await sqlite.get_unconsolidated_memory_ids(limit=100)
        assert len(remaining) == 4, "work was lost when the budget expired"

        stats = await c.consolidate_batch()                     # no budget
        assert stats["checked"] == 4
        assert await sqlite.get_unconsolidated_memory_ids(limit=100) == []

    async def test_memory_without_a_vector_is_not_marked_consolidated(self, store):
        """Otherwise a later vector backfill would never get it checked."""
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        mem = Memory(session_id=sid, role="user", content="no vector",
                     summary="no vector", timestamp=datetime.now(timezone.utc))
        mid = await sqlite.create_memory(mem)      # deliberately not embedded

        stats = await _consolidator(sqlite, vectors).consolidate_batch()

        assert stats["no_vector"] == 1
        assert mid in await sqlite.get_unconsolidated_memory_ids(limit=100)


class TestDryRun:
    async def test_dry_run_reports_without_mutating(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        await _add(sqlite, vectors, sid, content="a", summary="a",
                   vec=_vec(1.0), importance=0.9)
        drop = await _add(sqlite, vectors, sid, content="b", summary="b",
                          vec=_vec(1.0), importance=0.1)

        c = _consolidator(sqlite, vectors, consolidation_dry_run=True)
        stats = await c.consolidate_batch()

        assert stats["merged"] == 1
        assert stats.get("dry_run") is True
        assert (await sqlite.get_memory(drop)).is_archived is False
        # and nothing was marked consolidated, so a real run still sees them
        assert len(await sqlite.get_unconsolidated_memory_ids(limit=100)) == 2

    async def test_dry_run_findings_are_visible_without_log_config(self, store):
        """A dry run whose output only went to logger.info printed NOTHING at
        the CLI's default WARNING level — it would read as "no duplicates
        found", the opposite of the truth. Findings must come back in the
        stats and through on_status."""
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        await _add(sqlite, vectors, sid, content="a", summary="keep this one",
                   vec=_vec(1.0), importance=0.9)
        drop = await _add(sqlite, vectors, sid, content="b", summary="drop this one",
                          vec=_vec(1.0), importance=0.1)

        messages = []
        c = _consolidator(sqlite, vectors, consolidation_dry_run=True)
        stats = await c.consolidate_batch(on_status=messages.append)

        assert len(stats["would_merge"]) == 1
        proposal = stats["would_merge"][0]
        assert proposal["loser_id"] == drop
        assert "drop this one" in proposal["loser"]
        assert "keep this one" in proposal["winner"]
        assert any("would merge" in m for m in messages), (
            "nothing surfaced through on_status — the dry run would look empty"
        )

    async def test_live_run_reports_no_proposals_key(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        await _add(sqlite, vectors, sid, content="a", summary="a", vec=_vec(1.0))
        stats = await _consolidator(sqlite, vectors).consolidate_batch()
        assert "would_merge" not in stats


class TestNoRepeatedArchiving:
    """A memory that loses a merge is done — it must not be folded into a
    second winner as well.

    The first dry run over the real corpus (2026-08-06) showed #18177
    scheduled to absorb one memory and then be archived into FOUR different
    winners, copying its tags and access count into each. That's content
    duplication dressed up as consolidation.
    """

    async def test_loser_is_not_merged_into_multiple_winners(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        # Three memories on one vector; the weak one loses to whichever
        # strong memory is considered first and must then be left alone.
        weak = await _add(sqlite, vectors, sid, content="weak", summary="weak",
                          vec=_vec(1.0), importance=0.1, access=5)
        strong_a = await _add(sqlite, vectors, sid, content="A", summary="A",
                              vec=_vec(1.0), importance=0.8, access=0)
        strong_b = await _add(sqlite, vectors, sid, content="B", summary="B",
                              vec=_vec(1.0), importance=0.9, access=0)

        await _consolidator(sqlite, vectors).consolidate_batch()

        a = await sqlite.get_memory(strong_a)
        b = await sqlite.get_memory(strong_b)
        # The weak memory's access count may land on exactly one winner,
        # never on both.
        absorbed = [m for m in (a, b) if (m.access_count or 0) >= 5]
        assert len(absorbed) <= 1, (
            "an archived memory was folded into more than one winner"
        )
        assert (await sqlite.get_memory(weak)).is_archived is True

    async def test_a_memory_archived_early_stops_being_processed(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        loser = await _add(sqlite, vectors, sid, content="l", summary="l",
                           vec=_vec(1.0), importance=0.1)
        for i in range(3):
            await _add(sqlite, vectors, sid, content=f"w{i}", summary=f"w{i}",
                       vec=_vec(1.0), importance=0.9 + i * 0.01)

        stats = await _consolidator(sqlite, vectors).consolidate_batch()

        archived = [
            m for m in [await sqlite.get_memory(i) for i in range(1, 5)]
            if m and m.is_archived
        ]
        # Exactly one archive event per losing memory, not one per neighbour.
        assert stats["merged"] == len(archived)


class TestLoopTerminationSignal:
    """`nightly --job consolidate --loop` decides whether to keep going from
    the job's stats. Consolidation usually merges NOTHING on a given batch —
    at a correct threshold most memories aren't duplicates — so progress has
    to be signalled by `checked`, not `merged`. Keying off `merged` alone made
    --loop stop after one pass and print "nothing left to process" with
    thousands of memories still unexamined."""

    async def test_a_batch_with_no_merges_still_reports_progress(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        for i in range(3):
            await _add(sqlite, vectors, sid, content=f"m{i}", summary=f"m{i}",
                       vec=_orthogonal(i))

        stats = await _consolidator(sqlite, vectors).consolidate_batch()

        assert stats["merged"] == 0
        assert stats["checked"] == 3, (
            "a zero-merge batch reported no progress — --loop would stop early"
        )

    async def test_checked_is_in_the_cli_progress_keys(self):
        """Pins the wiring: the CLI must treat `checked` as progress."""
        import inspect
        from blipshell.ui import cli

        # nightly_cmd is a Click Command; the function is its .callback
        src = inspect.getsource(cli.nightly_cmd.callback)
        start = src.index('"resummarized"')
        keys_block = src[start:src.index("):", start)]
        assert '"checked"' in keys_block, (
            f"'checked' missing from the loop's progress keys: {keys_block}"
        )

    async def test_pool_shrinks_each_pass_so_the_loop_terminates(self, store):
        sqlite, vectors = store
        sid = await sqlite.create_session("s")
        for i in range(4):
            await _add(sqlite, vectors, sid, content=f"m{i}", summary=f"m{i}",
                       vec=_orthogonal(i))

        c = _consolidator(sqlite, vectors)
        first = await c.consolidate_batch()
        second = await c.consolidate_batch()

        assert first["checked"] == 4
        assert second["checked"] == 0, "consolidated memories were re-checked"
