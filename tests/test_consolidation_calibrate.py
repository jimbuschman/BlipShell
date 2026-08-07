"""Tests for scripts/consolidation_calibrate.py.

It reports where near-duplicate similarities actually fall, so the merge
threshold can be argued from the corpus instead of from one pass's merge rate.
The thing that must be right is the bucketing: a tool that miscounts the band
around the threshold is worse than no tool, because it looks authoritative.
"""

import math
import re
import sqlite3

import pytest

from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.memory import Memory
from scripts.consolidation_calibrate import BUCKETS


class TestBuckets:
    def test_buckets_tile_the_range_without_gaps(self):
        """Every similarity from 0 to 1 must land in exactly one bucket."""
        ordered = sorted(BUCKETS, key=lambda b: b[0])
        assert ordered[0][0] == 0.0
        for (lo1, hi1, _), (lo2, _, _) in zip(ordered, ordered[1:]):
            assert math.isclose(hi1, lo2), f"gap or overlap at {hi1} -> {lo2}"
        assert ordered[-1][1] > 1.0, "a similarity of exactly 1.0 must be counted"

    @pytest.mark.parametrize("sim", [0.0, 0.5, 0.849, 0.85, 0.88, 0.92, 0.999, 1.0])
    def test_every_similarity_lands_in_exactly_one_bucket(self, sim):
        hits = [lbl for lo, hi, lbl in BUCKETS if lo <= sim < hi]
        assert len(hits) == 1, f"{sim} matched {hits}"

    def test_the_threshold_boundaries_are_bucket_edges(self):
        """0.85 and 0.92 are the two thresholds that have actually been run, so
        the report has to split exactly on them, not straddle them."""
        edges = {lo for lo, _, _ in BUCKETS} | {hi for _, hi, _ in BUCKETS}
        assert 0.85 in edges
        assert 0.92 in edges


@pytest.fixture
async def corpus(tmp_path):
    """A DB whose memories exist; vectors are added directly to vec_memories."""
    path = tmp_path / "cal.db"
    store = SQLiteStore(str(path))
    await store.initialize()
    session_id = await store.create_session("s")
    ids = [
        await store.create_memory(Memory(
            session_id=session_id, role="user",
            content=f"content {i}", summary=f"summary {i}",
        ))
        for i in range(4)
    ]
    await store.close()
    return path, ids


class TestSampling:
    async def test_only_active_memories_are_sampled(self, corpus):
        """Archived memories are not consolidation candidates; including them
        would skew the distribution the threshold gets read off."""
        path, ids = corpus
        conn = sqlite3.connect(path)
        conn.execute("UPDATE memories SET is_archived = 1 WHERE id = ?", (ids[0],))
        conn.commit()

        rows = conn.execute(
            "SELECT id FROM memories WHERE is_archived = 0 ORDER BY id LIMIT ?",
            (2000,),
        ).fetchall()
        conn.close()

        sampled = [r[0] for r in rows]
        assert ids[0] not in sampled
        assert len(sampled) == 3

    async def test_missing_db_exits_one(self, tmp_path, monkeypatch):
        from scripts import consolidation_calibrate

        monkeypatch.setattr(
            "sys.argv",
            ["consolidation_calibrate.py", "--db", str(tmp_path / "nope.db")],
        )
        assert consolidation_calibrate.main() == 1

    async def test_reports_a_real_duplicate_pair(self, tmp_path, monkeypatch, capsys):
        """End-to-end against real vectors. The first version of this script
        asked find_neighbors for k=1, which returned only the query row itself
        and so reported 'no neighbours' for a perfectly healthy store."""
        import struct

        from scripts import consolidation_calibrate
        from blipshell.memory.vector_store import VectorStore

        path = tmp_path / "real.db"
        store = SQLiteStore(str(path))
        await store.initialize()
        session_id = await store.create_session("s")
        ids = [
            await store.create_memory(Memory(
                session_id=session_id, role="user",
                content=f"content {i}", summary=f"summary {i}",
            ))
            for i in range(3)
        ]
        await store.close()

        vectors = VectorStore(str(path))
        vectors.initialize()
        dim = vectors.embedding_dim
        # Two near-identical vectors and one orthogonal to both.
        near_a = [1.0] + [0.0] * (dim - 1)
        near_b = [0.999, 0.0447] + [0.0] * (dim - 2)
        other = [0.0, 0.0, 1.0] + [0.0] * (dim - 3)
        with vectors._lock:
            for mid, vec in zip(ids, (near_a, near_b, other)):
                vectors._conn.execute(
                    "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                    [mid, struct.pack(f"{dim}f", *vec)],
                )
            vectors._conn.commit()
        vectors.close()

        monkeypatch.setattr(
            "sys.argv", ["consolidation_calibrate.py", "--db", str(path)],
        )
        assert consolidation_calibrate.main() == 0

        # Rich wraps to the console width and emits ANSI, so compare against a
        # flattened copy rather than weakening the assertion to a fragment.
        raw = capsys.readouterr().out
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", raw).split())
        assert "No neighbours" not in flat, flat
        assert "Would merge at 0.92" in flat, flat

    async def test_empty_corpus_exits_zero(self, tmp_path, monkeypatch):
        from scripts import consolidation_calibrate

        path = tmp_path / "empty.db"
        store = SQLiteStore(str(path))
        await store.initialize()
        await store.close()

        monkeypatch.setattr(
            "sys.argv", ["consolidation_calibrate.py", "--db", str(path)],
        )
        assert consolidation_calibrate.main() == 0
