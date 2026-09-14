"""Blank rows cannot block the vector backfill.

A blank row has no vector and never will: it stays in the `v.rowid IS NULL`
set for ever. The filter used to run in PYTHON, after `LIMIT`, so a window
that happened to be all blanks returned `processed=0` — which stops `drain()`
cleanly, and stops it BEFORE the first embeddable row. With more blank rows
than the batch size, no valid record was ever reached: a permanent starvation
that reports itself as "nothing to do".

Also pinned here: `COALESCE(content, summary)` fell back only on NULL, so a
memory with `content = ''` and a perfectly good summary resolved to `''` and
was discarded as unembeddable.
"""

import sqlite3
from unittest.mock import MagicMock

import pytest

from blipshell.memory.vector_store import VectorStore


EMBED_DIM = 8


@pytest.fixture
def store(tmp_path):
    """A real VectorStore on a temp DB with a scripted embedder."""
    path = str(tmp_path / "v.db")

    # The source tables the backfill reads. Created before the store opens so
    # its own schema work sees them.
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY, session_id INTEGER, "
        "role TEXT, content TEXT, summary TEXT, is_archived INTEGER DEFAULT 0)"
    )
    conn.commit()
    conn.close()

    vs = VectorStore(db_path=path, embedding_model="fake", ollama_url="http://x",
                     embedding_dim=EMBED_DIM)
    vs.initialize()

    # Never reach a network. One vector per input, in order.
    client = MagicMock()

    def embed(model, input):
        texts = [input] if isinstance(input, str) else list(input)
        assert all(t and t.strip() for t in texts), f"blank text reached the embedder: {texts!r}"
        return {"embeddings": [[float(len(t))] * EMBED_DIM for t in texts]}

    client.embed = MagicMock(side_effect=embed)
    vs._ollama_client = client
    yield vs
    vs.close()


def _add(store, rows):
    """rows: list of (content, summary). Returns the ids, in order."""
    ids = []
    for content, summary in rows:
        cur = store._conn.execute(
            "INSERT INTO memories (session_id, role, content, summary) VALUES (1, 'user', ?, ?)",
            (content, summary),
        )
        ids.append(cur.lastrowid)
    store._conn.commit()
    return ids


def _vector_ids(store):
    return {r[0] for r in store._conn.execute("SELECT rowid FROM vec_memories").fetchall()}


# --- the starvation ---------------------------------------------------------------


def test_a_full_batch_of_blanks_does_not_hide_the_valid_records(store):
    """THE regression: 60 blanks ahead of 3 real rows, batch limit 50."""
    blank_ids = _add(store, [("", "") for _ in range(60)])
    good_ids = _add(store, [("real one", None), ("real two", None), ("real three", None)])

    stats = store.backfill_missing_vectors("memories", limit=50)

    assert stats["succeeded"] == 3
    assert stats["failed"] == 0
    assert stats["skipped_blank"] == 60
    assert _vector_ids(store) == set(good_ids)
    assert not (_vector_ids(store) & set(blank_ids))


def test_whitespace_only_rows_count_as_blank(store):
    """Tabs and newlines, not just empty strings (memory/blank_text.py)."""
    _add(store, [(w, w) for w in ("", " ", "\t", "\n", " \t\r\n ", "\u00a0")])
    good = _add(store, [("real", None)])

    stats = store.backfill_missing_vectors("memories", limit=50)

    assert stats["skipped_blank"] == 6
    assert stats["succeeded"] == 1
    assert _vector_ids(store) == set(good)


def test_mixed_batch_embeds_the_good_and_skips_the_blank(store):
    ids = _add(store, [
        ("alpha", None), ("", ""), ("beta", None), ("   ", "  "),
        ("gamma", None),
    ])
    good = [ids[0], ids[2], ids[4]]

    stats = store.backfill_missing_vectors("memories", limit=50)

    assert stats["processed"] == 3
    assert stats["succeeded"] == 3
    assert stats["skipped_blank"] == 2
    assert _vector_ids(store) == set(good)


def test_rows_outside_the_active_filter_are_not_counted_as_skipped(store):
    """Pre-existing scope, stated: the memories filter excludes a row whose
    content AND summary are both NULL, exactly as it excludes an archived one.
    Out of scope is not the same as skipped, and the figure says so."""
    _add(store, [(None, None), (None, None)])
    good = _add(store, [("real", None)])

    stats = store.backfill_missing_vectors("memories", limit=50)

    assert stats["succeeded"] == 1
    assert "skipped_blank" not in stats
    assert _vector_ids(store) == set(good)


def test_an_entirely_blank_collection_terminates_and_says_so(store):
    _add(store, [("", "") for _ in range(10)])

    stats = store.backfill_missing_vectors("memories", limit=50)

    assert stats["processed"] == 0
    assert stats["succeeded"] == 0
    assert stats["skipped_blank"] == 10
    assert _vector_ids(store) == set()


def test_an_empty_collection_reports_nothing_skipped(store):
    stats = store.backfill_missing_vectors("memories", limit=50)
    assert stats == {"processed": 0, "succeeded": 0, "failed": 0}


def test_draining_in_batches_reaches_every_valid_record(store):
    """The loop shape the real callers use — it must terminate AND finish."""
    _add(store, [("", "") for _ in range(120)])
    good = _add(store, [(f"real {i}", None) for i in range(25)])

    embedded, batches = 0, 0
    while True:
        stats = store.backfill_missing_vectors("memories", limit=10)
        batches += 1
        embedded += stats["succeeded"]
        if stats["processed"] == 0:
            break
        assert batches < 50, "drain did not terminate"

    assert embedded == 25
    assert _vector_ids(store) == set(good)


def test_skipped_blank_is_the_whole_backlog_not_the_window(store):
    """A figure that changed with the batch size would be unreadable."""
    _add(store, [("", "") for _ in range(40)])
    _add(store, [("real", None)])

    small = store.backfill_missing_vectors("memories", limit=1)
    assert small["skipped_blank"] == 40


# --- blank content falls back to the summary --------------------------------------


def test_blank_content_falls_back_to_the_summary(store):
    """COALESCE only falls back on NULL; '' resolved to '' and was dropped."""
    ids = _add(store, [
        ("", "the summary carries the text"),
        ("   ", "another good summary"),
        (None, "null content, good summary"),
        ("content wins", "summary loses"),
    ])

    stats = store.backfill_missing_vectors("memories", limit=50)

    assert stats["succeeded"] == 4
    assert "skipped_blank" not in stats
    assert _vector_ids(store) == set(ids)

    embedded = [c.kwargs["input"] for c in store._ollama_client.embed.call_args_list]
    flat = [t for batch in embedded for t in (batch if isinstance(batch, list) else [batch])]
    assert "the summary carries the text" in flat
    assert "content wins" in flat


def test_a_row_blank_in_both_columns_is_skipped_not_embedded(store):
    _add(store, [("", ""), ("  ", None), (None, "\t")])

    stats = store.backfill_missing_vectors("memories", limit=50)

    assert stats["skipped_blank"] == 3
    assert stats["processed"] == 0
    store._ollama_client.embed.assert_not_called()


# --- positional alignment ---------------------------------------------------------


def test_a_short_embedding_reply_fails_loudly_instead_of_misaligning(store):
    """The silent cross-wiring: one dropped input shifts every later vector."""
    ids = _add(store, [("alpha", None), ("beta", None), ("gamma", None)])

    def short(model, input):
        texts = [input] if isinstance(input, str) else list(input)
        return {"embeddings": [[1.0] * EMBED_DIM for _ in texts[:-1]]}

    store._ollama_client.embed = MagicMock(side_effect=short)

    stats = store.backfill_missing_vectors("memories", limit=50)

    assert stats["succeeded"] == 0
    assert stats["failed"] == 3
    assert _vector_ids(store) == set()
    assert set(ids)  # nothing was written under the wrong rowid


def test_ids_and_vectors_stay_paired_across_chunk_boundaries(store):
    """_embed_batch chunks at 32; the outer batch is 50."""
    rows = [(f"text number {i:03d}", None) for i in range(50)]
    ids = _add(store, rows)

    stats = store.backfill_missing_vectors("memories", limit=50)
    assert stats["succeeded"] == 50

    # The scripted embedder encodes len(text) into every component, so each
    # stored vector identifies the text it came from.
    import struct
    for (content, _), item_id in zip(rows, ids, strict=True):
        blob = store._conn.execute(
            "SELECT embedding FROM vec_memories WHERE rowid = ?", (item_id,)
        ).fetchone()[0]
        got = struct.unpack(f"{EMBED_DIM}f", blob)
        assert got[0] == pytest.approx(float(len(content))), f"row {item_id} holds another row's vector"
