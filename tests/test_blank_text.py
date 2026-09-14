"""The SQL and Python definitions of "blank" are the same definition.

The repair selected rows with SQLite one-argument `TRIM`, which strips
SPACES ONLY. Every Python check used `.strip()`, which strips all whitespace.
A summary of a single tab or newline was therefore blank to the embedder
(which refuses it by name) and invisible to `blipshell repair
--blank-summaries`, the command whose job is to find exactly those rows.
"""

import sqlite3

import pytest

from blipshell.memory.blank_text import (
    WHITESPACE,
    blank_sql,
    coalesce_nonblank_sql,
    first_nonblank,
    is_blank,
    nonblank_sql,
)

# Every shape the two definitions used to disagree about.
BLANK_VALUES = [
    None, "", " ", "  ", "\t", "\n", "\r\n", "\v", "\f",
    " \t\n ", "\t\t", "\n\n\n", "\u00a0", "\u2003", "\u3000",
]
MEANINGFUL_VALUES = [
    "x", "hello", " padded ", "\tleading tab", "trailing newline\n",
    " \n mixed whitespace around real text \t ", "0", "SKIP",
]


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, content TEXT, summary TEXT)")
    yield c
    c.close()


def _sql_is_blank(conn, value) -> bool:
    conn.execute("DELETE FROM t")
    conn.execute("INSERT INTO t (id, summary) VALUES (1, ?)", (value,))
    return bool(conn.execute(
        f"SELECT 1 FROM t WHERE {blank_sql('summary')}").fetchone())


# --- the two definitions agree ---------------------------------------------------


@pytest.mark.parametrize("value", BLANK_VALUES)
def test_blank_values_are_blank_in_python_and_in_sql(conn, value):
    assert is_blank(value) is True
    assert _sql_is_blank(conn, value) is True


@pytest.mark.parametrize("value", MEANINGFUL_VALUES)
def test_meaningful_values_survive_in_python_and_in_sql(conn, value):
    assert is_blank(value) is False
    assert _sql_is_blank(conn, value) is False


def test_sql_and_python_agree_on_every_whitespace_character(conn):
    """Generated, not enumerated: no character can slip between them."""
    for ch in WHITESPACE:
        assert is_blank(ch), repr(ch)
        assert _sql_is_blank(conn, ch), repr(ch)
        assert not is_blank(f"a{ch}b")
        assert not _sql_is_blank(conn, f"a{ch}b")


def test_nonblank_sql_is_the_exact_negation(conn):
    for value in BLANK_VALUES + MEANINGFUL_VALUES:
        conn.execute("DELETE FROM t")
        conn.execute("INSERT INTO t (id, summary) VALUES (1, ?)", (value,))
        blank = bool(conn.execute(
            f"SELECT 1 FROM t WHERE {blank_sql('summary')}").fetchone())
        nonblank = bool(conn.execute(
            f"SELECT 1 FROM t WHERE {nonblank_sql('summary')}").fetchone())
        assert blank is not nonblank, repr(value)


def test_the_old_predicate_is_what_this_replaces(conn):
    """Pin the defect: one-argument TRIM misses everything but spaces."""
    conn.execute("DELETE FROM t")
    conn.execute("INSERT INTO t (id, summary) VALUES (1, ?)", ("\n",))
    old = conn.execute(
        "SELECT 1 FROM t WHERE TRIM(COALESCE(summary, '')) = ''").fetchone()
    assert old is None, "SQLite one-arg TRIM strips spaces only — the premise"
    assert _sql_is_blank(conn, "\n") is True


# --- blank, not null, is the fallback condition ----------------------------------


def test_first_nonblank_skips_blank_not_just_none():
    assert first_nonblank(None, "summary") == "summary"
    assert first_nonblank("", "summary") == "summary"
    assert first_nonblank("   ", "summary") == "summary"
    assert first_nonblank("\t\n", "summary") == "summary"
    assert first_nonblank("content", "summary") == "content"
    assert first_nonblank(None, None) is None
    assert first_nonblank("", "  ") is None


def test_coalesce_nonblank_sql_falls_back_on_blank_not_only_null(conn):
    """COALESCE(content, summary) resolved '' to '' and threw the row away."""
    rows = [
        (1, "real content", "a summary"),
        (2, "", "fallback summary"),
        (3, "   ", "fallback summary"),
        (4, "\t\n", "fallback summary"),
        (5, None, "fallback summary"),
        (6, None, None),
        (7, " padded content ", None),
    ]
    conn.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
    expr = coalesce_nonblank_sql("content", "summary")
    got = dict(conn.execute(f"SELECT id, {expr} FROM t").fetchall())
    assert got == {
        1: "real content",
        2: "fallback summary",
        3: "fallback summary",
        4: "fallback summary",
        5: "fallback summary",
        6: None,
        7: " padded content ",  # untouched: the embedder sees it as stored
    }


def test_old_coalesce_kept_the_empty_string(conn):
    """Pin the defect this replaces."""
    conn.execute("INSERT INTO t VALUES (2, '', 'fallback summary')")
    old = conn.execute("SELECT COALESCE(content, summary) FROM t").fetchone()[0]
    assert old == "", "COALESCE only falls back on NULL — the premise"


# --- the shared definition actually governs the call sites -----------------------


def test_repair_selection_uses_the_shared_predicate():
    from blipshell.memory.processor import BLANK_SUMMARY_SQL
    assert blank_sql("summary") in BLANK_SUMMARY_SQL


@pytest.mark.asyncio
async def test_repair_finds_a_whitespace_only_summary(tmp_path):
    """End to end through the real selector: a newline summary is findable."""
    from blipshell.memory.processor import find_blank_summaries
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.models.memory import Memory

    store = SQLiteStore(str(tmp_path / "t.db"))
    await store.initialize()
    session_id = await store.create_session(title="t")
    ids = {}
    for label, summary in [("newline", "\n"), ("tab", "\t"),
                           ("spaces", "   "), ("mixed", " \t\r\n "),
                           ("empty", ""), ("real", "a real summary")]:
        ids[label] = await store.create_memory(Memory(
            session_id=session_id, role="user", summary=summary,
            content=f"content for {label}",
        ))

    found = {r["id"] for r in await find_blank_summaries(store, limit=100)}
    await store.close()

    assert ids["real"] not in found
    for label in ("newline", "tab", "spaces", "mixed", "empty"):
        assert ids[label] in found, f"{label} summary was not selected"


def test_coalesce_nonblank_sql_works_with_a_single_column(conn):
    """SQLite rejects a one-argument COALESCE, and four of the five vector
    collections have no fallback column at all."""
    conn.executemany("INSERT INTO t VALUES (?, ?, ?)", [
        (1, "real", None), (2, "", None), (3, "  ", None), (4, None, None),
    ])
    expr = coalesce_nonblank_sql("content")
    got = dict(conn.execute(f"SELECT id, {expr} FROM t").fetchall())
    assert got == {1: "real", 2: None, 3: None, 4: None}


def test_coalesce_nonblank_sql_refuses_zero_columns():
    with pytest.raises(ValueError):
        coalesce_nonblank_sql()
