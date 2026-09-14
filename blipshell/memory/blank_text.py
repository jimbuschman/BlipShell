"""One definition of "blank text", in Python AND in SQL.

There were two. The Python side — `summary_or_raw`, `VectorStore._embed`,
`_embed_batch`, the backfill filter — all asked `not text.strip()`, which
strips every character Python calls whitespace. The SQL side selected rows
with `TRIM(COALESCE(summary, '')) = ''`, and SQLite's one-argument `TRIM`
strips **spaces only** — not a tab, not a newline.

So a summary of "\\n" was blank to the embedder (which refuses it by name)
and INVISIBLE to `blipshell repair --blank-summaries`, the command whose
whole job is to find exactly those rows. The repair fixed ordinary spaces and
left every other whitespace-only summary in place, unfindable and unrendered.

`TRIM(X, Y)` takes a second argument: the set of characters to strip. Given
Python's own whitespace set, the two agree character for character — verified
by `tests/test_blank_text.py`, which compares the SQL predicate against
`str.strip()` over every code point Python calls whitespace.

Python is still the authority: a row selected by SQL is re-checked with
`is_blank` before anything is done to it.
"""

from __future__ import annotations

from typing import Optional

# Exactly the characters `str.strip()` removes. Generated, not hand-listed —
# a hand-listed set is how the SQL and Python definitions drifted apart in the
# first place. (Nothing above U+3000 is whitespace in Python.)
WHITESPACE: str = "".join(chr(c) for c in range(0x3001) if chr(c).isspace())

# The same set as a SQLite expression. `char(X1, X2, ...)` builds a string
# from code points, which keeps the literal readable in a logged query
# instead of embedding raw control characters in it.
WHITESPACE_SQL: str = "char(" + ",".join(
    str(ord(c)) for c in WHITESPACE
) + ")"


def is_blank(value: Optional[str]) -> bool:
    """True when there is no usable text here: None, empty, or whitespace."""
    return not value or not value.strip()


def first_nonblank(*values: Optional[str]) -> Optional[str]:
    """The first value carrying usable text, or None.

    The fallback SQL used `COALESCE(content, summary)`, which only falls back
    on NULL — a row with `content = ''` and a perfectly good summary resolved
    to `''` and was then thrown away as unembeddable. Blankness, not
    nullness, is the condition that matters.
    """
    for value in values:
        if not is_blank(value):
            return value
    return None


def blank_sql(column: str) -> str:
    """SQL predicate: this column holds no usable text.

    Matches `is_blank` exactly, including tabs, newlines and Unicode spaces.
    """
    return f"TRIM(COALESCE({column}, ''), {WHITESPACE_SQL}) = ''"


def nonblank_sql(column: str) -> str:
    """SQL predicate: this column holds usable text."""
    return f"TRIM(COALESCE({column}, ''), {WHITESPACE_SQL}) <> ''"


def coalesce_nonblank_sql(*columns: str) -> str:
    """SQL expression: the first column holding usable text, else NULL.

    The SQL half of `first_nonblank`. Each column yields its OWN value
    (untrimmed — the embedder should see the text as stored) or NULL when it
    is blank, and COALESCE picks the first survivor.
    """
    parts = [
        f"CASE WHEN {nonblank_sql(col)} THEN {col} END" for col in columns
    ]
    return f"COALESCE({', '.join(parts)})"
