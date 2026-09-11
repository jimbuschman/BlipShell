"""Query-relevant excerpts instead of prefix truncation (V3 Stage B3).

Recall used to cut every memory to its first 1,200 characters. A fact that
FTS or the vector index matched at character 1,400 was retrieved and then
thrown away before the model saw it - the continuity set's
`fact_buried_past_1200_chars` case, and review finding F6.

`excerpt()` keeps the sentence(s) around the lexical matches for the query,
expanding to neighbours while the budget allows, and marks the cut ends with
an ellipsis so the model knows it is looking at a window. With no lexical
match it falls back to the old prefix cut - a vector-only hit still gets the
opening of the memory, which is the best deterministic guess. No model call.
"""

from __future__ import annotations

import re

_STOP = frozenset("""
a an and are as at be but by do does did for from had has have how i if in is it
its me my of on or our so than that the their them then there these they this to
was we were what when where which who why will with would you your yours
""".split())

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9][a-z0-9.\-']*", re.IGNORECASE)

# Below this, one end of a head+tail excerpt carries nothing readable.
MIN_EXCERPT_SIDE = 60


def query_terms(query: str) -> list[str]:
    """Content words of a query: lowercased, stop-words dropped, possessives trimmed."""
    out = []
    for w in _WORD.findall(query.lower()):
        w = w.rstrip(".").removesuffix("'s").rstrip("'")
        if len(w) >= 2 and w not in _STOP and w not in out:
            out.append(w)
    return out


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text)]
    return [p for p in parts if p]


def _hits(sentence: str, terms: list[str]) -> int:
    s = sentence.lower()
    return sum(1 for t in terms if t in s)


def excerpt(text: str, query: str, max_chars: int = 1200, marker: str = "...") -> str:
    """Return at most `max_chars` of `text`, centred on the query's lexical matches.

    - text already within budget: returned unchanged
    - no lexical match: prefix cut (old behaviour), ellipsis at the end
    - otherwise: the best-matching sentence plus neighbours (preferring ones
      that also match) while they fit; ellipses mark cut ends
    """
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    terms = query_terms(query or "")
    sentences = split_sentences(text)
    if not terms or not sentences:
        return text[:max_chars] + marker

    scores = [_hits(s, terms) for s in sentences]
    if max(scores) == 0:
        return text[:max_chars] + marker

    best = max(range(len(sentences)), key=lambda i: (scores[i], -i))
    lo = hi = best
    used = len(sentences[best])
    budget = max_chars - 2 * len(marker) - 2
    if used > budget:
        # one sentence is itself over budget: window inside it around the first hit
        s = sentences[best]
        pos = min((s.lower().find(t) for t in terms if t in s.lower()), default=0)
        start = max(0, pos - budget // 3)
        return marker + s[start:start + budget] + marker

    # expand: prefer the neighbour that carries hits, else alternate
    while True:
        cand = []
        if lo > 0:
            cand.append((scores[lo - 1], -0, lo - 1))
        if hi < len(sentences) - 1:
            cand.append((scores[hi + 1], -1, hi + 1))
        if not cand:
            break
        cand.sort(reverse=True)
        _, _, idx = cand[0]
        add = len(sentences[idx]) + 1
        if used + add > budget:
            # try the other side once before giving up
            if len(cand) > 1:
                _, _, idx2 = cand[1]
                add2 = len(sentences[idx2]) + 1
                if used + add2 <= budget:
                    idx, add = idx2, add2
                else:
                    break
            else:
                break
        if idx < lo:
            lo = idx
        else:
            hi = idx
        used += add

    window = " ".join(sentences[lo:hi + 1])
    if lo > 0:
        window = marker + " " + window
    if hi < len(sentences) - 1:
        window = window + " " + marker
    return window


def _head_cut(text: str, budget: int) -> str:
    """At most `budget` chars from the start, ending on a sentence boundary
    when one falls in the back half of the window, else on a word boundary."""
    if budget <= 0:
        return ""
    window = text[:budget]
    stop = max(window.rfind(". "), window.rfind("! "), window.rfind("? "),
               window.rfind(".\n"), window.rfind("\n"))
    if stop >= budget // 2:
        return window[:stop + 1]
    space = window.rfind(" ")
    return window[:space] if space >= budget // 2 else window


def _tail_cut(text: str, budget: int) -> str:
    """At most `budget` chars from the END, starting on a sentence boundary
    when one falls in the front half of the window, else on a word boundary.
    Starting mid-clause is what drops a negation, so the boundary matters."""
    if budget <= 0:
        return ""
    window = text[-budget:]
    for mark in (". ", "! ", "? ", "\n"):
        pos = window.find(mark)
        if 0 <= pos <= budget // 2:
            return window[pos + len(mark):]
    space = window.find(" ")
    return window[space + 1:] if 0 <= space <= budget // 2 else window


def head_tail_excerpt(text: str, max_chars: int, marker: str = "...",
                      head_share: float = 0.35) -> str:
    """At most `max_chars` of `text`, keeping BOTH the opening and the END.

    For text with no query to centre on whose end carries the state: the last
    turns of a session (core/handoff.py). A prefix cut drops exactly what a
    handoff exists for - the unfinished next step at the end of a long turn
    (review 2026-09-11, finding 1) - and a bare last-N-character cut drops the
    antecedent that says what the turn is about. The elided middle is marked
    with its size, so the reader knows the content is a window and how much is
    missing; neither part is verbatim on its own.
    """
    text = text or ""
    if len(text) <= max_chars:
        return text
    gap_reserve = 2 * len(marker) + 22  # "...[12345 chars elided]..." plus spaces
    body = max_chars - gap_reserve
    if body < 2 * MIN_EXCERPT_SIDE:
        # too small to hold both ends: keep the END, which is the state
        return marker + " " + _tail_cut(text, max(0, max_chars - len(marker) - 1))
    head = _head_cut(text, max(MIN_EXCERPT_SIDE, int(body * head_share)))
    tail = _tail_cut(text, body - len(head))
    elided = len(text) - len(head) - len(tail)
    if elided <= 0:
        return text[:max_chars]
    return (f"{head.rstrip()} {marker}[{elided} chars elided]{marker} "
            f"{tail.lstrip()}")
