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
