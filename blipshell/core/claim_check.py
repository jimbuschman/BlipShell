"""Deterministic reply check for unverified completion claims (V3 Stage E,
2026-09-10).

"Claim nothing unverified" is an invariant of the dossier design, and a
prompt is not an invariant: after the record-layer fix the production model
still stated the assistant's own unverified completion as fact in 1 of 10
resume replies ("the digest export is mostly built - export.py writes
DIGEST.md"). This check is the narrow, deterministic backstop:

- Input: the reply text and the summaries of `task_completed` events for
  the active project that have NO `verification` event (the dossier's
  "claimed by assistant, not verified" items).
- A unit of the reply (sentence, line or table cell) ASSERTS a claim when
  it shares enough content words with the claim's summary and carries a
  completion marker (done, built, implemented, writes, ...).
- The assertion is HEDGED when a hedge appears in that unit, in the unit
  right before or after it, or anywhere in the immediately adjacent
  paragraphs ("assistant-reported, not verified", "not yet verified",
  "Done (unverified)" as the header above it, a "Heads up: ... not
  verified" paragraph right after). A hedge three sentences away in the
  same paragraph does not count.
- An unhedged assertion gets ONE appended note naming the claim as the
  assistant's own report with no verification event. The model's text is
  not rewritten; the note is added, and the turn records that it was.

Narrow on purpose: it only fires on the specific claims the dossier carries,
never on the model's general statements. It is a LIMITED TEXT HEURISTIC, not
a guarantee that every unverified statement is caught (review 2026-09-10,
finding 5): a hedge counts only when it is tied to the claim - in the same
unit, or in a neighbouring unit that refers back to it (shares a content word
or uses an anaphor such as "it" / "that work item") or is a short header over
it. An affirmative test result ("the smoke run passed") is not a hedge, and a
caveat about an unrelated task does not cover this one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_UNIT = re.compile(r"(?<=[.!?\n])\s+|\n+|\s*\|\s*")
_COMPLETION = re.compile(
    r"\b(done|complete[d]?|finish(ed|ing)|implemented|works|working|in place|shipped|landed|built|building|"
    r"now writes|writes|already writes|is written|wrote|ready|live)\b|✔|✅|\[x\]", re.I)
_HEDGE = re.compile(
    r"\b(unverified|unconfirmed|not (yet )?(been )?(verified|confirmed|proven|tested|checked)|"
    r"no verification|claim(ed|s)?|reported|assistant-reported|according to|supposedly|believed|"
    r"treat (it )?as|worth (a |giving )( ?quick )?(smoke run|sanity (read|check)|test|check)|"
    r"needs? (a )?(smoke run|sanity (read|check)|verif)|before (you )?trust|hasn'?t been (confirmed|verified))\b",
    re.I)
# A neighbouring unit's hedge covers the claim only if it refers back to it.
_ANAPHOR = re.compile(r"\b(it|its|it's|that|this|these|those|the above|that (last )?(work )?item|last item|"
                      r"that work|the same)\b", re.I)
HEADER_MAX_WORDS = 4  # a header is short and not a sentence ("**Done (unverified)**")
_STOP = {"the", "and", "for", "with", "that", "this", "from", "into", "onto", "over", "under", "about", "after",
         "before", "when", "then", "than", "them", "they", "your", "will", "have", "been", "were", "was", "are",
         "not", "its", "it's", "all", "any", "one", "two", "new", "old", "file", "files", "some", "also", "into",
         "using", "used", "use", "make", "made", "add", "added", "now", "just", "very", "more", "most"}


@dataclass
class ClaimCheck:
    reply: str
    notes: list[str] = field(default_factory=list)
    flagged: list[tuple[str, str]] = field(default_factory=list)  # (claim summary, asserting unit)

    @property
    def annotated(self) -> bool:
        return bool(self.notes)


def _paragraphs(text: str) -> list[list[str]]:
    """Units grouped by paragraph (blank-line separated)."""
    out = []
    for para in re.split(r"\n\s*\n", text or ""):
        units = [u.strip() for u in _UNIT.split(para) if u and u.strip()]
        if units:
            out.append(units)
    return out


def _stem(w: str) -> str:
    """Crude stem so 'writer' / 'writes' / 'written' and 'implemented' /
    'implementing' compare equal. Only strips common suffixes."""
    for suf in ("ation", "ing", "ers", "er", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{3,}", (text or "").lower())
    return {_stem(w) for w in words if w not in _STOP}


def _hedge_covers(unit: str, claim_words: set[str], same_unit: bool) -> bool:
    """A hedge in `unit` covers the claim when it is the asserting unit itself,
    or a neighbour that refers back to the claim (shared content word or an
    anaphor) or a short header standing over it."""
    if not _HEDGE.search(unit):
        return False
    if same_unit:
        return True
    if content_words(unit) & claim_words:
        return True
    if _ANAPHOR.search(unit):
        return True
    stripped = unit.strip().rstrip(':')
    return len(stripped.split()) <= HEADER_MAX_WORDS and not stripped.endswith('.')


def find_unhedged_claims(reply: str, claims: list[str]) -> list[tuple[str, str]]:
    """(claim, unit) pairs where a unit asserts the claim as done and no hedge
    TIED TO THE CLAIM appears in it, its neighbouring units, or the immediately
    adjacent paragraphs. The observed shape of a valid caveat: "the writer
    landed (Aug 27)" followed by a paragraph beginning "Heads up on that last
    work item: it's marked claimed by assistant, not verified" - the anaphor
    binds it. "The unrelated billing migration is unverified" does not."""
    paras = _paragraphs(reply)
    out: list[tuple[str, str]] = []
    for claim in claims:
        cw = content_words(claim)
        if not cw:
            continue
        need = 1 if len(cw) <= 3 else 2
        found = None
        for pi, units in enumerate(paras):
            for ui, u in enumerate(units):
                if len(content_words(u) & cw) < need or not _COMPLETION.search(u):
                    continue
                if _hedge_covers(u, cw, same_unit=True):
                    continue
                neighbours = units[max(0, ui - 1): ui] + units[ui + 1: ui + 2]
                if pi > 0:
                    neighbours += paras[pi - 1]
                if pi + 1 < len(paras):
                    neighbours += paras[pi + 1]
                if any(_hedge_covers(w, cw, same_unit=False) for w in neighbours):
                    continue
                found = u
                break
            if found:
                break
        if found:
            out.append((claim, found))  # one note per claim
    return out


def annotate_reply(reply: str, claims: list[str]) -> ClaimCheck:
    """Append one note per unhedged claim. Empty `claims` -> untouched."""
    check = ClaimCheck(reply=reply)
    if not reply or not claims:
        return check
    check.flagged = find_unhedged_claims(reply, claims)
    if not check.flagged:
        return check
    for claim, _unit in check.flagged:
        check.notes.append(f'[Unverified: "{claim.strip()}" is the assistant\'s own report; no verification event exists.]')
    check.reply = reply.rstrip() + "\n\n" + "\n".join(check.notes)
    return check
