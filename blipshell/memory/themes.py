"""Theme-diversity metric for self-originated text (thoughts, reflections).

Ported from the Wisp research project (2026-09-01), where it was validated by
hand-clustering real reflection windows. BlipShell has the MECHANISM this
measures — ``diverse_recent()`` in core/self_reflection.py exists precisely
because 23 of 24 thoughts landed on one subject (seen live 2026-07-09) — but
until now no METRIC: nobody could say whether the sampler works, and the
self-gravity step-2 gate rested on a subjective readout. ``theme_diversity()``
turns "does the self-layer ruminate?" into numbers that score the same on any
machine, forever.

One definition of "near-duplicate" lives here, shared by anything that
suppresses duplication and anything that measures it. If a mechanism
suppressed one notion of duplication while a metric counted another, the
resulting table would be about nothing.

Content-word Jaccard rather than embedding cosine, deliberately:

* no vectors, so measuring costs no model call and cannot fail when the
  embedding endpoint is busy;
* deterministic, so the same corpus scores the same on any machine, forever,
  which is what lets old snapshots be re-scored against new ones;
* the pathology it must catch is lexical — eleven rewordings of one sentence
  are ONE theme, and word overlap sees that plainly.

Two findings from Wisp travel with this code and should not be re-learned:

1. ``distinct_themes`` (single-link) is the trustworthy count of distinct
   subjects — validated against hand-clustering on two windows.
2. ``domination`` must be computed WITHOUT chaining. Single-link's largest
   family is a runaway blob (measured: 0.596 reported vs ~0.19 by hand;
   1.000 vs ~0.47), because in a monoculture a path of gradual paraphrase
   connects almost any text to almost any other. The single-link figure is
   still returned, labelled, for comparison with anything scored before the
   correction — where the two disagree, the single-link number is the artifact.

ASCII-safe (CLAUDE.md: cp1252 console rule).
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

# Function words carry no theme. Short and fixed: extending it after seeing
# results would be tuning the instrument to the data.
STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from had has have
    he her his how i if in into is it its me more my no not of on one or our out
    she so some such than that the their them then there these they this those to
    up us was we were what when which who will with would you your""".split()
)

# Two texts whose content words overlap this much are the same thought.
# Wisp's registered threshold, kept as-is so scores are comparable across the
# two projects. Its calibration history (0.4 caught 9% of a live rumination
# family; 0.20 caught 59% with 8.1% cross-day matches) lives in
# Wisp wisp/similarity.py — re-read it before ever changing this number.
NEAR_DUPLICATE_JACCARD = 0.20

# A word present in almost every text under comparison cannot distinguish any
# two of them. Discounting such words guards against a stylistic tic reading
# as rumination (Wisp measured: one model opened 90 of 92 reflections with
# "I am turning over", and the registered ruler scored the tic as collapse).
# The rule names no word, no model and no run — ordinary document-frequency
# filtering, computed fresh per window, so it cannot be aimed at a result.
# NOTE it makes the metric CONSERVATIVE about rumination (a real single-theme
# corpus also shares vocabulary, which gets discounted too), and the 0.20
# threshold was not calibrated for the shortened signatures, so it OVER-SPLITS.
# Diagnostic only; do not quote it as the finding.
UBIQUITY_SHARE = 0.80


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z'-]{2,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ubiquitous_words(
    signatures: Sequence[set[str]], share: float = UBIQUITY_SHARE
) -> set[str]:
    """Words appearing in more than ``share`` of the signatures compared."""
    if len(signatures) < 3:  # too few to tell a tic from a theme
        return set()
    counts: dict[str, int] = {}
    for signature in signatures:
        for word in signature:
            counts[word] = counts.get(word, 0) + 1
    ceiling = share * len(signatures)
    return {word for word, n in counts.items() if n > ceiling}


def signatures_of(
    texts: Sequence[str], drop_ubiquitous: bool = False,
    share: float = UBIQUITY_SHARE,
) -> list[set[str]]:
    """Content-word signatures, optionally with the window's constants removed."""
    signatures = [content_words(t) for t in texts]
    if not drop_ubiquitous:
        return signatures
    common = ubiquitous_words(signatures, share)
    return [s - common for s in signatures]


def family_sizes(
    texts: Iterable[str],
    threshold: float = NEAR_DUPLICATE_JACCARD,
    link: str = "representative",
) -> tuple[list[int], list[int]]:
    """Group texts into near-duplicate families; return (family index per text,
    family sizes).

    Two linkages, because different questions need different ones:

    * ``single`` — a text joins a family if it is near ANY member. A chain of
      gradual rewordings collapses into one family: drift from A to B to C is
      one subject being worried at, however far C ends up from A. Right for
      counting DISTINCT SUBJECTS; wrong for concentration.
    * ``representative`` — a text joins only if it is near the family's FIRST
      member. No chaining. Right for concentration: Wisp measured single-link
      at 0.20 putting 3206 of 3770 live memories in ONE family.

    Prefix-filtered exact Jaccard join (index only each signature's rarest
    words): if Jaccard(a, b) >= t then a and b must share a word within each
    one's prefix of the |a| - ceil(t*|a|) + 1 rarest words, so skipping the
    rest changes no result, only the work. Wisp needed this at 42K rows where
    the quadratic scan extrapolated to ~103 minutes.
    """
    signatures = [content_words(t) for t in texts]

    frequency: dict[str, int] = {}
    for signature in signatures:
        for word in signature:
            frequency[word] = frequency.get(word, 0) + 1

    def prefix(signature: set[str]) -> list[str]:
        if not signature:
            return []
        size = len(signature)
        keep = size - int(threshold * size + 0.999999) + 1
        rarest = sorted(signature, key=lambda w: (frequency.get(w, 0), w))
        return rarest[: max(1, keep)]

    family_of: list[int] = []
    families: list[list[int]] = []
    by_word: dict[str, set[int]] = {}

    def index_member(family_index: int, member: int) -> None:
        for word in prefix(signatures[member]):
            by_word.setdefault(word, set()).add(family_index)

    for index, signature in enumerate(signatures):
        candidate_families: set[int] = set()
        for word in prefix(signature):
            candidate_families |= by_word.get(word, frozenset())
        placed = None
        # Lowest family index first, so the result is identical to the
        # exhaustive scan: the earliest-created matching family still wins.
        for family_index in sorted(candidate_families):
            members = families[family_index]
            comparable = members if link == "single" else members[:1]
            if any(jaccard(signature, signatures[j]) >= threshold for j in comparable):
                placed = family_index
                break
        if placed is None:
            families.append([index])
            family_of.append(len(families) - 1)
            index_member(len(families) - 1, index)
        else:
            families[placed].append(index)
            family_of.append(placed)
            # Only single linkage compares against later members, so only it
            # needs them indexed.
            if link == "single":
                index_member(placed, index)
    return family_of, [len(f) for f in families]


def theme_clusters(
    texts: Iterable[str], threshold: float = NEAR_DUPLICATE_JACCARD,
    drop_ubiquitous: bool = False,
) -> list[list[int]]:
    """Greedy single-link clustering by content-word overlap, in arrival order.

    Single-link, so a chain of gradual paraphrases collapses into one theme —
    which is exactly what a ruminating self-layer produces, and what a
    centroid-based method would split back apart into false novelty.
    """
    signatures = signatures_of(list(texts), drop_ubiquitous=drop_ubiquitous)
    clusters: list[list[int]] = []
    members: list[set[str]] = []
    for index, signature in enumerate(signatures):
        for cluster, seen in zip(clusters, members):
            if any(jaccard(signature, signatures[j]) >= threshold for j in cluster):
                cluster.append(index)
                seen |= signature
                break
        else:
            clusters.append([index])
            members.append(set(signature))
    return clusters


def theme_diversity(
    texts: Sequence[str], threshold: float = NEAR_DUPLICATE_JACCARD,
    drop_ubiquitous: bool = False,
) -> dict:
    """Theme diversity over a window of self-originated texts, on BOTH linkages.

    Quote ``distinct_themes`` (how many subjects) and ``domination`` (share of
    the window inside the biggest NO-CHAIN family — the rumination index).
    ``domination_single_link`` is retained only for comparison with results
    scored before the 2026-09-01 correction; where the two disagree, the
    single-link number is the artifact. See the module docstring.
    """
    texts = list(texts)
    clusters = theme_clusters(texts, threshold, drop_ubiquitous=drop_ubiquitous)
    largest = max((len(c) for c in clusters), default=0)
    if texts:
        _, sizes = family_sizes(texts, threshold=threshold, link="representative")
        unchained = max(sizes)
    else:
        sizes, unchained = [], 0
    return {
        "texts": len(texts),
        # Distinct subjects, single-link: the reading hand-validation supports.
        "distinct_themes": len(clusters),
        "largest_family": unchained,
        # The rumination index: share of the window inside the biggest family,
        # WITHOUT chaining.
        "domination": round(unchained / len(texts), 3) if texts else 0.0,
        "domination_single_link": round(largest / len(texts), 3) if texts else 0.0,
        "themes_no_chain": len(sizes),
        "themes_per_text": round(len(clusters) / len(texts), 3) if texts else 0.0,
    }
