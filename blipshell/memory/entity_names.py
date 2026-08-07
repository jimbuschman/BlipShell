"""Name comparison rules shared by both entity-merging paths.

Two places decide whether two entity names refer to the same thing:

  * `EntityExtractor._resolve_entity` — at CREATION time, auto-merging above
    0.85 cosine.
  * `EntityMerger` — the retroactive nightly pass, auto-merging above 0.90.

These rules lived only on EntityMerger, which is the more conservative of the
two AND the one that is disabled by default. So the guard was on the path that
wasn't running, and the live path — creation-time resolution, enabled in
config — had none (deep-dive 2026-08-04).
"""

from __future__ import annotations

import re

_TOKEN_SPLIT = re.compile(r"[\s_\-:/.,]+")


def numeric_tokens(name: str) -> set[str]:
    """Tokens in `name` that contain a digit (version/number markers)."""
    return {
        t for t in _TOKEN_SPLIT.split(name.lower())
        if any(c.isdigit() for c in t)
    }


def normalize_name(name: str) -> str:
    """Collapse only spacing/joining punctuation (space _ - .) so pure lexical
    variants map together (chat gpt == chatgpt, phi-3 == phi3) while meaningful
    symbols are kept (c# != c++) and different numbers stay distinct
    (projectecho_v1 != _v2)."""
    return re.sub(r"[\s_\-.]+", "", name.lower())


def version_distinguished(a_name: str, b_name: str) -> bool:
    """True if two names differ in their numeric/version tokens.

    Embeddings rate names differing only by a version or instance number as
    near-identical — projectecho_v1 vs _v2 scores 0.996, and corememorybackup1
    vs 2, llama 3.2b vs 3.2, ws2812 vs ws2812b all behave the same way. That
    clears any plausible auto-merge threshold and collapses genuinely distinct
    things with no LLM veto. Such pairs are almost always separate versions
    rather than formatting variants, so they must never auto-merge.

    Pairs whose numeric tokens match (langchain-x vs langchain_x,
    deepseek-r1-7b vs deepseek-r1:7b) are unaffected.
    """
    return numeric_tokens(a_name) != numeric_tokens(b_name)
