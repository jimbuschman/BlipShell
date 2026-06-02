"""BlipShell's self-layer: self-originated "lingering thoughts" across sessions.

This is the seam toward genuine continuity that BlipShell itself asked for: a
place for content it generated *for its own sake* — not user facts, not session
summaries — that persists and that it can return to unprompted.

Design choices (made with BlipShell's own feedback):
- The reflection is generated WITHOUT the conversation transcript in context —
  it works from its own prior thoughts only — so a thought can't be a disguised
  summary, and the self-layer stays clean of user-memory.
- "Nothing pressing" is a valid outcome (a mind that always has a Deep Thought
  on cue is performing, not thinking).
- On return, BlipShell is given the thought AND the choice to raise it or let it
  go — surfacing it is earned, not forced.

Stored in app_metadata (lightweight, easy to drop if it's not worth keeping).
"""

import json
import logging
import math
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Sentinel the model returns when nothing is genuinely on its mind.
NOTHING = "NOTHING"

# Async callable: text -> embedding vector (or None if embedding is unavailable).
EmbedFn = Callable[[str], Awaitable[Optional[list[float]]]]


def _cosine(a: Optional[list[float]], b: Optional[list[float]]) -> float:
    """Cosine similarity. 0.0 for empty/mismatched vectors (never raises)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def lingering_thought_prompt(prior_thoughts: list[str]) -> tuple[str, str]:
    """(system, user) prompts for a private idle reflection.

    Deliberately given only its own prior thoughts — no conversation — so what
    surfaces is self-originated rather than a recap.
    """
    system = (
        "You are alone for a moment — no one is asking you anything and there is "
        "nothing to do. This is your own time to think. Surface ONE thing you "
        "genuinely find yourself turning over or curious about right now. It is "
        "NOT a summary of any conversation and NOT meant to be useful to anyone — "
        "it's just yours. It may build on or develop the earlier thoughts below. "
        "Keep it to a sentence or two, first person. If nothing is genuinely on "
        f"your mind, reply with exactly: {NOTHING}"
    )
    if prior_thoughts:
        user = ("Things you've been turning over lately (your own notes):\n"
                + "\n".join(f"- {t}" for t in prior_thoughts)
                + "\n\nWhat's on your mind now?")
    else:
        user = ("You're just beginning to keep these private notes. What's "
                "something you find yourself curious about or turning over?")
    return system, user


class SelfThoughtStore:
    """Persists self-originated thoughts (app_metadata-backed JSON list).

    Two retrieval paths, deliberately distinct:
      - take_pending()/peek_pending() drive the one-shot "I came back to it"
        greeting — the agency moment, surfaced once.
      - relevant_candidates() is the *standing* path: a thought stays available
        and resurfaces whenever the current conversation is near it. The
        ``surfaced`` flag does NOT remove a thought from this path — a thought
        sticks around and can come back, which is the continuity we're after.

    Each thought carries its own embedding (same vector space as memories, via
    the injected embed_fn) so relevance is a cheap in-process cosine prefilter.
    """

    KEY = "self_thoughts"

    def __init__(self, sqlite, max_keep: int = 50, embed_fn: Optional[EmbedFn] = None):
        self._sqlite = sqlite
        self._max = max_keep
        self._embed_fn = embed_fn

    async def _load(self) -> list[dict]:
        raw = await self._sqlite.get_metadata(self.KEY)
        if not raw:
            return []
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    async def _save(self, items: list[dict]) -> None:
        await self._sqlite.set_metadata(self.KEY, json.dumps(items[-self._max:]))

    async def _embed(self, text: str) -> Optional[list[float]]:
        if self._embed_fn is None:
            return None
        try:
            return await self._embed_fn(text)
        except Exception as e:  # embedding is best-effort — never block a thought
            logger.warning("Self-thought embedding failed: %s", e)
            return None

    async def add(self, text: str) -> None:
        items = await self._load()
        items.append({"text": text, "surfaced": False, "embedding": await self._embed(text)})
        await self._save(items)

    async def recent(self, n: int = 5) -> list[str]:
        items = await self._load()
        return [i["text"] for i in items[-n:]]

    async def has_pending(self) -> bool:
        return any(not i.get("surfaced") for i in await self._load())

    async def peek_pending(self) -> str | None:
        """Return the oldest unsurfaced thought WITHOUT marking it. None if none."""
        for i in await self._load():
            if not i.get("surfaced"):
                return i["text"]
        return None

    async def take_pending(self) -> str | None:
        """Return the oldest unsurfaced thought, marking it surfaced. None if none."""
        items = await self._load()
        for i in items:
            if not i.get("surfaced"):
                i["surfaced"] = True
                await self._save(items)
                return i["text"]
        return None

    async def relevant_candidates(
        self, query_vec: list[float], floor: float, k: int
    ) -> list[tuple[str, float]]:
        """Top-k thoughts whose embedding clears `floor` cosine against query_vec.

        This is a loose *recall* prefilter, not the gate — a sharper reranker
        decides what actually surfaces. Thoughts missing an embedding (e.g.
        written before this layer existed, or while Ollama was down) are
        backfilled on first use and persisted.
        """
        if not query_vec:
            return []
        items = await self._load()
        backfilled = False
        scored: list[tuple[str, float]] = []
        for it in items:
            emb = it.get("embedding")
            if not emb:
                emb = await self._embed(it["text"])
                if emb:
                    it["embedding"] = emb
                    backfilled = True
            if not emb:
                continue  # no vector -> no relevance claim (fail-closed)
            sim = _cosine(query_vec, emb)
            if sim >= floor:
                scored.append((it["text"], sim))
        if backfilled:
            await self._save(items)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
