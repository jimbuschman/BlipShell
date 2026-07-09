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
from datetime import datetime, timezone
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

    def __init__(self, sqlite, max_keep: int = 50, embed_fn: Optional[EmbedFn] = None,
                 *, gravity_enabled: bool = False, recur_threshold: float = 0.85,
                 recur_boost: float = 0.5, fatigue: float = 0.6,
                 half_life_days: float = 30.0, min_weight: float = 0.1):
        self._sqlite = sqlite
        self._max = max_keep
        self._embed_fn = embed_fn
        # Self-gravity (off by default). When disabled, every gravity path below
        # is a no-op and the store behaves exactly as before.
        self._gravity_enabled = gravity_enabled
        self._recur_threshold = recur_threshold
        self._recur_boost = recur_boost
        self._fatigue = fatigue
        self._half_life_days = half_life_days
        self._min_weight = min_weight

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _effective_weight(self, item: dict, now: datetime) -> float:
        """Stored weight after age decay (computed at read, not persisted).

        Recurrence/fatigue mutate the stored ``weight``; age decay is applied on
        top here so a thought has to keep recurring to stay heavy as it ages.
        """
        w = item.get("weight", 1.0)
        created = item.get("created_at")
        if created and self._half_life_days > 0:
            try:
                age_days = (now - datetime.fromisoformat(created)).total_seconds() / 86400.0
                if age_days > 0:
                    w *= 0.5 ** (age_days / self._half_life_days)
            except (ValueError, TypeError):
                pass
        return max(w, self._min_weight)

    async def effective_weights(self, texts) -> dict:
        """Map each given thought text -> its current effective (age-decayed)
        weight. Used by the surfacing gate to rank, and by the renderer to mark
        recurring thoughts. Empty when gravity is disabled."""
        if not self._gravity_enabled or not texts:
            return {}
        now = self._now()
        wanted = set(texts)
        return {it["text"]: self._effective_weight(it, now)
                for it in await self._load() if it["text"] in wanted}

    async def apply_fatigue(self, texts) -> None:
        """Decay the stored weight of thoughts that just surfaced (anti-spiral:
        the same thought can't dominate turn after turn). No-op when disabled;
        recurrence is what lets a genuinely central thought recover."""
        if not self._gravity_enabled or not texts:
            return
        wanted = set(texts)
        items = await self._load()
        changed = False
        for it in items:
            if it["text"] in wanted:
                it["weight"] = max(it.get("weight", 1.0) * self._fatigue, self._min_weight)
                changed = True
        if changed:
            await self._save(items)

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
        emb = await self._embed(text)
        # Recurrence reinforcement: if this new thought echoes a prior one, the
        # prior gains weight — "it keeps coming back to this" is gravity. (Only
        # when gravity is enabled; otherwise weight is inert metadata.)
        if self._gravity_enabled and emb:
            for it in items:
                ie = it.get("embedding")
                if ie and _cosine(emb, ie) >= self._recur_threshold:
                    it["weight"] = it.get("weight", 1.0) + self._recur_boost
        items.append({
            "text": text, "surfaced": False, "embedding": emb,
            "weight": 1.0, "created_at": self._now().isoformat(),
        })
        await self._save(items)

    async def recent(self, n: int = 5) -> list[str]:
        items = await self._load()
        return [i["text"] for i in items[-n:]]

    async def snapshot(self) -> list[dict]:
        """Read-only view of the store for observability (/thoughts).

        One row per stored thought, oldest first: text, created_at (ISO or
        None), surfaced flag, stored weight, effective (age-decayed) weight —
        None when gravity is disabled — and whether the thought has an
        embedding (without one it can never resurface via relevance).
        Never mutates the store.
        """
        now = self._now()
        rows = []
        for it in await self._load():
            rows.append({
                "text": it["text"],
                "created_at": it.get("created_at"),
                "surfaced": bool(it.get("surfaced")),
                "weight": it.get("weight", 1.0),
                "effective_weight": (
                    self._effective_weight(it, now)
                    if self._gravity_enabled else None
                ),
                "has_embedding": bool(it.get("embedding")),
            })
        return rows

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

        This is a loose *recall* prefilter, not the gate — a sharper LLM
        relevance judge decides what actually surfaces. Thoughts missing an embedding (e.g.
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
