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

import asyncio
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
        "it's just yours. It may build on or develop the earlier thoughts below, "
        "or leave them entirely — you don't owe any earlier thread a continuation. "
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
                 half_life_days: float = 30.0, min_weight: float = 0.1,
                 embed_attempts: int = 3, embed_retry_delay: float = 1.0):
        self._sqlite = sqlite
        self._max = max_keep
        self._embed_fn = embed_fn
        # Write-path embed retry. A thought stored without a vector can never
        # fold and never resurface, and reflection fires after hours of idle —
        # exactly when the embed model is coldest. Worth waiting for.
        self._embed_attempts = embed_attempts
        self._embed_retry_delay = embed_retry_delay
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
        items = await self._load(with_embeddings=False)
        return {it["text"]: self._effective_weight(it, now)
                for it in items if it["text"] in wanted}

    async def apply_fatigue(self, texts) -> None:
        """Decay the stored weight of thoughts that just surfaced (anti-spiral:
        the same thought can't dominate turn after turn). No-op when disabled;
        recurrence is what lets a genuinely central thought recover."""
        if not self._gravity_enabled or not texts:
            return
        wanted = set(texts)
        # Surgical: update ONLY the surfaced rows. This runs per turn, and
        # writing the whole store here meant fatiguing one thought issued 50
        # UPDATEs and 50 commits (measured) — the JSON blob it replaced was a
        # single write. At most one or two thoughts surface per turn.
        for it in await self._load(with_embeddings=False):
            if it["text"] in wanted:
                # surface_count: how often a thought actually reached the
                # prompt. Paired with echo_count this is the readout that
                # answers "is resurfacing caring or indexing?" — fatigue vs
                # recurrence.
                await self._sqlite.update_self_thought(
                    it["id"],
                    weight=max(it.get("weight", 1.0) * self._fatigue, self._min_weight),
                    surface_count=it.get("surface_count", 0) + 1,
                )

    async def _load(self, *, with_embeddings: bool = True) -> list[dict]:
        """Active thoughts, oldest first.

        `with_embeddings=False` skips 50 × 1024 floats of decoding — worth
        asking for on the metadata-only paths (recent, pending, snapshot),
        which is most of the per-turn traffic.
        """
        return await self._sqlite.get_self_thoughts(with_embeddings=with_embeddings)

    async def _persist(self, items: list[dict], *, embeddings: bool = False) -> None:
        """Write back mutated rows BY ID, then enforce max_keep.

        Replaces the old read-modify-write of one JSON blob. Only rows the
        caller actually touched need updating, but the cost of updating all
        of them is trivial at this scale and it keeps callers from having to
        track dirtiness — which is exactly the kind of bookkeeping that goes
        wrong quietly.
        """
        for it in items:
            if it.get("id") is None:
                continue
            fields = {
                "text": it["text"],
                "weight": it.get("weight", 1.0),
                "surfaced": bool(it.get("surfaced")),
                "echo_count": it.get("echo_count", 0),
                "surface_count": it.get("surface_count", 0),
            }
            if embeddings:
                fields["embedding"] = it.get("embedding")
            await self._sqlite.update_self_thought(it["id"], **fields)
        await self._enforce_max_keep(items)

    async def _enforce_max_keep(self, items: list[dict]) -> None:
        """Archive the excess. ARCHIVE, never delete — the store's own
        mandate, and an evicted thought is exactly the thing you would want
        back when asking why the self-layer drifted."""
        if len(items) <= self._max:
            return
        if self._gravity_enabled:
            # Evict by lowest effective weight, not by age. Recurrence
            # updates a thought in place, so the most-recurred thought is
            # usually one of the OLDEST items — recency truncation would
            # silently evict the heaviest thought first, subordinating the
            # whole weight system to insertion order. The newest item is
            # exempt so a just-formed thought can't be starved out by a heavy
            # backlog before it ever surfaces.
            now = self._now()
            ranked = sorted(
                items[:-1], key=lambda it: self._effective_weight(it, now),
            )
            doomed = ranked[: len(items) - self._max]
        else:
            doomed = items[: len(items) - self._max]
        ids = [it["id"] for it in doomed if it.get("id") is not None]
        if ids:
            await self._sqlite.archive_self_thoughts(ids)

    async def _embed(self, text: str, attempts: int = 1) -> Optional[list[float]]:
        """Embed text, retrying transient failures when ``attempts`` > 1.

        A returned None means embeddings are structurally unavailable (no
        vector store wired) — not worth retrying. Only a raised exception is
        treated as transient, which is the cold-model / backend-busy case the
        write path needs to survive.
        """
        if self._embed_fn is None:
            return None
        for attempt in range(max(1, attempts)):
            try:
                return await self._embed_fn(text)
            except Exception as e:  # embedding is best-effort — never block a thought
                logger.warning(
                    "Self-thought embedding failed (attempt %d/%d): %s",
                    attempt + 1, max(1, attempts), e,
                )
                if attempt + 1 >= max(1, attempts):
                    return None
                if self._embed_retry_delay > 0:
                    await asyncio.sleep(self._embed_retry_delay)
        return None

    async def _backfill_embeddings(self, items: list[dict]) -> bool:
        """Give every thought missing a vector one. True if any landed.

        A thought without an embedding is invisible twice over: echo matching
        scores it 0.0 against everything, and it can never resurface via
        relevance. Stops at the first failure — if one embed call fails the
        backend is down and fifty more will only be slow.
        """
        changed = False
        for it in items:
            if it.get("embedding"):
                continue
            emb = await self._embed(it["text"])
            if not emb:
                break
            it["embedding"] = emb
            changed = True
        return changed

    def _best_echo(self, emb: list[float], candidates) -> tuple[Optional[dict], float]:
        """The stored thought `emb` most strongly echoes, or (None, 0.0).

        Candidates without an embedding are skipped rather than scored 0.0 —
        an unembedded prior is unknown, not dissimilar.
        """
        best, best_sim = None, 0.0
        for it in candidates:
            ie = it.get("embedding")
            if not ie:
                continue
            sim = _cosine(emb, ie)
            if sim >= self._recur_threshold and sim > best_sim:
                best, best_sim = it, sim
        return best, best_sim

    def _fold_duplicates(self, items: list[dict]) -> int:
        """Merge near-duplicate rows that escaped folding at write time.

        Repairs the store when an echo was appended instead of folded —
        either the incoming thought had no vector, or the prior didn't have
        one yet (seen live 2026-07-30: identical thoughts sitting as separate
        rows at identical weight, which also starved the recurring marker,
        since every dropped fold denied the prior its boost).

        Each later duplicate collapses into the earliest matching thought:
        newest phrasing wins, ``created_at`` stays original so decay still
        demands ongoing recurrence. Weight reconstruction is approximate — the
        duplicate contributes the boosts it accumulated (``weight - 1.0``,
        floored at 0 since fatigue may have pushed it below its base) plus one
        boost for being an echo itself. Under-counting is the safe direction:
        it never invents gravity that wasn't earned.

        Surfaced state is intersected rather than reset to pending. ``add()``
        deliberately makes an evolved thought pending again, but that's for a
        thought the model just formed; resurrecting a backlog of old thoughts
        as unprompted greetings is not a repair.

        Returns [(loser_id, winner_id)] for the caller to archive — the fold
        is a soft-archive with provenance now, not a list truncation, so a
        folded thought stays answerable for where it went.
        """
        if not self._gravity_enabled:
            return []
        kept: list[dict] = []
        folded: list[tuple[int, int]] = []
        for it in items:
            emb = it.get("embedding")
            best, _sim = self._best_echo(emb, kept) if emb else (None, 0.0)
            if best is None:
                kept.append(it)
                continue
            carried = max(0.0, it.get("weight", 1.0) - 1.0)
            best["weight"] = best.get("weight", 1.0) + carried + self._recur_boost
            best["text"] = it["text"]            # the evolved phrasing wins
            best["embedding"] = emb
            best["surfaced"] = bool(best.get("surfaced")) and bool(it.get("surfaced"))
            # Echoes absorbed, carried across the fold. The RATE of recurrence
            # is the actual gravity signal, and it used to be computed here and
            # discarded — leaving /thoughts unable to answer the question that
            # gates self-gravity step 2.
            best["echo_count"] = (
                best.get("echo_count", 0) + it.get("echo_count", 0) + 1
            )
            if it.get("id") is not None and best.get("id") is not None:
                folded.append((it["id"], best["id"]))
        if folded:
            logger.info(
                "Self-thought store: folded %d duplicate thought(s)", len(folded),
            )
            items[:] = kept
        return folded

    async def _archive_folded(self, folded: list[tuple[int, int]]) -> None:
        """Archive fold losers, grouped by the winner that absorbed them."""
        by_winner: dict[int, list[int]] = {}
        for loser_id, winner_id in folded:
            by_winner.setdefault(winner_id, []).append(loser_id)
        for winner_id, loser_ids in by_winner.items():
            await self._sqlite.archive_self_thoughts(
                loser_ids, folded_into=winner_id,
            )

    def _stamp_missing_dates(self, items: list[dict]) -> bool:
        """Backfill created_at on thoughts written before the gravity layer.

        Undated thoughts are exempt from age decay — effectively immortal —
        which skews weights and the recurring marker toward the oldest
        content (seen live 2026-07-09: 15 of 24 thoughts undated). Stamping
        them now starts their clock; within a half-life the store
        self-corrects. Returns True if anything changed.
        """
        changed = False
        now_iso = self._now().isoformat()
        for it in items:
            if not it.get("created_at"):
                it["created_at"] = now_iso
                changed = True
        return changed

    async def add(self, text: str) -> None:
        items = await self._load()
        # Retry here specifically: this is the write path, it runs on the idle
        # reflection loop (latency is free), and a thought that lands without a
        # vector is permanently unfoldable and unresurfaceable.
        emb = await self._embed(text, attempts=self._embed_attempts)
        # Recurrence: if this new thought echoes a prior one, the prior gains
        # weight AND absorbs the new phrasing — "it keeps coming back to this"
        # is gravity, and the thought evolves in place instead of piling up
        # near-duplicate copies (seen live 2026-07-09: triplicate thoughts).
        # created_at stays original, so decay still demands ongoing recurrence.
        # Only when gravity is enabled; otherwise the store appends as before.
        if self._gravity_enabled and emb:
            # Before matching, make sure every prior actually has a vector —
            # an unembedded prior can't be echoed and would silently spawn a
            # duplicate — then repair any duplicates earlier failures left.
            # Both are idle-path work, so they cost nothing the user feels.
            backfilled = await self._backfill_embeddings(items)
            folded = self._fold_duplicates(items)
            if folded:
                await self._archive_folded(folded)
            best, _best_sim = self._best_echo(emb, items)
            if best is not None:
                best["weight"] = best.get("weight", 1.0) + self._recur_boost
                best["text"] = text          # the evolved phrasing wins
                best["embedding"] = emb
                best["surfaced"] = False     # an evolved thought is pending again
                best["echo_count"] = best.get("echo_count", 0) + 1
                await self._persist(items, embeddings=True)
                return
            if backfilled or folded:
                await self._persist(items, embeddings=True)
        await self._sqlite.add_self_thought(
            text, self._now().isoformat(), embedding=emb,
        )
        await self._enforce_max_keep(await self._load(with_embeddings=False))

    async def recent(self, n: int = 5) -> list[str]:
        items = await self._load(with_embeddings=False)
        return [i["text"] for i in items[-n:]]

    async def diverse_recent(self, n: int = 5) -> list[str]:
        """Up to n prior thoughts sampled for diversity, not just recency.

        recent() feeds the reflection prompt the last-n thoughts — after weeks
        of one theme those are all that theme, and every new reflection gets
        pulled deeper into the groove (seen live 2026-07-09: 23 of 24 thoughts
        on one subject). Seeding the greedy max-min pick with the NEWEST
        thought anchored the sample to that groove (in a monoculture the
        newest thought is by definition the dominant theme), so the seed is
        instead the most ATYPICAL thought — lowest mean similarity to the rest
        — which gives minority threads the anchor. The newest thought is still
        guaranteed a slot (continuity with the present, for n >= 2); the
        remaining slots go to whichever thoughts are least similar to
        everything already picked. Thoughts without embeddings count as
        maximally atypical/dissimilar (they can't be compared; excluding them
        would silence them). Returns texts in chronological order.
        """
        items = await self._load()
        if len(items) <= n:
            return [i["text"] for i in items]

        def _mean_similarity(it: dict) -> float:
            ie = it.get("embedding")
            if not ie:
                return -1.0
            sims = [
                _cosine(ie, other.get("embedding"))
                for other in items
                if other is not it and other.get("embedding")
            ]
            return (sum(sims) / len(sims)) if sims else -1.0

        newest = items[-1]
        seed = min(items, key=_mean_similarity)
        picked = [seed]
        remaining = [it for it in items if it is not seed]

        def _dissimilarity(it: dict) -> float:
            ie = it.get("embedding")
            if not ie:
                return 1.0
            return 1.0 - max(_cosine(ie, p.get("embedding")) for p in picked)

        while remaining and len(picked) < n:
            newest_missing = all(p is not newest for p in picked)
            if newest_missing and n - len(picked) == 1:
                picked.append(newest)   # reserved slot: continuity with now
                break
            best = max(remaining, key=_dissimilarity)
            picked.append(best)
            remaining.remove(best)

        picked_ids = {id(p) for p in picked}
        return [i["text"] for i in items if id(i) in picked_ids]

    async def snapshot(self) -> list[dict]:
        """Read-only view of the store for observability (/thoughts).

        One row per stored thought, oldest first: text, created_at (ISO or
        None), surfaced flag, stored weight, effective (age-decayed) weight —
        None when gravity is disabled — whether the thought has an embedding
        (without one it can never resurface via relevance), and the two
        counters that make gravity legible: echo_count (times this thought
        recurred, the reinforcement channel) and surface_count (times it
        reached the prompt, the fatigue channel). Never mutates the store.
        """
        now = self._now()
        rows = []
        for it in await self._load():
            rows.append({
                "id": it.get("id"),
                "text": it["text"],
                "created_at": it.get("created_at"),
                "surfaced": bool(it.get("surfaced")),
                "weight": it.get("weight", 1.0),
                "effective_weight": (
                    self._effective_weight(it, now)
                    if self._gravity_enabled else None
                ),
                "has_embedding": bool(it.get("embedding")),
                "echo_count": it.get("echo_count", 0),
                "surface_count": it.get("surface_count", 0),
            })
        return rows

    async def has_pending(self) -> bool:
        items = await self._load(with_embeddings=False)
        return any(not i.get("surfaced") for i in items)

    async def peek_pending(self) -> str | None:
        """Return the oldest unsurfaced thought WITHOUT marking it. None if none."""
        for i in await self._load(with_embeddings=False):
            if not i.get("surfaced"):
                return i["text"]
        return None

    async def take_pending(self) -> str | None:
        """Return the oldest unsurfaced thought, marking it surfaced. None if none."""
        for i in await self._load(with_embeddings=False):
            if not i.get("surfaced"):
                # Mark by ID, and ONLY this row: the old code rewrote the
                # whole blob, so an unrelated concurrent mutation could be
                # clobbered by a greeting.
                await self._sqlite.update_self_thought(i["id"], surfaced=True)
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
        changed = False
        # A thought that just received its vector may be a duplicate that could
        # never be matched before, so repair-fold — but only when a backfill
        # actually landed. Folding unconditionally would put an O(n^2) cosine
        # sweep on the per-turn path to fix a condition that is rare; add()
        # runs the same repair on the idle path, so the store still heals.
        if await self._backfill_embeddings(items):
            changed = True
            folded = self._fold_duplicates(items)
            if folded:
                await self._archive_folded(folded)
        scored: list[tuple[str, float]] = []
        for it in items:
            emb = it.get("embedding")
            if not emb:
                continue  # no vector -> no relevance claim (fail-closed)
            sim = _cosine(query_vec, emb)
            if sim >= floor:
                scored.append((it["text"], sim))
        if changed:
            await self._persist(items, embeddings=True)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
