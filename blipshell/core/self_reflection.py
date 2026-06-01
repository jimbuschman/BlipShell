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

logger = logging.getLogger(__name__)

# Sentinel the model returns when nothing is genuinely on its mind.
NOTHING = "NOTHING"


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
    """Persists self-originated thoughts (app_metadata-backed JSON list)."""

    KEY = "self_thoughts"

    def __init__(self, sqlite, max_keep: int = 50):
        self._sqlite = sqlite
        self._max = max_keep

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

    async def add(self, text: str) -> None:
        items = await self._load()
        items.append({"text": text, "surfaced": False})
        await self._save(items)

    async def recent(self, n: int = 5) -> list[str]:
        items = await self._load()
        return [i["text"] for i in items[-n:]]

    async def has_pending(self) -> bool:
        return any(not i.get("surfaced") for i in await self._load())

    async def take_pending(self) -> str | None:
        """Return the oldest unsurfaced thought, marking it surfaced. None if none."""
        items = await self._load()
        for i in items:
            if not i.get("surfaced"):
                i["surfaced"] = True
                await self._save(items)
                return i["text"]
        return None
