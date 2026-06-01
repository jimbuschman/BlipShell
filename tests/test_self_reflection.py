"""Self-layer: lingering-thought store + the idle reflection prompt.

The store holds self-originated thoughts (separate from user memory); the prompt
is given only prior thoughts (no transcript) so what surfaces can't be a recap.
"""

import json

import pytest

from blipshell.core.self_reflection import (
    NOTHING,
    SelfThoughtStore,
    lingering_thought_prompt,
)


class FakeStore:
    """Stand-in for sqlite app_metadata (get/set_metadata over a dict)."""

    def __init__(self):
        self.data: dict[str, str] = {}

    async def get_metadata(self, key):
        return self.data.get(key)

    async def set_metadata(self, key, value):
        self.data[key] = value


# --- prompt ----------------------------------------------------------------

def test_prompt_excludes_conversation_and_allows_nothing():
    system, user = lingering_thought_prompt([])
    assert NOTHING in system
    assert "summary" in system.lower()        # explicitly not a summary
    assert "private" in user.lower() or "beginning" in user.lower()


def test_prompt_feeds_prior_thoughts_for_threading():
    system, user = lingering_thought_prompt(["I keep wondering about continuity",
                                             "names vs identity"])
    assert "continuity" in user
    assert "names vs identity" in user


# --- store -----------------------------------------------------------------

async def test_add_and_recent():
    s = SelfThoughtStore(FakeStore())
    await s.add("first")
    await s.add("second")
    assert await s.recent(5) == ["first", "second"]


async def test_pending_lifecycle():
    s = SelfThoughtStore(FakeStore())
    assert await s.has_pending() is False
    await s.add("a thought")
    assert await s.has_pending() is True

    taken = await s.take_pending()
    assert taken == "a thought"
    assert await s.has_pending() is False        # consumed
    assert await s.take_pending() is None         # nothing left


async def test_surfaced_only_once_across_reload():
    store = FakeStore()
    s = SelfThoughtStore(store)
    await s.add("remember me")
    await s.take_pending()
    # A fresh store over the same backing data must see it already surfaced.
    s2 = SelfThoughtStore(store)
    assert await s2.has_pending() is False


async def test_caps_retained_thoughts():
    s = SelfThoughtStore(FakeStore(), max_keep=3)
    for i in range(6):
        await s.add(f"t{i}")
    kept = await s.recent(99)
    assert kept == ["t3", "t4", "t5"]            # only the last max_keep


async def test_corrupt_store_is_safe():
    store = FakeStore()
    store.data[SelfThoughtStore.KEY] = "{not json"
    s = SelfThoughtStore(store)
    assert await s.recent(5) == []                # falls back to empty, no crash
    await s.add("recovers")
    assert await s.recent(5) == ["recovers"]
