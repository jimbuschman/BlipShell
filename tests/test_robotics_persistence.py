"""Profile persistence: a cube's authored behaviors survive reconnects.

Without this, BlipShell re-authored from scratch every connect — inconsistent
body language ("O" one session, "Y" the next). Now the profile is saved per
cube type and reused (no LLM call) on later connects; reauthor() forces a fresh
one for tuning.
"""

import json

import pytest

from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics import RoboticsCore
from blipshell.robotics.cubes import VirtualLEDMatrix

PROFILE = {
    "semantic_role": "status display",
    "behaviors": [
        {"trigger": "thinking_started", "actions": [
            {"target": "led_matrix_01", "action": "display_text", "args": {"text": ".."}},
        ]},
    ],
}


class FakeStore:
    """In-memory stand-in for sqlite app_metadata."""

    def __init__(self):
        self.data: dict[str, str] = {}

    async def load(self, key):
        return self.data.get(key)

    async def save(self, key, value):
        self.data[key] = value


def _counting_gen(payload):
    calls = {"n": 0}

    async def gen(system, user):
        calls["n"] += 1
        return json.dumps(payload)

    return gen, calls


def _core(store, gen):
    return RoboticsCore(
        ToolRegistry(), generate_fn=gen,
        load_profile_fn=store.load, save_profile_fn=store.save,
    )


async def test_first_connect_authors_and_saves():
    store = FakeStore()
    gen, calls = _counting_gen(PROFILE)
    core = _core(store, gen)

    await core.connect(VirtualLEDMatrix())

    assert calls["n"] == 1                       # authored via LLM
    assert "led_matrix" in store.data            # saved by type
    assert len(core.get_profile("led_matrix_01").behaviors) == 1


async def test_reconnect_reuses_stored_without_llm():
    store = FakeStore()
    gen, calls = _counting_gen(PROFILE)

    core1 = _core(store, gen)
    await core1.connect(VirtualLEDMatrix())
    assert calls["n"] == 1

    # Fresh core (simulates a restart) sharing the same store.
    core2 = _core(store, gen)
    await core2.connect(VirtualLEDMatrix())

    assert calls["n"] == 1                        # NOT re-authored — reused
    assert len(core2.get_profile("led_matrix_01").behaviors) == 1


async def test_reuse_remaps_targets_to_new_instance():
    """A stored profile drives a different instance of the same type."""
    store = FakeStore()
    gen, calls = _counting_gen(PROFILE)

    core1 = _core(store, gen)
    await core1.connect(VirtualLEDMatrix(cube_id="led_matrix_01"))

    core2 = _core(store, gen)
    await core2.connect(VirtualLEDMatrix(cube_id="led_matrix_99"))

    assert calls["n"] == 1
    profile = core2.get_profile("led_matrix_99")
    assert profile.behaviors[0].actions[0].target == "led_matrix_99"  # remapped


async def test_reauthor_forces_fresh_generation():
    store = FakeStore()
    gen, calls = _counting_gen(PROFILE)
    core = _core(store, gen)
    await core.connect(VirtualLEDMatrix())
    assert calls["n"] == 1

    await core.reauthor("led_matrix_01")

    assert calls["n"] == 2                         # forced a fresh LLM author
    assert "led_matrix" in store.data


async def test_unreadable_stored_profile_falls_back_to_authoring():
    store = FakeStore()
    store.data["led_matrix"] = "{not valid json"
    gen, calls = _counting_gen(PROFILE)
    core = _core(store, gen)

    await core.connect(VirtualLEDMatrix())

    assert calls["n"] == 1                         # bad store -> authored fresh
    assert len(core.get_profile("led_matrix_01").behaviors) == 1
