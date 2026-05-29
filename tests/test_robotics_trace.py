"""Behavior tracer: observes what a behavior actually renders, non-destructively.

The detection is generic — it compares observable-state snapshots before/after
each action and flags any output immediately overwritten by the next action on
the same target. Nothing here encodes what 'clear' or 'display_text' mean.
"""

import pytest

from blipshell.robotics import CapabilityRegistry
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.rules import Behavior, BehaviorAction
from blipshell.robotics.trace import trace_behaviors


def _b(*actions, trigger="t", name=None):
    return Behavior(trigger=trigger, name=name, actions=[
        BehaviorAction(target=t, action=a, args=args) for (t, a, args) in actions
    ])


@pytest.fixture
async def registry():
    reg = CapabilityRegistry()
    await reg.connect(VirtualLEDMatrix())
    return reg


async def test_display_then_clear_flagged(registry):
    """The exact bug from the live run: HI shown then instantly cleared."""
    behavior = _b(
        ("led_matrix_01", "display_text", {"text": "HI"}),
        ("led_matrix_01", "clear", {}),
        name="greet",
    )
    issues = await trace_behaviors([behavior], registry)

    assert len(issues) == 1
    assert issues[0].action == "display_text"
    assert issues[0].behavior_label == "greet"
    assert "never visible" in issues[0].problem


async def test_single_display_is_clean(registry):
    behavior = _b(("led_matrix_01", "display_text", {"text": "Listening..."}))
    assert await trace_behaviors([behavior], registry) == []


async def test_clear_then_display_is_clean(registry):
    """Reordering so the visible output is last produces no issue."""
    behavior = _b(
        ("led_matrix_01", "clear", {}),
        ("led_matrix_01", "display_text", {"text": "HI"}),
    )
    assert await trace_behaviors([behavior], registry) == []


async def test_two_displays_first_is_flagged(registry):
    """display A then display B — A was never seen."""
    frame = [[0] * 8 for _ in range(8)]
    behavior = _b(
        ("led_matrix_01", "display_text", {"text": "A"}),
        ("led_matrix_01", "display_frame", {"frame": frame}),
    )
    issues = await trace_behaviors([behavior], registry)
    assert len(issues) == 1
    assert issues[0].action == "display_text"


async def test_trace_is_non_destructive(registry):
    """A dry-run must leave the cube exactly as it was."""
    cube = registry.get_cube("led_matrix_01")
    await registry.invoke("led_matrix_01", "display_text", {"text": "BEFORE"})
    before = cube.snapshot()

    await trace_behaviors([_b(
        ("led_matrix_01", "display_text", {"text": "HI"}),
        ("led_matrix_01", "clear", {}),
    )], registry)

    assert cube.snapshot() == before
    assert cube.last_text == "BEFORE"


async def test_redundant_action_not_flagged(registry):
    """An action that changes nothing (clear on blank) isn't treated as output."""
    behavior = _b(
        ("led_matrix_01", "clear", {}),       # blank -> blank, no change
        ("led_matrix_01", "display_text", {"text": "HI"}),
    )
    assert await trace_behaviors([behavior], registry) == []
