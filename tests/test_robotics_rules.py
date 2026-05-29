"""The ECA rules engine: events fire behaviors that drive cubes deterministically.

Covers the full reflex loop (publish event -> behavior runs -> cube state
changes) and the clean-failure case the design flagged: a behavior whose target
has disconnected logs an error and the engine keeps running.
"""

import pytest

from blipshell.robotics import CapabilityRegistry
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.rules import Behavior, BehaviorAction, RulesEngine


def _listening_behavior(cube_id="led_matrix_01"):
    return Behavior(
        name="show_listening",
        trigger="speech_detected",
        actions=[BehaviorAction(
            target=cube_id, action="display_text", args={"text": "Listening..."},
        )],
    )


@pytest.fixture
def engine_with_cube():
    caps = CapabilityRegistry()
    cube = VirtualLEDMatrix()
    engine = RulesEngine(caps)
    return caps, cube, engine


async def test_load_subscribes_triggers(engine_with_cube):
    caps, cube, engine = engine_with_cube
    engine.load([_listening_behavior()])
    assert engine.triggers == {"speech_detected"}


async def test_event_fires_behavior_and_drives_cube(engine_with_cube):
    caps, cube, engine = engine_with_cube
    await caps.connect(cube)
    engine.load([_listening_behavior()])

    await caps.event_bus.publish("speech_detected", {})

    assert cube.last_text == "Listening..."


async def test_unrelated_event_does_nothing(engine_with_cube):
    caps, cube, engine = engine_with_cube
    await caps.connect(cube)
    engine.load([_listening_behavior()])

    await caps.event_bus.publish("battery_low", {})

    assert cube.last_text is None


async def test_multiple_behaviors_same_trigger_run_in_order(engine_with_cube):
    caps, cube, engine = engine_with_cube
    await caps.connect(cube)
    engine.load([
        Behavior(trigger="wake", actions=[
            BehaviorAction(target="led_matrix_01", action="display_text", args={"text": "first"}),
        ]),
        Behavior(trigger="wake", actions=[
            BehaviorAction(target="led_matrix_01", action="display_text", args={"text": "second"}),
        ]),
    ])

    await caps.event_bus.publish("wake", {})

    assert cube.last_text == "second"  # ran in load order, last wins state


async def test_behavior_targeting_disconnected_cube_fails_cleanly(engine_with_cube):
    caps, cube, engine = engine_with_cube
    await caps.connect(cube)
    engine.load([_listening_behavior()])
    await caps.disconnect("led_matrix_01")

    # Must not raise — the engine logs an actionable error and survives.
    await caps.event_bus.publish("speech_detected", {})

    assert engine.last_results[-1]["result"].lower().startswith("error")
    assert cube.last_text is None


async def test_behavior_can_target_cube_that_connects_later(engine_with_cube):
    caps, cube, engine = engine_with_cube
    # Load behaviors BEFORE the cube exists — common at startup.
    engine.load([_listening_behavior()])
    await caps.connect(cube)

    await caps.event_bus.publish("speech_detected", {})

    assert cube.last_text == "Listening..."


async def test_reload_replaces_rules_no_double_fire(engine_with_cube):
    caps, cube, engine = engine_with_cube
    await caps.connect(cube)
    engine.load([_listening_behavior()])
    # Reload with a different behavior on the same trigger.
    engine.load([Behavior(trigger="speech_detected", actions=[
        BehaviorAction(target="led_matrix_01", action="display_text", args={"text": "new"}),
    ])])

    await caps.event_bus.publish("speech_detected", {})

    # Only the new behavior ran (old subscription was removed).
    fired = [r for r in engine.last_results if r["event"] == "speech_detected"]
    assert len(fired) == 1
    assert cube.last_text == "new"


async def test_results_history_records_dispatch(engine_with_cube):
    caps, cube, engine = engine_with_cube
    await caps.connect(cube)
    engine.load([_listening_behavior()])

    await caps.event_bus.publish("speech_detected", {})

    rec = engine.last_results[-1]
    assert rec["event"] == "speech_detected"
    assert rec["target"] == "led_matrix_01"
    assert rec["action"] == "display_text"
