"""The lifecycle bridge: Agent emits events that drive a connected cube.

_emit_robot_event is the seam that makes the cube BlipShell's body — when the
assistant thinks/responds, the matching event fires and the authored behavior
runs, with no manual triggering. Tested via a minimal ChatMixin stand-in so we
don't bootstrap a whole Agent.
"""

import asyncio

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics import RoboticsCore
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.rules import Behavior, BehaviorAction


class _MiniAgent(ChatMixin):
    """Just enough of an Agent to exercise _emit_robot_event."""

    def __init__(self, robotics):
        self.robotics = robotics
        self._background_tasks = set()


async def test_emit_reaches_event_bus():
    agent = _MiniAgent(RoboticsCore(ToolRegistry(), generate_fn=None))
    got = []

    async def handler(name, payload):
        got.append(name)

    agent.robotics.registry.event_bus.subscribe("thinking_started", handler)
    agent._emit_robot_event("thinking_started")
    await asyncio.sleep(0.02)  # let the fire-and-forget task run

    assert got == ["thinking_started"]


async def test_emit_drives_connected_cube_via_rules():
    agent = _MiniAgent(RoboticsCore(ToolRegistry(), generate_fn=None))
    cube = VirtualLEDMatrix()
    await agent.robotics.connect(cube)
    # Load a behavior by hand (no LLM): thinking_started -> show a cue.
    agent.robotics.rules.load([Behavior(
        trigger="thinking_started",
        actions=[BehaviorAction(target="led_matrix_01", action="display_text",
                                args={"text": "..."})],
    )])

    agent._emit_robot_event("thinking_started")
    await asyncio.sleep(0.02)

    assert cube.last_text == "..."


async def test_emit_is_noop_without_robotics():
    agent = _MiniAgent(None)
    # Must not raise even with no robotics core.
    agent._emit_robot_event("thinking_started")


async def test_emit_is_noop_when_no_behavior_bound():
    """Emitting an event nobody subscribed to is a cheap no-op, not an error."""
    agent = _MiniAgent(RoboticsCore(ToolRegistry(), generate_fn=None))
    agent._emit_robot_event("speech_detected")
    await asyncio.sleep(0.01)  # nothing should happen, and nothing should raise
