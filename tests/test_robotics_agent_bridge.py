"""Truthful lifecycle occasions drive a cube via BlipShell's OWN behaviors.

Approach (B): the chat pipeline emits only true events (the user spoke, it
started/finished thinking). What the cube does is decided entirely by the
behaviors BlipShell authored (its plugin), run by the rules engine — never by
hardcoded reactions here. Tested via a minimal ChatMixin stand-in.
"""

import asyncio

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics import RoboticsCore
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.rules import Behavior, BehaviorAction


class _MiniAgent(ChatMixin):
    def __init__(self, robotics):
        self.robotics = robotics
        self._background_tasks = set()


async def test_emit_publishes_true_occasion():
    agent = _MiniAgent(RoboticsCore(ToolRegistry(), generate_fn=None))
    seen = []

    async def handler(name, payload):
        seen.append(name)

    agent.robotics.registry.event_bus.subscribe("thinking_started", handler)
    agent._emit_robot_event("thinking_started")
    await asyncio.sleep(0.02)  # let the fire-and-forget task run

    assert seen == ["thinking_started"]


async def test_occasion_runs_blipshells_authored_behavior():
    """The reaction comes from a behavior 'BlipShell' authored, not from us."""
    agent = _MiniAgent(RoboticsCore(ToolRegistry(), generate_fn=None))
    cube = VirtualLEDMatrix()
    await agent.robotics.connect(cube)
    # Stand in for BlipShell's plugin: it chose to scroll a cue while thinking.
    agent.robotics.rules.load([Behavior(
        trigger="thinking_started",
        actions=[BehaviorAction(target="led_matrix_01", action="display_text",
                                args={"text": "..."})],
    )])

    agent._emit_robot_event("thinking_started")
    await asyncio.sleep(0.02)

    assert cube.last_text == "..."


async def test_occasion_with_no_authored_behavior_is_silent():
    """If BlipShell authored nothing for an occasion, nothing happens — its call."""
    agent = _MiniAgent(RoboticsCore(ToolRegistry(), generate_fn=None))
    cube = VirtualLEDMatrix()
    await agent.robotics.connect(cube)
    # No behaviors loaded.

    agent._emit_robot_event("thinking_started")
    await asyncio.sleep(0.02)

    assert cube.last_text is None  # silence is a valid outcome


async def test_emit_noop_without_robotics():
    _MiniAgent(None)._emit_robot_event("thinking_started")  # must not raise
