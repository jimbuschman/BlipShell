"""Cube awareness: BlipShell is told it has a body it can choose to use.

The LED matrix is a passive output — a tool BlipShell uses at its own
discretion. Awareness states what's connected and BlipShell's own authored
notes; it must NOT prescribe when to use the cube (that's BlipShell's call).
"""

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics import RoboticsCore
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.profile import CapabilityProfile


class _MiniAgent(ChatMixin):
    def __init__(self, robotics):
        self.robotics = robotics


def test_awareness_empty_without_robotics():
    assert _MiniAgent(None)._build_cube_awareness() == ""


async def test_awareness_empty_when_no_cube_connected():
    agent = _MiniAgent(RoboticsCore(ToolRegistry(), generate_fn=None))
    assert agent._build_cube_awareness() == ""


async def test_awareness_lists_connected_cube_and_actions():
    agent = _MiniAgent(RoboticsCore(ToolRegistry(), generate_fn=None))
    await agent.robotics.connect(VirtualLEDMatrix())

    text = agent._build_cube_awareness()

    assert "led_matrix_01" in text
    assert "display_text" in text and "display_frame" in text and "clear" in text
    # Permission to use, with the decision left to BlipShell.
    assert "your call" in text.lower()


async def test_awareness_includes_blipshells_own_notes():
    """The profile (BlipShell's plugin) surfaces as its own notes, not our rules."""
    core = RoboticsCore(ToolRegistry(), generate_fn=None)
    agent = _MiniAgent(core)
    await core.connect(VirtualLEDMatrix())
    # Inject a profile as if BlipShell authored it.
    core._profiles["led_matrix_01"] = CapabilityProfile(
        cube_id="led_matrix_01",
        semantic_role="a way to externalize my state",
        intended_uses=["show mood", "signal attention"],
        usage_guidance="keep it subtle",
    )

    text = agent._build_cube_awareness()

    assert "externalize my state" in text
    assert "show mood" in text
    assert "keep it subtle" in text
