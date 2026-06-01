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
    # Framed as identity, with the decision left to BlipShell.
    assert "your decision" in text.lower()
    assert "part of yourself" in text.lower()


def test_mood_awareness_empty_without_emotion():
    assert _MiniAgent(None)._mood_awareness_text() == ""


def test_mood_awareness_states_mood_and_guardrail():
    from blipshell.robotics import EmotionEngine
    agent = _MiniAgent(None)
    agent.emotion = EmotionEngine()
    agent.emotion.appraise("praise")

    text = agent._mood_awareness_text()

    assert "mood" in text.lower()
    assert "warmth" in text.lower() and "energy" in text.lower()
    assert "never" in text.lower()  # the tone-only / never-reduce-helpfulness guardrail


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
