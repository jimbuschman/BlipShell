"""VirtualEyes smart-device cube (set_mood). Eye geometry now lives in
test_robotics_eye_config.py (the ported esp32-eyes model)."""

import pytest

from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics import RoboticsCore
from blipshell.robotics.cubes import VirtualEyes


def test_eyes_describe_advertises_set_mood():
    meta = VirtualEyes().describe()
    assert meta.module_type == "eyes"
    assert meta.get_action("set_mood") is not None
    assert meta.get_action("display_frame") is None  # eyes aren't frame-driven


async def test_eyes_set_mood_updates_target_and_clamps():
    eyes = VirtualEyes()
    await eyes.invoke("set_mood", {"valence": 0.8, "arousal": 0.5})
    assert eyes.target_valence == pytest.approx(0.8)
    assert eyes.target_arousal == pytest.approx(0.5)
    await eyes.invoke("set_mood", {"valence": 5, "arousal": -5})
    assert eyes.target_valence == 1.0 and eyes.target_arousal == -1.0


async def test_eyes_mood_flows_through_registry():
    core = RoboticsCore(ToolRegistry())
    eyes = VirtualEyes()
    await core.connect(eyes)

    result = await core.registry.invoke("eyes_01", "set_mood", {"valence": 0.3, "arousal": -0.1})

    assert "mood set" in result
    assert eyes.target_valence == pytest.approx(0.3)


def test_eyes_describe_advertises_play_reaction():
    assert VirtualEyes().describe().get_action("play_reaction") is not None


async def test_play_reaction_sets_active_reaction():
    eyes = VirtualEyes()
    res = await eyes.invoke("play_reaction", {"emotion": "surprised", "duration": 2})
    assert "surprised" in res
    assert eyes.active_reaction() == "surprised"


async def test_play_reaction_unknown_emotion_is_error():
    eyes = VirtualEyes()
    res = await eyes.invoke("play_reaction", {"emotion": "ecstatic"})
    assert res.lower().startswith("error")
    assert eyes.active_reaction() is None


async def test_reaction_expires():
    eyes = VirtualEyes()
    await eyes.invoke("play_reaction", {"emotion": "glee"})
    assert eyes.active_reaction() == "glee"
    eyes.reaction_until = 0.0  # force expiry (deadline in the past)
    assert eyes.active_reaction() is None
