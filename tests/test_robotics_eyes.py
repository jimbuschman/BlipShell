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
