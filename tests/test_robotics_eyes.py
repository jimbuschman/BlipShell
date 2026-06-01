"""Procedural eye geometry: mood/blink/gaze -> per-eye shape parameters."""

import pytest

from blipshell.robotics.eyes import EyeShape, eye_geometry


def test_neutral_is_moderate_open_no_lids():
    e = eye_geometry(0.0, 0.0)
    assert e.openness == pytest.approx(0.55)
    assert e.lower_lid == 0.0
    assert e.upper_lid_inner == 0.0 and e.upper_lid_outer == 0.0
    assert e.width == pytest.approx(1.0)


def test_arousal_sets_openness_and_width():
    wide = eye_geometry(0.0, 1.0)
    sleepy = eye_geometry(0.0, -1.0)
    assert wide.openness == pytest.approx(1.0)
    assert sleepy.openness == pytest.approx(0.1)
    assert wide.openness > sleepy.openness
    assert wide.width > 1.0          # alert/surprised eyes widen
    assert sleepy.width == pytest.approx(1.0)  # low arousal doesn't widen


def test_blink_closes_eyes():
    assert eye_geometry(0.0, 1.0, blink=1.0).openness == pytest.approx(0.0)


def test_positive_valence_raises_lower_lid():
    happy = eye_geometry(1.0, 0.0)
    assert happy.lower_lid > 0.0
    assert happy.upper_lid_inner == 0.0  # no droop when happy


def test_calm_sadness_slants_outer_corner_down():
    """Sad (low valence, calm) drops the OUTER corners — the classic sad slant."""
    sad = eye_geometry(-1.0, -0.6)
    assert sad.upper_lid_outer > sad.upper_lid_inner
    assert sad.lower_lid == 0.0


def test_aroused_negative_furrows_inner_corner():
    """Angry (low valence, high arousal) drops the INNER corners — a furrow."""
    angry = eye_geometry(-1.0, 1.0)
    assert angry.upper_lid_inner > angry.upper_lid_outer


def test_displeased_neutral_arousal_is_symmetric():
    e = eye_geometry(-1.0, 0.0)
    assert e.upper_lid_inner == pytest.approx(e.upper_lid_outer)
    assert e.upper_lid_inner > 0.0


def test_gaze_passthrough_and_clamp():
    e = eye_geometry(0.0, 0.0, gaze=(2.0, -2.0))
    assert e.gaze_x == 1.0 and e.gaze_y == -1.0


def test_inputs_clamped():
    e = eye_geometry(5.0, -5.0)
    assert e.lower_lid <= 1.0
    assert e.openness == pytest.approx(0.1)  # arousal clamped to -1


def test_returns_eyeshape():
    assert isinstance(eye_geometry(0.3, 0.3), EyeShape)


# --- VirtualEyes smart-device cube ------------------------------------------

def test_eyes_describe_advertises_set_mood():
    from blipshell.robotics.cubes import VirtualEyes
    meta = VirtualEyes().describe()
    assert meta.module_type == "eyes"
    assert meta.get_action("set_mood") is not None
    assert meta.get_action("display_frame") is None  # eyes aren't frame-driven


async def test_eyes_set_mood_updates_target_and_clamps():
    from blipshell.robotics.cubes import VirtualEyes
    eyes = VirtualEyes()
    await eyes.invoke("set_mood", {"valence": 0.8, "arousal": 0.5})
    assert eyes.target_valence == pytest.approx(0.8)
    assert eyes.target_arousal == pytest.approx(0.5)
    await eyes.invoke("set_mood", {"valence": 5, "arousal": -5})
    assert eyes.target_valence == 1.0 and eyes.target_arousal == -1.0


async def test_eyes_mood_flows_through_registry():
    from blipshell.core.tools.base import ToolRegistry
    from blipshell.robotics import RoboticsCore
    from blipshell.robotics.cubes import VirtualEyes
    core = RoboticsCore(ToolRegistry())
    eyes = VirtualEyes()
    await core.connect(eyes)

    result = await core.registry.invoke("eyes_01", "set_mood", {"valence": 0.3, "arousal": -0.1})

    assert "mood set" in result
    assert eyes.target_valence == pytest.approx(0.3)
