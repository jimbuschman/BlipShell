"""Procedural face renderer: mood -> a readable 8x8 face."""

import pytest

from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics import EmotionEngine, RoboticsCore
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.mood_display import render_face


def test_dimensions():
    f = render_face(0.0, 0.0)
    assert len(f) == 8 and all(len(row) == 8 for row in f)
    assert all(v in (0, 1) for row in f for v in row)


def test_smile_vs_frown_differ():
    smile = render_face(0.8, 0.0)
    frown = render_face(-0.8, 0.0)
    assert smile != frown
    # Smile puts corners high (row 5); frown puts corners low (row 6).
    assert smile[5][2] == 1 and frown[6][2] == 1


def test_flat_mouth_for_neutral_valence():
    f = render_face(0.0, 0.5)
    # Flat mouth = the full bottom row of the mouth lit.
    assert f[6][2] == 1 and f[6][3] == 1 and f[6][4] == 1 and f[6][5] == 1


def test_arousal_changes_eye_openness():
    alert = render_face(0.0, 0.6)
    sleepy = render_face(0.0, -0.6)
    # Alert eyes span two rows; sleepy eyes only one.
    assert alert[1][2] == 1 and alert[2][2] == 1
    assert sleepy[1][2] == 0 and sleepy[2][2] == 1


def test_eyes_present_in_all_moods():
    for v, a in [(0.9, 0.9), (-0.9, -0.9), (0.0, 0.0)]:
        f = render_face(v, a)
        assert f[2][2] == 1 and f[2][5] == 1  # both eyes always drawn


async def test_mood_renders_as_face_on_connected_cube():
    """End to end: a positive mood draws a smiling face on the LED cube.

    Mirrors how the agent renders the emotion engine's state to the cube
    (emotion -> render_face -> display_frame). The cube never authors this.
    """
    core = RoboticsCore(ToolRegistry())  # no authoring — cube is a mood display
    cube = VirtualLEDMatrix()
    await core.connect(cube)

    emotion = EmotionEngine()
    for _ in range(3):
        emotion.appraise("praise")  # push clearly positive
    frame = render_face(emotion.state.valence, emotion.state.arousal)
    await core.registry.invoke("led_matrix_01", "display_frame", {"frame": frame})

    assert cube.frame == frame
    assert cube.frame[5][2] == 1  # smile corner up — positive mood shows a smile
