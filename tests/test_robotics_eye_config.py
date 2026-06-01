"""Ported esp32-eyes EyeConfig model: presets, mood mapping, tween, outline."""

import pytest

from blipshell.robotics.eye_config import (
    PRESETS,
    EyeConfig,
    eye_outline,
    lerp_config,
    mirror_config,
    mood_to_config,
    with_blink,
)


def test_eighteen_presets():
    assert len(PRESETS) == 18
    for name in ["normal", "happy", "sad", "angry", "surprised", "sleepy", "awe"]:
        assert name in PRESETS


def test_preset_slope_conventions():
    # Sad slants outer-down (negative); angry/furious slant inner-down (positive).
    assert PRESETS["sad"].slope_top < 0
    assert PRESETS["angry"].slope_top > 0
    assert PRESETS["furious"].slope_top > PRESETS["angry"].slope_top  # furious steeper


def test_mood_mapping_directions():
    happy = mood_to_config(1.0, 0.0)
    sad = mood_to_config(-1.0, -1.0)
    angry = mood_to_config(-1.0, 1.0)
    alert = mood_to_config(0.0, 1.0)
    sleepy = mood_to_config(0.0, -1.0)

    assert happy.height < PRESETS["normal"].height      # happy squishes short
    assert sad.slope_top < 0                             # calm sadness: outer-down
    assert angry.slope_top > 0                           # aroused negative: furrow
    assert alert.height > sleepy.height                  # arousal raises height
    assert alert.width > sleepy.width                    # and widens


def test_lerp_midpoint():
    a = EyeConfig(0, 0, 10, 40, 0.0, 0.0, 2, 2)
    b = EyeConfig(0, 0, 30, 40, 0.4, 0.0, 10, 10)
    mid = lerp_config(a, b, 0.5)
    assert mid.height == pytest.approx(20)
    assert mid.slope_top == pytest.approx(0.2)


def test_mirror_negates_slope_and_offset():
    c = EyeConfig(offset_x=-3, slope_top=0.3, slope_bottom=0.1)
    m = mirror_config(c)
    assert m.slope_top == -0.3 and m.slope_bottom == -0.1 and m.offset_x == 3


def test_blink_collapses_height():
    c = PRESETS["normal"]
    assert with_blink(c, 1.0).height == pytest.approx(3.0)   # clamped sliver
    assert with_blink(c, 0.0).height == c.height


def test_outline_returns_points_and_mirror_differs():
    pts = eye_outline(PRESETS["angry"], 32, 32)
    assert len(pts) > 8 and all(len(p) == 2 for p in pts)
    mirrored = eye_outline(mirror_config(PRESETS["angry"]), 32, 32)
    assert mirrored != pts  # the furrow slant flips


def test_outline_respects_size():
    small = eye_outline(EyeConfig(height=10, width=20, radius_top=0, radius_bottom=0), 50, 50)
    xs = [p[0] for p in small]
    ys = [p[1] for p in small]
    assert max(xs) - min(xs) == pytest.approx(20, abs=1.0)
    assert max(ys) - min(ys) == pytest.approx(10, abs=1.0)
