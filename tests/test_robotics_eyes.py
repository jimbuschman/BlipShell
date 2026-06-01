"""Procedural eye geometry: mood/blink/gaze -> per-eye shape parameters."""

import pytest

from blipshell.robotics.eyes import EyeShape, eye_geometry


def test_neutral_is_moderate_open_no_lids():
    e = eye_geometry(0.0, 0.0)
    assert e.openness == pytest.approx(0.6)
    assert e.lower_lid == 0.0
    assert e.upper_lid_inner == 0.0 and e.upper_lid_outer == 0.0


def test_arousal_sets_openness():
    wide = eye_geometry(0.0, 1.0)
    sleepy = eye_geometry(0.0, -1.0)
    assert wide.openness == pytest.approx(1.0)
    assert sleepy.openness == pytest.approx(0.2)
    assert wide.openness > sleepy.openness


def test_blink_closes_eyes():
    assert eye_geometry(0.0, 1.0, blink=1.0).openness == pytest.approx(0.0)
    half = eye_geometry(0.0, 0.0, blink=0.5)
    assert half.openness == pytest.approx(0.3)  # 0.6 * (1 - 0.5)


def test_positive_valence_raises_lower_lid():
    happy = eye_geometry(1.0, 0.0)
    assert happy.lower_lid > 0.0
    assert happy.upper_lid_inner == 0.0  # no droop when happy


def test_negative_valence_droops_upper_lids():
    sad = eye_geometry(-1.0, -0.5)  # sad + calm
    assert sad.upper_lid_inner > 0.0 and sad.upper_lid_outer > 0.0
    assert sad.lower_lid == 0.0
    # calm sadness is roughly symmetric (no furrow)
    assert sad.upper_lid_inner == pytest.approx(sad.upper_lid_outer)


def test_aroused_sadness_furrows_inner_corner():
    angry = eye_geometry(-1.0, 1.0)  # negative valence + high arousal
    # inner corner drops more than outer — a furrow, toward an angry look
    assert angry.upper_lid_inner > angry.upper_lid_outer


def test_gaze_passthrough_and_clamp():
    e = eye_geometry(0.0, 0.0, gaze=(2.0, -2.0))
    assert e.gaze_x == 1.0 and e.gaze_y == -1.0


def test_inputs_clamped():
    e = eye_geometry(5.0, -5.0)
    assert e.lower_lid <= 1.0
    assert e.openness == pytest.approx(0.2)  # arousal clamped to -1


def test_returns_eyeshape():
    assert isinstance(eye_geometry(0.3, 0.3), EyeShape)
