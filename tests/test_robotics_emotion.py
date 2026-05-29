"""The emotion engine: a persistent valence/arousal mood, evolved deterministically.

No display, no LLM, no clock — elapsed time is passed in — so the dynamics are
fully testable in isolation.
"""

import math

import pytest

from blipshell.robotics.emotion import (
    NEUTRAL_CALM,
    AffectState,
    EmotionEngine,
    mood_label,
)


def test_starts_at_baseline():
    e = EmotionEngine()
    assert e.state.valence == NEUTRAL_CALM.valence
    assert e.state.arousal == NEUTRAL_CALM.arousal


def test_appraisal_moves_in_expected_direction():
    e = EmotionEngine()
    e.appraise("praise")
    assert e.state.valence > NEUTRAL_CALM.valence  # praise lifts valence

    e2 = EmotionEngine()
    e2.appraise("task_failed")
    assert e2.state.valence < 0          # failure is negative
    assert e2.state.arousal > NEUTRAL_CALM.arousal  # and activating


def test_unknown_event_is_noop():
    e = EmotionEngine()
    before = e.dump()
    assert e.appraise("not_an_event") is False
    assert e.dump() == before


def test_intensity_scales_the_nudge():
    weak = EmotionEngine()
    strong = EmotionEngine()
    weak.appraise("praise", intensity=0.5)
    strong.appraise("praise", intensity=1.0)
    assert strong.state.valence > weak.state.valence


def test_state_is_clamped():
    e = EmotionEngine()
    for _ in range(20):
        e.appraise("praise")  # would overshoot +1 without clamping
    assert e.state.valence <= 1.0
    assert -1.0 <= e.state.arousal <= 1.0


def test_decay_moves_toward_baseline():
    e = EmotionEngine(relax_tau_seconds=100.0)
    e.appraise("praise")
    lifted = e.state.valence

    e.decay(100.0)  # one time-constant
    # Should be ~37% of the way from baseline remaining (e^-1).
    expected = NEUTRAL_CALM.valence + (lifted - NEUTRAL_CALM.valence) * math.exp(-1)
    assert e.state.valence == pytest.approx(expected, abs=1e-6)
    # And strictly closer to baseline than before.
    assert abs(e.state.valence - NEUTRAL_CALM.valence) < abs(lifted - NEUTRAL_CALM.valence)


def test_large_decay_returns_to_baseline():
    e = EmotionEngine(relax_tau_seconds=100.0)
    e.appraise("task_failed")
    e.decay(10_000.0)  # way past the time constant
    assert e.state.valence == pytest.approx(NEUTRAL_CALM.valence, abs=1e-3)
    assert e.state.arousal == pytest.approx(NEUTRAL_CALM.arousal, abs=1e-3)


def test_zero_or_negative_decay_is_noop():
    e = EmotionEngine()
    e.appraise("praise")
    before = e.dump()
    e.decay(0)
    e.decay(-5)
    assert e.dump() == before


def test_persistence_round_trip():
    e = EmotionEngine()
    e.appraise("praise")
    e.appraise("user_returned")
    saved = e.dump()

    e2 = EmotionEngine()
    e2.load(saved)
    assert e2.state.valence == pytest.approx(e.state.valence)
    assert e2.state.arousal == pytest.approx(e.state.arousal)


def test_load_corrupt_falls_back_to_baseline():
    e = EmotionEngine()
    e.load({"garbage": True})
    assert e.state.valence == NEUTRAL_CALM.valence
    assert e.state.arousal == NEUTRAL_CALM.arousal


def test_mood_labels_cover_the_space():
    assert mood_label(AffectState(valence=0.0, arousal=0.0)) == "neutral"
    assert mood_label(AffectState(valence=0.8, arousal=0.8)) == "excited"
    assert mood_label(AffectState(valence=0.8, arousal=-0.8)) == "content"
    assert mood_label(AffectState(valence=-0.8, arousal=0.8)) == "agitated"
    assert mood_label(AffectState(valence=-0.8, arousal=-0.8)) == "glum"
    assert mood_label(AffectState(valence=0.0, arousal=0.8)) == "alert"
    assert mood_label(AffectState(valence=0.0, arousal=-0.8)) == "sleepy"


def test_accumulation_builds_mood():
    e = EmotionEngine()
    v0 = e.state.valence
    e.appraise("interaction")
    v1 = e.state.valence
    e.appraise("interaction")
    v2 = e.state.valence
    assert v2 > v1 > v0  # repeated positives accumulate
