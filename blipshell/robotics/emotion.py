"""BlipShell's affective interior — a small, persistent emotional state.

This is the "something to express" that the body (Cozmo-style eyes, the LED)
renders. It is deliberately display-agnostic and behavior-agnostic: it observes
events and holds a mood; it does not decide what gets drawn and does not touch
BlipShell's responses. Wiring it to a display or to the chat pipeline happens
elsewhere — this module is just the interior, fully testable on its own.

The state is two continuous axes (a smooth slope, not discrete points):
    valence : -1 (negative) .. +1 (positive)
    arousal : -1 (calm/drowsy) .. +1 (energized/alert)

Events nudge the state; over time it relaxes back toward a baseline (homeostasis,
exponential decay). No clock is read here — elapsed time is passed in — so the
dynamics are deterministic and unit-testable.
"""

import math
from typing import Optional

from pydantic import BaseModel


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class AffectState(BaseModel):
    """A point in valence/arousal space. Both axes required so that incomplete
    stored data fails validation (→ fallback to baseline) rather than silently
    defaulting."""
    valence: float
    arousal: float


# Baseline temperament: neutral valence, slightly calm. This quietly defines
# personality — where the mood rests when nothing is happening.
NEUTRAL_CALM = AffectState(valence=0.0, arousal=-0.2)

# How an observed event shifts the mood: event -> (d_valence, d_arousal).
# Data-driven and tunable on purpose — affect dynamics live here, not scattered
# as magic numbers. Scaled by an optional per-call intensity.
DEFAULT_APPRAISALS: dict[str, tuple[float, float]] = {
    "task_succeeded": (0.30, 0.10),    # it worked — pleased, a little up
    "task_failed":    (-0.30, 0.20),   # error — negative and a bit stressed
    "user_corrected": (-0.20, 0.10),   # chastened
    "praise":         (0.40, 0.15),    # warmth from the user
    "user_returned":  (0.25, 0.20),    # glad you're back
    "interaction":    (0.05, 0.10),    # a normal turn — mild engagement
    "idle":           (0.00, -0.15),   # going quiet — winding down
    "long_absence":   (-0.10, -0.20),  # alone a while
}


def mood_label(state: AffectState) -> str:
    """A human-readable label for the current mood (logging / status only).

    The eyes use the raw valence/arousal; this is just for /status and logs.
    """
    v, a = state.valence, state.arousal
    if abs(v) < 0.2 and abs(a) < 0.2:
        return "neutral"
    hi_a, lo_a = a > 0.2, a < -0.2
    pos_v, neg_v = v > 0.2, v < -0.2
    if pos_v and hi_a:
        return "excited"
    if pos_v and lo_a:
        return "content"
    if pos_v:
        return "pleased"
    if neg_v and hi_a:
        return "agitated"
    if neg_v and lo_a:
        return "glum"
    if neg_v:
        return "displeased"
    return "alert" if hi_a else "sleepy"


class EmotionEngine:
    """Holds and evolves a valence/arousal mood. Display- and behavior-agnostic."""

    def __init__(
        self,
        baseline: AffectState = NEUTRAL_CALM,
        relax_tau_seconds: float = 300.0,
        appraisals: Optional[dict[str, tuple[float, float]]] = None,
    ):
        self.baseline = baseline
        self.tau = relax_tau_seconds  # larger = moods linger longer
        self.appraisals = appraisals if appraisals is not None else DEFAULT_APPRAISALS
        self.state = AffectState(valence=baseline.valence, arousal=baseline.arousal)

    def appraise(self, event: str, intensity: float = 1.0) -> bool:
        """Apply an event's emotional nudge. Returns False if event is unknown."""
        delta = self.appraisals.get(event)
        if delta is None:
            return False
        dv, da = delta
        self.state.valence = _clamp(self.state.valence + dv * intensity)
        self.state.arousal = _clamp(self.state.arousal + da * intensity)
        return True

    def decay(self, elapsed_seconds: float) -> None:
        """Relax the mood toward baseline over elapsed time (exponential)."""
        if elapsed_seconds <= 0:
            return
        factor = math.exp(-elapsed_seconds / self.tau)
        self.state.valence = self.baseline.valence + (self.state.valence - self.baseline.valence) * factor
        self.state.arousal = self.baseline.arousal + (self.state.arousal - self.baseline.arousal) * factor

    def label(self) -> str:
        return mood_label(self.state)

    # --- persistence --------------------------------------------------------

    def dump(self) -> dict:
        """Serialize the current state for storage."""
        return self.state.model_dump()

    def load(self, data: dict) -> None:
        """Restore a stored state. Caller should decay() afterward by elapsed time."""
        try:
            self.state = AffectState.model_validate(data)
            self.state.valence = _clamp(self.state.valence)
            self.state.arousal = _clamp(self.state.arousal)
        except Exception:
            # Corrupt store — fall back to baseline rather than crashing.
            self.state = AffectState(valence=self.baseline.valence, arousal=self.baseline.arousal)
