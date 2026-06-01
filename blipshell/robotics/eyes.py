"""Procedural eye geometry — Cozmo/Vector-style parametric eyes.

Pure and display-agnostic: maps an affective state (valence/arousal) plus a
blink amount and gaze offset into per-eye shape parameters. A renderer (the eye
window, later real OLED hardware) draws a rounded-rect base and cuts it with
"lids" according to these params; animation is the renderer tweening between
EyeShapes over time. No GUI, no hardware, no clock here — fully unit-testable.

Mapping (v1, two axes → expression):
    arousal  -> openness (eye height): low = sleepy/short, high = wide/alert
    valence  -> lids: positive raises the lower lid into a happy squint;
                negative droops the upper lids (sad), and when also aroused the
                inner corner drops more (a furrow, toward angry).
Reactions (surprise, big grin, etc.) will later override these briefly.
"""

from dataclasses import dataclass


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class EyeShape:
    """Geometry for one eye, all normalized 0..1 (or -1..1 for gaze).

    The renderer interprets these against the eye's allotted box: it draws a
    rounded rectangle of height `openness` and width-scale `width`, then fills
    the top with a lid whose coverage runs from `upper_lid_inner` (nose side) to
    `upper_lid_outer`, and raises the bottom by `lower_lid`. `gaze_*` shift it.
    """

    openness: float          # 0 = closed (blink) .. 1 = wide open
    lower_lid: float         # 0..1 raised from the bottom (happy squint)
    upper_lid_inner: float   # 0..1 top coverage at the inner (nose-side) corner
    upper_lid_outer: float   # 0..1 top coverage at the outer corner
    width: float = 1.0       # horizontal scale (>1 wider, e.g. surprised/awe)
    gaze_x: float = 0.0      # -1..1 horizontal look offset (saccades)
    gaze_y: float = 0.0      # -1..1 vertical look offset


def eye_geometry(
    valence: float,
    arousal: float,
    blink: float = 0.0,
    gaze: tuple[float, float] = (0.0, 0.0),
) -> EyeShape:
    """Compute the canonical (un-mirrored) eye shape for a mood + blink + gaze.

    Encoding follows Cozmo's actual expression vocabulary:
      - arousal -> openness (sleepy<->wide) AND width (alert/surprised widen).
      - positive valence -> lower lid raised into a happy arc.
      - negative valence -> upper lids droop, with the SLANT set by arousal:
        calm+sad -> OUTER corners drop ('\\ /', sad); aroused+sad -> INNER
        corners drop ('/ \\', a furrow, toward angry).
    The renderer mirrors inner/outer per eye (inner = toward the nose).
    """
    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))
    blink = _clamp(blink)

    # Openness from arousal; a blink collapses it. Width grows when alert.
    openness = _clamp(0.55 + 0.45 * arousal) * (1.0 - blink)
    width = 1.0 + 0.3 * max(0.0, arousal)

    # Positive valence raises the lower lid into a happy arc.
    lower_lid = _clamp(max(0.0, valence) * 0.5)

    # Negative valence droops the upper lids. Arousal skews the slant:
    # skew<0 (calm) -> outer drops (sad); skew>0 (aroused) -> inner drops (angry).
    droop = _clamp(max(0.0, -valence) * 0.7)
    skew = 0.5 * arousal
    upper_lid_inner = _clamp(droop * (1.0 + skew))
    upper_lid_outer = _clamp(droop * (1.0 - skew))

    gx = max(-1.0, min(1.0, gaze[0]))
    gy = max(-1.0, min(1.0, gaze[1]))
    return EyeShape(openness, lower_lid, upper_lid_inner, upper_lid_outer, width, gx, gy)
