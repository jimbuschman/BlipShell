"""Faithful port of the esp32-eyes EyeConfig model — the real OLED eye renderer.

Mirrors playfultechnology/esp32-eyes (GPL-AGPL, Luis Llamas / Alastair Aitchison):
the same EyeConfig parameters and the same 18 emotion presets the ESP32 firmware
uses, so the desktop sim renders what the 0.96" OLED hardware will show and the
two share one set of expression definitions.

An eye is a rounded rectangle (Height x Width, with separate top/bottom corner
radii) whose top and bottom edges can slope: +slope_top tilts the top edge down
toward the face's middle (a furrow → angry/focused), −slope_top tilts it down
toward the outside (→ sad/worried). eye_outline() returns the polygon a renderer
fills; reactions are named presets; mood (continuous valence/arousal) maps to a
config via mood_to_config(). (The esp32-eyes Inverse_* params aren't used by its
Draw routine, so they're omitted.)
"""

import math
from dataclasses import dataclass, replace


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class EyeConfig:
    offset_x: float = 0.0
    offset_y: float = 0.0
    height: float = 40.0
    width: float = 40.0
    slope_top: float = 0.0      # + = inner-down (furrow/angry); − = outer-down (sad)
    slope_bottom: float = 0.0
    radius_top: float = 8.0
    radius_bottom: float = 8.0


# The 18 emotion presets, transcribed from esp32-eyes EyePresets.h (the "_Alt"
# variants are dropped; one canonical config per emotion).
PRESETS: dict[str, EyeConfig] = {
    "normal":      EyeConfig(0, 0, 40, 40, 0.0, 0.0, 8, 8),
    "happy":       EyeConfig(0, 0, 10, 40, 0.0, 0.0, 10, 0),
    "glee":        EyeConfig(0, 0, 8, 40, 0.0, 0.0, 8, 0),
    "sad":         EyeConfig(0, 0, 15, 40, -0.5, 0.0, 1, 10),
    "worried":     EyeConfig(0, 0, 25, 40, -0.1, 0.0, 6, 10),
    "focused":     EyeConfig(0, 0, 14, 40, 0.2, 0.0, 3, 1),
    "annoyed":     EyeConfig(0, 0, 12, 40, 0.0, 0.0, 0, 10),
    "surprised":   EyeConfig(-2, 0, 45, 45, 0.0, 0.0, 16, 16),
    "skeptic":     EyeConfig(0, 0, 40, 40, 0.0, 0.0, 10, 10),
    "frustrated":  EyeConfig(3, -5, 12, 40, 0.0, 0.0, 0, 10),
    "unimpressed": EyeConfig(3, 0, 12, 40, 0.0, 0.0, 1, 10),
    "sleepy":      EyeConfig(0, -2, 14, 40, -0.5, -0.5, 3, 3),
    "suspicious":  EyeConfig(0, 0, 22, 40, 0.0, 0.0, 8, 3),
    "squint":      EyeConfig(-10, -3, 35, 35, 0.0, 0.0, 8, 8),
    "angry":       EyeConfig(-3, 0, 20, 40, 0.3, 0.0, 2, 12),
    "furious":     EyeConfig(-2, 0, 30, 40, 0.4, 0.0, 2, 8),
    "scared":      EyeConfig(-3, 0, 40, 40, -0.1, 0.0, 12, 8),
    "awe":         EyeConfig(2, 0, 35, 45, -0.1, 0.1, 12, 12),
}


def lerp_config(a: EyeConfig, b: EyeConfig, t: float) -> EyeConfig:
    """Interpolate every field for smooth tweening between two configs."""
    t = _clamp(t, 0.0, 1.0)
    def f(x, y):
        return x + (y - x) * t
    return EyeConfig(
        offset_x=f(a.offset_x, b.offset_x), offset_y=f(a.offset_y, b.offset_y),
        height=f(a.height, b.height), width=f(a.width, b.width),
        slope_top=f(a.slope_top, b.slope_top), slope_bottom=f(a.slope_bottom, b.slope_bottom),
        radius_top=f(a.radius_top, b.radius_top), radius_bottom=f(a.radius_bottom, b.radius_bottom),
    )


def mirror_config(c: EyeConfig) -> EyeConfig:
    """Mirror a config for the other eye (negate horizontal slope + offset) so a
    furrow/sad slant is symmetric across both eyes."""
    return replace(c, slope_top=-c.slope_top, slope_bottom=-c.slope_bottom, offset_x=-c.offset_x)


def with_blink(c: EyeConfig, blink: float) -> EyeConfig:
    """Collapse the eye height for a blink (0 = open, 1 = shut)."""
    return replace(c, height=max(3.0, c.height * (1.0 - _clamp(blink, 0.0, 1.0))))


def mood_to_config(valence: float, arousal: float) -> EyeConfig:
    """Map a continuous mood to an EyeConfig (our addition; esp32-eyes only has
    discrete presets). Tunable. arousal -> height/width/openness; positive
    valence -> short happy squish; negative valence -> sloped lids, with the
    slant sign set by arousal (calm = outer/sad, aroused = inner/angry furrow).
    """
    v = _clamp(valence, -1.0, 1.0)
    a = _clamp(arousal, -1.0, 1.0)
    pos, neg = max(0.0, v), max(0.0, -v)

    height = (16.0 + 22.0 * (a + 1.0) / 2.0) * (1.0 - 0.55 * pos)
    height = max(8.0, height)
    width = 38.0 + 8.0 * max(0.0, a)
    slope_top = neg * 0.5 * _clamp(a / 0.4, -1.0, 1.0)
    radius_top = _clamp(6.0 + 8.0 * max(0.0, a), 2.0, 16.0)
    radius_bottom = 2.0 if pos > 0.4 else radius_top
    return EyeConfig(0.0, 0.0, height, width, slope_top, 0.0, radius_top, radius_bottom)


def eye_outline(c: EyeConfig, cx: float, cy: float, samples: int = 6) -> list[tuple[float, float]]:
    """Polygon outline (logical coords) for one eye — a renderer fills this.

    Rounded rectangle with per-corner radii, then a y-shear applied to the top
    and bottom so the edges slope (matching esp32-eyes' delta = Height*Slope/2).
    """
    ecx, ecy = cx + c.offset_x, cy + c.offset_y
    w, h = max(2.0, c.width), max(2.0, c.height)
    left, right = ecx - w / 2, ecx + w / 2
    top, bottom = ecy - h / 2, ecy + h / 2
    rt = _clamp(c.radius_top, 0.0, min(w, h) / 2)
    rb = _clamp(c.radius_bottom, 0.0, min(w, h) / 2)

    pts: list[tuple[float, float]] = []

    def arc(cxc, cyc, r, a0, a1):
        if r <= 0:
            return
        for i in range(samples + 1):
            ang = a0 + (a1 - a0) * i / samples
            pts.append((cxc + r * math.cos(ang), cyc + r * math.sin(ang)))

    # Clockwise (screen y down): TL -> TR -> BR -> BL corners.
    if rt > 0:
        arc(left + rt, top + rt, rt, math.pi, 1.5 * math.pi)        # top-left
        arc(right - rt, top + rt, rt, 1.5 * math.pi, 2 * math.pi)   # top-right
    else:
        pts.append((left, top)); pts.append((right, top))
    if rb > 0:
        arc(right - rb, bottom - rb, rb, 0.0, 0.5 * math.pi)        # bottom-right
        arc(left + rb, bottom - rb, rb, 0.5 * math.pi, math.pi)     # bottom-left
    else:
        pts.append((right, bottom)); pts.append((left, bottom))

    # Slope: shear top points by slope_top, bottom points by slope_bottom. The
    # shift at the eye's edge equals Height*Slope/2 (esp32-eyes' delta).
    shear_top = h * c.slope_top / w
    shear_bot = h * c.slope_bottom / w
    out = []
    for x, y in pts:
        shift = shear_top if y <= ecy else shear_bot
        out.append((x, y + (x - ecx) * shift))
    return out
