"""Render a mood (valence/arousal) as a simple procedural face on an NxN grid.

A coarse stand-in for the eventual Cozmo-style procedural eyes: eyes' openness
comes from arousal, the mouth's curve from valence. The emotion engine holds the
genuine state; this only *renders* it — exactly like the planned eyes, BlipShell
never authors these pixels. The 8x8 grid quantizes a continuous mood to a few
readable faces; the OLED eyes will be smooth.
"""


def render_face(valence: float, arousal: float, width: int = 8, height: int = 8) -> list[list[int]]:
    """Procedurally draw a face for the given mood. Returns a height x width 0/1 grid."""
    f = [[0] * width for _ in range(height)]

    # Eyes: open & alert when aroused, a single half-lidded row when calm/low.
    eye_l, eye_r = 2, width - 3            # columns 2 and 5 on an 8-wide grid
    if arousal >= 0.0:
        for r in (1, 2):                   # tall, open eyes
            f[r][eye_l] = 1
            f[r][eye_r] = 1
    else:
        f[2][eye_l] = 1                    # low/sleepy — half-closed
        f[2][eye_r] = 1

    # Mouth (rows height-3 / height-2): smile, flat, or frown by valence.
    top, bot = height - 3, height - 2      # rows 5 and 6 on an 8-tall grid
    out_l, out_r = 2, width - 3            # mouth corners (2, 5)
    in_l, in_r = 3, width - 4              # mouth middle (3, 4)
    if valence > 0.25:                     # smile: corners up, middle down (U)
        f[top][out_l] = f[top][out_r] = 1
        f[bot][in_l] = f[bot][in_r] = 1
    elif valence < -0.25:                  # frown: middle up, corners down
        f[top][in_l] = f[top][in_r] = 1
        f[bot][out_l] = f[bot][out_r] = 1
    else:                                  # flat
        for c in (out_l, in_l, in_r, out_r):
            f[bot][c] = 1
    return f
