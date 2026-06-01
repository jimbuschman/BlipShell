"""A smart "eyes" display cube — BlipShell's face as Cozmo-style procedural eyes.

Unlike the LED matrix (which BlipShell drives frame-by-frame), the eyes device is
SMART: BlipShell sends only the current mood via set_mood(valence, arousal); the
device itself renders the procedural eyes and runs its own keep-alive (blinks,
saccades) locally, so the eyes stay alive independent of BlipShell/LLM latency —
the way Cozmo/Vector eyes do. This class is the contract + state; the actual
rendering + keep-alive live in the eye window (scripts/eyes_window.py).
"""

import time
from typing import Any

from blipshell.models.tools import ToolParameter, ToolParameterType
from blipshell.robotics.capability import ActionSpec, CubeMetadata
from blipshell.robotics.cube import Cube
from blipshell.robotics.eye_config import PRESETS


class VirtualEyes(Cube):
    """An eye-display cube. Holds the latest commanded mood; the window tweens to it.

    Also holds a transient *reaction*: a named expression that briefly overrides
    the mood, then expires on its own — the device handles the override/revert
    locally, so it's smooth and independent of BlipShell/LLM latency.
    """

    def __init__(self, cube_id: str = "eyes_01"):
        super().__init__()
        self.cube_id = cube_id
        # Latest mood BlipShell commanded. Defaults to neutral-calm.
        self.target_valence: float = 0.0
        self.target_arousal: float = -0.2
        # Transient reaction (name + monotonic expiry).
        self.current_reaction: str | None = None
        self.reaction_until: float = 0.0

    def describe(self) -> CubeMetadata:
        return CubeMetadata(
            cube_id=self.cube_id,
            module_type="eyes",
            description=(
                "A pair of expressive Cozmo-style eyes (a small OLED face). It is "
                "your face: it shows your current mood on its own and stays alive "
                "(blinks, glances) by itself. Drive it by setting your mood, not "
                "by drawing — set_mood takes valence (-1 sad .. +1 happy) and "
                "arousal (-1 calm/sleepy .. +1 alert/energized)."
            ),
            actions=[
                ActionSpec(
                    name="set_mood",
                    description="Set the mood the eyes express (valence and arousal, each -1..1).",
                    parameters=[
                        ToolParameter(name="valence", type=ToolParameterType.NUMBER,
                                      description="-1 (negative) .. +1 (positive)"),
                        ToolParameter(name="arousal", type=ToolParameterType.NUMBER,
                                      description="-1 (calm/sleepy) .. +1 (alert/energized)"),
                    ],
                ),
                ActionSpec(
                    name="play_reaction",
                    description=("Briefly show a named expression, then settle back to the "
                                 "mood. One of: " + ", ".join(sorted(PRESETS))),
                    parameters=[
                        ToolParameter(name="emotion", type=ToolParameterType.STRING,
                                      description="expression name (e.g. surprised, glee, annoyed)"),
                        ToolParameter(name="duration", type=ToolParameterType.NUMBER,
                                      description="seconds to hold (default 1.5)", required=False),
                    ],
                ),
            ],
        )

    async def invoke(self, action: str, args: dict[str, Any]) -> str:
        if action == "set_mood":
            self.target_valence = max(-1.0, min(1.0, float(args.get("valence", 0.0))))
            self.target_arousal = max(-1.0, min(1.0, float(args.get("arousal", 0.0))))
            return f"mood set to ({self.target_valence:.2f}, {self.target_arousal:.2f})"
        if action == "play_reaction":
            emotion = str(args.get("emotion", "")).lower()
            if emotion not in PRESETS:
                return (f"Error: unknown expression '{emotion}'. "
                        f"Available: {', '.join(sorted(PRESETS))}.")
            duration = float(args.get("duration", 1.5) or 1.5)
            self.current_reaction = emotion
            self.reaction_until = time.monotonic() + max(0.1, duration)
            return f"reacting: {emotion} for {duration:.1f}s"
        raise ValueError(f"unsupported action '{action}'")

    def active_reaction(self) -> str | None:
        """The reaction currently overriding the mood, or None if expired."""
        if self.current_reaction and time.monotonic() < self.reaction_until:
            return self.current_reaction
        return None
