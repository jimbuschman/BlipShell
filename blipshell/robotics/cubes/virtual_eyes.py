"""A smart "eyes" display cube — BlipShell's face as Cozmo-style procedural eyes.

Unlike the LED matrix (which BlipShell drives frame-by-frame), the eyes device is
SMART: BlipShell sends only the current mood via set_mood(valence, arousal); the
device itself renders the procedural eyes and runs its own keep-alive (blinks,
saccades) locally, so the eyes stay alive independent of BlipShell/LLM latency —
the way Cozmo/Vector eyes do. This class is the contract + state; the actual
rendering + keep-alive live in the eye window (scripts/eyes_window.py).
"""

from typing import Any

from blipshell.models.tools import ToolParameter, ToolParameterType
from blipshell.robotics.capability import ActionSpec, CubeMetadata
from blipshell.robotics.cube import Cube


class VirtualEyes(Cube):
    """An eye-display cube. Holds the latest commanded mood; the window tweens to it."""

    def __init__(self, cube_id: str = "eyes_01"):
        super().__init__()
        self.cube_id = cube_id
        # Latest mood BlipShell commanded. Defaults to neutral-calm.
        self.target_valence: float = 0.0
        self.target_arousal: float = -0.2

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
            ],
        )

    async def invoke(self, action: str, args: dict[str, Any]) -> str:
        if action == "set_mood":
            self.target_valence = max(-1.0, min(1.0, float(args.get("valence", 0.0))))
            self.target_arousal = max(-1.0, min(1.0, float(args.get("arousal", 0.0))))
            return f"mood set to ({self.target_valence:.2f}, {self.target_arousal:.2f})"
        raise ValueError(f"unsupported action '{action}'")
