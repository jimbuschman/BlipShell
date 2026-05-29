"""A software stand-in for an LED-matrix cube.

Implements the same describe()/invoke() contract a real ESP32 LED-matrix module
would, but instead of driving pixels it holds the display state in memory and
logs. This lets the whole LLM -> validate -> execute loop run with no hardware.

It demonstrates the constraint-enforcement split: structural validation (action
exists, required args present) is handled by the registry; *hardware* limits
(a frame must match the panel's exact dimensions) are enforced here, where the
"firmware" lives, and surfaced as actionable ValueErrors.
"""

import logging
from typing import Any

from blipshell.models.tools import ToolParameter, ToolParameterType
from blipshell.robotics.capability import ActionSpec, CubeMetadata
from blipshell.robotics.cube import Cube

logger = logging.getLogger(__name__)


class VirtualLEDMatrix(Cube):
    """An in-memory width x height LED matrix (default 8x8)."""

    def __init__(self, cube_id: str = "led_matrix_01", width: int = 8, height: int = 8):
        super().__init__()
        self.cube_id = cube_id
        self.width = width
        self.height = height
        # Observable state — what's currently "shown". Tests assert on these.
        self.last_text: str | None = None
        self.frame: list[list[int]] = self._blank_frame()

    def _blank_frame(self) -> list[list[int]]:
        return [[0] * self.width for _ in range(self.height)]

    def snapshot(self):
        """Observable state = the text cue and the pixel frame (immutable copy)."""
        return (self.last_text, tuple(tuple(row) for row in self.frame))

    def restore(self, snap) -> None:
        if snap is None:
            return
        self.last_text, frame = snap
        self.frame = [list(row) for row in frame]

    def describe(self) -> CubeMetadata:
        return CubeMetadata(
            cube_id=self.cube_id,
            module_type="led_matrix",
            description=(
                f"{self.width}x{self.height} LED matrix — a small attention/status "
                "display surface. Good for short notifications, listening/thinking "
                "indicators, emotional cues, and simple animations. Not for long text."
            ),
            actions=[
                ActionSpec(
                    name="display_text",
                    description="Scroll a short text string across the matrix.",
                    parameters=[
                        ToolParameter(
                            name="text",
                            type=ToolParameterType.STRING,
                            description="The text to display (scrolls if wider than the panel).",
                        ),
                    ],
                ),
                ActionSpec(
                    name="display_frame",
                    description=(
                        f"Set every pixel at once. Requires a {self.height}x{self.width} "
                        "array of 0/1 rows."
                    ),
                    parameters=[
                        ToolParameter(
                            name="frame",
                            type=ToolParameterType.ARRAY,
                            description=(
                                f"{self.height} rows of {self.width} values, each 0 (off) "
                                "or 1 (on)."
                            ),
                        ),
                    ],
                ),
                ActionSpec(
                    name="clear",
                    description="Turn all pixels off.",
                    parameters=[],
                ),
            ],
            constraints={"width": self.width, "height": self.height},
        )

    async def invoke(self, action: str, args: dict[str, Any]) -> str:
        if action == "display_text":
            text = str(args.get("text", ""))
            self.last_text = text
            logger.info("[%s] display_text: %r", self.cube_id, text)
            return f"Displaying '{text}' on {self.cube_id}."

        if action == "display_frame":
            frame = args.get("frame")
            self._validate_frame(frame)
            self.frame = [list(row) for row in frame]
            self.last_text = None
            logger.info("[%s] display_frame applied", self.cube_id)
            return f"Frame applied to {self.cube_id}."

        if action == "clear":
            self.frame = self._blank_frame()
            self.last_text = None
            logger.info("[%s] cleared", self.cube_id)
            return f"{self.cube_id} cleared."

        # Should be unreachable — the registry validates the action exists first.
        raise ValueError(f"unsupported action '{action}'")

    def _validate_frame(self, frame: Any) -> None:
        """Enforce the panel's exact dimensions. Raises actionable ValueError."""
        if not isinstance(frame, list) or not all(isinstance(row, list) for row in frame):
            raise ValueError(
                f"frame must be a {self.height}x{self.width} array of rows; "
                f"got {type(frame).__name__}"
            )
        if len(frame) != self.height or any(len(row) != self.width for row in frame):
            got = f"{len(frame)}x{len(frame[0]) if frame else 0}"
            raise ValueError(
                f"frame must be exactly {self.height}x{self.width}; got {got}"
            )
        if any(v not in (0, 1) for row in frame for v in row):
            raise ValueError("frame values must each be 0 (off) or 1 (on)")
