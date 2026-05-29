"""Modular cube robotics — software-first.

A self-describing hardware cube broadcasts its capabilities; the capability
registry tracks connected cubes and is the single validated path through which
actions are dispatched. The LLM (via BlipShell) reasons over the normalized
capability surface and authors behavior — it never touches hardware directly.

Everything here runs without hardware: virtual cubes implement the same
contract a real ESP32 module will, so the LLM -> validate -> execute loop and
the connect/disconnect lifecycle can be exercised entirely in software before
anything is soldered.
"""

from blipshell.robotics.capability import ActionSpec, CubeMetadata
from blipshell.robotics.cube import Cube
from blipshell.robotics.eventbus import EventBus
from blipshell.robotics.registry import CapabilityRegistry

__all__ = [
    "ActionSpec",
    "CubeMetadata",
    "Cube",
    "EventBus",
    "CapabilityRegistry",
]
