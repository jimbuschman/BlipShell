"""Capability schema for self-describing hardware cubes.

A cube (an ESP32 module today, a virtual stand-in during software bring-up)
advertises *what it can do* as structured metadata. The LLM never sees raw
hardware — only this normalized capability surface:

    - actions  : what the cube can be told to do (display_text, clear, ...)
    - events   : what the cube can report happening (motion_detected, ...)
    - constraints: physical limits the deterministic layer enforces (4x4, ...)

Action parameters reuse :class:`ToolParameter` so a cube's actions map directly
onto the LLM tool schema later — one schema, no translation layer to drift.
"""

from typing import Any

from pydantic import BaseModel, Field

from blipshell.models.tools import ToolParameter


class ActionSpec(BaseModel):
    """One thing a cube can be told to do — a single safe, tested API call.

    The deterministic firmware implements the action; this is just its
    advertised, validatable signature.
    """

    name: str  # e.g. "display_text"
    description: str  # human/LLM-facing: what it does, when to use it
    parameters: list[ToolParameter] = Field(default_factory=list)


class CubeMetadata(BaseModel):
    """Everything a cube broadcasts about itself when it connects.

    This is the contract the capability registry stores and the LLM reasons
    over. ``cube_id`` identifies a *physical instance* (two LED matrices have
    distinct ids); ``module_type`` identifies the *kind* of module.
    """

    cube_id: str  # unique instance id, e.g. "led_matrix_01"
    module_type: str  # kind of module, e.g. "led_matrix"
    description: str  # semantic hint for the LLM, e.g. "4x4 status display surface"
    actions: list[ActionSpec] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)  # event names this cube may emit
    constraints: dict[str, Any] = Field(default_factory=dict)  # physical limits

    def get_action(self, name: str) -> ActionSpec | None:
        """Return the named action spec, or None if this cube has no such action."""
        for action in self.actions:
            if action.name == name:
                return action
        return None
