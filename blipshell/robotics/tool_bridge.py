"""Bridge cube capabilities into BlipShell's LLM tool registry.

This is the "can-invoke" half of using cubes reliably: when a cube connects,
each of its actions is auto-registered as a Tool named ``cube_<id>_<action>``,
so the action appears in the LLM's tool list on *every* turn — the model never
has to remember the cube exists. When the cube disconnects, those tools are
torn down, so the tool list never advertises a capability that's gone.

No LLM-generated code is involved: an ActionSpec already carries a name,
description, and ToolParameter list, so the mapping to a ToolDefinition is a
direct, deterministic transform.
"""

import logging

from blipshell.core.tools.base import Tool, ToolRegistry
from blipshell.models.tools import ToolDefinition
from blipshell.robotics.capability import ActionSpec, CubeMetadata
from blipshell.robotics.registry import CapabilityRegistry

logger = logging.getLogger(__name__)


def _sanitize(part: str) -> str:
    """Make an id/action safe for a tool name (alnum + underscore only)."""
    return "".join(c if c.isalnum() else "_" for c in part)


def tool_name_for(cube_id: str, action: str) -> str:
    """The deterministic tool name for a cube action."""
    return f"cube_{_sanitize(cube_id)}_{_sanitize(action)}"


class CubeActionTool(Tool):
    """Wraps one cube action as an LLM-callable tool.

    Dispatch goes through the CapabilityRegistry, not the cube directly, so the
    same validation (cube still connected, args present) and actionable error
    strings apply whether the caller is the LLM or the rules engine.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        cube_id: str,
        meta_description: str,
        spec: ActionSpec,
    ):
        self._registry = registry
        self._cube_id = cube_id
        self._meta_description = meta_description
        self._spec = spec
        self._name = tool_name_for(cube_id, spec.name)

    def definition(self) -> ToolDefinition:
        # Prepend the cube's semantic role so the model knows what the surface
        # is *for*, not just the bare action verb.
        description = f"[{self._cube_id} — {self._meta_description}] {self._spec.description}"
        return ToolDefinition(
            name=self._name,
            description=description,
            parameters=self._spec.parameters,
        )

    async def execute(self, **kwargs) -> str:
        return await self._registry.invoke(self._cube_id, self._spec.name, kwargs)


class CubeToolBridge:
    """Keeps a ToolRegistry in sync with connected cubes.

    Attach once; thereafter every cube connect/disconnect adds or removes the
    matching tools automatically.
    """

    def __init__(
        self,
        capability_registry: CapabilityRegistry,
        tool_registry: ToolRegistry,
        group: str = "robotics",
    ):
        self._caps = capability_registry
        self._tools = tool_registry
        self._group = group
        # cube_id -> the tool names registered for it, so disconnect can undo.
        self._registered: dict[str, list[str]] = {}

    def attach(self) -> None:
        """Wire connect/disconnect listeners. Idempotent per registry."""
        self._caps.add_connect_listener(self._on_connect)
        self._caps.add_disconnect_listener(self._on_disconnect)

    def _on_connect(self, meta: CubeMetadata) -> None:
        names: list[str] = []
        for spec in meta.actions:
            tool = CubeActionTool(self._caps, meta.cube_id, meta.description, spec)
            self._tools.register(tool, group=self._group)
            names.append(tool.definition().name)
        self._registered[meta.cube_id] = names
        logger.info("Registered %d tool(s) for cube '%s'", len(names), meta.cube_id)

    def _on_disconnect(self, meta: CubeMetadata) -> None:
        names = self._registered.pop(meta.cube_id, [])
        for name in names:
            self._tools.unregister(name)
        logger.info("Unregistered %d tool(s) for cube '%s'", len(names), meta.cube_id)
