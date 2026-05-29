"""The capability registry — the robot core's source of truth.

Tracks which cubes are connected, what they can do, and is the *only* path
through which actions are dispatched. Every invoke() is validated here before
it reaches a cube:

    - is a cube with this id still connected?   (cubes can vanish at any moment)
    - does that cube actually have this action?
    - are the action's required parameters present?

Structural validation lives here; hardware/semantic constraints live in the
cube. On failure the registry returns an actionable error string — the same
"errors are prompts" discipline the executor tools use — so an LLM caller can
self-correct.

connect/disconnect fire listener callbacks so higher layers (LLM tool
registration, the rules engine) can react. Critically, disconnect tears the
cube out completely: its capabilities disappear and any later invoke() against
it fails cleanly rather than firing an action into the void.
"""

import logging
from typing import Any, Awaitable, Callable

from blipshell.robotics.capability import ActionSpec, CubeMetadata
from blipshell.robotics.cube import Cube
from blipshell.robotics.eventbus import EventBus

logger = logging.getLogger(__name__)

# Listener fired on connect/disconnect: (metadata) -> optional awaitable
ConnectListener = Callable[[CubeMetadata], Awaitable[None] | None]


class CapabilityRegistry:
    """Registers cubes, validates, and dispatches actions to them."""

    def __init__(self, event_bus: EventBus | None = None):
        self._cubes: dict[str, Cube] = {}
        self._metadata: dict[str, CubeMetadata] = {}
        self._event_bus = event_bus or EventBus()
        self._on_connect: list[ConnectListener] = []
        self._on_disconnect: list[ConnectListener] = []

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    # --- lifecycle listeners -------------------------------------------------

    def add_connect_listener(self, listener: ConnectListener) -> None:
        """Register a callback fired after a cube connects."""
        self._on_connect.append(listener)

    def add_disconnect_listener(self, listener: ConnectListener) -> None:
        """Register a callback fired after a cube disconnects."""
        self._on_disconnect.append(listener)

    async def _fire(self, listeners: list[ConnectListener], metadata: CubeMetadata) -> None:
        for listener in listeners:
            try:
                result = listener(metadata)
                if result is not None:
                    await result
            except Exception as e:
                logger.warning("Cube lifecycle listener failed for '%s': %s",
                               metadata.cube_id, e)

    # --- connect / disconnect ------------------------------------------------

    async def connect(self, cube: Cube) -> CubeMetadata:
        """Register a cube, wire its event sink, and notify listeners.

        Reconnecting a cube_id that's already present replaces it (treated as
        a fresh connect) — the prior instance is disconnected first so its
        listeners and event wiring don't leak.
        """
        metadata = cube.describe()
        if metadata.cube_id in self._cubes:
            logger.info("Cube '%s' reconnecting; replacing existing registration",
                        metadata.cube_id)
            await self.disconnect(metadata.cube_id)

        self._cubes[metadata.cube_id] = cube
        self._metadata[metadata.cube_id] = metadata
        cube.set_event_sink(self._event_bus.publish)
        logger.info("Cube connected: %s (%s) — %d action(s)",
                    metadata.cube_id, metadata.module_type, len(metadata.actions))
        await self._fire(self._on_connect, metadata)
        return metadata

    async def disconnect(self, cube_id: str) -> bool:
        """Remove a cube. Returns False if it wasn't connected.

        After this returns, the cube's capabilities are gone and any invoke()
        against ``cube_id`` fails cleanly. Listeners (e.g. LLM tool teardown)
        fire so nothing keeps advertising a capability that no longer exists.
        """
        cube = self._cubes.pop(cube_id, None)
        metadata = self._metadata.pop(cube_id, None)
        if cube is None or metadata is None:
            return False
        cube.set_event_sink(None)  # stop it publishing into the bus
        logger.info("Cube disconnected: %s", cube_id)
        await self._fire(self._on_disconnect, metadata)
        return True

    # --- introspection -------------------------------------------------------

    def is_connected(self, cube_id: str) -> bool:
        return cube_id in self._cubes

    def get_metadata(self, cube_id: str) -> CubeMetadata | None:
        return self._metadata.get(cube_id)

    def list_cubes(self) -> list[CubeMetadata]:
        """All currently connected cubes' metadata."""
        return list(self._metadata.values())

    def list_capabilities(self) -> list[tuple[str, ActionSpec]]:
        """Flat (cube_id, action) list across all connected cubes.

        This is the normalized surface the LLM reasons over — every action it
        could currently take, with no hardware detail leaking through.
        """
        return [
            (cube_id, action)
            for cube_id, meta in self._metadata.items()
            for action in meta.actions
        ]

    # --- dispatch ------------------------------------------------------------

    async def invoke(self, cube_id: str, action: str, args: dict[str, Any] | None = None) -> str:
        """Validate and dispatch an action to a connected cube.

        Returns the cube's result string, or an actionable error string if the
        target/action/args don't check out. Never raises for caller mistakes —
        the error text is the feedback.
        """
        args = args or {}

        cube = self._cubes.get(cube_id)
        metadata = self._metadata.get(cube_id)
        if cube is None or metadata is None:
            connected = ", ".join(self._cubes) or "none"
            return (f"Error: no cube '{cube_id}' is connected. "
                    f"Currently connected: {connected}.")

        spec = metadata.get_action(action)
        if spec is None:
            available = ", ".join(a.name for a in metadata.actions) or "none"
            return (f"Error: cube '{cube_id}' has no action '{action}'. "
                    f"Available actions: {available}.")

        missing = [
            p.name for p in spec.parameters
            if p.required and p.name not in args
        ]
        if missing:
            return (f"Error: action '{action}' on '{cube_id}' is missing required "
                    f"argument(s): {', '.join(missing)}.")

        try:
            return await cube.invoke(action, args)
        except ValueError as e:
            # Cube-enforced hardware/semantic constraint violation — actionable.
            return f"Error: {action} on '{cube_id}' rejected the request: {e}"
        except Exception as e:
            logger.error("Cube '%s' action '%s' failed: %s", cube_id, action, e)
            return f"Error: {action} on '{cube_id}' failed unexpectedly: {e}"
