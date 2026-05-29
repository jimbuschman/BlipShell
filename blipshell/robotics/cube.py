"""The cube interface contract.

This is the boundary every module — virtual today, ESP32 firmware tomorrow —
must satisfy. Two responsibilities only:

    describe()  -> advertise capabilities as CubeMetadata
    invoke()    -> deterministically perform one action, return a result string

The deterministic layer lives *behind* invoke(). The LLM never reaches past
this interface: it reads capabilities and requests actions; the cube owns all
hardware/timing/safety logic and enforces its own physical constraints.

Input cubes (sensors) report events by calling self.emit(); the registry wires
that to the event bus on connect. Output cubes (LED matrix) simply never emit.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from blipshell.robotics.capability import CubeMetadata

logger = logging.getLogger(__name__)

# Set by the registry on connect: (event_name, payload) -> awaitable
EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


class Cube(ABC):
    """Abstract base for a hardware module (or its virtual stand-in)."""

    def __init__(self):
        self._event_sink: EventSink | None = None

    @abstractmethod
    def describe(self) -> CubeMetadata:
        """Return this cube's capability metadata (broadcast on connect)."""
        ...

    @abstractmethod
    async def invoke(self, action: str, args: dict[str, Any]) -> str:
        """Perform one action deterministically and return a result string.

        Implementations should enforce their own *hardware* constraints here
        (frame dimensions, value ranges) and raise ValueError with an
        actionable message on violation. Structural validation (does the
        action exist, are required args present) is the registry's job.
        """
        ...

    def snapshot(self) -> Any:
        """Return a comparable snapshot of this cube's *observable* state.

        Used by the behavior tracer to (a) detect when one action's visible
        output is immediately replaced by the next, and (b) restore state after
        a dry-run so tracing never disturbs the live cube. The value must be
        equality-comparable; two equal snapshots mean nothing observable
        changed. Default None — a cube with no observable state (e.g. a pure
        sensor) is simply never flagged.
        """
        return None

    def restore(self, snap: Any) -> None:
        """Restore a snapshot taken by snapshot(). Default no-op."""
        return None

    def set_event_sink(self, sink: EventSink | None) -> None:
        """Wire (or clear) the bus this cube publishes events into."""
        self._event_sink = sink

    async def emit(self, event_name: str, payload: dict[str, Any] | None = None) -> None:
        """Publish an event to the core. No-op if not connected to a bus."""
        if self._event_sink is None:
            logger.debug("Cube emitted '%s' with no event sink wired; dropped", event_name)
            return
        await self._event_sink(event_name, payload or {})
