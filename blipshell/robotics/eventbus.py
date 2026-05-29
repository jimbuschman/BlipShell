"""A minimal async pub/sub event bus for the robot core.

Cubes (input modules: microphones, motion sensors) publish events; behaviors
and the rules engine subscribe. This is the seam that, on real hardware,
becomes MQTT topics — keep the surface small so the swap is mechanical.

Events are fire-and-forget: a misbehaving subscriber is logged and isolated,
never allowed to take down the publisher or the other subscribers.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# An event handler receives the event name and an arbitrary JSON-able payload.
EventHandler = Callable[[str, dict[str, Any]], Awaitable[None]]


class EventBus:
    """In-process async publish/subscribe."""

    def __init__(self):
        self._subscribers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Register a handler for an event name."""
        self._subscribers.setdefault(event_name, []).append(handler)
        logger.debug("Subscribed handler to event '%s'", event_name)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Remove a previously registered handler. No-op if not present."""
        handlers = self._subscribers.get(event_name)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            pass
        if not handlers:
            self._subscribers.pop(event_name, None)

    async def publish(self, event_name: str, payload: dict[str, Any] | None = None) -> int:
        """Deliver an event to all subscribers concurrently.

        Returns the number of handlers invoked. Subscriber exceptions are
        logged and swallowed so one bad handler can't break delivery to the
        others — the rules engine must keep running.
        """
        handlers = list(self._subscribers.get(event_name, ()))
        if not handlers:
            return 0
        payload = payload or {}
        results = await asyncio.gather(
            *(h(event_name, payload) for h in handlers),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Event handler for '%s' failed: %s", event_name, result)
        return len(handlers)
