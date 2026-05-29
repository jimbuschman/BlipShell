"""The Event-Condition-Action rules engine — the robot's reflexes.

This is the deterministic runtime the LLM never sits inside. The LLM *authors*
behaviors (offline, as data); the engine *runs* them locally with no LLM in the
loop, so a `speech_detected` event can light the matrix in milliseconds.

A Behavior is exactly the JSON shape the design doc specifies:

    {"trigger": "speech_detected",
     "actions": [{"target": "led_matrix_01", "action": "display_text",
                  "args": {"text": "Listening..."}}]}

Every action is dispatched through the CapabilityRegistry, so it inherits the
same validation and actionable-error handling. A rule that targets a cube which
has since disconnected logs a clean error and moves on — it never fires into a
void or crashes the engine.

Conflict arbitration (two behaviors driving one cube at once) is intentionally
*not* solved here yet — actions run in deterministic load order and that's the
documented limit. It's the next hard problem, called out so it isn't mistaken
for handled.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from blipshell.robotics.registry import CapabilityRegistry

logger = logging.getLogger(__name__)

_MAX_RESULT_HISTORY = 100


class BehaviorAction(BaseModel):
    """One action a behavior performs when triggered."""

    target: str  # cube_id to dispatch to
    action: str  # action name on that cube
    args: dict[str, Any] = Field(default_factory=dict)


class Behavior(BaseModel):
    """A trigger -> actions rule. Mirrors the LLM-authored behavior JSON."""

    trigger: str  # event name that fires this behavior
    actions: list[BehaviorAction] = Field(default_factory=list)
    name: str | None = None  # optional label for logs/observability
    # What the author expects the user to observe. Lets the trace/review step
    # judge whether the behavior's actual visible effect matches the goal.
    intent: str | None = None


class RulesEngine:
    """Subscribes behaviors to the event bus and dispatches their actions."""

    def __init__(self, registry: CapabilityRegistry):
        self._registry = registry
        self._bus = registry.event_bus
        self._by_trigger: dict[str, list[Behavior]] = {}
        # Recent (event, target, action, result) tuples for observability.
        self.last_results: list[dict[str, Any]] = []

    def load(self, behaviors: list[Behavior]) -> None:
        """Replace the active rule set and (re)subscribe to its triggers.

        Idempotent: prior subscriptions are removed first, so reloading an
        updated behavior set doesn't double-fire.
        """
        for trigger in self._by_trigger:
            self._bus.unsubscribe(trigger, self._handle)

        self._by_trigger = {}
        for behavior in behaviors:
            self._by_trigger.setdefault(behavior.trigger, []).append(behavior)

        for trigger in self._by_trigger:
            self._bus.subscribe(trigger, self._handle)

        logger.info("Rules engine loaded %d behavior(s) across %d trigger(s)",
                    len(behaviors), len(self._by_trigger))

    @property
    def triggers(self) -> set[str]:
        """Event names this engine currently reacts to."""
        return set(self._by_trigger)

    async def _handle(self, event_name: str, payload: dict[str, Any]) -> None:
        """Run every behavior bound to a fired event, in load order."""
        for behavior in self._by_trigger.get(event_name, ()):
            for act in behavior.actions:
                result = await self._registry.invoke(act.target, act.action, act.args)
                self._record(event_name, behavior, act, result)
                if result.lower().startswith("error"):
                    logger.warning("Behavior %s: %s",
                                   behavior.name or event_name, result)

    def _record(self, event_name, behavior, act, result) -> None:
        self.last_results.append({
            "event": event_name,
            "behavior": behavior.name,
            "target": act.target,
            "action": act.action,
            "result": result,
        })
        if len(self.last_results) > _MAX_RESULT_HISTORY:
            del self.last_results[:-_MAX_RESULT_HISTORY]
