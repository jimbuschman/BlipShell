"""Behavior tracer — observes what a behavior *actually* renders.

This is the feedback the LLM lacked: it authored behaviors blind, with no way
to know that "display HI then clear" makes HI flash for zero time. The tracer
dry-runs each behavior against the live cube (snapshotting and restoring state,
so it's non-destructive) and reports problems it *observes* — it does not encode
any domain rule about what actions mean.

The core observation is fully generic: actions in a behavior run instantly in
sequence (no delay exists), so if an action changes a cube's observable state
and the very next action on that same cube changes it again, the first output
was never visible. We detect that purely by comparing state snapshots before
and after each action — nothing here "knows" that clear blanks a display or
that display_text shows text. It just sees state replaced with no time between.
"""

import logging

from pydantic import BaseModel

from blipshell.robotics.registry import CapabilityRegistry
from blipshell.robotics.rules import Behavior

logger = logging.getLogger(__name__)


class TraceIssue(BaseModel):
    """One observed problem with a behavior's visible effect."""

    behavior_label: str
    action_index: int
    target: str
    action: str
    problem: str  # plain-language description, fed back to the LLM


async def trace_behaviors(
    behaviors: list[Behavior], registry: CapabilityRegistry,
) -> list[TraceIssue]:
    """Dry-run every behavior and collect observed issues. Non-destructive."""
    issues: list[TraceIssue] = []
    for behavior in behaviors:
        issues.extend(await _trace_one(behavior, registry))
    return issues


async def _trace_one(behavior: Behavior, registry: CapabilityRegistry) -> list[TraceIssue]:
    label = behavior.name or behavior.trigger
    targets = {a.target for a in behavior.actions if registry.is_connected(a.target)}
    if not targets:
        return []

    # Snapshot every involved cube so we can restore after the dry-run.
    snaps = {t: registry.get_cube(t).snapshot() for t in targets}
    changed: list[bool] = []
    try:
        for act in behavior.actions:
            cube = registry.get_cube(act.target)
            if cube is None:
                changed.append(False)
                continue
            before = cube.snapshot()
            await registry.invoke(act.target, act.action, act.args)
            changed.append(cube.snapshot() != before)
    finally:
        for cube_id, snap in snaps.items():
            cube = registry.get_cube(cube_id)
            if cube is not None:
                cube.restore(snap)

    # An action whose output is immediately overwritten by the next action on
    # the same target was never observable.
    issues: list[TraceIssue] = []
    for i, act in enumerate(behavior.actions):
        if not changed[i] or i + 1 >= len(behavior.actions):
            continue
        nxt = behavior.actions[i + 1]
        if nxt.target == act.target and changed[i + 1]:
            issues.append(TraceIssue(
                behavior_label=label,
                action_index=i,
                target=act.target,
                action=act.action,
                problem=(
                    f"the output of '{act.action}' was never visible — the next "
                    f"action ('{nxt.action}') changed '{act.target}' immediately, "
                    f"with no delay between them"
                ),
            ))
    return issues
