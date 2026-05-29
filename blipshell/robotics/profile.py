"""LLM-authored capability profiles — the "plugin the model writes itself".

When a cube connects, the LLM is asked what the new capability is *for* and
what behaviors it enables. Crucially the output is DATA, not code: a
CapabilityProfile carrying a semantic role, intended uses, usage guidance, and
a list of trigger->action Behaviors. The deterministic rules engine runs the
behaviors; the model never produces executable hardware logic.

The generator treats LLM output as untrusted: it extracts the JSON, then drops
any behavior action that targets a cube/action the registry doesn't actually
expose (the doc's "validate against allowed APIs"). A hallucinated action can
never reach a cube — it's filtered before it ever loads into the engine.
"""

import json
import logging
import re
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field, ValidationError

from blipshell.robotics.capability import CubeMetadata
from blipshell.robotics.events import CORE_EVENTS
from blipshell.robotics.registry import CapabilityRegistry
from blipshell.robotics.rules import Behavior

logger = logging.getLogger(__name__)

# (system, user) -> raw LLM text. Wraps router.generate(REASONING, ...).
GenerateFn = Callable[[str, str], Awaitable[str]]


class CapabilityProfile(BaseModel):
    """What the LLM decided a cube is for, plus the behaviors it authored."""

    cube_id: str
    semantic_role: str = ""
    intended_uses: list[str] = Field(default_factory=list)
    usage_guidance: str = ""
    behaviors: list[Behavior] = Field(default_factory=list)


def build_profile_prompt(meta: CubeMetadata, registry: CapabilityRegistry) -> tuple[str, str]:
    """Build (system, user) prompts asking the LLM to author a profile.

    The user prompt lists exactly which (cube, action, args) calls are legal and
    which triggers exist, so a cooperative model stays in bounds — and the
    generator validates regardless.
    """
    system = (
        "You design behaviors for a modular robot. A new hardware module (a "
        "'cube') just connected. Decide what it is useful for and author "
        "behaviors that react to events by invoking the module's actions.\n\n"
        "Respond with a SINGLE JSON object, no prose, of the form:\n"
        '{\n'
        '  "semantic_role": "<one line: what this surface is for>",\n'
        '  "intended_uses": ["<short use>", ...],\n'
        '  "usage_guidance": "<one line of advice on using it well>",\n'
        '  "behaviors": [\n'
        '    {"trigger": "<event name>", "actions": [\n'
        '      {"target": "<cube_id>", "action": "<action>", "args": {...}}\n'
        '    ]}\n'
        '  ]\n'
        "}\n\n"
        "Only use the cube_ids, actions, and triggers listed below. Keep "
        "behaviors simple and few."
    )

    # Enumerate the legal action surface.
    action_lines = []
    for cube_id, action in registry.list_capabilities():
        params = ", ".join(p.name for p in action.parameters) or "no args"
        action_lines.append(f"  - {cube_id}.{action.name}({params}): {action.description}")

    # Trigger vocabulary = core events + any connected cube's advertised events.
    triggers = dict(CORE_EVENTS)
    for cube_meta in registry.list_cubes():
        for ev in cube_meta.events:
            triggers.setdefault(ev, "(emitted by a connected cube)")
    trigger_lines = [f"  - {name}: {desc}" for name, desc in triggers.items()]

    user = (
        f"New cube connected:\n"
        f"  cube_id: {meta.cube_id}\n"
        f"  type: {meta.module_type}\n"
        f"  description: {meta.description}\n"
        f"  constraints: {json.dumps(meta.constraints)}\n\n"
        f"Available actions (only these may be targeted):\n"
        + "\n".join(action_lines)
        + "\n\nAvailable triggers (only these may be used):\n"
        + "\n".join(trigger_lines)
        + "\n\nAuthor the profile JSON now."
    )
    return system, user


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of an LLM response (tolerates code fences/prose)."""
    # Strip ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # Find the outermost { ... } span.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in LLM response")
    return json.loads(text[start:end + 1])


class ProfileGenerator:
    """Turns a connected cube into a validated CapabilityProfile via the LLM."""

    def __init__(self, generate_fn: GenerateFn):
        self._generate = generate_fn

    async def generate(
        self, meta: CubeMetadata, registry: CapabilityRegistry,
    ) -> CapabilityProfile:
        """Ask the LLM for a profile and return it with invalid behaviors dropped."""
        system, user = build_profile_prompt(meta, registry)
        raw = await self._generate(system, user)
        data = _extract_json(raw)

        behaviors = self._validate_behaviors(data.get("behaviors", []), registry)
        return CapabilityProfile(
            cube_id=meta.cube_id,
            semantic_role=str(data.get("semantic_role", "")),
            intended_uses=[str(u) for u in data.get("intended_uses", []) if u],
            usage_guidance=str(data.get("usage_guidance", "")),
            behaviors=behaviors,
        )

    @staticmethod
    def _validate_behaviors(raw_behaviors: Any, registry: CapabilityRegistry) -> list[Behavior]:
        """Keep only behaviors whose every action targets a real cube action.

        A hallucinated target/action drops the whole behavior (not just the bad
        action) — a half-valid behavior is more dangerous than none.
        """
        valid_actions = {(cid, a.name) for cid, a in registry.list_capabilities()}
        kept: list[Behavior] = []
        if not isinstance(raw_behaviors, list):
            return kept

        for entry in raw_behaviors:
            try:
                behavior = Behavior.model_validate(entry)
            except ValidationError as e:
                logger.warning("Dropping malformed behavior %r: %s", entry, e)
                continue
            bad = [
                (a.target, a.action) for a in behavior.actions
                if (a.target, a.action) not in valid_actions
            ]
            if bad:
                logger.warning(
                    "Dropping behavior '%s' — references unavailable action(s): %s",
                    behavior.name or behavior.trigger, bad,
                )
                continue
            kept.append(behavior)
        return kept
