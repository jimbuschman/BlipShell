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
from blipshell.robotics.trace import TraceIssue, trace_behaviors

logger = logging.getLogger(__name__)

# How many times the model may revise after seeing observed problems.
DEFAULT_MAX_REVISIONS = 2

# (system, user) -> raw LLM text. Wraps router.generate(REASONING, ...).
GenerateFn = Callable[[str, str], Awaitable[str]]


class CapabilityProfile(BaseModel):
    """What the LLM decided a cube is for, plus the behaviors it authored."""

    cube_id: str
    semantic_role: str = ""
    intended_uses: list[str] = Field(default_factory=list)
    usage_guidance: str = ""
    behaviors: list[Behavior] = Field(default_factory=list)
    # Set by the self-review loop: how many revisions the model made after
    # seeing observed problems, and any problems still unresolved at the end.
    revision_count: int = 0
    unresolved_issues: list[str] = Field(default_factory=list)


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
        '    {"trigger": "<event name>", "intent": "<what the user should observe>",\n'
        '     "actions": [\n'
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


def build_revise_prompt(
    profile: "CapabilityProfile", issues: list[TraceIssue],
) -> tuple[str, str]:
    """Build (system, user) prompts asking the LLM to fix observed problems.

    The system prompt explains the platform constraint that caused the problem
    (instant, delay-free execution) but does NOT prescribe a fix — reorder,
    remove, replace, combine, whatever the model judges best.
    """
    system = (
        "You are revising robot behaviors that did not produce the effect you "
        "intended. Actions in a behavior run immediately one after another with "
        "NO delay — there is no way to pause between them. So if one action "
        "displays something and the next action on the same target changes the "
        "display, the first output is never seen.\n\n"
        "Fix each flagged behavior so its visible result matches its intent. You "
        "may reorder, remove, replace, or combine actions however you judge best. "
        "Respond with a SINGLE JSON object in the same format as before "
        "(semantic_role, intended_uses, usage_guidance, behaviors with intent), "
        "no prose."
    )
    issue_lines = [
        f"  - behavior '{i.behavior_label}': {i.problem}" for i in issues
    ]
    user = (
        "Your current profile:\n"
        + json.dumps(_profile_to_dict(profile))
        + "\n\nProblems observed when the behaviors were actually run:\n"
        + "\n".join(issue_lines)
        + "\n\nReturn the corrected profile JSON."
    )
    return system, user


def _profile_to_dict(profile: "CapabilityProfile") -> dict[str, Any]:
    """Serialize a profile back to the authoring JSON shape (for revision)."""
    return {
        "semantic_role": profile.semantic_role,
        "intended_uses": profile.intended_uses,
        "usage_guidance": profile.usage_guidance,
        "behaviors": [
            {
                "trigger": b.trigger,
                "intent": b.intent,
                "actions": [
                    {"target": a.target, "action": a.action, "args": a.args}
                    for a in b.actions
                ],
            }
            for b in profile.behaviors
        ],
    }


class ProfileGenerator:
    """Turns a connected cube into a validated, self-reviewed CapabilityProfile.

    Authoring is a loop: the LLM proposes behaviors, the tracer runs them and
    reports what was actually observed, and the LLM revises until the observed
    effect is clean or the revision budget is spent. This is the "self-test and
    adjust" step — without it the model authors blind.
    """

    def __init__(self, generate_fn: GenerateFn):
        self._generate = generate_fn

    async def generate(
        self,
        meta: CubeMetadata,
        registry: CapabilityRegistry,
        max_revisions: int = DEFAULT_MAX_REVISIONS,
    ) -> CapabilityProfile:
        """Author, then trace-review-revise until clean or budget exhausted."""
        system, user = build_profile_prompt(meta, registry)
        profile = await self._author(system, user, meta, registry)
        return await self.revise_until_clean(profile, meta, registry, max_revisions)

    async def revise_until_clean(
        self,
        profile: CapabilityProfile,
        meta: CubeMetadata,
        registry: CapabilityRegistry,
        max_revisions: int = DEFAULT_MAX_REVISIONS,
    ) -> CapabilityProfile:
        """Trace an existing profile and revise it until clean or budget spent.

        Exposed separately from generate() so a flawed profile can be fed in
        directly (e.g. to demonstrate self-correction on a known bug).
        """
        issues = await trace_behaviors(profile.behaviors, registry)
        revisions = 0
        while issues and revisions < max_revisions:
            logger.info("Profile for '%s': %d observed issue(s), revising (%d/%d)",
                        meta.cube_id, len(issues), revisions + 1, max_revisions)
            rsystem, ruser = build_revise_prompt(profile, issues)
            try:
                profile = await self._author(rsystem, ruser, meta, registry)
            except Exception as e:
                logger.warning("Revision %d failed for '%s': %s — keeping prior",
                               revisions + 1, meta.cube_id, e)
                break
            revisions += 1
            issues = await trace_behaviors(profile.behaviors, registry)

        profile.revision_count = revisions
        profile.unresolved_issues = [i.problem for i in issues]
        if issues:
            logger.warning("Profile for '%s' still has %d issue(s) after %d revision(s)",
                           meta.cube_id, len(issues), revisions)
        return profile

    async def _author(
        self, system: str, user: str, meta: CubeMetadata, registry: CapabilityRegistry,
    ) -> CapabilityProfile:
        """One LLM round: generate, extract JSON, validate behaviors."""
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
