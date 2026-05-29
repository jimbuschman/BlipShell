"""LLM-authored capability profiles: JSON extraction and untrusted-output validation.

Uses a fake generate_fn (canned JSON) so the validation/parsing logic is tested
with no LLM. Real-model output quality is validated separately on the Ollama PC.
"""

import json

import pytest

from blipshell.robotics import CapabilityRegistry
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.profile import (
    ProfileGenerator,
    _extract_json,
    build_profile_prompt,
    build_revise_prompt,
)
from blipshell.robotics.trace import TraceIssue

VALID_PROFILE = {
    "semantic_role": "status display surface",
    "intended_uses": ["listening cue", "notifications"],
    "usage_guidance": "Keep text short.",
    "behaviors": [
        {"trigger": "speech_detected", "actions": [
            {"target": "led_matrix_01", "action": "display_text", "args": {"text": "Listening..."}},
        ]},
    ],
}


def _fake_gen(payload):
    """Build a generate_fn that returns the given object as JSON text."""
    async def gen(system, user):
        return json.dumps(payload)
    return gen


@pytest.fixture
async def registry_with_cube():
    reg = CapabilityRegistry()
    await reg.connect(VirtualLEDMatrix())
    return reg


# --- _extract_json ----------------------------------------------------------

def test_extract_bare_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_prose():
    text = 'Sure! Here is the profile:\n{"a": 1, "b": 2}\nHope that helps.'
    assert _extract_json(text) == {"a": 1, "b": 2}


def test_extract_json_missing_raises():
    with pytest.raises(ValueError):
        _extract_json("there is no json here")


# --- prompt -----------------------------------------------------------------

async def test_prompt_lists_actions_and_triggers(registry_with_cube):
    meta = registry_with_cube.get_metadata("led_matrix_01")
    system, user = build_profile_prompt(meta, registry_with_cube)

    assert "led_matrix_01.display_text" in user  # legal action surface
    assert "speech_detected" in user             # core trigger vocabulary
    assert "JSON" in system


# --- generation + validation ------------------------------------------------

async def test_generate_valid_profile(registry_with_cube):
    meta = registry_with_cube.get_metadata("led_matrix_01")
    gen = ProfileGenerator(_fake_gen(VALID_PROFILE))

    profile = await gen.generate(meta, registry_with_cube)

    assert profile.cube_id == "led_matrix_01"
    assert profile.semantic_role == "status display surface"
    assert len(profile.behaviors) == 1
    assert profile.behaviors[0].trigger == "speech_detected"


async def test_generate_drops_unknown_action(registry_with_cube):
    """A hallucinated action is filtered before it can ever load."""
    payload = {
        "semantic_role": "x",
        "behaviors": [
            {"trigger": "speech_detected", "actions": [
                {"target": "led_matrix_01", "action": "launch_missiles", "args": {}},
            ]},
        ],
    }
    meta = registry_with_cube.get_metadata("led_matrix_01")
    gen = ProfileGenerator(_fake_gen(payload))

    profile = await gen.generate(meta, registry_with_cube)

    assert profile.behaviors == []  # whole behavior dropped


async def test_generate_drops_unknown_target(registry_with_cube):
    payload = {
        "behaviors": [
            {"trigger": "speech_detected", "actions": [
                {"target": "ghost_cube", "action": "display_text", "args": {"text": "hi"}},
            ]},
        ],
    }
    meta = registry_with_cube.get_metadata("led_matrix_01")
    gen = ProfileGenerator(_fake_gen(payload))

    profile = await gen.generate(meta, registry_with_cube)

    assert profile.behaviors == []


async def test_generate_keeps_valid_drops_invalid_mixed(registry_with_cube):
    payload = {
        "behaviors": [
            {"trigger": "speech_detected", "actions": [
                {"target": "led_matrix_01", "action": "display_text", "args": {"text": "ok"}},
            ]},
            {"trigger": "user_present", "actions": [
                {"target": "led_matrix_01", "action": "nope", "args": {}},
            ]},
        ],
    }
    meta = registry_with_cube.get_metadata("led_matrix_01")
    gen = ProfileGenerator(_fake_gen(payload))

    profile = await gen.generate(meta, registry_with_cube)

    assert len(profile.behaviors) == 1
    assert profile.behaviors[0].trigger == "speech_detected"


async def test_generate_tolerates_malformed_behavior_entry(registry_with_cube):
    payload = {"behaviors": ["not a behavior", {"trigger": "x"}]}  # junk + no actions
    meta = registry_with_cube.get_metadata("led_matrix_01")
    gen = ProfileGenerator(_fake_gen(payload))

    profile = await gen.generate(meta, registry_with_cube)

    # The string entry is dropped; the action-less one is valid-but-empty.
    assert all(isinstance(b.trigger, str) for b in profile.behaviors)


# --- self-test / review / adjust loop ---------------------------------------

# Reproduces the live-run bug: greet by showing HI then immediately clearing it.
BAD_GREET = {
    "semantic_role": "status display",
    "behaviors": [
        {"trigger": "user_present", "intent": "briefly greet the user", "actions": [
            {"target": "led_matrix_01", "action": "display_text", "args": {"text": "HI"}},
            {"target": "led_matrix_01", "action": "clear", "args": {}},
        ]},
    ],
}
# The fix: drop the clear so the greeting actually shows.
FIXED_GREET = {
    "semantic_role": "status display",
    "behaviors": [
        {"trigger": "user_present", "intent": "briefly greet the user", "actions": [
            {"target": "led_matrix_01", "action": "display_text", "args": {"text": "HI"}},
        ]},
    ],
}


def _sequence_gen(*payloads):
    """generate_fn that returns each payload in turn, repeating the last."""
    state = {"n": 0}

    async def gen(system, user):
        i = min(state["n"], len(payloads) - 1)
        state["n"] += 1
        return json.dumps(payloads[i])

    return gen


async def test_self_review_fixes_flagged_behavior(registry_with_cube):
    """Author a flash bug, observe it, revise to a clean profile."""
    meta = registry_with_cube.get_metadata("led_matrix_01")
    gen = ProfileGenerator(_sequence_gen(BAD_GREET, FIXED_GREET))

    profile = await gen.generate(meta, registry_with_cube)

    assert profile.revision_count == 1
    assert profile.unresolved_issues == []
    assert len(profile.behaviors[0].actions) == 1  # the clear was dropped


async def test_self_review_gives_up_after_budget(registry_with_cube):
    """If the model never fixes it, stop after max_revisions and report it."""
    meta = registry_with_cube.get_metadata("led_matrix_01")
    gen = ProfileGenerator(_sequence_gen(BAD_GREET))  # always returns the bad one

    profile = await gen.generate(meta, registry_with_cube, max_revisions=2)

    assert profile.revision_count == 2
    assert profile.unresolved_issues  # still flagged
    assert "never visible" in profile.unresolved_issues[0]


async def test_clean_profile_needs_no_revision(registry_with_cube):
    meta = registry_with_cube.get_metadata("led_matrix_01")
    gen = ProfileGenerator(_sequence_gen(FIXED_GREET))

    profile = await gen.generate(meta, registry_with_cube)

    assert profile.revision_count == 0
    assert profile.unresolved_issues == []


async def test_revise_until_clean_fixes_seeded_flaw(registry_with_cube):
    """The inject-flaw path: feed a known-bad profile straight in, get it fixed."""
    from blipshell.robotics.profile import CapabilityProfile
    from blipshell.robotics.rules import Behavior, BehaviorAction

    meta = registry_with_cube.get_metadata("led_matrix_01")
    flawed = CapabilityProfile(cube_id="led_matrix_01", behaviors=[Behavior(
        trigger="user_present", intent="briefly greet",
        actions=[
            BehaviorAction(target="led_matrix_01", action="display_text", args={"text": "HI"}),
            BehaviorAction(target="led_matrix_01", action="clear", args={}),
        ],
    )])
    gen = ProfileGenerator(_sequence_gen(FIXED_GREET))  # model returns the fix

    fixed = await gen.revise_until_clean(flawed, meta, registry_with_cube)

    assert fixed.revision_count == 1
    assert fixed.unresolved_issues == []
    assert len(fixed.behaviors[0].actions) == 1


def test_revise_prompt_includes_issue_and_constraint():
    from blipshell.robotics.profile import CapabilityProfile

    profile = CapabilityProfile(cube_id="led_matrix_01")
    issues = [TraceIssue(
        behavior_label="greet", action_index=0, target="led_matrix_01",
        action="display_text", problem="the output of 'display_text' was never visible",
    )]
    system, user = build_revise_prompt(profile, issues)

    assert "no delay" in system.lower()       # explains the platform constraint
    assert "never visible" in user            # surfaces the observed problem
    assert "greet" in user
