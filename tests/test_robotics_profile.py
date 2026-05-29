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
)

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
