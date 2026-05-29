"""RoboticsCore: the assembled reactive-provisioning loop, end to end.

Connect a cube -> its actions become tools AND the LLM authors behaviors that
compile into the rules engine -> an event drives the cube. Disconnect unwinds
all of it. Driven by a fake generate_fn so it runs with no LLM.
"""

import json

import pytest

from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics import RoboticsCore
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.tool_bridge import tool_name_for

PROFILE_JSON = json.dumps({
    "semantic_role": "status display",
    "intended_uses": ["listening cue"],
    "usage_guidance": "short text",
    "behaviors": [
        {"trigger": "speech_detected", "actions": [
            {"target": "led_matrix_01", "action": "display_text", "args": {"text": "Listening..."}},
        ]},
    ],
})


async def _fake_gen(system, user):
    return PROFILE_JSON


@pytest.fixture
def tools():
    return ToolRegistry()


async def test_connect_registers_tools_and_authors_behaviors(tools):
    core = RoboticsCore(tools, generate_fn=_fake_gen)
    cube = VirtualLEDMatrix()

    await core.connect(cube)

    # Tools registered (can-invoke)...
    assert tool_name_for("led_matrix_01", "display_text") in tools.get_tool_names()
    # ...and behaviors authored + compiled (uses-well).
    profile = core.get_profile("led_matrix_01")
    assert profile is not None and len(profile.behaviors) == 1
    assert core.rules.triggers == {"speech_detected"}


async def test_full_reactive_loop_event_drives_cube(tools):
    core = RoboticsCore(tools, generate_fn=_fake_gen)
    cube = VirtualLEDMatrix()
    await core.connect(cube)

    await core.registry.event_bus.publish("speech_detected", {})

    assert cube.last_text == "Listening..."


async def test_disconnect_unwinds_everything(tools):
    core = RoboticsCore(tools, generate_fn=_fake_gen)
    cube = VirtualLEDMatrix()
    await core.connect(cube)

    await core.disconnect("led_matrix_01")

    assert tools.get_tool_names() == []          # tools gone
    assert core.get_profile("led_matrix_01") is None  # profile gone
    assert core.rules.triggers == set()          # behaviors removed


async def test_profile_failure_does_not_break_connect(tools):
    async def boom(system, user):
        raise RuntimeError("LLM down")

    core = RoboticsCore(tools, generate_fn=boom)
    cube = VirtualLEDMatrix()

    await core.connect(cube)  # must not raise

    # Cube is usable: tools registered, just no auto behaviors.
    assert tool_name_for("led_matrix_01", "display_text") in tools.get_tool_names()
    assert core.get_profile("led_matrix_01") is None
    assert core.rules.triggers == set()


async def test_no_generate_fn_still_registers_tools(tools):
    core = RoboticsCore(tools, generate_fn=None)
    cube = VirtualLEDMatrix()

    await core.connect(cube)

    assert tool_name_for("led_matrix_01", "display_text") in tools.get_tool_names()
    assert core.get_profile("led_matrix_01") is None


async def test_two_cubes_behaviors_merge_and_unmerge(tools):
    core = RoboticsCore(tools, generate_fn=_fake_gen)
    # Second cube's canned profile references led_b.
    await core.connect(VirtualLEDMatrix(cube_id="led_matrix_01"))

    async def gen_b(system, user):
        return json.dumps({
            "behaviors": [{"trigger": "user_present", "actions": [
                {"target": "led_b", "action": "clear", "args": {}},
            ]}],
        })
    core.profiles._generate = gen_b  # swap the fake for the second connect
    await core.connect(VirtualLEDMatrix(cube_id="led_b"))

    assert core.rules.triggers == {"speech_detected", "user_present"}

    await core.disconnect("led_b")
    assert core.rules.triggers == {"speech_detected"}  # only led_b's behavior removed
