"""Cube actions auto-register as LLM tools and tear down on disconnect.

This is the "can-invoke" guarantee: a connected cube's actions are in the tool
registry (and so in the LLM's tool list every turn); a disconnected cube's are
gone, so the tool list never lies about what's available.
"""

import pytest

from blipshell.core.tools.base import ToolRegistry
from blipshell.models.tools import ToolCall, ToolParameterType
from blipshell.robotics import CapabilityRegistry
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.tool_bridge import CubeToolBridge, tool_name_for


@pytest.fixture
def wired():
    """A capability registry + tool registry joined by an attached bridge."""
    caps = CapabilityRegistry()
    tools = ToolRegistry()
    CubeToolBridge(caps, tools).attach()
    return caps, tools


async def test_connect_registers_tools(wired):
    caps, tools = wired
    await caps.connect(VirtualLEDMatrix())

    names = set(tools.get_tool_names())
    assert tool_name_for("led_matrix_01", "display_text") in names
    assert tool_name_for("led_matrix_01", "display_frame") in names
    assert tool_name_for("led_matrix_01", "clear") in names


async def test_tool_definition_carries_schema_and_context(wired):
    caps, tools = wired
    await caps.connect(VirtualLEDMatrix())

    tool = tools.get_tool(tool_name_for("led_matrix_01", "display_text"))
    defn = tool.definition()

    # Cube's semantic role surfaces in the description for the LLM.
    assert "led_matrix_01" in defn.description
    assert "LED matrix" in defn.description
    # Parameter schema maps straight across from the ActionSpec.
    assert len(defn.parameters) == 1
    assert defn.parameters[0].name == "text"
    assert defn.parameters[0].type == ToolParameterType.STRING


async def test_tool_executes_through_registry(wired):
    caps, tools = wired
    cube = VirtualLEDMatrix()
    await caps.connect(cube)

    tool = tools.get_tool(tool_name_for("led_matrix_01", "display_text"))
    result = await tool.execute(text="Listening...")

    assert "Listening..." in result
    assert cube.last_text == "Listening..."


async def test_end_to_end_via_tool_call(wired):
    """A full ToolCall through ToolRegistry.execute_tool_call reaches the cube."""
    caps, tools = wired
    cube = VirtualLEDMatrix()
    await caps.connect(cube)

    call = ToolCall(
        id="1",
        name=tool_name_for("led_matrix_01", "display_text"),
        arguments={"text": "hi"},
    )
    result = await tools.execute_tool_call(call)

    assert result.success
    assert cube.last_text == "hi"


async def test_disconnect_unregisters_tools(wired):
    caps, tools = wired
    await caps.connect(VirtualLEDMatrix())
    assert tools.get_tool_names()  # present

    await caps.disconnect("led_matrix_01")

    assert tools.get_tool_names() == []


async def test_reconnect_reregisters_tools(wired):
    caps, tools = wired
    await caps.connect(VirtualLEDMatrix())
    await caps.disconnect("led_matrix_01")

    await caps.connect(VirtualLEDMatrix())

    assert tool_name_for("led_matrix_01", "display_text") in tools.get_tool_names()


async def test_two_cubes_get_distinct_tools(wired):
    caps, tools = wired
    await caps.connect(VirtualLEDMatrix(cube_id="led_a"))
    await caps.connect(VirtualLEDMatrix(cube_id="led_b"))

    names = set(tools.get_tool_names())
    assert tool_name_for("led_a", "display_text") in names
    assert tool_name_for("led_b", "display_text") in names
    assert len(names) == 6  # 3 actions x 2 cubes


async def test_disconnect_one_leaves_the_other(wired):
    caps, tools = wired
    await caps.connect(VirtualLEDMatrix(cube_id="led_a"))
    await caps.connect(VirtualLEDMatrix(cube_id="led_b"))

    await caps.disconnect("led_a")

    names = set(tools.get_tool_names())
    assert tool_name_for("led_a", "display_text") not in names
    assert tool_name_for("led_b", "display_text") in names
    assert len(names) == 3
