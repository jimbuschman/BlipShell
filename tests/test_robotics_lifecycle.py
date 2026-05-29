"""Connect/disconnect lifecycle and validation for the cube robotics layer.

These exercise the highest-risk behavior the design flagged: a cube can vanish
at any moment, and when it does its capabilities and dispatch path must
invalidate cleanly — no actions firing into a gone target, no stale
capabilities still advertised.
"""

import pytest

from blipshell.robotics import CapabilityRegistry
from blipshell.robotics.cubes import VirtualLEDMatrix


@pytest.fixture
def registry():
    return CapabilityRegistry()


# --- connect / capability surface -------------------------------------------

async def test_connect_registers_capabilities(registry):
    cube = VirtualLEDMatrix()
    meta = await registry.connect(cube)

    assert registry.is_connected("led_matrix_01")
    assert meta.module_type == "led_matrix"
    actions = {a.name for _, a in registry.list_capabilities()}
    assert actions == {"display_text", "display_frame", "clear"}


async def test_connect_fires_listener(registry):
    seen = []
    registry.add_connect_listener(lambda meta: seen.append(meta.cube_id))

    await registry.connect(VirtualLEDMatrix())

    assert seen == ["led_matrix_01"]


# --- invoke / dispatch ------------------------------------------------------

async def test_invoke_display_text_updates_state(registry):
    cube = VirtualLEDMatrix()
    await registry.connect(cube)

    result = await registry.invoke("led_matrix_01", "display_text", {"text": "Listening..."})

    assert "Listening..." in result
    assert cube.last_text == "Listening..."


async def test_invoke_display_frame_updates_state(registry):
    cube = VirtualLEDMatrix()
    await registry.connect(cube)
    frame = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]

    result = await registry.invoke("led_matrix_01", "display_frame", {"frame": frame})

    assert "applied" in result.lower()
    assert cube.frame == frame


async def test_clear_resets_state(registry):
    cube = VirtualLEDMatrix()
    await registry.connect(cube)
    await registry.invoke("led_matrix_01", "display_text", {"text": "hi"})

    await registry.invoke("led_matrix_01", "clear", {})

    assert cube.last_text is None
    assert cube.frame == [[0] * 4 for _ in range(4)]


# --- validation: errors are actionable, never exceptions --------------------

async def test_invoke_unknown_cube_is_actionable_error(registry):
    result = await registry.invoke("nonexistent", "display_text", {"text": "hi"})
    assert result.lower().startswith("error")
    assert "nonexistent" in result


async def test_invoke_unknown_action_lists_available(registry):
    await registry.connect(VirtualLEDMatrix())
    result = await registry.invoke("led_matrix_01", "explode", {})
    assert "no action 'explode'" in result
    assert "display_text" in result  # tells the caller what it *can* do


async def test_invoke_missing_required_arg(registry):
    await registry.connect(VirtualLEDMatrix())
    result = await registry.invoke("led_matrix_01", "display_text", {})
    assert "missing required" in result
    assert "text" in result


async def test_frame_constraint_enforced_by_cube(registry):
    await registry.connect(VirtualLEDMatrix())
    # 2x2 frame on a 4x4 panel — cube-level hardware constraint violation.
    result = await registry.invoke("led_matrix_01", "display_frame", {"frame": [[1, 0], [0, 1]]})
    assert result.lower().startswith("error")
    assert "4x4" in result


async def test_frame_bad_values_rejected(registry):
    await registry.connect(VirtualLEDMatrix())
    frame = [[2, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    result = await registry.invoke("led_matrix_01", "display_frame", {"frame": frame})
    assert result.lower().startswith("error")
    assert "0" in result and "1" in result


# --- disconnect invalidation (the load-bearing case) ------------------------

async def test_disconnect_removes_capabilities(registry):
    await registry.connect(VirtualLEDMatrix())
    assert await registry.disconnect("led_matrix_01") is True

    assert not registry.is_connected("led_matrix_01")
    assert registry.list_capabilities() == []
    assert registry.get_metadata("led_matrix_01") is None


async def test_invoke_after_disconnect_fails_cleanly(registry):
    cube = VirtualLEDMatrix()
    await registry.connect(cube)
    await registry.disconnect("led_matrix_01")

    result = await registry.invoke("led_matrix_01", "display_text", {"text": "hi"})

    assert result.lower().startswith("error")
    assert "no cube 'led_matrix_01' is connected" in result
    # The gone cube must not have been touched.
    assert cube.last_text is None


async def test_disconnect_fires_listener(registry):
    gone = []
    registry.add_disconnect_listener(lambda meta: gone.append(meta.cube_id))
    await registry.connect(VirtualLEDMatrix())

    await registry.disconnect("led_matrix_01")

    assert gone == ["led_matrix_01"]


async def test_disconnect_unknown_cube_returns_false(registry):
    assert await registry.disconnect("never_existed") is False


async def test_reconnect_restores_capabilities(registry):
    cube = VirtualLEDMatrix()
    await registry.connect(cube)
    await registry.disconnect("led_matrix_01")

    await registry.connect(cube)

    assert registry.is_connected("led_matrix_01")
    result = await registry.invoke("led_matrix_01", "display_text", {"text": "back"})
    assert cube.last_text == "back"


async def test_reconnect_same_id_replaces_without_leaking(registry):
    """Connecting an id that's already present replaces it (one disconnect fired)."""
    disconnects = []
    registry.add_disconnect_listener(lambda meta: disconnects.append(meta.cube_id))

    await registry.connect(VirtualLEDMatrix())
    await registry.connect(VirtualLEDMatrix())  # same id, reconnect path

    assert disconnects == ["led_matrix_01"]  # old instance torn down exactly once
    assert len(registry.list_cubes()) == 1


# --- events: input cubes publish through the wired sink ---------------------

async def test_cube_emits_through_bus_when_connected(registry):
    received = []

    async def handler(name, payload):
        received.append((name, payload))

    registry.event_bus.subscribe("motion_detected", handler)
    cube = VirtualLEDMatrix()
    await registry.connect(cube)

    await cube.emit("motion_detected", {"zone": "front"})

    assert received == [("motion_detected", {"zone": "front"})]


async def test_cube_emit_after_disconnect_is_silenced(registry):
    received = []

    async def handler(name, payload):
        received.append(name)

    registry.event_bus.subscribe("motion_detected", handler)
    cube = VirtualLEDMatrix()
    await registry.connect(cube)
    await registry.disconnect("led_matrix_01")

    await cube.emit("motion_detected", {"zone": "front"})

    assert received == []  # sink was torn out on disconnect
