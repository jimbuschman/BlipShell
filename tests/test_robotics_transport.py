"""TCP transport: a cube connecting over a socket behaves like an in-process one.

Uses a tiny in-process socket client (no GUI) standing in for the cube process.
Covers auto-registration on connect, action forwarding + results, events, clean
teardown on socket drop, and the no-response timeout.
"""

import asyncio
import json

import pytest

from blipshell.core.tools.base import ToolRegistry
from blipshell.robotics import RoboticsCore
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.tool_bridge import tool_name_for
from blipshell.robotics.transport import CubeServer

# Realistic metadata: exactly what a real LED-matrix cube would announce.
LED_META = VirtualLEDMatrix().describe().model_dump()


class FakeCubeClient:
    """Stand-in for the cube process: connects, answers invokes, sends events."""

    def __init__(self, metadata: dict, auto_result: str | None = "ok"):
        self.metadata = metadata
        self.auto_result = auto_result  # None = never respond (timeout test)
        self.received: list[dict] = []
        self._reader = None
        self._writer = None

    async def connect(self, port: int) -> None:
        self._reader, self._writer = await asyncio.open_connection("127.0.0.1", port)
        await self._send({"type": "hello", "metadata": self.metadata})

    async def serve(self) -> None:
        while True:
            line = await self._reader.readline()
            if not line:
                break
            msg = json.loads(line)
            if msg.get("type") == "invoke":
                self.received.append(msg)
                if self.auto_result is not None:
                    await self._send({"type": "result", "id": msg["id"],
                                      "result": self.auto_result})

    async def send_event(self, name: str, payload: dict | None = None) -> None:
        await self._send({"type": "event", "name": name, "payload": payload or {}})

    async def _send(self, msg: dict) -> None:
        self._writer.write((json.dumps(msg) + "\n").encode("utf-8"))
        await self._writer.drain()

    async def close(self) -> None:
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except Exception:
            pass


async def _wait(cond, timeout: float = 2.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not cond():
        if loop.time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.01)


@pytest.fixture
async def server():
    tools = ToolRegistry()
    core = RoboticsCore(tools, generate_fn=None)  # no LLM authoring in transport tests
    srv = CubeServer(core, host="127.0.0.1", port=0, invoke_timeout=0.5)
    await srv.start()
    try:
        yield core, tools, srv
    finally:
        await srv.stop()


async def test_cube_auto_registers_on_connect(server):
    core, tools, srv = server
    client = FakeCubeClient(LED_META)
    await client.connect(srv.port)
    serve = asyncio.create_task(client.serve())

    await _wait(lambda: core.registry.is_connected("led_matrix_01"))

    assert tool_name_for("led_matrix_01", "display_text") in tools.get_tool_names()
    serve.cancel()
    await client.close()


async def test_invoke_is_forwarded_and_result_returned(server):
    core, tools, srv = server
    client = FakeCubeClient(LED_META, auto_result="rendered")
    await client.connect(srv.port)
    serve = asyncio.create_task(client.serve())
    await _wait(lambda: core.registry.is_connected("led_matrix_01"))

    result = await core.registry.invoke("led_matrix_01", "display_text", {"text": "hi"})

    assert "rendered" in result
    assert any(m["action"] == "display_text" and m["args"] == {"text": "hi"}
               for m in client.received)
    serve.cancel()
    await client.close()


async def test_event_from_cube_reaches_bus(server):
    core, tools, srv = server
    got = []

    async def handler(name, payload):
        got.append((name, payload))

    core.registry.event_bus.subscribe("user_present", handler)
    client = FakeCubeClient(LED_META)
    await client.connect(srv.port)
    serve = asyncio.create_task(client.serve())
    await _wait(lambda: core.registry.is_connected("led_matrix_01"))

    await client.send_event("user_present", {"zone": "front"})

    await _wait(lambda: bool(got))
    assert got[0] == ("user_present", {"zone": "front"})
    serve.cancel()
    await client.close()


async def test_disconnect_unwinds_on_socket_drop(server):
    core, tools, srv = server
    client = FakeCubeClient(LED_META)
    await client.connect(srv.port)
    serve = asyncio.create_task(client.serve())
    await _wait(lambda: core.registry.is_connected("led_matrix_01"))

    serve.cancel()
    await client.close()

    await _wait(lambda: not core.registry.is_connected("led_matrix_01"))
    assert [n for n in tools.get_tool_names() if n.startswith("cube_")] == []


async def test_invoke_times_out_if_cube_silent(server):
    core, tools, srv = server
    client = FakeCubeClient(LED_META, auto_result=None)  # never answers
    await client.connect(srv.port)
    serve = asyncio.create_task(client.serve())
    await _wait(lambda: core.registry.is_connected("led_matrix_01"))

    result = await core.registry.invoke("led_matrix_01", "clear", {})

    assert "did not respond" in result
    serve.cancel()
    await client.close()
