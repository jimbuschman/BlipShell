"""TCP transport — cubes connect to BlipShell over a socket, like ESP32 will.

A cube process connects to the CubeServer, announces itself (hello + metadata),
and from then on receives action commands and returns results, and may push
events. On the BlipShell side each connection is wrapped in a RemoteCube that
satisfies the same Cube contract as an in-process cube — so the registry, tool
bridge, profile authoring, and rules engine all work unchanged. Connection is
automatic: the moment a cube's socket arrives, the core registers it; when the
socket drops, the core disconnects it.

Protocol — newline-delimited JSON, one message per line:
  cube -> core : {"type":"hello","metadata":{...CubeMetadata...}}
                 {"type":"result","id":<int>,"result":"<str>"}
                 {"type":"event","name":"<str>","payload":{...}}
  core -> cube : {"type":"invoke","id":<int>,"action":"<str>","args":{...}}

Self-correction note: RemoteCube does not implement snapshot/restore yet, so the
behavior tracer degrades gracefully (no observable state -> nothing flagged).
Wire-level snapshot/restore is a planned follow-up.
"""

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from blipshell.robotics.capability import CubeMetadata
from blipshell.robotics.cube import Cube

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_INVOKE_TIMEOUT = 10.0

# Sends one action to the remote cube and awaits its result string.
InvokeSender = Callable[[str, dict[str, Any]], Awaitable[str]]


class RemoteCube(Cube):
    """A cube living in another process, reached over the wire.

    describe() returns the metadata it announced; invoke() forwards the action
    to the cube and returns whatever it reports back.
    """

    def __init__(self, metadata: CubeMetadata, send_invoke: InvokeSender):
        super().__init__()
        self._metadata = metadata
        self._send_invoke = send_invoke

    def describe(self) -> CubeMetadata:
        return self._metadata

    async def invoke(self, action: str, args: dict[str, Any]) -> str:
        return await self._send_invoke(action, args)


class _CubeConnection:
    """Handles one connected cube: hello, invoke/result correlation, events."""

    def __init__(self, reader, writer, core, invoke_timeout: float):
        self._reader = reader
        self._writer = writer
        self._core = core
        self._invoke_timeout = invoke_timeout
        self._cube: RemoteCube | None = None
        self._cube_id: str | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._write_lock = asyncio.Lock()

    async def run(self) -> None:
        try:
            hello = await self._read_msg()
            if not hello or hello.get("type") != "hello":
                logger.warning("Cube connection sent no valid hello; closing")
                return
            try:
                metadata = CubeMetadata.model_validate(hello["metadata"])
            except Exception as e:
                logger.warning("Cube hello had invalid metadata: %s", e)
                return

            self._cube = RemoteCube(metadata, self._send_invoke)
            self._cube_id = metadata.cube_id
            # Read subsequent messages concurrently so result frames can resolve
            # while core.connect() (profile authoring) issues invokes.
            read_task = asyncio.create_task(self._read_loop())
            try:
                await self._core.connect(self._cube)
                await read_task
            finally:
                read_task.cancel()
        finally:
            await self._teardown()

    async def _teardown(self) -> None:
        # Fail any in-flight invokes so callers don't hang.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result(f"Error: cube '{self._cube_id}' disconnected mid-action")
        self._pending.clear()
        if self._cube_id is not None:
            try:
                await self._core.disconnect(self._cube_id)
            except Exception as e:
                logger.warning("Error disconnecting cube '%s': %s", self._cube_id, e)
        try:
            self._writer.close()
        except Exception:
            pass

    async def _read_loop(self) -> None:
        while True:
            msg = await self._read_msg()
            if msg is None:
                break  # EOF — cube went away
            kind = msg.get("type")
            if kind == "result":
                fut = self._pending.pop(msg.get("id"), None)
                if fut and not fut.done():
                    fut.set_result(str(msg.get("result", "")))
            elif kind == "event":
                await self._core.registry.event_bus.publish(
                    str(msg.get("name", "")), msg.get("payload") or {},
                )
            else:
                logger.debug("Ignoring unknown message from cube: %r", kind)

    async def _send_invoke(self, action: str, args: dict[str, Any]) -> str:
        self._next_id += 1
        mid = self._next_id
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[mid] = fut
        await self._write_msg({"type": "invoke", "id": mid, "action": action, "args": args})
        try:
            return await asyncio.wait_for(fut, timeout=self._invoke_timeout)
        except asyncio.TimeoutError:
            self._pending.pop(mid, None)
            return (f"Error: cube '{self._cube_id}' did not respond to '{action}' "
                    f"within {self._invoke_timeout}s")

    async def _read_msg(self) -> dict | None:
        try:
            line = await self._reader.readline()
        except (ConnectionError, asyncio.IncompleteReadError):
            return None
        if not line:
            return None
        line = line.strip()
        if not line:
            return await self._read_msg()
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Cube sent malformed JSON: %s", e)
            return await self._read_msg()

    async def _write_msg(self, msg: dict) -> None:
        data = (json.dumps(msg) + "\n").encode("utf-8")
        async with self._write_lock:
            self._writer.write(data)
            await self._writer.drain()


class CubeServer:
    """Listens for cube connections and wires each into the robotics core."""

    def __init__(
        self,
        core,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        invoke_timeout: float = DEFAULT_INVOKE_TIMEOUT,
    ):
        self._core = core
        self._host = host
        self._port = port
        self._invoke_timeout = invoke_timeout
        self._server: asyncio.AbstractServer | None = None

    @property
    def port(self) -> int:
        """The bound port (useful when started with port=0 for tests)."""
        if self._server and self._server.sockets:
            return self._server.sockets[0].getsockname()[1]
        return self._port

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, self._host, self._port,
        )
        logger.info("Cube server listening on %s:%d", self._host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception as e:
                logger.debug("Cube server close error: %s", e)
            self._server = None

    async def _handle_client(self, reader, writer) -> None:
        peer = writer.get_extra_info("peername")
        logger.info("Cube connected from %s", peer)
        conn = _CubeConnection(reader, writer, self._core, self._invoke_timeout)
        try:
            await conn.run()
        except Exception as e:
            logger.warning("Cube connection error (%s): %s", peer, e)
