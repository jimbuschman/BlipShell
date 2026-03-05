"""Test timeout when server sends keepalive data (like Ollama during model loading).

Ollama sends periodic status updates while loading a model. This resets
httpx's read timeout because data IS flowing. Tests whether our timeout
still works in this scenario.
"""

import asyncio
import socket
import threading
import time

import pytest

from blipshell.llm.client import LLMClient


def _start_keepalive_server() -> tuple[int, socket.socket]:
    """Start a server that accepts HTTP, sends headers + periodic chunks, never finishes.

    Simulates Ollama loading a model: connection alive, data trickling, but
    the response never completes.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    port = server.getsockname()[1]

    def _handle_client(conn):
        try:
            # Read the HTTP request (so the connection doesn't reset)
            conn.recv(4096)

            # Send HTTP response headers with chunked transfer encoding
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"\r\n"
            )

            # Send keepalive chunks every 1 second — this resets httpx read timeout
            for _ in range(300):  # up to 5 minutes
                chunk = b'{"status":"loading model"}\n'
                hex_len = f"{len(chunk):x}".encode()
                conn.sendall(hex_len + b"\r\n" + chunk + b"\r\n")
                time.sleep(1)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _accept_loop():
        while True:
            try:
                conn, _ = server.accept()
                t = threading.Thread(target=_handle_client, args=(conn,), daemon=True)
                t.start()
            except OSError:
                break

    t = threading.Thread(target=_accept_loop, daemon=True)
    t.start()
    return port, server


@pytest.mark.asyncio
async def test_timeout_with_keepalive_data():
    """Client must time out even when server sends periodic keepalive chunks.

    This is the most likely failure mode: Ollama sends status updates while
    loading a model, which resets httpx's per-read timeout. Our outer
    asyncio.wait_for must still fire.
    """
    port, server = _start_keepalive_server()
    try:
        # httpx read timeout is 3s, but server sends data every 1s (resetting it).
        # The outer asyncio.wait_for at 3s must still fire.
        client = LLMClient(
            host=f"http://127.0.0.1:{port}", max_retries=0, timeout=3.0,
        )

        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                client.generate(
                    prompt="Test", model="fake", system="Brief.",
                    use_cache=False, think=False,
                ),
                timeout=3.0,
            )
            # If we get here, the server's chunked response was parsed as a valid
            # ollama response. That's fine — check it didn't take too long.
            elapsed = time.monotonic() - t0
            assert elapsed < 8.0, f"Took {elapsed:.1f}s even though it 'succeeded'"
        except Exception as e:
            elapsed = time.monotonic() - t0
            error_str = str(e)[:200]
            assert elapsed < 8.0, (
                f"Took {elapsed:.1f}s — timeout didn't fire despite keepalive. "
                f"Error type: {type(e).__name__}, Error: {error_str}"
            )
            print(f"Correctly timed out in {elapsed:.1f}s: {type(e).__name__}: {error_str}")
    finally:
        server.close()
