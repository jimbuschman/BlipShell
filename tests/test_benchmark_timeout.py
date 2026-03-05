"""Test that benchmark LLM calls actually time out.

Uses REAL sockets (not mocks) to prove timeouts work end-to-end.
A TCP server accepts connections but never sends data, simulating
a hung Ollama server. The LLMClient must time out and raise.
"""

import asyncio
import socket
import threading
import time

import pytest

from blipshell.llm.client import LLMClient


def _start_black_hole_server() -> tuple[int, socket.socket]:
    """Start a TCP server that accepts connections but never responds.

    Returns (port, server_socket). Caller must close the socket.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    port = server.getsockname()[1]

    # Accept connections in a thread so they don't queue up
    def _accept_loop():
        conns = []
        while True:
            try:
                conn, _ = server.accept()
                conns.append(conn)  # hold open, never respond
            except OSError:
                break  # server closed
        for c in conns:
            try:
                c.close()
            except Exception:
                pass

    t = threading.Thread(target=_accept_loop, daemon=True)
    t.start()
    return port, server


@pytest.mark.asyncio
async def test_generate_times_out_real_socket():
    """LLMClient.generate() times out against a server that never responds."""
    port, server = _start_black_hole_server()
    try:
        client = LLMClient(
            host=f"http://127.0.0.1:{port}", max_retries=0, timeout=3.0,
        )
        t0 = time.monotonic()
        with pytest.raises(Exception):  # TimeoutError, httpx.ReadTimeout, or similar
            await client.generate(prompt="Hello", model="fake", use_cache=False)
        elapsed = time.monotonic() - t0

        assert elapsed < 8.0, f"Took {elapsed:.1f}s — timeout didn't fire (expected ~3s)"
        assert elapsed >= 2.0, f"Took {elapsed:.1f}s — too fast, something else errored"
    finally:
        server.close()


@pytest.mark.asyncio
async def test_outer_wait_for_real_socket():
    """asyncio.wait_for wrapping client.generate() against a real hung server."""
    port, server = _start_black_hole_server()
    try:
        # Client timeout is 30s, but outer wait_for is 3s — outer should win
        client = LLMClient(
            host=f"http://127.0.0.1:{port}", max_retries=0, timeout=30.0,
        )
        t0 = time.monotonic()
        with pytest.raises(Exception):
            await asyncio.wait_for(
                client.generate(prompt="Hello", model="fake", use_cache=False),
                timeout=3.0,
            )
        elapsed = time.monotonic() - t0

        assert elapsed < 8.0, f"Took {elapsed:.1f}s — outer wait_for didn't fire (expected ~3s)"
    finally:
        server.close()


@pytest.mark.asyncio
async def test_sequential_timeouts_dont_compound():
    """3 timed-out calls to a hung server should take ~9s, not minutes."""
    port, server = _start_black_hole_server()
    try:
        client = LLMClient(
            host=f"http://127.0.0.1:{port}", max_retries=0, timeout=3.0,
        )
        t0 = time.monotonic()
        for _ in range(3):
            try:
                await client.generate(prompt="Hello", model="fake", use_cache=False)
            except Exception:
                pass
        elapsed = time.monotonic() - t0

        assert elapsed < 15.0, f"Took {elapsed:.1f}s — timeouts compounding (expected ~9s)"
    finally:
        server.close()


@pytest.mark.asyncio
async def test_benchmark_pattern_real_socket():
    """Test the exact pattern used by benchmark_background.py bench functions."""
    port, server = _start_black_hole_server()
    try:
        client = LLMClient(
            host=f"http://127.0.0.1:{port}", max_retries=0, timeout=3.0,
        )
        timeout = 3.0

        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                client.generate(
                    prompt="Test prompt", model="fake-model",
                    system="Be brief.", use_cache=False, think=False,
                ),
                timeout=timeout,
            )
            assert False, "Should have timed out"
        except Exception as e:
            elapsed = time.monotonic() - t0
            # Record like the benchmark does
            error_str = str(e)[:120]
            assert elapsed < 8.0, (
                f"Took {elapsed:.1f}s — benchmark would hang here. "
                f"Error: {error_str}"
            )
    finally:
        server.close()
