"""A simulate scenario must leave nothing that keeps the process alive.

aiosqlite's connection worker is a NON-daemon thread; an agent whose SQLite
was never closed keeps the interpreter running after main returns. Every
Stage E gate run on 2026-09-09 wrote its JSON and then hung until killed.
The runner's cleanup now releases the stores the way the CLI does.
"""

from __future__ import annotations

import threading

import pytest

from blipshell.simulate.models import SimScenario
from blipshell.simulate.runner import SimRunner


def _store_threads() -> list[str]:
    return [t.name for t in threading.enumerate()
            if not t.daemon and t is not threading.main_thread() and "connection_worker" in t.name]


async def _store_threads_after_release(timeout_s: float = 3.0) -> list[str]:
    """Store threads still alive once the release has had time to land.

    aiosqlite 0.22's `Connection.close()` awaits a `stop()` future that the
    worker thread resolves from INSIDE itself (`close_and_stop`); the thread
    then still has to unwind its run loop and exit, so `threading.enumerate()`
    can list it for a few milliseconds after `close()` has returned - longer
    on a loaded machine. The invariant under test is that cleanup RELEASES
    the thread, not that it is gone within zero milliseconds of `close()`
    returning. The instant snapshot failed twice in a row under load
    (2026-09-11) while an instrumented full run found no test leaving a
    thread behind - the "leak" was this test's own agent, checked too soon."""
    import asyncio
    deadline = asyncio.get_event_loop().time() + timeout_s
    left = _store_threads()
    while left and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
        left = _store_threads()
    return left


async def test_end_session_alone_leaves_the_sqlite_thread_alive(tmp_path):
    """Negative control: the failure is real without the cleanup."""
    from blipshell.benchmark.continuity import bootstrap_headless_agent
    agent, _ = await bootstrap_headless_agent(tmp_path / "a.db")
    try:
        await agent.start_session()
        await agent.end_session()
        assert _store_threads(), "expected aiosqlite's non-daemon worker thread to still be alive"
    finally:
        await agent.force_cleanup()


async def test_runner_cleanup_releases_the_stores(tmp_path):
    from blipshell.benchmark.continuity import bootstrap_headless_agent
    agent, _ = await bootstrap_headless_agent(tmp_path / "b.db")
    await agent.start_session()
    scenario = SimScenario(name="x", description="", category="t", steps=[])
    await SimRunner(quiet=True)._cleanup(agent, scenario)
    assert await _store_threads_after_release() == []
    assert agent.sqlite._db is None or getattr(agent.sqlite, "_closed", True)


async def test_preflight_agent_is_fully_released(monkeypatch):
    """The pre-flight validator boots its own agent; it must release the
    stores too, or the interpreter hangs at exit on its SQLite thread."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from blipshell.simulate import runner as r

    agent = SimpleNamespace(
        start_session=AsyncMock(), end_session=AsyncMock(), force_cleanup=AsyncMock(),
        activate_project=AsyncMock(), deactivate_project=AsyncMock(),
        tool_registry=SimpleNamespace(get_tool_names=lambda: ["read_file"]),
        sqlite=SimpleNamespace(list_projects=AsyncMock(return_value=[])),
    )

    async def fake_bootstrap(self):
        return agent, object(), object()

    monkeypatch.setattr(r.SimRunner, "_bootstrap_agent", fake_bootstrap)
    errors = await r.SimRunner(quiet=True)._preflight_validate([])
    assert errors == []
    agent.end_session.assert_awaited_once()
    agent.force_cleanup.assert_awaited_once()
