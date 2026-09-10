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
    assert _store_threads() == []
    assert agent.sqlite._db is None or getattr(agent.sqlite, "_closed", True)
