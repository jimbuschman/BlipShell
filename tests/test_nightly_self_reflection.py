"""Nightly self_reflection job (core/nightly.py).

The idle loop and on-return reflection both need the process to be around a
3h+ gap; measured throughput on open-chat-close usage was ~1 thought/month
against a step-2 gate that needs 10 NEW thoughts. The nightly job is the
steady cadence. These tests drive the real job handler against the real
SQLite store with a canned router — logic and wiring only, per the
validation split; nothing here says anything about thought QUALITY.
"""

from unittest.mock import AsyncMock, MagicMock

from blipshell.core.nightly import _OLLAMA_JOBS, JOB_ORDER, NightlyRunner
from blipshell.llm.router import TaskType
from blipshell.models.config import BlipShellConfig, ReflectionConfig

THOUGHT = "I keep circling back to how the garden changes when nobody watches it."


def _runner(sqlite, *, reflection: ReflectionConfig, generate_text: str = THOUGHT):
    config = BlipShellConfig(reflection=reflection)
    router = MagicMock()
    router.generate = AsyncMock(return_value=generate_text)
    return NightlyRunner(config, sqlite, vectors=None, router=router,
                         processor=None), router


def test_job_is_registered():
    assert "self_reflection" in JOB_ORDER
    # Embedding the stored thought needs Ollama; with it down the job must be
    # skipped in milliseconds, not burn the 300s job timeout.
    assert "self_reflection" in _OLLAMA_JOBS


async def test_generates_and_stores_one_thought(sqlite_store):
    runner, router = _runner(sqlite_store, reflection=ReflectionConfig())
    result = await runner._job_self_reflection(lambda msg: None)

    assert result["generated"] == 1
    thoughts = await sqlite_store.get_self_thoughts(with_embeddings=False)
    assert [t["text"] for t in thoughts] == [THOUGHT]
    # Routed through the dedicated reflection key, not reasoning.
    assert router.generate.await_args.args[0] == TaskType.REFLECTION
    # The theme readout rides along on every run.
    assert result["themes"]["texts"] == 1
    assert result["themes"]["distinct_themes"] == 1


async def test_nothing_pressing_stores_nothing(sqlite_store):
    runner, _ = _runner(sqlite_store, reflection=ReflectionConfig(),
                        generate_text="NOTHING")
    result = await runner._job_self_reflection(lambda msg: None)

    assert result["generated"] == 0
    assert result["nothing_pressing"] is True
    assert await sqlite_store.get_self_thoughts(with_embeddings=False) == []


async def test_skips_when_reflection_disabled(sqlite_store):
    runner, router = _runner(
        sqlite_store, reflection=ReflectionConfig(enabled=False),
    )
    result = await runner._job_self_reflection(lambda msg: None)
    assert result["skipped"] == "reflection.enabled=false"
    router.generate.assert_not_awaited()


async def test_skips_when_nightly_disabled(sqlite_store):
    runner, router = _runner(
        sqlite_store, reflection=ReflectionConfig(nightly_enabled=False),
    )
    result = await runner._job_self_reflection(lambda msg: None)
    assert result["skipped"] == "reflection.nightly_enabled=false"
    router.generate.assert_not_awaited()


async def test_skips_when_a_thought_already_formed_today(sqlite_store):
    from datetime import datetime, timezone

    await sqlite_store.add_self_thought(
        "A thought the idle loop already formed this afternoon.",
        datetime.now(timezone.utc).isoformat(),
    )
    runner, router = _runner(sqlite_store, reflection=ReflectionConfig())
    result = await runner._job_self_reflection(lambda msg: None)

    assert result["generated"] == 0
    assert "nightly_min_gap_hours" in result["skipped"]
    router.generate.assert_not_awaited()
    # The readout still lands, so a skipped night still contributes a point.
    assert result["themes"]["texts"] == 1


async def test_generates_when_newest_thought_is_old(sqlite_store):
    from datetime import datetime, timedelta, timezone

    stale = datetime.now(timezone.utc) - timedelta(hours=48)
    await sqlite_store.add_self_thought("An older thought.", stale.isoformat())
    runner, _ = _runner(sqlite_store, reflection=ReflectionConfig())
    result = await runner._job_self_reflection(lambda msg: None)

    assert result["generated"] == 1
    thoughts = await sqlite_store.get_self_thoughts(with_embeddings=False)
    assert len(thoughts) == 2


async def test_min_gap_zero_disables_the_guard(sqlite_store):
    from datetime import datetime, timezone

    await sqlite_store.add_self_thought(
        "Fresh thought.", datetime.now(timezone.utc).isoformat(),
    )
    runner, _ = _runner(
        sqlite_store,
        reflection=ReflectionConfig(nightly_min_gap_hours=0.0),
    )
    result = await runner._job_self_reflection(lambda msg: None)
    assert result["generated"] == 1


async def test_run_job_dispatches_self_reflection(sqlite_store):
    runner, _ = _runner(
        sqlite_store, reflection=ReflectionConfig(enabled=False),
    )
    # Dispatch through the public entry point: an unregistered handler would
    # raise ValueError("Unknown job") here.
    result = await runner.run_job("self_reflection")
    assert result["skipped"] == "reflection.enabled=false"
