"""NightlyRunner orchestration + report honesty.

The startup notification prints "all clean" whenever the report's warnings
and errors are both empty. _build_and_store_report used to collect only
status == "error", so timeouts, Ollama-down skips, partial progress and a
failed backup all rendered as a clean night (deep-dive 2026-08-04).

These drive the REAL run() loop with fake jobs (ok / raises / hangs) and
assert both the per-job isolation contract and what reaches the report.
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from blipshell.core import nightly as nightly_mod
from blipshell.core.nightly import NightlyRunner


class FakeMeta:
    """Minimal sqlite stand-in capturing metadata writes."""

    def __init__(self):
        self.data = {}

    async def set_metadata(self, key, value):
        self.data[key] = value

    async def get_metadata(self, key):
        return self.data.get(key)


class _FakeDatabase:
    """run() checks config.database.path for an import lock — that needs to be
    a real path string, not a Mock (pathlib rejects Mock)."""

    def __init__(self, path):
        self.path = str(path)


class _FakeConfig:
    def __init__(self, path):
        self.database = _FakeDatabase(path)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "nightly_test.db")


def _runner(db_path, sqlite=None):
    r = NightlyRunner(
        config=_FakeConfig(db_path), sqlite=sqlite or FakeMeta(), vectors=Mock(),
        router=Mock(), processor=Mock(),
    )
    # Ollama healthy unless a test says otherwise
    r._check_ollama_health = AsyncMock(return_value=True)
    return r


def _install_jobs(runner, monkeypatch, jobs: dict):
    """Point run_job at a dict of {job_name: async callable}."""
    async def fake_run_job(job_name, on_status=None):
        return await jobs[job_name]()
    monkeypatch.setattr(runner, "run_job", fake_run_job)
    monkeypatch.setattr(nightly_mod, "JOB_ORDER", list(jobs.keys()))


def _report(sqlite):
    return json.loads(sqlite.data["nightly_report"])


class TestJobIsolation:
    async def test_one_failure_does_not_abort_the_rest(self, monkeypatch, db_path):
        """The documented contract: jobs are isolated."""
        ran = []

        async def ok_a():
            ran.append("a")
            return {"did": 1}

        async def boom():
            ran.append("b")
            raise RuntimeError("job b exploded")

        async def ok_c():
            ran.append("c")
            return {"did": 1}

        sqlite = FakeMeta()
        runner = _runner(db_path, sqlite)
        _install_jobs(runner, monkeypatch, {"a": ok_a, "b": boom, "c": ok_c})

        results = await runner.run()

        assert ran == ["a", "b", "c"]         # loop continued past the failure
        assert results["jobs"]["a"]["status"] == "ok"
        assert results["jobs"]["b"]["status"] == "error"
        assert results["jobs"]["c"]["status"] == "ok"

    async def test_hung_job_times_out_and_loop_continues(self, monkeypatch, db_path):
        monkeypatch.setattr(nightly_mod, "_JOB_TIMEOUT", 0.05)
        ran = []

        async def hangs():
            ran.append("hangs")
            await asyncio.sleep(5)
            return {"never": True}

        async def after():
            ran.append("after")
            return {"did": 1}

        sqlite = FakeMeta()
        runner = _runner(db_path, sqlite)
        _install_jobs(runner, monkeypatch, {"hangs": hangs, "after": after})

        results = await runner.run()

        assert ran == ["hangs", "after"]
        assert results["jobs"]["hangs"]["status"] == "timeout"
        assert results["jobs"]["after"]["status"] == "ok"


class TestReportHonesty:
    async def test_timeout_reaches_errors(self, monkeypatch, db_path):
        monkeypatch.setattr(nightly_mod, "_JOB_TIMEOUT", 0.05)

        async def hangs():
            await asyncio.sleep(5)

        sqlite = FakeMeta()
        runner = _runner(db_path, sqlite)
        _install_jobs(runner, monkeypatch, {"hangs": hangs})

        await runner.run()

        report = _report(sqlite)
        assert report["errors"], "a timed-out job reported the night as clean"
        assert any("hangs" in e for e in report["errors"])

    async def test_ollama_down_skips_reach_warnings_grouped(self, monkeypatch, db_path):
        """All LLM jobs skip at once — that must surface, but as ONE grouped
        warning rather than N identical lines."""
        async def never_runs():
            raise AssertionError("job should have been skipped")

        sqlite = FakeMeta()
        runner = _runner(db_path, sqlite)
        runner._check_ollama_health = AsyncMock(return_value=False)
        llm_jobs = {j: never_runs for j in ("batch_tag", "tag_discovery", "consolidate")}
        _install_jobs(runner, monkeypatch, llm_jobs)

        await runner.run()

        report = _report(sqlite)
        skip_warnings = [w for w in report["warnings"] if "skipped" in w]
        assert len(skip_warnings) == 1, f"expected one grouped warning, got {skip_warnings}"
        for job in llm_jobs:
            assert job in skip_warnings[0]

    async def test_tag_discovery_and_consolidate_are_ollama_gated(self):
        """Both call out to Ollama (tag_discovery via router.generate,
        consolidate via an embedded search query). Omitting them meant each
        burned the full job timeout with Ollama down."""
        assert "tag_discovery" in nightly_mod._OLLAMA_JOBS
        assert "consolidate" in nightly_mod._OLLAMA_JOBS
        # centroid_tag only reads stored embeddings — must NOT be gated
        assert "centroid_tag" not in nightly_mod._OLLAMA_JOBS

    async def test_partial_progress_reaches_warnings(self, monkeypatch, db_path):
        async def stopped():
            return {"processed": 10, "stopped_early": True}

        sqlite = FakeMeta()
        runner = _runner(db_path, sqlite)
        _install_jobs(runner, monkeypatch, {"batch_tag": stopped})

        await runner.run()

        report = _report(sqlite)
        assert any("stopped early" in w for w in report["warnings"])

    async def test_per_item_timeouts_reach_warnings(self, monkeypatch, db_path):
        """friction_analysis reports these as `timed_out`, not `failed`."""
        async def partial():
            return {"analyzed": 3, "timed_out": 2}

        sqlite = FakeMeta()
        runner = _runner(db_path, sqlite)
        _install_jobs(runner, monkeypatch, {"friction_analysis": partial})

        await runner.run()

        report = _report(sqlite)
        assert any("timed out" in w for w in report["warnings"])

    async def test_failed_backup_reaches_warnings(self, monkeypatch, db_path):
        """_job_backup swallows its exception and returns a soft warning."""
        async def bad_backup():
            return {"backup_path": None, "warning": "disk full"}

        sqlite = FakeMeta()
        runner = _runner(db_path, sqlite)
        _install_jobs(runner, monkeypatch, {"backup": bad_backup})

        await runner.run()

        report = _report(sqlite)
        assert any("disk full" in w for w in report["warnings"])

    async def test_clean_night_stays_clean(self, monkeypatch, db_path):
        """The fix must not cry wolf: a genuinely clean run reports nothing.
        Includes a config-gated no-op, which is intentional and not a warning."""
        async def ok():
            return {"processed": 5}

        async def config_skip():
            return {"skipped": "entity_merge_enabled=false"}

        sqlite = FakeMeta()
        runner = _runner(db_path, sqlite)
        _install_jobs(runner, monkeypatch, {"prune": ok, "merge_entities": config_skip})

        await runner.run()

        report = _report(sqlite)
        assert report["errors"] == []
        assert report["warnings"] == [], f"clean night produced warnings: {report['warnings']}"

    async def test_job_statuses_block_is_self_describing(self, monkeypatch, db_path):
        monkeypatch.setattr(nightly_mod, "_JOB_TIMEOUT", 0.05)

        async def ok():
            return {}

        async def boom():
            raise RuntimeError("x")

        async def hangs():
            await asyncio.sleep(5)

        sqlite = FakeMeta()
        runner = _runner(db_path, sqlite)
        _install_jobs(runner, monkeypatch, {"a": ok, "b": boom, "c": hangs})

        await runner.run()

        statuses = _report(sqlite)["job_statuses"]
        assert statuses == {"ok": 1, "error": 1, "timeout": 1}


class TestStalenessNotification:
    """_check_nightly_report had no age check, so a dead scheduler reported
    the same successful run as current forever. On a desktop the 2am window
    is missed often — silence is the wrong default (deep-dive 2026-08-04)."""

    def _mixin(self, sqlite):
        """The notification lives on SessionMixin; exercise it standalone."""
        from blipshell.core.agent_session import SessionMixin

        obj = SessionMixin.__new__(SessionMixin)
        obj.sqlite = sqlite
        return obj

    async def _set_run(self, sqlite, *, age_hours, warnings=None, errors=None):
        import time as _time

        completed = _time.time() - age_hours * 3600
        await sqlite.set_metadata("nightly_last_run", json.dumps({
            "completed_at": completed, "elapsed_s": 42.0, "jobs": {},
        }))
        await sqlite.set_metadata("nightly_report", json.dumps({
            "warnings": warnings or [], "errors": errors or [],
        }))

    async def test_fresh_clean_run_reports_clean(self):
        sqlite = FakeMeta()
        await self._set_run(sqlite, age_hours=6)
        msg = await self._mixin(sqlite)._check_nightly_report()
        assert "all clean" in msg

    async def test_stale_run_warns_instead_of_all_clean(self):
        sqlite = FakeMeta()
        await self._set_run(sqlite, age_hours=24 * 9)   # nine days
        msg = await self._mixin(sqlite)._check_nightly_report()
        assert "all clean" not in msg
        assert "hasn't run" in msg
        assert "9 days" in msg

    async def test_staleness_wins_over_a_clean_report(self):
        """An old report saying 'clean' describes a corpus nine days gone."""
        sqlite = FakeMeta()
        await self._set_run(sqlite, age_hours=24 * 9, warnings=[], errors=[])
        msg = await self._mixin(sqlite)._check_nightly_report()
        assert "hasn't run" in msg

    async def test_just_inside_threshold_still_normal(self):
        sqlite = FakeMeta()
        await self._set_run(sqlite, age_hours=30)
        msg = await self._mixin(sqlite)._check_nightly_report()
        assert "hasn't run" not in msg

    async def test_never_run_is_reported(self):
        sqlite = FakeMeta()
        msg = await self._mixin(sqlite)._check_nightly_report()
        assert msg is not None and "never run" in msg

    async def test_fresh_run_with_issues_points_at_details(self):
        sqlite = FakeMeta()
        await self._set_run(sqlite, age_hours=3, errors=["batch_tag: timed out"])
        msg = await self._mixin(sqlite)._check_nightly_report()
        assert "1 error(s)" in msg
        assert "/nightly report" in msg
