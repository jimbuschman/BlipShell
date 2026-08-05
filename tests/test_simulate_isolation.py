"""Simulation must not write to the production database.

SimRunner loaded the ambient config and used config.database.path as-is.
Every scenario starts a session and _cleanup calls the full end_session, so
a complete run wrote ~38 real sessions plus LLM-generated lessons and mutated
project digests into the live corpus — the same corpus retrieval reads from
and `benchmark realdata` samples (deep-dive 2026-08-04).

These test path resolution and temp-DB lifecycle only; they never bootstrap an
Agent (that needs the `openai` package and a live model, i.e. the Ollama PC).
"""

from pathlib import Path

import pytest

from blipshell.simulate.runner import SimRunner


class _FakeDatabase:
    def __init__(self, path):
        self.path = path


class _FakeConfig:
    """Just enough config surface for _resolve_db_path."""

    def __init__(self, path="data/blipshell.db"):
        self.database = _FakeDatabase(path)


class TestDatabaseIsolationByDefault:
    def test_default_run_uses_a_temp_db_not_the_configured_one(self):
        cfg = _FakeConfig("data/blipshell.db")
        runner = SimRunner()

        resolved = runner._resolve_db_path(cfg)

        assert resolved is not None
        assert "blipshell.db" not in resolved
        assert Path(resolved).name == "sim.db"
        assert "blipshell_sim_" in resolved
        runner._discard_temp_db()

    def test_temp_path_is_stable_within_a_run(self):
        """All scenarios in one suite share the same throwaway DB."""
        runner = SimRunner()
        cfg = _FakeConfig()
        try:
            first = runner._resolve_db_path(cfg)
            second = runner._resolve_db_path(cfg)
            assert first == second
        finally:
            runner._discard_temp_db()

    def test_temp_db_is_removed_after_the_suite(self, tmp_path):
        runner = SimRunner()
        resolved = runner._resolve_db_path(_FakeConfig())
        parent = Path(resolved).parent
        parent.mkdir(parents=True, exist_ok=True)
        (parent / "sim.db").write_text("fake db")
        assert parent.exists()

        runner._discard_temp_db()

        assert not parent.exists(), "simulation temp database was left behind"
        assert runner._temp_db_dir is None

    def test_discard_is_safe_when_no_temp_db_was_created(self):
        runner = SimRunner(use_real_db=True)
        runner._discard_temp_db()      # must not raise


class TestExplicitOverrides:
    def test_explicit_db_path_is_honored(self, tmp_path):
        target = str(tmp_path / "scratch.db")
        runner = SimRunner(db_path=target)

        assert runner._resolve_db_path(_FakeConfig()) == target
        assert runner._temp_db_dir is None, "explicit --db must not create a temp DB"

    def test_real_db_opts_out_of_isolation(self):
        """--real-db returns None, meaning "leave config.database.path alone"."""
        runner = SimRunner(use_real_db=True)
        assert runner._resolve_db_path(_FakeConfig("data/blipshell.db")) is None

    def test_real_db_warns_loudly(self):
        messages = []
        runner = SimRunner(use_real_db=True, on_status=messages.append)

        runner._resolve_db_path(_FakeConfig())

        assert any("WARNING" in m and "REAL database" in m for m in messages), (
            "running against the live corpus must be announced"
        )
