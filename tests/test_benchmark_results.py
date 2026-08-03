"""Benchmark results as committed files (2026-08-03).

The failure this replaced: results lived in a gitignored SQLite DB, so on a
two-PC setup they never synced and the comparison corpus could never
accumulate. These tests pin the properties that make the file store fix that --
per-run filenames (so two machines never conflict), latest-run-wins selection,
provenance capture, and tolerance of a corrupt file.

Deterministic, no LLM, no DB.
"""

import json

import pytest

from blipshell.benchmark.report import build_report, render_markdown
from blipshell.benchmark.results import (
    SCHEMA_VERSION,
    ResultsStore,
    slugify_model,
)


def _rows(model="m", quality=0.9):
    """A minimal harness-shaped row set: one scoring row, one latency row."""
    return [
        {"run_group": "g", "model": model, "suite": "pipeline",
         "task_type": "ranking", "metric": "accuracy", "value": quality,
         "unit": "ratio", "tier": "deep", "is_baseline": False,
         "run_ts": "t", "raw": {"n": 5}},
        {"run_group": "g", "model": model, "suite": "pipeline",
         "task_type": "pipeline", "metric": "latency_s", "value": 1.5,
         "unit": "seconds", "tier": "deep", "is_baseline": False,
         "run_ts": "t", "raw": None},
    ]


class TestFilenames:
    def test_slugify_handles_slashes_and_colons(self):
        assert slugify_model("minimax/minimax-m3") == "minimax_minimax-m3"
        assert slugify_model("kimi-k2.7-code:cloud") == "kimi-k2.7-code_cloud"

    def test_one_file_per_run_so_two_machines_never_conflict(self, tmp_path):
        """The whole reason for per-run files instead of one appended log: two
        boxes writing different filenames merge with a plain `git pull`."""
        s = ResultsStore(tmp_path)
        a = s.write_run(model="qwen3:14b", run_group="g1",
                        run_ts="2026-08-03T10:00:00+00:00", rows=_rows())
        b = s.write_run(model="kimi-k2.7-code:cloud", run_group="g2",
                        run_ts="2026-08-03T11:00:00+00:00", rows=_rows())
        assert a != b
        assert len(list(tmp_path.glob("*.json"))) == 2

    def test_filenames_sort_chronologically(self, tmp_path):
        s = ResultsStore(tmp_path)
        s.write_run(model="m", run_group="g", rows=_rows(),
                    run_ts="2026-08-03T09:00:00+00:00")
        s.write_run(model="m", run_group="g", rows=_rows(),
                    run_ts="2026-12-01T09:00:00+00:00")
        names = sorted(p.name for p in tmp_path.glob("*.json"))
        assert names[0].startswith("20260803")
        assert names[1].startswith("20261201")


class TestRoundTrip:
    def test_written_run_is_readable(self, tmp_path):
        s = ResultsStore(tmp_path)
        s.write_run(model="qwen3:14b", run_group="g",
                    run_ts="2026-08-03T10:00:00+00:00", rows=_rows(quality=0.77))
        runs = s.load_runs()
        assert len(runs) == 1
        assert runs[0]["model"] == "qwen3:14b"
        assert runs[0]["schema"] == SCHEMA_VERSION
        assert len(runs[0]["rows"]) == 2

    def test_raw_payload_survives(self, tmp_path):
        s = ResultsStore(tmp_path)
        s.write_run(model="m", run_group="g", run_ts="2026-08-03T10:00:00+00:00",
                    rows=_rows())
        row = next(r for r in s.load_runs()[0]["rows"] if r["metric"] == "accuracy")
        assert row["raw"] == {"n": 5}

    def test_model_rows_reattaches_header_fields(self, tmp_path):
        """build_report reads task_type/metric/value per row; model and run_ts
        live once in the header and must be re-attached on read."""
        s = ResultsStore(tmp_path)
        s.write_run(model="qwen3:14b", run_group="g7",
                    run_ts="2026-08-03T10:00:00+00:00", rows=_rows())
        rows = s.model_rows()["qwen3:14b"]
        assert all(r["model"] == "qwen3:14b" for r in rows)
        assert all(r["run_group"] == "g7" for r in rows)


class TestLatestWins:
    def test_latest_run_supersedes_earlier_one_for_the_same_model(self, tmp_path):
        s = ResultsStore(tmp_path)
        s.write_run(model="m", run_group="old", rows=_rows(quality=0.10),
                    run_ts="2026-08-01T10:00:00+00:00")
        s.write_run(model="m", run_group="new", rows=_rows(quality=0.90),
                    run_ts="2026-08-03T10:00:00+00:00")
        rows = s.model_rows()["m"]
        acc = next(r["value"] for r in rows if r["metric"] == "accuracy")
        assert acc == 0.90

    def test_history_keeps_every_run_for_drift_inspection(self, tmp_path):
        s = ResultsStore(tmp_path)
        s.write_run(model="m", run_group="a", rows=_rows(0.1),
                    run_ts="2026-08-01T10:00:00+00:00")
        s.write_run(model="m", run_group="b", rows=_rows(0.9),
                    run_ts="2026-08-03T10:00:00+00:00")
        hist = s.history("m")
        assert [h["run_group"] for h in hist] == ["a", "b"]

    def test_models_are_independent(self, tmp_path):
        s = ResultsStore(tmp_path)
        s.write_run(model="a", run_group="g", rows=_rows(0.2),
                    run_ts="2026-08-01T10:00:00+00:00")
        s.write_run(model="b", run_group="g", rows=_rows(0.8),
                    run_ts="2026-08-02T10:00:00+00:00")
        assert set(s.model_rows()) == {"a", "b"}


class TestProvenance:
    def test_captures_when_and_where(self, tmp_path):
        s = ResultsStore(tmp_path)
        s.write_run(model="m", run_group="g", rows=_rows(), tier="deep",
                    judge_model="anthropic/claude-opus-4.8",
                    run_ts="2026-08-03T10:00:00+00:00")
        p = s.provenance()["m"]
        assert p["run_ts"].startswith("2026-08-03")
        assert p["tier"] == "deep"
        assert p["judge_model"] == "anthropic/claude-opus-4.8"
        assert "git_sha" in p and "host" in p

    def test_jobs_scope_is_recorded(self, tmp_path):
        """A --jobs-scoped run measured less than a full one; the report must be
        able to say so rather than presenting a partial run as complete."""
        s = ResultsStore(tmp_path)
        s.write_run(model="m", run_group="g", rows=_rows(),
                    run_ts="2026-08-03T10:00:00+00:00",
                    jobs={"session_review", "pipeline"})
        assert s.load_runs()[0]["jobs"] == ["pipeline", "session_review"]


class TestCorruptionTolerance:
    def test_one_bad_file_does_not_kill_the_report(self, tmp_path):
        """Losing a run is recoverable; losing the whole report is not."""
        s = ResultsStore(tmp_path)
        s.write_run(model="good", run_group="g", rows=_rows(),
                    run_ts="2026-08-03T10:00:00+00:00")
        (tmp_path / "20260101T000000__broken.json").write_text("{not json",
                                                               encoding="utf-8")
        runs = s.load_runs()
        assert [r["model"] for r in runs] == ["good"]

    def test_malformed_payload_is_skipped(self, tmp_path):
        s = ResultsStore(tmp_path)
        (tmp_path / "20260101T000000__x.json").write_text(
            json.dumps({"model": "x"}), encoding="utf-8")  # no "rows"
        assert s.load_runs() == []

    def test_future_schema_is_skipped_not_misread(self, tmp_path):
        s = ResultsStore(tmp_path)
        (tmp_path / "20260101T000000__x.json").write_text(
            json.dumps({"schema": SCHEMA_VERSION + 1, "model": "x", "rows": []}),
            encoding="utf-8")
        assert s.load_runs() == []

    def test_missing_directory_reads_as_empty(self, tmp_path):
        assert ResultsStore(tmp_path / "nope").load_runs() == []


class TestReportIntegration:
    def test_report_builds_from_the_file_store(self, tmp_path):
        s = ResultsStore(tmp_path)
        s.write_run(model="qwen3:14b", run_group="g", rows=_rows(0.80),
                    run_ts="2026-08-03T10:00:00+00:00")
        s.write_run(model="kimi-k2.7-code:cloud", run_group="g2", rows=_rows(0.95),
                    run_ts="2026-08-03T11:00:00+00:00")
        report = build_report(s.model_rows(), provenance=s.provenance())
        assert set(report["models"]) == {"qwen3:14b", "kimi-k2.7-code:cloud"}
        ranking = next(c for c in report["categories"] if c["key"] == "ranking")
        assert ranking["best_model"] == "kimi-k2.7-code:cloud"

    def test_provenance_table_renders(self, tmp_path):
        s = ResultsStore(tmp_path)
        s.write_run(model="m", run_group="g", rows=_rows(),
                    run_ts="2026-08-03T10:00:00+00:00")
        md = render_markdown(build_report(s.model_rows(), provenance=s.provenance()))
        assert "## Provenance" in md
        assert "2026-08-03" in md

    def test_report_without_provenance_still_renders(self):
        """build_report stays usable without the new argument."""
        md = render_markdown(build_report({"m": _rows()}))
        assert "# BlipShell model benchmark" in md
        assert "## Provenance" not in md

    def test_scoped_run_is_flagged_so_composites_arent_compared_naively(self, tmp_path):
        """A --jobs run measured fewer categories, so its COMPOSITE averages
        over fewer jobs. Presenting that next to a full run without saying so
        invites exactly the wrong conclusion."""
        s = ResultsStore(tmp_path)
        s.write_run(model="scoped", run_group="g", rows=_rows(),
                    run_ts="2026-08-03T10:00:00+00:00", jobs={"pipeline"})
        md = render_markdown(build_report(s.model_rows(), provenance=s.provenance()))
        assert "| pipeline |" in md            # scope column
        assert "did not measure every category" in md

    def test_full_run_reports_all_jobs_scope(self, tmp_path):
        s = ResultsStore(tmp_path)
        s.write_run(model="full", run_group="g", rows=_rows(),
                    run_ts="2026-08-03T10:00:00+00:00")
        md = render_markdown(build_report(s.model_rows(), provenance=s.provenance()))
        assert "all jobs" in md
        assert "did not measure every category" not in md

    def test_migrated_run_says_pre_migration_not_just_a_dash(self, tmp_path):
        """'Commit unknowable because it predates the migration' and 'git was
        unavailable' are different facts; a bare dash conflates them."""
        s = ResultsStore(tmp_path)
        (tmp_path / "20260624T120000__old.json").write_text(json.dumps({
            "schema": SCHEMA_VERSION, "model": "old", "run_group": "g",
            "run_ts": "2026-06-24T12:00:00+00:00", "tier": "deep",
            "migrated_from_db": True, "git_sha": None,
            "rows": [{"suite": "pipeline", "task_type": "ranking",
                      "metric": "accuracy", "value": 0.5, "unit": "ratio",
                      "raw": None}],
        }), encoding="utf-8")
        md = render_markdown(build_report(s.model_rows(), provenance=s.provenance()))
        assert "pre-migration" in md


class TestLegacyMigration:
    def _legacy_db(self, tmp_path, rows):
        import sqlite3
        p = tmp_path / "benchmark.db"
        c = sqlite3.connect(p)
        c.execute("""CREATE TABLE benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_group TEXT, model TEXT,
            suite TEXT, task_type TEXT, metric TEXT, value REAL, unit TEXT,
            tier TEXT, is_baseline INT, run_ts TEXT, raw_json TEXT)""")
        for r in rows:
            c.execute(
                "INSERT INTO benchmark_runs (run_group, model, suite, task_type,"
                " metric, value, unit, tier, is_baseline, run_ts, raw_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)", r)
        c.commit()
        c.close()
        return p

    def test_reads_legacy_rows_and_decodes_raw(self, tmp_path):
        from blipshell.benchmark.results import rows_from_legacy_db
        db = self._legacy_db(tmp_path, [
            ("g", "m", "pipeline", "ranking", "accuracy", 0.9, "ratio", "deep",
             0, "2026-06-24T12:00:00+00:00", json.dumps({"n": 3})),
        ])
        rows = rows_from_legacy_db(db)
        assert len(rows) == 1
        assert rows[0]["raw"] == {"n": 3}
        assert "raw_json" not in rows[0]

    def test_absent_table_reads_as_empty(self, tmp_path):
        import sqlite3
        from blipshell.benchmark.results import rows_from_legacy_db
        p = tmp_path / "empty.db"
        sqlite3.connect(p).close()
        assert rows_from_legacy_db(p) == []

    def test_grouping_splits_by_run_and_model(self, tmp_path):
        from scripts.migrate_benchmark_results import group_runs
        rows = [
            {"run_group": "g1", "model": "a", "metric": "accuracy"},
            {"run_group": "g1", "model": "a", "metric": "latency_s"},
            {"run_group": "g2", "model": "b", "metric": "accuracy"},
            {"run_group": "g1", "model": "b", "metric": "accuracy"},
        ]
        runs = group_runs(rows)
        assert len(runs) == 3
        assert len(runs[("g1", "a")]) == 2

    def test_rows_without_a_model_are_dropped(self):
        from scripts.migrate_benchmark_results import group_runs
        assert group_runs([{"run_group": "g", "metric": "accuracy"}]) == {}


class TestStoreNoLongerHoldsResults:
    def test_results_table_is_gone_from_the_sqlite_schema(self):
        """Single source of truth -- keeping both would recreate exactly the
        dual-store drift the ChromaDB -> sqlite-vec migration removed."""
        from blipshell.benchmark.store import SCHEMA_SQL
        assert "benchmark_runs" not in SCHEMA_SQL
        assert "model_catalog" in SCHEMA_SQL

    @pytest.mark.parametrize("gone", [
        "record_run", "record_many", "metrics_for_group",
        "latest_run_group", "models_with_runs", "baseline_metrics",
    ])
    def test_result_methods_removed(self, gone):
        from blipshell.benchmark.store import BenchmarkStore
        assert not hasattr(BenchmarkStore, gone)
