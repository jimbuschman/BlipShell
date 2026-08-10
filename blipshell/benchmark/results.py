"""Benchmark results as committed, versioned files — the single source of truth.

Why this replaced the SQLite results table (2026-08-03):

Results used to live in `data/benchmark.db`, which is gitignored. On a two-PC
setup that means results are machine-local and never sync, so the comparison
corpus can never accumulate: every "compare against the rest" needs every model
re-run on whichever box you're standing at. Nobody does that, which is why the
2026-06-24 report compared four cloud models and omitted `qwen3:14b` — the
local model actually serving half the jobs. The feature was never missing; the
data just couldn't survive.

So: one JSON file per run, in a committed directory.

- **One file per run, not one appended log.** Two machines writing different
  filenames never conflict in git; two machines appending to one file conflict
  on every run. Merging results becomes `git pull`.
- **Committed.** Both boxes see every result ever produced, and `git log` over
  this directory is free regression history — you can see a score move and
  find the commit that moved it.
- **Provenance on every run.** `git_sha` and `host` are recorded, because a
  benchmark you keep for months is only trustworthy if you can tell which code
  produced a number. Scores that predate a scorer change are not comparable to
  scores after it, and without the sha you cannot know which is which.

The model catalog stays in SQLite: it's a refetchable cache from OpenRouter, it
is genuinely throwaway, and committing 345 rows of third-party pricing that
`benchmark discover` rebuilds on demand would be noise. Precious data syncs,
caches don't.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Bump when the on-disk shape changes incompatibly. Readers skip unknown
# majors rather than silently misreading them.
SCHEMA_VERSION = 1

RESULTS_DIRNAME = "benchmark_results"

# Fields copied from a harness row into the stored file. run_group/model/run_ts/
# tier live once in the file header, so they are not repeated per row.
_ROW_FIELDS = ("suite", "task_type", "metric", "value", "unit", "raw",
               "values", "spread")


def slugify_model(model: str) -> str:
    """Filesystem-safe model name. `minimax/minimax-m3` -> `minimax_minimax-m3`."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def _compact_ts(run_ts: str) -> str:
    """ISO timestamp -> sortable filename component (20260803T141530)."""
    return re.sub(r"[^0-9T]", "", run_ts.replace("+00:00", "").split(".")[0])


def _git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    """Short HEAD sha, or None outside a repo / without git.

    Best-effort provenance — a missing sha must never fail a benchmark run.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=5, check=False,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:  # git missing, not a repo, timeout — all non-fatal
        return None


class ResultsStore:
    """Read/write benchmark runs as JSON files under a committed directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ------------------------------------------------------------------ write

    def write_run(
        self,
        *,
        model: str,
        run_group: str,
        run_ts: str,
        rows: list[dict],
        tier: str = "deep",
        judge_model: Optional[str] = None,
        jobs: Optional[set] = None,
        repo_root: Optional[Path] = None,
    ) -> Path:
        """Persist one benchmark run. Returns the file written.

        Rows are the harness's own dicts; per-run constants are hoisted into the
        header so the file reads cleanly and stays diffable.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA_VERSION,
            "model": model,
            "run_group": run_group,
            "run_ts": run_ts,
            "tier": tier,
            "judge_model": judge_model,
            "jobs": sorted(jobs) if jobs else None,
            "git_sha": _git_sha(repo_root or self.root.parent),
            "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or None,
            "rows": [{k: r.get(k) for k in _ROW_FIELDS} for r in rows],
        }
        path = self.root / f"{_compact_ts(run_ts)}__{slugify_model(model)}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote %d benchmark rows to %s", len(rows), path)
        return path

    def write_transcripts(
        self, *, model: str, run_ts: str, calls: list[dict],
    ) -> Path:
        """Persist every candidate call of a run, next to its result file.

        Separate file so the metric rows stay small and diffable; committed
        for the same reason the results are — analysis happens on whichever
        machine reads them, not necessarily the one that ran the model.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (
            f"{_compact_ts(run_ts)}__{slugify_model(model)}__transcripts.json"
        )
        path.write_text(
            json.dumps({"model": model, "run_ts": run_ts, "calls": calls},
                       indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Wrote %d transcript calls to %s", len(calls), path)
        return path

    # ------------------------------------------------------------------- read

    def load_runs(self) -> list[dict]:
        """Every stored run, oldest first. Unreadable files are skipped loudly.

        A corrupt or hand-edited file must not take the whole report down —
        losing one run is recoverable, losing the report is not.
        """
        if not self.root.is_dir():
            return []
        runs = []
        for path in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping unreadable benchmark result %s: %s", path.name, e)
                continue
            if not isinstance(data, dict) or "rows" not in data:
                logger.warning("Skipping malformed benchmark result %s", path.name)
                continue
            if data.get("schema", 0) > SCHEMA_VERSION:
                logger.warning(
                    "Skipping %s — schema %s is newer than this build understands (%s)",
                    path.name, data.get("schema"), SCHEMA_VERSION,
                )
                continue
            data["_path"] = str(path)
            runs.append(data)
        runs.sort(key=lambda d: str(d.get("run_ts") or ""))
        return runs

    def latest_per_model(self) -> dict[str, dict]:
        """model -> its most recent run. Later run_ts wins."""
        latest: dict[str, dict] = {}
        for run in self.load_runs():  # already sorted oldest-first
            model = run.get("model")
            if model:
                latest[model] = run
        return latest

    def model_rows(self) -> dict[str, list[dict]]:
        """model -> newest measurement of each metric, shaped for build_report().

        Merged **per metric, not per run.** A `--jobs`-scoped run measures a few
        jobs; taking the latest run wholesale would let a 3-row session_review
        run erase a model's 21-row full run and leave it with one column and a
        composite averaged over a single job — which is exactly what happened to
        kimi-k2.7-code on 2026-08-03 (it briefly showed the table's best
        composite, 0.925, computed from one number).

        So each (suite, task_type, metric) is resolved independently and the
        newest run that measured it wins. Rows therefore carry their own
        `run_ts`, because one model's column can legitimately be assembled from
        several runs on different dates. `provenance()` reports that spread.

        build_report() reads `task_type`, `metric`, `value` off each row, so the
        hoisted header fields are re-attached here rather than stored per row.
        """
        merged: dict[str, dict[tuple, dict]] = {}
        for run in self.load_runs():  # oldest first, so later runs overwrite
            model = run.get("model")
            if not model:
                continue
            bucket = merged.setdefault(model, {})
            for r in run.get("rows") or []:
                if not isinstance(r, dict):
                    continue
                key = (r.get("suite"), r.get("task_type"), r.get("metric"))
                enriched = dict(r)
                enriched.update({
                    "model": model,
                    "run_group": run.get("run_group"),
                    "run_ts": run.get("run_ts"),
                    "tier": run.get("tier"),
                    "git_sha": run.get("git_sha"),
                })
                bucket[key] = enriched
        return {m: list(b.values()) for m, b in merged.items()}

    def history(self, model: str) -> list[dict]:
        """Every run for one model, oldest first — for score-drift inspection."""
        return [r for r in self.load_runs() if r.get("model") == model]

    def provenance(self) -> dict[str, dict]:
        """model -> where its numbers came from, across every contributing run.

        Because model_rows() merges per metric, a model's column can be built
        from several runs on different dates and different commits. Reporting
        only the newest run would hide that, so this reports the spread:
        ``run_ts`` is the newest contributing run, ``oldest_ts`` the oldest, and
        ``run_count`` how many runs were combined. A wide spread is a signal to
        re-run, not a reason to trust the row.
        """
        out: dict[str, dict] = {}
        for run in self.load_runs():  # oldest first
            model = run.get("model")
            if not model:
                continue
            p = out.setdefault(model, {
                "oldest_ts": run.get("run_ts"),
                "run_count": 0,
                "shas": [],
                "migrated_from_db": False,
            })
            p["run_ts"] = run.get("run_ts")          # newest wins
            p["host"] = run.get("host") or p.get("host")
            p["tier"] = run.get("tier") or p.get("tier")
            p["judge_model"] = run.get("judge_model") or p.get("judge_model")
            p["run_count"] += 1
            if run.get("git_sha"):
                p["shas"].append(run["git_sha"])
            if run.get("migrated_from_db"):
                p["migrated_from_db"] = True
        for p in out.values():
            # Newest known sha; None when every contributing run predates
            # sha capture (all migrated runs).
            p["git_sha"] = p["shas"][-1] if p["shas"] else None
            p["mixed_code"] = len(set(p["shas"])) > 1
            p.pop("shas", None)
        return out


def results_dir(config_path: Optional[str]) -> Path:
    """`benchmark_results/` next to the config file (repo root), cwd-independent.

    Same anchoring rule as the report dir: `blipshell` is an installed CLI run
    from anywhere, and a cwd-relative path silently splits results across
    directories depending on where you invoked it.
    """
    from blipshell.core.config import DEFAULT_CONFIG_PATH

    base = Path(config_path).resolve().parent if config_path else DEFAULT_CONFIG_PATH.parent
    return base / RESULTS_DIRNAME


def rows_from_legacy_db(db_path: str | Path) -> list[dict]:
    """Read benchmark_runs out of a pre-2026-08-03 benchmark.db.

    Used only by scripts/migrate_benchmark_results.py. Returns raw rows with
    `raw_json` decoded back into `raw`; grouping into runs is the caller's job.
    """
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        names = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "benchmark_runs" not in names:
            return []
        rows = []
        for r in conn.execute("SELECT * FROM benchmark_runs ORDER BY run_ts, id"):
            d = dict(r)
            raw = d.pop("raw_json", None)
            if raw:
                try:
                    d["raw"] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    d["raw"] = raw
            rows.append(d)
        return rows
    finally:
        conn.close()
