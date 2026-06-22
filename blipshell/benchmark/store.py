"""Dedicated SQLite store for benchmark results and the model catalog.

Lives in its own `data/benchmark.db` — deliberately NOT the production memory
DB. Benchmark data is independent throwaway data, and isolating it means zero
migrations on `blipshell.db` and a store that can be rebuilt or deleted freely.

Two tables:
  - benchmark_runs: one row per (model, task_type, metric) measured in a run.
  - model_catalog:  discovery cache (OpenRouter / Artificial Analysis feeds).

Schema is idempotent CREATE TABLE IF NOT EXISTS, mirroring sqlite_store.py.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_group     TEXT NOT NULL,            -- ties one `benchmark run` invocation together
    model         TEXT NOT NULL,
    suite         TEXT NOT NULL,            -- pipeline | realdata | tool_calling | coding | reasoning
    task_type     TEXT NOT NULL,            -- ranking | summarization | tool_calling | coding | ...
    metric        TEXT NOT NULL,            -- accuracy | judge_quality | tool_pass_rate | latency_s | ...
    value         REAL,                     -- NULL = not measured / judge failed
    unit          TEXT DEFAULT '',          -- 'ratio' | 'seconds' | 'usd_per_1m' | ...
    tier          TEXT DEFAULT 'quick',     -- quick | full
    is_baseline   INTEGER DEFAULT 0,        -- 1 = production baseline for comparison
    run_ts        TEXT NOT NULL,            -- ISO timestamp (stamped by caller)
    raw_json      TEXT                      -- optional raw payload for drill-down
);

CREATE INDEX IF NOT EXISTS idx_bench_model      ON benchmark_runs(model);
CREATE INDEX IF NOT EXISTS idx_bench_group      ON benchmark_runs(run_group);
CREATE INDEX IF NOT EXISTS idx_bench_task       ON benchmark_runs(task_type, metric);
CREATE INDEX IF NOT EXISTS idx_bench_baseline   ON benchmark_runs(is_baseline);

CREATE TABLE IF NOT EXISTS model_catalog (
    model               TEXT NOT NULL,
    source              TEXT NOT NULL,      -- openrouter | artificial_analysis | ollama
    context_length      INTEGER,
    price_in            REAL,               -- $/1M prompt tokens
    price_out           REAL,               -- $/1M completion tokens
    vision              INTEGER DEFAULT 0,
    intelligence_index  REAL,               -- Artificial Analysis Intelligence Index
    tok_per_s           REAL,
    ttft_s              REAL,
    created_ts          TEXT,               -- provider's model-created timestamp (if any)
    fetched_ts          TEXT NOT NULL,      -- when we last pulled it
    raw_json            TEXT,
    PRIMARY KEY (model, source)
);

CREATE INDEX IF NOT EXISTS idx_catalog_fetched ON model_catalog(fetched_ts);
"""


class BenchmarkStore:
    """Async store over data/benchmark.db."""

    def __init__(self, db_path: str = "data/benchmark.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> "BenchmarkStore":
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path, timeout=60)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        return self

    async def close(self):
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> "BenchmarkStore":
        return await self.initialize()

    async def __aexit__(self, *exc):
        await self.close()

    # ------------------------------------------------------------------ runs

    async def record_run(
        self,
        *,
        run_group: str,
        model: str,
        suite: str,
        task_type: str,
        metric: str,
        value: Optional[float],
        run_ts: str,
        unit: str = "",
        tier: str = "quick",
        is_baseline: bool = False,
        raw: Any = None,
    ) -> int:
        """Insert one measured metric row. Returns the row id."""
        assert self._db is not None, "BenchmarkStore not initialized"
        cur = await self._db.execute(
            """INSERT INTO benchmark_runs
               (run_group, model, suite, task_type, metric, value, unit, tier,
                is_baseline, run_ts, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_group, model, suite, task_type, metric, value, unit, tier,
                1 if is_baseline else 0, run_ts,
                json.dumps(raw) if raw is not None else None,
            ),
        )
        await self._db.commit()
        return cur.lastrowid

    async def record_many(self, rows: list[dict]) -> None:
        """Bulk-insert metric rows. Each dict matches record_run kwargs."""
        for row in rows:
            await self.record_run(**row)

    async def clear_baseline(self) -> None:
        """Demote any existing baseline rows (a new --baseline run supersedes)."""
        assert self._db is not None
        await self._db.execute("UPDATE benchmark_runs SET is_baseline = 0 WHERE is_baseline = 1")
        await self._db.commit()

    async def latest_run_group(self, model: str) -> Optional[str]:
        """Most recent run_group for a model (by run_ts)."""
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT run_group FROM benchmark_runs WHERE model = ? "
            "ORDER BY run_ts DESC, id DESC LIMIT 1",
            (model,),
        )
        row = await cur.fetchone()
        return row["run_group"] if row else None

    async def metrics_for_group(self, run_group: str) -> list[dict]:
        """All metric rows for a run_group."""
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT * FROM benchmark_runs WHERE run_group = ? ORDER BY task_type, metric",
            (run_group,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def baseline_metrics(self) -> list[dict]:
        """All metric rows flagged as the production baseline."""
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT * FROM benchmark_runs WHERE is_baseline = 1 ORDER BY task_type, metric"
        )
        return [dict(r) for r in await cur.fetchall()]

    async def models_with_runs(self) -> list[str]:
        """Distinct models that have any recorded run."""
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT DISTINCT model FROM benchmark_runs ORDER BY model"
        )
        return [r["model"] for r in await cur.fetchall()]

    # -------------------------------------------------------------- catalog

    async def upsert_catalog(self, entry: dict) -> None:
        """Upsert one model_catalog row. Keyed by (model, source)."""
        assert self._db is not None
        await self._db.execute(
            """INSERT INTO model_catalog
               (model, source, context_length, price_in, price_out, vision,
                intelligence_index, tok_per_s, ttft_s, created_ts, fetched_ts, raw_json)
               VALUES (:model, :source, :context_length, :price_in, :price_out, :vision,
                       :intelligence_index, :tok_per_s, :ttft_s, :created_ts, :fetched_ts, :raw_json)
               ON CONFLICT(model, source) DO UPDATE SET
                   context_length=excluded.context_length,
                   price_in=excluded.price_in,
                   price_out=excluded.price_out,
                   vision=excluded.vision,
                   intelligence_index=excluded.intelligence_index,
                   tok_per_s=excluded.tok_per_s,
                   ttft_s=excluded.ttft_s,
                   created_ts=excluded.created_ts,
                   fetched_ts=excluded.fetched_ts,
                   raw_json=excluded.raw_json""",
            {
                "model": entry["model"],
                "source": entry["source"],
                "context_length": entry.get("context_length"),
                "price_in": entry.get("price_in"),
                "price_out": entry.get("price_out"),
                "vision": 1 if entry.get("vision") else 0,
                "intelligence_index": entry.get("intelligence_index"),
                "tok_per_s": entry.get("tok_per_s"),
                "ttft_s": entry.get("ttft_s"),
                "created_ts": entry.get("created_ts"),
                "fetched_ts": entry["fetched_ts"],
                "raw_json": json.dumps(entry.get("raw")) if entry.get("raw") is not None else None,
            },
        )
        await self._db.commit()

    async def catalog_models(self, source: Optional[str] = None) -> list[dict]:
        """All catalog rows, optionally filtered by source."""
        assert self._db is not None
        if source:
            cur = await self._db.execute(
                "SELECT * FROM model_catalog WHERE source = ? ORDER BY model", (source,)
            )
        else:
            cur = await self._db.execute("SELECT * FROM model_catalog ORDER BY model")
        return [dict(r) for r in await cur.fetchall()]

    async def catalog_lookup(self, model: str) -> Optional[dict]:
        """Best catalog entry for a model name (any source; prefers one with price)."""
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT * FROM model_catalog WHERE model = ? "
            "ORDER BY (price_in IS NOT NULL) DESC, fetched_ts DESC LIMIT 1",
            (model,),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def known_catalog_keys(self) -> set[tuple[str, str]]:
        """Set of (model, source) already in the catalog — used to flag 'new' models."""
        assert self._db is not None
        cur = await self._db.execute("SELECT model, source FROM model_catalog")
        return {(r["model"], r["source"]) for r in await cur.fetchall()}
