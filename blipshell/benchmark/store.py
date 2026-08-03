"""SQLite cache for the discovery model catalog.

Lives in its own `data/benchmark.db` — deliberately NOT the production memory
DB, and gitignored, because everything in here is a refetchable third-party
cache that `benchmark discover` rebuilds on demand.

**Results used to live here too, and no longer do** (2026-08-03). A gitignored
DB cannot sync across the two-PC setup, so the comparison corpus could never
accumulate and every report was missing models the other machine had measured.
Results are now committed files — see `benchmark/results.py` for the full
reasoning. Keeping both would recreate the dual-store drift the ChromaDB ->
sqlite-vec migration existed to eliminate, so `benchmark_runs` is gone rather
than deprecated. `results.rows_from_legacy_db()` still reads it out of an old
DB for the one-shot migration.

One table:
  - model_catalog: discovery cache (OpenRouter / Artificial Analysis feeds).

Schema is idempotent CREATE TABLE IF NOT EXISTS, mirroring sqlite_store.py.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
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
