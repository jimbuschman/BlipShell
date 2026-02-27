"""Nightly job runner for BlipShell.

Orchestrates overnight maintenance: backup, cleanup, tagging, pruning,
consolidation, and tag discovery. Can run interactively via /nightly
or headless via `blipshell nightly`.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Optional

from blipshell.memory.batch_tagger import BatchTagger
from blipshell.memory.centroid_tagger import CentroidTagger
from blipshell.memory.consolidation import MemoryConsolidator
from blipshell.memory.tag_discovery import TagDiscovery
from blipshell.memory.tagger import register_topic_patterns

logger = logging.getLogger(__name__)

# Jobs run in this order. Each is isolated — one failure doesn't abort the rest.
JOB_ORDER = [
    "backup",
    "cleanup",
    "centroid_tag",
    "batch_tag",
    "prune",
    "consolidate",
    "tag_discovery",
]


class NightlyRunner:
    """Orchestrates nightly maintenance jobs."""

    def __init__(self, config, sqlite, chroma, router, processor):
        self.config = config
        self.sqlite = sqlite
        self.chroma = chroma
        self.router = router
        self.processor = processor

    @classmethod
    async def create_from_config(cls, config_path: str | None = None) -> NightlyRunner:
        """Factory: build a NightlyRunner from config without a full Agent.

        Creates its own SQLite, ChromaDB, Router, and Processor instances.
        Caller must call close() when done.
        """
        from blipshell.core.config import ConfigManager
        from blipshell.llm.endpoints import EndpointManager
        from blipshell.llm.router import LLMRouter
        from blipshell.memory.chroma_store import ChromaStore
        from blipshell.memory.processor import MemoryProcessor
        from blipshell.memory.sqlite_store import SQLiteStore

        config_mgr = ConfigManager(config_path)
        config = config_mgr.load()

        sqlite = SQLiteStore(config.database.path)
        await sqlite.initialize()

        chroma = ChromaStore(
            persist_dir=config.database.chroma_path,
            embedding_model=config.models.embedding,
            ollama_url=config.endpoints[0].url if config.endpoints else "http://localhost:11434",
        )
        chroma.initialize()

        endpoint_mgr = EndpointManager(config.endpoints, config.llm)
        router = LLMRouter(config.models, endpoint_mgr)
        processor = MemoryProcessor(sqlite, chroma, router, config=config.memory)

        return cls(config, sqlite, chroma, router, processor)

    async def run(
        self,
        on_status: Optional[Callable[[str], None]] = None,
        jobs: Optional[list[str]] = None,
    ) -> dict:
        """Run all (or specified) nightly jobs in sequence.

        Returns stats dict with per-job results.
        """
        run_jobs = jobs or JOB_ORDER
        started_at = time.time()
        results = {}

        def _status(msg: str):
            if on_status:
                on_status(msg)
            logger.info("Nightly: %s", msg)

        _status(f"Starting nightly run ({len(run_jobs)} jobs)...")

        for job_name in run_jobs:
            if job_name not in JOB_ORDER:
                _status(f"Unknown job: {job_name}, skipping.")
                results[job_name] = {"status": "skipped", "error": "unknown job"}
                continue

            _status(f"Running job: {job_name}...")
            t0 = time.monotonic()
            try:
                job_result = await self.run_job(job_name, on_status=on_status)
                elapsed = time.monotonic() - t0
                job_result["status"] = "ok"
                job_result["elapsed_s"] = round(elapsed, 1)
                results[job_name] = job_result
                _status(f"  {job_name} completed in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.monotonic() - t0
                logger.error("Nightly job %s failed: %s", job_name, e, exc_info=True)
                results[job_name] = {
                    "status": "error",
                    "error": str(e),
                    "elapsed_s": round(elapsed, 1),
                }
                _status(f"  {job_name} FAILED: {e}")

        completed_at = time.time()
        total_elapsed = completed_at - started_at

        # Persist run stats
        try:
            await self.sqlite.set_metadata("nightly_last_run", json.dumps({
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_s": round(total_elapsed, 1),
                "jobs": {k: v.get("status", "unknown") for k, v in results.items()},
            }))
        except Exception:
            pass

        _status(f"Nightly run complete in {total_elapsed:.0f}s.")
        return {
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_s": round(total_elapsed, 1),
            "jobs": results,
        }

    async def run_job(
        self,
        job_name: str,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Run a single job by name. Returns job-specific stats dict."""
        handlers = {
            "backup": self._job_backup,
            "cleanup": self._job_cleanup,
            "centroid_tag": self._job_centroid_tag,
            "batch_tag": self._job_batch_tag,
            "prune": self._job_prune,
            "consolidate": self._job_consolidate,
            "tag_discovery": self._job_tag_discovery,
        }
        handler = handlers.get(job_name)
        if not handler:
            raise ValueError(f"Unknown job: {job_name}")
        return await handler(on_status)

    async def _job_backup(self, on_status) -> dict:
        """Run pre-operation backup."""
        try:
            from scripts.backup_db import backup_before_destructive
            result = backup_before_destructive(
                "nightly",
                db_path=self.config.database.path,
                chroma_path=self.config.database.chroma_path,
            )
            return {"backup_path": str(result) if result else None}
        except Exception as e:
            logger.warning("Backup failed (non-fatal): %s", e)
            return {"backup_path": None, "warning": str(e)}

    async def _job_cleanup(self, on_status) -> dict:
        """Reprocess failed messages."""
        unprocessed = await self.sqlite.get_unprocessed_messages(limit=500)
        if not unprocessed:
            return {"processed": 0, "failed": 0, "total": 0}

        import asyncio
        processed = 0
        failed = 0
        for msg in unprocessed:
            try:
                await asyncio.wait_for(
                    self.processor.process_message(
                        text=msg["content"],
                        role=msg["role"],
                        session_id=msg["session_id"],
                    ),
                    timeout=120,
                )
                await self.sqlite.mark_message_processed(msg["id"])
                processed += 1
            except Exception:
                failed += 1

        return {"processed": processed, "failed": failed, "total": len(unprocessed)}

    async def _job_centroid_tag(self, on_status) -> dict:
        """Run centroid-based tag assignment."""
        tagger = CentroidTagger(self.sqlite, self.chroma, self.config.memory)
        return await tagger.run(on_status=on_status)

    async def _job_batch_tag(self, on_status) -> dict:
        """Run LLM batch tag assignment."""
        tagger = BatchTagger(self.sqlite, self.router, self.config.memory)
        return await tagger.tag_all(on_status=on_status)

    async def _job_prune(self, on_status) -> dict:
        """Archive old low-value memories."""
        cfg = self.config.memory
        if cfg.auto_prune_days <= 0:
            return {"pruned": 0, "skipped": "auto_prune_days=0"}

        ids_to_archive = await self.sqlite.get_archived_memory_ids(
            days_old=cfg.auto_prune_days,
            max_importance=cfg.prune_max_importance,
            max_rank=cfg.prune_max_rank,
        )
        count = await self.sqlite.archive_old_memories(
            days_old=cfg.auto_prune_days,
            max_importance=cfg.prune_max_importance,
            max_rank=cfg.prune_max_rank,
        )
        for mid in ids_to_archive:
            try:
                self.chroma.delete_memory(mid)
            except Exception:
                pass
        return {"pruned": count}

    async def _job_consolidate(self, on_status) -> dict:
        """Merge near-duplicate memories."""
        if self.config.memory.consolidation_batch_size <= 0:
            return {"merged": 0, "skipped": "consolidation_batch_size=0"}

        consolidator = MemoryConsolidator(
            self.sqlite, self.chroma, self.config.memory,
        )
        return await consolidator.consolidate_batch()

    async def _job_tag_discovery(self, on_status) -> dict:
        """Run LLM-powered tag pattern discovery."""
        cfg = self.config.memory
        discovery = TagDiscovery(
            self.sqlite, self.router,
            interval_days=0,  # force run (don't check interval)
            sample_size=cfg.tag_discovery_sample_size,
        )
        stats = await discovery.maybe_run()
        if stats["discovered"] > 0:
            new_patterns = await self.sqlite.get_discovered_tag_patterns()
            register_topic_patterns(new_patterns)
        return stats

    async def close(self):
        """Clean up resources."""
        try:
            await self.sqlite.close()
        except Exception:
            pass
