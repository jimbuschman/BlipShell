"""Nightly job runner for BlipShell.

Orchestrates overnight maintenance: backup, cleanup, tagging, pruning,
consolidation, and tag discovery. Can run interactively via /nightly
or headless via `blipshell nightly`.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from blipshell.core.import_lock import is_import_active, read_lock_info
from blipshell.memory.batch_tagger import BatchTagger
from blipshell.memory.centroid_tagger import CentroidTagger
from blipshell.memory.consolidation import MemoryConsolidator
from blipshell.memory.tag_discovery import TagDiscovery
from blipshell.memory.tagger import register_topic_patterns

logger = logging.getLogger(__name__)

# Jobs run in this order. Each is isolated — one failure doesn't abort the rest.
JOB_ORDER = [
    "backup",
    "backfill_vectors",
    "clean_empty_sessions",
    "cleanup",
    "backfill_summaries",
    "resummarize",
    "backfill_lessons",
    "score_lessons",
    "clean_junk_lessons",
    "session_reflections",
    "self_reflection",
    "revote_lessons",
    "friction_analysis",
    "entity_extraction",
    "entity_cleanup",
    "merge_entities",
    "prune_entities",
    "centroid_tag",
    "batch_tag",
    "prune",
    "consolidate",
    "clean_neutral_tags",
    "tag_discovery",
    "rebuild_digests",
    "update_user_model",
    "export_mirror",
    "health_check",
]

# Jobs that require Ollama (LLM calls or embedding). Skipped if Ollama is down.
# Getting this set wrong is expensive: an omitted job runs with Ollama down and
# burns the full _JOB_TIMEOUT instead of skipping in milliseconds.
#   - tag_discovery -> TagDiscovery.maybe_run() -> router.generate()
#   - consolidate   -> vectors.search_memories() embeds the query text
# (centroid_tag is deliberately absent: it only reads stored embeddings via
# get_embeddings_by_ids and never calls out.)
_OLLAMA_JOBS = {
    "backfill_vectors", "backfill_summaries", "resummarize",
    "backfill_lessons", "score_lessons", "clean_junk_lessons",
    "session_reflections", "self_reflection", "revote_lessons",
    "friction_analysis",
    "entity_extraction", "batch_tag",
    "merge_entities",
    "rebuild_digests",
    "tag_discovery",
    "consolidate",
    "update_user_model",
}

# Max time per job (seconds). Prevents a single hung job from burning hours.
_JOB_TIMEOUT = 300  # 5 minutes per job

# Per-job overrides for jobs whose legitimate work is a long batch. The
# nightly runs headless at 2am with the whole night available — a real
# backlog being SLOW is not the failure the cap exists for (HUNG is; the
# per-call LLM timeouts and per-session caps below cover that).
_JOB_TIMEOUTS = {
    "backfill_lessons": 3600,
}

# Sentinel distinguishing "caller didn't specify a budget, use the nightly
# default" from an explicit None ("no budget, full pass"). Used by
# _job_merge_entities so the scheduled run gets budgeted while the standalone
# script can opt out.
_DEFAULT_BUDGET = object()

# Per-session cap for friction analysis. A between-iterations time budget is not
# enough on its own: one slow/hung LLM call would block the loop past
# _JOB_TIMEOUT (the budget check only runs between sessions). Each session's LLM
# work is wrapped in wait_for(this), and the loop won't START a session unless a
# full per-session budget fits before the cap. Module-level so tests can shrink it.
_FRICTION_SESSION_TIMEOUT = 90.0

# Per-session hang insurance for lesson backfill, NOT a slowness penalty:
# generous, because a large session legitimately chunks into many LLM calls.
# A session that hits it is skipped for the night and retried the next run —
# the loop continues, so one bad session can't block the rest (the 2026-09-04
# timeout: a between-sessions budget alone couldn't stop one oversized
# session from eating the whole job window between checks).
_LESSON_SESSION_TIMEOUT = 900.0


class NightlyRunner:
    """Orchestrates nightly maintenance jobs."""

    def __init__(self, config, sqlite, vectors, router, processor):
        self.config = config
        self.sqlite = sqlite
        self.vectors = vectors
        self.router = router
        self.processor = processor

    async def _check_ollama_health(self, timeout: float = 10.0) -> bool:
        """Quick check: is Ollama responding? Returns False if down."""
        try:
            import ollama
            from blipshell.models.config import get_ollama_url
            url = get_ollama_url(self.config.endpoints)
            client = ollama.Client(host=url)
            # list() is a cheap API call — if it hangs, Ollama is wedged
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, client.list),
                timeout=timeout,
            )
            return True
        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            return False

    @classmethod
    async def create_from_config(
        cls,
        config_path: str | None = None,
        *,
        local_only: bool = False,
    ) -> NightlyRunner:
        """Factory: build a NightlyRunner from config without a full Agent.

        Creates its own SQLite, VectorStore, Router, and Processor instances.
        Caller must call close() when done.

        Args:
            local_only: If True, disable all non-Ollama endpoints so every
                LLM call routes through local Ollama. Useful for bulk nightly
                runs that would overwhelm cloud rate limits.
        """
        from blipshell.core.config import ConfigManager
        from blipshell.llm.endpoints import EndpointManager
        from blipshell.llm.router import LLMRouter
        from blipshell.memory.processor import MemoryProcessor
        from blipshell.memory.sqlite_store import SQLiteStore
        from blipshell.memory.vector_store import VectorStore
        from blipshell.models.config import get_ollama_url

        config_mgr = ConfigManager(config_path)
        config = config_mgr.load()

        endpoints = config.endpoints
        if local_only:
            # Keep only Ollama endpoints, disable cloud (OpenAI-compatible)
            for ep in endpoints:
                if ep.provider != "ollama":
                    ep.enabled = False
            logger.info("Local-only mode: disabled %d cloud endpoints",
                        sum(1 for ep in endpoints if not ep.enabled))

        sqlite = SQLiteStore(config.database.path)
        await sqlite.initialize()

        vectors = VectorStore(
            db_path=config.database.path,
            embedding_model=config.models.embedding,
            ollama_url=get_ollama_url(endpoints),
            embedding_dim=config.database.embedding_dimensions,
        )
        vectors.initialize()

        endpoint_mgr = EndpointManager(endpoints, config.llm)
        router = LLMRouter(config.models, endpoint_mgr)
        processor = MemoryProcessor(sqlite, vectors, router, config=config.memory)

        return cls(config, sqlite, vectors, router, processor)

    async def run(
        self,
        on_status: Optional[Callable[[str], None]] = None,
        jobs: Optional[list[str]] = None,
        force: bool = False,
    ) -> dict:
        """Run all (or specified) nightly jobs in sequence.

        Skips automatically if an import lock is present (force=True overrides).
        Returns stats dict with per-job results.
        """
        run_jobs = jobs or JOB_ORDER
        started_at = time.time()
        results = {}

        def _status(msg: str):
            if on_status:
                on_status(msg)
            logger.info("Nightly: %s", msg)

        # Skip if an import is in progress — both writers fight for the
        # same SQLite file and create orphan vectors.
        db_path = getattr(getattr(self.config, "database", None), "path", None)
        if db_path and not force and is_import_active(db_path):
            info = read_lock_info(db_path) or {}
            op = info.get("operation", "import")
            _status(f"Skipping nightly: {op} in progress (use force=True to override)")
            return {
                "skipped": True,
                "reason": f"{op} in progress",
                "lock_info": info,
                "started_at": started_at,
            }

        _status(f"Starting nightly run ({len(run_jobs)} jobs)...")

        # Pre-flight: check if Ollama is responsive before running LLM jobs.
        # If it's down, skip LLM-dependent jobs instead of timing out on each.
        ollama_ok = await self._check_ollama_health()
        if not ollama_ok:
            _status("Ollama not responding — LLM-dependent jobs will be skipped")

        for job_name in run_jobs:
            if job_name not in JOB_ORDER:
                _status(f"Unknown job: {job_name}, skipping.")
                results[job_name] = {"status": "skipped", "error": "unknown job"}
                continue

            # Skip LLM jobs if Ollama is down
            if not ollama_ok and job_name in _OLLAMA_JOBS:
                _status(f"  {job_name} skipped (Ollama down)")
                results[job_name] = {"status": "skipped", "error": "Ollama not responding"}
                continue

            _status(f"Running job: {job_name}...")
            t0 = time.monotonic()
            job_timeout = _JOB_TIMEOUTS.get(job_name, _JOB_TIMEOUT)
            try:
                job_result = await asyncio.wait_for(
                    self.run_job(job_name, on_status=on_status),
                    timeout=job_timeout,
                )
                elapsed = time.monotonic() - t0
                job_result["status"] = "ok"
                job_result["elapsed_s"] = round(elapsed, 1)
                results[job_name] = job_result
                _status(f"  {job_name} completed in {elapsed:.1f}s")
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - t0
                logger.error("Nightly job %s timed out after %ds", job_name, job_timeout)
                results[job_name] = {
                    "status": "timeout",
                    "error": f"Timed out after {job_timeout}s",
                    "elapsed_s": round(elapsed, 1),
                }
                _status(f"  {job_name} TIMED OUT after {job_timeout}s")
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
        except Exception as e:
            logger.warning("Failed to persist nightly run metadata: %s", e)

        _status(f"Nightly run complete in {total_elapsed:.0f}s.")

        full_results = {
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_s": round(total_elapsed, 1),
            "jobs": results,
        }

        # Build and store structured report for startup notification
        try:
            await self._build_and_store_report(full_results)
        except Exception as e:
            logger.warning("Failed to build nightly report: %s", e)

        return full_results

    async def run_job(
        self,
        job_name: str,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Run a single job by name. Returns job-specific stats dict."""
        handlers = {
            "backup": self._job_backup,
            "update_user_model": self._job_update_user_model,
            "backfill_vectors": self._job_backfill_vectors,
            "clean_empty_sessions": self._job_clean_empty_sessions,
            "cleanup": self._job_cleanup,
            "backfill_summaries": self._job_backfill_summaries,
            "resummarize": self._job_resummarize,
            "backfill_lessons": self._job_backfill_lessons,
            "score_lessons": self._job_score_lessons,
            "clean_junk_lessons": self._job_clean_junk_lessons,
            "session_reflections": self._job_session_reflections,
            "self_reflection": self._job_self_reflection,
            "revote_lessons": self._job_revote_lessons,
            "friction_analysis": self._job_friction_analysis,
            "entity_extraction": self._job_entity_extraction,
            "entity_cleanup": self._job_entity_cleanup,
            "centroid_tag": self._job_centroid_tag,
            "batch_tag": self._job_batch_tag,
            "prune": self._job_prune,
            "merge_entities": self._job_merge_entities,
            "prune_entities": self._job_prune_entities,
            "consolidate": self._job_consolidate,
            "clean_neutral_tags": self._job_clean_neutral_tags,
            "tag_discovery": self._job_tag_discovery,
            "rebuild_digests": self._job_rebuild_digests,
            "export_mirror": self._job_export_mirror,
            "health_check": self._job_health_check,
        }
        handler = handlers.get(job_name)
        if not handler:
            raise ValueError(f"Unknown job: {job_name}")
        return await handler(on_status or (lambda msg: None))

    async def _job_backup(self, on_status) -> dict:
        """Run pre-operation backup."""
        try:
            from scripts.backup_db import backup_before_destructive
            result = backup_before_destructive(
                "nightly",
                db_path=self.config.database.path,
            )
            return {"backup_path": str(result) if result else None}
        except Exception as e:
            logger.warning("Backup failed (non-fatal): %s", e)
            return {"backup_path": None, "warning": str(e)}

    async def _job_backfill_vectors(self, on_status) -> dict:
        """Backfill any missing vector embeddings.

        "reflections" was missing from this list until 2026-09-02, so only
        reflections embedded at write time had vectors — the 2026-09 audit
        found 1,720 of 1,808 invisible to search_lessons' reflections leg.
        """
        total = {"succeeded": 0, "failed": 0}
        for collection in ("memories", "core_memories", "lessons", "entities",
                           "reflections"):
            stats = self.vectors.backfill_missing_vectors(collection, limit=500)
            total["succeeded"] += stats.get("succeeded", 0)
            total["failed"] += stats.get("failed", 0)
            if stats.get("succeeded", 0) > 0:
                on_status(f"Backfilled {stats['succeeded']} {collection} vectors")
        return total

    async def _job_clean_empty_sessions(self, on_status) -> dict:
        """Delete sessions with zero memories (app started but user never chatted)."""
        count = await self.sqlite.delete_empty_sessions(min_age_hours=24)
        if count:
            on_status(f"Deleted {count} empty sessions")
        return {"deleted": count}

    async def _job_cleanup(self, on_status) -> dict:
        """Reprocess failed memories."""
        unprocessed = await self.sqlite.get_unprocessed_memories(limit=500)
        if not unprocessed:
            return {"processed": 0, "failed": 0, "total": 0}

        processed = 0
        failed = 0
        for msg in unprocessed:
            try:
                await self.processor.process_message(
                    text=msg["content"],
                    role=msg["role"],
                    session_id=msg["session_id"],
                    memory_id=msg["id"],
                )
                processed += 1
            except Exception as e:
                logger.warning("Failed to reprocess memory %d: %s", msg["id"], e)
                failed += 1

        return {"processed": processed, "failed": failed, "total": len(unprocessed)}

    async def _job_backfill_summaries(self, on_status) -> dict:
        """Generate summaries for sessions that were imported without them."""
        from scripts.backfill_session_summaries import summarize_session

        sessions = await self.sqlite.get_sessions_without_summaries(limit=500)
        if not sessions:
            return {"processed": 0, "total": 0}

        processed = 0
        failed = 0
        for session in sessions:
            sid = session["id"]
            try:
                summary, title = await summarize_session(
                    self.sqlite, self.router, sid,
                )
                if summary:
                    await self.sqlite.update_session(
                        sid, summary=summary, title=title,
                    )
                    processed += 1
            except Exception as e:
                logger.error("Backfill failed for session %d: %s", sid, e)
                failed += 1

        return {"processed": processed, "failed": failed, "total": len(sessions)}

    async def _job_resummarize(self, on_status) -> dict:
        """Re-summarize memories where summary = content (import failures)."""
        from blipshell.llm.prompts import summarize_memory
        from blipshell.llm.router import TaskType

        memories = await self.sqlite.get_unsummarized_memories()
        if not memories:
            return {"resummarized": 0, "remaining": 0, "total": 0}

        total = len(memories)
        on_status(f"Re-summarizing {total} memories...")
        resummarized = 0
        failed = 0
        for i, mem in enumerate(memories):
            if (i + 1) % 50 == 0:
                on_status(f"Re-summarizing {i + 1}/{total} ({resummarized} done, {failed} failed)...")
            try:
                sum_system, sum_prompt = summarize_memory(mem.content)
                summary = await self.router.generate(
                    TaskType.SUMMARIZATION,
                    sum_prompt,
                    system=sum_system,
                )
                if summary and summary.strip().upper() != "SKIP" and summary.strip() != mem.content.strip():
                    await self.sqlite.update_memory(mem.id, summary=summary.strip())
                    # Update vector store embedding with proper summary
                    try:
                        self.vectors.add_memory(mem.id, summary.strip())
                    except Exception as e:
                        logger.debug("vector store update failed for memory %d: %s", mem.id, e)
                    resummarized += 1
                else:
                    # Mark it so we don't retry — set summary to truncated content
                    truncated = mem.content[:300] + "..." if len(mem.content) > 300 else mem.content
                    await self.sqlite.update_memory(mem.id, summary=truncated)
                    resummarized += 1
            except Exception as e:
                logger.error("Re-summarize failed for memory %d: %s", mem.id, e)
                failed += 1

        # Count remaining
        remaining_rows = await self.sqlite._db.execute_fetchall(
            "SELECT COUNT(*) as cnt FROM memories WHERE summary = content "
            "AND is_archived = 0 AND length(content) > 200"
        )
        remaining = remaining_rows[0][0] if remaining_rows else 0

        return {
            "resummarized": resummarized,
            "failed": failed,
            "remaining": remaining,
            "total": len(memories),
        }

    async def _job_centroid_tag(self, on_status) -> dict:
        """Run centroid-based tag assignment."""
        tagger = CentroidTagger(self.sqlite, self.vectors, self.config.memory)
        return await tagger.run(on_status=on_status)

    async def _job_batch_tag(self, on_status) -> dict:
        """Run LLM batch tag assignment.

        Passes a time budget that's slightly less than the per-job timeout
        so the tagger exits cleanly with partial progress instead of being
        killed mid-batch by the outer ``wait_for``. Remaining work resumes
        the next nightly run.
        """
        tagger = BatchTagger(self.sqlite, self.router, self.config.memory)
        return await tagger.tag_all(
            on_status=on_status,
            time_budget_seconds=_JOB_TIMEOUT - 30,
        )

    async def _job_prune(self, on_status) -> dict:
        """Archive old low-value memories, then sweep their vectors.

        Previous approach deleted vectors one-by-one immediately after
        archiving, but the async SQLite connection (aiosqlite) and the
        sync vector connection (sqlite3) both write to the same DB file
        and race for the write lock. Now: archive first, commit, then
        sweep all orphan vectors in one batch through a single connection.
        """
        cfg = self.config.memory
        if cfg.auto_prune_days <= 0:
            return {"pruned": 0, "skipped": "auto_prune_days=0"}

        count = await self.sqlite.archive_old_memories(
            days_old=cfg.auto_prune_days,
            max_importance=cfg.prune_max_importance,
            max_rank=cfg.prune_max_rank,
        )

        # Sweep orphan vectors in one batch — no per-ID delete loop, no
        # lock contention between aiosqlite and sync sqlite3 connections.
        sweep = self.vectors.cleanup_orphan_vectors()
        return {
            "pruned": count,
            "orphan_sweep": sweep,
        }

    async def _job_merge_entities(
        self, on_status, on_progress=None, time_budget_seconds=_DEFAULT_BUDGET,
    ) -> dict:
        """Retroactively merge duplicate entities already in the graph.

        Consolidates duplicates that creation-time resolution missed (e.g. those
        created while nightly extraction ran without dedup). Conservative,
        config-driven, dry-run aware. Runs before prune_entities so merged-away
        husks don't get treated as independent low-value nodes.

        The full-graph scan can't finish in one _JOB_TIMEOUT, so by default this
        runs merge_pass under a time budget of (_JOB_TIMEOUT - 30): it makes
        resumable partial progress every scheduled run instead of always timing
        out (merged husks are archived, so the graph converges over nights).
        on_progress (scanned, total, stats) is forwarded for a heartbeat. The
        standalone scripts/merge_entities.py runner passes
        time_budget_seconds=None for an uncapped full pass / complete preview.
        """
        cfg = self.config.memory
        if not getattr(cfg, "entity_merge_enabled", False):
            return {"skipped": "entity_merge_enabled=false"}
        if self.vectors is None:
            return {"skipped": "no vector store (similarity search unavailable)"}

        from blipshell.memory.entity_merger import EntityMerger

        merger = EntityMerger(
            self.sqlite, self.router, self.vectors,
            auto_threshold=cfg.entity_merge_auto_threshold,
            llm_threshold=cfg.entity_merge_llm_threshold,
            max_candidates=cfg.entity_merge_max_candidates,
            edge_sample=cfg.entity_merge_edge_sample,
        )
        if time_budget_seconds is _DEFAULT_BUDGET:
            time_budget_seconds = _JOB_TIMEOUT - 30
        dry_run = getattr(cfg, "entity_merge_dry_run", True)
        result = await merger.merge_pass(
            dry_run=dry_run, on_progress=on_progress,
            time_budget_seconds=time_budget_seconds,
        )
        # If a time budget cut the scan short, say so — the numbers are partial
        # and the rest is picked up next run.
        if result.get("stopped_early"):
            scan_note = (
                f" [partial: scanned {result['entities_scanned']}/"
                f"{result['entities_total']}, resumes next run]"
            )
        else:
            scan_note = ""

        if dry_run:
            # Persist the confidence-sorted plan so it can be read through before
            # committing — the in-result sample is just a teaser. Keep the
            # returned/persisted dict small by dropping the full plan from it.
            # Note: under a time budget this preview is PARTIAL; use the
            # standalone scripts/merge_entities.py for a complete one.
            plan = result.pop("plan", [])
            preview_path = Path(self.config.database.path).parent / "entity_merge_preview.json"
            try:
                preview_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
                result["preview_file"] = str(preview_path)
            except Exception as e:
                logger.warning("Failed to write merge preview: %s", e)
            on_status(
                f"[dry-run] {result['would_merge']} entity merges proposed "
                f"({result['auto_merges']} auto, {result['llm_merges']} LLM, "
                f"{result['llm_rejects']} rejected) — nothing changed; "
                f"sorted preview -> {preview_path}{scan_note}"
            )
        else:
            on_status(
                f"Merged {result['merged']} duplicate entities "
                f"({result['auto_merges']} auto, {result['llm_merges']} LLM){scan_note}"
            )
        return result

    async def _job_prune_entities(self, on_status) -> dict:
        """Soft-archive low-value entities from the graph (reversible, dry-run aware).

        Thresholds are config-driven (entity_prune_* on MemoryConfig). An entity
        is a candidate when it is older than min_age_days AND has <= max_mentions
        mentions AND <= max_degree relationships. Disabled by default; when
        enabled it runs in dry-run mode (log-only) until entity_prune_dry_run is
        explicitly set false. Archiving flips a flag — it never deletes, so it is
        fully reversible.
        """
        cfg = self.config.memory
        if not getattr(cfg, "entity_prune_enabled", False):
            return {"skipped": "entity_prune_enabled=false"}

        candidates = await self.sqlite.get_prunable_entities(
            min_age_days=cfg.entity_prune_min_age_days,
            max_mentions=cfg.entity_prune_max_mentions,
            max_degree=cfg.entity_prune_max_degree,
        )

        dry_run = getattr(cfg, "entity_prune_dry_run", True)
        sample = [
            {"id": c["id"], "name": c["name"], "type": c["entity_type"],
             "mentions": c["mentions"], "degree": c["degree"]}
            for c in candidates[:25]
        ]

        if dry_run:
            on_status(
                f"[dry-run] {len(candidates)} entities WOULD be archived "
                f"(age>{cfg.entity_prune_min_age_days}d, mentions<="
                f"{cfg.entity_prune_max_mentions}, degree<="
                f"{cfg.entity_prune_max_degree}) — nothing changed"
            )
            for c in sample:
                logger.info(
                    "[dry-run] would archive entity id=%d '%s' (%s) mentions=%d degree=%d",
                    c["id"], c["name"], c["type"], c["mentions"], c["degree"],
                )
            return {
                "dry_run": True,
                "would_prune": len(candidates),
                "sample": sample,
            }

        archived = await self.sqlite.archive_entities([c["id"] for c in candidates])
        on_status(f"Soft-archived {archived} low-value entities")
        return {
            "dry_run": False,
            "archived": archived,
            "candidates": len(candidates),
            "sample": sample,
        }

    async def _job_consolidate(self, on_status) -> dict:
        """Merge near-duplicate memories."""
        if self.config.memory.consolidation_batch_size <= 0:
            return {"merged": 0, "skipped": "consolidation_batch_size=0"}

        consolidator = MemoryConsolidator(
            self.sqlite, self.vectors, self.config.memory,
        )
        # Resumable: leave headroom under the job timeout so a partial pass
        # returns its stats (and its stopped_early flag) instead of being
        # hard-killed with nothing recorded.
        if consolidator.dry_run:
            on_status("  consolidate: DRY RUN — proposing merges, changing nothing")
        return await consolidator.consolidate_batch(
            time_budget_seconds=_JOB_TIMEOUT - 30,
            on_status=on_status,
        )

    async def _job_clean_neutral_tags(self, on_status) -> dict:
        """Remove 'neutral' tag from memories that have other tags.

        The 'neutral' tag is too broad (39% of memories) and adds no
        discriminative value for search. Stripping it from multi-tagged
        memories reduces noise without losing information.
        """
        removed = 0
        try:
            # Find memories tagged 'neutral' that also have other tags
            rows = await self.sqlite._db.execute_fetchall(
                """
                SELECT mt.memory_id
                FROM memory_tags mt
                JOIN tags t ON t.id = mt.tag_id
                WHERE t.name = 'neutral'
                  AND mt.memory_id IN (
                      SELECT mt2.memory_id FROM memory_tags mt2
                      GROUP BY mt2.memory_id HAVING COUNT(*) > 1
                  )
                """
            )
            if rows:
                memory_ids = [r[0] for r in rows]
                tag_row = await self.sqlite._db.execute_fetchall(
                    "SELECT id FROM tags WHERE name = 'neutral'"
                )
                if tag_row:
                    neutral_tag_id = tag_row[0][0]
                    for mid in memory_ids:
                        await self.sqlite._db.execute(
                            "DELETE FROM memory_tags WHERE memory_id = ? AND tag_id = ?",
                            (mid, neutral_tag_id),
                        )
                        removed += 1
                    await self.sqlite._db.commit()
        except Exception as e:
            logger.warning("clean_neutral_tags error: %s", e)
        return {"neutral_tags_removed": removed}

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

    async def _job_backfill_lessons(self, on_status) -> dict:
        """Re-extract lessons from sessions with messages but no lessons."""
        sessions = await self.sqlite.get_sessions_missing_lessons(limit=500)
        if not sessions:
            return {"processed": 0, "total": 0}

        # Two layers, same pattern (and same bug history) as friction_analysis:
        # (1) a between-sessions budget that won't START a session it can't
        # finish inside the job's window, and (2) a per-session wait_for as
        # hang insurance, so one wedged extraction can't run to the hard job
        # kill (the 2026-09-04 timeout: budget alone was not enough). A
        # timed-out session is simply skipped and retried the next run — the
        # loop continues past it, so it can't block the rest of the backlog.
        budget = _JOB_TIMEOUTS["backfill_lessons"] - _LESSON_SESSION_TIMEOUT - 15
        start = time.monotonic()
        stopped_early = False
        processed = 0
        failed = 0
        timed_out = 0
        cloud_routed = 0
        for session in sessions:
            if time.monotonic() - start > budget:
                stopped_early = True
                logger.info(
                    "backfill_lessons hit time budget (%ds), stopping with "
                    "partial progress (%d/%d)", budget, processed, len(sessions))
                break
            sid = session["id"]
            project = session.get("project")
            try:
                was_cloud = await asyncio.wait_for(
                    self._backfill_one_session_lesson(sid, project),
                    timeout=_LESSON_SESSION_TIMEOUT,
                )
                cloud_routed += 1 if was_cloud else 0
                processed += 1
            except asyncio.TimeoutError:
                timed_out += 1
                logger.warning(
                    "backfill_lessons: session %d exceeded %.0fs (hang "
                    "insurance), skipping — retried next run",
                    sid, _LESSON_SESSION_TIMEOUT)
            except Exception as e:
                logger.error("Lesson backfill failed for session %d: %s", sid, e)
                failed += 1

        return {"processed": processed, "failed": failed,
                "timed_out": timed_out,
                "cloud_routed": cloud_routed, "total": len(sessions),
                "stopped_early": stopped_early}

    async def _backfill_one_session_lesson(self, sid: int,
                                           project: Optional[str]) -> bool:
        """Extract the lesson for one session; returns True if cloud-routed.

        Split out so the caller can bound the whole thing (message fetch,
        chunking, every LLM call) with a single ``wait_for``.
        """
        from blipshell.memory.manager import estimate_tokens

        messages = await self.sqlite.get_session_messages_for_lesson(sid)
        conversation_lines = [f"{m['role']}: {m['content']}" for m in messages]
        conversation_text = "\n".join(conversation_lines)
        tokens = estimate_tokens(conversation_text)
        # Route large sessions to bigger-context endpoint
        min_ctx = tokens + 4096 if tokens > 28000 else None
        await self.processor.process_lesson(
            conversation_text, sid, project=project,
            min_context_tokens=min_ctx,
        )
        return min_ctx is not None

    async def _job_score_lessons(self, on_status) -> dict:
        """Score lessons that still have default rank=3/importance=0.5."""
        from blipshell.llm.prompts import rank_lesson
        from blipshell.llm.router import TaskType
        from blipshell.memory.processor import MemoryProcessor

        unscored = await self.sqlite.get_unscored_lessons()
        if not unscored:
            return {"scored": 0, "total": 0}

        total = len(unscored)
        on_status(f"Scoring {total} unscored lessons...")
        scored = 0
        failed = 0
        for i, lesson in enumerate(unscored):
            if (i + 1) % 50 == 0:
                on_status(f"Scoring lessons {i + 1}/{total} ({scored} done)...")
            try:
                ri_system, ri_prompt = rank_lesson(lesson.content)
                ri_text = await self.router.generate(
                    TaskType.RANKING_IMPORTANCE,
                    ri_prompt,
                    system=ri_system,
                )
                rank, importance = MemoryProcessor._parse_rank_and_importance(ri_text)
                await self.sqlite.update_lesson_scores(lesson.id, rank, importance)
                scored += 1
            except Exception as e:
                logger.error("Lesson %d scoring failed: %s", lesson.id, e)
                failed += 1

        return {"scored": scored, "failed": failed, "total": len(unscored)}

    async def _job_clean_junk_lessons(self, on_status) -> dict:
        """Delete junk lessons (SKIP, empty, short) and dedup near-identical ones."""
        db = self.sqlite._db
        deleted_junk = 0
        deleted_dupes = 0

        # Phase 1: Delete obvious junk
        junk_rows = await db.execute_fetchall(
            "SELECT id, content FROM lessons WHERE "
            "TRIM(content) = 'SKIP' OR TRIM(content) = '' OR length(TRIM(content)) < 20"
        )
        for row in junk_rows:
            lid = row[0]
            await db.execute("DELETE FROM lesson_tags WHERE lesson_id = ?", (lid,))
            await db.execute("DELETE FROM lessons WHERE id = ?", (lid,))
            try:
                self.vectors.delete_lesson(lid)
            except Exception as e:
                logger.debug("Failed to delete junk lesson %d vector: %s", lid, e)
            deleted_junk += 1
        if deleted_junk:
            await db.commit()

        # Phase 2: Dedup near-identical lessons via vector store similarity
        lessons = await self.sqlite.get_all_lessons()
        seen_ids = set()
        for lesson in lessons:
            if lesson.id in seen_ids:
                continue
            try:
                similar = self.vectors.search_lessons(lesson.content, n_results=5)
                for s in similar:
                    other_id = s.get("id")
                    if other_id == lesson.id or other_id in seen_ids:
                        continue
                    if s.get("similarity", 0) > 0.92:
                        # Keep the one with higher importance, or the older one
                        await db.execute("DELETE FROM lesson_tags WHERE lesson_id = ?", (other_id,))
                        await db.execute("DELETE FROM lessons WHERE id = ?", (other_id,))
                        try:
                            self.vectors.delete_lesson(other_id)
                        except Exception as e:
                            logger.debug("Failed to delete dupe lesson %d vector: %s", other_id, e)
                        seen_ids.add(other_id)
                        deleted_dupes += 1
            except Exception as e:
                logger.debug("Lesson dedup check failed for %d: %s", lesson.id, e)

        if deleted_dupes:
            await db.commit()

        # Phase 3: fold paraphrase families (2026-09-02 audit follow-up).
        # The vector dedup above catches near-verbatim copies (>0.92); the
        # audit found the real accumulation was PARAPHRASES — 456 lessons in
        # 127 Jaccard families that vectors scored as distinct. Threshold is
        # deliberately stricter (default 0.35) than the audit's 0.20 measuring
        # ruler: the hand-reviewed one-shot found 6 false families in 127 at
        # 0.20, and an unattended job gets no human review, so it only merges
        # blatant rewordings. Keep-best (importance, then length, then
        # newest); receipts written next to the database before deletion.
        folded = 0
        fold_threshold = getattr(
            self.config.memory, "lesson_family_fold_threshold", 0.35)
        if fold_threshold and fold_threshold < 1.0:
            from collections import defaultdict

            from blipshell.memory.themes import family_sizes

            rows = await db.execute_fetchall(
                "SELECT id, importance, content FROM lessons ORDER BY id")
            ids = [r[0] for r in rows]
            info = {r[0]: (r[1] or 0.0, r[2] or "") for r in rows}
            fam, _ = family_sizes([info[i][1] for i in ids],
                                  threshold=fold_threshold,
                                  link="representative")
            groups = defaultdict(list)
            for idx, f in enumerate(fam):
                groups[f].append(ids[idx])
            doomed: list[int] = []
            for members in groups.values():
                if len(members) < 2:
                    continue
                keeper = max(members, key=lambda lid: (
                    info[lid][0], len(info[lid][1]), lid))
                doomed.extend(m for m in members if m != keeper)
            if doomed:
                receipt_rows = await db.execute_fetchall(
                    f"SELECT id, content, summary, timestamp, "
                    f"source_session_id, project FROM lessons "
                    f"WHERE id IN ({','.join('?' * len(doomed))})", doomed)
                from datetime import datetime, timezone
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                receipt = (Path(self.config.database.path).parent
                           / f"lessons_folded_{stamp}.json")
                try:
                    receipt.write_text(json.dumps(
                        [dict(zip(("id", "content", "summary", "timestamp",
                                   "source_session_id", "project"), r))
                         for r in receipt_rows], indent=1, ensure_ascii=False),
                        encoding="utf-8")
                except Exception as e:
                    logger.warning("Fold receipts failed, skipping fold: %s", e)
                else:
                    for lid in doomed:
                        await db.execute(
                            "DELETE FROM lesson_tags WHERE lesson_id = ?", (lid,))
                        await db.execute(
                            "DELETE FROM lessons WHERE id = ?", (lid,))
                        try:
                            self.vectors.delete_lesson(lid)
                        except Exception:
                            pass  # orphan vectors swept by maintenance
                        folded += 1
                    await db.commit()
                    # A folded lesson's session must not look never-extracted,
                    # or backfill_lessons re-creates the duplicate next run.
                    await self.sqlite.add_lesson_backfill_exclusions(
                        r[4] for r in receipt_rows if r[4] is not None)
                    on_status(f"  folded {folded} paraphrase-duplicate "
                              f"lesson(s); receipts -> {receipt.name}")

        return {"deleted_junk": deleted_junk, "deleted_dupes": deleted_dupes,
                "folded": folded}

    async def _job_session_reflections(self, on_status) -> dict:
        """Generate holistic reflections for unreflected sessions.

        Like friction analysis, each session is one LLM call, so a full batch
        exceeds the per-job timeout. Process under a time budget and exit with
        partial progress — reflected sessions are recorded in
        ``session_reflections`` and excluded next run, so the next nightly run
        resumes the backlog instead of being hard-killed mid-loop.
        """
        sessions = await self.sqlite.get_sessions_missing_reflections(limit=200)
        if not sessions:
            return {"processed": 0, "skipped": 0, "total": 0}

        budget = _JOB_TIMEOUT - 30
        start = time.monotonic()
        processed = 0
        skipped = 0
        failed = 0
        cloud_routed = 0
        stopped_early = False
        for session in sessions:
            if time.monotonic() - start > budget:
                stopped_early = True
                logger.info(
                    "session_reflections hit time budget (%ds), stopping with "
                    "partial progress (%d/%d sessions)",
                    budget, processed, len(sessions),
                )
                break
            sid = session["id"]
            summary = session["summary"]
            project = session.get("project")
            try:
                chunks, total_tokens = await self.processor.prepare_conversation_for_reflection(
                    sid, summary,
                )
                # Route large sessions to bigger-context endpoint
                min_ctx = total_tokens + 4096 if total_tokens > 28000 else None
                if min_ctx:
                    cloud_routed += 1
                result = await self.processor.process_reflection(
                    session_id=sid,
                    session_summary=summary,
                    conversation_chunks=chunks,
                    project=project,
                    min_context_tokens=min_ctx,
                )
                if result is None:
                    skipped += 1
                else:
                    processed += 1
            except Exception as e:
                logger.error("Session reflection failed for session %d: %s", sid, e)
                failed += 1

        return {
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "cloud_routed": cloud_routed,
            "total": len(sessions),
            "stopped_early": stopped_early,
        }

    async def _job_self_reflection(self, on_status) -> dict:
        """Form one self-originated lingering thought, app open or not.

        The idle loop and on-return reflection only fire around a 3h+ gap the
        PROCESS witnesses; on open-chat-close usage that measured out to ~1
        thought/month (scripts/diagnose_self_thoughts.py), which left the
        self-gravity step-2 gate ("10 NEW thoughts") roughly a year away. The
        nightly run happens regardless of how the app is used, so this is
        where a steady cadence lives: one thought per night, generated exactly
        the way the agent's own loop generates one — diversity-sampled priors,
        same prompt, TaskType.REFLECTION routing, NOTHING respected.

        Also reports theme-diversity stats over the active thought corpus
        (blipshell/memory/themes.py) every run, generated or skipped — the
        step-2 gate question "is resurfacing caring or indexing?" finally gets
        a number next to the feeling, and a nightly series shows drift.
        """
        from datetime import datetime, timezone

        from blipshell.core.self_reflection import (
            NOTHING, SelfThoughtStore, lingering_thought_prompt,
        )
        from blipshell.llm.router import TaskType
        from blipshell.memory.themes import theme_diversity

        refl = self.config.reflection
        if not refl.enabled:
            return {"skipped": "reflection.enabled=false"}
        if not getattr(refl, "nightly_enabled", False):
            return {"skipped": "reflection.nightly_enabled=false"}

        async def _embed(text: str):
            if self.vectors is None:
                return None
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.vectors.embed_text, text)

        store = SelfThoughtStore(
            self.sqlite,
            max_keep=refl.max_keep,
            embed_fn=_embed,
            gravity_enabled=refl.gravity_enabled,
            recur_threshold=refl.gravity_recur_threshold,
            recur_boost=refl.gravity_recur_boost,
            fatigue=refl.gravity_fatigue,
            half_life_days=refl.gravity_half_life_days,
            min_weight=refl.gravity_min_weight,
        )

        def _themes_now(items: list[dict]) -> dict:
            return theme_diversity([it["text"] for it in items])

        items = await self.sqlite.get_self_thoughts(with_embeddings=False)

        # Don't pile a second thought onto a day that already produced one
        # (idle loop or on-return reflection) — cadence, not volume, is the
        # point, and near-simultaneous thoughts are the raw material of the
        # duplicate-fold pathologies the store already had to repair once.
        min_gap = getattr(refl, "nightly_min_gap_hours", 12.0)
        newest_iso = items[-1].get("created_at") if items else None
        if newest_iso and min_gap > 0:
            try:
                newest = datetime.fromisoformat(newest_iso)
                if newest.tzinfo is None:
                    newest = newest.replace(tzinfo=timezone.utc)
                age_hours = (
                    datetime.now(timezone.utc) - newest
                ).total_seconds() / 3600.0
                if age_hours < min_gap:
                    return {
                        "generated": 0,
                        "skipped": (
                            f"newest thought is {age_hours:.1f}h old "
                            f"(< nightly_min_gap_hours={min_gap})"
                        ),
                        "themes": _themes_now(items),
                    }
            except (ValueError, TypeError):
                pass  # unparseable stamp: don't let it block the cadence

        prior = await store.diverse_recent(5)
        system, user = lingering_thought_prompt(prior)
        text = (
            await self.router.generate(TaskType.REFLECTION, user, system=system)
        ).strip()
        if not text or text.upper().startswith(NOTHING):
            on_status("  self_reflection: nothing pressing")
            return {
                "generated": 0,
                "nothing_pressing": True,
                "themes": _themes_now(items),
            }

        await store.add(text)
        on_status(f"  self_reflection: {text[:100]}")
        items = await self.sqlite.get_self_thoughts(with_embeddings=False)
        return {
            "generated": 1,
            "thought": text[:200],
            "themes": _themes_now(items),
        }

    _REVOTE_WATERMARK = "lesson_revote_watermark"

    async def _job_revote_lessons(self, on_status) -> dict:
        """Revote lessons against the night's fresh session reflections.

        ExpeL-style lifecycle (2026-09-02 audit follow-up): each new
        reflection is paired with the lessons most similar to it, and the
        LOCAL model judges CONFIRMS / CONTRADICTS / NEUTRAL. Votes move
        importance (down harder than up); the lessons pool's top-30 cut does
        the rest. Demotion only — never deletion — so a lesson can recover.

        Ships disabled, and dry-run when enabled: it reports what it WOULD
        do until lesson_revote_dry_run is explicitly false. The watermark
        always advances to the last reflection read, so a dry-run night is a
        report on that night's evidence, not a growing replay.
        """
        from blipshell.llm.router import TaskType
        from blipshell.memory.lesson_revote import (
            JUDGE_SYSTEM, adjusted_importance, parse_verdict, revote_prompt,
        )

        cfg = self.config.memory
        if not getattr(cfg, "lesson_revote_enabled", False):
            return {"skipped": "lesson_revote_enabled=false"}
        if self.vectors is None:
            return {"skipped": "no vector store (lesson pairing unavailable)"}

        since = await self.sqlite.get_metadata(self._REVOTE_WATERMARK)
        reflections = await self.sqlite.get_reflection_texts_since(
            since, limit=12,
        )
        if not reflections:
            return {"reflections": 0, "pairs": 0}

        lessons = {l.id: l for l in await self.sqlite.get_all_lessons()}
        dry_run = getattr(cfg, "lesson_revote_dry_run", True)
        per_reflection = getattr(cfg, "lesson_revote_per_reflection", 3)
        max_pairs = getattr(cfg, "lesson_revote_max_pairs", 40)

        stats = {"reflections": len(reflections), "pairs": 0,
                 "confirms": 0, "contradicts": 0, "neutral": 0,
                 "no_verdict": 0, "dry_run": dry_run}
        votes: list[dict] = []
        loop = asyncio.get_running_loop()
        last_read = since
        for text, created_at in reflections:
            last_read = created_at
            if not text or text.startswith("Session skipped"):
                continue
            try:
                similar = await loop.run_in_executor(
                    None, functools.partial(
                        self.vectors.search_lessons, text,
                        n_results=per_reflection),
                )
            except Exception as e:
                logger.warning("Lesson pairing failed: %s", e)
                continue
            for hit in similar:
                if stats["pairs"] >= max_pairs:
                    break
                lesson = lessons.get(hit.get("id"))
                if lesson is None:
                    continue
                stats["pairs"] += 1
                try:
                    reply = await self.router.generate(
                        TaskType.REASONING,
                        revote_prompt(lesson.content, text),
                        system=JUDGE_SYSTEM,
                    )
                except Exception as e:
                    logger.warning("Revote judge failed: %s", e)
                    stats["no_verdict"] += 1
                    continue
                verdict = parse_verdict(reply)
                if verdict is None:
                    stats["no_verdict"] += 1
                    continue
                stats[verdict.lower()] += 1
                if verdict == "NEUTRAL":
                    continue
                new_importance = adjusted_importance(
                    lesson.importance, verdict,
                    getattr(cfg, "lesson_revote_up", 0.05),
                    getattr(cfg, "lesson_revote_down", 0.15),
                )
                votes.append({"lesson_id": lesson.id, "verdict": verdict,
                              "old": lesson.importance, "new": new_importance})
                if not dry_run and new_importance != lesson.importance:
                    await self.sqlite.update_lesson_scores(
                        lesson.id, lesson.rank, new_importance,
                    )
                    lesson.importance = new_importance

        await self.sqlite.set_metadata(self._REVOTE_WATERMARK, last_read)
        stats["votes"] = votes[:20]
        mode = "[dry-run] would move" if dry_run else "moved"
        moved = [v for v in votes if v["old"] != v["new"]]
        on_status(f"  revote_lessons: {stats['pairs']} pairs judged, "
                  f"{mode} {len(moved)} importances "
                  f"({stats['confirms']} confirm / {stats['contradicts']} contradict)")
        return stats

    async def _job_friction_analysis(self, on_status) -> dict:
        """Analyze recent sessions for system-level friction.

        Each session needs a SESSION_REVIEW LLM call, so a full 200-session
        batch far exceeds the per-job timeout. Two layers keep it under the cap:
        (1) a between-sessions time budget that won't START a session it can't
        finish, and (2) a per-session ``wait_for`` so one slow/hung call can't
        block the loop past the budget check (the bug that let this job time
        out at 300s despite the budget). Completed sessions are marked in
        ``friction_log`` (real items or a NONE sentinel); timed-out/unanalyzed
        ones stay unmarked so the next nightly run resumes them.
        """
        sessions = await self.sqlite.get_sessions_missing_friction_analysis(limit=200)
        if not sessions:
            return {"processed": 0, "friction_items": 0, "total": 0}

        # Reserve a full per-session budget before the cap so the loop never
        # STARTS a session it can't finish in time — and each session's LLM work
        # is independently bounded below, so a single hung call can't run past it.
        budget = _JOB_TIMEOUT - _FRICTION_SESSION_TIMEOUT - 15
        start = time.monotonic()
        processed = 0
        total_items = 0
        failed = 0
        timed_out = 0
        stopped_early = False
        for session in sessions:
            if time.monotonic() - start > budget:
                stopped_early = True
                logger.info(
                    "friction_analysis hit time budget (%.0fs), stopping with "
                    "partial progress (%d/%d sessions)",
                    budget, processed, len(sessions),
                )
                break
            sid = session["id"]
            summary = session["summary"]
            project = session.get("project")
            try:
                # Bound the per-session LLM work. On timeout the loop regains
                # control (the session is left unmarked → retried next run) even
                # if the underlying call is a non-cancellable router/Ollama gate.
                items = await asyncio.wait_for(
                    self._analyze_one_session_friction(sid, summary, project),
                    timeout=_FRICTION_SESSION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "friction_analysis: session %d exceeded %.0fs, skipping "
                    "(retried next run)", sid, _FRICTION_SESSION_TIMEOUT,
                )
                timed_out += 1
                continue
            except Exception as e:
                logger.error("Friction analysis failed for session %d: %s", sid, e)
                failed += 1
                continue

            if items is None:
                continue  # no chunks — leave unmarked so it's retried later

            try:
                for item in items:
                    await self.sqlite.add_friction_entry(
                        session_id=sid,
                        source=item["source"],
                        category=item["category"],
                        description=item["description"],
                    )
                total_items += len(items)
                processed += 1
                # Even if NONE, mark session as analyzed by inserting a sentinel
                if not items:
                    await self.sqlite.add_friction_entry(
                        session_id=sid, source="nightly",
                        category="NONE", description="No friction detected",
                    )
            except Exception as e:
                logger.error("Friction write failed for session %d: %s", sid, e)
                failed += 1

        return {
            "timed_out": timed_out,
            "processed": processed,
            "friction_items": total_items,
            "failed": failed,
            "total": len(sessions),
            "stopped_early": stopped_early,
        }

    async def _analyze_one_session_friction(self, sid, summary, project):
        """Prepare a session's conversation and run friction analysis on it.

        Returns the list of friction items (empty = analyzed, no friction), or
        None when there's nothing to analyze (no chunks). Split out so the
        caller can bound the whole thing with a single ``wait_for``.
        """
        chunks, _ = await self.processor.prepare_conversation_for_reflection(
            sid, summary,
        )
        if not chunks:
            return None
        # Use first chunk only for friction (don't need full multi-chunk)
        conversation_text = chunks[0]
        if len(chunks) > 1:
            conversation_text += f"\n\n[... {len(chunks) - 1} more parts omitted]"
        return await self.processor.analyze_session_friction(
            session_id=sid,
            session_summary=summary,
            conversation_text=conversation_text,
            project=project,
        )

    async def _job_entity_extraction(self, on_status) -> dict:
        """Catch up on unextracted memories."""
        from blipshell.memory.entity_extractor import EntityExtractor

        # Mirror the live worker path (worker.py) so the nightly catch-up honors
        # the same entity-resolution config. Previously this read a non-existent
        # attribute (entity_resolution_enabled) and always fell back to False,
        # so catch-up extraction ran without dedup and created duplicate entities.
        er_cfg = self.config.memory.entity_resolution
        extractor = EntityExtractor(
            self.sqlite, self.router, self.vectors,
            batch_size=500,
            entity_resolution_enabled=er_cfg.enabled,
            entity_auto_merge_threshold=er_cfg.embedding_auto_merge_threshold,
            entity_llm_threshold=er_cfg.llm_arbitration_threshold,
            entity_max_candidates=er_cfg.max_candidates,
        )
        return await extractor.extract_batch(concurrency=1)

    async def _job_entity_cleanup(self, on_status) -> dict:
        """Clean up bad entities: pronouns, long names, invalid types, commentary."""
        from scripts.cleanup_entities import (
            DELETE_NAMES, VALID_TYPES, has_commentary,
            strip_formatting, clean_entity_type,
        )

        db = self.sqlite._db
        cursor = await db.execute("SELECT id, name, entity_type FROM entities")
        all_rows = await cursor.fetchall()

        deleted = 0
        type_fixed = 0
        renamed = 0
        deleted_entity_ids: list[int] = []
        for row in all_rows:
            eid, name, etype = row["id"], row["name"], row["entity_type"]
            name_lower = name.strip().lower()

            # Delete: commentary, pronouns, single-char, long names
            should_delete = (
                has_commentary(name)
                or name_lower in DELETE_NAMES
                or len(name.strip()) <= 1
                or len(name) > 60
                or name.replace(".", "").replace("-", "").strip().isdigit()
            )
            if should_delete:
                await db.execute("DELETE FROM entity_mentions WHERE entity_id = ?", (eid,))
                await db.execute("DELETE FROM entity_relationships WHERE subject_id = ? OR object_id = ?", (eid, eid))
                await db.execute("DELETE FROM entity_aliases WHERE canonical_entity_id = ?", (eid,))
                await db.execute("DELETE FROM entities WHERE id = ?", (eid,))
                deleted_entity_ids.append(eid)
                deleted += 1
                continue

            # Fix invalid entity types
            if etype not in VALID_TYPES:
                fixed = clean_entity_type(etype)
                await db.execute("UPDATE entities SET entity_type = ? WHERE id = ?", (fixed, eid))
                type_fixed += 1

            # Strip formatting prefixes
            cleaned = strip_formatting(name)
            if cleaned != name and cleaned and len(cleaned) > 1:
                await db.execute("UPDATE entities SET name = ? WHERE id = ?", (cleaned, eid))
                renamed += 1

        if deleted or type_fixed or renamed:
            await db.commit()

        # Batch-delete entity vectors AFTER async commit releases the lock.
        # Per-ID deletes during the loop caused lock contention between
        # aiosqlite and sync sqlite3 connections on the same DB file.
        if deleted_entity_ids and self.vectors:
            vec_deleted = 0
            for eid in deleted_entity_ids:
                try:
                    self.vectors.delete_entity(eid)
                    vec_deleted += 1
                except Exception:
                    pass  # orphan, will be cleaned eventually
            logger.info("Deleted %d/%d entity vectors", vec_deleted, len(deleted_entity_ids))

        # Clean orphaned relationships/mentions
        cursor = await db.execute(
            "DELETE FROM entity_relationships WHERE "
            "subject_id NOT IN (SELECT id FROM entities) OR "
            "object_id NOT IN (SELECT id FROM entities)"
        )
        orphan_rels = cursor.rowcount
        cursor = await db.execute(
            "DELETE FROM entity_mentions WHERE "
            "entity_id NOT IN (SELECT id FROM entities)"
        )
        orphan_mentions = cursor.rowcount
        cursor = await db.execute(
            "DELETE FROM entity_aliases WHERE "
            "canonical_entity_id NOT IN (SELECT id FROM entities)"
        )
        orphan_aliases = cursor.rowcount
        if orphan_rels or orphan_mentions or orphan_aliases:
            await db.commit()

        # This job writes entity rows through the raw connection rather than
        # the store's methods, so nothing invalidated the cached name list —
        # searches would keep matching deleted and pre-rename names for the
        # life of the process.
        self.sqlite._invalidate_entity_name_cache()

        return {
            "deleted": deleted,
            "type_fixed": type_fixed,
            "renamed": renamed,
            "orphan_rels": orphan_rels,
            "orphan_mentions": orphan_mentions,
            "orphan_aliases": orphan_aliases,
        }

    async def _job_update_user_model(self, on_status) -> dict:
        """Revise the user-model document from reflections since last run.

        Runs AFTER session_reflections in JOB_ORDER, so tonight's reflections
        are available as evidence tonight rather than tomorrow. Routed to the
        LOCAL model inside UserModel — this document is the distilled
        personal layer and deliberately never leaves the machine.
        """
        from blipshell.memory.user_model import UserModel

        return await UserModel(self.sqlite, self.router).revise_from_reflections()

    async def _job_export_mirror(self, on_status) -> dict:
        """Write the human-readable memory mirror (memory/mirror.py).

        Runs AFTER update_user_model in JOB_ORDER so tonight's revision is
        what gets mirrored. Read-only against the store, no LLM — never
        skipped for Ollama being down.
        """
        from blipshell.memory.mirror import export_mirror

        stats = await export_mirror(self.sqlite, self.config.database.path)
        on_status(f"  memory mirror -> {stats['dir']}")
        return stats

    async def _job_health_check(self, on_status) -> dict:
        """Run audit_db checks and return structured findings."""
        from scripts.audit_db import run_audit

        result = run_audit(
            db_path=self.config.database.path,
            skip_vectors=False,
            skip_endpoints=True,  # endpoints may not be available during nightly
        )
        warnings = [f for f in result.findings if f["severity"] == "WARNING"]
        errors = [f for f in result.findings if f["severity"] == "ERROR"]
        return {
            "total_findings": len(result.findings),
            "warnings": len(warnings),
            "errors": len(errors),
            "findings": result.findings,
        }

    async def _build_and_store_report(self, results: dict):
        """Build a structured nightly report and store in app_metadata.

        Every failure mode a job can report has to land in warnings/errors,
        because the startup notification prints "all clean" whenever both are
        empty. This used to collect only status == "error", so the worst
        realistic night — Ollama wedged, every LLM job skipped, two jobs
        timing out, backup failed — reported itself as clean
        (deep-dive 2026-08-04).
        """
        warnings = []
        errors = []
        statuses: dict[str, int] = {}
        # Run-level skips are grouped by reason: with Ollama down that's ~14
        # jobs at once, and 14 identical warnings buries everything else.
        skipped_by_reason: dict[str, list[str]] = {}

        for job_name, job_result in results.get("jobs", {}).items():
            status = job_result.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1

            if status == "error":
                errors.append(f"{job_name}: {job_result.get('error', 'unknown')}")
            elif status == "timeout":
                # A job that ran out of time did NOT finish its work.
                errors.append(
                    f"{job_name}: {job_result.get('error', f'timed out after {_JOB_TIMEOUT}s')}"
                )
            elif status == "skipped":
                # Only run-level skips carry this status (Ollama down, unknown
                # job). Config-gated no-ops finish with status "ok" and a
                # "skipped" key, and are intentional — not reported here.
                reason = job_result.get("error", "unknown reason")
                skipped_by_reason.setdefault(reason, []).append(job_name)

            # Collect health check findings
            if job_name == "health_check" and status == "ok":
                for finding in job_result.get("findings", []):
                    if finding["severity"] == "ERROR":
                        errors.append(
                            f"[health] {finding['check']}: {finding['message']}"
                        )
                    elif finding["severity"] == "WARNING":
                        warnings.append(
                            f"[health] {finding['check']}: {finding['message']}"
                        )

            # Flag jobs with failures
            if job_result.get("failed", 0) > 0:
                warnings.append(f"{job_name}: {job_result['failed']} failures")
            # Per-item timeouts inside a job that itself finished (friction
            # analysis reports these as `timed_out`, NOT `failed`, so the
            # check above misses them entirely).
            if job_result.get("timed_out", 0) > 0:
                warnings.append(
                    f"{job_name}: {job_result['timed_out']} items timed out"
                )
            # Partial progress — the job ran out of budget and will resume
            # next night. Not an error, but the work is not done.
            if job_result.get("stopped_early"):
                warnings.append(
                    f"{job_name}: stopped early (time budget) — work remains"
                )
            # A handler that swallowed its own exception and reported it as a
            # soft warning (e.g. _job_backup returning backup_path=None).
            if job_result.get("warning"):
                warnings.append(f"{job_name}: {job_result['warning']}")

        for reason, jobs in skipped_by_reason.items():
            warnings.append(
                f"{len(jobs)} job(s) skipped ({reason}): {', '.join(sorted(jobs))}"
            )

        report = {
            "timestamp": results.get("completed_at"),
            "elapsed_s": results.get("elapsed_s"),
            "warnings": warnings,
            "errors": errors,
            "job_statuses": statuses,
            "summary": {
                job: {k: v for k, v in data.items() if k != "findings"}
                for job, data in results.get("jobs", {}).items()
            },
        }

        await self.sqlite.set_metadata("nightly_report", json.dumps(report))

    async def _job_rebuild_digests(self, on_status) -> dict:
        """Rebuild project digests only for projects missing one."""
        from blipshell.memory.project_digest import ProjectDigestManager

        digest_mgr = ProjectDigestManager(self.sqlite, self.router, self.vectors)
        projects = await self.sqlite.list_projects()
        rebuilt = 0
        skipped = 0
        for project in projects:
            name = project.get("name")
            if not name:
                continue
            # Skip projects that already have a digest — they update incrementally
            # on session close. Only rebuild missing ones.
            existing = await digest_mgr.get_digest(name)
            if existing:
                skipped += 1
                continue
            try:
                digest = await digest_mgr.bootstrap_digest(name)
                if digest:
                    rebuilt += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error("Digest rebuild failed for '%s': %s", name, e)
                skipped += 1

        # Mirror every project's digest + lessons into its repo
        # (.blipshell/DIGEST.md). Runs for ALL projects, not just rebuilt
        # ones: lessons change without the digest changing, and the export
        # itself skips identical rewrites. Best-effort per project.
        from blipshell.memory.digest_export import export_digest
        exported = 0
        for project in projects:
            if project.get("name"):
                if await export_digest(self.sqlite, project["name"]):
                    exported += 1
        return {"rebuilt": rebuilt, "skipped": skipped,
                "exported": exported, "total": len(projects)}

    async def close(self):
        """Clean up resources."""
        try:
            await self.sqlite.close()
        except Exception as e:
            logger.debug("NightlyRunner close error: %s", e)
