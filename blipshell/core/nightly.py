"""Nightly job runner for BlipShell.

Orchestrates overnight maintenance: backup, cleanup, tagging, pruning,
consolidation, and tag discovery. Can run interactively via /nightly
or headless via `blipshell nightly`.
"""

from __future__ import annotations

import asyncio
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
    "health_check",
]

# Jobs that require Ollama (LLM calls or embedding). Skipped if Ollama is down.
_OLLAMA_JOBS = {
    "backfill_vectors", "backfill_summaries", "resummarize",
    "backfill_lessons", "score_lessons", "clean_junk_lessons",
    "session_reflections", "friction_analysis",
    "entity_extraction", "batch_tag",
    "merge_entities",
    "rebuild_digests",
}

# Max time per job (seconds). Prevents a single hung job from burning hours.
_JOB_TIMEOUT = 300  # 5 minutes per job

# Sentinel distinguishing "caller didn't specify a budget, use the nightly
# default" from an explicit None ("no budget, full pass"). Used by
# _job_merge_entities so the scheduled run gets budgeted while the standalone
# script can opt out.
_DEFAULT_BUDGET = object()


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
            try:
                job_result = await asyncio.wait_for(
                    self.run_job(job_name, on_status=on_status),
                    timeout=_JOB_TIMEOUT,
                )
                elapsed = time.monotonic() - t0
                job_result["status"] = "ok"
                job_result["elapsed_s"] = round(elapsed, 1)
                results[job_name] = job_result
                _status(f"  {job_name} completed in {elapsed:.1f}s")
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - t0
                logger.error("Nightly job %s timed out after %ds", job_name, _JOB_TIMEOUT)
                results[job_name] = {
                    "status": "timeout",
                    "error": f"Timed out after {_JOB_TIMEOUT}s",
                    "elapsed_s": round(elapsed, 1),
                }
                _status(f"  {job_name} TIMED OUT after {_JOB_TIMEOUT}s")
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
            "backfill_vectors": self._job_backfill_vectors,
            "clean_empty_sessions": self._job_clean_empty_sessions,
            "cleanup": self._job_cleanup,
            "backfill_summaries": self._job_backfill_summaries,
            "resummarize": self._job_resummarize,
            "backfill_lessons": self._job_backfill_lessons,
            "score_lessons": self._job_score_lessons,
            "clean_junk_lessons": self._job_clean_junk_lessons,
            "session_reflections": self._job_session_reflections,
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
        """Backfill any missing vector embeddings."""
        total = {"succeeded": 0, "failed": 0}
        for collection in ("memories", "core_memories", "lessons", "entities"):
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
        return await consolidator.consolidate_batch()

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
        from blipshell.memory.manager import estimate_tokens

        sessions = await self.sqlite.get_sessions_missing_lessons(limit=500)
        if not sessions:
            return {"processed": 0, "total": 0}

        processed = 0
        failed = 0
        cloud_routed = 0
        for session in sessions:
            sid = session["id"]
            project = session.get("project")
            try:
                messages = await self.sqlite.get_session_messages_for_lesson(sid)
                conversation_lines = [f"{m['role']}: {m['content']}" for m in messages]
                conversation_text = "\n".join(conversation_lines)
                tokens = estimate_tokens(conversation_text)
                # Route large sessions to bigger-context endpoint
                min_ctx = tokens + 4096 if tokens > 28000 else None
                if min_ctx:
                    cloud_routed += 1
                await self.processor.process_lesson(
                    conversation_text, sid, project=project,
                    min_context_tokens=min_ctx,
                )
                processed += 1
            except Exception as e:
                logger.error("Lesson backfill failed for session %d: %s", sid, e)
                failed += 1

        return {"processed": processed, "failed": failed, "cloud_routed": cloud_routed, "total": len(sessions)}

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

        return {"deleted_junk": deleted_junk, "deleted_dupes": deleted_dupes}

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

    async def _job_friction_analysis(self, on_status) -> dict:
        """Analyze recent sessions for system-level friction.

        Each session needs a REASONING LLM call, so a full 200-session batch
        far exceeds the per-job timeout. We process under a time budget
        slightly below ``_JOB_TIMEOUT`` and exit cleanly with partial
        progress — completed sessions are marked in ``friction_log`` (real
        items or a NONE sentinel), so the next nightly run resumes where this
        one stopped instead of being hard-killed mid-loop by ``wait_for``.
        """
        sessions = await self.sqlite.get_sessions_missing_friction_analysis(limit=200)
        if not sessions:
            return {"processed": 0, "friction_items": 0, "total": 0}

        budget = _JOB_TIMEOUT - 30
        start = time.monotonic()
        processed = 0
        total_items = 0
        failed = 0
        stopped_early = False
        for session in sessions:
            if time.monotonic() - start > budget:
                stopped_early = True
                logger.info(
                    "friction_analysis hit time budget (%ds), stopping with "
                    "partial progress (%d/%d sessions)",
                    budget, processed, len(sessions),
                )
                break
            sid = session["id"]
            summary = session["summary"]
            project = session.get("project")
            try:
                chunks, _ = await self.processor.prepare_conversation_for_reflection(
                    sid, summary,
                )
                if not chunks:
                    continue
                # Use first chunk only for friction (don't need full multi-chunk)
                conversation_text = chunks[0]
                if len(chunks) > 1:
                    conversation_text += f"\n\n[... {len(chunks) - 1} more parts omitted]"

                items = await self.processor.analyze_session_friction(
                    session_id=sid,
                    session_summary=summary,
                    conversation_text=conversation_text,
                    project=project,
                )
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
                logger.error("Friction analysis failed for session %d: %s", sid, e)
                failed += 1

        return {
            "processed": processed,
            "friction_items": total_items,
            "failed": failed,
            "total": len(sessions),
            "stopped_early": stopped_early,
        }

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

        return {
            "deleted": deleted,
            "type_fixed": type_fixed,
            "renamed": renamed,
            "orphan_rels": orphan_rels,
            "orphan_mentions": orphan_mentions,
            "orphan_aliases": orphan_aliases,
        }

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
        """Build a structured nightly report and store in app_metadata."""
        warnings = []
        errors = []

        for job_name, job_result in results.get("jobs", {}).items():
            if job_result.get("status") == "error":
                errors.append(f"{job_name}: {job_result.get('error', 'unknown')}")
            # Collect health check findings
            if job_name == "health_check" and job_result.get("status") == "ok":
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

        report = {
            "timestamp": results.get("completed_at"),
            "elapsed_s": results.get("elapsed_s"),
            "warnings": warnings,
            "errors": errors,
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
        return {"rebuilt": rebuilt, "skipped": skipped, "total": len(projects)}

    async def close(self):
        """Clean up resources."""
        try:
            await self.sqlite.close()
        except Exception as e:
            logger.debug("NightlyRunner close error: %s", e)
