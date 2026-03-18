"""Session lifecycle mixin for Agent.

Extracts session start, memory loading, pruning, and startup tasks.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass  # All types accessed via self

from blipshell.memory.consolidation import MemoryConsolidator
from blipshell.memory.manager import PoolItem
from blipshell.memory.tag_discovery import TagDiscovery
from blipshell.memory.tagger import register_topic_patterns

logger = logging.getLogger(__name__)


class SessionMixin:
    """Session lifecycle methods mixed into Agent."""

    async def start_session(
        self,
        project: Optional[str] = None,
        resume_session_id: Optional[int] = None,
    ) -> int:
        """Start or resume a session."""
        await self.initialize()
        self._file_changes = []
        self._files_read = set()

        session_id = await self.session_manager.start_session(
            project=project,
            resume_session_id=resume_session_id,
        )

        # Register memory tools now that we have session_id
        self._register_memory_tools()

        # Register task/workflow tools
        self._register_task_tools()

        # Retry any failed ChromaDB operations from previous sessions
        await self._retry_chroma_queue()

        # Load core memories into Core pool
        await self._load_core_memories()

        # Load lessons into Core pool
        await self._load_lessons()

        # Summarize sessions that never closed properly (crash, Ollama died, etc.)
        await self._summarize_orphaned_sessions()

        # Load recent session context into RecentHistory
        await self._load_recent_sessions()

        # Check for nightly report warnings/errors
        self._nightly_notification = await self._check_nightly_report()

        # Load pending follow-ups for proactive session opener
        self._pending_follow_ups = await self._load_follow_ups()

        # Load session notes (persistent state surviving compaction)
        await self._load_session_notes()

        return session_id

    async def _retry_chroma_queue(self):
        """Retry failed ChromaDB operations queued from previous sessions."""
        try:
            from blipshell.memory.chroma_retry import process_retry_queue
            stats = await process_retry_queue(self.sqlite, self.chroma, limit=100)
            if stats["processed"] > 0:
                logger.info(
                    "ChromaDB retry queue: %d processed, %d succeeded, %d still failing",
                    stats["processed"], stats["succeeded"], stats["failed"],
                )
        except Exception as e:
            logger.debug("ChromaDB retry queue processing failed: %s", e)

    async def _load_core_memories(self):
        """Load active core memories into the Core pool."""
        core_memories = await self.sqlite.get_active_core_memories()
        for cm in core_memories:
            self.memory_manager.add_memory("Core", PoolItem(
                text=cm.content,
                session_role="system",
                priority_score=cm.importance + 1.0,  # boost core memories
            ))
        logger.info("Loaded %d core memories", len(core_memories))

    async def _load_lessons(self):
        """Load lessons into the Core pool."""
        lessons = await self.sqlite.get_all_lessons()
        for lesson in lessons:
            self.memory_manager.add_memory("Core", PoolItem(
                text=lesson.content,
                session_role="system2",  # marks as lesson for pool labeling
                priority_score=lesson.importance,
            ))
        logger.info("Loaded %d lessons", len(lessons))

    async def _auto_prune_memories(self):
        """Prune old low-value memories on startup (disabled when auto_prune_days=0)."""
        cfg = self.config.memory
        if cfg.auto_prune_days <= 0:
            return
        try:
            # Get IDs before archiving (for ChromaDB cleanup)
            ids_to_archive = await self.sqlite.get_archived_memory_ids(
                days_old=cfg.auto_prune_days,
                max_importance=cfg.prune_max_importance,
                max_rank=cfg.prune_max_rank,
            )
            # Archive in SQLite
            count = await self.sqlite.archive_old_memories(
                days_old=cfg.auto_prune_days,
                max_importance=cfg.prune_max_importance,
                max_rank=cfg.prune_max_rank,
            )
            # Remove from ChromaDB
            for mid in ids_to_archive:
                try:
                    self.chroma.delete_memory(mid)
                except Exception as e:
                    logger.warning("Failed to delete memory %d from ChromaDB (queued): %s", mid, e)
                    from blipshell.memory.chroma_retry import queue_failed_op, OP_DELETE, COLLECTION_MEMORIES
                    await queue_failed_op(
                        self.sqlite, OP_DELETE, COLLECTION_MEMORIES,
                        mid, error=str(e),
                    )
            if count:
                logger.info("Auto-pruned %d memories", count)
        except Exception as e:
            logger.error("Auto-prune failed: %s", e)

    async def _auto_consolidate_memories(self):
        """Merge near-duplicate memories on startup (disabled when batch_size=0)."""
        if self.config.memory.consolidation_batch_size <= 0:
            return
        try:
            consolidator = MemoryConsolidator(
                self.sqlite, self.chroma, self.config.memory,
            )
            stats = await consolidator.consolidate_batch()
            if stats["merged"] > 0:
                logger.info(
                    "Consolidated %d duplicate memories (checked %d)",
                    stats["merged"], stats["checked"],
                )
        except Exception as e:
            logger.error("Memory consolidation failed: %s", e)

    async def _load_discovered_tags(self):
        """Load previously discovered tag patterns into the tagger."""
        try:
            discovered = await self.sqlite.get_discovered_tag_patterns()
            if discovered:
                register_topic_patterns(discovered)
                total = sum(len(v) for v in discovered.values())
                logger.info("Loaded %d discovered tag patterns", total)
        except Exception as e:
            logger.error("Failed to load discovered tags: %s", e)

    async def _auto_tag_discovery(self):
        """Run LLM-powered tag discovery if enough time has elapsed."""
        try:
            cfg = self.config.memory
            discovery = TagDiscovery(
                self.sqlite, self.router,
                interval_days=cfg.tag_discovery_interval_days,
                sample_size=cfg.tag_discovery_sample_size,
            )
            stats = await discovery.maybe_run()
            if stats["discovered"] > 0:
                # Reload newly discovered patterns into tagger
                new_patterns = await self.sqlite.get_discovered_tag_patterns()
                register_topic_patterns(new_patterns)
                logger.info("Discovered %d new tag patterns", stats["discovered"])
        except Exception as e:
            logger.error("Tag discovery failed: %s", e)

    async def _enqueue_startup_background_tasks(self):
        """Enqueue entity extraction and unprocessed messages to the background worker.

        Replaces the old blocking _auto_extract_entities() and _sweep_unprocessed_messages()
        so startup completes in seconds instead of minutes.
        """
        if not self._memory_worker or not self._memory_worker.is_alive:
            logger.warning("Memory worker not running, skipping background startup tasks")
            return

        from blipshell.memory.worker import WorkItem, WorkType

        # Entity extraction — worker processes in background
        self._memory_worker.enqueue(
            WorkItem(work_type=WorkType.EXTRACT_ENTITIES, text="startup")
        )

        # Unprocessed memory sweep — enqueue each as PROCESS_MESSAGE
        try:
            unprocessed = await self.sqlite.get_unprocessed_memories(limit=50)
            if unprocessed:
                logger.info(
                    "Enqueueing %d unprocessed memories for background processing",
                    len(unprocessed),
                )
                for msg in unprocessed:
                    self._memory_worker.enqueue(WorkItem(
                        work_type=WorkType.PROCESS_MESSAGE,
                        text=msg["content"],
                        role=msg["role"],
                        session_id=msg["session_id"],
                        memory_id=msg["id"],
                    ))
        except Exception as e:
            logger.warning("Failed to enqueue unprocessed memories: %s", e)

    async def _backfill_entity_embeddings(self):
        """One-time backfill: embed all existing entities into ChromaDB for resolution.

        Tracks completion via app_metadata so it only runs once.
        """
        try:
            marker = await self.sqlite.get_metadata("entity_embeddings_backfilled")
            if marker:
                return  # already done

            # Load all entities from SQLite
            cursor = await self.sqlite._db.execute(
                "SELECT id, name, entity_type FROM entities"
            )
            rows = await cursor.fetchall()
            if not rows:
                await self.sqlite.set_metadata("entity_embeddings_backfilled", "1")
                return

            # Batch upsert into ChromaDB (chunks of 500 to avoid OOM)
            batch_size = 500
            total = len(rows)
            for i in range(0, total, batch_size):
                chunk = rows[i:i + batch_size]
                ids = [r["id"] for r in chunk]
                names = [r["name"] for r in chunk]
                types = [r["entity_type"] for r in chunk]
                try:
                    self.chroma.upsert_entities_batch(ids, names, types)
                except Exception as e:
                    logger.warning("Entity backfill batch failed (queueing individually): %s", e)
                    from blipshell.memory.chroma_retry import queue_failed_op, OP_UPSERT, COLLECTION_ENTITIES
                    for eid, ename, etype in zip(ids, names, types):
                        await queue_failed_op(
                            self.sqlite, OP_UPSERT, COLLECTION_ENTITIES, eid,
                            document=ename, metadata={"entity_type": etype}, error=str(e),
                        )

            await self.sqlite.set_metadata("entity_embeddings_backfilled", "1")
            logger.info("Backfilled %d entity embeddings into ChromaDB", total)
        except Exception as e:
            logger.error("Entity embedding backfill failed: %s", e)

    async def _load_recent_sessions(self):
        """Load context from recent sessions into RecentHistory pool.

        Two tiers:
        1. Last substantive session (>= 5 messages): load top memories with timestamps
        2. Other recent sessions: load summaries only
        """
        sessions = await self.sqlite.list_sessions(limit=10)
        current_id = self.session_manager.session_id
        now = datetime.now(timezone.utc)

        loaded_substantive = False
        for s in sessions:
            if s.id == current_id:
                continue

            # Tier 1: Load top memories from last substantive session
            if not loaded_substantive and s.message_count >= 5:
                memories = await self.sqlite.get_memories_by_session(s.id)
                good_memories = [
                    m for m in memories
                    if m.summary and not m.is_archived and m.importance >= 0.3
                ]
                good_memories.sort(key=lambda m: m.importance, reverse=True)

                for m in good_memories[:20]:
                    ts = m.timestamp
                    if ts and ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    label = ""
                    if ts:
                        delta = now - ts
                        hours = delta.total_seconds() / 3600
                        if hours < 1:
                            label = f"[{int(delta.total_seconds() / 60)}m ago]"
                        elif hours < 24:
                            label = f"[{int(hours)}h ago]"
                        elif delta.days < 7:
                            label = f"[{delta.days}d ago]"
                        else:
                            label = f"[{ts.strftime('%Y-%m-%d')}]"
                    self.memory_manager.add_memory("RecentHistory", PoolItem(
                        text=f"{label} {m.summary}",
                        session_role="system",
                        priority_score=2.0 + m.importance,
                        session_id=s.id,
                    ))

                if s.summary:
                    self.memory_manager.add_memory("RecentHistory", PoolItem(
                        text=f"[Previous session summary] {s.summary}",
                        session_role="system",
                        priority_score=3.0,
                        session_id=s.id,
                    ))
                loaded_substantive = True
                logger.info(
                    "Loaded %d memories from previous session %d into RecentHistory",
                    min(len(good_memories), 20), s.id,
                )
                continue

            # Tier 2: Other recent sessions get summary only
            text = s.summary
            if not text:
                memories = await self.sqlite.get_memories_by_session(s.id)
                if not memories:
                    continue
                text = "; ".join(
                    m.summary or m.content for m in memories[:10]
                )
            self.memory_manager.add_memory("RecentHistory", PoolItem(
                text=text,
                session_role="system",
                priority_score=2.0,
                session_id=s.id,
            ))

    async def _summarize_orphaned_sessions(self):
        """Generate summaries for recent sessions that never closed properly.

        When a session doesn't get end_session() (crash, killed, Ollama died),
        it has no summary and no title. This detects and fixes those on startup.
        """
        from blipshell.llm.prompts import summarize_session_conversation, generate_session_title
        from blipshell.llm.router import TaskType

        sessions = await self.sqlite.list_sessions(limit=10)
        current_id = self.session_manager.session_id
        for s in sessions:
            if s.id == current_id:
                continue
            if s.summary:
                continue
            if s.message_count < 3:
                continue

            memories = await self.sqlite.get_memories_by_session(s.id)
            summaries = [m.summary for m in memories if m.summary]
            if not summaries:
                continue

            try:
                text = "\n".join(summaries[:20])
                summary = await self.router.generate(
                    TaskType.SUMMARIZATION,
                    summarize_session_conversation(text),
                )
                title = await self.router.generate(
                    TaskType.SUMMARIZATION,
                    generate_session_title(summary),
                )
                title = title.strip().strip('"').strip("'")
                await self.sqlite.update_session(
                    s.id, summary=summary, title=title,
                )
                logger.info(
                    "Generated summary for orphaned session %d: %s",
                    s.id, title[:60],
                )
            except Exception as e:
                logger.warning(
                    "Failed to summarize orphaned session %d: %s", s.id, e,
                )

    async def _load_follow_ups(self) -> str:
        """Load pending follow-ups and format for injection into first turn."""
        try:
            project = self.active_project["name"] if self.active_project else None
            items = await self.sqlite.get_pending_follow_ups(project=project, limit=10)
            if not items:
                return ""

            lines = ["[OPEN FOLLOW-UPS from previous sessions]"]
            for item in items:
                line = f"- #{item['id']}: {item['content']}"
                if item.get("due_hint"):
                    line += f" (due: {item['due_hint']})"
                lines.append(line)
            lines.append(
                "Mention relevant items naturally. "
                "Use resolve_followup when an item is addressed."
            )
            logger.info("Loaded %d pending follow-ups", len(items))
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Failed to load follow-ups: %s", e)
            return ""

    async def _load_session_notes(self):
        """Load session notes from database into agent state for system prompt injection."""
        try:
            if not self.config.notes.enabled:
                return
            session_id = self.session_manager.session_id if self.session_manager else None
            if not session_id:
                return
            notes = await self.sqlite.get_session_notes(session_id)
            self._session_notes = notes
            if notes:
                logger.info("Loaded %d session notes", len(notes))
        except Exception as e:
            logger.debug("Failed to load session notes: %s", e)

    async def _check_nightly_report(self) -> str | None:
        """Check last nightly run status. Always reports when it ran, highlights issues."""
        try:
            import json
            from datetime import datetime as _dt

            raw = await self.sqlite.get_metadata("nightly_last_run")
            if not raw:
                return None
            data = json.loads(raw)
            completed_at = data.get("completed_at")
            if not completed_at:
                return None

            last_run = _dt.fromtimestamp(completed_at)
            elapsed = data.get("elapsed_s", 0)
            time_str = last_run.strftime("%Y-%m-%d %I:%M %p")

            # Check for warnings/errors in the report
            report_raw = await self.sqlite.get_metadata("nightly_report")
            warnings = []
            errors = []
            if report_raw:
                report = json.loads(report_raw)
                warnings = report.get("warnings", [])
                errors = report.get("errors", [])

            if errors or warnings:
                parts = []
                if errors:
                    parts.append(f"{len(errors)} error(s)")
                if warnings:
                    parts.append(f"{len(warnings)} warning(s)")
                return f"Nightly ran {time_str} ({elapsed:.0f}s) — {', '.join(parts)}. Use /nightly report for details."
            else:
                return f"Nightly ran {time_str} ({elapsed:.0f}s) — all clean."
        except Exception:
            return None
