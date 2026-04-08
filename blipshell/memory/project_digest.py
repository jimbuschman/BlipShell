"""Project Digest — auto-maintained summary of project state across sessions.

Generates and updates a structured digest stored in projects.metadata_json["digest"].
Bootstrap uses semantic search (ChromaDB) + keyword search to find relevant memories,
then processes them in batches. Incremental updates fold a single new session summary.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from blipshell.llm.prompts import (
    generate_initial_digest,
    update_digest_incremental,
    update_digest_with_sessions,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore

if TYPE_CHECKING:
    from blipshell.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

BATCH_SIZE = 5  # memories per batch during bootstrap
MAX_MEMORIES = 40  # cap total memories to keep LLM cost reasonable


class ProjectDigestManager:
    """Manages project digest lifecycle: bootstrap, incremental update, retrieval."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        router: LLMRouter,
        vectors: VectorStore | None = None,
    ):
        self.sqlite = sqlite
        self.router = router
        self.vectors = vectors

    async def get_digest(self, project_name: str) -> str | None:
        """Load existing digest from project metadata."""
        project = await self.sqlite.get_project(project_name)
        if not project:
            return None
        meta = json.loads(project.get("metadata_json") or "{}")
        return meta.get("digest")

    async def bootstrap_digest(self, project_name: str) -> str:
        """Generate digest from scratch using memory search.

        Data source priority (merged, deduplicated):
        1. ChromaDB semantic search for the project name (finds related memories)
        2. SQLite keyword search on memory summaries (catches exact matches)
        3. Session summaries tagged or mentioning the project (bonus)

        Processes in batches, folding each into a running digest.
        Returns the final digest (also saved to DB).
        """
        memories = await self._gather_memories(project_name)

        if not memories:
            logger.info("No data found for project '%s'", project_name)
            return ""

        logger.info(
            "Bootstrap digest for '%s': gathered %d memories",
            project_name, len(memories),
        )
        return await self._bootstrap_from_memories(project_name, memories)

    async def _gather_memories(self, project_name: str) -> list[dict]:
        """Gather memory summaries from multiple sources, deduplicated."""
        import re

        seen_summaries: set[str] = set()
        results: list[dict] = []

        def _add(summary: str, importance: float = 0.5):
            key = summary.strip().lower()[:100]
            if key not in seen_summaries:
                seen_summaries.add(key)
                results.append({"summary": summary, "importance": importance})

        # Source 1: ChromaDB semantic search (most powerful — finds related content)
        if self.vectors:
            # Split project name for broader search
            parts = re.split(r'[-_\s]+', project_name)
            queries = [project_name]
            # Also search individual meaningful words
            for p in parts:
                if len(p) >= 3 and p.lower() not in queries:
                    queries.append(p)

            for query in queries:
                try:
                    chroma_results = self.vectors.search_memories(
                        query=query, n_results=30,
                    )
                    for cr in chroma_results:
                        doc = cr.get("document", "")
                        sim = cr.get("similarity", 0.0)
                        if doc and sim > 0.3:
                            _add(doc, sim)
                except Exception as e:
                    logger.warning("ChromaDB search failed for '%s': %s", query, e)

        # Source 2: SQLite keyword search on memory summaries
        keyword_results = await self.sqlite.search_memories_by_keyword(
            project_name, limit=50,
        )
        for m in keyword_results:
            if m.get("summary"):
                _add(m["summary"], m.get("importance", 0.5))

        # Source 3: Session summaries (bonus — use when available)
        sessions = await self.sqlite.list_sessions_chronological(project_name)
        if not sessions:
            sessions = await self.sqlite.search_sessions_by_keyword(project_name)
        for s in sessions or []:
            if s.summary:
                _add(f"[Session: {s.title or 'Untitled'}] {s.summary}", 0.7)

        # Sort by importance descending, cap total
        results.sort(key=lambda r: r["importance"], reverse=True)
        return results[:MAX_MEMORIES]

    async def _bootstrap_from_memories(
        self, project_name: str, memories: list[dict],
    ) -> str:
        """Build digest from memory summaries in batches."""
        digest = ""

        for i in range(0, len(memories), BATCH_SIZE):
            batch = memories[i:i + BATCH_SIZE]
            batch_text = "\n".join(
                f"- {m['summary']}" for m in batch if m.get("summary")
            )
            if not batch_text:
                continue

            if not digest:
                system, user = generate_initial_digest(project_name, batch_text)
                digest = await self.router.generate(
                    TaskType.REASONING, user, system=system,
                )
                logger.info(
                    "Bootstrap digest for '%s': initial batch (%d memories)",
                    project_name, len(batch),
                )
            else:
                system, user = update_digest_with_sessions(digest, batch_text)
                digest = await self.router.generate(
                    TaskType.REASONING, user, system=system,
                )
                logger.info(
                    "Bootstrap digest for '%s': folded batch %d (%d memories)",
                    project_name, i // BATCH_SIZE + 1, len(batch),
                )

        digest = await self._fold_lessons(project_name, digest)

        if digest:
            await self._save_digest(project_name, digest, [])
            logger.info(
                "Bootstrap digest for '%s' complete (%d chars, %d memories)",
                project_name, len(digest), len(memories),
            )
        return digest

    async def _fold_lessons(self, project_name: str, digest: str) -> str:
        """Fold project-scoped lessons into digest if any exist."""
        lessons = await self.sqlite.get_lessons_by_project(project_name)
        if lessons and digest:
            lesson_text = "\n".join(f"- {l.content}" for l in lessons[:20])
            system, user = update_digest_with_sessions(
                digest,
                f"[Behavioral lessons learned across sessions]\n{lesson_text}",
            )
            digest = await self.router.generate(
                TaskType.REASONING, user, system=system,
            )
            logger.info(
                "Bootstrap digest for '%s': folded %d lessons",
                project_name, len(lessons),
            )
        return digest

    async def update_digest(
        self,
        project_name: str,
        session_summary: str,
        session_title: str,
        session_id: int,
    ) -> str:
        """Incrementally update digest with a single new session.

        If no digest exists yet, triggers a full bootstrap.
        Returns the updated digest.
        """
        current = await self.get_digest(project_name)
        if not current:
            return await self.bootstrap_digest(project_name)

        # Check if this session was already folded in
        project = await self.sqlite.get_project(project_name)
        meta = json.loads(project.get("metadata_json") or "{}")
        existing_ids = meta.get("digest_session_ids", [])
        if session_id in existing_ids:
            logger.debug("Session %d already in digest for '%s'", session_id, project_name)
            return current

        system, user = update_digest_incremental(current, session_summary, session_title)
        updated = await self.router.generate(
            TaskType.REASONING, user, system=system,
        )

        existing_ids.append(session_id)
        await self._save_digest(project_name, updated, existing_ids)
        logger.info(
            "Updated digest for '%s' with session %d (%d chars)",
            project_name, session_id, len(updated),
        )
        return updated

    async def _save_digest(
        self, project_name: str, digest: str, session_ids: list[int],
    ):
        """Persist digest to project metadata_json."""
        project = await self.sqlite.get_project(project_name)
        if not project:
            return
        meta = json.loads(project.get("metadata_json") or "{}")
        meta["digest"] = digest
        meta["digest_updated_at"] = datetime.now(timezone.utc).isoformat()
        meta["digest_session_ids"] = session_ids
        await self.sqlite.update_project(
            project_name, metadata_json=json.dumps(meta),
        )
