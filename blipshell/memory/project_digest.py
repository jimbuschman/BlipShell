"""Project Digest — auto-maintained summary of project state across sessions.

Generates and updates a structured digest stored in projects.metadata_json["digest"].
Bootstrap processes all sessions chronologically in batches; incremental updates
fold a single new session into the existing digest.
"""

import json
import logging
from datetime import datetime, timezone

from blipshell.llm.prompts import (
    generate_initial_digest,
    update_digest_incremental,
    update_digest_with_sessions,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

BATCH_SIZE = 5  # sessions per batch during bootstrap


class ProjectDigestManager:
    """Manages project digest lifecycle: bootstrap, incremental update, retrieval."""

    def __init__(self, sqlite: SQLiteStore, router: LLMRouter):
        self.sqlite = sqlite
        self.router = router

    async def get_digest(self, project_name: str) -> str | None:
        """Load existing digest from project metadata."""
        project = await self.sqlite.get_project(project_name)
        if not project:
            return None
        meta = json.loads(project.get("metadata_json") or "{}")
        return meta.get("digest")

    async def bootstrap_digest(self, project_name: str) -> str:
        """Generate digest from scratch using session or memory summaries.

        Data source priority:
        1. Sessions tagged with the project name
        2. Sessions mentioning the project name in title/summary (keyword search)
        3. Individual memory summaries mentioning the project name

        Processes in batches, folding each into a running digest.
        Returns the final digest (also saved to DB).
        """
        # --- Try session-based sources first ---
        sessions = await self.sqlite.list_sessions_chronological(project_name)
        if not sessions:
            sessions = await self.sqlite.search_sessions_by_keyword(project_name)
            if sessions:
                logger.info(
                    "No tagged sessions for '%s', found %d via keyword search",
                    project_name, len(sessions),
                )

        if sessions:
            return await self._bootstrap_from_sessions(project_name, sessions)

        # --- Fallback: use individual memory summaries ---
        memories = await self.sqlite.search_memories_by_keyword(project_name)
        if memories:
            logger.info(
                "No sessions found for '%s', using %d memory summaries instead",
                project_name, len(memories),
            )
            return await self._bootstrap_from_memories(project_name, memories)

        logger.info("No data found for project '%s' (no sessions or memories)", project_name)
        return ""

    async def _bootstrap_from_sessions(
        self, project_name: str, sessions: list,
    ) -> str:
        """Build digest from session summaries."""
        session_ids = []
        digest = ""

        for i in range(0, len(sessions), BATCH_SIZE):
            batch = sessions[i:i + BATCH_SIZE]
            batch_summaries = "\n".join(
                f"- [{s.title or 'Untitled'}] {s.summary}"
                for s in batch if s.summary
            )
            if not batch_summaries:
                continue

            session_ids.extend(s.id for s in batch)

            if not digest:
                system, user = generate_initial_digest(project_name, batch_summaries)
                digest = await self.router.generate(
                    TaskType.REASONING, user, system=system,
                )
                logger.info(
                    "Bootstrap digest for '%s': initial batch (%d sessions)",
                    project_name, len(batch),
                )
            else:
                system, user = update_digest_with_sessions(digest, batch_summaries)
                digest = await self.router.generate(
                    TaskType.REASONING, user, system=system,
                )
                logger.info(
                    "Bootstrap digest for '%s': folded batch %d (%d sessions)",
                    project_name, i // BATCH_SIZE + 1, len(batch),
                )

        digest = await self._fold_lessons(project_name, digest)

        if digest:
            await self._save_digest(project_name, digest, session_ids)
            logger.info(
                "Bootstrap digest for '%s' complete (%d chars, %d sessions)",
                project_name, len(digest), len(session_ids),
            )
        return digest

    async def _bootstrap_from_memories(
        self, project_name: str, memories: list[dict],
    ) -> str:
        """Build digest from individual memory summaries (fallback path)."""
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
                    "Bootstrap digest for '%s': initial memory batch (%d memories)",
                    project_name, len(batch),
                )
            else:
                system, user = update_digest_with_sessions(digest, batch_text)
                digest = await self.router.generate(
                    TaskType.REASONING, user, system=system,
                )
                logger.info(
                    "Bootstrap digest for '%s': folded memory batch %d (%d memories)",
                    project_name, i // BATCH_SIZE + 1, len(batch),
                )

        digest = await self._fold_lessons(project_name, digest)

        if digest:
            await self._save_digest(project_name, digest, [])
            logger.info(
                "Bootstrap digest for '%s' complete from memories (%d chars, %d memories)",
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
