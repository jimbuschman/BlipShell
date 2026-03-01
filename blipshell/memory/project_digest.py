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
        """Generate digest from scratch using all session summaries.

        Processes in batches of 5 chronologically, folding each batch
        into a running digest to avoid context overflow.
        Returns the final digest (also saved to DB).

        Falls back to keyword search if no sessions are tagged with the
        project name (common for imported sessions that predate project creation).
        """
        sessions = await self.sqlite.list_sessions_chronological(project_name)
        if not sessions:
            # Fallback: search for sessions mentioning the project by name
            sessions = await self.sqlite.search_sessions_by_keyword(project_name)
            if sessions:
                logger.info(
                    "No tagged sessions for '%s', found %d via keyword search",
                    project_name, len(sessions),
                )
        if not sessions:
            logger.info("No sessions with summaries for project '%s'", project_name)
            return ""

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
                # First batch — generate initial digest
                system, user = generate_initial_digest(project_name, batch_summaries)
                digest = await self.router.generate(
                    TaskType.REASONING, user, system=system,
                )
                logger.info(
                    "Bootstrap digest for '%s': initial batch (%d sessions)",
                    project_name, len(batch),
                )
            else:
                # Subsequent batches — fold into running digest
                system, user = update_digest_with_sessions(digest, batch_summaries)
                digest = await self.router.generate(
                    TaskType.REASONING, user, system=system,
                )
                logger.info(
                    "Bootstrap digest for '%s': folded batch %d (%d sessions)",
                    project_name, i // BATCH_SIZE + 1, len(batch),
                )

        # Fold in project-scoped lessons if any
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

        if digest:
            await self._save_digest(project_name, digest, session_ids)
            logger.info(
                "Bootstrap digest for '%s' complete (%d chars, %d sessions)",
                project_name, len(digest), len(session_ids),
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
