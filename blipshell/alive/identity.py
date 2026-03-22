"""Identity Synthesizer — progressive batching of thoughts into self-authored identity.

Follows the ProjectDigestManager pattern: gather thoughts in batches,
fold each batch into a running identity draft, stabilize against previous version.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from blipshell.alive.prompts import (
    fold_thoughts_into_identity,
    stabilize_identity,
    synthesize_identity,
)
from blipshell.llm.router import TaskType
from blipshell.models.alive import IdentityVersion

if TYPE_CHECKING:
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.llm.router import LLMRouter

logger = logging.getLogger(__name__)

BATCH_SIZE = 10  # thoughts per synthesis batch


class IdentitySynthesizer:
    """Synthesizes the AI's self-authored identity from accumulated thoughts."""

    def __init__(self, sqlite: SQLiteStore, router: LLMRouter):
        self.sqlite = sqlite
        self.router = router

    async def synthesize(self, trigger: str = "nightly") -> IdentityVersion | None:
        """Generate a new identity version from active thoughts.

        Uses progressive batching:
        1. Gather high-confidence active thoughts
        2. Process in batches of BATCH_SIZE, folding into running draft
        3. Stabilize against previous identity (gradual evolution)
        4. Store new version

        Returns the new IdentityVersion or None if no thoughts to process.
        """
        # Gather inputs
        thoughts = await self.sqlite.get_active_thoughts(limit=200)
        if not thoughts:
            logger.info("No active thoughts for identity synthesis")
            return None

        # Sort by confidence descending — most confident thoughts first
        thoughts.sort(key=lambda t: t.get("confidence", 0.5), reverse=True)

        # Get previous identity for stabilization
        prev_row = await self.sqlite.get_current_identity()
        prev_identity = prev_row["content"] if prev_row else None

        # Get stats for context
        stats = {
            "total_thoughts": len(thoughts),
        }
        try:
            cursor = await self.sqlite._db.execute(
                "SELECT COUNT(*) FROM memories WHERE is_archived = 0"
            )
            stats["total_memories"] = (await cursor.fetchone())[0]
            cursor = await self.sqlite._db.execute(
                "SELECT COUNT(*) FROM sessions"
            )
            stats["total_sessions"] = (await cursor.fetchone())[0]
        except Exception:
            pass

        # Progressive batching
        batches = [
            thoughts[i:i + BATCH_SIZE]
            for i in range(0, len(thoughts), BATCH_SIZE)
        ]

        draft = None
        for i, batch in enumerate(batches):
            batch_text = "\n".join(
                f"- [{t.get('category', 'observation')}] (conf: {t.get('confidence', 0.5):.1f}) "
                f"{t.get('content', '')}"
                for t in batch
            )

            if draft is None:
                # First batch — generate initial draft
                system, user = synthesize_identity(
                    previous_identity=prev_identity,
                    thoughts_batch=batch_text,
                    stats=stats,
                )
            else:
                # Subsequent batches — fold into existing draft
                system, user = fold_thoughts_into_identity(
                    current_draft=draft,
                    new_thoughts_batch=batch_text,
                )

            try:
                draft = await self.router.generate(
                    TaskType.REASONING, user, system=system,
                )
            except Exception as e:
                logger.error("Identity synthesis batch %d failed: %s", i, e)
                if draft is None:
                    return None
                break  # keep whatever draft we have

        if not draft:
            return None

        # Stabilize against previous identity (if exists)
        if prev_identity and draft:
            try:
                system, user = stabilize_identity(prev_identity, draft)
                stabilized = await self.router.generate(
                    TaskType.REASONING, user, system=system,
                )
                if stabilized and len(stabilized.strip()) > 50:
                    draft = stabilized
            except Exception as e:
                logger.warning("Identity stabilization failed, using raw draft: %s", e)

        # Store
        version_id = await self.sqlite.add_identity_version(
            content=draft.strip(),
            trigger=trigger,
            thought_count=len(thoughts),
        )

        version = IdentityVersion(
            id=version_id,
            version_number=version_id,  # approximate — real number set by add_identity_version
            content=draft.strip(),
            trigger=trigger,
            thought_count=len(thoughts),
        )

        logger.info(
            "Identity synthesis complete: version %d, %d thoughts processed, trigger=%s",
            version_id, len(thoughts), trigger,
        )
        return version

    async def get_current_identity(self) -> str:
        """Get the current identity text."""
        row = await self.sqlite.get_current_identity()
        return row["content"] if row else ""

    async def get_identity_history(self, limit: int = 10) -> list[IdentityVersion]:
        """Get identity version history."""
        rows = await self.sqlite.get_identity_history(limit=limit)
        return [
            IdentityVersion(
                id=r["id"],
                version_number=r["version_number"],
                content=r["content"],
                trigger=r["trigger"],
                thought_count=r["thought_count"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
