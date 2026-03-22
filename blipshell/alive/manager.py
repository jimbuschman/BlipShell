"""Alive Manager — facade wiring together the alive subsystems.

Provides session lifecycle hooks (on_session_start, on_session_end)
and coordinates ThoughtEngine, IdentitySynthesizer, and initiative queue.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from blipshell.alive.thought_engine import ThoughtEngine

if TYPE_CHECKING:
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import AliveConfig

logger = logging.getLogger(__name__)


class AliveManager:
    """Orchestrates the alive system: thoughts, identity, initiative.

    Lifecycle:
    - initialize() creates sub-components
    - on_session_start() loads context for injection
    - on_session_end() triggers thought extraction
    - run_identity_synthesis() regenerates identity from thoughts
    """

    def __init__(self):
        self.sqlite: SQLiteStore | None = None
        self.chroma: ChromaStore | None = None
        self.router: LLMRouter | None = None
        self.config: AliveConfig | None = None
        self.thought_engine: ThoughtEngine | None = None
        self._identity_synthesizer = None  # lazy import to avoid circular deps

    def initialize(
        self,
        config: AliveConfig,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        router: LLMRouter,
    ):
        """Wire up sub-components."""
        self.config = config
        self.sqlite = sqlite
        self.chroma = chroma
        self.router = router

        self.thought_engine = ThoughtEngine(sqlite, chroma, router, config)
        logger.info("AliveManager initialized")

    async def on_session_start(self, session_id: int) -> dict:
        """Load alive context for session injection.

        Returns dict with:
        - identity: str — current self-authored identity text
        - initiative_items: list[dict] — pending items to bring up
        - recent_thoughts: list[dict] — recent thoughts for context
        """
        result = {
            "identity": "",
            "initiative_items": [],
            "recent_thoughts": [],
        }

        try:
            # Load current identity
            identity = await self.sqlite.get_current_identity()
            if identity:
                result["identity"] = identity["content"]

            # Load pending initiative items
            inject_count = self.config.initiative.inject_count
            items = await self.sqlite.get_pending_initiative(limit=inject_count)
            result["initiative_items"] = items

            # Load recent thoughts for internal context
            thoughts = await self.sqlite.get_recent_thoughts(limit=5)
            result["recent_thoughts"] = thoughts

        except Exception as e:
            logger.error("Failed to load alive context: %s", e)

        return result

    async def on_session_end(
        self,
        session_id: int,
        session_summary: str,
        conversation_text: str,
    ):
        """Post-session: extract thoughts and update initiative queue."""
        if not self.thought_engine:
            return

        # Get current identity for context
        identity = ""
        try:
            id_row = await self.sqlite.get_current_identity()
            if id_row:
                identity = id_row["content"]
        except Exception:
            pass

        # Extract thoughts
        try:
            thoughts = await self.thought_engine.generate_thoughts(
                session_summary=session_summary,
                conversation_text=conversation_text,
                session_id=session_id,
                current_identity=identity,
            )
            logger.info("Extracted %d thoughts from session %d", len(thoughts), session_id)
        except Exception as e:
            logger.error("Thought extraction failed for session %d: %s", session_id, e)

    async def consume_initiative(self, item_id: int, session_id: int):
        """Mark an initiative item as raised."""
        await self.sqlite.consume_initiative(item_id, session_id)

    async def run_identity_synthesis(self, trigger: str = "manual") -> str | None:
        """Run identity synthesis. Returns new identity text or None."""
        from blipshell.alive.identity import IdentitySynthesizer

        if not self._identity_synthesizer:
            self._identity_synthesizer = IdentitySynthesizer(
                self.sqlite, self.router,
            )

        try:
            version = await self._identity_synthesizer.synthesize(trigger=trigger)
            if version:
                return version.content
        except Exception as e:
            logger.error("Identity synthesis failed: %s", e)

        return None

    async def get_alive_status(self) -> dict:
        """Get status overview for /alive command."""
        thought_count = await self.sqlite.get_thought_count()
        identity = await self.sqlite.get_current_identity()
        pending = await self.sqlite.get_pending_initiative(limit=100)
        logs = await self.sqlite.get_monologue_logs(limit=1)

        return {
            "enabled": self.config.enabled if self.config else False,
            "thought_count": thought_count,
            "identity_version": identity["version_number"] if identity else 0,
            "identity_trigger": identity["trigger"] if identity else None,
            "identity_date": identity["created_at"] if identity else None,
            "initiative_pending": len(pending),
            "last_monologue": logs[0]["created_at"] if logs else None,
            "last_monologue_thoughts": logs[0]["thoughts_generated"] if logs else 0,
        }
