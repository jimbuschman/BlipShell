"""Session lifecycle management (port of SessionManager.cs).

Handles in-memory message tracking, text cleaning,
dump-to-memory lifecycle, and session summary generation.
"""

import logging
import re
from datetime import datetime
from typing import Optional

from blipshell.llm.prompts import (
    generate_session_title,
    summarize_session_conversation,
    summarize_session_summaries,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.manager import MemoryManager, PoolItem, estimate_tokens
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.session import MessageRole, Session, SessionMessage

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages conversation sessions with memory integration.

    Port of SessionManager.cs with enhancements:
    - In-memory message tracking
    - Text cleaning
    - Dump-to-memory lifecycle
    - Session summary generation (chunk 20 → summarize → meta-summarize → title)
    - Named projects
    - Session resume
    """

    def __init__(
        self,
        sqlite: SQLiteStore,
        memory_manager: MemoryManager,
        processor: MemoryProcessor,
        router: LLMRouter,
        summary_chunk_size: int = 20,
    ):
        self.sqlite = sqlite
        self.memory_manager = memory_manager
        self.processor = processor
        self.router = router
        self.summary_chunk_size = summary_chunk_size

        self.session_id: Optional[int] = None
        self.project: Optional[str] = None
        self._messages: list[SessionMessage] = []
        self._dumped_indices: set[int] = set()
        self._currently_saving = False

    async def start_session(
        self, project: Optional[str] = None, resume_session_id: Optional[int] = None
    ) -> int:
        """Start a new session or resume an existing one."""
        if resume_session_id:
            session = await self.sqlite.get_session(resume_session_id)
            if session:
                self.session_id = session.id
                self.project = session.project
                # Load existing messages into memory manager
                memories = await self.sqlite.get_memories_by_session(session.id)
                for mem in memories:
                    self._messages.append(SessionMessage(
                        role=MessageRole(mem.role),
                        content=mem.content,
                        timestamp=mem.timestamp,
                    ))
                    self._dumped_indices.add(len(self._messages) - 1)
                logger.info("Resumed session %d (%d messages)", session.id, len(memories))
                return session.id

        # Create new session
        self.project = project
        self.session_id = await self.sqlite.create_session(
            title="New Session",
            project=project,
        )
        self._messages.clear()
        self._dumped_indices.clear()
        logger.info("Started new session %d (project=%s)", self.session_id, project)
        return self.session_id

    def add_message(self, role: MessageRole, content: str, tool_calls: list[dict] | None = None):
        """Add a message to the current session."""
        cleaned = self._clean_text(content)
        msg = SessionMessage(
            role=role,
            content=cleaned,
            timestamp=datetime.utcnow(),
            token_count=estimate_tokens(cleaned),
            tool_calls=tool_calls,
        )
        self._messages.append(msg)

        # Add to memory manager ActiveSession pool
        self.memory_manager.add_memory("ActiveSession", PoolItem(
            text=f"{role.value}: {cleaned}",
            session_role=role.value,
            priority_score=1.0 if role == MessageRole.USER else 0.8,
            session_id=self.session_id or 0,
        ))

    def get_messages(self) -> list[SessionMessage]:
        """Get all messages in the current session."""
        return list(self._messages)

    def get_ollama_messages(self) -> list[dict]:
        """Get messages formatted for Ollama API."""
        return [msg.to_ollama_message() for msg in self._messages]

    def get_undumped_messages(self) -> list[SessionMessage]:
        """Get messages not yet dumped to persistent memory."""
        return [
            msg for i, msg in enumerate(self._messages)
            if i not in self._dumped_indices
        ]

    async def dump_to_memory(self):
        """Dump undumped messages to persistent memory.

        Port of MemoryDB.DumpConversationToMemory().
        """
        if self._currently_saving or not self.session_id:
            return

        self._currently_saving = True
        try:
            undumped = [
                (i, msg) for i, msg in enumerate(self._messages)
                if i not in self._dumped_indices
            ]

            for idx, msg in undumped:
                if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                    await self.processor.process_message(
                        text=msg.content,
                        role=msg.role.value,
                        session_id=self.session_id,
                    )
                    self._dumped_indices.add(idx)

            await self.sqlite.update_session(
                self.session_id,
                last_active=datetime.utcnow().isoformat(),
                message_count=len(self._messages),
            )
        except Exception as e:
            logger.error("Failed to dump session to memory: %s", e)
        finally:
            self._currently_saving = False

    async def end_session(self):
        """End the current session: dump remaining messages, generate summary."""
        if not self.session_id:
            return

        # Dump any remaining messages
        await self.dump_to_memory()

        # Generate session summary
        await self._create_session_summary()

        logger.info("Session %d ended", self.session_id)

    async def _create_session_summary(self):
        """Generate session summary using chunked summarization.

        Port of MemoryDB.CreateSessionSummary():
        - Chunk messages into groups of 20
        - Summarize each chunk
        - Meta-summarize all chunk summaries
        - Generate title from final summary
        """
        if not self.session_id:
            return

        memories = await self.sqlite.get_memories_by_session(self.session_id)
        if not memories:
            return

        texts = [m.summary or m.content for m in memories]

        if len(texts) > self.summary_chunk_size:
            # Chunk and summarize
            chunk_summaries = []
            for i in range(0, len(texts), self.summary_chunk_size):
                chunk = texts[i:i + self.summary_chunk_size]
                chunk_text = "\n".join(chunk)
                try:
                    chunk_summary = await self.router.generate(
                        TaskType.SUMMARIZATION,
                        summarize_session_summaries(chunk_text),
                    )
                    chunk_summaries.append(chunk_summary)
                except Exception as e:
                    logger.error("Chunk summarization failed: %s", e)
                    chunk_summaries.append(chunk_text[:200])

            # Meta-summarize
            try:
                summary = await self.router.generate(
                    TaskType.SUMMARIZATION,
                    summarize_session_summaries("\n".join(chunk_summaries)),
                )
            except Exception as e:
                logger.error("Meta-summarization failed: %s", e)
                summary = "\n".join(chunk_summaries)
        else:
            # Direct summarize
            all_text = "\n".join(texts)
            try:
                summary = await self.router.generate(
                    TaskType.SUMMARIZATION,
                    summarize_session_conversation(all_text),
                )
            except Exception as e:
                logger.error("Session summarization failed: %s", e)
                summary = all_text[:500]

        # Generate title
        try:
            title = await self.router.generate(
                TaskType.SUMMARIZATION,
                generate_session_title(summary),
            )
        except Exception as e:
            logger.error("Title generation failed: %s", e)
            title = f"Session {self.session_id}"

        await self.sqlite.update_session(
            self.session_id,
            title=title.strip(),
            summary=summary.strip(),
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for storage (port of SessionManager.CleanText)."""
        if not text or not text.strip():
            return ""

        cleaned = text.strip()
        cleaned = cleaned.replace("\r\n", "\n")
        cleaned = cleaned.replace("\t", " ")
        cleaned = re.sub(r"  +", " ", cleaned)  # collapse double spaces
        return cleaned

    @property
    def message_count(self) -> int:
        return len(self._messages)
