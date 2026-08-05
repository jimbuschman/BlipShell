"""Session lifecycle management (port of SessionManager.cs).

Handles in-memory message tracking, text cleaning,
dump-to-memory lifecycle, and session summary generation.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
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

    # No timeouts on session-close operations. OllamaGate serializes all
    # local Ollama calls, so each step may wait behind background work +
    # model swaps. Artificial timeouts just lose work. If the user wants
    # to bail, they Ctrl+C.

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
        self._memory_db_ids: dict[int, int] = {}  # message index → memories row ID
        self._pending_persists: list[asyncio.Task] = []  # pending save_raw_memory tasks
        self._currently_saving = False
        # Sessions already closed. end_session deliberately leaves session_id
        # intact (callers still read it), so without this a second call would
        # re-run summarization, lesson extraction and the digest update —
        # paying for the LLM work twice and duplicating lessons. The web app
        # and telegram both call end_session more than once.
        self._ended_sessions: set[int] = set()

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
                    # Restore image refs (vision turns) from metadata_json if present.
                    images = None
                    if getattr(mem, "metadata_json", None):
                        try:
                            images = json.loads(mem.metadata_json).get("images")
                        except (ValueError, AttributeError):
                            images = None
                    self._messages.append(SessionMessage(
                        role=MessageRole(mem.role),
                        content=mem.content,
                        timestamp=mem.timestamp,
                        images=images,
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
        self._memory_db_ids.clear()
        logger.info("Started new session %d (project=%s)", self.session_id, project)
        return self.session_id

    def add_message(self, role: MessageRole, content: str, tool_calls: list[dict] | None = None,
                    images: list[dict] | None = None):
        """Add a message to the current session.

        Persists to memories table immediately (is_processed=0) so messages
        survive crashes. The memory pipeline later updates the row with
        summary, rank, importance, and sets is_processed=1.

        `images` is an optional list of ImageRef dicts (vision input). The bytes
        live on disk; only the refs travel with the message and get re-sent on
        later turns (persist & replay) while in the recent-history window.
        """
        cleaned = self._clean_text(content)
        if not cleaned and not images:
            return  # skip empty text-only messages — they contaminate history
        now = datetime.now(timezone.utc)
        msg = SessionMessage(
            role=role,
            content=cleaned,
            timestamp=now,
            token_count=estimate_tokens(cleaned),
            tool_calls=tool_calls,
            images=images,
        )
        self._messages.append(msg)

        # Persist to SQLite immediately (tracked task — awaited before enqueue)
        # Capture index NOW (before another add_message increments _messages)
        if self.session_id and role in (MessageRole.USER, MessageRole.ASSISTANT):
            msg_idx = len(self._messages) - 1
            task = asyncio.ensure_future(self._persist_message(
                self.session_id, role.value, cleaned, now.isoformat(), msg_idx, images,
            ))
            self._pending_persists.append(task)

        # Add to memory manager ActiveSession pool
        pool_text = f"{role.value}: {cleaned}"
        if images:
            names = ", ".join(i.get("orig_name", "image") for i in images)
            pool_text += f" [image: {names}]"
        self.memory_manager.add_memory("ActiveSession", PoolItem(
            text=pool_text,
            session_role=role.value,
            priority_score=1.0 if role == MessageRole.USER else 0.8,
            session_id=self.session_id or 0,
        ))

    async def _persist_message(self, session_id: int, role: str, content: str, timestamp: str,
                               msg_idx: int, images: list[dict] | None = None):
        """Persist a raw message to memories table (is_processed=0).

        Image refs (if any) go in metadata_json so vision turns survive a resume.
        """
        try:
            metadata = json.dumps({"images": images}) if images else None
            mem_id = await self.sqlite.save_raw_memory(
                session_id, role, content, timestamp, metadata=metadata,
            )
            # Store the memory ID so the pipeline can update it later
            self._memory_db_ids[msg_idx] = mem_id
        except Exception as e:
            logger.error("Failed to persist raw memory: %s", e)

    async def flush_pending_persists(self):
        """Await all pending persist tasks so _memory_db_ids is populated."""
        if self._pending_persists:
            await asyncio.gather(*self._pending_persists, return_exceptions=True)
            self._pending_persists.clear()

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
        No timeouts by design (see the class-level note): a failed message is
        skipped (not marked as dumped) so it can be retried next time.
        """
        if self._currently_saving or not self.session_id:
            return

        self._currently_saving = True
        dumped_count = 0
        skipped_count = 0
        try:
            # The raw-persist tasks must land first: process_message with
            # memory_id=None creates a SECOND row for a message whose raw
            # persist is still in flight (duplicate memory, seen when project
            # activation dumps mid-turn).
            await self.flush_pending_persists()
            undumped = [
                (i, msg) for i, msg in enumerate(self._messages)
                if i not in self._dumped_indices
            ]

            for idx, msg in undumped:
                if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                    try:
                        mem_id = self._memory_db_ids.get(idx)
                        await self.processor.process_message(
                            text=msg.content,
                            role=msg.role.value,
                            session_id=self.session_id,
                            memory_id=mem_id,
                        )
                        self._dumped_indices.add(idx)
                        dumped_count += 1
                    except Exception as e:
                        logger.error("dump_to_memory: message %d failed: %s", idx, e)
                        skipped_count += 1

            if dumped_count > 0 or skipped_count > 0:
                logger.info(
                    "dump_to_memory: %d saved, %d skipped", dumped_count, skipped_count,
                )

            await self.sqlite.update_session(
                self.session_id,
                last_active=datetime.now(timezone.utc).isoformat(),
                message_count=len(self._messages),
            )
        except Exception as e:
            logger.error("Failed to dump session to memory: %s", e)
        finally:
            self._currently_saving = False

    async def end_session(self, on_status=None) -> dict[str, str]:
        """End the current session: dump remaining messages, generate summary, extract lessons.

        No artificial timeouts — each step runs to completion. OllamaGate
        serializes local calls, so waits can be long but work gets done.
        User can Ctrl+C if they need to bail early.

        Critical: message_count and a fallback title are saved FIRST so
        that even if LLM calls (summary, lessons) fail, the session is
        still identifiable and recent-session loading works correctly.

        Every step is isolated, so a failure never blocks the rest — but a
        failure is REPORTED. Previously each step logged and moved on while
        on_status announced the next one, so a session that produced no
        summary, no lessons and no digest looked like a clean close
        (deep-dive 2026-08-04). Returns {step: "ok"|"failed: ..."}.
        """
        if not self.session_id:
            return {}
        if self.session_id in self._ended_sessions:
            logger.info("Session %d already ended — skipping close work", self.session_id)
            return {"status": "already_ended"}

        def _status(msg: str):
            if on_status:
                on_status(msg)
            logger.info(msg)

        outcomes: dict[str, str] = {}

        async def _step(name: str, label: str, coro_fn):
            """Run one close step, reporting its OUTCOME rather than just its
            intent. A step may return a status string ("skipped: ...") to
            distinguish a deliberate no-op from work actually done."""
            _status(label)
            try:
                reported = await coro_fn()
                outcomes[name] = reported if isinstance(reported, str) else "ok"
            except Exception as e:
                logger.error("%s failed: %s", name, e, exc_info=True)
                outcomes[name] = f"failed: {e}"
                _status(f"  {name} FAILED: {e}")

        undumped_count = len(self._messages) - len(self._dumped_indices)

        # Save message_count and fallback title FIRST — before any LLM calls.
        # This ensures the session is identifiable even if everything else fails.
        await _step(
            "bookkeeping", "Saving session record...",
            lambda: self.sqlite.update_session(
                self.session_id,
                message_count=len(self._messages),
                title=self._make_fallback_title(),
                last_active=datetime.now(timezone.utc).isoformat(),
            ),
        )

        await _step("dump", f"Saving {undumped_count} messages...", self.dump_to_memory)
        # Summary overwrites the fallback title with an LLM-generated one
        await _step("summary", "Generating session summary...", self._create_session_summary)
        await _step("lessons", "Extracting lessons...", self._extract_lessons)
        if self.project:
            await _step("digest", "Updating project digest...", self._update_project_digest)

        self._ended_sessions.add(self.session_id)

        failed = {k: v for k, v in outcomes.items() if v != "ok"}
        if failed:
            logger.warning(
                "Session %d ended with %d failed step(s): %s",
                self.session_id, len(failed), ", ".join(sorted(failed)),
            )
            _status(
                f"Session closed with {len(failed)} problem(s): {', '.join(sorted(failed))}"
            )
        else:
            logger.info("Session %d ended", self.session_id)
        return outcomes

    def _make_fallback_title(self) -> str:
        """Create a fallback title from the first user message.

        Used as a placeholder before the LLM generates a proper title.
        Better than 'New Session' for identification and search.
        """
        for msg in self._messages:
            if msg.role == MessageRole.USER and msg.content.strip():
                text = msg.content.strip().replace("\n", " ")
                if len(text) > 80:
                    text = text[:77] + "..."
                return text
        return f"Session {self.session_id}"

    async def _update_project_digest(self):
        """Update the project digest with this session's summary."""
        if not self.project or not self.session_id:
            return
        session = await self.sqlite.get_session(self.session_id)
        if not session or not session.summary:
            return
        from blipshell.memory.project_digest import ProjectDigestManager
        digest_mgr = ProjectDigestManager(self.sqlite, self.router)
        await digest_mgr.update_digest(
            self.project, session.summary, session.title or "", self.session_id,
        )

    async def _extract_lessons(self) -> str | None:
        """Extract lessons from the session conversation.

        Only runs if there were enough messages to be meaningful (5+).
        Sends the conversation to the LLM for lesson extraction.

        Raises on failure — end_session's step wrapper logs AND reports it.
        This used to swallow its own exception, which meant end_session saw
        a successful close for a session that produced no lessons at all.
        """
        if not self.session_id:
            return "skipped: no session"
        if len(self._messages) < 5:
            return f"skipped: only {len(self._messages)} messages (need 5)"

        # Build conversation text from messages
        conversation_lines = []
        for msg in self._messages:
            conversation_lines.append(f"{msg.role.value}: {msg.content}")
        conversation_text = "\n".join(conversation_lines)

        await self.processor.process_lesson(
            conversation_text, self.session_id, project=self.project,
        )
        logger.info("Lessons extracted for session %d", self.session_id)
        return None

    async def _create_session_summary(self):
        """Generate session summary using chunked summarization.

        Port of MemoryDB.CreateSessionSummary():
        - Chunk messages into groups of 20
        - Summarize each chunk
        - Meta-summarize all chunk summaries
        - Generate title from final summary
        """
        if not self.session_id:
            return "skipped: no session"

        memories = await self.sqlite.get_memories_by_session(self.session_id)
        if not memories:
            return "skipped: no memories to summarize"

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
