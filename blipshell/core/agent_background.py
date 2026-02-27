"""Background memory processing mixin for Agent.

Extracts memory worker callbacks and reflection logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    pass  # All types accessed via self

from blipshell.llm.prompts import reflect_on_response, summarize_session_chunk
from blipshell.llm.router import TaskType
from blipshell.models.session import MessageRole

logger = logging.getLogger(__name__)


class BackgroundMixin:
    """Background memory processing methods mixed into Agent."""

    async def _background_memory_processing(self):
        """Enqueue undumped messages to the memory worker thread."""
        try:
            if self.session_manager.message_count % 5 == 0:
                self._enqueue_undumped_messages()
        except Exception as e:
            logger.error("Background memory processing error: %s", e)

    def _enqueue_undumped_messages(self):
        """Push undumped session messages to the memory worker queue."""
        if not self._memory_worker or not self._memory_worker.is_alive:
            logger.debug("Memory worker not available, skipping enqueue")
            return

        from blipshell.memory.worker import WorkItem, WorkType

        undumped = [
            (i, msg) for i, msg in enumerate(self.session_manager._messages)
            if i not in self.session_manager._dumped_indices
        ]
        for idx, msg in undumped:
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                db_id = self.session_manager._message_db_ids.get(idx)
                self._memory_worker.enqueue(WorkItem(
                    work_type=WorkType.PROCESS_MESSAGE,
                    text=msg.content,
                    role=msg.role.value,
                    session_id=self.session_manager.session_id,
                    message_db_id=db_id,
                ))
                self.session_manager._dumped_indices.add(idx)

    async def _summarize_overflow(self, text: str) -> str:
        """Callback for memory manager overflow summarization."""
        return await self.router.generate(
            TaskType.SUMMARIZATION,
            summarize_session_chunk(text),
        )

    async def _reflect_on_response(
        self,
        user_message: str,
        original_response: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Run a second LLM pass to critique and improve the response.

        Returns the improved response if changes were suggested,
        otherwise returns the original.
        """
        try:
            prompt = reflect_on_response(user_message, original_response)
            improved = await self.router.generate(TaskType.REASONING, prompt)
            improved = improved.strip()

            if not improved or improved == "NO_CHANGES":
                if on_token:
                    on_token("[No changes needed]\n")
                return original_response

            if on_token:
                on_token(improved)
            return improved
        except Exception as e:
            logger.error("Reflection failed: %s", e)
            if on_token:
                on_token(f"[Reflection failed: {e}]\n")
            return original_response
