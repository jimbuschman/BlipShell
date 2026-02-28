"""Main agent loop (ports OllamaChat.SendMessageToOllama + Form1.RunChat).

Key improvement: Uses native Ollama tool calling instead of parsing
tool calls from markdown code blocks.

Extended with:
- Task planner + executor (Phase 1): complex messages get decomposed
- Background task manager (Phase 2): async long-running tasks
- Workflow system (Phase 4): named reusable templates

Modularized via mixins:
- ToolsMixin: tool registration
- BackgroundMixin: memory worker callbacks, reflection
- ProjectMixin: project activation/deactivation, context scanning
- SessionMixin: session lifecycle, memory loading, startup tasks
- ChatMixin: chat pipeline, message building, event logging
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

from blipshell.core.agent_background import BackgroundMixin
from blipshell.core.agent_chat import ChatMixin
from blipshell.core.agent_project import ProjectMixin
from blipshell.core.agent_session import SessionMixin
from blipshell.core.agent_tools import ToolsMixin
from blipshell.core.background import BackgroundTaskManager
from blipshell.core.config import ConfigManager
from blipshell.core.executor import TaskExecutor
from blipshell.core.planner import TaskPlanner
from blipshell.core.repo_map import RepoMap
from blipshell.core.tools.base import ToolRegistry
from blipshell.core.workflows import WorkflowExecutor, WorkflowRegistry
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.job_queue import LLMJobQueue
from blipshell.llm.model_settings import ModelSettingsRegistry
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.manager import MemoryManager, estimate_tokens
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.search import MemorySearch
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import BlipShellConfig, get_ollama_url
from blipshell.models.session import MessageRole, SessionMessage
from blipshell.session.manager import SessionManager

logger = logging.getLogger(__name__)


class Agent(
    ToolsMixin,
    BackgroundMixin,
    ProjectMixin,
    SessionMixin,
    ChatMixin,
):
    """Main BlipShell agent that orchestrates the full chat loop.

    Lifecycle per message:
    1. Load core memories → load lessons → search relevant memories
    2. Calculate token budget
    3. Gather memory context from all pools
    4. Build message list (system + memory context + conversation)
    5. Classify complexity: simple → direct chat, complex → plan + execute
    6. Send to Ollama with native tool calling
    7. Handle tool call loop (max N iterations)
    8. Update session + memory pools
    9. Background: process memories (summarize, embed, tag, rank)
    """

    def __init__(self, config: BlipShellConfig, config_manager: ConfigManager):
        self.config = config
        self.config_manager = config_manager

        # Infrastructure
        self.sqlite: Optional[SQLiteStore] = None
        self.chroma: Optional[ChromaStore] = None
        self.endpoint_manager: Optional[EndpointManager] = None
        self.router: Optional[LLMRouter] = None
        self.job_queue: Optional[LLMJobQueue] = None
        self.model_settings = ModelSettingsRegistry()

        # Memory
        self.memory_manager: Optional[MemoryManager] = None
        self.processor: Optional[MemoryProcessor] = None
        self.search: Optional[MemorySearch] = None

        # Session
        self.session_manager: Optional[SessionManager] = None

        # Tools
        self.tool_registry = ToolRegistry()

        # Task planning + execution (Phase 1)
        self.task_planner: Optional[TaskPlanner] = None
        self.task_executor: Optional[TaskExecutor] = None

        # Background tasks (Phase 2)
        self.background_manager: Optional[BackgroundTaskManager] = None

        # Workflows (Phase 4)
        self.workflow_registry: Optional[WorkflowRegistry] = None
        self.workflow_executor: Optional[WorkflowExecutor] = None

        # MCP (Model Context Protocol) servers
        self.mcp_manager = None

        self._health_check_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()
        self._memory_worker = None  # MemoryWorker — dedicated thread for background processing
        self._last_endpoint_used: Optional[str] = None
        self._initialized = False
        self.think_enabled: bool = False  # /think toggle — off for fast simple chat, complex auto-enables
        self.reflect_enabled: bool = False  # /reflect toggle — second-pass self-critique
        self._turn_number: int = 0
        self._last_context_stats: Optional[dict] = None

        # Token usage tracking (per-endpoint, per-session)
        # { endpoint_name: { "prompt_tokens": int, "completion_tokens": int, "requests": int } }
        self._token_usage: dict[str, dict[str, int]] = {}

        # Interactive callbacks (wired by CLI)
        self._ask_user_callback: Optional[Callable] = None

        # Project (BlipCode)
        self.active_project: Optional[dict] = None
        self._project_context: str = ""
        self._file_changes: list[dict] = []
        self._files_read: set[str] = set()  # tracks files/dirs already read this session
        self._repo_map: Optional[RepoMap] = None

    async def initialize(self, on_status=None):
        """Initialize all subsystems.

        Args:
            on_status: Optional callback(msg: str) for progress reporting.
        """
        if self._initialized:
            return

        def _status(msg: str):
            if on_status:
                on_status(msg)

        _status("Loading database...")
        self.sqlite = SQLiteStore(self.config.database.path)
        await self.sqlite.initialize()

        _status("Connecting to ChromaDB...")
        self.chroma = ChromaStore(
            persist_dir=self.config.database.chroma_path,
            embedding_model=self.config.models.embedding,
            ollama_url=get_ollama_url(self.config.endpoints),
        )
        self.chroma.initialize()

        # Endpoint manager
        self.endpoint_manager = EndpointManager(self.config.endpoints, self.config.llm)

        # Router
        self.router = LLMRouter(self.config.models, self.endpoint_manager, pii_enabled=self.config.pii.enabled)

        # Job queue
        self.job_queue = LLMJobQueue()
        self.job_queue.start()

        # Memory manager — use endpoint context_tokens for pool sizing
        endpoint_ctx = self.endpoint_manager.get_context_tokens_for_role(
            "tool_calling", default=65536,
        )
        self.memory_manager = MemoryManager(self.config.memory, context_tokens=endpoint_ctx)
        self.memory_manager.set_summarize_callback(self._summarize_overflow)

        # Processor
        self.processor = MemoryProcessor(self.sqlite, self.chroma, self.router,
                                         config=self.config.memory)

        # Search
        self.search = MemorySearch(
            self.sqlite, self.chroma, self.router,
            config=self.config.memory,
        )

        # Background memory worker (dedicated thread with own event loop + connections)
        _status("Starting memory worker...")
        from blipshell.memory.worker import MemoryWorker
        self._memory_worker = MemoryWorker(self.config, self.chroma)
        self._memory_worker.start()

        # Session manager
        self.session_manager = SessionManager(
            self.sqlite, self.memory_manager, self.processor, self.router,
            summary_chunk_size=self.config.session.summary_chunk_size,
        )

        # Task planner + executor
        # NOTE: ComplexityClassifier removed — model decides its own complexity.
        # !plan CLI prefix sets force_plan=True directly. See git history for heuristic.
        self.task_planner = TaskPlanner(
            self.router, self.sqlite, self.config.planner,
        )
        self.task_executor = TaskExecutor(
            router=self.router,
            sqlite=self.sqlite,
            tool_registry=self.tool_registry,
            config=self.config.planner,
            system_prompt=self.config.agent.system_prompt,
            max_tool_iterations=self.config.agent.max_tool_iterations,
            processor=self.processor,
        )

        # Background task manager (Phase 2)
        self.background_manager = BackgroundTaskManager(
            self.router, self.sqlite, self.config.worker,
            processor=self.processor,
        )

        # Workflow system (Phase 4)
        self.workflow_registry = WorkflowRegistry("workflows")
        self.workflow_executor = WorkflowExecutor(
            self.workflow_registry, self.task_executor, self.sqlite,
        )

        # Register tools
        self._register_tools()

        # Connect MCP servers (if configured)
        if self.config.mcp_servers:
            _status("Connecting MCP servers...")
            await self._connect_mcp_servers()

        # Load per-model behavioral settings
        if self.config.model_settings:
            self.model_settings.load(self.config.model_settings)

        # Load discovered tag patterns into tagger
        await self._load_discovered_tags()

        _status("Checking endpoints...")
        await self.endpoint_manager.startup_health_check()

        # Auto-backup if >24h since last backup
        _status("Checking backups...")
        await self._maybe_auto_backup()

        _status("Running tag discovery...")
        await self._auto_tag_discovery()

        _status("Backfilling entity embeddings...")
        await self._backfill_entity_embeddings()

        # Start periodic health check (re-detects endpoints that come/go)
        self._health_check_task = self.endpoint_manager.start_health_loop(
            interval=60, on_check=self.router.clear_failed_models,
        )

        # Queue background tasks instead of blocking startup
        await self._enqueue_startup_background_tasks()

        self._initialized = True
        logger.info("Agent initialized")

    async def _maybe_auto_backup(self):
        """Auto-backup if more than 24 hours since last backup.

        Runs synchronously in a thread to avoid blocking the event loop.
        Logs warnings on failure but never blocks startup.
        """
        try:
            from scripts.backup_db import get_last_backup_time, run_backup, rotate_backups

            last = get_last_backup_time()
            if last is not None:
                hours_ago = (datetime.now() - last).total_seconds() / 3600
                if hours_ago < 24:
                    logger.debug("Last backup %.1fh ago — skipping auto-backup", hours_ago)
                    return

            logger.info("Auto-backup: no recent backup found, creating one...")
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: run_backup(
                    db_path=self.config.database.path,
                    chroma_path=self.config.database.chroma_path,
                    quiet=True,
                ),
            )
            if result:
                logger.info("Auto-backup created: %s", result)
                # Rotate, keeping last 5
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: rotate_backups(keep=5),
                )
            else:
                logger.warning("Auto-backup failed")
        except Exception as e:
            logger.warning("Auto-backup error (non-fatal): %s", e)

    async def night_cleanup(
        self,
        on_status: Callable[[str], None] | None = None,
        timeout_per_message: int = 120,
    ) -> dict:
        """Reprocess failed messages with relaxed timeouts.

        Unlike the startup sweep (30s, limit 50), this uses 120s per message
        and processes up to 500 messages. Designed for manual/scheduled runs
        when the system isn't under interactive load.

        Returns:
            Dict with processed, failed, total counts.
        """
        def _status(msg: str):
            if on_status:
                on_status(msg)
            logger.info("Night cleanup: %s", msg)

        _status("Fetching unprocessed messages...")
        unprocessed = await self.sqlite.get_unprocessed_messages(limit=500)
        if not unprocessed:
            _status("No unprocessed messages found.")
            return {"processed": 0, "failed": 0, "total": 0}

        _status(f"Found {len(unprocessed)} unprocessed messages, reprocessing with {timeout_per_message}s timeout...")
        processed = 0
        failed = 0
        for i, msg in enumerate(unprocessed):
            _status(f"Processing message {i + 1}/{len(unprocessed)} (id={msg['id']})...")
            try:
                await asyncio.wait_for(
                    self.processor.process_message(
                        text=msg["content"],
                        role=msg["role"],
                        session_id=msg["session_id"],
                    ),
                    timeout=timeout_per_message,
                )
                await self.sqlite.mark_message_processed(msg["id"])
                processed += 1
            except asyncio.TimeoutError:
                logger.warning(
                    "Night cleanup: message %d timed out after %ds",
                    msg["id"], timeout_per_message,
                )
                failed += 1
            except Exception as e:
                logger.warning("Night cleanup: message %d failed: %s", msg["id"], e)
                failed += 1

        result = {"processed": processed, "failed": failed, "total": len(unprocessed)}
        _status(f"Done: {processed}/{len(unprocessed)} processed, {failed} failed")
        return result

    async def run_nightly(
        self,
        on_status: Callable[[str], None] | None = None,
        job: str | None = None,
    ) -> dict:
        """Run nightly maintenance jobs using the agent's existing connections."""
        from blipshell.core.nightly import NightlyRunner

        runner = NightlyRunner(
            self.config, self.sqlite, self.chroma,
            self.router, self.processor,
        )
        jobs = [job] if job else None
        return await runner.run(on_status=on_status, jobs=jobs)

    def _record_token_usage_from_chunk(self, chunk):
        """Extract and accumulate token usage from an Ollama response/chunk.

        Ollama returns prompt_eval_count (input tokens) and eval_count (output tokens)
        in the final chunk of a streaming response and in non-streaming responses.
        """
        # Extract counts — handle both object attrs and dict keys
        prompt_tokens = getattr(chunk, "prompt_eval_count", None)
        eval_tokens = getattr(chunk, "eval_count", None)
        if prompt_tokens is None and isinstance(chunk, dict):
            prompt_tokens = chunk.get("prompt_eval_count")
        if eval_tokens is None and isinstance(chunk, dict):
            eval_tokens = chunk.get("eval_count")

        if prompt_tokens is None and eval_tokens is None:
            return

        endpoint_name = self._last_endpoint_used or "unknown"
        if endpoint_name not in self._token_usage:
            self._token_usage[endpoint_name] = {
                "prompt_tokens": 0, "completion_tokens": 0, "requests": 0,
            }
        stats = self._token_usage[endpoint_name]
        stats["prompt_tokens"] += prompt_tokens or 0
        stats["completion_tokens"] += eval_tokens or 0
        stats["requests"] += 1

    async def _on_tool_executed(self, name: str, arguments: dict, result) -> None:
        """Callback for ChatLoop — tracks files and logs events."""
        # Track files/dirs already read
        if result.success and name in ("read_file", "list_directory"):
            read_path = arguments.get("path", "")
            if read_path:
                self._files_read.add(read_path)

        # Track file modifications
        if result.success and name in ("write_file", "edit_file"):
            file_path = arguments.get("path", "")
            self._file_changes.append({
                "path": file_path,
                "tool": name,
                "turn_number": self._turn_number,
            })
            # Invalidate repo map cache for edited files
            if self._repo_map and file_path.endswith(".py"):
                self._repo_map.invalidate(file_path)
            await self._log_event("file_modified", {
                "path": file_path, "tool": name,
            })

    # ── Session end & cleanup ────────────────────────────────────────────────

    async def end_session(self, on_status=None):
        """End the current session and clean up."""
        def _status(msg: str):
            if on_status:
                on_status(msg)

        # Cancel asyncio background tasks (lightweight, main-loop only)
        await self._cancel_background_tasks()
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None

        # Enqueue any remaining undumped messages to worker before shutdown
        self._enqueue_undumped_messages()

        # Drain memory worker FIRST — must finish all DB writes before
        # end_session runs summary/lessons on the main loop's connection.
        # Without this ordering, worker and main loop compete for the SQLite
        # write lock, causing "database is locked" errors and lesson timeouts.
        if self._memory_worker and self._memory_worker.is_alive:
            depth = self._memory_worker.queue_depth
            if depth > 0:
                _status(f"Draining memory queue ({depth} items)...")
            self._memory_worker.shutdown(timeout=60.0)

        if self.session_manager:
            await self.session_manager.end_session(on_status=on_status)

        if self.job_queue:
            await self.job_queue.stop()
        # Close ChromaDB after all writes are done (worker is stopped).
        # If the worker didn't exit in time, skip — the daemon thread is still
        # using ChromaDB and closing it causes errors. Process exit kills it.
        if self.chroma:
            if self._memory_worker and self._memory_worker.is_alive:
                logger.warning("Memory worker still alive, deferring ChromaDB close to process exit")
            else:
                self.chroma.close()

    async def force_cleanup(self):
        """Cancel all background tasks so the process can exit cleanly.

        Order matters: cancel in-flight writes → stop worker → close ChromaDB → close SQLite.
        """
        # 1. Cancel background memory processing tasks (prevents mid-write corruption)
        await self._cancel_background_tasks()

        # 2. Stop memory worker thread (short timeout for forced cleanup)
        if self._memory_worker:
            self._memory_worker.shutdown(timeout=5.0)

        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
        if self.job_queue:
            try:
                await self.job_queue.stop()
            except Exception:
                pass
        # 2. Disconnect MCP servers
        if self.mcp_manager:
            try:
                await self.mcp_manager.disconnect_all()
            except Exception:
                pass
        # 3. Close ChromaDB before SQLite (ChromaDB may reference SQLite data)
        if self.chroma:
            try:
                self.chroma.close()
            except Exception:
                pass
        # 4. Close SQLite last
        if self.sqlite:
            try:
                await self.sqlite.close()
            except Exception:
                pass

    async def _cancel_background_tasks(self):
        """Cancel all tracked background tasks and wait for them to finish."""
        if not self._background_tasks:
            return
        for task in self._background_tasks:
            task.cancel()
        # Give cancelled tasks a moment to clean up
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

    # ── Status & introspection ───────────────────────────────────────────────

    @property
    def file_changes(self) -> list[dict]:
        """Files modified during this session."""
        return list(self._file_changes)

    @property
    def last_endpoint_used(self) -> Optional[str]:
        """Name of the endpoint that handled the last chat request."""
        return self._last_endpoint_used

    def get_status(self) -> dict:
        """Get agent status for display."""
        return {
            "session_id": self.session_manager.session_id if self.session_manager else None,
            "project": self.session_manager.project if self.session_manager else None,
            "message_count": self.session_manager.message_count if self.session_manager else 0,
            "memory_usage": self.memory_manager.get_usage() if self.memory_manager else {},
            "endpoints": self.endpoint_manager.get_status() if self.endpoint_manager else [],
            "tools": self.tool_registry.get_tool_names(),
            "job_queue_pending": self.job_queue.pending_count if self.job_queue else 0,
            "planner_enabled": self.config.planner.enabled,
            "workflows_loaded": len(self.workflow_registry.list_all()) if self.workflow_registry else 0,
        }

    def get_context_info(self) -> dict:
        """Get context window usage info for /context display.

        Returns pool budgets, usage, context limit, and headroom — using
        the stats captured during the most recent _build_messages() call.
        """
        context_role = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        context_limit = self.endpoint_manager.get_context_tokens_for_role(
            context_role, default=65536,
        ) if self.endpoint_manager else 65536

        pool_usage = self.memory_manager.get_usage() if self.memory_manager else {}

        # Total tokens currently used across all pools
        total_pool_tokens = sum(s.get("used", 0) for s in pool_usage.values())

        # Message count from session
        message_count = self.session_manager.message_count if self.session_manager else 0

        # Estimate tokens used by session messages (rough — messages in memory
        # pools are already counted, but gives the user a sense of total usage)
        session_tokens = 0
        if self.session_manager:
            for msg in self.session_manager.get_messages():
                session_tokens += getattr(msg, "token_count", 0) or 0

        return {
            "context_limit": context_limit,
            "overhead_reserve": MemoryManager.OVERHEAD_TOKENS,
            "pool_usage": pool_usage,
            "total_pool_tokens": total_pool_tokens,
            "message_count": message_count,
            "session_tokens": session_tokens,
            "last_context_stats": self._last_context_stats,
            "turn_number": self._turn_number,
        }

    def get_token_usage(self) -> dict[str, dict[str, int]]:
        """Get token usage per endpoint for this session."""
        return dict(self._token_usage)

    async def compact_conversation(self, focus: str = "") -> str:
        """Compact older conversation messages to free context space.

        Summarizes older messages into a condensed form while keeping recent
        messages intact. Returns a status message describing what was compacted.
        """
        if not self.session_manager:
            return "No active session."

        messages = self.session_manager.get_messages()
        if len(messages) < 6:
            return "Too few messages to compact (need at least 6)."

        # Keep the most recent 4 messages untouched
        keep_recent = 4
        older = messages[:-keep_recent]
        recent = messages[-keep_recent:]

        # Build text from older messages for summarization
        older_text = []
        for msg in older:
            prefix = msg.role.value
            older_text.append(f"{prefix}: {msg.content[:500]}")
        text_to_summarize = "\n".join(older_text)

        focus_hint = f" Focus on: {focus}" if focus else ""
        summary_prompt = (
            f"Summarize the following conversation history into a concise summary "
            f"(max 300 words) that preserves key facts, decisions, and context.{focus_hint}\n\n"
            f"{text_to_summarize}"
        )

        # Use the summarization model to create a compact summary
        try:
            summary = await self.router.generate(
                TaskType.SUMMARIZATION, summary_prompt,
            )
        except Exception as e:
            return f"Compaction failed (LLM error): {e}"

        # Replace older messages with a single system summary message
        compacted_msg = SessionMessage(
            role=MessageRole.SYSTEM,
            content=f"[Compacted conversation summary]\n{summary}",
            timestamp=recent[0].timestamp if recent else messages[0].timestamp,
            token_count=estimate_tokens(summary),
        )

        # Replace the session manager's message list
        self.session_manager._messages = [compacted_msg] + list(recent)
        # Mark compacted indices as dumped (they were already processed)
        self.session_manager._dumped_indices = {0}

        old_count = len(older)
        old_tokens = sum(getattr(m, "token_count", 0) or 0 for m in older)
        new_tokens = estimate_tokens(summary)
        return (
            f"Compacted {old_count} older messages ({old_tokens:,} tokens) "
            f"into a summary ({new_tokens:,} tokens). "
            f"Saved ~{old_tokens - new_tokens:,} tokens."
        )
