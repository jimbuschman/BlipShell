"""Main agent loop (ports OllamaChat.SendMessageToOllama + Form1.RunChat).

Key improvement: Uses native Ollama tool calling instead of parsing
tool calls from markdown code blocks.

Extended with:
- Task planner + executor (Phase 1): complex messages get decomposed
- Background task manager (Phase 2): async long-running tasks
- Workflow system (Phase 4): named reusable templates
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import AsyncIterator, Callable, Optional

from blipshell.core.background import BackgroundTaskManager
from blipshell.core.config import ConfigManager
from blipshell.core.executor import TaskExecutor, build_executor_narrative
from blipshell.core.planner import TaskPlanner
from blipshell.core.repo_map import RepoMap
from blipshell.core.tools.base import ToolRegistry
from blipshell.core.tools.code_tools import GlobTool, GrepTool
from blipshell.core.tools.git_tools import (
    GitAddTool, GitCommitTool, GitDiffTool, GitStatusTool,
)
from blipshell.core.tools.filesystem import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from blipshell.core.tools.project_tools import CreateProjectTool
from blipshell.core.tools.memory_tools import (
    ListSessionsTool,
    PromoteToCoreMemoryTool,
    SaveCoreMemoryTool,
    SearchMemoriesTool,
)
from blipshell.core.tools.interaction_tools import AskUserTool, TaskCompleteTool
from blipshell.core.tools.shell import ShellTool
from blipshell.core.tools.task_tools import (
    CheckBackgroundTaskTool,
    ListBackgroundTasksTool,
    RunWorkflowTool,
    StartBackgroundTaskTool,
)
from blipshell.core.tools.web import WebFetchTool, WebSearchTool
from blipshell.core.workflows import WorkflowExecutor, WorkflowRegistry
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.exceptions import is_model_error
from blipshell.llm.job_queue import LLMJobQueue
from blipshell.llm.model_settings import ModelSettingsRegistry
from blipshell.llm.prompts import reflect_on_response, summarize_session_chunk
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.consolidation import MemoryConsolidator
from blipshell.memory.manager import MemoryManager, PoolItem, estimate_tokens
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.query_profiles import classify_query, compute_pool_budgets
from blipshell.memory.search import MemorySearch
from blipshell.memory.tag_discovery import TagDiscovery
from blipshell.memory.tagger import register_topic_patterns
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import BlipShellConfig, get_ollama_url
from blipshell.models.session import MessageRole, SessionMessage
from blipshell.session.manager import SessionManager

logger = logging.getLogger(__name__)


class Agent:
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

    def _register_tools(self):
        """Register all tools with their group for selective inclusion."""
        cfg = self.config.tools

        # Filesystem group
        self.tool_registry.register(ReadFileTool(
            max_file_size=cfg.filesystem.max_file_size,
            blocked_paths=cfg.filesystem.blocked_paths,
            files_read=self._files_read,
        ), group="filesystem")
        self.tool_registry.register(WriteFileTool(
            blocked_paths=cfg.filesystem.blocked_paths,
        ), group="filesystem")
        self.tool_registry.register(EditFileTool(), group="filesystem")
        self.tool_registry.register(ListDirectoryTool(), group="filesystem")

        # Shell group
        self.tool_registry.register(ShellTool(
            timeout=cfg.shell.timeout,
            allowed_commands=cfg.shell.allowed_commands,
        ), group="shell")

        # Web group
        self.tool_registry.register(WebSearchTool(), group="web")
        self.tool_registry.register(WebFetchTool(
            max_size=cfg.web.max_fetch_size,
            timeout=cfg.web.timeout,
        ), group="web")

    def _register_memory_tools(self):
        """Register memory tools (needs session_id, so called after session start)."""
        session_id = self.session_manager.session_id if self.session_manager else None

        self.tool_registry.register(SearchMemoriesTool(self.search, session_id), group="memory")
        self.tool_registry.register(SaveCoreMemoryTool(self.processor, session_id), group="memory")
        self.tool_registry.register(PromoteToCoreMemoryTool(
            self.sqlite, self.processor, session_id,
        ), group="memory")
        self.tool_registry.register(ListSessionsTool(self.sqlite), group="memory")
        self.tool_registry.register(CreateProjectTool(self.sqlite), group="general")

    def _register_task_tools(self):
        """Register background task and workflow tools (needs session_id)."""
        session_id = self.session_manager.session_id if self.session_manager else None

        self.tool_registry.register(StartBackgroundTaskTool(
            self.background_manager, session_id,
        ), group="tasks")
        self.tool_registry.register(CheckBackgroundTaskTool(
            self.background_manager,
        ), group="tasks")
        self.tool_registry.register(ListBackgroundTasksTool(
            self.background_manager, session_id,
        ), group="tasks")

        if self.workflow_executor:
            self.tool_registry.register(RunWorkflowTool(
                self.workflow_executor, session_id,
            ), group="tasks")

    def set_ask_user_callback(self, callback):
        """Set the callback for ask_user tool (wired by CLI)."""
        self._ask_user_callback = callback

    async def activate_project(self, name: str) -> dict:
        """Activate a project by name. Loads context and re-registers tools.

        Returns the project dict from the DB.
        Raises KeyError if project not found.
        """
        project = await self.sqlite.get_project(name)
        if not project:
            raise KeyError(f"Project '{name}' not found")

        # Dump memory if switching from another project (preserve conversation)
        if self.active_project and self.active_project["name"] != name:
            if self.session_manager:
                await self.session_manager.dump_to_memory()

        self.active_project = project
        root = project.get("root_path")

        # Re-register file tools with project root
        self._register_tools_with_root(root)

        # Register coding tools
        self.tool_registry.register(GrepTool(root_path=root), group="coding")
        self.tool_registry.register(GlobTool(root_path=root), group="coding")

        # Register git tools
        self.tool_registry.register(GitStatusTool(root_path=root), group="coding")
        self.tool_registry.register(GitDiffTool(root_path=root), group="coding")
        self.tool_registry.register(GitAddTool(root_path=root), group="coding")
        self.tool_registry.register(GitCommitTool(root_path=root), group="coding")

        # Register interaction tools for execution
        self.tool_registry.register(
            AskUserTool(callback=self._ask_user_callback), group="general",
        )
        self.tool_registry.register(TaskCompleteTool(), group="general")

        # Initialize repo map for code structure context
        self._repo_map = RepoMap(root)

        # Tool rules disabled (CC approach: model picks tools freely)

        # Tag current session with this project
        if self.session_manager and self.session_manager.session_id:
            await self.sqlite.update_session_project(
                self.session_manager.session_id, name,
            )
            self.session_manager.project = name

        # Touch last_active
        await self.sqlite.touch_project(name)

        # Load project context — use cache if fresh (< 1 hour)
        settings = json.loads(project.get("settings_json") or "{}")
        cached = settings.get("project_context")
        cached_at = settings.get("project_context_cached_at", 0)

        if cached and (time.time() - cached_at) < 3600:
            self._project_context = cached
            logger.info("Using cached project context for '%s'", name)
        else:
            self._project_context = await self._scan_project_context(project)
            settings["project_context"] = self._project_context
            settings["project_context_cached_at"] = time.time()
            await self.sqlite.update_project(
                name, settings_json=json.dumps(settings),
            )

        # Sync executor with project state
        if self.task_executor:
            self.task_executor.active_project = self.active_project
            self.task_executor.project_context = self._project_context
            self.task_executor.files_read = self._files_read

        logger.info("Activated project '%s' at %s", name, root)
        return project

    async def deactivate_project(self):
        """Deactivate the current project, reset tools to defaults."""
        if not self.active_project:
            return

        self.active_project = None
        self._project_context = ""
        self._repo_map = None
        # Re-register file tools without root
        self._register_tools_with_root(None)

        # Remove coding and git tools
        self.tool_registry.unregister("grep_files")
        self.tool_registry.unregister("glob_files")
        self.tool_registry.unregister("git_status")
        self.tool_registry.unregister("git_diff")
        self.tool_registry.unregister("git_add")
        self.tool_registry.unregister("git_commit")
        self.tool_registry.unregister("ask_user")
        self.tool_registry.unregister("task_complete")

        # Clear executor project state
        if self.task_executor:
            self.task_executor.active_project = None
            self.task_executor.project_context = ""

        logger.info("Deactivated project")

    def _register_tools_with_root(self, root_path: str | None):
        """Re-register file and shell tools with a root_path (or None to reset)."""
        cfg = self.config.tools

        # Unregister existing file/shell tools and re-register with root_path
        for name in ("read_file", "write_file", "edit_file", "list_directory", "run_command"):
            self.tool_registry.unregister(name)

        self.tool_registry.register(ReadFileTool(
            max_file_size=cfg.filesystem.max_file_size,
            blocked_paths=cfg.filesystem.blocked_paths,
            root_path=root_path,
            files_read=self._files_read,
        ), group="filesystem")
        self.tool_registry.register(WriteFileTool(
            blocked_paths=cfg.filesystem.blocked_paths,
            root_path=root_path,
        ), group="filesystem")
        self.tool_registry.register(EditFileTool(root_path=root_path), group="filesystem")
        self.tool_registry.register(ListDirectoryTool(root_path=root_path), group="filesystem")
        self.tool_registry.register(ShellTool(
            timeout=cfg.shell.timeout,
            allowed_commands=cfg.shell.allowed_commands,
            cwd=root_path,
        ), group="shell")

    async def _scan_project_context(self, project: dict) -> str:
        """Scan a project directory and build a context string for the LLM."""
        import subprocess
        from pathlib import Path

        root = project.get("root_path")
        if not root or not Path(root).is_dir():
            return f"Project: {project['name']}\nRoot path not accessible."

        root_path = Path(root)
        parts = [
            f"Project: {project['name']}",
            f"Root: {root}",
        ]
        if project.get("description"):
            parts.append(f"Description: {project['description']}")
        if project.get("language"):
            parts.append(f"Language: {project['language']}")
        if project.get("git_url"):
            parts.append(f"Git: {project['git_url']}")

        # Git info
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root, capture_output=True, text=True, timeout=5,
            )
            if branch.returncode == 0:
                parts.append(f"Branch: {branch.stdout.strip()}")

            log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=root, capture_output=True, text=True, timeout=5,
            )
            if log.returncode == 0 and log.stdout.strip():
                parts.append(f"\nRecent commits:\n{log.stdout.strip()}")
        except Exception:
            pass

        # Code map: AST-based structure of Python files (replaces file tree)
        if self._repo_map:
            code_map = self._repo_map.build(max_lines=120)
            if code_map:
                parts.append(f"\nCode structure (classes, functions):\n{code_map}")

        # Compact file tree (top level only, for non-Python files/dirs)
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                     ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
                     ".vs", ".idea", ".vscode", "backups"}
        tree_lines = []
        for entry in sorted(root_path.iterdir()):
            if entry.name in skip_dirs:
                continue
            prefix = "[DIR] " if entry.is_dir() else "      "
            tree_lines.append(f"  {prefix}{entry.name}")
        if tree_lines:
            parts.append(f"\nTop-level layout:\n" + "\n".join(tree_lines[:40]))

        # BLIPSHELL.md — project-level instructions (loaded in full, like CLAUDE.md)
        blipshell_md = root_path / "BLIPSHELL.md"
        if blipshell_md.is_file():
            try:
                content = blipshell_md.read_text(encoding="utf-8", errors="replace")
                parts.append(f"\n=== BLIPSHELL.md (project instructions) ===\n{content}")
                logger.info("Loaded BLIPSHELL.md from %s (%d chars)", root, len(content))
            except Exception:
                pass

        # Key files
        key_files = ["README.md", "README.rst", "README.txt", "readme.md",
                     "pyproject.toml", "setup.py", "setup.cfg",
                     "package.json", "Cargo.toml", "go.mod",
                     "requirements.txt", "Makefile", "CLAUDE.md"]
        for fname in key_files:
            fpath = root_path / fname
            if fpath.is_file():
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    lines = content.splitlines()[:60]
                    truncated = "\n".join(lines)
                    if len(content.splitlines()) > 60:
                        truncated += "\n... (truncated)"
                    parts.append(f"\n=== {fname} ===\n{truncated}")
                except Exception:
                    pass

        return "\n".join(parts)

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

        # Load core memories into Core pool
        await self._load_core_memories()

        # Load lessons into Core pool
        await self._load_lessons()

        # Load recent session summaries into RecentHistory
        await self._load_recent_sessions()

        return session_id

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
                except Exception:
                    pass
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

        # Unprocessed message sweep — enqueue each as PROCESS_MESSAGE
        try:
            unprocessed = await self.sqlite.get_unprocessed_messages(limit=50)
            if unprocessed:
                logger.info(
                    "Enqueueing %d unprocessed messages for background processing",
                    len(unprocessed),
                )
                for msg in unprocessed:
                    self._memory_worker.enqueue(WorkItem(
                        work_type=WorkType.PROCESS_MESSAGE,
                        text=msg["content"],
                        role=msg["role"],
                        session_id=msg["session_id"],
                        message_db_id=msg["id"],
                    ))
        except Exception as e:
            logger.warning("Failed to enqueue unprocessed messages: %s", e)

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
                self.chroma.upsert_entities_batch(ids, names, types)

            await self.sqlite.set_metadata("entity_embeddings_backfilled", "1")
            logger.info("Backfilled %d entity embeddings into ChromaDB", total)
        except Exception as e:
            logger.error("Entity embedding backfill failed: %s", e)

    async def _load_recent_sessions(self):
        """Load recent session summaries into RecentHistory pool."""
        sessions = await self.sqlite.list_sessions(limit=3)
        current_id = self.session_manager.session_id
        for s in sessions:
            if s.id == current_id or not s.summary:
                continue
            self.memory_manager.add_memory("RecentHistory", PoolItem(
                text=s.summary,
                session_role="system",
                priority_score=2.0,
                session_id=s.id,
            ))

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

    async def chat(
        self,
        user_message: str,
        on_token: Optional[Callable[[str], None]] = None,
        force_plan: bool = False,
    ) -> str:
        """Process a user message through the full agent pipeline.

        Routes between simple chat and planned execution based on
        complexity classification.

        Args:
            user_message: The user's input
            on_token: Optional callback for streaming tokens
            force_plan: If True, skip classification and go straight to planning

        Returns:
            The assistant's complete response
        """
        # Add user message to session
        self.session_manager.add_message(MessageRole.USER, user_message)

        # Decide execution path — only force_plan triggers the executor.
        # The model handles complexity naturally through the unified chat loop.
        needs_planning = force_plan

        # Event: turn_start
        self._turn_number += 1
        session_id = self.session_manager.session_id
        await self._log_event("turn_start", {
            "query_length": len(user_message),
            "route": "planned" if needs_planning else "simple",
        })

        if needs_planning:
            logger.info("Message classified as complex — using planned execution")
            response = await self._chat_planned(user_message, on_token=on_token)
        else:
            logger.info("Message classified as simple — using direct chat")
            response = await self._chat_simple(user_message, on_token=on_token)

        # Self-reflection: second LLM pass to catch errors/gaps
        if self.reflect_enabled and response and not response.startswith("Error:"):
            if on_token:
                on_token("\n\n\x1b[2m[Reflecting...]\x1b[0m\n\n")
            response = await self._reflect_on_response(user_message, response, on_token)

        # Add assistant response to session (skip empty — prevents cascade of
        # blank responses where an empty assistant message confuses the model)
        if response and response.strip():
            self.session_manager.add_message(MessageRole.ASSISTANT, response)

        # Background: dump to memory periodically (tracked for clean shutdown)
        task = asyncio.create_task(self._background_memory_processing())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return response

    async def _chat_simple(
        self,
        user_message: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Simple chat path — uses unified ChatLoop with endpoint fallback."""
        from blipshell.core.chat_loop import ChatLoop, LoopConfig, LoopResult

        # Search relevant memories for recall
        await self._search_relevant_memories(user_message)

        # Build message list
        messages = self._build_messages(user_message)

        # Event: context_built (stats computed in _build_messages)
        if self._last_context_stats:
            await self._log_event("context_built", self._last_context_stats)

        # Route to coding model when a project is active, otherwise tool_calling
        task_type = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING

        # Get model (with fallback if primary is known to be down)
        model = self.router.get_model(task_type)
        using_fallback = False

        if self.router.is_model_failed(model):
            fallback = self.router.get_fallback_model(task_type)
            if fallback:
                logger.info("Skipping failed model '%s', using fallback '%s'", model, fallback)
                model = fallback
                using_fallback = True

        # Always pass all tools — let the model decide what to use.
        tools = self.tool_registry.get_all_ollama_tools() or None
        max_iterations = self.config.agent.max_tool_iterations if tools else 0
        logger.info("Passing %d tools (max_iterations=%d)",
                     len(tools) if tools else 0, max_iterations)

        loop = ChatLoop(self.tool_registry, on_token)
        config = LoopConfig(
            budget=max_iterations,
            enable_dedup=True,
            auto_continue_on_exhaustion=True,
        )

        # Try primary, then fallback on error
        result = None
        endpoint_name = ""
        full_response = ""

        for attempt in range(2):  # primary + one fallback
            endpoint = await self.endpoint_manager.get_endpoint_for_role(task_type)
            if not endpoint:
                if attempt == 0 and not using_fallback:
                    fallback = self.router.get_fallback_model(task_type)
                    if fallback and fallback != model:
                        model = fallback
                        using_fallback = True
                        continue
                full_response = "Error: No available LLM endpoint."
                break

            self._last_endpoint_used = endpoint.name
            endpoint_name = endpoint.name
            client = endpoint.client

            chat_kwargs: dict = {}
            if endpoint.context_tokens:
                chat_kwargs["options"] = {"num_ctx": endpoint.context_tokens}
            if not self.think_enabled:
                chat_kwargs["think"] = False

            endpoint.start_request()
            try:
                result = await loop.run(
                    client=client,
                    messages=messages,
                    model=model,
                    tools=tools,
                    chat_kwargs=chat_kwargs,
                    config=config,
                    on_tool_executed=self._on_tool_executed,
                    on_stream_done=self._record_token_usage_from_chunk,
                )
                endpoint.record_success(0)
                full_response = result.response
                break  # Success
            except Exception as e:
                if is_model_error(e):
                    logger.warning(
                        "Model-level error on endpoint '%s' (not penalizing): %s",
                        endpoint.name, e,
                    )
                    self.router.mark_model_failed(model)
                else:
                    endpoint.record_failure()

                if attempt == 0 and not using_fallback:
                    fallback = self.router.get_fallback_model(task_type)
                    if fallback and fallback != model:
                        logger.warning("Primary model '%s' failed, falling back to '%s'", model, fallback)
                        model = fallback
                        using_fallback = True
                        if on_token:
                            on_token(f"\n\x1b[33m[Falling back to {fallback}]\x1b[0m\n")
                        continue  # Retry with fallback

                logger.error("Chat error: %s", e)
                full_response = f"Error: {e}"
                break
            finally:
                endpoint.complete_request()

        # Event: llm_complete
        await self._log_event("llm_complete", {
            "endpoint": endpoint_name,
            "model": model,
            "fallback": using_fallback,
            "tool_calls": result.tool_call_names if result else [],
            "response_length": len(full_response),
        })

        return full_response

    async def _chat_planned(
        self,
        user_message: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Planned chat path — dynamic iterative execution (no pre-generated plan).

        The LLM decides what to do next after each action, stopping when done.
        This is how Claude Code, Cursor, and modern coding agents work.

        Memory integration:
        - Searches relevant memories before execution (context IN)
        - Passes recent chat history so design discussions carry over
        - Feeds the coding result through the memory pipeline (context OUT)
        """
        if on_token:
            on_token("[Preparing context...]\n")

        # Search relevant memories for this task
        memory_context = ""
        try:
            results = await self.search.search(
                query=user_message,
                current_session_id=self.session_manager.session_id,
                n_results=10,
                active_project=self.active_project["name"] if self.active_project else None,
            )
            if results:
                memory_context = "Relevant memories from past sessions:\n"
                for r in results:
                    memory_context += f"- {r.summary}\n"
            # Log search results for /flow observability
            await self._log_event("search_complete", {
                "memory_results": len(results) if results else 0,
                "lesson_results": 0,
                "final_returned": len(results) if results else 0,
                "skipped": None if results else "no results",
            })
        except Exception as e:
            logger.error("Memory search for executor failed: %s", e)

        # Get recent chat history (design discussion context)
        chat_history = []
        for msg in self.session_manager.get_messages()[-10:]:
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                chat_history.append(msg.to_ollama_message())

        if on_token:
            on_token("[Executing task...]\n\n")

        def on_step_complete(step_num, result_summary):
            if on_token:
                on_token(f"\n[Iteration {step_num} complete]\n")

        try:
            result = await self.task_executor.execute_dynamic(
                user_message,
                on_step_complete=on_step_complete,
                on_token=on_token,
                memory_context=memory_context,
                chat_history=chat_history,
                log_event=self._log_event,
            )
        except Exception as e:
            logger.error("Dynamic execution failed: %s", e)
            # Fallback to simple chat
            if on_token:
                on_token("[Execution failed, falling back to direct chat]\n")
            return await self._chat_simple(user_message, on_token=on_token)

        # Build a clean narrative from the executor transcript and feed through memory
        try:
            narrative = build_executor_narrative(self.task_executor.last_messages)
            if narrative and narrative.strip():
                await self.session_manager.processor.process_message(
                    text=narrative,
                    role="assistant",
                    session_id=self.session_manager.session_id,
                )
        except Exception as e:
            logger.error("Failed to process coding narrative into memory: %s", e)

        return result

    async def _search_relevant_memories(self, query: str):
        """Search for relevant memories and lessons, add to Recall pool."""
        # Search conversation memories
        memory_count = 0
        try:
            active_proj = self.active_project["name"] if self.active_project else None
            results = await self.search.search(
                query=query,
                current_session_id=self.session_manager.session_id,
                n_results=10,
                active_project=active_proj,
            )
            memory_count = len(results)
            for r in results:
                self.memory_manager.add_memory("Recall", PoolItem(
                    text=r.summary,
                    session_role="system",
                    priority_score=r.boosted_score,
                ))
        except Exception as e:
            logger.error("Memory search failed: %s", e)

        # Search lessons semantically (closes the lessons loop)
        lesson_count = 0
        try:
            active_proj = self.active_project["name"] if self.active_project else None
            lesson_results = await self.search.search_lessons(
                query, n_results=5, active_project=active_proj,
            )
            for lr in lesson_results:
                similarity = lr.get("similarity", 0.0)
                if similarity < 0.4:
                    continue
                lesson_count += 1
                self.memory_manager.add_memory("Recall", PoolItem(
                    text=lr.get("document", ""),
                    session_role="system2",  # labeled as "RelevantLessons" in context
                    priority_score=similarity + 0.1,  # slight boost for lessons
                ))
        except Exception as e:
            logger.error("Lesson search failed: %s", e)

        # Event: search_complete
        search_stats = self.search.last_search_stats or {}
        await self._log_event("search_complete", {
            "memory_results": memory_count,
            "lesson_results": lesson_count,
            **search_stats,
        })

    def _build_messages(self, user_message: str) -> list[dict]:
        """Build the full message list with memory context.

        Port of OllamaChat.SendMessageToOllama message building.
        Uses dynamic context window based on the active endpoint.
        """
        user_tokens = estimate_tokens(user_message)

        # Use endpoint-specific context window if available
        # Route to coding endpoint's context window when a project is active
        context_role = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        context_limit = self.endpoint_manager.get_context_tokens_for_role(
            context_role,
            default=65536,
        )

        available = (
            context_limit
            - user_tokens
            - MemoryManager.OVERHEAD_TOKENS
        )

        # Classify query and compute dynamic pool budgets
        profile = classify_query(user_message)
        pool_budgets = compute_pool_budgets(
            profile, available, self.memory_manager.get_hard_caps(),
        )
        logger.debug("Query profile: %s", profile)

        # Gather memory from all pools with dynamic budgets
        memory_items = self.memory_manager.gather_memory(
            token_budget=available, pool_budgets=pool_budgets,
        )

        # Build memory context string organized by pool
        # Order: Core first (stable facts, LLM attends to start), history in
        # the middle (lowest attention), Recall last (most relevant, LLM
        # attends to end) — mitigates lost-in-the-middle effect.
        pool_labels = {
            "Core": "CoreFoundation",
            "Lessons": "RelevantLessons",
            "Recall": "RelevantMemory",
            "RecentHistory": "RecentHistory",
            "Buffer": "RecentHistory",
            "ActiveSession": "ActiveSession",
        }
        pool_order = ["Core", "Lessons", "RecentHistory", "Buffer", "ActiveSession", "Recall"]
        context_parts: dict[str, list[str]] = {}
        for item in memory_items:
            pool = item.pool_name
            if pool not in context_parts:
                context_parts[pool] = []
            context_parts[pool].append(f"   - {item.text}")

        # Compute context stats for observability
        pool_usage = {}
        for item in memory_items:
            p = item.pool_name
            if p not in pool_usage:
                pool_usage[p] = {"items": 0, "tokens": 0}
            pool_usage[p]["items"] += 1
            pool_usage[p]["tokens"] += item.estimated_tokens
        self._last_context_stats = {
            "query_profile": profile,
            "context_limit": context_limit,
            "available_tokens": available,
            "pool_budgets": pool_budgets,
            "pool_usage": pool_usage,
            "total_context_items": len(memory_items),
        }

        memory_text = ""
        for pool_name in pool_order:
            if pool_name not in context_parts:
                continue
            label = pool_labels.get(pool_name, pool_name)
            memory_text += f"{label}:\n" + "\n".join(context_parts[pool_name]) + "\n\n"
        # Include any pools not in the explicit order (future-proofing)
        for pool_name, items in context_parts.items():
            if pool_name not in pool_order:
                label = pool_labels.get(pool_name, pool_name)
                memory_text += f"{label}:\n" + "\n".join(items) + "\n\n"

        # Build messages
        system_prompt = self.config.agent.system_prompt

        # Get per-model settings for the active model
        task_type_for_model = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        active_model = self.router.get_model(task_type_for_model)
        ms = self.model_settings.get(active_model)

        if self.active_project and self._project_context:
            root_path = self.active_project.get("root_path", "")
            tool_limit = ms.max_tool_calls

            system_prompt += (
                "\n\n--- PROJECT CONTEXT ---\n"
                f"Project: \"{self.active_project['name']}\" at {root_path}\n"
                "File tools resolve relative paths against this root.\n\n"
                "Answer questions conversationally. Only use tools when the user asks you "
                "to create, modify, find, or run something.\n\n"
                "Examples:\n"
                '- "How does search work?" -> Answer from context. No tools.\n'
                '- "Create hello.py" -> Use write_file.\n'
                '- "Fix the bug in search.py" -> read_file, then edit_file.\n\n'
                f"Target under {tool_limit} tool calls. Do not explore endlessly.\n\n"
                "# Scratchpad\n"
                f"Project: data/scratchpad_{self.active_project['name']}.md | "
                "General: data/scratchpad.md\n"
                "For decisions, plans, and TODOs that should survive across sessions.\n\n"
            )

            # Add model-specific extra instructions
            if ms.extra_instructions:
                system_prompt += f"MODEL-SPECIFIC INSTRUCTIONS:\n{ms.extra_instructions}\n\n"

            system_prompt += self._project_context

        # Consolidate all context into a single system message (CC approach)
        scratchpad = self._read_scratchpad()
        if scratchpad:
            system_prompt += f"\n\n--- SCRATCHPAD ---\n{scratchpad}"

        if memory_text.strip():
            system_prompt += f"\n\n{memory_text}"

        if self._files_read:
            files_list = "\n".join(f"  - {f}" for f in sorted(self._files_read))
            system_prompt += (
                "\n\nFILES ALREADY READ THIS SESSION (do NOT re-read these):\n"
                + files_list
            )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history from ActiveSession (last messages)
        for msg in self.session_manager.get_messages()[-20:]:
            messages.append(msg.to_ollama_message())

        return messages

    def _read_scratchpad(self) -> str:
        """Read scratchpad files (general + project-specific) for context injection.

        Returns combined scratchpad content, or empty string if none exist.
        """
        parts = []
        # Project-specific scratchpad
        if self.active_project:
            proj_path = os.path.join("data", f"scratchpad_{self.active_project['name']}.md")
            if os.path.exists(proj_path):
                try:
                    with open(proj_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        parts.append(f"[Project: {self.active_project['name']}]\n{content}")
                except Exception:
                    pass
        # General scratchpad
        general_path = os.path.join("data", "scratchpad.md")
        if os.path.exists(general_path):
            try:
                with open(general_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    parts.append(f"[General]\n{content}")
            except Exception:
                pass
        return "\n\n".join(parts)

    async def _log_event(self, event_type: str, data: dict):
        """Log a conversation flow event. Fire-and-forget safe."""
        try:
            session_id = self.session_manager.session_id if self.session_manager else 0
            await self.sqlite.log_turn_event(
                session_id, self._turn_number, event_type, data,
            )
        except Exception as e:
            logger.debug("Failed to log event %s: %s", event_type, e)

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
        # 2. Close ChromaDB before SQLite (ChromaDB may reference SQLite data)
        if self.chroma:
            try:
                self.chroma.close()
            except Exception:
                pass
        # 3. Close SQLite last
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
