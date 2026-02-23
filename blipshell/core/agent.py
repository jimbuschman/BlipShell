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
import time
from datetime import datetime
from typing import AsyncIterator, Callable, Optional

from blipshell.core.background import BackgroundTaskManager
from blipshell.core.config import ConfigManager
from blipshell.core.executor import TaskExecutor, build_executor_narrative
from blipshell.core.planner import ComplexityClassifier, TaskPlanner
from blipshell.core.repo_map import RepoMap
from blipshell.core.tool_rules import ToolRuleEngine, create_coding_rules, create_default_rules
from blipshell.core.tools.base import ToolRegistry, detect_tool_groups
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
from blipshell.core.tools.interaction_tools import AskUserTool
from blipshell.core.tools.shell import ShellTool
from blipshell.core.tools.task_tools import (
    CheckBackgroundTaskTool,
    ListBackgroundTasksTool,
    RunWorkflowTool,
    StartBackgroundTaskTool,
)
from blipshell.core.tools.web import WebFetchTool, WebSearchTool
from blipshell.core.workflows import WorkflowExecutor, WorkflowRegistry
from blipshell.llm.client import LLMClient
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.exceptions import is_model_error
from blipshell.llm.job_queue import LLMJobQueue
from blipshell.llm.model_settings import ModelSettingsRegistry
from blipshell.llm.prompts import reflect_on_response, summarize_session_chunk
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.consolidation import MemoryConsolidator
from blipshell.memory.entity_extractor import EntityExtractor
from blipshell.memory.manager import MemoryManager, PoolItem, estimate_tokens
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.query_profiles import classify_query, compute_pool_budgets
from blipshell.memory.search import MemorySearch
from blipshell.memory.tag_discovery import TagDiscovery
from blipshell.memory.tagger import register_topic_patterns
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import BlipShellConfig, get_ollama_url
from blipshell.models.session import MessageRole
from blipshell.models.tools import ToolCall
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
        self._tool_rules: ToolRuleEngine = create_default_rules()

        # Task planning + execution (Phase 1)
        self.complexity_classifier: Optional[ComplexityClassifier] = None
        self.task_planner: Optional[TaskPlanner] = None
        self.task_executor: Optional[TaskExecutor] = None

        # Background tasks (Phase 2)
        self.background_manager: Optional[BackgroundTaskManager] = None

        # Workflows (Phase 4)
        self.workflow_registry: Optional[WorkflowRegistry] = None
        self.workflow_executor: Optional[WorkflowExecutor] = None

        self._health_check_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()
        self._last_endpoint_used: Optional[str] = None
        self._initialized = False
        self.think_enabled: bool = False  # /think toggle — off for fast simple chat, complex auto-enables
        self.reflect_enabled: bool = False  # /reflect toggle — second-pass self-critique
        self._turn_number: int = 0
        self._last_context_stats: Optional[dict] = None

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
        self.router = LLMRouter(self.config.models, self.endpoint_manager)

        # Job queue
        self.job_queue = LLMJobQueue()
        self.job_queue.start()

        # Memory manager
        self.memory_manager = MemoryManager(self.config.memory)
        self.memory_manager.set_summarize_callback(self._summarize_overflow)

        # Processor
        self.processor = MemoryProcessor(self.sqlite, self.chroma, self.router,
                                         config=self.config.memory)

        # Search
        self.search = MemorySearch(
            self.sqlite, self.chroma, self.router,
            config=self.config.memory,
        )

        # Session manager
        self.session_manager = SessionManager(
            self.sqlite, self.memory_manager, self.processor, self.router,
            summary_chunk_size=self.config.session.summary_chunk_size,
        )

        # Task planner + executor (Phase 1)
        self.complexity_classifier = ComplexityClassifier(self.config.planner)
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

        _status("Extracting entities...")
        await self._auto_extract_entities()

        _status("Backfilling entity embeddings...")
        await self._backfill_entity_embeddings()

        # Start periodic health check (re-detects endpoints that come/go)
        self._health_check_task = self.endpoint_manager.start_health_loop(
            interval=60, on_check=self.router.clear_failed_models,
        )

        # Process any messages that failed during previous session close
        _status("Checking for unprocessed messages...")
        await self._sweep_unprocessed_messages()

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

    async def _sweep_unprocessed_messages(self):
        """Reprocess messages that failed during previous session close.

        Scans session_messages for is_processed=False rows and runs them
        through the memory pipeline. Non-fatal — logs and skips on error.
        """
        try:
            unprocessed = await self.sqlite.get_unprocessed_messages(limit=50)
            if not unprocessed:
                return

            logger.info("Found %d unprocessed messages, reprocessing...", len(unprocessed))
            processed = 0
            for msg in unprocessed:
                try:
                    await self.processor.process_message(
                        text=msg["content"],
                        role=msg["role"],
                        session_id=msg["session_id"],
                    )
                    await self.sqlite.mark_message_processed(msg["id"])
                    processed += 1
                except Exception as e:
                    logger.warning(
                        "Sweep: message %d failed: %s", msg["id"], e,
                    )
            logger.info("Sweep complete: %d/%d processed", processed, len(unprocessed))
        except Exception as e:
            logger.warning("Startup sweep error (non-fatal): %s", e)

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

        # Register ask_user tool for interactive clarification during execution
        self.tool_registry.register(
            AskUserTool(callback=self._ask_user_callback), group="general",
        )

        # Initialize repo map for code structure context
        self._repo_map = RepoMap(root)

        # Switch to coding-mode tool rules (more permissive)
        self._tool_rules = create_coding_rules()

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

        # Sync planner + executor with project state so plan generation
        # and step execution use the correct model/context
        if self.task_planner:
            self.task_planner.active_project = self.active_project
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
        self._tool_rules = create_default_rules()

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

        # Clear planner + executor project state
        if self.task_planner:
            self.task_planner.active_project = None
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

    async def _auto_extract_entities(self):
        """Extract entity relationship triples from unprocessed memories on startup."""
        try:
            er_config = self.config.memory.entity_resolution
            extractor = EntityExtractor(
                self.sqlite, self.router,
                chroma=self.chroma,
                batch_size=self.config.memory.entity_extraction_batch_size,
                entity_resolution_enabled=er_config.enabled,
                entity_auto_merge_threshold=er_config.embedding_auto_merge_threshold,
                entity_llm_threshold=er_config.llm_arbitration_threshold,
                entity_max_candidates=er_config.max_candidates,
            )
            stats = await extractor.extract_batch()
            if stats["triples"] > 0:
                logger.info(
                    "Extracted %d entity triples from %d memories",
                    stats["triples"], stats["extracted"],
                )
            if stats["errors"] > 0:
                logger.warning(
                    "Entity extraction had %d errors", stats["errors"],
                )
        except Exception as e:
            logger.error("Entity extraction failed: %s", e)

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

    @staticmethod
    def _extract_response(response) -> tuple[str, list | None]:
        """Extract content and tool_calls from an Ollama response.

        Handles both dict responses (old ollama) and object responses (ollama 0.4+).
        """
        # Try object attribute access first (ollama 0.4+)
        msg = getattr(response, "message", None)
        if msg is not None:
            content = getattr(msg, "content", "") or ""
            tool_calls = getattr(msg, "tool_calls", None)
            return content, tool_calls

        # Fallback to dict access (older ollama)
        if isinstance(response, dict):
            msg = response.get("message", {})
            return msg.get("content", ""), msg.get("tool_calls", None)

        return "", None

    @staticmethod
    def _extract_tool_call_info(tc) -> tuple[str, dict, str]:
        """Extract name, arguments, and id from a tool call object or dict.

        Returns (name, arguments, tool_call_id). Handles both Ollama (args
        as dict) and OpenAI-compatible APIs (args as JSON string).
        """
        # Object access (ollama 0.4+)
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", "") or ""
            args = getattr(fn, "arguments", {}) or {}
            tc_id = getattr(tc, "id", "") or ""
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            return name, args, tc_id

        # Dict access
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            tc_id = tc.get("id", "")
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            return fn.get("name", ""), args, tc_id

        return "", {}, ""

    @staticmethod
    def _format_tool_arg_hint(tool_call: ToolCall) -> str:
        """Format a short argument hint for tool call display."""
        args = tool_call.arguments
        if not args:
            return ""
        if "pattern" in args:
            return f" {args['pattern'][:50]}"
        if "path" in args:
            return f" {args['path']}"
        if "command" in args:
            return f" {args['command'][:60]}"
        if "query" in args:
            return f" {args['query'][:50]}"
        if "message" in args:
            return f" {args['message'][:60]}"
        if "paths" in args:
            return f" {args['paths'][:60]}"
        return ""

    @staticmethod
    def _should_auto_continue(text: str) -> bool:
        """Detect if the LLM stopped but clearly intends to continue.

        Catches two patterns:
        1. Permission-asking: "Should I proceed?", "Want me to...", etc.
        2. Continuation-intent: "Now let me verify...", "Next I'll...", etc.
           where the LLM narrates its next action but doesn't actually do it.

        Returns True if the LLM should be nudged to continue.

        NOTE: Only called when tool_call_names is non-empty (model already
        used tools this turn), so these patterns indicate mid-task pausing,
        not normal conversational questions.
        """
        text_lower = text.lower().strip()

        # Pattern 1: asking permission (question in last 200 chars)
        # Tightened: only match explicit "should I do X" patterns, not
        # generic questions like "what do you think?" or "let me know"
        if "?" in text_lower[-200:]:
            tail_200 = text_lower[-200:]
            permission_patterns = [
                "should i proceed", "should i continue",
                "should i go ahead", "should i start",
                "want me to proceed", "want me to continue",
                "shall i proceed", "shall i continue",
                "would you like me to proceed", "would you like me to continue",
                "ready to proceed", "ready to continue",
            ]
            if any(p in tail_200 for p in permission_patterns):
                return True

        # Pattern 2: continuation intent — the LLM says it's going to do
        # something but stopped without actually calling a tool.
        # Tightened: only match "let me [verb]" and "next i'll [verb]"
        # patterns, not generic "i need to" or "need to add" which can
        # appear in normal explanations.
        tail = text_lower[-300:]
        continuation_patterns = [
            "now let me ", "let me also ", "let me check",
            "let me fix", "let me update", "let me verify",
            "let me read", "let me look", "next i'll",
            "next i need to", "i'll also ",
        ]
        if any(p in tail for p in continuation_patterns):
            return True

        return False

    async def _continue_tool_loop(
        self,
        messages: list[dict],
        client,
        model: str,
        tools,
        chat_kwargs: dict,
        initial_content: str,
        initial_tool_calls: list,
        tool_call_names: list[str],
        remaining_iterations: int,
        on_token=None,
        task_type=None,
    ) -> str:
        """Continue the tool call loop after auto-nudge triggered new tool calls."""
        # Process the initial tool calls from the nudge response
        messages.append({"role": "assistant", "content": initial_content,
                        "tool_calls": initial_tool_calls})

        for tc in initial_tool_calls:
            name, arguments, tc_id = self._extract_tool_call_info(tc)
            tool_call_names.append(name)
            tool_call = ToolCall(id=tc_id, name=name, arguments=arguments)

            if on_token:
                arg_hint = self._format_tool_arg_hint(tool_call)
                on_token(f"\n\x1b[36m\x1b[1m[Tool: {tool_call.name}{arg_hint}]\x1b[0m\n")

            result = await self.tool_registry.execute_tool_call(tool_call)
            result.tool_call_id = tc_id
            messages.append(result.to_ollama_message())

            if result.success and name in ("read_file", "list_directory"):
                read_path = arguments.get("path", "")
                if read_path:
                    self._files_read.add(read_path)
            if result.success and name in ("write_file", "edit_file"):
                file_path = arguments.get("path", "")
                self._file_changes.append({
                    "path": file_path, "tool": name,
                    "turn_number": self._turn_number,
                })

            if on_token:
                preview = result.result[:120].replace("\n", " ")
                if result.success:
                    on_token(f"\x1b[2m[{preview}]\x1b[0m\n\n")
                else:
                    on_token(f"\x1b[31m[{preview}]\x1b[0m\n\n")

        # Continue the main loop for remaining iterations
        full_response = ""
        for iteration in range(max(remaining_iterations, 5)):
            try:
                endpoint = await self.endpoint_manager.get_endpoint_for_role(task_type)
                if endpoint:
                    endpoint.start_request()

                # Apply tool rules in continue loop too
                cont_tools = None
                if tools and iteration < max(remaining_iterations, 5) - 1:
                    cont_tools = self._tool_rules.filter_tools(tools, tool_call_names)
                    if not cont_tools:
                        cont_tools = None

                response = await client.chat(
                    messages=messages, model=model,
                    tools=cont_tools,
                    **chat_kwargs,
                )

                content, new_tc = self._extract_response(response)

                if new_tc and iteration < max(remaining_iterations, 5) - 1:
                    messages.append({"role": "assistant", "content": content,
                                    "tool_calls": new_tc})
                    for tc in new_tc:
                        name, arguments, tc_id = self._extract_tool_call_info(tc)
                        tool_call_names.append(name)
                        tool_call = ToolCall(id=tc_id, name=name, arguments=arguments)

                        if on_token:
                            arg_hint = self._format_tool_arg_hint(tool_call)
                            on_token(f"\n\x1b[36m\x1b[1m[Tool: {tool_call.name}{arg_hint}]\x1b[0m\n")

                        result = await self.tool_registry.execute_tool_call(tool_call)
                        result.tool_call_id = tc_id
                        messages.append(result.to_ollama_message())

                        if result.success and name in ("read_file", "list_directory"):
                            read_path = arguments.get("path", "")
                            if read_path:
                                self._files_read.add(read_path)
                        if result.success and name in ("write_file", "edit_file"):
                            file_path = arguments.get("path", "")
                            self._file_changes.append({
                                "path": file_path, "tool": name,
                                "turn_number": self._turn_number,
                            })

                        if on_token:
                            preview = result.result[:120].replace("\n", " ")
                            if result.success:
                                on_token(f"\x1b[2m[{preview}]\x1b[0m\n\n")
                            else:
                                on_token(f"\x1b[31m[{preview}]\x1b[0m\n\n")

                    if endpoint:
                        endpoint.record_success(0)
                    continue
                else:
                    full_response = content
                    if on_token and content:
                        on_token(content)
                    if endpoint:
                        endpoint.record_success(0)
                    break
            except Exception as e:
                logger.error("Continue tool loop error: %s", e)
                full_response = f"Error: {e}"
                break
            finally:
                if endpoint:
                    endpoint.complete_request()

        if not full_response:
            full_response = "[Completed tool calls — no final summary generated]"

        return full_response

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

        # Decide execution path
        needs_planning = force_plan or self.complexity_classifier.needs_planning(user_message)

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

        # Add assistant response to session
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
        """Simple chat path — existing flat tool-calling loop."""
        # Search relevant memories for recall
        await self._search_relevant_memories(user_message)

        # Build message list
        messages = self._build_messages(user_message)

        # Event: context_built (stats computed in _build_messages)
        if self._last_context_stats:
            await self._log_event("context_built", self._last_context_stats)

        # Route to coding model when a project is active, otherwise tool_calling
        task_type = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING

        # Get model and client (with fallback if cloud is down)
        model = self.router.get_model(task_type)
        client = await self.router.get_client(task_type)
        using_fallback = False

        # Skip straight to fallback if primary model is known to be down
        if self.router.is_model_failed(model):
            fallback = self.router.get_fallback_model(task_type)
            if fallback:
                logger.info("Skipping failed model '%s', using fallback '%s'", model, fallback)
                model = fallback
                using_fallback = True

        if not client:
            # Try fallback model
            fallback = self.router.get_fallback_model(task_type)
            if fallback:
                logger.warning("Primary endpoint down, using fallback model '%s'", fallback)
                model = fallback
                using_fallback = True
                client = await self.router.get_client(task_type)
            if not client:
                return "Error: No available LLM endpoint."

        # Tool gating: when a project is active, pass all tools (coding mode).
        # Otherwise, detect which tool groups the message needs and only pass those.
        # Pure conversation (no groups detected) → no tools → faster response.
        ms = self.model_settings.get(model)
        if self.active_project:
            tools = self.tool_registry.get_all_ollama_tools() or None
        else:
            needed_groups = detect_tool_groups(user_message)
            if needed_groups:
                # Always include memory tools when any group is detected
                needed_groups.add("memory")
                needed_groups.add("general")
                needed_groups.add("tasks")
                tools = self.tool_registry.get_tools_for_groups(needed_groups) or None
            else:
                tools = None
        # Use per-model tool call limit if configured
        max_iterations = self.config.agent.max_tool_iterations if tools else 0
        if self.active_project and tools:
            max_iterations = max(max_iterations, ms.max_tool_calls)
        logger.info("Passing %d tools to model (groups=%s, max_iterations=%d)",
                     len(tools) if tools else 0,
                     "all" if self.active_project else (needed_groups if tools else "none"),
                     max_iterations)
        full_response = ""
        tool_call_names: list[str] = []
        endpoint_name = ""

        for iteration in range(max_iterations + 1):
            endpoint = None
            try:
                endpoint = await self.endpoint_manager.get_endpoint_for_role(task_type)
                if endpoint:
                    endpoint.start_request()
                    self._last_endpoint_used = endpoint.name
                    endpoint_name = endpoint.name

                # Pass context window size to Ollama so it doesn't truncate
                ctx_tokens = endpoint.context_tokens if endpoint and endpoint.context_tokens else None
                chat_kwargs = {}
                if ctx_tokens:
                    chat_kwargs["options"] = {"num_ctx": ctx_tokens}

                # Pass thinking mode toggle to Ollama (models that don't support it ignore it)
                if not self.think_enabled:
                    chat_kwargs["think"] = False

                # Apply tool rules to filter available tools based on call history
                iter_tools = None
                if tools and iteration < max_iterations:
                    iter_tools = self._tool_rules.filter_tools(tools, tool_call_names)
                    if not iter_tools:
                        iter_tools = None  # all tools filtered out → force text response

                response = await client.chat(
                    messages=messages,
                    model=model,
                    tools=iter_tools,
                    **chat_kwargs,
                )

                content, tool_calls = self._extract_response(response)
                logger.info("LLM response: tool_calls=%s, content_len=%d, tools_offered=%d",
                           bool(tool_calls), len(content),
                           len(iter_tools) if iter_tools else 0)

                if tool_calls and iteration < max_iterations:
                    # Process tool calls
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                    for tc in tool_calls:
                        name, arguments, tc_id = self._extract_tool_call_info(tc)
                        tool_call_names.append(name)
                        tool_call = ToolCall(id=tc_id, name=name, arguments=arguments)

                        if on_token:
                            arg_hint = self._format_tool_arg_hint(tool_call)
                            on_token(f"\n\x1b[36m\x1b[1m[Tool: {tool_call.name}{arg_hint}]\x1b[0m\n")

                        result = await self.tool_registry.execute_tool_call(tool_call)
                        result.tool_call_id = tc_id
                        messages.append(result.to_ollama_message())

                        # Track files/dirs already read (prevents re-reading across turns)
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

                        if on_token:
                            preview = result.result[:120].replace("\n", " ")
                            if result.success:
                                on_token(f"\x1b[2m[{preview}]\x1b[0m\n\n")
                            else:
                                on_token(f"\x1b[31m[{preview}]\x1b[0m\n\n")

                    if endpoint:
                        endpoint.record_success(0)
                    continue  # Loop back for LLM to process tool results
                else:
                    # No tool calls — use the response directly
                    full_response = content
                    if on_token and content:
                        on_token(content)
                    if endpoint:
                        endpoint.record_success(0)
                    break
            except Exception as e:
                if endpoint:
                    if is_model_error(e):
                        logger.warning(
                            "Model-level error on endpoint '%s' (not penalizing): %s",
                            endpoint.name, e,
                        )
                        self.router.mark_model_failed(model)
                    else:
                        endpoint.record_failure()

                # Try fallback model if we haven't already
                if not using_fallback:
                    fallback = self.router.get_fallback_model(task_type)
                    if fallback and fallback != model:
                        logger.warning("Primary model '%s' failed, falling back to '%s'", model, fallback)
                        model = fallback
                        using_fallback = True
                        if on_token:
                            on_token(f"\n\x1b[33m[Falling back to {fallback}]\x1b[0m\n")
                        continue  # Retry the iteration with fallback model

                logger.error("Chat error: %s", e)
                full_response = f"Error: {e}"
                break
            finally:
                if endpoint:
                    endpoint.complete_request()

        # Auto-continue: if the loop ended without a text response (hit iteration
        # limit mid-task), nudge the model to wrap up instead of going silent.
        if not full_response and tool_call_names:
            if on_token:
                on_token("\n\x1b[2m[Continuing...]\x1b[0m\n")
            messages.append({
                "role": "user",
                "content": (
                    "You hit the tool call limit. Summarize what you've done so far "
                    "and what remains. Do NOT call any more tools — just respond."
                ),
            })
            try:
                response = await client.chat(
                    messages=messages, model=model, tools=None,
                    **chat_kwargs,
                )
                content, _ = self._extract_response(response)
                full_response = content
                if on_token and content:
                    on_token(content)
            except Exception as e:
                logger.error("Auto-continue failed: %s", e)
                full_response = f"[Hit tool limit after {len(tool_call_names)} calls]"

        # Auto-nudge: if the LLM stopped mid-task (asking permission or
        # narrating intent without acting), nudge it to continue.
        # Only fires if tool calls have already been made — otherwise the LLM
        # is having a conversation and its questions are meant for the user.
        if (full_response and tool_call_names and self._should_auto_continue(full_response)):
            if on_token:
                on_token("\n\x1b[2m[Auto-continuing...]\x1b[0m\n")
            messages.append({"role": "assistant", "content": full_response})
            messages.append({
                "role": "user",
                "content": (
                    "Yes, go ahead. Execute the full task autonomously. "
                    "Do not ask for permission again — just do it."
                ),
            })
            try:
                response = await client.chat(
                    messages=messages, model=model, tools=tools,
                    **chat_kwargs,
                )
                content, new_tool_calls = self._extract_response(response)
                if new_tool_calls:
                    # The LLM wants to use tools — re-enter the tool loop
                    full_response = await self._continue_tool_loop(
                        messages, client, model, tools, chat_kwargs,
                        content, new_tool_calls, tool_call_names,
                        max_iterations - len(tool_call_names),
                        on_token, task_type,
                    )
                elif content:
                    full_response = content
                    if on_token:
                        on_token(content)
            except Exception as e:
                logger.error("Auto-nudge failed: %s", e)
                # Keep original response

        # Event: llm_complete
        await self._log_event("llm_complete", {
            "endpoint": endpoint_name,
            "model": model,
            "fallback": using_fallback,
            "tool_calls": tool_call_names,
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
            default=self.config.memory.total_context_tokens,
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
                f"You are working on the project \"{self.active_project['name']}\".\n"
                f"Project root: {root_path}\n"
                "All file tools (read_file, write_file, edit_file, list_directory) resolve "
                "relative paths against this project root. Use relative paths like "
                "'blipshell/ui/cli.py', NOT absolute paths.\n"
                "The run_command tool also runs from the project root.\n\n"
                "INTERACTION MODE:\n"
                "- When the user is discussing, asking questions, or exploring ideas — just have a conversation.\n"
                "- When the user asks you to make changes, create files, or implement something — use your tools.\n"
                "- For non-trivial changes, briefly describe your approach before starting.\n"
                "- Ask clarifying questions if requirements are ambiguous — don't guess.\n"
                "- You have tools available but do NOT use them unless the user is requesting action.\n\n"
                "TOOL DISCIPLINE:\n"
                "- Read a file ONCE, then use what you learned. Do NOT re-read files.\n"
                "- List a directory ONCE. Do NOT re-list directories.\n"
                "- Do NOT run the same grep or glob search twice.\n"
                "- Always read a file before editing it.\n"
                "- Use grep_files/glob_files tools, NOT shell grep/find/wc.\n"
                "- NEVER launch interactive or full-screen apps via run_command (TUI, curses, Textual .run()). They destroy the terminal.\n"
                "- NEVER create documentation files (.md, README) unless explicitly asked.\n"
                f"- Target UNDER {tool_limit} tool calls per task. Read, write, test — do not explore endlessly.\n\n"
                "PLATFORM: Windows.\n"
                f"- Project root: {root_path}\n"
                "- Do NOT use Linux commands (ls, cat, grep, head, tail, find, wc) in shell.\n"
                "- Use 'dir' not 'ls', 'type' not 'cat'. Or better: use the file/grep/glob tools.\n"
                "- Do NOT use 'cd' in run_command — it already runs from the project root.\n\n"
            )

            # Add model-specific extra instructions
            if ms.extra_instructions:
                system_prompt += f"MODEL-SPECIFIC INSTRUCTIONS:\n{ms.extra_instructions}\n\n"

            system_prompt += self._project_context

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if memory_text.strip():
            messages.append({"role": "system", "content": memory_text})

        # Inject files-already-read list so the model doesn't re-read across turns
        if self._files_read:
            files_list = "\n".join(f"  - {f}" for f in sorted(self._files_read))
            messages.append({"role": "system", "content": (
                "FILES ALREADY READ THIS SESSION (do NOT re-read these):\n"
                + files_list
                + "\nSkip straight to your task — do not list directories or read files you already have."
            )})

        # Add conversation history from ActiveSession (last messages)
        for msg in self.session_manager.get_messages()[-20:]:
            messages.append(msg.to_ollama_message())

        return messages

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
        """Background task to dump and process session memories."""
        try:
            if self.session_manager.message_count % 5 == 0:
                await self.session_manager.dump_to_memory()
        except Exception as e:
            logger.error("Background memory processing error: %s", e)

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
        # Cancel background memory tasks first to prevent writes during shutdown
        await self._cancel_background_tasks()
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
        if self.session_manager:
            await self.session_manager.end_session(on_status=on_status)
        if self.job_queue:
            await self.job_queue.stop()
        # Close ChromaDB after all writes are done
        if self.chroma:
            self.chroma.close()

    async def force_cleanup(self):
        """Cancel all background tasks so the process can exit cleanly.

        Order matters: cancel in-flight writes → close ChromaDB → close SQLite.
        """
        # 1. Cancel background memory processing tasks (prevents mid-write corruption)
        await self._cancel_background_tasks()

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
