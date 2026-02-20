"""Main agent loop (ports OllamaChat.SendMessageToOllama + Form1.RunChat).

Key improvement: Uses native Ollama tool calling instead of parsing
tool calls from markdown code blocks.

Extended with:
- Task planner + executor (Phase 1): complex messages get decomposed
- Background task manager (Phase 2): async long-running tasks
- Workflow system (Phase 4): named reusable templates
"""

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator, Callable, Optional

from blipshell.core.background import BackgroundTaskManager
from blipshell.core.config import ConfigManager
from blipshell.core.executor import TaskExecutor
from blipshell.core.planner import ComplexityClassifier, TaskPlanner
from blipshell.core.tools.base import ToolRegistry, detect_tool_groups
from blipshell.core.tools.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from blipshell.core.tools.memory_tools import (
    ListSessionsTool,
    PromoteToCoreMemoryTool,
    SaveCoreMemoryTool,
    SearchMemoriesTool,
)
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

        # Memory
        self.memory_manager: Optional[MemoryManager] = None
        self.processor: Optional[MemoryProcessor] = None
        self.search: Optional[MemorySearch] = None

        # Session
        self.session_manager: Optional[SessionManager] = None

        # Tools
        self.tool_registry = ToolRegistry()

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
        self._last_endpoint_used: Optional[str] = None
        self._initialized = False
        self.think_enabled: bool = False  # /think toggle — off for fast simple chat, complex auto-enables
        self.reflect_enabled: bool = False  # /reflect toggle — second-pass self-critique
        self._turn_number: int = 0
        self._last_context_stats: Optional[dict] = None

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
        )

        # Background task manager (Phase 2)
        self.background_manager = BackgroundTaskManager(
            self.router, self.sqlite, self.config.worker,
        )

        # Workflow system (Phase 4)
        self.workflow_registry = WorkflowRegistry("workflows")
        self.workflow_executor = WorkflowExecutor(
            self.workflow_registry, self.task_executor, self.sqlite,
        )

        # Register tools
        self._register_tools()

        _status("Pruning old memories...")
        await self._auto_prune_memories()

        _status("Consolidating memories...")
        await self._auto_consolidate_memories()

        # Load discovered tag patterns into tagger
        await self._load_discovered_tags()

        _status("Checking endpoints...")
        await self.endpoint_manager.startup_health_check()

        _status("Running tag discovery...")
        await self._auto_tag_discovery()

        _status("Extracting entities...")
        await self._auto_extract_entities()

        # Start periodic health check (re-detects endpoints that come/go)
        self._health_check_task = self.endpoint_manager.start_health_loop(interval=60)

        self._initialized = True
        logger.info("Agent initialized")

    def _register_tools(self):
        """Register all tools with their group for selective inclusion."""
        cfg = self.config.tools

        # Filesystem group
        self.tool_registry.register(ReadFileTool(
            max_file_size=cfg.filesystem.max_file_size,
            blocked_paths=cfg.filesystem.blocked_paths,
        ), group="filesystem")
        self.tool_registry.register(WriteFileTool(
            blocked_paths=cfg.filesystem.blocked_paths,
        ), group="filesystem")
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

    async def start_session(
        self,
        project: Optional[str] = None,
        resume_session_id: Optional[int] = None,
    ) -> int:
        """Start or resume a session."""
        await self.initialize()

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
        """Prune old low-value memories on startup."""
        cfg = self.config.memory
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
        """Merge near-duplicate memories on startup."""
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
            extractor = EntityExtractor(
                self.sqlite, self.router,
                batch_size=self.config.memory.entity_extraction_batch_size,
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
    def _extract_tool_call_info(tc) -> tuple[str, dict]:
        """Extract name and arguments from a tool call object or dict."""
        # Object access (ollama 0.4+)
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", "") or ""
            args = getattr(fn, "arguments", {}) or {}
            return name, args

        # Dict access
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            return fn.get("name", ""), fn.get("arguments", {})

        return "", {}

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
                on_token("\n\n[Reflecting...]\n\n")
            response = await self._reflect_on_response(user_message, response, on_token)

        # Add assistant response to session
        self.session_manager.add_message(MessageRole.ASSISTANT, response)

        # Background: dump to memory periodically
        asyncio.create_task(self._background_memory_processing())

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

        # Get model and client (with fallback if cloud is down)
        model = self.router.get_model(TaskType.REASONING)
        client = await self.router.get_client(TaskType.REASONING)
        using_fallback = False
        if not client:
            # Try fallback model
            fallback = self.router.get_fallback_model(TaskType.REASONING)
            if fallback:
                logger.warning("Primary endpoint down, using fallback model '%s'", fallback)
                model = fallback
                using_fallback = True
                client = await self.router.get_client(TaskType.REASONING)
            if not client:
                return "Error: No available LLM endpoint."

        # Always pass all tools — the model decides whether to use them
        tools = self.tool_registry.get_all_ollama_tools() or None
        max_iterations = self.config.agent.max_tool_iterations if tools else 0
        logger.info("Passing %d tools to model", len(tools) if tools else 0)
        full_response = ""
        tool_call_names: list[str] = []
        endpoint_name = ""

        for iteration in range(max_iterations + 1):
            endpoint = None
            try:
                endpoint = await self.endpoint_manager.get_endpoint_for_role(TaskType.REASONING)
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

                response = await client.chat(
                    messages=messages,
                    model=model,
                    tools=tools if iteration < max_iterations else None,
                    **chat_kwargs,
                )

                content, tool_calls = self._extract_response(response)
                logger.info("LLM response: tool_calls=%s, content_len=%d",
                           bool(tool_calls), len(content))

                if tool_calls and iteration < max_iterations:
                    # Process tool calls
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                    for tc in tool_calls:
                        name, arguments = self._extract_tool_call_info(tc)
                        tool_call_names.append(name)
                        tool_call = ToolCall(name=name, arguments=arguments)

                        if on_token:
                            on_token(f"\n[Tool: {tool_call.name}]\n")

                        result = await self.tool_registry.execute_tool_call(tool_call)
                        messages.append(result.to_ollama_message())

                        if on_token:
                            on_token(f"[Result: {result.result[:200]}]\n\n")

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
                    else:
                        endpoint.record_failure()
                logger.error("Chat error: %s", e)
                full_response = f"Error: {e}"
                break
            finally:
                if endpoint:
                    endpoint.complete_request()

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
        """Planned chat path — decompose into steps and execute sequentially."""
        session_id = self.session_manager.session_id

        # Generate plan
        if on_token:
            on_token("[Planning...]\n")

        try:
            plan = await self.task_planner.create_plan(
                user_message, session_id=session_id,
            )
        except Exception as e:
            logger.error("Plan generation failed: %s", e)
            # Fallback to simple chat
            if on_token:
                on_token("[Plan generation failed, falling back to direct chat]\n")
            return await self._chat_simple(user_message, on_token=on_token)

        # Show plan to user
        if on_token:
            on_token(f"\n[Plan ({len(plan.steps)} steps):]\n")
            for step in plan.steps:
                tool_hint = f" ({step.tool_hint})" if step.tool_hint else ""
                on_token(f"  {step.step_number}. {step.description}{tool_hint}\n")
            on_token("\n[Executing...]\n\n")

        # Execute plan
        def on_step_start(step_num, total, description):
            if on_token:
                on_token(f"\n--- Step {step_num}/{total}: {description} ---\n")

        def on_step_complete(step_num, total, result_summary):
            if on_token:
                on_token(f"\n[Step {step_num}/{total} complete]\n")

        try:
            result = await self.task_executor.execute_plan(
                plan,
                on_step_start=on_step_start,
                on_step_complete=on_step_complete,
                on_token=on_token,
            )
        except Exception as e:
            logger.error("Plan execution failed: %s", e)
            result = f"Plan execution failed: {e}"

        return result

    async def _search_relevant_memories(self, query: str):
        """Search for relevant memories and lessons, add to Recall pool."""
        # Search conversation memories
        memory_count = 0
        try:
            results = await self.search.search(
                query=query,
                current_session_id=self.session_manager.session_id,
                n_results=10,
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
            lesson_results = await self.search.search_lessons(query, n_results=5)
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
        context_limit = self.endpoint_manager.get_context_tokens_for_role(
            TaskType.REASONING,
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
        messages = [
            {"role": "system", "content": self.config.agent.system_prompt},
        ]

        if memory_text.strip():
            messages.append({"role": "system", "content": memory_text})

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

    async def end_session(self):
        """End the current session and clean up."""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
        if self.session_manager:
            await self.session_manager.end_session()
        if self.job_queue:
            await self.job_queue.stop()

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
