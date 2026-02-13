"""Main agent loop (ports OllamaChat.SendMessageToOllama + Form1.RunChat).

Key improvement: Uses native Ollama tool calling instead of parsing
tool calls from markdown code blocks.
"""

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator, Callable, Optional

from blipshell.core.config import ConfigManager
from blipshell.core.tools.base import ToolRegistry
from blipshell.core.tools.filesystem import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from blipshell.core.tools.memory_tools import (
    GetSessionSummaryTool,
    ListSessionsTool,
    SaveCoreMemoryTool,
    SearchMemoriesTool,
)
from blipshell.core.tools.shell import ShellTool
from blipshell.core.tools.web import WebFetchTool, WebSearchTool
from blipshell.llm.client import LLMClient
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.job_queue import LLMJobQueue
from blipshell.llm.prompts import summarize_session_chunk
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.manager import MemoryManager, PoolItem, estimate_tokens
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.search import MemorySearch
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import BlipShellConfig
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
    5. Send to Ollama with native tool calling
    6. Handle tool call loop (max N iterations)
    7. Update session + memory pools
    8. Background: process memories (summarize, embed, tag, rank)
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

        self._initialized = False

    async def initialize(self):
        """Initialize all subsystems."""
        if self._initialized:
            return

        # Database
        self.sqlite = SQLiteStore(self.config.database.path)
        await self.sqlite.initialize()

        # ChromaDB
        self.chroma = ChromaStore(
            persist_dir=self.config.database.chroma_path,
            embedding_model=self.config.models.embedding,
            ollama_url=self.config.endpoints[0].url if self.config.endpoints else "http://localhost:11434",
        )
        self.chroma.initialize()

        # Endpoint manager
        self.endpoint_manager = EndpointManager(self.config.endpoints)

        # Router
        self.router = LLMRouter(self.config.models, self.endpoint_manager)

        # Job queue
        self.job_queue = LLMJobQueue()
        self.job_queue.start()

        # Memory manager
        self.memory_manager = MemoryManager(self.config.memory)
        self.memory_manager.set_summarize_callback(self._summarize_overflow)

        # Processor
        self.processor = MemoryProcessor(self.sqlite, self.chroma, self.router)

        # Search
        self.search = MemorySearch(
            self.sqlite, self.chroma, self.router,
            min_rank=self.config.memory.min_rank_threshold,
            search_limit=self.config.memory.recall_search_limit,
        )

        # Session manager
        self.session_manager = SessionManager(
            self.sqlite, self.memory_manager, self.processor, self.router,
            summary_chunk_size=self.config.session.summary_chunk_size,
        )

        # Register tools
        self._register_tools()

        self._initialized = True
        logger.info("Agent initialized")

    def _register_tools(self):
        """Register all tools."""
        cfg = self.config.tools

        self.tool_registry.register(ReadFileTool(
            max_file_size=cfg.filesystem.max_file_size,
            blocked_paths=cfg.filesystem.blocked_paths,
        ))
        self.tool_registry.register(WriteFileTool(blocked_paths=cfg.filesystem.blocked_paths))
        self.tool_registry.register(EditFileTool())
        self.tool_registry.register(ListDirectoryTool())
        self.tool_registry.register(ShellTool(
            timeout=cfg.shell.timeout,
            allowed_commands=cfg.shell.allowed_commands,
        ))
        self.tool_registry.register(WebSearchTool())
        self.tool_registry.register(WebFetchTool(
            max_size=cfg.web.max_fetch_size,
            timeout=cfg.web.timeout,
        ))

    def _register_memory_tools(self):
        """Register memory tools (needs session_id, so called after session start)."""
        session_id = self.session_manager.session_id if self.session_manager else None

        self.tool_registry.register(SearchMemoriesTool(self.search, session_id))
        self.tool_registry.register(SaveCoreMemoryTool(self.processor, session_id))
        self.tool_registry.register(ListSessionsTool(self.sqlite))
        self.tool_registry.register(GetSessionSummaryTool(self.sqlite))

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

    async def chat(
        self,
        user_message: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Process a user message through the full agent pipeline.

        Args:
            user_message: The user's input
            on_token: Optional callback for streaming tokens

        Returns:
            The assistant's complete response
        """
        # Add user message to session
        self.session_manager.add_message(MessageRole.USER, user_message)

        # Search relevant memories for recall
        await self._search_relevant_memories(user_message)

        # Build message list
        messages = self._build_messages(user_message)

        # Get model and client
        model = self.router.get_model(TaskType.REASONING)
        client = self.router.get_client(TaskType.REASONING)
        if not client:
            return "Error: No available LLM endpoint."

        # Get tools
        tools = self.tool_registry.get_all_ollama_tools()

        # Tool call loop
        max_iterations = self.config.agent.max_tool_iterations
        full_response = ""

        for iteration in range(max_iterations + 1):
            endpoint = self.endpoint_manager.get_endpoint_for_role(TaskType.REASONING)
            if endpoint:
                endpoint.start_request()

            try:
                # Non-streaming call with tools to check for tool calls
                response = await client.chat(
                    messages=messages,
                    model=model,
                    tools=tools,
                )

                msg = response.get("message", {})
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", None)

                if tool_calls and iteration < max_iterations:
                    # Process tool calls
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        tool_call = ToolCall(
                            name=fn.get("name", ""),
                            arguments=fn.get("arguments", {}),
                        )

                        if on_token:
                            on_token(f"\n[Tool: {tool_call.name}]\n")

                        result = await self.tool_registry.execute_tool_call(tool_call)
                        messages.append(result.to_ollama_message())

                        if on_token:
                            on_token(f"[Result: {result.result[:200]}]\n\n")

                    continue  # Loop back for LLM to process tool results
                else:
                    # No tool calls or max iterations — stream the final response
                    if self.config.agent.stream and on_token:
                        # Re-send as streaming for token-by-token output
                        full_response = await self._stream_response(
                            client, messages, model, None, on_token
                        )
                    else:
                        full_response = content
                    break

                if endpoint:
                    endpoint.record_success(0)
            except Exception as e:
                if endpoint:
                    endpoint.record_failure()
                logger.error("Chat error: %s", e)
                full_response = f"Error: {e}"
                break
            finally:
                if endpoint:
                    endpoint.complete_request()

        # Add assistant response to session
        self.session_manager.add_message(MessageRole.ASSISTANT, full_response)

        # Background: dump to memory periodically
        asyncio.create_task(self._background_memory_processing())

        return full_response

    async def _stream_response(
        self,
        client: LLMClient,
        messages: list[dict],
        model: str,
        tools: list[dict] | None,
        on_token: Callable[[str], None],
    ) -> str:
        """Stream a response, calling on_token for each chunk."""
        full = []
        async for chunk in client.chat_stream(messages=messages, model=model, tools=tools):
            content = chunk.get("message", {}).get("content", "")
            if content:
                full.append(content)
                on_token(content)
        return "".join(full)

    async def _search_relevant_memories(self, query: str):
        """Search for relevant memories and add to Recall pool."""
        try:
            results = await self.search.search(
                query=query,
                current_session_id=self.session_manager.session_id,
                n_results=10,
            )
            for r in results:
                self.memory_manager.add_memory("Recall", PoolItem(
                    text=r.summary,
                    session_role="system",
                    priority_score=r.boosted_score,
                ))
        except Exception as e:
            logger.error("Memory search failed: %s", e)

    def _build_messages(self, user_message: str) -> list[dict]:
        """Build the full message list with memory context.

        Port of OllamaChat.SendMessageToOllama message building.
        """
        user_tokens = estimate_tokens(user_message)
        available = (
            self.config.memory.total_context_tokens
            - user_tokens
            - MemoryManager.OVERHEAD_TOKENS
        )

        # Gather memory from all pools
        memory_items = self.memory_manager.gather_memory(token_budget=available)

        # Build memory context string organized by pool
        context_parts = {}
        for item in memory_items:
            pool = item.pool_name
            if pool not in context_parts:
                label = {
                    "Core": "CoreFoundation",
                    "Lessons": "RelevantLessons",
                    "Recall": "RelevantMemory",
                    "RecentHistory": "RecentHistory",
                    "Buffer": "RecentHistory",
                    "ActiveSession": "ActiveSession",
                }.get(pool, pool)
                context_parts[pool] = (label, [])
            context_parts[pool][1].append(f"   - {item.text}")

        memory_text = ""
        for pool_name, (label, items) in context_parts.items():
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

    async def end_session(self):
        """End the current session and clean up."""
        if self.session_manager:
            await self.session_manager.end_session()
        if self.job_queue:
            await self.job_queue.stop()

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
        }
