"""Memory tools: LLM can search/save memories, list sessions."""

import json
from typing import Optional

from blipshell.core.tools.base import Tool
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.search import MemorySearch
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType


class SearchMemoriesTool(Tool):
    read_only = True

    def __init__(self, search: MemorySearch, current_session_id: int | None = None):
        self.search = search
        self.current_session_id = current_session_id

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_memories",
            description=(
                "Search past conversation memories for specific information.\n\n"
                "Your context already includes automatically-retrieved memories, but they're "
                "limited to the top results for the user's current message. Use this tool when:\n"
                "- You need more detail about a topic than what's in your context\n"
                "- The user asks about something specific from a past conversation\n"
                "- You want to verify a fact before stating it\n"
                "- The automatically-loaded context doesn't cover what the user is asking about\n\n"
                "Returns full conversation content, not just summaries."
            ),
            parameters=[
                ToolParameter(name="query", type=ToolParameterType.STRING,
                              description="Search query — be specific (e.g. 'cat name' not 'what do you know about me')"),
                ToolParameter(name="max_results", type=ToolParameterType.INTEGER,
                              description="Maximum results to return (default 5)", required=False),
            ],
        )

    async def execute(self, query: str, max_results: int = 5, **kwargs) -> str:
        results = await self.search.search(
            query=query,
            current_session_id=self.current_session_id,
            n_results=max_results,
        )

        if not results:
            return "No relevant memories found."

        output = []
        for r in results:
            # Return raw content (actual conversation), not one-line summaries.
            text = r.text if r.text and len(r.text) > len(r.summary or "") else (r.summary or r.text or "")
            if len(text) > 1200:
                text = text[:1200] + "..."
            ts = ""
            if r.timestamp:
                ts = f" | {r.timestamp.strftime('%Y-%m-%d')}"
            output.append(
                f"[Score: {r.boosted_score:.2f}{ts}]\n"
                f"{text}\n"
            )
        return "\n---\n".join(output)


class SaveCoreMemoryTool(Tool):
    """Allows the LLM to save important information as core memories.

    Replaces the C# FRAMEWORK_MEMORY_ENTRY parsing from SystemFunctionCall.cs.
    """

    def __init__(self, processor: MemoryProcessor, session_id: int | None = None):
        self.processor = processor
        self.session_id = session_id

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="save_core_memory",
            description=(
                "Save an important, lasting fact about the user as a permanent core memory. "
                "Core memories are always loaded into your context — use this sparingly.\n\n"
                "When to use:\n"
                "- User's name, job, location, or key personal details\n"
                "- Stated preferences that affect how you should respond (e.g. 'I prefer concise answers')\n"
                "- Important decisions or facts the user explicitly asks you to remember\n"
                "- Skills, tools, or technologies the user works with regularly\n\n"
                "When NOT to use:\n"
                "- For conversation details — these are saved automatically as regular memories\n"
                "- For temporary or situational information (e.g. 'I'm working on X today')\n"
                "- If you're unsure whether it's important — it probably isn't\n"
                "- For information already in your context from a previous core memory"
            ),
            parameters=[
                ToolParameter(name="content", type=ToolParameterType.STRING,
                              description="The information to remember permanently"),
                ToolParameter(name="category", type=ToolParameterType.STRING,
                              description="Category: general, preference, fact, or personality",
                              required=False),
            ],
        )

    async def execute(self, content: str, category: str = "general", **kwargs) -> str:
        mem_id = await self.processor.process_core_memory(
            text=content,
            session_id=self.session_id,
        )
        return f"Core memory saved (ID: {mem_id}): {content[:100]}"


class ListSessionsTool(Tool):
    read_only = True

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_sessions",
            description="List recent conversation sessions with their titles and summaries.",
            parameters=[
                ToolParameter(name="limit", type=ToolParameterType.INTEGER,
                              description="Maximum sessions to return (default 10)", required=False),
                ToolParameter(name="project", type=ToolParameterType.STRING,
                              description="Filter by project name", required=False),
            ],
        )

    async def execute(self, limit: int = 10, project: str | None = None, **kwargs) -> str:
        sessions = await self.sqlite.list_sessions(limit=limit, project=project)

        if not sessions:
            return "No sessions found."

        output = []
        for s in sessions:
            title = s.title or "Untitled"
            summary = (s.summary or "No summary")[:100]
            output.append(f"Session #{s.id}: {title}\n  {summary}\n  Messages: {s.message_count}")

        return "\n\n".join(output)


class PromoteToCoreMemoryTool(Tool):
    """Allows the LLM to promote an existing memory or lesson to core memory status."""

    def __init__(self, sqlite: SQLiteStore, processor: MemoryProcessor,
                 session_id: int | None = None):
        self.sqlite = sqlite
        self.processor = processor
        self.session_id = session_id

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="promote_to_core_memory",
            description=(
                "Promote an existing memory or lesson to a permanent core memory.\n\n"
                "When to use:\n"
                "- A search result or loaded memory contains a key fact that should always be available\n"
                "- A lesson learned from a past session is broadly useful (not just project-specific)\n\n"
                "When NOT to use:\n"
                "- To save NEW information — use save_core_memory instead\n"
                "- If the information is already a core memory — check your context first\n"
                "- For project-specific details that are only relevant during that project\n\n"
                "Requires the source_id from a memory or lesson you've seen in search results or context."
            ),
            parameters=[
                ToolParameter(name="source_type", type=ToolParameterType.STRING,
                              description="Type of source: 'memory' or 'lesson'",
                              enum=["memory", "lesson"]),
                ToolParameter(name="source_id", type=ToolParameterType.INTEGER,
                              description="ID of the memory or lesson to promote"),
                ToolParameter(name="category", type=ToolParameterType.STRING,
                              description="Category: general, preference, fact, or personality",
                              required=False),
            ],
        )

    async def execute(self, source_type: str, source_id: int,
                      category: str = "general", **kwargs) -> str:
        if source_type == "memory":
            memory = await self.sqlite.get_memory(source_id)
            if not memory:
                return f"Memory {source_id} not found."
            content = memory.summary or memory.content
        elif source_type == "lesson":
            lessons = await self.sqlite.get_all_lessons()
            lesson = next((l for l in lessons if l.id == source_id), None)
            if not lesson:
                return f"Lesson {source_id} not found."
            content = lesson.content
        else:
            return f"Invalid source_type: {source_type}. Use 'memory' or 'lesson'."

        mem_id = await self.processor.process_core_memory(
            text=content,
            session_id=self.session_id,
        )
        return (
            f"Promoted {source_type} #{source_id} to core memory (ID: {mem_id}): "
            f"{content[:100]}"
        )


class GetSessionSummaryTool(Tool):
    read_only = True

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_session_summary",
            description="Get the detailed summary of a specific conversation session.",
            parameters=[
                ToolParameter(name="session_id", type=ToolParameterType.INTEGER,
                              description="The session ID to get the summary for"),
            ],
        )

    async def execute(self, session_id: int, **kwargs) -> str:
        session = await self.sqlite.get_session(session_id)
        if not session:
            return f"Session {session_id} not found."

        title = session.title or "Untitled"
        summary = session.summary or "No summary available."
        return f"Session #{session.id}: {title}\nProject: {session.project or 'None'}\n\n{summary}"
