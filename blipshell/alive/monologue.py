"""Inner Monologue — background thinking loop with tool-assisted research.

Two-phase cycle:
1. Research phase: model gets memories + optional focus, can call tools
   (search_memories, search_thoughts, web_search) to gather information.
2. Reflection phase: model gets original context + research results,
   produces thoughts, refinements, initiative items, and next focus.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from blipshell.alive.prompts import inner_monologue_cycle, monologue_research_phase
from blipshell.alive.thought_engine import ThoughtEngine
from blipshell.models.alive import MonologueCycleResult

if TYPE_CHECKING:
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import AliveConfig

logger = logging.getLogger(__name__)

# Max tool calls per research phase (prevent runaway loops)
MAX_RESEARCH_TOOL_CALLS = 5

# Tool definitions for the research phase (Ollama tool format)
RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": "Search your memory database for a specific topic. Returns summaries of relevant memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memories",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_thoughts",
            "description": "Search your existing thoughts on a topic. Returns relevant thoughts you've had before.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in your thoughts",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Use when you're curious about something and want to learn more.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


class InnerMonologue:
    """Two-phase monologue: research (with tools) → reflect (produce thoughts)."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        router: LLMRouter,
        config: AliveConfig,
        thought_engine: ThoughtEngine,
    ):
        self.sqlite = sqlite
        self.chroma = chroma
        self.router = router
        self.config = config
        self.thought_engine = thought_engine

    async def run_cycle(self) -> MonologueCycleResult:
        """Run one thinking cycle: research → reflect."""
        start = time.monotonic()
        cycle_num = await self.sqlite.get_next_monologue_cycle()

        # Load previous focus (self-directed prompt from last cycle)
        next_focus = await self._load_next_focus()

        # 1. Select memories to review
        memories = await self._select_memories(next_focus)
        memory_texts = [
            m.get("summary") or m.get("content", "")
            for m in memories
        ]

        # 2. Get recent thoughts for continuity
        recent = await self.thought_engine.get_recent_thoughts(limit=5)
        thought_texts = [t.get("content", "") for t in recent]

        # 3. Get current identity
        identity_row = await self.sqlite.get_current_identity()
        identity = identity_row["content"] if identity_row else ""

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Phase 1: Research (tool-calling loop)
        tool_results, tool_call_count = await self._research_phase(
            identity, memory_texts, next_focus, now_str,
        )

        # Phase 2: Reflect (produce thoughts)
        system, user = inner_monologue_cycle(
            current_identity=identity,
            memories=memory_texts,
            recent_thoughts=thought_texts,
            current_datetime=now_str,
            next_focus=next_focus,
            tool_results=tool_results if tool_results else None,
        )

        try:
            raw = await self.router.generate(
                "reasoning", user, system=system,
            )
        except Exception as e:
            logger.error("Monologue reflection failed: %s", e)
            elapsed = time.monotonic() - start
            result = MonologueCycleResult(
                cycle_number=cycle_num,
                memories_reviewed=len(memories),
                tool_calls_made=tool_call_count,
                elapsed_s=elapsed,
            )
            await self.sqlite.add_monologue_log(
                cycle_number=cycle_num,
                memories_reviewed=len(memories),
                elapsed_s=elapsed,
                raw_output=f"ERROR: {e}",
            )
            return result

        if not raw or raw.strip().upper() == "SKIP":
            elapsed = time.monotonic() - start
            result = MonologueCycleResult(
                cycle_number=cycle_num,
                memories_reviewed=len(memories),
                tool_calls_made=tool_call_count,
                elapsed_s=elapsed,
            )
            await self.sqlite.add_monologue_log(
                cycle_number=cycle_num,
                memories_reviewed=len(memories),
                elapsed_s=elapsed,
                raw_output="SKIP",
            )
            return result

        # 5. Parse and store results
        thoughts_gen, thoughts_ref, initiative_added = await self._process_output(
            raw, cycle_num,
        )

        # 6. Parse and store next focus
        new_focus = self._parse_next_focus(raw)
        if new_focus:
            await self._save_next_focus(new_focus)

        elapsed = time.monotonic() - start
        result = MonologueCycleResult(
            cycle_number=cycle_num,
            memories_reviewed=len(memories),
            thoughts_generated=thoughts_gen,
            thoughts_refined=thoughts_ref,
            initiative_items_added=initiative_added,
            tool_calls_made=tool_call_count,
            next_focus=new_focus,
            elapsed_s=elapsed,
        )

        await self.sqlite.add_monologue_log(
            cycle_number=cycle_num,
            memories_reviewed=len(memories),
            thoughts_generated=thoughts_gen,
            thoughts_refined=thoughts_ref,
            initiative_items_added=initiative_added,
            raw_output=raw[:2000],  # cap for storage
            elapsed_s=elapsed,
        )

        logger.info(
            "Monologue cycle %d: %d memories → %d thoughts, %d refined, "
            "%d initiative, %d tool calls (%.1fs)%s",
            cycle_num, len(memories), thoughts_gen, thoughts_ref,
            initiative_added, tool_call_count, elapsed,
            f" | next: {new_focus[:60]}..." if new_focus else "",
        )
        return result

    async def _research_phase(
        self,
        identity: str,
        memory_texts: list[str],
        next_focus: str | None,
        current_datetime: str,
    ) -> tuple[list[str], int]:
        """Phase 1: Let the model call tools to research before reflecting.

        Returns (tool_results, tool_call_count).
        """
        from blipshell.llm.router import TaskType

        model, client = await self.router.get_model_and_client("tool_calling")
        if not client:
            logger.warning("No client available for research phase")
            return [], 0

        system, user = monologue_research_phase(
            current_identity=identity,
            memories=memory_texts,
            next_focus=next_focus,
            current_datetime=current_datetime,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        tool_results = []
        tool_call_count = 0

        for _ in range(MAX_RESEARCH_TOOL_CALLS + 1):
            try:
                response = await client.chat(
                    messages=messages,
                    model=model,
                    tools=RESEARCH_TOOLS,
                )
            except Exception as e:
                logger.warning("Research phase LLM call failed: %s", e)
                break

            msg = response.get("message", {})

            # Check for tool calls
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                # No tool calls — model is done researching
                # Check if it said DONE or just produced text
                break

            if tool_call_count >= MAX_RESEARCH_TOOL_CALLS:
                break

            # Execute tool calls
            messages.append(msg)  # append assistant message with tool calls

            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                result_text = await self._execute_tool(name, args)
                tool_results.append(f"[{name}({args.get('query', '')})] {result_text}")
                tool_call_count += 1

                messages.append({
                    "role": "tool",
                    "content": result_text,
                })

        return tool_results, tool_call_count

    async def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a monologue tool and return the result as text."""
        query = args.get("query", "")
        if not query:
            return "(empty query)"

        try:
            if name == "search_memories":
                results = self.chroma.search_memories(query, n_results=5)
                if not results:
                    return "No relevant memories found."
                texts = []
                for r in results:
                    doc = r.get("document", "")
                    if len(doc) > 200:
                        doc = doc[:200] + "..."
                    texts.append(doc)
                return "\n".join(f"- {t}" for t in texts)

            elif name == "search_thoughts":
                results = self.chroma.search_thoughts(query, n_results=5)
                if not results:
                    return "No relevant thoughts found."
                texts = []
                for r in results:
                    doc = r.get("document", "")
                    texts.append(doc)
                return "\n".join(f"- {t}" for t in texts)

            elif name == "web_search":
                try:
                    from ddgs import DDGS
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=3))
                    if not results:
                        return "No web results found."
                    texts = []
                    for r in results:
                        title = r.get("title", "")
                        body = r.get("body", "")
                        if len(body) > 200:
                            body = body[:200] + "..."
                        texts.append(f"{title}: {body}")
                    return "\n".join(f"- {t}" for t in texts)
                except Exception as e:
                    return f"Web search failed: {e}"

            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            logger.warning("Monologue tool %s failed: %s", name, e)
            return f"Tool error: {e}"

    async def _select_memories(self, next_focus: str | None = None) -> list[dict]:
        """Select diverse memories for review. Uses next_focus as search seed if available."""
        per_cycle = self.config.inner_monologue.memories_per_cycle
        memories = []

        # If we have a focus from last cycle, use it to find relevant memories
        if next_focus:
            try:
                focused = self.chroma.search_memories(next_focus, n_results=min(4, per_cycle))
                for r in focused:
                    memories.append({
                        "id": r["id"],
                        "summary": r.get("document", ""),
                        "importance": 0.5,
                    })
            except Exception as e:
                logger.debug("Failed to get focus-related memories: %s", e)

        # Recent memories (what just happened)
        try:
            cursor = await self.sqlite._db.execute(
                """SELECT id, summary, content, importance, timestamp
                   FROM memories
                   WHERE is_archived = 0 AND summary IS NOT NULL
                   ORDER BY timestamp DESC LIMIT ?""",
                (min(3, per_cycle - len(memories)),),
            )
            rows = await cursor.fetchall()
            memories.extend([dict(r) for r in rows])
        except Exception as e:
            logger.debug("Failed to get recent memories: %s", e)

        # Random high-importance (long-term reflection)
        try:
            random_mems = await self.sqlite.get_random_high_importance_memories(
                limit=min(3, per_cycle - len(memories)),
            )
            memories.extend(random_mems)
        except Exception as e:
            logger.debug("Failed to get random memories: %s", e)

        # Recently accessed (what the user was thinking about)
        try:
            accessed = await self.sqlite.get_recently_accessed_memories(
                limit=min(2, per_cycle - len(memories)),
            )
            memories.extend(accessed)
        except Exception as e:
            logger.debug("Failed to get recently accessed memories: %s", e)

        # Related to latest thought (follow the thread)
        try:
            recent_thoughts = await self.thought_engine.get_recent_thoughts(limit=1)
            if recent_thoughts and len(memories) < per_cycle:
                related = self.chroma.search_memories(
                    recent_thoughts[0].get("content", ""),
                    n_results=min(2, per_cycle - len(memories)),
                )
                for r in related:
                    memories.append({
                        "id": r["id"],
                        "summary": r.get("document", ""),
                        "importance": 0.5,
                    })
        except Exception as e:
            logger.debug("Failed to get thought-related memories: %s", e)

        # Deduplicate by ID
        seen = set()
        unique = []
        for m in memories:
            mid = m.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                unique.append(m)

        return unique[:per_cycle]

    @staticmethod
    def _clean_raw(raw: str) -> str:
        """Strip markdown formatting from LLM output before parsing."""
        # Remove ** bold ** and * italic * around field names
        return re.sub(r'\*+', '', raw)

    async def _process_output(
        self, raw: str, cycle_num: int,
    ) -> tuple[int, int, int]:
        """Parse monologue output and store thoughts/refinements/initiative items.

        Returns (thoughts_generated, thoughts_refined, initiative_added).
        """
        cleaned = self._clean_raw(raw)
        thoughts_gen = 0
        thoughts_ref = 0
        initiative_added = 0

        # Parse new thoughts (reuse ThoughtEngine parser)
        from blipshell.alive.thought_engine import ThoughtEngine
        thoughts = ThoughtEngine._parse_thoughts(raw)  # parser does its own cleaning

        for thought in thoughts:
            if thought.confidence < self.config.thought.min_confidence:
                continue
            # Dedup
            if await self.thought_engine._is_duplicate(thought.content):
                continue

            thought_id = await self.sqlite.add_thought(
                content=thought.content,
                category=thought.category.value,
                confidence=thought.confidence,
                source_type="monologue",
            )
            try:
                self.chroma.add_thought(
                    thought_id, thought.content,
                    metadata={"category": thought.category.value},
                )
            except Exception:
                pass
            thoughts_gen += 1

        # Parse refinements: REFINE: <id>
        refine_blocks = re.findall(
            r'REFINE:\s*(?:The\s+)?["\']?.*?["\']?\s*\(?(\d+)\)?\s*\n\s*CONFIDEN(?:CE|T):\s*([\d.]+)\s*\n\s*THOUGHT:\s*(.+?)(?=\n\s*(?:CATEGORY|REFINE|INITIATIVE|NEXT_FOCUS):|$)',
            cleaned, re.IGNORECASE | re.DOTALL,
        )
        for match in refine_blocks:
            try:
                parent_id = int(match[0])
                confidence = max(0.0, min(1.0, float(match[1])))
                content = match[2].strip()
                if len(content) < 10:
                    continue
                await self.thought_engine.refine_thought(
                    parent_id, content, confidence, source_type="monologue",
                )
                thoughts_ref += 1
            except (ValueError, Exception) as e:
                logger.debug("Failed to parse refinement: %s", e)

        # Parse initiative items: INITIATIVE: <category>
        initiative_blocks = re.findall(
            r'INITIATIVE:\s*(\w+)\s*\n\s*PRIORITY:\s*([\d.]+)\s*\n\s*CONTENT:\s*(.+?)(?=\n\s*(?:CATEGORY|REFINE|INITIATIVE|NEXT_FOCUS):|$)',
            cleaned, re.IGNORECASE | re.DOTALL,
        )
        for match in initiative_blocks:
            try:
                category = match[0].lower()
                priority = max(0.0, min(1.0, float(match[1])))
                content = match[2].strip()
                if len(content) < 10:
                    continue
                await self.sqlite.add_initiative_item(
                    content=content,
                    category=category,
                    priority=priority,
                    source_type="monologue",
                )
                initiative_added += 1
            except (ValueError, Exception) as e:
                logger.debug("Failed to parse initiative item: %s", e)

        return thoughts_gen, thoughts_ref, initiative_added

    @staticmethod
    def _parse_next_focus(raw: str) -> str | None:
        """Parse NEXT_FOCUS from monologue output."""
        cleaned = InnerMonologue._clean_raw(raw)
        match = re.search(
            r'NEXT_FOCUS:\s*(.+?)(?=\n\s*(?:CATEGORY|REFINE|INITIATIVE|NEXT_FOCUS):|$)',
            cleaned, re.IGNORECASE | re.DOTALL,
        )
        if match:
            focus = match.group(1).strip()
            if len(focus) >= 10:
                return focus
        return None

    async def _load_next_focus(self) -> str | None:
        """Load the next focus from the previous cycle."""
        try:
            return await self.sqlite.get_metadata("alive_next_focus")
        except Exception:
            return None

    async def _save_next_focus(self, focus: str):
        """Save the next focus for the next cycle."""
        try:
            await self.sqlite.set_metadata("alive_next_focus", focus)
        except Exception as e:
            logger.debug("Failed to save next focus: %s", e)
