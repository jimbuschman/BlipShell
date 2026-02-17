"""ChatGPT data import — conversations, personality, and memories.

Parses a ChatGPT data export (conversations.json) and imports
conversations through BlipShell's memory pipeline. Also supports
importing a personality/system prompt and manual ChatGPT memories.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from blipshell.core.config import ConfigManager
from blipshell.llm.prompts import ask_importance, rank_memory, summarize_memory
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.noise import should_skip_memory
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.tagger import tag_message
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for parsed ChatGPT conversations
# ---------------------------------------------------------------------------

@dataclass
class ParsedMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None


@dataclass
class ParsedConversation:
    title: str
    created_at: Optional[float] = None
    messages: list[ParsedMessage] = field(default_factory=list)


@dataclass
class ImportStats:
    conversations_imported: int = 0
    conversations_skipped: int = 0
    conversations_reimported: int = 0
    messages_processed: int = 0
    messages_skipped_noise: int = 0
    lessons_extracted: int = 0


# ---------------------------------------------------------------------------
# Conversation parser
# ---------------------------------------------------------------------------

def parse_conversations(file_path: str | Path) -> list[ParsedConversation]:
    """Parse a ChatGPT conversations.json export file.

    ChatGPT export format:
    - Array of conversation objects, each with:
      - title, create_time, update_time
      - mapping: {node_id: {message: {...}, parent, children: [...]}}
      - current_node: ID of the last message
    - Messages form a tree; we linearize by walking from root following children.
    - author.role is one of: system, user, assistant, tool
    - content.parts is a list of strings (concatenated)
    - Some messages have null content or are system/tool messages — skip those
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected a JSON array of conversations")

    conversations = []
    for conv_obj in raw:
        parsed = _parse_single_conversation(conv_obj)
        if parsed and parsed.messages:
            conversations.append(parsed)

    return conversations


def _parse_single_conversation(conv_obj: dict) -> Optional[ParsedConversation]:
    """Parse a single conversation object from the ChatGPT export."""
    title = conv_obj.get("title") or "Untitled"
    created_at = conv_obj.get("create_time")
    mapping = conv_obj.get("mapping", {})

    if not mapping:
        return None

    # Find the root node (no parent, or parent is None)
    root_id = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            root_id = node_id
            break

    if root_id is None:
        # Fallback: pick the first node
        root_id = next(iter(mapping))

    # Walk the tree depth-first, following first child at each level
    messages = []
    _walk_tree(mapping, root_id, messages)

    return ParsedConversation(
        title=title,
        created_at=created_at,
        messages=messages,
    )


def _walk_tree(mapping: dict, root_id: str, messages: list[ParsedMessage]):
    """Iteratively walk the conversation tree, collecting user/assistant messages."""
    stack = [root_id]
    visited = set()

    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)

        node = mapping.get(node_id)
        if not node:
            continue

        msg = node.get("message")
        if msg:
            parsed = _extract_message(msg)
            if parsed:
                messages.append(parsed)

        # Push children in reverse order so first child is processed first
        children = node.get("children", [])
        for child_id in reversed(children):
            if child_id not in visited:
                stack.append(child_id)


def _extract_message(msg: dict) -> Optional[ParsedMessage]:
    """Extract a ParsedMessage from a ChatGPT message node, or None if skippable."""
    if not msg:
        return None

    author = msg.get("author", {})
    role = author.get("role", "")

    # Only keep user and assistant messages
    if role not in ("user", "assistant"):
        return None

    content_obj = msg.get("content", {})
    parts = content_obj.get("parts", [])

    # Concatenate text parts, skip non-string parts (images, etc.)
    text_parts = [p for p in parts if isinstance(p, str)]
    content = "\n".join(text_parts).strip()

    if not content:
        return None

    timestamp = msg.get("create_time")

    return ParsedMessage(
        role=role,
        content=content,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Conversation import
# ---------------------------------------------------------------------------

async def import_conversations(
    sqlite: SQLiteStore,
    chroma: ChromaStore,
    router: LLMRouter,
    config: MemoryConfig,
    conversations: list[ParsedConversation],
    on_progress: Optional[Callable[[int, int, str, "ImportStats"], None]] = None,
    skip_lessons: bool = False,
) -> ImportStats:
    """Import parsed ChatGPT conversations through the BlipShell memory pipeline.

    For each conversation:
    1. Create a BlipShell session
    2. Process each message through the memory pipeline (noise, summarize, embed, tag, rank)
    3. Optionally extract lessons from the full conversation

    Args:
        sqlite: SQLite store instance
        chroma: ChromaDB store instance
        router: LLM router for summarization/ranking
        config: Memory configuration
        conversations: Parsed conversations to import
        on_progress: Callback(current_index, total, title) for progress tracking
        skip_lessons: If True, skip lesson extraction (faster)

    Returns:
        ImportStats with counts of imported items
    """
    processor = MemoryProcessor(sqlite, chroma, router, config=config)
    stats = ImportStats()
    total = len(conversations)

    for i, conv in enumerate(conversations):
        if on_progress:
            on_progress(i, total, conv.title, stats)

        # Resume support: skip complete conversations, re-import incomplete ones
        session_id, db_count = await sqlite.get_session_message_count(conv.title)
        if session_id is not None:
            # Run noise filter (instant, no LLM) to get the real expected count.
            non_noise = sum(
                1 for m in conv.messages if not should_skip_memory(m.content)
            )
            if db_count >= non_noise:
                logger.info("Skipping already imported: %s", conv.title)
                stats.conversations_skipped += 1
                continue
            else:
                # Incomplete — delete and re-import
                logger.info(
                    "Re-importing incomplete '%s' (had %d/%d messages)",
                    conv.title, db_count, non_noise,
                )
                try:
                    # Clean up ChromaDB entries for old memories
                    cursor = await sqlite._db.execute(
                        "SELECT id FROM memories WHERE session_id = ?",
                        (session_id,),
                    )
                    old_ids = [row[0] for row in await cursor.fetchall()]
                    for mid in old_ids:
                        try:
                            chroma.delete_memory(mid)
                        except Exception:
                            pass
                    # Clean up lessons in ChromaDB
                    cursor = await sqlite._db.execute(
                        "SELECT id FROM lessons WHERE source_session_id = ?",
                        (session_id,),
                    )
                    old_lesson_ids = [row[0] for row in await cursor.fetchall()]
                    for lid in old_lesson_ids:
                        try:
                            chroma.delete_lesson(lid)
                        except Exception:
                            pass
                    await sqlite.delete_session_cascade(session_id, old_ids)
                    stats.conversations_reimported += 1
                except Exception as e:
                    logger.error("Failed to clean up incomplete '%s': %s", conv.title, e)
                    continue

        try:
            await _import_single_conversation(
                sqlite, chroma, router, processor, config, conv, stats,
                skip_lessons,
            )
            stats.conversations_imported += 1
        except Exception as e:
            logger.error("Failed to import conversation '%s': %s", conv.title, e)

        # Small delay between conversations to avoid overwhelming the LLM endpoint
        if i < total - 1:
            await asyncio.sleep(0.5)

    return stats


async def _import_single_conversation(
    sqlite: SQLiteStore,
    chroma: ChromaStore,
    router: LLMRouter,
    processor: MemoryProcessor,
    config: MemoryConfig | None,
    conv: ParsedConversation,
    stats: ImportStats,
    skip_lessons: bool,
):
    """Import a single conversation using batch-by-task-type pipeline.

    Instead of processing each message through the full 7-step pipeline
    (which swaps LLM models at every step), this batches all messages by
    task type so each model loads only once per conversation.

    Batch order (minimizes model swaps):
      1. Noise filter + tagging  (no LLM)
      2. Summarize all           (glm4)
      3. Rank all                (qwen2.5:14b)
      4. DB create + tag all     (no LLM)
      5. Embed all               (nomic-embed-text)
      6. Importance all + ranks  (qwen3:14b)
      7. Lessons                 (qwen3:14b — same model, no swap)
    """
    # Create a session for this conversation (preserve original date)
    conv_ts = (
        datetime.fromtimestamp(conv.created_at, tz=UTC)
        if conv.created_at else None
    )
    session_id = await sqlite.create_session(title=conv.title, created_at=conv_ts)

    recency_bonus = config.importance_recency_bonus if config else 0.1
    tag_bonus = config.importance_tag_bonus if config else 0.05

    title_short = conv.title[:40]
    total_msgs = len(conv.messages)

    # ── Step 1: Filter noise + tag (no LLM) ────────────────────────────
    valid_messages: list[tuple[ParsedMessage, list[str]]] = []
    for msg in conv.messages:
        if should_skip_memory(msg.content):
            stats.messages_skipped_noise += 1
            continue
        tags = tag_message(msg.content)
        valid_messages.append((msg, tags))
    noise_count = total_msgs - len(valid_messages)
    print(f"  [{title_short}] {len(valid_messages)}/{total_msgs} messages kept ({noise_count} noise)")

    # ── Step 2: Summarize all (one model load) ─────────────────────────
    # Messages where summary comes back as "SKIP" are filtered out.
    surviving: list[tuple[ParsedMessage, list[str], str]] = []
    for i, (msg, tags) in enumerate(valid_messages):
        print(f"  [{title_short}] Summarizing {i+1}/{len(valid_messages)}...", end="\r")
        try:
            sum_system, sum_prompt = summarize_memory(msg.content)
            summary = await router.generate(
                TaskType.SUMMARIZATION, sum_prompt, system=sum_system,
            )
            if summary.strip().upper() == "SKIP":
                logger.debug("Memory skipped (meta/self-referential): %s", msg.content[:50])
                stats.messages_skipped_noise += 1
                continue
        except Exception as e:
            logger.error("Summarization failed, using raw text: %s", e)
            summary = msg.content
        surviving.append((msg, tags, summary))
    print(f"  [{title_short}] Summarized {len(surviving)} messages" + " " * 20)

    # ── Step 3: Rank all (one model load) ──────────────────────────────
    ranks: list[int] = []
    for i, (msg, _tags, _summary) in enumerate(surviving):
        print(f"  [{title_short}] Ranking {i+1}/{len(surviving)}...", end="\r")
        try:
            rank_system, rank_prompt = rank_memory(msg.content)
            rank_text = await router.generate(
                TaskType.RANKING, rank_prompt, system=rank_system,
            )
            ranks.append(MemoryProcessor._parse_rank(rank_text))
        except Exception as e:
            logger.error("Ranking failed: %s", e)
            ranks.append(3)
    print(f"  [{title_short}] Ranked {len(ranks)} messages" + " " * 20)

    # ── Step 4: DB create + tag all (no LLM) ──────────────────────────
    memory_ids: list[int] = []
    for msg, tags, summary in surviving:
        ts = (
            datetime.fromtimestamp(msg.timestamp, tz=UTC)
            if msg.timestamp else None
        )
        memory = Memory(
            session_id=session_id,
            role=msg.role,
            content=msg.content,
            summary=summary,
            timestamp=ts or datetime.now(timezone.utc),
            memory_type=MemoryType.CONVERSATION,
        )
        memory_id = await sqlite.create_memory(memory)
        memory_ids.append(memory_id)
        try:
            await sqlite.tag_memory(memory_id, tags)
        except Exception as e:
            logger.error("Tagging failed: %s", e)
    print(f"  [{title_short}] Saved {len(memory_ids)} memories to DB")

    # ── Step 5: Embed all (one model load) ─────────────────────────────
    for i, memory_id in enumerate(memory_ids):
        msg, _tags, summary = surviving[i]
        try:
            chroma.add_memory(memory_id, summary, {
                "session_id": str(session_id),
                "role": msg.role,
            })
        except Exception as e:
            logger.error("ChromaDB embed failed: %s", e)
    print(f"  [{title_short}] Embedded {len(memory_ids)} memories")

    # ── Step 6: Importance all + update ranks (one model load) ─────────
    for i, memory_id in enumerate(memory_ids):
        print(f"  [{title_short}] Importance {i+1}/{len(memory_ids)}...", end="\r")
        _msg, tags, _summary = surviving[i]
        try:
            imp_system, imp_prompt = ask_importance(surviving[i][0].content)
            importance_text = await router.generate(
                TaskType.IMPORTANCE, imp_prompt, system=imp_system,
            )
            importance = MemoryProcessor._parse_float(importance_text, default=0.3)
        except Exception:
            importance = 0.3

        importance += recency_bonus
        if len(tags) > 6:
            importance += tag_bonus
        importance = min(importance, 1.0)

        try:
            await sqlite.update_memory(memory_id, rank=ranks[i], importance=importance)
        except Exception as e:
            logger.error("Rank/importance update failed: %s", e)
    print(f"  [{title_short}] Scored importance for {len(memory_ids)} memories" + " " * 20)

    # Update session message count
    msg_count = len(memory_ids)
    stats.messages_processed += msg_count
    await sqlite.update_session(session_id, message_count=msg_count)

    # ── Step 7: Lessons (same model as importance — no swap) ───────────
    if not skip_lessons and len(conv.messages) >= 4:
        print(f"  [{title_short}] Extracting lessons...")
        try:
            full_text = _build_conversation_text(conv)
            await processor.process_lesson(full_text, session_id)
            stats.lessons_extracted += 1
        except Exception as e:
            logger.error("Lesson extraction failed for '%s': %s", conv.title, e)

    print(f"  [{title_short}] Done ({msg_count} messages processed)")


def _build_conversation_text(conv: ParsedConversation) -> str:
    """Build a plain-text representation of a conversation for lesson extraction."""
    lines = []
    for msg in conv.messages:
        prefix = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Personality / system prompt import
# ---------------------------------------------------------------------------

def import_personality(config_manager: ConfigManager, personality_text: str):
    """Import a personality description into the BlipShell system prompt.

    Prepends the personality text to the existing system prompt, keeping
    BlipShell's tool-calling instructions intact.

    Args:
        config_manager: Config manager instance (must have loaded config)
        personality_text: The personality/style text to prepend
    """
    existing_prompt = config_manager.get("agent.system_prompt", "")
    new_prompt = f"{personality_text.strip()}\n\n{existing_prompt}"
    config_manager.set("agent.system_prompt", new_prompt)
    config_manager.save()


# ---------------------------------------------------------------------------
# Manual ChatGPT memories import (as core memories)
# ---------------------------------------------------------------------------

async def import_memories_as_core(
    sqlite: SQLiteStore,
    chroma: ChromaStore,
    router: LLMRouter,
    config: MemoryConfig,
    memories_text: str,
) -> int:
    """Import ChatGPT memories as BlipShell core memories.

    Accepts a text where each non-empty line is a separate memory.
    Each line is stored as a core memory via the memory processor.

    Args:
        sqlite: SQLite store instance
        chroma: ChromaDB store instance
        router: LLM router
        config: Memory configuration
        memories_text: Text with one memory per line

    Returns:
        Count of imported core memories
    """
    processor = MemoryProcessor(sqlite, chroma, router, config=config)
    count = 0

    lines = [line.strip() for line in memories_text.splitlines() if line.strip()]

    for line in lines:
        try:
            await processor.process_core_memory(line)
            count += 1
        except Exception as e:
            logger.error("Failed to import core memory: %s — %s", line[:50], e)

    return count
