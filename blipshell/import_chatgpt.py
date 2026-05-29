"""ChatGPT data import — conversations, personality, and memories.

Parses a ChatGPT data export (conversations.json) and imports
conversations through BlipShell's memory pipeline. Also supports
importing a personality/system prompt and manual ChatGPT memories.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from blipshell.core.config import ConfigManager
from blipshell.import_common import (
    ImportStats,
    ParsedConversation,
    ParsedMessage,
    import_conversations,
)
from blipshell.llm.router import LLMRouter
from blipshell.memory.vector_store import VectorStore
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import MemoryConfig

logger = logging.getLogger(__name__)

# Re-export shared symbols so existing callers (cli.py) don't break
__all__ = [
    "ParsedMessage",
    "ParsedConversation",
    "ImportStats",
    "import_conversations",
    "parse_conversations",
    "import_personality",
    "import_memories_as_core",
]


# ---------------------------------------------------------------------------
# Conversation parser (ChatGPT-specific)
# ---------------------------------------------------------------------------

def parse_conversations(file_path: str | Path) -> list[ParsedConversation]:
    """Parse a ChatGPT conversations export file.

    Supports two containers (both hold the same per-conversation shape):
    - JSON array — the official `conversations.json` (one big `[...]`)
    - JSONL — one conversation object per line (third-party exporters)

    Each conversation object has:
      - title, create_time, update_time, conversation_id
      - mapping: {node_id: {message: {...}, parent, children: [...]}}
      - current_node: ID of the last message
    Messages form a tree; we linearize by walking from root following children.
    author.role is one of system/user/assistant/tool; only user/assistant kept.
    """
    path = Path(file_path)
    records = _load_records(path)

    conversations = []
    for conv_obj in records:
        parsed = _parse_single_conversation(conv_obj)
        if parsed and parsed.messages:
            conversations.append(parsed)

    return conversations


def _load_records(path: Path) -> list[dict]:
    """Load conversation records from a JSON array or JSONL file.

    Auto-detects: tries whole-file JSON first; if that yields a list, use it;
    if it yields a single dict, wrap it; otherwise fall back to line-by-line
    JSONL parsing. This is robust to the .json/.jsonl extension being wrong.
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()

    # JSON array or single object — try whole-file parse first.
    if stripped.startswith("["):
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError("Expected a JSON array of conversations")
        return raw
    if stripped.startswith("{"):
        # Could be a single conversation object, or JSONL (many objects, one
        # per line). A single object parses cleanly; JSONL raises "Extra data".
        try:
            obj = json.loads(text)
            return [obj] if isinstance(obj, dict) else obj
        except json.JSONDecodeError:
            pass  # fall through to JSONL

    # JSONL: one JSON object per non-empty line.
    records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{path.name}: line {line_no} is not valid JSON ({e}). "
                "Expected a JSON array or JSONL (one conversation per line)."
            ) from e
    if not records:
        raise ValueError(f"{path.name}: no conversations found")
    return records


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

    # conversation_id (fallback to id) identifies the source conversation for
    # dedup; update_time lets re-imports detect a conversation that has grown.
    external_id = conv_obj.get("conversation_id") or conv_obj.get("id")
    external_updated_at = conv_obj.get("update_time")

    return ParsedConversation(
        title=title,
        created_at=created_at,
        messages=messages,
        external_id=external_id,
        external_updated_at=external_updated_at,
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
    vectors: VectorStore,
    router: LLMRouter,
    config: MemoryConfig,
    memories_text: str,
) -> int:
    """Import ChatGPT memories as BlipShell core memories.

    Accepts a text where each non-empty line is a separate memory.
    Each line is stored as a core memory via the memory processor.

    Args:
        sqlite: SQLite store instance
        vectors: Vector store instance
        router: LLM router
        config: Memory configuration
        memories_text: Text with one memory per line

    Returns:
        Count of imported core memories
    """
    processor = MemoryProcessor(sqlite, vectors, router, config=config)
    count = 0

    lines = [line.strip() for line in memories_text.splitlines() if line.strip()]

    for line in lines:
        try:
            await processor.process_core_memory(line)
            count += 1
        except Exception as e:
            logger.error("Failed to import core memory: %s — %s", line[:50], e)

    return count
