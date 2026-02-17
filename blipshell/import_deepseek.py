"""DeepSeek data import — conversation parser.

Parses a DeepSeek data export (conversations_deepseek.json) into
ParsedConversation objects for the shared import pipeline.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from blipshell.import_common import ParsedConversation, ParsedMessage

logger = logging.getLogger(__name__)


def parse_conversations(file_path: str | Path) -> list[ParsedConversation]:
    """Parse a DeepSeek conversations export file.

    DeepSeek export format:
    - Top-level: JSON array of conversation objects
    - Conversation fields: title, inserted_at (ISO 8601 with +08:00 offset)
    - Uses a tree/mapping structure similar to ChatGPT
    - Messages have fragments[] array with types:
      - REQUEST  → user message content (concatenate)
      - RESPONSE → assistant message content (concatenate)
      - THINK    → chain-of-thought reasoning (skip)
      - SEARCH   → web search metadata (skip)
    - Node IDs are sequential ("root", "1", "2", ...) instead of UUIDs
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
    """Parse a single DeepSeek conversation object."""
    title = conv_obj.get("title") or "Untitled"
    inserted_at_str = conv_obj.get("inserted_at")
    mapping = conv_obj.get("mapping", {})

    if not mapping:
        return None

    # Parse conversation timestamp
    created_at = _parse_iso_timestamp(inserted_at_str)

    # Find root node
    root_id = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            root_id = node_id
            break

    if root_id is None:
        root_id = "root" if "root" in mapping else next(iter(mapping))

    # Walk the tree collecting messages
    messages = []
    _walk_tree(mapping, root_id, messages)

    return ParsedConversation(
        title=title,
        created_at=created_at,
        messages=messages,
    )


def _walk_tree(mapping: dict, root_id: str, messages: list[ParsedMessage]):
    """Iteratively walk the DeepSeek conversation tree, collecting messages."""
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

        msg_obj = node.get("message")
        if msg_obj:
            parsed = _extract_messages_from_fragments(msg_obj)
            messages.extend(parsed)

        # Push children in reverse order so first child is processed first
        children = node.get("children", [])
        for child_id in reversed(children):
            if child_id not in visited:
                stack.append(child_id)


def _extract_messages_from_fragments(msg_obj: dict) -> list[ParsedMessage]:
    """Extract ParsedMessages from a DeepSeek message node's fragments.

    A single node can have multiple fragments. We group consecutive fragments
    of the same role and concatenate their content.
    """
    fragments = msg_obj.get("fragments", [])
    if not fragments:
        return []

    timestamp = _parse_iso_timestamp(msg_obj.get("inserted_at"))

    results = []
    current_role = None
    current_parts = []

    for frag in fragments:
        frag_type = frag.get("type", "")

        if frag_type == "REQUEST":
            role = "user"
        elif frag_type == "RESPONSE":
            role = "assistant"
        else:
            # Skip THINK, SEARCH, and any unknown types
            continue

        content = (frag.get("content") or "").strip()
        if not content:
            continue

        if role == current_role:
            current_parts.append(content)
        else:
            # Flush previous
            if current_role and current_parts:
                text = "\n".join(current_parts).strip()
                if text:
                    results.append(ParsedMessage(
                        role=current_role,
                        content=text,
                        timestamp=timestamp,
                    ))
            current_role = role
            current_parts = [content]

    # Flush final
    if current_role and current_parts:
        text = "\n".join(current_parts).strip()
        if text:
            results.append(ParsedMessage(
                role=current_role,
                content=text,
                timestamp=timestamp,
            ))

    return results


def _parse_iso_timestamp(ts_str: Optional[str]) -> Optional[float]:
    """Parse an ISO 8601 timestamp string to a Unix epoch float."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None
