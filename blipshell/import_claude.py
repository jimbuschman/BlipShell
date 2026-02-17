"""Claude data import — official export and scraped conversations.

Provides two parsers for Claude conversation data:
  - parse_conversations()         — official Claude export (conversations.json)
  - parse_scraped_conversations() — scraped from .txt files (conversations_export.json)

Both return list[ParsedConversation] for the shared import pipeline.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from blipshell.import_common import ParsedConversation, ParsedMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Official Claude export parser
# ---------------------------------------------------------------------------

def parse_conversations(file_path: str | Path) -> list[ParsedConversation]:
    """Parse an official Claude conversations.json export file.

    Format:
    - Top-level: JSON array of conversation objects
    - Conversation fields: uuid, name (title), created_at (ISO 8601), chat_messages (flat array)
    - Message fields: sender ("human"/"assistant"), text (plain text), created_at (ISO 8601)
    - Skip empty conversations (no chat_messages)
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected a JSON array of conversations")

    conversations = []
    for conv_obj in raw:
        parsed = _parse_official_conversation(conv_obj)
        if parsed and parsed.messages:
            conversations.append(parsed)

    return conversations


def _parse_official_conversation(conv_obj: dict) -> Optional[ParsedConversation]:
    """Parse a single conversation from the official Claude export."""
    title = conv_obj.get("name") or "Untitled"
    created_at_str = conv_obj.get("created_at")
    chat_messages = conv_obj.get("chat_messages", [])

    if not chat_messages:
        return None

    # Parse conversation timestamp to epoch float
    created_at = _parse_iso_timestamp(created_at_str)

    messages = []
    for msg_obj in chat_messages:
        parsed = _extract_official_message(msg_obj)
        if parsed:
            messages.append(parsed)

    return ParsedConversation(
        title=title,
        created_at=created_at,
        messages=messages,
    )


def _extract_official_message(msg_obj: dict) -> Optional[ParsedMessage]:
    """Extract a ParsedMessage from an official Claude message object."""
    sender = msg_obj.get("sender", "")

    # Map Claude sender names to standard roles
    if sender == "human":
        role = "user"
    elif sender == "assistant":
        role = "assistant"
    else:
        return None

    # Use the text field (clean plain-text rendering, includes tool results inline)
    content = (msg_obj.get("text") or "").strip()
    if not content:
        return None

    timestamp = _parse_iso_timestamp(msg_obj.get("created_at"))

    return ParsedMessage(
        role=role,
        content=content,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Scraped Claude export parser
# ---------------------------------------------------------------------------

def parse_scraped_conversations(file_path: str | Path) -> list[ParsedConversation]:
    """Parse a scraped Claude conversations_export.json file.

    Format:
    - Top-level: JSON object with conversations array
    - Conversation fields: title, source_file, messages (flat array)
    - Message fields: role ("user"/"assistant"), content (plain text),
      timestamp (date-only string, only on assistant messages)
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or "conversations" not in raw:
        raise ValueError("Expected a JSON object with a 'conversations' array")

    conversations = []
    for conv_obj in raw["conversations"]:
        parsed = _parse_scraped_conversation(conv_obj)
        if parsed and parsed.messages:
            conversations.append(parsed)

    return conversations


def _parse_scraped_conversation(conv_obj: dict) -> Optional[ParsedConversation]:
    """Parse a single conversation from the scraped Claude export."""
    title = conv_obj.get("title") or "Untitled"
    msg_list = conv_obj.get("messages", [])

    if not msg_list:
        return None

    # Try to get a timestamp from the first assistant message for created_at
    created_at = None
    for msg_obj in msg_list:
        ts = _parse_date_timestamp(msg_obj.get("timestamp"))
        if ts is not None:
            created_at = ts
            break

    messages = []
    for msg_obj in msg_list:
        parsed = _extract_scraped_message(msg_obj)
        if parsed:
            messages.append(parsed)

    _interpolate_timestamps(messages)

    return ParsedConversation(
        title=title,
        created_at=created_at,
        messages=messages,
    )


def _extract_scraped_message(msg_obj: dict) -> Optional[ParsedMessage]:
    """Extract a ParsedMessage from a scraped Claude message object."""
    role = msg_obj.get("role", "")
    if role not in ("user", "assistant"):
        return None

    content = (msg_obj.get("content") or "").strip()
    if not content:
        return None

    timestamp = _parse_date_timestamp(msg_obj.get("timestamp"))

    return ParsedMessage(
        role=role,
        content=content,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Timestamp interpolation
# ---------------------------------------------------------------------------

def _interpolate_timestamps(messages: list[ParsedMessage]):
    """Fill in missing timestamps by interpolating from known anchors.

    The scraped data only has date-level timestamps on assistant messages.
    This spreads unknown timestamps evenly between known anchors:
      - Gap before first anchor → all get the first anchor's timestamp
      - Gap between two anchors → spread evenly across the interval
      - Gap after last anchor  → all get the last anchor's timestamp
    """
    if not messages:
        return

    # Collect indices of messages that have timestamps (anchors)
    anchors = [(i, m.timestamp) for i, m in enumerate(messages) if m.timestamp is not None]

    if not anchors:
        return  # No timestamps at all, nothing to interpolate from

    # Fill gap before the first anchor
    first_idx, first_ts = anchors[0]
    for i in range(0, first_idx):
        messages[i].timestamp = first_ts

    # Fill gaps between consecutive anchors
    for k in range(len(anchors) - 1):
        start_idx, start_ts = anchors[k]
        end_idx, end_ts = anchors[k + 1]
        gap = end_idx - start_idx
        if gap <= 1:
            continue
        for i in range(start_idx + 1, end_idx):
            # Linear interpolation
            frac = (i - start_idx) / gap
            messages[i].timestamp = start_ts + frac * (end_ts - start_ts)

    # Fill gap after the last anchor
    last_idx, last_ts = anchors[-1]
    for i in range(last_idx + 1, len(messages)):
        messages[i].timestamp = last_ts


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _parse_iso_timestamp(ts_str: Optional[str]) -> Optional[float]:
    """Parse an ISO 8601 timestamp string to a Unix epoch float."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _parse_date_timestamp(ts_str: Optional[str]) -> Optional[float]:
    """Parse a date-only string (e.g. '2025-11-21') to a Unix epoch float."""
    if not ts_str:
        return None
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        # Fall back to ISO parse in case it's a full timestamp
        return _parse_iso_timestamp(ts_str)
