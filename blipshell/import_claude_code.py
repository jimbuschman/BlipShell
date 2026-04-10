"""Claude Code JSONL import — conversations from Claude Code CLI sessions.

Parses the JSONL conversation logs stored in ~/.claude/projects/<project>/*.jsonl.
Each JSONL file is one session containing user prompts, assistant responses with
tool calls (Read, Edit, Bash, Grep, etc.), and tool results.

The parser extracts:
  - User text messages (plain prompts)
  - Assistant text responses (explanations, summaries)
  - Tool usage is included as concise annotations within the assistant message
    (tool name + key inputs) so the memory pipeline can learn from what actions
    were taken, not just what was said.
  - Tool results are included only for short outputs and errors.

Produces list[ParsedConversation] for the shared import pipeline.

Usage:
  blipshell import-claude code <path>            # directory or single .jsonl
  blipshell import-claude code <path> --max 5    # limit to first 5 sessions
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from blipshell.import_common import ParsedConversation, ParsedMessage

logger = logging.getLogger(__name__)

# Maximum characters of tool result to include (keeps noise down)
_MAX_TOOL_RESULT_LEN = 500
# Tool result content shorter than this is always included
_MIN_TOOL_RESULT_LEN = 200


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_claude_code_sessions(
    path: str | Path,
) -> list[ParsedConversation]:
    """Parse Claude Code JSONL conversation files.

    Args:
        path: Either a directory containing .jsonl files (e.g. a project folder
              inside ~/.claude/projects/) or a single .jsonl file.
              If a top-level projects directory is given, all project
              subdirectories are scanned recursively.

    Returns:
        list[ParsedConversation] ready for the shared import pipeline.
    """
    path = Path(path)
    jsonl_files = _discover_jsonl_files(path)

    if not jsonl_files:
        logger.warning("No .jsonl files found at %s", path)
        return []

    # Try to load session indexes for title lookups
    title_map = _build_title_map(path)

    conversations: list[ParsedConversation] = []
    for jsonl_path in jsonl_files:
        try:
            conv = _parse_jsonl_file(jsonl_path, title_map)
            if conv and conv.messages:
                conversations.append(conv)
        except Exception as e:
            logger.error("Failed to parse %s: %s", jsonl_path, e)

    # Sort by created_at (oldest first) for consistent import order
    conversations.sort(key=lambda c: c.created_at or 0)
    return conversations


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_jsonl_files(path: Path) -> list[Path]:
    """Find all Claude Code conversation JSONL files at the given path."""
    if path.is_file() and path.suffix == ".jsonl":
        return [path]

    if not path.is_dir():
        return []

    jsonl_files: list[Path] = []

    # Check if this looks like the top-level .claude/projects directory
    # (contains subdirectories named like C--Users-...) or a single project dir
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    has_project_subdirs = any(
        d.name.startswith("C-") or d.name.startswith("D-")
        for d in subdirs
    )

    if has_project_subdirs:
        # Top-level projects directory — scan all subdirectories
        for subdir in subdirs:
            for f in subdir.iterdir():
                if f.is_file() and f.suffix == ".jsonl":
                    jsonl_files.append(f)
    else:
        # Single project directory
        for f in path.iterdir():
            if f.is_file() and f.suffix == ".jsonl":
                jsonl_files.append(f)

    return sorted(jsonl_files)


# ---------------------------------------------------------------------------
# Session index for titles
# ---------------------------------------------------------------------------

def _build_title_map(path: Path) -> dict[str, str]:
    """Build a session_id -> summary/title map from sessions-index.json files."""
    title_map: dict[str, str] = {}
    search_dirs: list[Path] = []

    if path.is_file():
        search_dirs.append(path.parent)
    elif path.is_dir():
        # Check subdirectories too
        search_dirs.append(path)
        for d in path.iterdir():
            if d.is_dir():
                search_dirs.append(d)

    for d in search_dirs:
        index_path = d / "sessions-index.json"
        if not index_path.exists():
            continue
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            for entry in index.get("entries", []):
                sid = entry.get("sessionId", "")
                summary = entry.get("summary", "")
                if sid and summary:
                    title_map[sid] = summary
        except Exception as e:
            logger.warning("Failed to read session index %s: %s", index_path, e)

    return title_map


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

def _parse_jsonl_file(
    jsonl_path: Path,
    title_map: dict[str, str],
) -> Optional[ParsedConversation]:
    """Parse a single Claude Code JSONL conversation file.

    Consecutive assistant messages are merged into a single turn so that
    tool-only messages (e.g. a Read followed by another Read) don't create
    many tiny low-value memories. The merged message contains the assistant's
    text plus concise tool annotations, giving the memory pipeline a coherent
    picture of what the assistant did in each turn.
    """
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()

    session_id = jsonl_path.stem  # UUID filename without .jsonl
    raw_messages: list[ParsedMessage] = []
    first_timestamp: Optional[float] = None
    first_user_prompt: Optional[str] = None
    project_path: Optional[str] = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type", "")

        # Skip non-message types (progress, file-history-snapshot, system, etc.)
        if msg_type not in ("user", "assistant"):
            continue

        timestamp = _parse_iso_timestamp(obj.get("timestamp"))
        if first_timestamp is None and timestamp is not None:
            first_timestamp = timestamp

        # Extract project path from first user message
        if project_path is None and msg_type == "user":
            project_path = obj.get("cwd")

        message = obj.get("message", {})
        role = message.get("role", "")
        if role not in ("user", "assistant"):
            continue

        content = _extract_content(message, role)
        if not content or not content.strip():
            continue

        if first_user_prompt is None and role == "user":
            first_user_prompt = content[:120]

        raw_messages.append(ParsedMessage(
            role=role,
            content=content,
            timestamp=timestamp,
        ))

    if not raw_messages:
        return None

    # Merge consecutive same-role messages into single turns.
    # This is critical for assistant messages: Claude Code often emits
    # many assistant lines per turn (text, tool_use, text, tool_use, ...),
    # and we want one coherent memory per assistant turn, not dozens of
    # "[Tool: Read] file: ..." fragments.
    messages = _merge_consecutive(raw_messages)

    # Determine title: session index summary > first user prompt > session ID
    title = title_map.get(session_id, "")
    if not title:
        title = _make_title_from_prompt(first_user_prompt) if first_user_prompt else session_id

    # Prefix with project name for context
    if project_path:
        project_name = Path(project_path).name
        title = f"[{project_name}] {title}"

    return ParsedConversation(
        title=title,
        created_at=first_timestamp,
        messages=messages,
    )


def _merge_consecutive(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    """Merge consecutive messages with the same role into single turns.

    For assistant messages this is essential: Claude Code emits separate JSONL
    lines for each streaming chunk (text block, tool_use block, etc.), but
    they're all part of a single assistant turn. Merging them produces one
    coherent memory per turn instead of many fragments.

    Tool-only content (e.g. just "[Tool: Read] file: ...") is folded into
    the surrounding text. If an entire merged assistant turn is nothing but
    tool annotations with no explanatory text, it's still kept — the tool
    sequence itself is useful context for learning.
    """
    if not messages:
        return []

    merged: list[ParsedMessage] = []
    current = messages[0]

    for msg in messages[1:]:
        if msg.role == current.role:
            # Merge: append content, keep earliest timestamp
            current = ParsedMessage(
                role=current.role,
                content=current.content + "\n\n" + msg.content,
                timestamp=current.timestamp or msg.timestamp,
            )
        else:
            merged.append(current)
            current = msg

    merged.append(current)
    return merged


def _extract_content(message: dict, role: str) -> str:
    """Extract readable text content from a Claude Code message.

    For user messages: extracts the text (either plain string or content blocks).
    For assistant messages: extracts text blocks and includes concise tool usage
    annotations so the memory pipeline captures what actions were taken.
    """
    raw_content = message.get("content", "")

    # User messages: content can be a plain string or a list of content blocks
    if role == "user":
        if isinstance(raw_content, str):
            return raw_content.strip()
        if isinstance(raw_content, list):
            # User messages with tool_result blocks are responses to tool calls.
            # We skip these — the interesting content was in the assistant's tool_use.
            has_tool_result = any(
                isinstance(c, dict) and c.get("type") == "tool_result"
                for c in raw_content
            )
            if has_tool_result:
                return ""
            # Extract any text blocks
            parts = []
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts).strip()
        return ""

    # Assistant messages: content is always a list of content blocks
    if not isinstance(raw_content, list):
        return str(raw_content).strip() if raw_content else ""

    parts: list[str] = []
    for block in raw_content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type", "")

        if block_type == "text":
            text = (block.get("text") or "").strip()
            if text:
                parts.append(text)

        elif block_type == "tool_use":
            tool_desc = _format_tool_use(block)
            if tool_desc:
                parts.append(tool_desc)

        # Skip thinking blocks — internal reasoning, not actionable content

    return "\n\n".join(parts)


def _format_tool_use(block: dict) -> str:
    """Format a tool_use block as a concise annotation.

    Examples:
        [Tool: Read] file: src/controllers/AuthController.cs
        [Tool: Edit] file: src/models/User.cs
        [Tool: Bash] command: dotnet build
        [Tool: Grep] pattern: "async Task" in *.cs
    """
    name = block.get("name", "unknown")
    inp = block.get("input", {})

    if name == "Read":
        path = inp.get("file_path", "")
        return f"[Tool: Read] file: {_short_path(path)}"

    elif name == "Edit":
        path = inp.get("file_path", "")
        old = (inp.get("old_string") or "")[:80]
        new = (inp.get("new_string") or "")[:80]
        return f"[Tool: Edit] file: {_short_path(path)} | old: {old!r} -> new: {new!r}"

    elif name == "Write":
        path = inp.get("file_path", "")
        content_len = len(inp.get("content", ""))
        return f"[Tool: Write] file: {_short_path(path)} ({content_len} chars)"

    elif name == "Bash":
        cmd = (inp.get("command") or "")[:200]
        return f"[Tool: Bash] {cmd}"

    elif name == "Grep":
        pattern = inp.get("pattern", "")
        glob = inp.get("glob", "")
        path = inp.get("path", "")
        desc = f"[Tool: Grep] pattern: {pattern!r}"
        if glob:
            desc += f" in {glob}"
        if path:
            desc += f" at {_short_path(path)}"
        return desc

    elif name == "Glob":
        pattern = inp.get("pattern", "")
        return f"[Tool: Glob] {pattern}"

    elif name == "Agent":
        desc = inp.get("description", inp.get("prompt", "")[:100])
        return f"[Tool: Agent] {desc}"

    else:
        # Generic: just show the tool name and input keys
        keys = list(inp.keys())[:5]
        return f"[Tool: {name}] {', '.join(keys)}"


def _short_path(path: str) -> str:
    """Shorten a file path to the last 3 components for readability."""
    if not path:
        return ""
    parts = Path(path).parts
    if len(parts) <= 3:
        return path
    return str(Path(*parts[-3:]))


# ---------------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------------

def _make_title_from_prompt(prompt: str) -> str:
    """Create a session title from the first user prompt."""
    # Clean up and truncate
    title = prompt.replace("\n", " ").strip()
    if len(title) > 80:
        title = title[:77] + "..."
    return title


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
