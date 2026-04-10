"""Claude Code memory file import — structured knowledge from Claude Code's memory system.

Parses the markdown memory files stored in ~/.claude/projects/<project>/memory/.
Each project directory can contain:
  - MEMORY.md — an index file with structured knowledge (architecture, preferences, etc.)
  - Individual memory files (feedback_*.md, project_*.md, etc.) with YAML frontmatter

These are already-summarized knowledge, so they bypass the LLM summarization pipeline
and are inserted directly. Only the embedding model is needed.

Type mapping:
  Claude Code type  →  BlipShell MemoryType
  feedback          →  LESSON
  project           →  FACT
  user              →  PREFERENCE
  reference         →  FACT
  MEMORY.md index   →  CORE

Usage:
  blipshell import-claude memories <path>          # projects dir or single project
  blipshell import-claude memories <path> --dry-run
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from blipshell.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)

# Frontmatter type → BlipShell MemoryType
_TYPE_MAP = {
    "feedback": MemoryType.LESSON,
    "project": MemoryType.FACT,
    "user": MemoryType.PREFERENCE,
    "reference": MemoryType.FACT,
}

# Default importance scores per type (these are already curated knowledge)
_IMPORTANCE_MAP = {
    MemoryType.LESSON: 0.85,
    MemoryType.FACT: 0.75,
    MemoryType.PREFERENCE: 0.80,
    MemoryType.CORE: 0.90,
}

_RANK_MAP = {
    MemoryType.LESSON: 4,
    MemoryType.FACT: 3,
    MemoryType.PREFERENCE: 4,
    MemoryType.CORE: 5,
}


@dataclass
class ParsedMemory:
    """A single parsed memory file."""
    name: str
    description: str
    memory_type: MemoryType
    content: str  # Full content (used as both content and summary)
    source_file: str  # Original file path for dedup
    project_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class MemoryImportStats:
    memories_imported: int = 0
    memories_skipped: int = 0  # Already exists
    projects_scanned: int = 0
    files_scanned: int = 0
    parse_errors: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_claude_code_memories(path: str | Path) -> list[ParsedMemory]:
    """Parse Claude Code memory files from a projects directory or single project.

    Args:
        path: Either ~/.claude/projects (scans all), a single project dir,
              or a single project's memory/ subdirectory.

    Returns:
        list[ParsedMemory] ready for direct insertion.
    """
    path = Path(path)
    memory_dirs = _discover_memory_dirs(path)

    if not memory_dirs:
        logger.warning("No memory directories found at %s", path)
        return []

    all_memories: list[ParsedMemory] = []
    for mem_dir, project_name in memory_dirs:
        memories = _parse_memory_dir(mem_dir, project_name)
        all_memories.extend(memories)

    return all_memories


async def import_memories(
    sqlite,
    vectors,
    memories: list[ParsedMemory],
    stats: MemoryImportStats,
    dry_run: bool = False,
) -> MemoryImportStats:
    """Import parsed memories directly into BlipShell's storage.

    No LLM calls except embedding. Memories are inserted as pre-summarized
    knowledge with appropriate types, importance, and tags.
    """
    for mem in memories:
        stats.files_scanned += 1

        # Dedup: check if a memory with this source file already exists
        existing = await _find_existing_memory(sqlite, mem.source_file)
        if existing:
            logger.info("Skipping (already imported): %s", mem.source_file)
            stats.memories_skipped += 1
            continue

        if dry_run:
            stats.memories_imported += 1
            continue

        # Create a session to hold these memories (one per project)
        session_title = f"[Claude Code Memory] {mem.project_name or 'global'}"
        session_id = await _get_or_create_session(sqlite, session_title)

        importance = _IMPORTANCE_MAP.get(mem.memory_type, 0.7)
        rank = _RANK_MAP.get(mem.memory_type, 3)

        # Build metadata with source tracking
        import json
        metadata = json.dumps({
            "source": "claude_code_memory",
            "source_file": mem.source_file,
            "original_name": mem.name,
            "original_description": mem.description,
        })

        # Use the content as both content and summary (already condensed)
        summary = mem.content
        if mem.description:
            summary = f"{mem.description}\n\n{mem.content}"

        memory = Memory(
            session_id=session_id,
            role="assistant",  # These are system knowledge, not user messages
            content=mem.content,
            summary=summary,
            timestamp=datetime.now(timezone.utc),
            rank=rank,
            importance=importance,
            memory_type=mem.memory_type,
            metadata_json=metadata,
        )

        memory_id = await sqlite.create_memory(memory)

        # Tag with source + project + type-derived tags
        tags = ["claude-code-memory"]
        if mem.project_name:
            tags.append(f"project:{mem.project_name}")
        tags.extend(mem.tags)
        await sqlite.tag_memory(memory_id, tags)

        # Embed for vector search
        vectors.add_memory(memory_id, summary)

        # Mark as processed
        await sqlite.update_memory(memory_id, is_processed=True)

        logger.info("Imported memory %d: %s (%s)", memory_id, mem.name, mem.memory_type.value)
        stats.memories_imported += 1

    return stats


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_memory_dirs(path: Path) -> list[tuple[Path, Optional[str]]]:
    """Find all memory/ subdirectories and their project names.

    Returns list of (memory_dir_path, project_name) tuples.
    """
    results: list[tuple[Path, Optional[str]]] = []

    # If pointing directly at a memory/ directory
    if path.is_dir() and path.name == "memory":
        project_name = _project_name_from_dir(path.parent)
        results.append((path, project_name))
        return results

    if not path.is_dir():
        return results

    # Check if this is a single project dir (has memory/ subdirectory)
    mem_dir = path / "memory"
    if mem_dir.is_dir():
        project_name = _project_name_from_dir(path)
        results.append((mem_dir, project_name))
        return results

    # Top-level projects directory — scan all subdirectories
    for subdir in sorted(path.iterdir()):
        if not subdir.is_dir():
            continue
        mem_dir = subdir / "memory"
        if mem_dir.is_dir():
            project_name = _project_name_from_dir(subdir)
            results.append((mem_dir, project_name))

    return results


def _project_name_from_dir(project_dir: Path) -> Optional[str]:
    """Extract a human-readable project name from a Claude Code project directory.

    Directory names look like: C--Users-[user]-source-repos-jimbuschman-BlipShell
    We want the last meaningful component: "BlipShell"
    """
    name = project_dir.name
    # Split on -- (drive separator) and - (path separator)
    # The directory format is: C--Users-name-path-to-project
    parts = name.split("-")
    # Filter out empty strings and drive letters
    parts = [p for p in parts if p and len(p) > 1]
    if parts:
        return parts[-1]
    return name


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_memory_dir(
    mem_dir: Path, project_name: Optional[str]
) -> list[ParsedMemory]:
    """Parse all memory files in a single project's memory directory."""
    memories: list[ParsedMemory] = []

    for f in sorted(mem_dir.iterdir()):
        if not f.is_file() or f.suffix != ".md":
            continue

        try:
            if f.name == "MEMORY.md":
                mem = _parse_memory_index(f, project_name)
            else:
                mem = _parse_memory_file(f, project_name)

            if mem and mem.content.strip():
                memories.append(mem)
        except Exception as e:
            logger.error("Failed to parse %s: %s", f, e)

    return memories


def _parse_memory_index(
    file_path: Path, project_name: Optional[str]
) -> Optional[ParsedMemory]:
    """Parse a MEMORY.md index file as a single CORE memory."""
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    # MEMORY.md has no frontmatter — it's the index itself
    # Strip any frontmatter if present (shouldn't be, but be safe)
    content = _strip_frontmatter(text)
    if not content.strip():
        return None

    title = f"{project_name} project overview" if project_name else "Project overview"

    return ParsedMemory(
        name=title,
        description=f"Claude Code memory index for {project_name or 'unknown project'}",
        memory_type=MemoryType.CORE,
        content=content,
        source_file=str(file_path),
        project_name=project_name,
        tags=["overview", "claude-code-index"],
    )


def _parse_memory_file(
    file_path: Path, project_name: Optional[str]
) -> Optional[ParsedMemory]:
    """Parse a single memory file with YAML frontmatter."""
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    frontmatter, content = _extract_frontmatter(text)
    if not content.strip():
        return None

    name = frontmatter.get("name", file_path.stem)
    description = frontmatter.get("description", "")
    fm_type = frontmatter.get("type", "").lower()

    memory_type = _TYPE_MAP.get(fm_type, MemoryType.FACT)

    # Derive tags from the filename and type
    tags: list[str] = []
    if fm_type:
        tags.append(fm_type)
    # Extract topic from filename (e.g., "feedback_code_review_quality" → "code-review-quality")
    stem = file_path.stem
    for prefix in ("feedback_", "project_", "user_", "reference_"):
        if stem.startswith(prefix):
            topic = stem[len(prefix):].replace("_", "-")
            tags.append(topic)
            break

    return ParsedMemory(
        name=name,
        description=description,
        memory_type=memory_type,
        content=content,
        source_file=str(file_path),
        project_name=project_name,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _extract_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract YAML-like frontmatter and body from a markdown file.

    Returns (frontmatter_dict, body_content).
    Uses simple key: value parsing (no full YAML dependency needed).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    fm_text = match.group(1)
    body = text[match.end():]

    # Simple key: value parsing
    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip().strip('"').strip("'")

    return fm, body


def _strip_frontmatter(text: str) -> str:
    """Remove frontmatter from text if present."""
    _, body = _extract_frontmatter(text)
    return body


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------

# Cache session IDs to avoid repeated lookups
_session_cache: dict[str, int] = {}


async def _get_or_create_session(sqlite, title: str) -> int:
    """Get existing session by title or create a new one."""
    if title in _session_cache:
        return _session_cache[title]

    session_id, count = await sqlite.get_session_message_count(title)
    if session_id is not None:
        _session_cache[title] = session_id
        return session_id

    session_id = await sqlite.create_session(title=title)
    _session_cache[title] = session_id
    return session_id


async def _find_existing_memory(sqlite, source_file: str) -> bool:
    """Check if a memory with this source_file has already been imported."""
    # Search by metadata_json containing the source file
    row = await sqlite._db.execute_fetchall(
        "SELECT id FROM memories WHERE metadata_json LIKE ? LIMIT 1",
        (f'%{source_file.replace(chr(92), chr(92)*2)}%',),
    )
    return len(row) > 0
