"""Path parser and tier permission rules for the memory filesystem.

Translates LLM-facing /memories/{tier}/... paths into structured data the
backend can act on. Enforces tier permissions and validates against
traversal, character whitelists, and depth caps.

Tier layout:
    /memories/lessons/<project>/<id>-<slug>.md   (read-only, pipeline-derived)
    /memories/core/<id>-<slug>.md                (read/write, requires approval)
    /memories/digests/<project>.md               (read-only, auto-generated)
    /memories/sessions/<id>-<slug>.md            (read-only, historical)
    /memories/friction/<id>-<slug>.md            (read-only, system-observed)
    /memories/notes/<name>.md                    (read/write, current-session notes)

The notes tier is a filesystem view over the existing session_notes store
(sessions.metadata_json) — the same data the save_note/get_notes tools use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Tier(str, Enum):
    LESSONS = "lessons"
    CORE = "core"
    DIGESTS = "digests"
    SESSIONS = "sessions"
    FRICTION = "friction"
    NOTES = "notes"


# Per-tier permission table. `approval_for` lists operations that require
# the agent's approval callback (mirrors the existing tools_requiring_approval
# pattern but at path granularity).
#
# Lessons are READ-ONLY: they are pipeline-derived (extracted + scored + embedded)
# and hand-edits would bypass scoring and pollute the corpus. Working notes go
# through the NOTES tier (backed by the existing session_notes store).
TIER_PERMISSIONS: dict[Tier, dict] = {
    Tier.LESSONS:  {"read": True, "create": False, "edit": False, "delete": False,
                    "approval_for": frozenset()},
    Tier.CORE:     {"read": True, "create": True,  "edit": True,  "delete": True,
                    "approval_for": frozenset({"create", "edit", "delete"})},
    Tier.DIGESTS:  {"read": True, "create": False, "edit": False, "delete": False,
                    "approval_for": frozenset()},
    Tier.SESSIONS: {"read": True, "create": False, "edit": False, "delete": False,
                    "approval_for": frozenset()},
    Tier.FRICTION: {"read": True, "create": False, "edit": False, "delete": False,
                    "approval_for": frozenset()},
    Tier.NOTES:    {"read": True, "create": True,  "edit": True,  "delete": True,
                    "approval_for": frozenset()},
}


MAX_PATH_DEPTH = 5
MAX_SLUG_LENGTH = 60

# Filename forms we accept. Order matters — try id-first then plain.
_FILENAME_WITH_ID = re.compile(r"^(\d+)(?:-([a-z0-9_-]+))?\.md$")
_FILENAME_PLAIN = re.compile(r"^([a-z0-9_-]+)\.md$")
_SEGMENT = re.compile(r"^[a-z0-9_-]+$")


class PathError(ValueError):
    """Invalid /memories/ path. Message is intended for LLM consumption."""


@dataclass(frozen=True)
class MemoryPath:
    """Structured representation of a /memories/... path.

    Field meaning per tier:
        lessons:  project (None for tier root / set for project subdir)
                  filename (None for directory listing)
        core:     filename
        digests:  filename (which is the project name)
        sessions: filename
        friction: filename
        notes:    filename (slug carries the note name / key)

    is_root = path is '/memories' or '/memories/' (no tier yet).
    is_directory = tier is set but filename is None (list mode).
    """

    raw: str
    tier: Optional[Tier]
    project: Optional[str] = None
    filename: Optional[str] = None
    file_id: Optional[int] = None
    slug: Optional[str] = None

    @property
    def is_root(self) -> bool:
        return self.tier is None

    @property
    def is_directory(self) -> bool:
        return self.tier is not None and self.filename is None


def slugify(text: str, max_length: int = MAX_SLUG_LENGTH) -> str:
    """Generate a stable kebab-case slug from arbitrary text.

    Lowercases, collapses non-alphanumeric runs to single hyphens, trims
    leading/trailing hyphens, caps length at word boundaries when possible.
    Returns 'untitled' if input has no slug-able characters.
    """
    if not text:
        return "untitled"
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    if not s:
        return "untitled"
    if len(s) <= max_length:
        return s
    # Try to cut at a hyphen boundary near the limit.
    cut = s.rfind("-", 0, max_length)
    return (s[:cut] if cut > max_length // 2 else s[:max_length]).strip("-") or "untitled"


def build_filename(item_id: int, source_text: str) -> str:
    """Compose '<id>-<slug>.md' for a stored item."""
    return f"{item_id}-{slugify(source_text)}.md"


def parse(raw: str) -> MemoryPath:
    """Parse a path string into MemoryPath. Raises PathError on invalid input.

    Accepts:
        /memories                          -> root, lists tiers
        /memories/                         -> root, lists tiers
        /memories/<tier>                   -> tier listing
        /memories/<tier>/                  -> tier listing
        /memories/lessons/<project>        -> project listing
        /memories/lessons/<project>/       -> project listing
        /memories/lessons/<project>/<file>.md
        /memories/core/<file>.md
        /memories/digests/<project>.md
        /memories/sessions/<file>.md
        /memories/friction/<file>.md
        /memories/scratch/<session_id>     -> session-scoped listing
        /memories/scratch/<session_id>/<name>.md
    """
    if raw is None:
        raise PathError("Path is required")
    s = raw.strip()
    if not s:
        raise PathError("Path is empty")

    # Reject traversal upfront — these never have legitimate meaning here.
    if ".." in s.split("/"):
        raise PathError("Path traversal (..) is not allowed")
    if "\\" in s:
        raise PathError("Use forward slashes (/) in memory paths")
    if "\x00" in s:
        raise PathError("Null bytes not allowed in path")

    # Normalize: ensure leading slash, strip trailing slash for splitting
    if not s.startswith("/"):
        raise PathError(f"Path must start with /memories (got {raw!r})")
    stripped = s.rstrip("/")

    parts = [p for p in stripped.split("/") if p]
    if not parts or parts[0] != "memories":
        raise PathError(f"Path must start with /memories (got {raw!r})")

    if len(parts) > MAX_PATH_DEPTH:
        raise PathError(f"Path too deep (max {MAX_PATH_DEPTH} segments)")

    # /memories or /memories/
    if len(parts) == 1:
        return MemoryPath(raw=raw, tier=None)

    # /memories/<tier>...
    tier_str = parts[1]
    try:
        tier = Tier(tier_str)
    except ValueError:
        valid = ", ".join(t.value for t in Tier)
        raise PathError(f"Unknown tier {tier_str!r}. Valid tiers: {valid}")

    if len(parts) == 2:
        return MemoryPath(raw=raw, tier=tier)

    # Validate intermediate path segments (everything except the final filename).
    for seg in parts[2:-1]:
        if not _SEGMENT.match(seg):
            raise PathError(
                f"Invalid path segment {seg!r}. Use lowercase letters, "
                "digits, hyphens, underscores only."
            )

    last = parts[-1]
    last_is_file = last.endswith(".md")

    if tier == Tier.LESSONS:
        return _parse_lessons(raw, parts, last, last_is_file)
    elif tier == Tier.NOTES:
        return _parse_notes(raw, parts, last, last_is_file)
    elif tier in (Tier.CORE, Tier.SESSIONS, Tier.FRICTION):
        # /memories/<tier>/<file>.md only — no subdirectories
        if len(parts) != 3:
            raise PathError(
                f"{tier.value} does not support subdirectories. "
                f"Use /memories/{tier.value}/<file>.md"
            )
        if not last_is_file:
            raise PathError(f"{tier.value} entries must end with .md")
        file_id, slug = _parse_filename(last)
        return MemoryPath(raw=raw, tier=tier, filename=last,
                          file_id=file_id, slug=slug)
    elif tier == Tier.DIGESTS:
        # /memories/digests/<project>.md
        if len(parts) != 3:
            raise PathError(
                "digests are flat: /memories/digests/<project>.md"
            )
        if not last_is_file:
            raise PathError("Digest paths must end with .md")
        # Digest filename IS the project name (no id-prefix).
        slug = last[:-3]  # strip .md
        if not _SEGMENT.match(slug):
            raise PathError(
                f"Invalid digest filename {last!r}. "
                "Format: /memories/digests/<project>.md"
            )
        return MemoryPath(raw=raw, tier=tier, project=slug, filename=last, slug=slug)
    else:
        raise PathError(f"Tier {tier.value} parsing not implemented")  # pragma: no cover


def _parse_lessons(raw: str, parts: list[str], last: str, last_is_file: bool) -> MemoryPath:
    # /memories/lessons/<project> or /memories/lessons/<project>/
    if len(parts) == 3 and not last_is_file:
        if not _SEGMENT.match(last):
            raise PathError(f"Invalid project name {last!r}")
        return MemoryPath(raw=raw, tier=Tier.LESSONS, project=last)
    # /memories/lessons/<project>/<file>.md
    if len(parts) == 4 and last_is_file:
        project = parts[2]
        if not _SEGMENT.match(project):
            raise PathError(f"Invalid project name {project!r}")
        file_id, slug = _parse_filename(last)
        return MemoryPath(raw=raw, tier=Tier.LESSONS, project=project,
                          filename=last, file_id=file_id, slug=slug)
    raise PathError(
        "Invalid lessons path. Expected /memories/lessons/<project> or "
        "/memories/lessons/<project>/<file>.md"
    )


def _parse_notes(raw: str, parts: list[str], last: str, last_is_file: bool) -> MemoryPath:
    """Notes are flat and current-session-implicit: /memories/notes/<name>.md.

    The slug field carries the note name (the key into session_notes).
    """
    # /memories/notes/<name>.md only (tier listing handled earlier when len==2).
    if len(parts) != 3:
        raise PathError(
            "Notes are flat: /memories/notes/<name>.md (no subdirectories)"
        )
    if not last_is_file:
        raise PathError("Note paths must end with .md")
    stem = last[:-3]
    if not _SEGMENT.match(stem):
        raise PathError(
            f"Invalid note name {last!r}. Use lowercase letters, digits, "
            "hyphens, underscores; must end with .md"
        )
    return MemoryPath(raw=raw, tier=Tier.NOTES, filename=last, slug=stem)


def _parse_filename(filename: str) -> tuple[Optional[int], Optional[str]]:
    """Parse '<id>-<slug>.md' or '<slug>.md'. Returns (id, slug)."""
    m = _FILENAME_WITH_ID.match(filename)
    if m:
        return int(m.group(1)), m.group(2)
    m = _FILENAME_PLAIN.match(filename)
    if m:
        return None, m.group(1)
    raise PathError(
        f"Invalid filename {filename!r}. Use '<id>-<slug>.md' or '<slug>.md' "
        "with lowercase letters, digits, hyphens, underscores."
    )


def check_permission(tier: Tier, operation: str) -> bool:
    """Return True if `operation` is permitted on `tier`.

    operation: one of 'read', 'create', 'edit', 'delete'.
    """
    perms = TIER_PERMISSIONS.get(tier)
    if perms is None:
        return False
    return bool(perms.get(operation, False))


def requires_approval(tier: Tier, operation: str) -> bool:
    """True if this (tier, operation) pair needs the approval callback."""
    perms = TIER_PERMISSIONS.get(tier)
    if perms is None:
        return False
    return operation in perms["approval_for"]
