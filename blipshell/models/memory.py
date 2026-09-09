"""Memory-related Pydantic models."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Type of memory entry."""
    CONVERSATION = "conversation"
    CORE = "core"
    LESSON = "lesson"
    SESSION_SUMMARY = "session_summary"
    FACT = "fact"
    EVENT = "event"
    PREFERENCE = "preference"
    SKILL = "skill"


class Tag(BaseModel):
    """A tag associated with a memory."""
    id: Optional[int] = None
    name: str
    category: str = "topic"  # topic, behavior, background


class Memory(BaseModel):
    """A single memory entry (port of C# Memory DTO)."""
    id: Optional[int] = None
    session_id: Optional[int] = None
    role: str  # "user" or "assistant"
    content: str
    summary: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rank: int = 1  # 1-5 quality/relevance rank (1 = unscored default)
    importance: float = 0.0  # 0.0 - 1.0 importance score
    tags: list[str] = Field(default_factory=list)
    memory_type: MemoryType = MemoryType.CONVERSATION
    is_archived: bool = False
    metadata_json: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    consolidated_at: Optional[datetime] = None
    entities_extracted_at: Optional[datetime] = None


# Provenance for DERIVED records (V3 B4). Raw memories carry `role`; the layers
# built from them - core memories, lessons, the user model - used to carry
# nothing, so "Jim said this" and "the model concluded this" were
# indistinguishable once distilled. Every creation site now says which.
SOURCE_TYPES = (
    "user_statement",       # the user said it (verbatim or a direct paraphrase)
    "assistant_inference",  # the model concluded/proposed it
    "tool_observation",     # a tool returned it
    "reflection",           # produced by a nightly/self-review pass over transcripts
    "import",               # brought in from an external export
    "unknown",              # pre-B4 rows
)
VERIFICATION_STATES = ("stated", "inferred", "verified", "contradicted", "unknown")


def default_verification(source_type: str) -> str:
    """The verification state a record starts in, given how it was produced."""
    return {
        "user_statement": "stated",
        "import": "stated",
        "tool_observation": "verified",
        "assistant_inference": "inferred",
        "reflection": "inferred",
    }.get(source_type, "unknown")


def provenance_tag(source_type: str, verification_state: str) -> str:
    """Rendered prefix for a derived record: '' for what the user stated,
    '[inferred] ' for a model's conclusion, '[contradicted] ' when a later
    fact overrode it, '' for pre-B4 rows (unknown - we do not invent a label)."""
    if verification_state == "inferred":
        return "[inferred] "
    if verification_state == "contradicted":
        return "[contradicted] "
    return ""


# Provenance for DERIVED records (V3 B4). Raw memories carry `role`; the layers
# built from them - core memories, lessons, the user model - used to carry
# nothing, so "Jim said this" and "the model concluded this" were
# indistinguishable once distilled. Every creation site now says which.
SOURCE_TYPES = (
    "user_statement",       # the user said it (verbatim or a direct paraphrase)
    "assistant_inference",  # the model concluded/proposed it
    "tool_observation",     # a tool returned it
    "reflection",           # produced by a nightly/self-review pass over transcripts
    "import",               # brought in from an external export
    "unknown",              # pre-B4 rows
)
VERIFICATION_STATES = ("stated", "inferred", "verified", "contradicted", "unknown")


def default_verification(source_type: str) -> str:
    """The verification state a record starts in, given how it was produced."""
    return {
        "user_statement": "stated",
        "import": "stated",
        "tool_observation": "verified",
        "assistant_inference": "inferred",
        "reflection": "inferred",
    }.get(source_type, "unknown")


def provenance_tag(source_type: str, verification_state: str) -> str:
    """Rendered prefix for a derived record: '' for what the user stated,
    '[inferred] ' for a model's conclusion, '[contradicted] ' when a later
    fact overrode it, '' for pre-B4 rows (unknown - we do not invent a label)."""
    if verification_state == "inferred":
        return "[inferred] "
    if verification_state == "contradicted":
        return "[contradicted] "
    return ""


class CoreMemory(BaseModel):
    """A persistent core memory (user preferences, facts, personality traits)."""
    id: Optional[int] = None
    content: str
    category: str = "general"  # general, preference, fact, personality
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    source_session_id: Optional[int] = None
    source_type: str = "unknown"
    verification_state: str = ""  # "" -> default_verification(source_type) at create
    source_type: str = "unknown"
    verification_state: str = ""  # "" -> default_verification(source_type) at create


class Lesson(BaseModel):
    """An extracted lesson from conversations (port of C# Lesson DTO)."""
    id: Optional[int] = None
    content: str
    summary: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rank: int = 3
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    source_session_id: Optional[int] = None
    project: Optional[str] = None  # project context for scoped lesson search
    file_id: Optional[int] = None
    added_by: str = "system"       # which path wrote it: session_review | correction_detector | user | reprocess | ...
    source_type: str = "unknown"
    verification_state: str = ""   # "" -> default_verification(source_type) at create
    added_by: str = "system"       # which path wrote it: session_review | correction_detector | user | reprocess | ...
    source_type: str = "unknown"
    verification_state: str = ""   # "" -> default_verification(source_type) at create


class MemorySearchResult(BaseModel):
    """Result from semantic memory search."""
    memory: Memory
    similarity: float  # cosine similarity score from ChromaDB
    boosted_score: float  # after importance/recency boosting


