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
    rank: int = 0  # 1-5 quality/relevance rank
    importance: float = 0.0  # 0.0 - 1.0 importance score
    tags: list[str] = Field(default_factory=list)
    memory_type: MemoryType = MemoryType.CONVERSATION
    is_archived: bool = False
    metadata_json: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class CoreMemory(BaseModel):
    """A persistent core memory (user preferences, facts, personality traits)."""
    id: Optional[int] = None
    content: str
    category: str = "general"  # general, preference, fact, personality
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    source_session_id: Optional[int] = None


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
    file_id: Optional[int] = None


class MemorySearchResult(BaseModel):
    """Result from semantic memory search."""
    memory: Memory
    similarity: float  # cosine similarity score from ChromaDB
    boosted_score: float  # after importance/recency boosting


