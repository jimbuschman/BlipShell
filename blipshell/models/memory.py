"""Memory-related Pydantic models."""

from datetime import datetime
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
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rank: int = 0  # 1-5 quality/relevance rank
    importance: float = 0.0  # 0.0 - 1.0 importance score
    tags: list[str] = Field(default_factory=list)
    memory_type: MemoryType = MemoryType.CONVERSATION
    is_archived: bool = False
    metadata_json: Optional[str] = None


class CoreMemory(BaseModel):
    """A persistent core memory (user preferences, facts, personality traits)."""
    id: Optional[int] = None
    content: str
    category: str = "general"  # general, preference, fact, personality
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    source_session_id: Optional[int] = None


class Lesson(BaseModel):
    """An extracted lesson from conversations (port of C# Lesson DTO)."""
    id: Optional[int] = None
    content: str
    summary: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
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


class MemoryItem(BaseModel):
    """Generic memory item for the token budget pool system."""
    id: str  # unique identifier
    content: str
    token_count: int = 0
    priority: int = 0  # higher = more important to keep
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = ""  # which pool this belongs to
    metadata: dict = Field(default_factory=dict)


class MemoryPool(BaseModel):
    """Configuration for a memory token budget pool."""
    name: str
    percentage: float  # fraction of total budget
    max_tokens: Optional[int] = None  # hard cap
    priority: int = 0  # rollover priority (higher = gets leftover tokens first)
    current_tokens: int = 0
    items: list[MemoryItem] = Field(default_factory=list)

    @property
    def budget(self) -> int:
        """Calculate budget from total context and percentage."""
        return self.current_tokens

    def calculate_budget(self, available_tokens: int) -> int:
        """Calculate token budget based on available tokens and percentage."""
        budget = int(available_tokens * self.percentage)
        if self.max_tokens is not None:
            budget = min(budget, self.max_tokens)
        self.current_tokens = budget
        return budget
