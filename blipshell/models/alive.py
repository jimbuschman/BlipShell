"""Pydantic models for BlipShell Alive — thoughts, identity, initiative."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ThoughtCategory(str, Enum):
    BELIEF = "belief"
    OPINION = "opinion"
    OBSERVATION = "observation"
    QUESTION = "question"
    PREFERENCE = "preference"
    PATTERN = "pattern"


class ThoughtSource(str, Enum):
    REFLECTION = "reflection"      # post-session reflection
    MONOLOGUE = "monologue"        # inner monologue cycle
    SESSION = "session"            # during active session
    CORRECTION = "correction"      # user correction detected


class InitiativeCategory(str, Enum):
    QUESTION = "question"
    REVISIT = "revisit"
    OBSERVATION = "observation"
    FOLLOW_UP = "follow_up"


class InitiativeStatus(str, Enum):
    PENDING = "pending"
    RAISED = "raised"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


class Thought(BaseModel):
    id: Optional[int] = None
    content: str
    category: ThoughtCategory = ThoughtCategory.OBSERVATION
    confidence: float = 0.5
    source_type: ThoughtSource = ThoughtSource.REFLECTION
    source_session_id: Optional[int] = None
    source_memory_id: Optional[int] = None
    parent_thought_id: Optional[int] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class InitiativeItem(BaseModel):
    id: Optional[int] = None
    content: str
    category: InitiativeCategory = InitiativeCategory.QUESTION
    priority: float = 0.5
    source_type: str = "monologue"
    source_thought_id: Optional[int] = None
    status: InitiativeStatus = InitiativeStatus.PENDING
    raised_session_id: Optional[int] = None
    created_at: Optional[datetime] = None
    raised_at: Optional[datetime] = None


class IdentityVersion(BaseModel):
    id: Optional[int] = None
    version_number: int
    content: str
    trigger: str = "nightly"
    thought_count: int = 0
    created_at: Optional[datetime] = None


class MonologueCycleResult(BaseModel):
    """Result of one inner monologue cycle."""
    cycle_number: int
    memories_reviewed: int = 0
    thoughts_generated: int = 0
    thoughts_refined: int = 0
    initiative_items_added: int = 0
    tool_calls_made: int = 0
    next_focus: str | None = None
    elapsed_s: float = 0.0
