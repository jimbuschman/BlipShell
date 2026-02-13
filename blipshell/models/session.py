"""Session-related Pydantic models."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SessionMessage(BaseModel):
    """A single message in a session (port of C# SessionMessage)."""
    id: Optional[int] = None
    session_id: Optional[int] = None
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: int = 0
    tool_calls: Optional[list[dict]] = None  # native Ollama tool call data
    tool_call_id: Optional[str] = None  # for tool response messages
    is_summarized: bool = False

    def to_ollama_message(self) -> dict:
        """Convert to Ollama message format."""
        msg = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


class Session(BaseModel):
    """A conversation session (port of C# ConversationMemory)."""
    id: Optional[int] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    project: Optional[str] = None  # named project context
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    is_archived: bool = False
    message_count: int = 0
    metadata_json: Optional[str] = None
    messages: list[SessionMessage] = Field(default_factory=list)
