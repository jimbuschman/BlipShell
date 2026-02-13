"""Session-specific enums and constants."""

from enum import Enum


class SessionState(str, Enum):
    """State of a session."""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ARCHIVED = "archived"
