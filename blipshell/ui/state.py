"""Per-process UI state shared between the chat loop and its commands.

These were module-level globals in cli.py, which meant any command handler
that toggled one had to live in cli.py too (a `global` statement can't reach
across modules). Holding them on one object lets handlers move out while
still mutating the same state the renderer reads.

`session_approved_tools` in particular is security-relevant: it records which
dangerous tools the user has waived confirmation for, for this process only.
It is deliberately NOT persisted — approval should not survive a restart.
"""

from dataclasses import dataclass, field


@dataclass
class UIState:
    verbose_tools: bool = False
    """Show full tool results instead of a one-line summary (/verbose)."""

    tool_batch_history: list = field(default_factory=list)
    """Recent (calls, results) batches, newest last, for /expand."""

    session_approved_tools: set[str] = field(default_factory=set)
    """Tools auto-approved for this session only (/approve)."""

    MAX_BATCH_HISTORY = 50

    def record_tool_batch(self, calls, results) -> None:
        """Append a batch, trimming in place so the list object is stable."""
        self.tool_batch_history.append((calls, results))
        if len(self.tool_batch_history) > self.MAX_BATCH_HISTORY:
            del self.tool_batch_history[:-self.MAX_BATCH_HISTORY]


ui_state = UIState()
