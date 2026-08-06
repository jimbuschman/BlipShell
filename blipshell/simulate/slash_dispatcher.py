"""Programmatic slash command executor for simulation.

This used to be a hand-maintained COPY of cli.py's dispatch ladder — its
docstring claimed to mirror "cli.py lines 400-551 exactly", a line reference
that was long stale, and it had drifted to dispatching noticeably fewer
commands than the real CLI. So the simulation harness, whose entire purpose is
catching command-parsing and state-mutation bugs, could not reach `/thoughts`,
`/guardrails`, `/verbose`, `/expand`, `/research`, `/cube`, `/followups`,
`/friction` or `/notes` at all (deep-dive 2026-08-04).

It now drives the SAME registry the CLI does, so drift is impossible by
construction: a command the CLI can run, simulation can run. Output is
captured to a buffer instead of the terminal.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from rich.console import Console

from blipshell.simulate.models import SlashResult
from blipshell.ui.commands import QUIT, CommandContext, Rewrite, registry
from blipshell.ui.state import UIState

import blipshell.ui.command_handlers  # noqa: F401  (registers the commands)

if TYPE_CHECKING:
    from blipshell.core.agent import Agent
    from blipshell.core.config import BlipShellConfig

logger = logging.getLogger(__name__)


class SlashCommandDispatcher:
    """Executes slash commands against an Agent programmatically."""

    def __init__(self, agent: Agent, config: BlipShellConfig):
        self.agent = agent
        self.config = config
        # Simulation gets its own UI state rather than sharing the CLI's
        # process-wide instance — a scenario toggling /verbose must not leak
        # into another scenario.
        self.ui = UIState()

    @property
    def session_approved_tools(self) -> set[str]:
        """Kept for scenarios that assert on approval state."""
        return self.ui.session_approved_tools

    def _make_console(self, buf: io.StringIO) -> Console:
        """A Rich Console that writes to a string buffer."""
        return Console(file=buf, force_terminal=False, no_color=True, width=120)

    def known_commands(self) -> list[str]:
        """Every dispatchable command name — derived, never hand-listed."""
        return registry.names()

    async def execute(self, command_str: str) -> SlashResult:
        """Execute a slash command string (e.g. '/project blipshell')."""
        if not command_str.startswith("/"):
            return SlashResult(
                command=command_str, output="", success=False,
                error="Not a slash command (must start with /)",
            )

        parts = command_str[1:].lower().split()
        if not parts:
            return SlashResult(command=command_str, output="", success=False,
                               error="Empty command")

        if registry.get(parts[0]) is None:
            return SlashResult(
                command=command_str, output="", success=False,
                error=f"Unknown command: /{parts[0]}",
            )

        buf = io.StringIO()
        ctx = CommandContext(
            agent=self.agent,
            config=self.config,
            raw=command_str,
            parts=parts,
            args=command_str[1:].split()[1:],   # original case preserved
            ui=self.ui,
            console=self._make_console(buf),
        )

        try:
            outcome = await registry.dispatch(ctx)
        except Exception as e:
            logger.warning("Slash command %s failed: %s", command_str, e)
            return SlashResult(
                command=command_str, output=buf.getvalue(), success=False,
                error=f"{type(e).__name__}: {e}",
            )

        output = buf.getvalue()
        if outcome is QUIT:
            output = output or "quit"
        elif isinstance(outcome, Rewrite):
            # The CLI would hand this to the normal message path; simulation
            # reports it rather than running an LLM turn.
            output = output or f"[rewrite] {outcome.text}"

        return SlashResult(command=command_str, output=output, success=True)
