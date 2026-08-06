"""Slash-command registry for the interactive CLI.

Replaces a 230-line if/elif ladder in cli.py. Two things that ladder made
impossible are the reason this exists:

1. The help text was a hand-written block listing what commands exist, so it
   could silently disagree with the code — and did. Help is now DERIVED from
   the registry, so an unlisted command is impossible by construction.

2. `simulate/slash_dispatcher.py` was a hand-maintained second copy of the
   dispatch logic. Its own docstring claimed to mirror "cli.py lines 400-551
   exactly"; that line reference was long stale and it dispatched noticeably
   fewer commands than the CLI, so the simulation harness — whose job is
   catching command-parsing bugs — could not reach `/thoughts` at all.
   Simulation now drives this same registry.

A handler takes a CommandContext and returns:
  * None      — handled, prompt for the next input
  * QUIT      — leave the chat loop
  * Rewrite   — replace the input and let the normal message path handle it
                (this is how /research becomes "!research ...")

Handlers may be sync or async; the dispatcher awaits when needed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

QUIT = object()
"""Sentinel: end the chat loop."""


@dataclass
class Rewrite:
    """Replace the user's input and fall through to the normal message path."""
    text: str


@dataclass
class CommandContext:
    """Everything a handler is allowed to touch.

    `parts` is lowercased for matching; `args` preserves the user's original
    case. The old ladder parsed the raw string three different ways — lowered
    tokens, cased tokens, and manual `user_input[len("/cmd "):]` slices — which
    is three chances to disagree. `rest` is that third form, computed once.
    """
    agent: Any
    config: Any
    raw: str
    parts: list[str]
    args: list[str]
    ui: Any
    console: Any

    @property
    def name(self) -> str:
        return self.parts[0] if self.parts else ""

    @property
    def rest(self) -> str:
        """Everything after the command word, original case, stripped."""
        without_slash = self.raw[1:] if self.raw.startswith("/") else self.raw
        _, _, tail = without_slash.partition(" ")
        return tail.strip()

    def arg(self, i: int, default: str = "") -> str:
        """Lowercased positional arg (parts[0] is the command itself)."""
        return self.parts[i + 1] if len(self.parts) > i + 1 else default


@dataclass
class Command:
    names: tuple[str, ...]
    handler: Callable
    help: str
    usage: str = ""
    section: str = "General"
    hidden: bool = False

    @property
    def name(self) -> str:
        return self.names[0]

    def render_label(self) -> str:
        label = f"/{self.name}"
        if self.usage:
            label += f" {self.usage}"
        return label


class CommandRegistry:
    def __init__(self):
        self._by_name: dict[str, Command] = {}
        self._ordered: list[Command] = []

    def register(self, cmd: Command) -> None:
        for n in cmd.names:
            if n in self._by_name:
                raise ValueError(f"duplicate slash command: /{n}")
            self._by_name[n] = cmd
        self._ordered.append(cmd)

    def command(self, *names: str, help: str, usage: str = "",
                section: str = "General", hidden: bool = False):
        def deco(fn):
            self.register(Command(names, fn, help, usage, section, hidden))
            return fn
        return deco

    def get(self, name: str) -> Optional[Command]:
        return self._by_name.get(name)

    def all(self) -> list[Command]:
        return list(self._ordered)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    async def dispatch(self, ctx: CommandContext):
        """Run the matching handler. Returns None | QUIT | Rewrite.

        An unknown command reports itself rather than being silently treated
        as chat input.
        """
        cmd = self.get(ctx.name)
        if cmd is None:
            ctx.console.print(f"[yellow]Unknown command: /{ctx.name}[/yellow]")
            ctx.console.print("[dim]Use /help to see available commands.[/dim]")
            return None
        result = cmd.handler(ctx)
        if inspect.isawaitable(result):
            result = await result
        return result


registry = CommandRegistry()
