"""The slash-command registry, and the two drift bugs it exists to prevent.

Before this, `/help` was a hand-written string listing what commands existed,
and `simulate/slash_dispatcher.py` was a hand-maintained copy of the dispatch
ladder. Both could disagree with the code, and both did: the simulate copy had
drifted to dispatching fewer commands than the CLI, so the harness whose job is
catching command-parsing bugs couldn't reach `/thoughts` at all
(deep-dive 2026-08-04).

Both are now derived from one registry, so the drift is structurally
impossible — these tests assert that property rather than a snapshot.
"""

import io

import pytest
from rich.console import Console

from blipshell.ui.commands import (
    QUIT, Command, CommandContext, CommandRegistry, Rewrite, registry,
)
from blipshell.ui.command_handlers import render_help
from blipshell.ui.state import UIState


def _ctx(raw, agent=None, config=None, ui=None, buf=None):
    parts = raw[1:].lower().split()
    return CommandContext(
        agent=agent, config=config, raw=raw, parts=parts,
        args=raw[1:].split()[1:], ui=ui or UIState(),
        console=Console(file=buf or io.StringIO(), no_color=True, width=100),
    )


class TestContextParsing:
    """The old ladder parsed the same input three ways — lowered tokens, cased
    tokens, and manual `user_input[len("/cmd "):]` slices. Three chances to
    disagree; `rest` is that third form computed once."""

    def test_rest_preserves_case_and_spacing(self):
        ctx = _ctx("/feedback Too Verbose, keep it SHORT")
        assert ctx.rest == "Too Verbose, keep it SHORT"

    def test_rest_is_empty_without_args(self):
        assert _ctx("/feedback").rest == ""

    def test_parts_are_lowercased_for_matching(self):
        assert _ctx("/Project INFO").parts == ["project", "info"]

    def test_args_preserve_case(self):
        assert _ctx("/project MyProject").args == ["MyProject"]

    def test_arg_helper_indexes_past_the_command(self):
        ctx = _ctx("/think ON")
        assert ctx.arg(0) == "on"
        assert ctx.arg(1, "fallback") == "fallback"

    def test_rest_handles_extra_whitespace(self):
        assert _ctx("/code   src/app.py  fix it").rest == "src/app.py  fix it"


class TestDispatch:
    async def test_unknown_command_is_reported_not_silently_ignored(self):
        buf = io.StringIO()
        r = CommandRegistry()
        out = await r.dispatch(_ctx("/nope", buf=buf))
        assert out is None
        assert "Unknown command" in buf.getvalue()

    async def test_sync_and_async_handlers_both_work(self):
        r = CommandRegistry()
        seen = []

        @r.command("sync", help="s")
        def _s(ctx):
            seen.append("sync")

        @r.command("async", help="a")
        async def _a(ctx):
            seen.append("async")

        await r.dispatch(_ctx("/sync"))
        await r.dispatch(_ctx("/async"))
        assert seen == ["sync", "async"]

    async def test_quit_sentinel_propagates(self):
        r = CommandRegistry()

        @r.command("bye", help="q")
        def _b(ctx):
            return QUIT

        assert await r.dispatch(_ctx("/bye")) is QUIT

    async def test_rewrite_propagates(self):
        """How /research becomes a normal '!research ...' message."""
        r = CommandRegistry()

        @r.command("go", help="g")
        def _g(ctx):
            return Rewrite("!research " + ctx.rest)

        out = await r.dispatch(_ctx("/go quantum widgets"))
        assert isinstance(out, Rewrite)
        assert out.text == "!research quantum widgets"

    async def test_aliases_resolve_to_the_same_command(self):
        r = CommandRegistry()
        hits = []

        @r.command("quit", "exit", "q", help="quit")
        def _q(ctx):
            hits.append(ctx.name)

        for name in ("/quit", "/exit", "/q"):
            await r.dispatch(_ctx(name))
        assert len(hits) == 3

    def test_duplicate_registration_is_rejected(self):
        """Two commands claiming one name is a silent shadowing bug; the old
        elif ladder would just never reach the second."""
        r = CommandRegistry()
        r.register(Command(("dup",), lambda ctx: None, "first"))
        with pytest.raises(ValueError, match="duplicate"):
            r.register(Command(("dup",), lambda ctx: None, "second"))


class TestHelpIsDerived:
    """/help can no longer disagree with what's dispatchable."""

    def test_every_registered_command_appears_in_help(self):
        rendered = render_help().renderable
        missing = [
            c.name for c in registry.all()
            if not c.hidden and f"/{c.name}" not in rendered
        ]
        assert not missing, f"commands missing from /help: {missing}"

    def test_help_lists_no_command_that_does_not_exist(self):
        import re

        rendered = render_help().renderable
        # Command labels sit at the start of a line after the bold tag
        listed = set(re.findall(r"\[bold\]/(\w+)", rendered))
        unknown = {n for n in listed if registry.get(n) is None}
        assert not unknown, f"/help advertises non-existent commands: {unknown}"

    def test_aliases_are_shown(self):
        rendered = render_help().renderable
        assert "/exit" in rendered and "/q" in rendered

    def test_usage_brackets_are_escaped_not_eaten_as_markup(self):
        """"[on|off]" is literal text; Rich would otherwise read it as a tag
        and drop it, silently breaking both the hint and the alignment."""
        console = Console(file=io.StringIO(), no_color=True, width=200)
        console.print(render_help())
        out = console.file.getvalue()
        assert "[on|off]" in out


class TestSimulationSharesTheRegistry:
    """The bug this whole extraction exists to kill."""

    def test_simulate_dispatches_everything_the_cli_does(self):
        from blipshell.simulate.slash_dispatcher import SlashCommandDispatcher

        d = SlashCommandDispatcher(agent=None, config=None)
        assert set(d.known_commands()) == set(registry.names())

    def test_previously_unreachable_commands_are_reachable(self):
        """These nine were missing from the hand-maintained copy — /thoughts
        being the alive layer's only observability command."""
        from blipshell.simulate.slash_dispatcher import SlashCommandDispatcher

        d = SlashCommandDispatcher(agent=None, config=None)
        for name in ("thoughts", "guardrails", "verbose", "expand", "research",
                     "cube", "followups", "friction", "notes"):
            assert name in d.known_commands(), name

    async def test_simulate_reports_unknown_commands(self):
        from blipshell.simulate.slash_dispatcher import SlashCommandDispatcher

        d = SlashCommandDispatcher(agent=None, config=None)
        result = await d.execute("/definitely-not-a-command")
        assert result.success is False
        assert "Unknown command" in result.error

    async def test_simulate_rejects_non_slash_input(self):
        from blipshell.simulate.slash_dispatcher import SlashCommandDispatcher

        d = SlashCommandDispatcher(agent=None, config=None)
        result = await d.execute("just a message")
        assert result.success is False

    async def test_simulate_gets_isolated_ui_state(self):
        """A scenario toggling /verbose must not leak into the next one, or
        into the real CLI's process-wide state."""
        from blipshell.simulate.slash_dispatcher import SlashCommandDispatcher
        from blipshell.ui.state import ui_state as cli_state

        a = SlashCommandDispatcher(agent=None, config=None)
        b = SlashCommandDispatcher(agent=None, config=None)
        await a.execute("/verbose")

        assert a.ui.verbose_tools is True
        assert b.ui.verbose_tools is False
        assert cli_state.verbose_tools is False
