"""Slash-command completion dropdown (ui/input.py SlashCommandCompleter).

The completer feeds prompt_toolkit's live menu: typing '/' lists visible
commands with their help text; it must stay inert for ordinary chat text and
after the command word, so conversation input never sprouts a menu. Pure
Document-driven tests — no terminal required (the interactive rendering is
the one part only a live smoke test can validate, per CLAUDE.md's note on
cli.py terminal plumbing).
"""

from prompt_toolkit.document import Document

from blipshell.ui.commands import Command, CommandRegistry
from blipshell.ui.input import SlashCommandCompleter


def _registry() -> CommandRegistry:
    reg = CommandRegistry()
    reg.register(Command(("help", "h"), lambda c: None, "Show commands"))
    reg.register(Command(("thoughts",), lambda c: None, "Show the self-thought store"))
    reg.register(Command(("think",), lambda c: None, "Toggle thinking"))
    reg.register(Command(("quit", "q", "exit"), lambda c: None, "Exit BlipShell"))
    reg.register(Command(("debug-secret",), lambda c: None, "internal", hidden=True))
    return reg


def _complete(text: str) -> list:
    completer = SlashCommandCompleter(_registry())
    return list(completer.get_completions(Document(text, len(text)), None))


def test_bare_slash_lists_all_visible_commands():
    names = {c.text for c in _complete("/")}
    assert names == {"/help", "/thoughts", "/think", "/quit"}


def test_prefix_narrows():
    assert {c.text for c in _complete("/th")} == {"/thoughts", "/think"}


def test_hidden_commands_never_appear():
    assert all("secret" not in c.text for c in _complete("/"))
    assert _complete("/debug") == []


def test_alias_prefix_completes_but_command_appears_once():
    completions = _complete("/q")
    assert [c.text for c in completions] == ["/quit"]
    # A prefix matching only the alias still finds the command, via the alias.
    assert [c.text for c in _complete("/ex")] == ["/exit"]


def test_help_text_is_the_annotation():
    (completion,) = _complete("/quit")
    meta = completion.display_meta_text
    assert "Exit BlipShell" in meta


def test_inert_for_chat_text_and_after_command_word():
    assert _complete("why do you think") == []
    assert _complete("hello /help") == []
    assert _complete("/project new") == []   # subcommands not completed
    assert _complete("") == []


def test_replaces_the_whole_typed_prefix():
    (completion,) = _complete("/quit")
    assert completion.start_position == -len("/quit")
    (completion,) = _complete("/qu")
    assert completion.start_position == -len("/qu")
