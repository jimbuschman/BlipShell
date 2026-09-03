"""Input handling using prompt_toolkit.

Provides PromptSession-based input for the main chat loop,
tool approval, and ask_user prompts. Supports:
- Input history (persisted to data/.blipshell_history)
- Bracketed paste (multi-line paste captured as single input)
- ANSI-formatted prompts matching Rich styling

Falls back to plain input() if prompt_toolkit can't initialize
(e.g., non-standard terminal environments).
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory, InMemoryHistory

logger = logging.getLogger(__name__)


class SlashCommandCompleter(Completer):
    """Dropdown of slash commands while the FIRST word starts with '/'.

    Live menu (complete_while_typing): typing '/' lists every visible command
    with its help text as the annotation; further characters narrow the list;
    Tab/arrows select. Inert everywhere else — once a space is typed, or for
    ordinary chat text, no completions are offered, so the model conversation
    itself never grows a dropdown.
    """

    def __init__(self, registry):
        self._registry = registry

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        prefix = text[1:].lower()
        for cmd in sorted(self._registry.visible_commands(),
                          key=lambda c: c.name):
            # Match the primary name first; fall back to any alias so a typed
            # alias prefix still completes — but offer each command once.
            matching = next(
                (n for n in cmd.names if n.startswith(prefix)), None,
            )
            if matching is None:
                continue
            yield Completion(
                text=f"/{matching}",
                start_position=-len(text),
                display=f"/{matching}" if matching != cmd.name
                        else cmd.render_label(),
                display_meta=cmd.help,
            )


class SafeFileHistory(FileHistory):
    """FileHistory that sanitizes surrogate characters before writing.

    Windows clipboard can produce unpaired UTF-16 surrogates (e.g. from
    emoji) that crash prompt_toolkit's UTF-8 encoding in store_string().
    """

    def store_string(self, string: str) -> None:
        # Replace surrogates character-by-character — .encode() with
        # surrogateescape fails on actual surrogates in UTF-8 mode.
        clean = "".join(
            c if ord(c) < 0xD800 or ord(c) > 0xDFFF else "\ufffd"
            for c in string
        )
        super().store_string(clean)

# History file lives alongside other BlipShell data
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_HISTORY_FILE = _DATA_DIR / ".blipshell_history"


def create_chat_session(bottom_toolbar=None, completer=None) -> Optional[PromptSession]:
    """Create the main chat PromptSession with persistent history.

    Args:
        bottom_toolbar: Optional callable returning toolbar text (ANSI string or plain).
        completer: Optional prompt_toolkit Completer (the '/' command dropdown).

    Returns None if prompt_toolkit can't initialize (non-console environment).
    """
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        return PromptSession(
            history=SafeFileHistory(str(_HISTORY_FILE)),
            # prompt_toolkit HARD-disables complete_while_typing whenever
            # enable_history_search is on (shortcuts/prompt.py: "Make sure
            # that complete_while_typing is disabled when
            # enable_history_search is enabled") — with both requested, the
            # '/' dropdown silently never fired (live report 2026-09-03).
            # With a completer, the dropdown wins; plain up/down history
            # cycling still works, only prefix-search-on-up is lost.
            enable_history_search=completer is None,
            mouse_support=False,
            multiline=False,
            bottom_toolbar=bottom_toolbar,
            completer=completer,
            # Live dropdown as '/' is typed; the completer itself is inert for
            # non-command text, so ordinary chat input never shows a menu.
            complete_while_typing=completer is not None,
        )
    except Exception as e:
        logger.warning("prompt_toolkit unavailable, falling back to basic input: %s", e)
        return None


def create_simple_session() -> Optional[PromptSession]:
    """Create a no-history session for tool approval and ask_user prompts.

    Returns None if prompt_toolkit can't initialize.
    """
    try:
        return PromptSession(
            history=InMemoryHistory(),
            multiline=False,
        )
    except Exception:
        return None


async def async_prompt(session: Optional[PromptSession], prompt_text) -> str:
    """Run a prompt_toolkit prompt, with fallback to built-in input().

    If session is None (prompt_toolkit unavailable), uses plain input().
    """
    if session is None:
        # Fallback: strip ANSI formatting for plain input
        plain = str(prompt_text) if not isinstance(prompt_text, str) else prompt_text
        return input(plain)

    try:
        return await session.prompt_async(prompt_text)
    except NotImplementedError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, session.prompt, prompt_text)


def format_chat_prompt(project_name: Optional[str] = None) -> ANSI:
    """Build ANSI-formatted chat prompt matching current Rich styling."""
    parts = []
    if project_name:
        parts.append(f"\x1b[1;36m{project_name}\x1b[0m ")
    parts.append("\x1b[1;32m> \x1b[0m")
    return ANSI("".join(parts))


# Pre-built ANSI prompts for simple inputs
APPROVAL_PROMPT = ANSI(
    "\x1b[1;33m(a)\x1b[0mllow  "
    "\x1b[1;33m(s)\x1b[0mession  "
    "\x1b[1;33m(d)\x1b[0meny  "
    "\x1b[1;33m>\x1b[0m "
)

SIMPLE_PROMPT = ANSI("\x1b[1;33m> \x1b[0m")
