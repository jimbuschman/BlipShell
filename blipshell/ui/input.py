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
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory, InMemoryHistory

logger = logging.getLogger(__name__)

# History file lives alongside other BlipShell data
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_HISTORY_FILE = _DATA_DIR / ".blipshell_history"


def create_chat_session() -> Optional[PromptSession]:
    """Create the main chat PromptSession with persistent history.

    Returns None if prompt_toolkit can't initialize (non-console environment).
    """
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        return PromptSession(
            history=FileHistory(str(_HISTORY_FILE)),
            enable_history_search=True,
            mouse_support=False,
            multiline=False,
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
