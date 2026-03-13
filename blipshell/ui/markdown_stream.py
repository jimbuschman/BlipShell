"""Streaming Markdown renderer for terminal output.

Applies ANSI formatting as tokens arrive from the LLM. Handles:
- Headers (#, ##, ###) → bold + colored (blue, cyan, green)
- Code blocks (```) → syntax highlighted via Rich.Syntax on completion
- Bold (**text**) → ANSI bold
- Italic (*text* / _text_) → ANSI italic
- Inline code (`text`) → cyan
- Bullet lists (- item, * item) → dim bullet
- Numbered lists (1. item) → dim number
- Horizontal rules (---) → dim line

Code blocks stream as dim text for responsiveness, then get replaced
with syntax-highlighted output when the closing fence arrives.
"""

import io
import os
import re

# ANSI escape codes
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"
CYAN = "\x1b[36m"
BLUE = "\x1b[34m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"

# Cursor control
CURSOR_UP = "\x1b[{n}A"       # Move cursor up n lines
ERASE_LINE = "\x1b[2K"         # Erase entire current line
CURSOR_COL0 = "\x1b[G"         # Move cursor to column 0

# Patterns
_NUMBERED_LIST = re.compile(r"^(\s*\d+[.)]\s)")
_BULLET_LIST = re.compile(r"^(\s*[-*+]\s)")
_HR_PATTERN = re.compile(r"^(\s*)([-*_])\s*\2\s*\2[\s\2]*$")


class MarkdownStreamer:
    """Token-by-token Markdown to ANSI formatter with syntax highlighting."""

    def __init__(self, syntax_highlight: bool = True):
        self._line_buffer = ""
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines: list[str] = []     # Buffered code block content
        self._code_display_lines = 0          # Lines shown on terminal for current block
        self._pending_output = ""
        self._syntax_highlight = syntax_highlight

    @staticmethod
    def _get_term_width() -> int:
        try:
            return os.get_terminal_size().columns
        except (OSError, ValueError):
            return 80

    def feed(self, token: str) -> str:
        """Accept a token and return ANSI-formatted output."""
        output = []

        for char in token:
            if char == "\n":
                formatted = self._format_line(self._line_buffer)
                output.append(formatted)
                output.append("\n")
                self._line_buffer = ""
            else:
                self._line_buffer += char

        # For partial lines (no newline yet), emit raw chars for responsiveness.
        if self._line_buffer and not any(c == "\n" for c in token):
            new_chars = token
            max_emit = self._get_term_width() - 1
            available = max_emit - len(self._pending_output)
            if available > 0:
                emit = new_chars[:available]
                if self._in_code_block:
                    output.append(f"{DIM}{emit}{RESET}")
                else:
                    output.append(emit)
                self._pending_output += emit

        return "".join(output)

    def _format_line(self, line: str) -> str:
        """Format a complete line with Markdown styling."""
        stripped = line.lstrip()

        # Code block fence
        if stripped.startswith("```"):
            if not self._in_code_block:
                # Opening fence
                self._in_code_block = True
                self._code_lang = stripped[3:].strip()
                self._code_lines = []
                self._code_display_lines = 0
                erase = self._erase_pending()
                label = f"```{self._code_lang}" if self._code_lang else "```"
                self._code_display_lines += 1
                return f"{erase}{DIM}{label}{RESET}"
            else:
                # Closing fence — render syntax highlighted block
                return self._close_code_block()

        # Inside code block — buffer and show dim
        if self._in_code_block:
            self._code_lines.append(line)
            erase = self._erase_pending()
            self._code_display_lines += 1
            return f"{erase}{DIM}{line}{RESET}"

        # Regular text
        erase = self._erase_pending()

        # Horizontal rule
        if _HR_PATTERN.match(stripped) and len(stripped) >= 3:
            width = self._get_term_width()
            return f"{erase}{DIM}{'-' * min(width - 1, 40)}{RESET}"

        # Headers with colors
        if stripped.startswith("### "):
            return f"{erase}{BOLD}{GREEN}{stripped}{RESET}"
        if stripped.startswith("## "):
            return f"{erase}{BOLD}{CYAN}{stripped}{RESET}"
        if stripped.startswith("# "):
            return f"{erase}{BOLD}{BLUE}{stripped}{RESET}"

        # Numbered lists
        m = _NUMBERED_LIST.match(line)
        if m:
            prefix = m.group(1)
            rest = line[m.end():]
            return f"{erase}{DIM}{prefix}{RESET}{self._format_inline(rest)}"

        # Bullet lists
        m = _BULLET_LIST.match(line)
        if m:
            prefix = m.group(1)
            rest = line[m.end():]
            return f"{erase}{DIM}{prefix}{RESET}{self._format_inline(rest)}"

        # Apply inline formatting
        formatted = self._format_inline(line)
        return f"{erase}{formatted}"

    def _close_code_block(self) -> str:
        """Close a code block and render with syntax highlighting."""
        code_content = "\n".join(self._code_lines)
        lang = self._code_lang
        display_lines = self._code_display_lines
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines = []
        self._code_display_lines = 0

        erase = self._erase_pending()

        # Try Rich Syntax highlighting
        if self._syntax_highlight and code_content.strip():
            highlighted = self._render_syntax(code_content, lang)
            if highlighted:
                # Erase the dim lines we streamed, replace with highlighted version
                erase_block = self._erase_n_lines(display_lines)
                return f"{erase}{erase_block}{highlighted}"

        # Fallback: just close the dim block
        return f"{erase}{DIM}```{RESET}"

    def _render_syntax(self, code: str, lang: str) -> str | None:
        """Render code with Rich Syntax, return ANSI string or None on failure."""
        try:
            from rich.syntax import Syntax
            from rich.console import Console

            # Map common language aliases
            lang_map = {
                "py": "python",
                "js": "javascript",
                "ts": "typescript",
                "sh": "bash",
                "yml": "yaml",
                "": "text",
            }
            lexer = lang_map.get(lang, lang) or "text"

            syntax = Syntax(
                code,
                lexer,
                theme="monokai",
                line_numbers=False,
                word_wrap=False,
                padding=(0, 1),
            )

            # Render to string
            string_io = io.StringIO()
            temp_console = Console(
                file=string_io,
                force_terminal=True,
                width=self._get_term_width(),
                no_color=False,
            )
            temp_console.print(syntax, end="")
            return string_io.getvalue()
        except Exception:
            return None

    @staticmethod
    def _erase_n_lines(n: int) -> str:
        """Generate ANSI to erase n previously-printed lines and reposition cursor."""
        if n <= 0:
            return ""
        # Move cursor up n lines, erasing each one
        parts = []
        for _ in range(n):
            parts.append(f"\x1b[A\x1b[2K")
        return "".join(parts)

    def _format_inline(self, text: str) -> str:
        """Apply inline Markdown formatting (bold, italic, code)."""
        result = []
        i = 0
        while i < len(text):
            # Bold: **text**
            if text[i:i+2] == "**":
                end = text.find("**", i + 2)
                if end != -1:
                    result.append(f"{BOLD}{text[i+2:end]}{RESET}")
                    i = end + 2
                    continue
            # Inline code: `text`
            if text[i] == "`":
                end = text.find("`", i + 1)
                if end != -1:
                    result.append(f"{CYAN}{text[i+1:end]}{RESET}")
                    i = end + 1
                    continue
            # Italic: *text* (but not **)
            if text[i] == "*" and (i + 1 >= len(text) or text[i + 1] != "*"):
                end = text.find("*", i + 1)
                if end != -1 and end > i + 1:
                    result.append(f"{ITALIC}{text[i+1:end]}{RESET}")
                    i = end + 1
                    continue
            result.append(text[i])
            i += 1
        return "".join(result)

    def _erase_pending(self) -> str:
        """Generate ANSI to erase previously emitted partial output."""
        if not self._pending_output:
            return ""
        n = len(self._pending_output)
        self._pending_output = ""
        return f"\x1b[{n}D\x1b[{n}X"

    def reset_line(self):
        """Discard buffered partial line state.

        Call this when external output (ANSI tool displays) has moved the
        cursor, making the erase-and-replace mechanism invalid.
        """
        self._line_buffer = ""
        self._pending_output = ""

    def flush(self) -> str:
        """Flush any remaining buffered content (call at end of stream)."""
        if self._line_buffer:
            erase = self._erase_pending()
            if self._in_code_block:
                # Unclosed code block — flush as dim
                self._code_lines.append(self._line_buffer)
                result = f"{erase}{DIM}{self._line_buffer}{RESET}"
            else:
                result = f"{erase}{self._format_inline(self._line_buffer)}"
            self._line_buffer = ""
            return result
        return ""
