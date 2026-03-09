"""Lightweight streaming Markdown renderer for terminal output.

Applies ANSI formatting as tokens arrive from the LLM. Handles:
- Headers (#, ##, ###) → bold
- Code blocks (```) → dimmed with language hint
- Bold (**text**) → ANSI bold
- Inline code (`text`) → cyan
- Bullet lists (- item) → dim bullet

Designed for incremental token-by-token input. Formatting is applied
at line boundaries to avoid partial-token issues.
"""

import os

# ANSI escape codes
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
RESET = "\x1b[0m"
GREEN = "\x1b[32m"


class MarkdownStreamer:
    """Token-by-token Markdown to ANSI formatter."""

    def __init__(self):
        self._line_buffer = ""
        self._in_code_block = False
        self._code_lang = ""
        self._pending_output = ""

    @staticmethod
    def _get_term_width() -> int:
        try:
            return os.get_terminal_size().columns
        except (OSError, ValueError):
            return 80

    def feed(self, token: str) -> str:
        """Accept a token and return ANSI-formatted output.

        Call this for each token from the LLM stream. Returns the
        formatted string to write to stdout.
        """
        output = []

        for char in token:
            if char == "\n":
                # Process the completed line
                formatted = self._format_line(self._line_buffer)
                output.append(formatted)
                output.append("\n")
                self._line_buffer = ""
            else:
                self._line_buffer += char

        # For partial lines (no newline yet), emit raw chars for responsiveness.
        # Cap at terminal width — CUB (Cursor Backward) used by _erase_pending()
        # cannot cross terminal line wraps, so we must keep pending output within
        # a single terminal line for erase-and-replace to work correctly.
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
                self._in_code_block = True
                self._code_lang = stripped[3:].strip()
                # Erase any partial output we emitted for this line
                erase = self._erase_pending()
                if self._code_lang:
                    return f"{erase}{DIM}```{self._code_lang}{RESET}"
                return f"{erase}{DIM}```{RESET}"
            else:
                self._in_code_block = False
                self._code_lang = ""
                erase = self._erase_pending()
                return f"{erase}{DIM}```{RESET}"

        # Inside code block — dim everything
        if self._in_code_block:
            erase = self._erase_pending()
            return f"{erase}{DIM}{line}{RESET}"

        # Erase any partial output we emitted while buffering
        erase = self._erase_pending()

        # Headers
        if stripped.startswith("### "):
            return f"{erase}{BOLD}{stripped}{RESET}"
        if stripped.startswith("## "):
            return f"{erase}{BOLD}{stripped}{RESET}"
        if stripped.startswith("# "):
            return f"{erase}{BOLD}{stripped}{RESET}"

        # Apply inline formatting
        formatted = self._format_inline(line)
        return f"{erase}{formatted}"

    def _format_inline(self, text: str) -> str:
        """Apply inline Markdown formatting (bold, code)."""
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
            result.append(text[i])
            i += 1
        return "".join(result)

    def _erase_pending(self) -> str:
        """Generate ANSI to erase previously emitted partial output."""
        if not self._pending_output:
            return ""
        # Move cursor back and clear what we emitted
        n = len(self._pending_output)
        self._pending_output = ""
        return f"\x1b[{n}D\x1b[{n}X"

    def reset_line(self):
        """Discard buffered partial line state.

        Call this when external output (ANSI tool displays) has moved the
        cursor, making the erase-and-replace mechanism invalid. The
        previously emitted raw text stays on screen as-is.
        """
        self._line_buffer = ""
        self._pending_output = ""

    def flush(self) -> str:
        """Flush any remaining buffered content (call at end of stream)."""
        if self._line_buffer:
            # Erase the raw partial output we emitted, replace with formatted
            erase = self._erase_pending()
            if self._in_code_block:
                result = f"{erase}{DIM}{self._line_buffer}{RESET}"
            else:
                result = f"{erase}{self._format_inline(self._line_buffer)}"
            self._line_buffer = ""
            return result
        return ""
