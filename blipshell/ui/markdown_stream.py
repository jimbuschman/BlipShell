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

        # For partial lines (no newline yet), buffer them.
        # We need to emit something for responsiveness, but we can't
        # format until we see the full line. Compromise: emit raw text
        # for partial lines, unless we're in a code block.
        if self._line_buffer and not any(c == "\n" for c in token):
            # Partial line — emit the new characters raw (or dimmed in code block)
            new_chars = token
            if self._in_code_block:
                output.append(f"{DIM}{new_chars}{RESET}")
            else:
                output.append(new_chars)
            # But we've already added to _line_buffer, so when the line
            # completes we'd double-output. Track what we've emitted.
            self._pending_output += new_chars

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

    def flush(self) -> str:
        """Flush any remaining buffered content (call at end of stream)."""
        if self._line_buffer:
            # Format whatever's left without erasing (it's the final partial line)
            if self._in_code_block:
                result = f"{DIM}{self._line_buffer}{RESET}"
            else:
                result = self._format_inline(self._line_buffer)
            self._line_buffer = ""
            self._pending_output = ""
            return result
        return ""
