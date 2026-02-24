"""Filesystem tools: read, write, edit, list files."""

import ast
import difflib
import logging
import os
from pathlib import Path

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


class ReadFileTool(Tool):
    def __init__(self, max_file_size: int = 1048576, blocked_paths: list[str] | None = None,
                 root_path: str | None = None, files_read: set[str] | None = None,
                 file_cache: dict[str, str] | None = None):
        self.max_file_size = max_file_size
        self.blocked_paths = blocked_paths or []
        self.root_path = root_path
        self.files_read = files_read  # shared set from agent — checked to prevent re-reads
        self.file_cache = file_cache  # optional cache — if set, re-reads return cached content

    # Default line limit for large files — prevents context window flooding
    DEFAULT_MAX_LINES = 1000

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description=(
                "Read the contents of a file. For large files, returns a window of lines "
                "with a note about total file size. Use start_line to read specific sections. "
                "Do not re-read files already in the conversation."
            ),
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="Absolute or relative file path to read"),
                ToolParameter(name="start_line", type=ToolParameterType.INTEGER,
                              description="Line number to start reading from (1-based, default: 1)",
                              required=False),
                ToolParameter(name="max_lines", type=ToolParameterType.INTEGER,
                              description="Maximum number of lines to return (default: 1000)",
                              required=False),
            ],
        )

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute() and self.root_path:
            return (Path(self.root_path) / p).resolve()
        return p.resolve()

    async def execute(self, path: str, start_line: int = 1, max_lines: int = 0, **kwargs) -> str:
        resolved = self._resolve(path)

        # Re-read handling: behavior depends on whether a cache is available.
        # With cache (executor/planned path): return cached content (fast, no disk IO)
        # Without cache (simple chat path): block re-reads entirely
        if self.files_read is not None:
            resolved_str = str(resolved)
            if resolved_str in self.files_read or path in self.files_read:
                # Check cache first — return cached content if available
                if self.file_cache is not None:
                    cached = self.file_cache.get(path) or self.file_cache.get(resolved_str)
                    if cached:
                        logger.debug("Returning cached content for %s", path)
                        return self._paginate(cached, start_line, max_lines)
                # No cache — block the re-read
                return (
                    f"ALREADY READ: '{path}' was already read this session. "
                    "The content is in our conversation above. Do not re-read."
                )

        if self._is_blocked(str(resolved)):
            return f"Error: Access to '{path}' is blocked."
        if not resolved.is_file():
            return (
                f"Error: File '{path}' not found. "
                "Use list_directory to see what files exist, or glob_files to search."
            )
        if resolved.stat().st_size > self.max_file_size:
            return f"Error: File '{path}' exceeds max size ({self.max_file_size} bytes)."

        content = resolved.read_text(encoding="utf-8", errors="replace")

        # Populate cache with full content (executor path)
        if self.file_cache is not None:
            self.file_cache[path] = content
            self.file_cache[str(resolved)] = content

        return self._paginate(content, start_line, max_lines)

    def _paginate(self, content: str, start_line: int = 1, max_lines: int = 0) -> str:
        """Return a window of lines with line numbers and a size note for large files."""
        all_lines = content.splitlines()
        total = len(all_lines)
        limit = max_lines if max_lines > 0 else self.DEFAULT_MAX_LINES

        # Clamp start_line to valid range (1-based)
        start = max(0, min(start_line - 1, total - 1))  # 0-based, clamped

        # If start_line is past the end, say so
        if start_line > total:
            return (
                f"File has {total} lines. start_line={start_line} is past the end. "
                f"Use start_line=1 to read from the beginning, or "
                f"start_line={max(1, total - limit + 1)} to read the last section."
            )

        window = all_lines[start:start + limit]
        end_line = start + len(window)

        # For small files that fit in one page, return plain content (no pagination noise)
        if total <= limit and start == 0:
            return content

        # Add line numbers for easier navigation
        numbered = []
        for i, line in enumerate(window, start=start + 1):
            numbered.append(f"{i:>5}: {line}")
        result = "\n".join(numbered)

        # Tell the model what they're seeing and how to see more
        if end_line < total:
            result += (
                f"\n\n[Showing lines {start + 1}-{end_line} of {total}. "
                f"Use start_line={end_line + 1} to continue reading.]"
            )
        else:
            result = f"[Lines {start + 1}-{end_line} of {total}]\n" + result

        return result

    def _is_blocked(self, path: str) -> bool:
        return any(blocked in path for blocked in self.blocked_paths)


class WriteFileTool(Tool):
    def __init__(self, blocked_paths: list[str] | None = None, root_path: str | None = None):
        self.blocked_paths = blocked_paths or []
        self.root_path = root_path

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="File path to write to"),
                ToolParameter(name="content", type=ToolParameterType.STRING,
                              description="Content to write"),
            ],
        )

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute() and self.root_path:
            return (Path(self.root_path) / p).resolve()
        return p.resolve()

    async def execute(self, path: str = "", content: str = "", **kwargs) -> str:
        if not path:
            return "Error: 'path' argument is required."
        if not content:
            return "Error: 'content' argument is required — provide the file content to write."
        resolved = self._resolve(path)
        if any(blocked in str(resolved) for blocked in self.blocked_paths):
            return f"Error: Access to '{path}' is blocked."

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"


class EditFileTool(Tool):
    """Edit a file by replacing a specific string.

    Uses layered matching (inspired by Aider) when exact match fails:
    1. Exact match
    2. Whitespace-normalized match (trailing spaces, indentation)
    3. Fuzzy match via SequenceMatcher (threshold 0.6)

    This dramatically reduces "text not found" failures that waste tool calls.
    """

    FUZZY_THRESHOLD = 0.6

    def __init__(self, root_path: str | None = None):
        self.root_path = root_path

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute() and self.root_path:
            return (Path(self.root_path) / p).resolve()
        return p.resolve()

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="edit_file",
            description=(
                "Replace a specific string in a file with new content. "
                "The old_text must match exactly (whitespace and indentation matter). "
                "Read the file first to get the exact text. If the edit fails, "
                "re-read the file once and retry with the correct text."
            ),
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="File path to edit"),
                ToolParameter(name="old_text", type=ToolParameterType.STRING,
                              description="Exact text to find and replace (copy from the file)"),
                ToolParameter(name="new_text", type=ToolParameterType.STRING,
                              description="Replacement text"),
            ],
        )

    async def execute(self, path: str = "", old_text: str = "", new_text: str = "", **kwargs) -> str:
        if not path:
            return "Error: 'path' argument is required."
        if not old_text:
            return "Error: 'old_text' argument is required — specify the text to find and replace."
        resolved = self._resolve(path)
        if not resolved.is_file():
            if resolved.is_dir():
                return (
                    f"Error: '{path}' is a directory, not a file. "
                    "Use list_directory to see files in it."
                )
            return (
                f"Error: File '{path}' not found. "
                "Check the path with list_directory or glob_files."
            )

        content = resolved.read_text(encoding="utf-8")

        # Layer 1: Exact match
        if old_text in content:
            new_content = content.replace(old_text, new_text, 1)
            lint_err = self._lint_python(resolved, new_content)
            if lint_err:
                return lint_err
            resolved.write_text(new_content, encoding="utf-8")
            return f"Successfully edited {path}"

        # Layer 2: Whitespace-normalized match
        ws_result = self._try_whitespace_match(content, old_text, new_text)
        if ws_result is not None:
            lint_err = self._lint_python(resolved, ws_result)
            if lint_err:
                return lint_err
            resolved.write_text(ws_result, encoding="utf-8")
            logger.info("edit_file used whitespace-normalized match for %s", path)
            return f"Successfully edited {path} (whitespace-normalized match)"

        # Layer 3: Fuzzy match
        fuzzy_result, ratio = self._try_fuzzy_match(content, old_text, new_text)
        if fuzzy_result is not None:
            lint_err = self._lint_python(resolved, fuzzy_result)
            if lint_err:
                return lint_err
            resolved.write_text(fuzzy_result, encoding="utf-8")
            logger.info("edit_file used fuzzy match (%.0f%%) for %s", ratio * 100, path)
            return f"Successfully edited {path} (fuzzy match, {ratio:.0%} similarity)"

        # All layers failed — provide helpful error with closest match
        total_lines = len(content.splitlines())
        hint = self._find_closest_match_hint(content, old_text)
        error_msg = (
            f"Error: Text to replace not found in '{path}' ({total_lines} lines). "
            "The old_text must match the file exactly. "
            "Re-read the file with read_file to get the correct text."
        )
        if hint:
            error_msg += f"\n\nClosest match (lines {hint['start_line']}-{hint['end_line']}):\n{hint['text']}"
        return error_msg

    @staticmethod
    def _lint_python(file_path: Path, content: str) -> str | None:
        """Check if the edited content has valid Python syntax.

        Returns an error message if syntax is broken, None if OK.
        Only checks .py files — other file types are passed through.
        Based on SWE-agent research: linting every edit improved success by 3%.
        """
        if file_path.suffix != ".py":
            return None
        try:
            ast.parse(content, filename=str(file_path))
            return None
        except SyntaxError as e:
            return (
                f"Error: Edit would create a syntax error in {file_path.name} "
                f"at line {e.lineno}: {e.msg}. "
                "The file was NOT modified. Fix the new_text and retry."
            )

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Strip trailing whitespace per line and normalize to single newlines."""
        return "\n".join(line.rstrip() for line in text.splitlines())

    def _try_whitespace_match(
        self, content: str, old_text: str, new_text: str,
    ) -> str | None:
        """Try matching after normalizing whitespace on both sides.

        Also tries adjusting indentation: if old_text's indentation doesn't
        match but the content is otherwise identical, adapts to the file's
        indentation.
        """
        norm_content = self._normalize_whitespace(content)
        norm_old = self._normalize_whitespace(old_text)

        if norm_old in norm_content:
            # Find the actual text in the original content that corresponds
            # to this normalized match, then replace it
            return self._replace_normalized(content, old_text, new_text)

        # Try indentation-adjusted match: strip all leading whitespace,
        # find the match, then apply new_text with the file's indentation
        dedented_old = "\n".join(line.lstrip() for line in old_text.splitlines())
        content_lines = content.splitlines()

        match_start = self._find_dedented_match(content_lines, dedented_old)
        if match_start is not None:
            old_line_count = len(old_text.splitlines())
            matched_lines = content_lines[match_start:match_start + old_line_count]

            # Detect the file's indentation for this block
            file_indent = ""
            for line in matched_lines:
                stripped = line.lstrip()
                if stripped:
                    file_indent = line[:len(line) - len(stripped)]
                    break

            # Apply new_text with detected indentation
            new_lines = new_text.splitlines()
            # Detect old_text's indentation to compute relative indentation
            old_indent = ""
            for line in old_text.splitlines():
                stripped = line.lstrip()
                if stripped:
                    old_indent = line[:len(line) - len(stripped)]
                    break

            adjusted_new_lines = []
            for line in new_lines:
                stripped = line.lstrip()
                if not stripped:
                    adjusted_new_lines.append("")
                    continue
                # Compute relative indent from old_text base
                current_indent = line[:len(line) - len(stripped)]
                if current_indent.startswith(old_indent):
                    relative = current_indent[len(old_indent):]
                else:
                    relative = ""
                adjusted_new_lines.append(file_indent + relative + stripped)

            original_text = "\n".join(matched_lines)
            adjusted_new = "\n".join(adjusted_new_lines)
            return content.replace(original_text, adjusted_new, 1)

        return None

    @staticmethod
    def _replace_normalized(content: str, old_text: str, new_text: str) -> str:
        """Replace old_text in content using whitespace-normalized matching.

        Finds the region in content that matches old_text after whitespace
        normalization, then replaces that exact region with new_text.
        """
        content_lines = content.splitlines()
        old_lines = old_text.splitlines()
        norm_old_lines = [line.rstrip() for line in old_lines]

        for i in range(len(content_lines) - len(old_lines) + 1):
            candidate = content_lines[i:i + len(old_lines)]
            if [line.rstrip() for line in candidate] == norm_old_lines:
                # Found the match — replace these lines
                before = content_lines[:i]
                after = content_lines[i + len(old_lines):]
                result_lines = before + new_text.splitlines() + after
                return "\n".join(result_lines)

        # Shouldn't reach here if _normalize_whitespace matched, but fallback
        return content

    @staticmethod
    def _find_dedented_match(content_lines: list[str], dedented_old: str) -> int | None:
        """Find where dedented old_text matches in content (ignoring indentation)."""
        dedented_old_lines = dedented_old.splitlines()
        old_count = len(dedented_old_lines)

        for i in range(len(content_lines) - old_count + 1):
            candidate = [line.lstrip() for line in content_lines[i:i + old_count]]
            if candidate == dedented_old_lines:
                return i

        return None

    def _try_fuzzy_match(
        self, content: str, old_text: str, new_text: str,
    ) -> tuple[str | None, float]:
        """Try fuzzy matching using SequenceMatcher.

        Slides a window over the content looking for the best match.
        Returns (new_content, similarity_ratio) or (None, 0.0).
        """
        content_lines = content.splitlines()
        old_lines = old_text.splitlines()
        old_count = len(old_lines)

        if old_count == 0 or old_count > len(content_lines):
            return None, 0.0

        best_ratio = 0.0
        best_start = -1
        old_joined = "\n".join(old_lines)

        # Slide a window of old_count lines across the file
        for i in range(len(content_lines) - old_count + 1):
            candidate = "\n".join(content_lines[i:i + old_count])
            ratio = difflib.SequenceMatcher(None, old_joined, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        if best_ratio >= self.FUZZY_THRESHOLD and best_start >= 0:
            matched_text = "\n".join(content_lines[best_start:best_start + old_count])
            new_content = content.replace(matched_text, new_text, 1)
            return new_content, best_ratio

        return None, best_ratio

    @staticmethod
    def _find_closest_match_hint(content: str, old_text: str) -> dict | None:
        """Find the closest matching region in the file for error reporting.

        Returns a dict with start_line, end_line, and text snippet,
        or None if no reasonable match found.
        """
        content_lines = content.splitlines()
        old_lines = old_text.splitlines()
        old_count = len(old_lines)

        if old_count == 0 or old_count > len(content_lines):
            return None

        best_ratio = 0.0
        best_start = -1
        old_joined = "\n".join(old_lines)

        for i in range(len(content_lines) - old_count + 1):
            candidate = "\n".join(content_lines[i:i + old_count])
            ratio = difflib.SequenceMatcher(None, old_joined, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        # Only show hint if there's at least some similarity (> 0.3)
        if best_ratio > 0.3 and best_start >= 0:
            matched_lines = content_lines[best_start:best_start + old_count]
            # Truncate hint to 10 lines max
            if len(matched_lines) > 10:
                matched_lines = matched_lines[:10] + ["... (truncated)"]
            return {
                "start_line": best_start + 1,
                "end_line": best_start + old_count,
                "text": "\n".join(matched_lines),
            }

        return None


class ListDirectoryTool(Tool):
    def __init__(self, root_path: str | None = None):
        self.root_path = root_path

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute() and self.root_path:
            return (Path(self.root_path) / p).resolve()
        return p.resolve()

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_directory",
            description="List files and directories at the given path.",
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="Directory path to list", required=False),
            ],
        )

    async def execute(self, path: str = ".", **kwargs) -> str:
        resolved = self._resolve(path)
        if not resolved.is_dir():
            if resolved.is_file():
                return f"Error: '{path}' is a file, not a directory. Use read_file to read it."
            return f"Error: '{path}' does not exist. Try list_directory on the parent path."

        entries = []
        try:
            for entry in sorted(resolved.iterdir()):
                prefix = "[DIR] " if entry.is_dir() else "      "
                entries.append(f"{prefix}{entry.name}")
        except PermissionError:
            return f"Error: Permission denied for '{path}'."

        if not entries:
            return f"Directory '{path}' is empty."
        return "\n".join(entries)
