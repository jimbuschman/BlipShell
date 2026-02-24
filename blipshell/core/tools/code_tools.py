"""Code-focused tools: grep (search file contents) and glob (find files)."""

import os
import re
from pathlib import Path

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

# Directories to always skip when walking a file tree.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox",
             ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs", ".hg"}

# Binary extensions to skip when grepping.
BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp",
               ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
               ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
               ".exe", ".dll", ".so", ".dylib", ".o", ".obj",
               ".pyc", ".pyo", ".whl", ".egg",
               ".db", ".sqlite", ".sqlite3",
               ".pdf", ".doc", ".docx", ".xls", ".xlsx"}


class GrepTool(Tool):
    """Search file contents with a regex pattern across a directory tree."""

    def __init__(self, root_path: str | None = None):
        self.root_path = root_path

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="grep_files",
            description=(
                "Search for a regex pattern across files in a directory. "
                "Returns matching lines as 'file:line_number: content'. "
                "The path must be a directory, not a file. To search within a single file, "
                "use read_file instead. Results are capped at max_results (default 50)."
            ),
            parameters=[
                ToolParameter(name="pattern", type=ToolParameterType.STRING,
                              description="Regex pattern to search for"),
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="Directory to search in (default: project root). Must be a directory, not a file path.",
                              required=False),
                ToolParameter(name="include", type=ToolParameterType.STRING,
                              description="Glob filter for files to include (e.g. '*.py', '*.ts')",
                              required=False),
                ToolParameter(name="max_results", type=ToolParameterType.INTEGER,
                              description="Maximum number of matching lines to return (default: 50)",
                              required=False),
            ],
        )

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        include: str = "",
        max_results: int = 50,
        **kwargs,
    ) -> str:
        search_root = self._resolve(path)
        if not search_root.is_dir():
            if search_root.is_file():
                return (
                    f"Error: '{path}' is a file, not a directory. "
                    f"Use read_file to read this file, or search its parent directory: "
                    f"grep_files(pattern='{pattern}', path='{search_root.parent}')"
                )
            return f"Error: '{path}' does not exist. Use list_directory to see available paths."

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex '{pattern}': {e}. Use a valid Python regex."

        matches = []
        for file_path in self._walk_files(search_root, include):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            for line_num, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = self._rel_path(file_path)
                    matches.append(f"{rel}:{line_num}: {line.rstrip()}")
                    if len(matches) >= max_results:
                        return "\n".join(matches) + f"\n... (truncated at {max_results} results)"

        if not matches:
            return f"No matches found for pattern '{pattern}' in {path}"
        return "\n".join(matches)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute() and self.root_path:
            return (Path(self.root_path) / p).resolve()
        return p.resolve()

    def _rel_path(self, file_path: Path) -> str:
        """Return path relative to root_path if possible."""
        if self.root_path:
            try:
                return str(file_path.relative_to(Path(self.root_path).resolve()))
            except ValueError:
                pass
        return str(file_path)

    def _walk_files(self, root: Path, include: str = ""):
        """Walk directory yielding files, skipping non-code dirs and binaries."""
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skipped directories in-place
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() in BINARY_EXTS:
                    continue
                if include:
                    if not fpath.match(include):
                        continue
                yield fpath


class GlobTool(Tool):
    """Find files matching a glob pattern."""

    def __init__(self, root_path: str | None = None):
        self.root_path = root_path

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="glob_files",
            description=(
                "Find files matching a glob pattern in a directory tree. "
                "Returns file paths sorted by modification time (newest first). "
                "Examples: '**/*.py', 'src/**/*.ts', '*.json'"
            ),
            parameters=[
                ToolParameter(name="pattern", type=ToolParameterType.STRING,
                              description="Glob pattern to match (e.g. '**/*.py')"),
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="Directory to search in (default: project root or cwd)",
                              required=False),
                ToolParameter(name="max_results", type=ToolParameterType.INTEGER,
                              description="Maximum number of files to return (default: 100)",
                              required=False),
            ],
        )

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 100,
        **kwargs,
    ) -> str:
        search_root = self._resolve(path)
        if not search_root.is_dir():
            return f"Error: '{path}' is not a directory."

        results = []
        try:
            for fpath in search_root.glob(pattern):
                if not fpath.is_file():
                    continue
                # Skip non-code dirs
                if any(part in SKIP_DIRS for part in fpath.parts):
                    continue
                results.append(fpath)
        except (OSError, ValueError) as e:
            return f"Error: {e}"

        if not results:
            return f"No files found matching '{pattern}' in {path}"

        # Sort by modification time, newest first
        results.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        results = results[:max_results]

        lines = []
        for fpath in results:
            rel = self._rel_path(fpath)
            size = fpath.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1048576:
                size_str = f"{size / 1024:.0f}KB"
            else:
                size_str = f"{size / 1048576:.1f}MB"
            lines.append(f"{rel}  ({size_str})")

        result = "\n".join(lines)
        if len(results) >= max_results:
            result += f"\n... (truncated at {max_results} files)"
        return result

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute() and self.root_path:
            return (Path(self.root_path) / p).resolve()
        return p.resolve()

    def _rel_path(self, file_path: Path) -> str:
        if self.root_path:
            try:
                return str(file_path.relative_to(Path(self.root_path).resolve()))
            except ValueError:
                pass
        return str(file_path)
