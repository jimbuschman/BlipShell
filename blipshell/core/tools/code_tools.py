"""Code-focused tools: grep (search file contents), glob (find files), repo_map (code structure)."""

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

if TYPE_CHECKING:
    from blipshell.core.repo_map import RepoMap

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
    read_only = True

    # Shorthand type → glob mapping (matches ripgrep --type conventions)
    TYPE_MAP = {
        "py": "*.py", "python": "*.py",
        "js": "*.js", "javascript": "*.js",
        "ts": "*.ts", "typescript": "*.ts",
        "tsx": "*.tsx", "jsx": "*.jsx",
        "java": "*.java", "go": "*.go", "rust": "*.rs", "rs": "*.rs",
        "c": "*.c", "cpp": "*.cpp", "h": "*.h", "hpp": "*.hpp",
        "rb": "*.rb", "ruby": "*.rb",
        "php": "*.php", "swift": "*.swift", "kt": "*.kt", "kotlin": "*.kt",
        "css": "*.css", "html": "*.html", "xml": "*.xml",
        "json": "*.json", "yaml": "*.yaml", "yml": "*.yml", "toml": "*.toml",
        "md": "*.md", "markdown": "*.md", "txt": "*.txt",
        "sh": "*.sh", "bash": "*.sh", "ps1": "*.ps1",
        "sql": "*.sql", "graphql": "*.graphql",
    }

    def __init__(self, root_path: str | None = None):
        self.root_path = root_path

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="grep_files",
            description=(
                "Search for a regex pattern across files in a directory tree.\n\n"
                "Output modes:\n"
                "- 'content' (default): matching lines as 'file:line_number: content'\n"
                "- 'files_with_matches': only unique file paths that contain a match\n"
                "- 'count': match count per file as 'file: N'\n\n"
                "Features:\n"
                "- Context lines: show N lines before/after each match\n"
                "- Multiline: match patterns across line boundaries\n"
                "- Case insensitive: ignore case when matching\n"
                "- Type filter: shorthand for file type (e.g. 'py' instead of '*.py')\n\n"
                "IMPORTANT:\n"
                "- The path must be a DIRECTORY, not a file. To search within a single file, "
                "use read_file instead.\n"
                "- Use 'include' for glob patterns or 'type_filter' for common extensions.\n"
                "- Results are capped at max_results (default 50)."
            ),
            parameters=[
                ToolParameter(name="pattern", type=ToolParameterType.STRING,
                              description="Regex pattern to search for"),
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="Directory to search in (default: project root). Must be a directory, not a file path.",
                              required=False),
                ToolParameter(name="include", type=ToolParameterType.STRING,
                              description="Glob filter for files to include (e.g. '*.py', '*.{ts,tsx}')",
                              required=False),
                ToolParameter(name="type_filter", type=ToolParameterType.STRING,
                              description="File type shorthand (e.g. 'py', 'js', 'ts', 'rust'). Alternative to include.",
                              required=False),
                ToolParameter(name="max_results", type=ToolParameterType.INTEGER,
                              description="Maximum number of results to return (default: 50)",
                              required=False),
                ToolParameter(name="output_mode", type=ToolParameterType.STRING,
                              description="Output format: 'content' (matching lines, default), 'files_with_matches' (file paths only), 'count' (match count per file)",
                              required=False),
                ToolParameter(name="context_lines", type=ToolParameterType.INTEGER,
                              description="Number of lines to show before AND after each match (default: 0)",
                              required=False),
                ToolParameter(name="before_context", type=ToolParameterType.INTEGER,
                              description="Lines to show before each match (overrides context_lines for before)",
                              required=False),
                ToolParameter(name="after_context", type=ToolParameterType.INTEGER,
                              description="Lines to show after each match (overrides context_lines for after)",
                              required=False),
                ToolParameter(name="multiline", type=ToolParameterType.BOOLEAN,
                              description="Match pattern across line boundaries (default: false)",
                              required=False),
                ToolParameter(name="case_insensitive", type=ToolParameterType.BOOLEAN,
                              description="Ignore case when matching (default: false)",
                              required=False),
            ],
        )

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        include: str = "",
        type_filter: str = "",
        max_results: int = 50,
        output_mode: str = "content",
        context_lines: int = 0,
        before_context: int = 0,
        after_context: int = 0,
        multiline: bool = False,
        case_insensitive: bool = False,
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

        if output_mode not in ("content", "files_with_matches", "count"):
            return f"Error: output_mode must be 'content', 'files_with_matches', or 'count'. Got '{output_mode}'."

        # Resolve type_filter to include glob
        effective_include = include
        if type_filter and not include:
            mapped = self.TYPE_MAP.get(type_filter.lower())
            if mapped:
                effective_include = mapped
            else:
                return f"Error: Unknown type_filter '{type_filter}'. Known types: {', '.join(sorted(set(self.TYPE_MAP.keys())))}"

        # Compile regex
        flags = re.IGNORECASE if case_insensitive else 0
        if multiline:
            flags |= re.DOTALL
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex '{pattern}': {e}. Use a valid Python regex."

        # Compute effective before/after context
        ctx_before = before_context if before_context else context_lines
        ctx_after = after_context if after_context else context_lines

        if output_mode == "files_with_matches":
            return self._search_files_only(regex, search_root, effective_include, max_results, multiline)
        elif output_mode == "count":
            return self._search_count(regex, search_root, effective_include, max_results, multiline)
        else:
            return self._search_content(regex, search_root, effective_include, max_results, multiline, ctx_before, ctx_after)

    def _search_content(
        self, regex, search_root: Path, include: str, max_results: int,
        multiline: bool, ctx_before: int, ctx_after: int,
    ) -> str:
        """Return matching lines with optional context."""
        matches = []
        for file_path in self._walk_files(search_root, include):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            rel = self._rel_path(file_path)
            lines = text.splitlines()

            if multiline:
                # Find all match positions and map to line numbers
                match_lines = set()
                for m in regex.finditer(text):
                    start_line = text[:m.start()].count("\n") + 1
                    end_line = text[:m.end()].count("\n") + 1
                    for ln in range(start_line, end_line + 1):
                        match_lines.add(ln)

                for ln in sorted(match_lines):
                    if ln <= len(lines):
                        line_text = self._cap_line(lines[ln - 1])
                        matches.append(f"{rel}:{ln}: {line_text}")
                        if len(matches) >= max_results:
                            return "\n".join(matches) + f"\n... (truncated at {max_results} results)"
            elif ctx_before > 0 or ctx_after > 0:
                # Context mode: track which lines to output
                match_line_nums = []
                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        match_line_nums.append(line_num)

                if not match_line_nums:
                    continue

                # Build output with context windows
                output_ranges = []
                for ln in match_line_nums:
                    start = max(1, ln - ctx_before)
                    end = min(len(lines), ln + ctx_after)
                    output_ranges.append((start, end, ln))

                # Merge overlapping ranges
                merged = [output_ranges[0]]
                for start, end, match_ln in output_ranges[1:]:
                    prev_start, prev_end, _ = merged[-1]
                    if start <= prev_end + 1:
                        merged[-1] = (prev_start, max(prev_end, end), merged[-1][2])
                    else:
                        merged.append((start, end, match_ln))

                last_end = 0
                for start, end, _ in merged:
                    if last_end > 0 and start > last_end + 1:
                        matches.append("--")
                    for ln in range(start, end + 1):
                        line_text = self._cap_line(lines[ln - 1])
                        prefix = ">" if ln in match_line_nums else " "
                        matches.append(f"{rel}:{ln}:{prefix} {line_text}")
                        if len(matches) >= max_results:
                            return "\n".join(matches) + f"\n... (truncated at {max_results} results)"
                    last_end = end
            else:
                # Simple line-by-line matching (original behavior)
                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        line_text = self._cap_line(line)
                        matches.append(f"{rel}:{line_num}: {line_text}")
                        if len(matches) >= max_results:
                            return "\n".join(matches) + f"\n... (truncated at {max_results} results)"

        if not matches:
            return f"No matches found for pattern '{regex.pattern}' in {search_root}"
        return "\n".join(matches)

    def _search_files_only(
        self, regex, search_root: Path, include: str, max_results: int, multiline: bool,
    ) -> str:
        """Return only file paths that contain at least one match."""
        matched_files = []
        for file_path in self._walk_files(search_root, include):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            if multiline:
                if regex.search(text):
                    matched_files.append(self._rel_path(file_path))
            else:
                for line in text.splitlines():
                    if regex.search(line):
                        matched_files.append(self._rel_path(file_path))
                        break

            if len(matched_files) >= max_results:
                return "\n".join(matched_files) + f"\n... (truncated at {max_results} files)"

        if not matched_files:
            return f"No files match pattern '{regex.pattern}' in {search_root}"
        return "\n".join(matched_files)

    def _search_count(
        self, regex, search_root: Path, include: str, max_results: int, multiline: bool,
    ) -> str:
        """Return match count per file."""
        file_counts = []
        for file_path in self._walk_files(search_root, include):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            if multiline:
                count = len(regex.findall(text))
            else:
                count = sum(1 for line in text.splitlines() if regex.search(line))

            if count > 0:
                file_counts.append((self._rel_path(file_path), count))
                if len(file_counts) >= max_results:
                    break

        if not file_counts:
            return f"No matches found for pattern '{regex.pattern}' in {search_root}"

        # Sort by count descending
        file_counts.sort(key=lambda x: x[1], reverse=True)
        total = sum(c for _, c in file_counts)
        lines = [f"{path}: {count}" for path, count in file_counts]
        lines.append(f"\nTotal: {total} matches in {len(file_counts)} files")
        if len(file_counts) >= max_results:
            lines.append(f"... (truncated at {max_results} files)")
        return "\n".join(lines)

    @staticmethod
    def _cap_line(line: str) -> str:
        """Cap a single line to prevent huge matches from blowing up context."""
        text = line.rstrip()
        if len(text) > 200:
            return text[:200] + "..."
        return text

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
    read_only = True

    def __init__(self, root_path: str | None = None):
        self.root_path = root_path

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="glob_files",
            description=(
                "Find files matching a glob pattern in a directory tree. "
                "Returns file paths with sizes, sorted by modification time (newest first).\n\n"
                "When to use:\n"
                "- To find files by name or extension (e.g. '**/*.py', 'src/**/*.ts')\n"
                "- To see all files of a certain type in the project\n\n"
                "When NOT to use:\n"
                "- To search file CONTENTS → use grep_files instead\n"
                "- To see a specific directory → use list_directory instead"
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


class RepoMapTool(Tool):
    """Query the project's code structure map — classes, functions, types across all languages."""
    read_only = True

    def __init__(self, repo_map: "RepoMap"):
        self._repo_map = repo_map

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="repo_map",
            description=(
                "Get a structural map of the project's source code: classes, functions, "
                "methods, types (structs, interfaces, enums) across all supported languages "
                "(Python, JS/TS, Go, Rust, Java, C/C++).\n\n"
                "Use this BEFORE grepping or reading files to understand the codebase layout. "
                "Much faster than exploring with list_directory + read_file.\n\n"
                "Examples:\n"
                "- repo_map() → full project structure\n"
                "- repo_map(path='src/core') → only files under src/core/\n"
                "- repo_map(language='python') → only Python files\n"
                "- repo_map(query='search') → only symbols containing 'search'\n"
                "- repo_map(query='Router', language='go') → Go symbols matching 'Router'"
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type=ToolParameterType.STRING,
                    description="Subdirectory to limit the map to (e.g. 'src/core', 'blipshell/memory'). Default: entire project.",
                    required=False,
                ),
                ToolParameter(
                    name="language",
                    type=ToolParameterType.STRING,
                    description="Filter by language: python, javascript, typescript, go, rust, java, c, cpp. Default: all languages.",
                    required=False,
                ),
                ToolParameter(
                    name="query",
                    type=ToolParameterType.STRING,
                    description="Filter to symbols (class/function/type names) containing this substring (case-insensitive). Default: show all.",
                    required=False,
                ),
                ToolParameter(
                    name="max_lines",
                    type=ToolParameterType.INTEGER,
                    description="Maximum output lines (default: 200). Use higher values for large projects.",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        path: str = "",
        language: str = "",
        query: str = "",
        max_lines: int = 200,
        **kwargs,
    ) -> str:
        result = self._repo_map.build(
            max_lines=max_lines,
            path_filter=path,
            language_filter=language,
            symbol_query=query,
        )
        if not result:
            parts = []
            if path:
                parts.append(f"path='{path}'")
            if language:
                parts.append(f"language='{language}'")
            if query:
                parts.append(f"query='{query}'")
            filter_desc = ", ".join(parts) if parts else "no filters"
            return f"No code definitions found ({filter_desc}). Try broader filters or check the path."
        return result
