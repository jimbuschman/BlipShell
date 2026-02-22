"""AST-based repository map for project context.

Parses Python files to extract class and function definitions, building
a compact code map that gives LLMs structural understanding of a codebase
without reading every file. Inspired by Aider's repo map (which uses
tree-sitter + PageRank), but uses Python's stdlib ast module for zero
external dependencies.

The map is injected into the project context instead of (or alongside)
the raw file tree, dramatically reducing the need for list_directory
and exploratory read_file calls.

Cache: Maps are cached per-file by mtime, so only changed files get
re-parsed on subsequent calls.
"""

import ast
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories to skip when walking the tree
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs", ".hg",
    ".vs", ".idea", ".vscode", "backups", "Clean DB",
}

# Max files to parse (prevent runaway on huge repos)
MAX_FILES = 200

# Max total lines in the map output
MAX_MAP_LINES = 150


@dataclass
class FileDefs:
    """Definitions extracted from a single Python file."""
    rel_path: str
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    mtime: float = 0.0


def _extract_defs(file_path: Path) -> FileDefs:
    """Parse a Python file and extract top-level and class-level definitions.

    Returns a FileDefs with class names (including method summaries)
    and standalone function names.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        logger.debug("Failed to parse %s: %s", file_path, e)
        return FileDefs(rel_path=str(file_path))

    defs = FileDefs(
        rel_path=str(file_path),
        mtime=file_path.stat().st_mtime,
    )

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in ast.iter_child_nodes(node):
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    # Skip dunder methods except __init__
                    if item.name.startswith("__") and item.name != "__init__":
                        continue
                    args = _format_args(item.args)
                    prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                    methods.append(f"{prefix}{item.name}({args})")

            method_summary = ""
            if methods:
                # Show first 5 methods, elide the rest
                shown = methods[:5]
                if len(methods) > 5:
                    shown.append(f"... +{len(methods) - 5} more")
                method_summary = " { " + ", ".join(shown) + " }"
            defs.classes.append(f"{node.name}{method_summary}")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = _format_args(node.args)
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            defs.functions.append(f"{prefix}{node.name}({args})")

    return defs


def _format_args(args: ast.arguments) -> str:
    """Format function arguments compactly."""
    parts = []
    # Regular positional args (skip 'self' and 'cls')
    for arg in args.args:
        if arg.arg in ("self", "cls"):
            continue
        parts.append(arg.arg)

    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return ", ".join(parts)


class RepoMap:
    """Builds and caches an AST-based code map for a project directory."""

    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self._cache: dict[str, FileDefs] = {}  # rel_path -> FileDefs

    def build(self, max_lines: int = MAX_MAP_LINES) -> str:
        """Build the repo map string.

        Returns a compact representation of all Python files' structure.
        Uses cache for unchanged files (by mtime).
        """
        start = time.monotonic()
        py_files = self._find_python_files()

        all_defs: list[FileDefs] = []
        cache_hits = 0

        for fpath in py_files:
            rel = self._rel_path(fpath)
            mtime = fpath.stat().st_mtime

            # Check cache
            cached = self._cache.get(rel)
            if cached and cached.mtime == mtime:
                all_defs.append(cached)
                cache_hits += 1
                continue

            # Parse and cache
            defs = _extract_defs(fpath)
            defs.rel_path = rel
            self._cache[rel] = defs
            all_defs.append(defs)

        # Build output
        lines = []
        for defs in all_defs:
            if not defs.classes and not defs.functions:
                continue

            file_parts = []
            for cls in defs.classes:
                file_parts.append(f"  class {cls}")
            for func in defs.functions:
                file_parts.append(f"  {func}")

            lines.append(defs.rel_path)
            lines.extend(file_parts)

            if len(lines) >= max_lines:
                lines.append(f"... ({len(all_defs) - len(lines)} more files)")
                break

        elapsed = (time.monotonic() - start) * 1000
        logger.debug(
            "Repo map: %d files, %d cached, %d lines, %.0fms",
            len(py_files), cache_hits, len(lines), elapsed,
        )

        if not lines:
            return ""

        return "\n".join(lines)

    def _find_python_files(self) -> list[Path]:
        """Find all .py files in the repo, respecting skip dirs."""
        results = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune skipped directories
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for fname in filenames:
                if fname.endswith(".py"):
                    results.append(Path(dirpath) / fname)
                    if len(results) >= MAX_FILES:
                        return results

        return results

    def _rel_path(self, fpath: Path) -> str:
        """Get path relative to project root."""
        try:
            return str(fpath.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(fpath)

    def invalidate(self, rel_path: str):
        """Remove a file from cache (e.g., after edit)."""
        self._cache.pop(rel_path, None)

    def clear_cache(self):
        """Clear the entire cache."""
        self._cache.clear()
