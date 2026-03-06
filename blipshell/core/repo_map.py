"""Multi-language repository map for project context.

Parses source files to extract class/function/type definitions, building
a compact code map that gives LLMs structural understanding of a codebase
without reading every file. Inspired by Aider's repo map approach.

Python: Uses stdlib `ast` module for accurate parsing (zero dependencies).
Other languages: Uses regex-based extraction for JS/TS, Go, Rust, Java, C/C++.

The map is injected into the project context (system prompt) for small projects
and available on-demand via the `repo_map` tool for large projects.

Cache: Maps are cached per-file by mtime, so only changed files get
re-parsed on subsequent calls.
"""

import ast
import logging
import os
import re
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
MAX_FILES = 300

# Max total lines in the map output
MAX_MAP_LINES = 200

# Language extension mapping
LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".hh", ".cxx"],
}

# Reverse lookup: extension -> language
EXT_TO_LANG: dict[str, str] = {}
for _lang, _exts in LANG_EXTENSIONS.items():
    for _ext in _exts:
        EXT_TO_LANG[_ext] = _lang

# All supported extensions
ALL_EXTENSIONS = set(EXT_TO_LANG.keys())


@dataclass
class FileDefs:
    """Definitions extracted from a source file."""
    rel_path: str
    language: str = ""
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)  # structs, interfaces, enums, typedefs
    mtime: float = 0.0


# ---------------------------------------------------------------------------
# Python extractor (AST-based, accurate)
# ---------------------------------------------------------------------------

def _extract_python(file_path: Path) -> FileDefs:
    """Parse a Python file with ast and extract definitions."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        logger.debug("Failed to parse %s: %s", file_path, e)
        return FileDefs(rel_path=str(file_path), language="python")

    defs = FileDefs(
        rel_path=str(file_path),
        language="python",
        mtime=file_path.stat().st_mtime,
    )

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in ast.iter_child_nodes(node):
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    if item.name.startswith("__") and item.name != "__init__":
                        continue
                    args = _format_py_args(item.args)
                    prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                    methods.append(f"{prefix}{item.name}({args})")

            method_summary = ""
            if methods:
                shown = methods[:5]
                if len(methods) > 5:
                    shown.append(f"... +{len(methods) - 5} more")
                method_summary = " { " + ", ".join(shown) + " }"
            defs.classes.append(f"{node.name}{method_summary}")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = _format_py_args(node.args)
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            defs.functions.append(f"{prefix}{node.name}({args})")

    return defs


def _format_py_args(args: ast.arguments) -> str:
    """Format Python function arguments compactly."""
    parts = []
    for arg in args.args:
        if arg.arg in ("self", "cls"):
            continue
        parts.append(arg.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Regex-based extractors for other languages
# ---------------------------------------------------------------------------

# Patterns match top-level definitions only (no leading whitespace or minimal indent).
# Each returns (kind, name, detail) tuples.

_JS_TS_PATTERNS = [
    # class Foo extends Bar {
    re.compile(r"^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?"),
    # interface Foo {
    re.compile(r"^(?:export\s+)?interface\s+(\w+)"),
    # type Foo = ...
    re.compile(r"^(?:export\s+)?type\s+(\w+)"),
    # enum Foo {
    re.compile(r"^(?:export\s+)?(?:const\s+)?enum\s+(\w+)"),
]

_JS_TS_FUNC_PATTERNS = [
    # function foo(...)  /  async function foo(...)  /  export function foo(...)
    re.compile(r"^(?:export\s+)?(?:export\s+default\s+)?(?:async\s+)?function\s*\*?\s+(\w+)\s*\(([^)]*)\)"),
    # const foo = (...) =>  /  const foo = function(...)
    re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function\s*)?\(([^)]*)\)"),
]


def _extract_js_ts(file_path: Path) -> FileDefs:
    """Extract definitions from JavaScript/TypeScript files via regex."""
    defs = FileDefs(
        rel_path=str(file_path),
        language=EXT_TO_LANG.get(file_path.suffix.lower(), "javascript"),
        mtime=file_path.stat().st_mtime,
    )
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, UnicodeDecodeError):
        return defs

    # Track current class for method extraction
    current_class = None
    class_methods: list[str] = []
    brace_depth = 0

    for line in lines:
        stripped = line.rstrip()

        # Track brace depth for class body detection
        if current_class is not None:
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0:
                # Class ended
                summary = ""
                if class_methods:
                    shown = class_methods[:5]
                    if len(class_methods) > 5:
                        shown.append(f"... +{len(class_methods) - 5} more")
                    summary = " { " + ", ".join(shown) + " }"
                defs.classes.append(f"{current_class}{summary}")
                current_class = None
                class_methods = []
                continue

            # Look for methods inside class
            m = re.match(r"\s+(?:async\s+)?(?:static\s+)?(?:get\s+|set\s+)?(\w+)\s*\(([^)]*)\)", stripped)
            if m and m.group(1) not in ("if", "for", "while", "switch", "catch", "constructor"):
                class_methods.append(f"{m.group(1)}({_compact_params(m.group(2))})")
            elif re.match(r"\s+(?:async\s+)?constructor\s*\(", stripped):
                class_methods.append("constructor(...)")
            continue

        # Top-level class detection
        for pat in _JS_TS_PATTERNS:
            m = pat.match(stripped)
            if m:
                name = m.group(1)
                if pat.pattern.startswith(r"^(?:export\s+)?(?:abstract\s+)?class"):
                    current_class = name
                    class_methods = []
                    brace_depth = stripped.count("{") - stripped.count("}")
                elif "interface" in pat.pattern:
                    defs.types.append(f"interface {name}")
                elif "type" in pat.pattern:
                    defs.types.append(f"type {name}")
                elif "enum" in pat.pattern:
                    defs.types.append(f"enum {name}")
                break
        else:
            # Top-level function detection
            for pat in _JS_TS_FUNC_PATTERNS:
                m = pat.match(stripped)
                if m:
                    name = m.group(1)
                    params = _compact_params(m.group(2))
                    prefix = "async " if "async" in stripped.split(name)[0] else ""
                    defs.functions.append(f"{prefix}{name}({params})")
                    break

    # Handle class that wasn't closed (file-end)
    if current_class:
        summary = ""
        if class_methods:
            shown = class_methods[:5]
            if len(class_methods) > 5:
                shown.append(f"... +{len(class_methods) - 5} more")
            summary = " { " + ", ".join(shown) + " }"
        defs.classes.append(f"{current_class}{summary}")

    return defs


_GO_PATTERNS = [
    # func FuncName(params) returnType {
    re.compile(r"^func\s+(\w+)\s*\(([^)]*)\)"),
    # func (r *Receiver) MethodName(params) returnType {
    re.compile(r"^func\s+\([^)]+\)\s+(\w+)\s*\(([^)]*)\)"),
]

_GO_TYPE_PATTERNS = [
    # type Name struct {
    re.compile(r"^type\s+(\w+)\s+(struct|interface)\b"),
]


def _extract_go(file_path: Path) -> FileDefs:
    """Extract definitions from Go files via regex."""
    defs = FileDefs(rel_path=str(file_path), language="go", mtime=file_path.stat().st_mtime)
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, UnicodeDecodeError):
        return defs

    for line in lines:
        stripped = line.rstrip()

        for pat in _GO_TYPE_PATTERNS:
            m = pat.match(stripped)
            if m:
                defs.types.append(f"{m.group(2)} {m.group(1)}")
                break
        else:
            for pat in _GO_PATTERNS:
                m = pat.match(stripped)
                if m:
                    name = m.group(1)
                    params = _compact_params(m.group(2))
                    defs.functions.append(f"{name}({params})")
                    break

    return defs


_RUST_PATTERNS = [
    # pub fn name(params) -> Type {
    re.compile(r"^(?:pub(?:\([\w:]+\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)"),
]

_RUST_TYPE_PATTERNS = [
    # struct Name  /  pub struct Name<T>
    re.compile(r"^(?:pub(?:\([\w:]+\))?\s+)?struct\s+(\w+)"),
    # enum Name
    re.compile(r"^(?:pub(?:\([\w:]+\))?\s+)?enum\s+(\w+)"),
    # trait Name
    re.compile(r"^(?:pub(?:\([\w:]+\))?\s+)?trait\s+(\w+)"),
    # impl Name  /  impl Trait for Name
    re.compile(r"^impl(?:<[^>]*>)?\s+(?:(\w+)\s+for\s+)?(\w+)"),
]


def _extract_rust(file_path: Path) -> FileDefs:
    """Extract definitions from Rust files via regex."""
    defs = FileDefs(rel_path=str(file_path), language="rust", mtime=file_path.stat().st_mtime)
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, UnicodeDecodeError):
        return defs

    for line in lines:
        stripped = line.rstrip()

        for pat in _RUST_TYPE_PATTERNS:
            m = pat.match(stripped)
            if m:
                if "struct" in pat.pattern:
                    defs.types.append(f"struct {m.group(1)}")
                elif "enum" in pat.pattern:
                    defs.types.append(f"enum {m.group(1)}")
                elif "trait" in pat.pattern:
                    defs.types.append(f"trait {m.group(1)}")
                elif "impl" in pat.pattern:
                    trait_name = m.group(1)
                    type_name = m.group(2)
                    if trait_name:
                        defs.types.append(f"impl {trait_name} for {type_name}")
                    else:
                        defs.types.append(f"impl {type_name}")
                break
        else:
            for pat in _RUST_PATTERNS:
                m = pat.match(stripped)
                if m:
                    name = m.group(1)
                    params = _compact_params(m.group(2))
                    prefix = "async " if "async" in stripped.split("fn")[0] else ""
                    defs.functions.append(f"{prefix}{name}({params})")
                    break

    return defs


_JAVA_CLASS_RE = re.compile(
    r"^(?:public|private|protected)?\s*(?:abstract\s+)?(?:static\s+)?"
    r"(?:final\s+)?(?:class|interface|enum|record)\s+(\w+)"
)
_JAVA_METHOD_RE = re.compile(
    r"^\s{2,8}(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?"
    r"(?:abstract\s+)?(?:synchronized\s+)?(?:\w+(?:<[^>]*>)?(?:\[\])*\s+)"
    r"(\w+)\s*\(([^)]*)\)"
)


def _extract_java(file_path: Path) -> FileDefs:
    """Extract definitions from Java files via regex."""
    defs = FileDefs(rel_path=str(file_path), language="java", mtime=file_path.stat().st_mtime)
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, UnicodeDecodeError):
        return defs

    current_class = None
    class_methods: list[str] = []

    for line in lines:
        stripped = line.rstrip()

        m = _JAVA_CLASS_RE.match(stripped)
        if m:
            # Save previous class
            if current_class:
                summary = ""
                if class_methods:
                    shown = class_methods[:5]
                    if len(class_methods) > 5:
                        shown.append(f"... +{len(class_methods) - 5} more")
                    summary = " { " + ", ".join(shown) + " }"
                defs.classes.append(f"{current_class}{summary}")

            current_class = m.group(1)
            class_methods = []
            continue

        if current_class:
            m = _JAVA_METHOD_RE.match(stripped)
            if m:
                name = m.group(1)
                params = _compact_params(m.group(2))
                class_methods.append(f"{name}({params})")

    # Save last class
    if current_class:
        summary = ""
        if class_methods:
            shown = class_methods[:5]
            if len(class_methods) > 5:
                shown.append(f"... +{len(class_methods) - 5} more")
            summary = " { " + ", ".join(shown) + " }"
        defs.classes.append(f"{current_class}{summary}")

    return defs


_C_FUNC_RE = re.compile(
    r"^(?:static\s+)?(?:inline\s+)?(?:extern\s+)?(?:const\s+)?"
    r"(?:unsigned\s+|signed\s+|long\s+|short\s+)*"
    r"(?:\w+(?:\s*\*+)?)\s+"
    r"(\w+)\s*\(([^)]*)\)\s*\{"
)

_C_TYPE_RE = re.compile(
    r"^(?:typedef\s+)?(?:struct|union|enum)\s+(\w+)"
)


def _extract_c_cpp(file_path: Path) -> FileDefs:
    """Extract definitions from C/C++ files via regex."""
    lang = "cpp" if file_path.suffix.lower() in (".cpp", ".hpp", ".cc", ".hh", ".cxx") else "c"
    defs = FileDefs(rel_path=str(file_path), language=lang, mtime=file_path.stat().st_mtime)
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, UnicodeDecodeError):
        return defs

    for line in lines:
        stripped = line.rstrip()

        m = _C_TYPE_RE.match(stripped)
        if m:
            defs.types.append(m.group(0).split("{")[0].strip())
            continue

        # C++ class
        cm = re.match(r"^(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?", stripped)
        if cm and lang == "cpp":
            defs.classes.append(cm.group(1))
            continue

        m = _C_FUNC_RE.match(stripped)
        if m:
            name = m.group(1)
            # Skip common false positives
            if name not in ("if", "for", "while", "switch", "return", "sizeof"):
                params = _compact_params(m.group(2))
                defs.functions.append(f"{name}({params})")

    return defs


def _compact_params(params: str) -> str:
    """Compact a parameter string — keep just names or short type+name pairs."""
    if not params or not params.strip():
        return ""
    parts = [p.strip() for p in params.split(",")]
    result = []
    for p in parts:
        if not p:
            continue
        # For typed params (e.g., "int x", "string name"), keep last word
        tokens = p.split()
        if len(tokens) >= 2:
            # Strip pointer/reference markers for display
            name = tokens[-1].lstrip("*&")
            result.append(name)
        else:
            result.append(p.lstrip("*&"))
    compact = ", ".join(result)
    if len(compact) > 60:
        return compact[:57] + "..."
    return compact


# ---------------------------------------------------------------------------
# Language dispatcher
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, callable] = {
    "python": _extract_python,
    "javascript": _extract_js_ts,
    "typescript": _extract_js_ts,
    "go": _extract_go,
    "rust": _extract_rust,
    "java": _extract_java,
    "c": _extract_c_cpp,
    "cpp": _extract_c_cpp,
}


def _extract_defs(file_path: Path) -> FileDefs:
    """Extract definitions from a file based on its extension."""
    lang = EXT_TO_LANG.get(file_path.suffix.lower())
    if not lang:
        return FileDefs(rel_path=str(file_path))
    extractor = _EXTRACTORS.get(lang)
    if not extractor:
        return FileDefs(rel_path=str(file_path))
    return extractor(file_path)


class RepoMap:
    """Builds and caches a multi-language code map for a project directory."""

    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        self._cache: dict[str, FileDefs] = {}  # rel_path -> FileDefs

    def build(
        self,
        max_lines: int = MAX_MAP_LINES,
        path_filter: str = "",
        language_filter: str = "",
        symbol_query: str = "",
    ) -> str:
        """Build the repo map string.

        Args:
            max_lines: Maximum output lines.
            path_filter: Only include files under this subdirectory.
            language_filter: Only include files of this language (e.g. "python", "typescript").
            symbol_query: Only include files with symbols matching this substring (case-insensitive).

        Returns a compact representation of source files' structure.
        Uses cache for unchanged files (by mtime).
        """
        start = time.monotonic()
        source_files = self._find_source_files(path_filter, language_filter)

        all_defs: list[FileDefs] = []
        cache_hits = 0

        for fpath in source_files:
            rel = self._rel_path(fpath)
            try:
                mtime = fpath.stat().st_mtime
            except OSError:
                continue

            cached = self._cache.get(rel)
            if cached and cached.mtime == mtime:
                all_defs.append(cached)
                cache_hits += 1
                continue

            defs = _extract_defs(fpath)
            defs.rel_path = rel
            self._cache[rel] = defs
            all_defs.append(defs)

        # Apply symbol query filter
        if symbol_query:
            q = symbol_query.lower()
            filtered = []
            for defs in all_defs:
                matched_classes = [c for c in defs.classes if q in c.lower()]
                matched_funcs = [f for f in defs.functions if q in f.lower()]
                matched_types = [t for t in defs.types if q in t.lower()]
                if matched_classes or matched_funcs or matched_types:
                    filtered_defs = FileDefs(
                        rel_path=defs.rel_path,
                        language=defs.language,
                        classes=matched_classes,
                        functions=matched_funcs,
                        types=matched_types,
                        mtime=defs.mtime,
                    )
                    filtered.append(filtered_defs)
            all_defs = filtered

        # Build output
        lines = []
        files_with_defs = 0
        for defs in all_defs:
            if not defs.classes and not defs.functions and not defs.types:
                continue

            file_parts = []
            for cls in defs.classes:
                file_parts.append(f"  class {cls}")
            for typ in defs.types:
                file_parts.append(f"  {typ}")
            for func in defs.functions:
                file_parts.append(f"  {func}")

            lines.append(defs.rel_path)
            lines.extend(file_parts)
            files_with_defs += 1

            if len(lines) >= max_lines:
                remaining = len(all_defs) - files_with_defs
                if remaining > 0:
                    lines.append(f"... ({remaining} more files, use path or language filter to narrow)")
                break

        elapsed = (time.monotonic() - start) * 1000
        logger.debug(
            "Repo map: %d files scanned, %d cached, %d with defs, %d lines, %.0fms",
            len(source_files), cache_hits, files_with_defs, len(lines), elapsed,
        )

        if not lines:
            if symbol_query:
                return f"No symbols matching '{symbol_query}' found."
            return ""

        return "\n".join(lines)

    def _find_source_files(
        self, path_filter: str = "", language_filter: str = "",
    ) -> list[Path]:
        """Find all supported source files in the repo."""
        search_root = self.root
        if path_filter:
            candidate = (self.root / path_filter).resolve()
            if candidate.is_dir():
                search_root = candidate

        # Determine which extensions to include
        if language_filter:
            lang_key = language_filter.lower()
            exts = set(LANG_EXTENSIONS.get(lang_key, []))
            if not exts:
                # Try matching by extension shorthand
                for lang, lang_exts in LANG_EXTENSIONS.items():
                    if lang_key in lang or any(lang_key in e for e in lang_exts):
                        exts.update(lang_exts)
            if not exts:
                exts = ALL_EXTENSIONS
        else:
            exts = ALL_EXTENSIONS

        results = []
        for dirpath, dirnames, filenames in os.walk(search_root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for fname in filenames:
                suffix = os.path.splitext(fname)[1].lower()
                if suffix in exts:
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
