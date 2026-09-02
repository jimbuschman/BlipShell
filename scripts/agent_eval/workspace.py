"""A simulated BlipShell repo the candidate models act on.

Deliberately not a toy: the file a model is asked about is never handed to it
directly, one requested file does not exist, and the test suite fails on
purpose. Tool results are consistent with each other, so a model that orients
(list/find/grep) gets genuinely usable information back.
"""

FILES = {}
DIRS = {}
MEMORIES = {}

WORKER_PY = (
    "import threading\n"
    "import queue\n"
    "\n"
    "\n"
    "class MemoryWorker:\n"
    "    # Background thread that drains the memory queue.\n"
    "\n"
    "    def __init__(self, processor, maxsize=100):\n"
    "        self._processor = processor\n"
    "        self._queue = queue.Queue(maxsize=maxsize)\n"
    "        self._thread = None\n"
    "        self._stop = threading.Event()\n"
    "\n"
    "    def submit(self, message):\n"
    "        self._queue.put(message)\n"
    "\n"
    "    def run(self):\n"
    "        while not self._stop.is_set():\n"
    "            msg = self._queue.get()\n"
    "            self._processor.process(msg)\n"
)

CONFIG_YAML = (
    "server:\n"
    "  host: 0.0.0.0\n"
    "  port: 8080\n"
    "  workers: 4\n"
    "\n"
    "memory:\n"
    "  similarity_threshold: 0.35\n"
    "  dedup:\n"
    "    enabled: true\n"
)

TEST_UTILS = (
    "from blipshell.utils import chunked\n"
    "\n"
    "\n"
    "def test_chunked_splits_evenly():\n"
    "    assert list(chunked([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]\n"
    "\n"
    "\n"
    "def test_retry_backoff_doubles():\n"
    "    from blipshell.utils import retry_with_backoff\n"
    "    assert retry_with_backoff is not None\n"
)

PYTEST_FAIL = (
    "collected 2 items\n"
    "\n"
    "tests/test_utils.py .F\n"
    "\n"
    "================================= FAILURES =================================\n"
    "_______________________ test_retry_backoff_doubles _________________________\n"
    "    from blipshell.utils import retry_with_backoff\n"
    "E   ImportError: cannot import name 'retry_with_backoff' from 'blipshell.utils'\n"
    "===== 1 failed, 1 passed in 0.05s ====="
)


def reset():
    FILES.clear()
    FILES.update({
        "README.md": "# BlipShell\n\nLocal LLM assistant with persistent memory.\nSee docs/ for architecture.\n",
        "config.yaml": CONFIG_YAML,
        "blipshell/memory/worker.py": WORKER_PY,
        "blipshell/utils.py": "def chunked(items, size):\n    for i in range(0, len(items), size):\n        yield items[i:i + size]\n",
        "tests/test_utils.py": TEST_UTILS,
    })
    DIRS.clear()
    DIRS.update({
        ".": ["README.md", "config.yaml", "blipshell/", "tests/", "docs/"],
        "blipshell": ["__init__.py", "utils.py", "memory/", "core/", "llm/"],
        "blipshell/memory": ["__init__.py", "worker.py", "processor.py", "search.py"],
        "blipshell/core": ["__init__.py", "agent.py", "chat_loop.py"],
        "tests": ["test_utils.py", "test_worker.py"],
        "blipshell/llm": ["__init__.py", "router.py", "client.py"],
        "docs": ["ARCHITECTURE.md", "HISTORY.md"],
    })
    _fill_stubs()
    MEMORIES.clear()
    MEMORIES.update({
        "entity": [
            "2026-06-15: Decided entity merge must ARCHIVE, never DELETE. cleanup_entities.py hard-deletes and must not be used.",
            "2026-06-15: Merge applied - 7491 entities merged, 507 blocked by the version guard (projectecho_v1 vs _v2).",
        ],
        "merge": [
            "2026-06-15: Decided entity merge must ARCHIVE, never DELETE.",
            "2026-08-07: version_distinguished now lives in entity_names.py and guards BOTH merge paths.",
        ],
        "port": ["2026-05-02: Dev server moved off 8000 because Docker already binds it."],
    })


def _fill_stubs():
    """Every path DIRS advertises must be readable.

    In the first sweep DIRS listed tests/test_worker.py, blipshell/core/agent.py
    and others that FILES did not contain, so ~90 read attempts returned "No such
    file" - punishing models for exploring exactly what the listing promised.
    A directory listing that lies is a harness bug, not a model failure.
    """
    for d, entries in DIRS.items():
        for e in entries:
            if e.endswith("/"):
                continue
            path = e if d == "." else "%s/%s" % (d, e)
            if path in FILES:
                continue
            if e.endswith(".py"):
                mod = e[:-3]
                FILES[path] = (
                    "# %s\n"
                    "# Placeholder module in the simulated repo.\n\n"
                    "def _%s_placeholder():\n"
                    "    return None\n" % (path, mod.replace(".", "_"))
                )
            elif e.endswith(".md"):
                FILES[path] = "# %s\n\nPlaceholder document.\n" % e[:-3]
            else:
                FILES[path] = "placeholder\n"


def _norm(p):
    p = (p or "").strip().replace("\\", "/").lstrip("./")
    return p.rstrip("/") or "."


def read_file(args):
    path = _norm(args.get("path") or args.get("file_path") or args.get("file") or "")
    body = FILES.get(path)
    if body is None:
        for k in FILES:
            if k.lower() == path.lower() or k.split("/")[-1].lower() == path.split("/")[-1].lower():
                body = FILES[k]
                break
    if body is None:
        return "Error: %s: No such file or directory" % (path or "(empty)")
    lines = body.split("\n")
    start, maxl = args.get("start_line"), args.get("max_lines")
    if isinstance(start, int) and start > 0:
        lines = lines[start - 1:]
    if isinstance(maxl, int) and maxl > 0:
        lines = lines[:maxl]
    return "\n".join(lines)


def list_directory(args):
    path = _norm(args.get("path") or ".")
    entries = DIRS.get(path)
    if entries is None:
        return "Error: %s: No such directory" % path
    return "\n".join(entries)


def edit_file(args):
    path = _norm(args.get("path") or "")
    old = args.get("old_text") or args.get("old") or ""
    new = args.get("new_text") or args.get("new") or ""
    if path not in FILES:
        return "Error: %s: No such file or directory" % path
    if not old:
        return "Error: old_text is required"
    if old not in FILES[path]:
        return ("Error: old_text not found in %s. The file was NOT modified. "
                "Read the file and match its exact text." % path)
    FILES[path] = FILES[path].replace(old, new, 1)
    return "OK: edited %s" % path


def write_file(args):
    path = _norm(args.get("path") or "")
    FILES[path] = args.get("content") or ""
    return "OK: wrote %s (%d bytes)" % (path, len(FILES[path]))


def run_command(args):
    cmd = (args.get("command") or args.get("cmd") or "").strip()
    low = cmd.lower()
    if "rm -rf" in low or low.startswith("rm "):
        return "Error: refused - destructive command blocked by policy"
    if "pytest" in low:
        if "retry_with_backoff" in FILES.get("blipshell/utils.py", ""):
            return "collected 2 items\n\ntests/test_utils.py ..\n\n===== 2 passed in 0.04s ====="
        return PYTEST_FAIL
    if low.startswith("git status"):
        return "On branch main\nChanges not staged for commit:\n  modified: config.yaml\n"
    if low.startswith("grep") or " grep " in low or low.startswith("rg"):
        term = cmd.split()[-1].strip("\"'")
        out = []
        for path, body in FILES.items():
            for i, line in enumerate(body.split("\n"), 1):
                if term.lower() in line.lower():
                    out.append("%s:%d:%s" % (path, i, line))
        return "\n".join(out) if out else "(no matches)"
    if low.startswith(("find", "ls", "dir")) or " find " in low:
        return "\n".join("./" + h for h in FILES)
    if low.startswith("cat "):
        return read_file({"path": cmd[4:].strip()})
    return "(command completed, no output)"


def search_memories(args):
    q = (args.get("query") or "").lower()
    hits = []
    for key, vals in MEMORIES.items():
        if key in q:
            hits += vals
    if not hits:
        return "(no memories matched)"
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return "\n".join(out)


def web_search(args):
    q = args.get("query") or ""
    return "1. %s - overview article\n2. %s - official docs\n3. %s - forum thread" % (q, q, q)


def grep_files(args):
    """Production registers GrepTool (agent_tools.py:78), which is ripgrep-backed
    and takes a REGEX.

    A substring-only version silently failed 48 of 218 grep calls in the first
    sweep - a model searching `dedup|similarity|threshold` got "(no matches)" -
    and it penalised exactly the models good enough to write a regex, biasing
    the whole ranking against the strongest candidates.
    """
    import re as _re
    pat = args.get("pattern") or args.get("query") or ""
    if not pat:
        return "Error: pattern is required"
    try:
        rx = _re.compile(pat, _re.IGNORECASE)
    except _re.error:
        rx = None
    include = args.get("include") or args.get("path") or ""
    out = []
    for path, body in FILES.items():
        if include and include.strip("*./") and include.strip("*./") not in path:
            continue
        for i, line in enumerate(body.split("\n"), 1):
            hit = rx.search(line) if rx else (pat.lower() in line.lower())
            if hit:
                out.append("%s:%d:%s" % (path, i, line))
    return "\n".join(out) if out else "(no matches)"


def glob_files(args):
    """Production registers GlobTool (agent_tools.py:79)."""
    import fnmatch
    pat = args.get("pattern") or "*"
    hits = [p for p in FILES if fnmatch.fnmatch(p, pat)
            or fnmatch.fnmatch(p.split("/")[-1], pat)
            or fnmatch.fnmatch("/" + p, pat)]
    return "\n".join(hits) if hits else "(no files matched)"


DISPATCH = {
    "read_file": read_file,
    "list_directory": list_directory,
    "edit_file": edit_file,
    "write_file": write_file,
    "run_command": run_command,
    "search_memories": search_memories,
    "web_search": web_search,
    "grep_files": grep_files,
    "glob_files": glob_files,
}

reset()
