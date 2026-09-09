"""Every unguarded third-party import in blipshell/ is a declared dependency.

The 2026-09-08 external review installed the declared package + dev extra
into a clean environment and test collection failed in five files: nightly
imports centroid_tagger, which imports numpy at module scope, and numpy was
not in pyproject.toml. A machine with leftover packages hides that class of
bug forever; this test does not. It walks the source with `ast`, keeps
module-scope imports that are not inside try/except or TYPE_CHECKING
(those are the optional-extra pattern: presidio, spacy), maps each import
root to its distribution via importlib.metadata, and checks the distribution
is declared under [project] dependencies or an optional extra.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "blipshell"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared() -> set[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    specs = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        specs.extend(extra)
    out = set()
    for spec in specs:
        name = re.split(r"[\[><=!~;\s]", spec, maxsplit=1)[0]
        out.add(_norm(name))
    return out


class _Imports(ast.NodeVisitor):
    """Collect import roots that would execute unconditionally on import."""

    def __init__(self):
        self.roots: set[str] = set()
        self._guard = 0

    def visit_Try(self, node):  # optional-dependency pattern
        self._guard += 1
        self.generic_visit(node)
        self._guard -= 1

    def visit_If(self, node):
        test = ast.unparse(node.test)
        if "TYPE_CHECKING" in test:
            for n in node.orelse:
                self.visit(n)
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # a lazy import inside a function runs only when that code path does;
        # it is still a real dependency, so it counts
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node):
        if not self._guard:
            for a in node.names:
                self.roots.add(a.name.split(".")[0])

    def visit_ImportFrom(self, node):
        if not self._guard and node.module and node.level == 0:
            self.roots.add(node.module.split(".")[0])


def _source_imports() -> dict[str, set[str]]:
    """root module -> files importing it (unguarded)."""
    where: dict[str, set[str]] = {}
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        v = _Imports()
        v.visit(tree)
        for r in v.roots:
            where.setdefault(r, set()).add(str(py.relative_to(REPO)))
    return where


# Module -> distribution for packages that may not be installed on the dev box
# (packages_distributions() can only map what is present). Keep this short:
# every entry is a dependency this test cannot fully verify here.
NOT_INSTALLED_HERE = {"telegram": "python-telegram-bot"}

# Repo-local packages imported by the application. `scripts/` lives at the repo
# root, outside the `blipshell` package: it resolves under an editable install
# (the .pth puts the repo root on sys.path) and would NOT under a wheel install.
# nightly.py imports scripts.backup_db and scripts.backfill_session_summaries
# lazily. Tracked in V3_PLAN "Explicitly NOT doing / open"; not a pyproject
# dependency, so exempt here rather than silently passing.
REPO_LOCAL = {"scripts"}


def test_every_unguarded_third_party_import_is_declared():
    declared = _declared()
    dists = packages_distributions()  # module -> [distribution, ...]
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    problems = []
    for root, files in sorted(_source_imports().items()):
        if root in stdlib or root in {"blipshell", "tests"} or root in REPO_LOCAL:
            continue
        candidates = {_norm(d) for d in dists.get(root, [])}
        if not candidates and root in NOT_INSTALLED_HERE:
            candidates = {_norm(NOT_INSTALLED_HERE[root])}
        if not candidates:
            problems.append(f"{root}: not installed here, cannot map to a distribution "
                            f"(imported unguarded by {sorted(files)[0]})")
            continue
        if not candidates & declared:
            problems.append(f"{root} -> {sorted(candidates)}: not in pyproject dependencies "
                            f"(imported unguarded by {sorted(files)[0]})")
    assert not problems, "\n".join(problems)


def test_numpy_is_declared_and_used():
    """The specific regression: centroid_tagger imports numpy at module scope
    and nightly imports centroid_tagger."""
    assert "numpy" in _declared()
    assert "numpy" in _source_imports()


def test_guarded_imports_are_not_required():
    """Sanity check on the walker: an import inside try/except is optional and
    must not be demanded of the base install."""
    src = "try:\n    import presidio_analyzer\nexcept ImportError:\n    presidio_analyzer = None\nimport json\n"
    v = _Imports()
    v.visit(ast.parse(src))
    assert v.roots == {"json"}
