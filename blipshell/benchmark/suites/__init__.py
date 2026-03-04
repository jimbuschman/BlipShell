"""Suite registry for the unified benchmark system."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blipshell.benchmark.suites.base import BenchmarkSuite

# Lazy imports to avoid circular deps and heavy startup cost
_SUITE_CLASSES: dict[str, str] = {
    "pipeline": "blipshell.benchmark.suites.pipeline:PipelineSuite",
    "coding": "blipshell.benchmark.suites.coding:CodingSuite",
    "tagging": "blipshell.benchmark.suites.tagging:TaggingSuite",
    "reflection": "blipshell.benchmark.suites.reflection:ReflectionSuite",
    "embedding": "blipshell.benchmark.suites.embedding:EmbeddingSuite",
}


def _load_suite(key: str) -> BenchmarkSuite:
    """Lazy-load and instantiate a suite class."""
    module_path, class_name = _SUITE_CLASSES[key].rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def get_suite(name: str) -> BenchmarkSuite:
    """Get a suite by name."""
    if name not in _SUITE_CLASSES:
        raise ValueError(f"Unknown suite: {name!r}. Available: {list(_SUITE_CLASSES)}")
    return _load_suite(name)


def list_suites() -> list[tuple[str, str]]:
    """Return list of (name, description) for all registered suites."""
    result = []
    for name in _SUITE_CLASSES:
        suite = _load_suite(name)
        result.append((name, suite.description))
    return result


SUITES = list(_SUITE_CLASSES.keys())
