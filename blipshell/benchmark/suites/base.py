"""Base class for benchmark suites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from blipshell.benchmark.models import SuiteResult
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import BlipShellConfig


class BenchmarkSuite(ABC):
    """Protocol for benchmark suites."""

    name: str
    description: str
    task_types: list[str]
    needs_db: bool
    needs_router: bool
    quick_samples: int
    thorough_samples: int

    @abstractmethod
    async def run(
        self,
        models: list[str],
        *,
        router_factory: Callable[[str], LLMRouter] | None = None,
        config: BlipShellConfig | None = None,
        db_path: str | None = None,
        ollama_url: str = "http://localhost:11434",
        thorough: bool = False,
        on_status: Callable[[str], None] | None = None,
        config_path: str | None = None,
    ) -> list[SuiteResult]:
        """Run the suite across all models.

        Args:
            models: Model names to benchmark.
            router_factory: Callable that creates an LLMRouter for a given model.
            config: Full config (needed for cloud models).
            db_path: Path to SQLite DB (for suites that need_db).
            ollama_url: Ollama server URL.
            thorough: If True, use thorough_samples; else quick_samples.
            on_status: Callback for progress updates.
            config_path: Path to config.yaml (for interactive suite agent bootstrap).

        Returns:
            One SuiteResult per model.
        """
        ...
