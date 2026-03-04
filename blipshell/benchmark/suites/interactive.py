"""Interactive suite — tool calling and coding (multi-turn harness).

Stub implementation: all scorers return 0.0. Real scorers added in step 3.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from blipshell.benchmark.models import SuiteResult, TaskScore
from blipshell.benchmark.suites.base import BenchmarkSuite

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)

# Roles in this suite and their task_name keys
ROLES = ["tool_calling", "coding"]


class InteractiveSuite(BenchmarkSuite):
    name = "interactive"
    description = "Tool calling accuracy and coding task completion (multi-turn)"
    task_types = ["tool_calling", "coding"]
    needs_db = False
    needs_router = True
    quick_samples = 0  # fixed test set
    thorough_samples = 0

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
    ) -> list[SuiteResult]:
        results = []
        for model in models:
            if on_status:
                on_status(f"[interactive] Testing {model}")
            scores = [
                TaskScore(task_name=role, quality=0.0, speed_s=0.0, samples=0)
                for role in ROLES
            ]
            results.append(SuiteResult(
                suite_name=self.name, model=model, scores=scores,
            ))
        return results
