"""Tagging suite — tag assignment precision/recall/F1 against ground truth.

Uses well-tagged memories from DB as ground truth, tests model ability
to assign the correct tags from the available tag vocabulary.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable

from blipshell.benchmark.models import SuiteResult, TaskScore
from blipshell.benchmark.suites.base import BenchmarkSuite

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)


def _parse_tag_response(text: str, valid_tags: set[str]) -> list[str]:
    """Parse LLM tag assignment response, validating against known tags."""
    tags = []
    for line in text.strip().split("\n"):
        line = line.strip().lstrip("-•*").strip()
        if not line:
            continue
        # Try exact match first
        if line.lower() in valid_tags:
            tags.append(line.lower())
            continue
        # Try first word/phrase
        for tag in valid_tags:
            if tag in line.lower():
                tags.append(tag)
                break
    return list(dict.fromkeys(tags))  # deduplicate preserving order


class TaggingSuite(BenchmarkSuite):
    name = "tagging"
    description = "Tag assignment quality: precision/recall/F1 vs ground truth"
    task_types = ["reasoning"]
    needs_db = True
    needs_router = True
    quick_samples = 10
    thorough_samples = 30

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
        if not db_path:
            logger.warning("Tagging suite requires DB path, skipping")
            return []

        n = self.thorough_samples if thorough else self.quick_samples

        # Load samples and tag vocabulary
        from blipshell.memory.sqlite_store import SQLiteStore
        store = SQLiteStore(db_path)
        await store.initialize()

        try:
            samples = await store.get_well_tagged_memory_sample(min_tags=3, limit=n)
            all_tags = await store.get_all_tag_names()
        finally:
            await store.close()

        if not samples or not all_tags:
            logger.warning("No tagged samples or tags found in DB")
            return []

        valid_tags = {t.lower() for t in all_tags}

        if on_status:
            on_status(f"[tagging] Loaded {len(samples)} samples, {len(all_tags)} tags")

        results = []
        for model in models:
            if on_status:
                on_status(f"[tagging] Testing {model}")
            router = router_factory(model) if router_factory else None
            if not router:
                continue
            sr = await self._benchmark_model(model, router, samples, all_tags, valid_tags, on_status)
            results.append(sr)
        return results

    async def _benchmark_model(
        self,
        model: str,
        router: LLMRouter,
        samples: list[dict],
        all_tags: list[str],
        valid_tags: set[str],
        on_status: Callable | None,
    ) -> SuiteResult:
        from blipshell.llm.router import TaskType

        times = []
        precisions = []
        recalls = []
        f1s = []
        errors = 0

        tag_list_str = ", ".join(sorted(all_tags)[:100])  # cap for context
        system_prompt = (
            "You are a memory classification assistant. "
            "Given a memory summary, assign 1-5 tags from the provided list. "
            "Reply with ONLY the tag names, one per line."
        )

        for sample in samples:
            summary = sample.get("summary", "")
            ground_truth = {t.lower() for t in sample.get("tags", [])}
            if not summary or not ground_truth:
                continue

            user_prompt = (
                f"Available tags: {tag_list_str}\n\n"
                f"Memory: {summary}\n\n"
                f"Assign 1-5 tags from the list above:"
            )

            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_prompt, system=system_prompt, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                predicted = set(_parse_tag_response(raw, valid_tags))
                if not predicted:
                    errors += 1
                    continue

                tp = len(predicted & ground_truth)
                precision = tp / len(predicted) if predicted else 0
                recall = tp / len(ground_truth) if ground_truth else 0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            except Exception as e:
                logger.debug("tagging error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0

        scores = [
            TaskScore(
                task_name="tag_f1",
                quality=round(avg_f1, 3),
                speed_s=round(avg_speed, 2),
                samples=len(samples),
                errors=errors,
                detail={
                    "precision": round(avg_precision, 3),
                    "recall": round(avg_recall, 3),
                    "f1": round(avg_f1, 3),
                },
            ),
        ]

        return SuiteResult(
            suite_name=self.name, model=model,
            scores=scores, elapsed_s=round(sum(times), 1),
        )
