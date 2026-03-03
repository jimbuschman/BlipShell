"""Pipeline suite — rank+importance, dedup, summarization using real DB data.

Tests the production memory pipeline calls. Most valuable suite for
deciding if a new model is better for the pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable

from blipshell.benchmark.models import SuiteResult, TaskScore
from blipshell.benchmark.shared import load_dedup_pairs, load_stratified_sample
from blipshell.benchmark.suites.base import BenchmarkSuite

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)


class PipelineSuite(BenchmarkSuite):
    name = "pipeline"
    description = "Memory pipeline: rank+importance+type, dedup, summarization (real DB data)"
    task_types = ["ranking_importance", "summarization", "reasoning"]
    needs_db = True
    needs_router = True
    quick_samples = 5
    thorough_samples = 20

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
            logger.warning("Pipeline suite requires DB path, skipping")
            return []

        n = self.thorough_samples if thorough else self.quick_samples
        messages = load_stratified_sample(db_path, n)
        if not messages:
            logger.warning("No sample messages found in DB")
            return []

        dedup_n = max(n // 2, 3)
        dedup_pairs = load_dedup_pairs(db_path, dedup_n)

        if on_status:
            on_status(f"[pipeline] Loaded {len(messages)} messages, {len(dedup_pairs)} dedup pairs")

        results = []
        for model in models:
            if on_status:
                on_status(f"[pipeline] Testing {model}")
            router = router_factory(model) if router_factory else None
            if not router:
                continue
            sr = await self._benchmark_model(model, router, messages, dedup_pairs, on_status)
            results.append(sr)
        return results

    async def _benchmark_model(
        self,
        model: str,
        router: LLMRouter,
        messages: list[dict],
        dedup_pairs: list[tuple[dict, list[str]]],
        on_status: Callable | None,
    ) -> SuiteResult:
        scores = []
        total_start = time.monotonic()

        ri_score = await self._bench_rank_importance(router, messages, on_status)
        scores.append(ri_score)

        dedup_score = await self._bench_dedup(router, dedup_pairs, on_status)
        scores.append(dedup_score)

        summ_score = await self._bench_summarize(router, messages[:10], on_status)
        scores.append(summ_score)

        elapsed = time.monotonic() - total_start
        return SuiteResult(
            suite_name=self.name, model=model,
            scores=scores, elapsed_s=round(elapsed, 1),
        )

    async def _bench_rank_importance(
        self, router: LLMRouter, messages: list[dict], on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import rank_importance_and_classify
        from blipshell.llm.router import TaskType
        from blipshell.memory.processor import MemoryProcessor

        times = []
        rank_within_1 = 0
        imp_within_02 = 0
        type_match = 0
        errors = 0
        total = len(messages)

        for msg in messages:
            content = msg.get("content", "") or msg.get("summary", "")
            if not content:
                continue
            sys_p, user_p = rank_importance_and_classify(content)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.RANKING_IMPORTANCE, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                rank, imp, mtype = MemoryProcessor._parse_rank_importance_type(raw)

                orig_rank = msg.get("rank", 0)
                if orig_rank and abs(rank - orig_rank) <= 1:
                    rank_within_1 += 1

                orig_imp = msg.get("importance")
                if orig_imp is not None and abs(imp - orig_imp) <= 0.2:
                    imp_within_02 += 1

                orig_type = msg.get("memory_type", "")
                if orig_type and mtype.lower() == orig_type.lower():
                    type_match += 1
            except Exception as e:
                logger.debug("rank_importance error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        quality = rank_within_1 / total if total else 0

        return TaskScore(
            task_name="rank_importance",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=total,
            errors=errors,
            detail={
                "rank_within_1": f"{rank_within_1}/{total}",
                "importance_within_0.2": f"{imp_within_02}/{total}",
                "type_match": f"{type_match}/{total}",
            },
        )

    async def _bench_dedup(
        self,
        router: LLMRouter,
        pairs: list[tuple[dict, list[str]]],
        on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import decide_memory_action
        from blipshell.llm.router import TaskType
        from blipshell.memory.processor import MemoryProcessor

        times = []
        actions: dict[str, int] = {}
        errors = 0
        total = len(pairs)

        for msg, existing_summaries in pairs:
            summary = msg.get("summary", "") or msg.get("content", "")
            if not summary or not existing_summaries:
                continue
            try:
                sys_p, user_p = decide_memory_action(summary, existing_summaries)
            except Exception:
                # Prompt function might not exist or have different sig
                errors += 1
                continue

            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                action, _ = MemoryProcessor._parse_memory_action(raw)
                actions[action] = actions.get(action, 0) + 1
            except Exception as e:
                logger.debug("dedup error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        # Quality: conservative dedup is better (more KEEP/ADD is safer)
        safe_actions = actions.get("ADD", 0) + actions.get("KEEP", 0)
        quality = safe_actions / total if total else 0.5

        return TaskScore(
            task_name="dedup",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=total,
            errors=errors,
            detail={"actions": actions},
        )

    async def _bench_summarize(
        self, router: LLMRouter, messages: list[dict], on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import summarize_memory
        from blipshell.llm.router import TaskType

        times = []
        good = 0
        skip_count = 0
        errors = 0
        total = len(messages)

        for msg in messages:
            content = msg.get("content", "") or msg.get("summary", "")
            if not content:
                continue
            sys_p, user_p = summarize_memory(content)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.SUMMARIZATION, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                text = raw.strip()
                if text.upper() == "SKIP":
                    skip_count += 1
                elif len(text) > 5 and len(text.split()) <= 40:
                    good += 1
            except Exception as e:
                logger.debug("summarize error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        quality = good / total if total else 0

        return TaskScore(
            task_name="summarization",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=total,
            errors=errors,
            detail={"good": good, "skips": skip_count, "total": total},
        )
