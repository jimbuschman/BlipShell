"""Prompts suite — hardcoded tests for entity extraction, contradiction, ranking.

No DB required. Uses curated test data to measure prompt quality.
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

# ---------------------------------------------------------------------------
# Hardcoded test data
# ---------------------------------------------------------------------------

TEST_MESSAGES = [
    # Greeting/filler — should be rank 1, low importance, SKIP summary
    {"role": "user", "content": "hey", "expect_rank": 1, "expect_imp": 0.1},
    # Short filler
    {"role": "user", "content": "ok thanks", "expect_rank": 1, "expect_imp": 0.1},
    # System noise — should be rank 1, SKIP
    {
        "role": "user",
        "content": (
            "[System: <important_rules>\nYou are in agent mode.\n"
            "If you need to use multiple tools...</important_rules>]\ntest"
        ),
        "expect_rank": 1,
        "expect_imp": 0.1,
    },
    # Technical question — should be rank 3-4, medium importance
    {
        "role": "user",
        "content": (
            "I'm using a MAX98357 with the esp32 and the sound it produces "
            "is terrible. Mostly sounds like just noise/garbage."
        ),
        "expect_rank": 3,
        "expect_imp": 0.5,
    },
    # Personal fact — should be rank 4-5, high importance
    {
        "role": "user",
        "content": (
            "for some reason my daughters laptop when we got in dec only "
            "wants to load 10 blocks in minecraft, thats pretty terrbile"
        ),
        "expect_rank": 4,
        "expect_imp": 0.7,
    },
    # Code review (assistant) — should be rank 3-4
    {
        "role": "assistant",
        "content": (
            "Now I can see the worker.py file! Let me provide a code review:\n\n"
            "## Code Review: worker.py\n\n"
            "### Issues Found:\n"
            "1. Missing stop() call on shutdown\n"
            "2. No retry logic for failed HTTP requests\n"
            "3. No model fallback mechanism"
        ),
        "expect_rank": 3,
        "expect_imp": 0.5,
    },
    # Decision/preference — should be rank 4-5, high importance
    {
        "role": "user",
        "content": (
            "yeah i think ill go with the two-module design for the desk robot. "
            "main board plus a sidecar for the sensors. JST connectors between them."
        ),
        "expect_rank": 4,
        "expect_imp": 0.8,
    },
    # Casual/minor — should be rank 2, low importance
    {"role": "user", "content": "sanding paint when its still tacky?",
     "expect_rank": 2, "expect_imp": 0.3},
]

ENTITY_TEST_SUMMARIES = [
    ("User asked about Python performance tuning for data analysis.",
     ["Python"]),
    ("User decided to use a two-module design for the desk robot with JST connectors.",
     ["desk robot", "JST"]),
    ("Assistant explained how to configure Ollama with GPU acceleration.",
     ["Ollama"]),
    ("User's daughter has a Minecraft performance issue on her HP laptop.",
     ["Minecraft", "HP"]),
    ("User said hello.", []),  # expect NONE
]

CONTRADICTION_PAIRS = [
    ("User prefers dark mode", "User prefers light mode", True),
    ("User uses Windows 10", "User upgraded to Windows 11", True),
    ("User likes Python", "User dislikes Python", True),
    ("User likes coffee", "User likes tea", False),
    ("User has a cat named Luna", "User works at Acme", False),
    ("User knows Python", "User also knows Rust", False),
]


class PromptsSuite(BenchmarkSuite):
    name = "prompts"
    description = "Hardcoded prompt quality: ranking, entity extraction, contradiction"
    task_types = ["ranking_importance", "reasoning", "summarization"]
    needs_db = False
    needs_router = True
    quick_samples = 0  # fixed data
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
                on_status(f"[prompts] Testing {model}")
            router = router_factory(model) if router_factory else None
            if not router:
                logger.warning("No router for %s, skipping", model)
                continue
            sr = await self._benchmark_model(model, router, on_status)
            results.append(sr)
        return results

    async def _benchmark_model(
        self, model: str, router: LLMRouter, on_status: Callable | None,
    ) -> SuiteResult:
        from blipshell.llm.router import TaskType

        scores = []
        total_start = time.monotonic()

        # 1. Rank + Importance
        ri_score = await self._bench_rank_importance(router, on_status)
        scores.append(ri_score)

        # 2. Entity extraction
        ent_score = await self._bench_entity_extraction(router, on_status)
        scores.append(ent_score)

        # 3. Contradiction detection
        contra_score = await self._bench_contradiction(router, on_status)
        scores.append(contra_score)

        # 4. Summarization
        summ_score = await self._bench_summarization(router, on_status)
        scores.append(summ_score)

        elapsed = time.monotonic() - total_start
        return SuiteResult(
            suite_name=self.name, model=model,
            scores=scores, elapsed_s=round(elapsed, 1),
        )

    async def _bench_rank_importance(
        self, router: LLMRouter, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import rank_importance_and_classify
        from blipshell.llm.router import TaskType
        from blipshell.memory.processor import MemoryProcessor

        times = []
        rank_close = 0  # within +/-1 of expected
        imp_close = 0   # within 0.2 of expected
        errors = 0
        total = len(TEST_MESSAGES)

        for msg in TEST_MESSAGES:
            sys_p, user_p = rank_importance_and_classify(msg["content"])
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.RANKING_IMPORTANCE, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                rank, imp, _ = MemoryProcessor._parse_rank_importance_type(raw)
                expected_rank = msg.get("expect_rank", 3)
                expected_imp = msg.get("expect_imp", 0.5)

                if abs(rank - expected_rank) <= 1:
                    rank_close += 1
                if abs(imp - expected_imp) <= 0.2:
                    imp_close += 1
            except Exception as e:
                logger.debug("rank_importance error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        quality = (rank_close + imp_close) / (total * 2) if total else 0

        return TaskScore(
            task_name="rank_importance",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=total,
            errors=errors,
            detail={
                "rank_within_1": rank_close,
                "importance_within_0.2": imp_close,
                "total": total,
            },
        )

    async def _bench_entity_extraction(
        self, router: LLMRouter, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import extract_entities
        from blipshell.llm.router import TaskType
        from blipshell.memory.processor import MemoryProcessor

        times = []
        hit_count = 0
        total_expected = 0
        errors = 0

        for summary, expected_entities in ENTITY_TEST_SUMMARIES:
            sys_p, user_p = extract_entities(summary)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                triples = MemoryProcessor._parse_triples(raw)
                found_entities = {t[0].lower() for t in triples} | {t[2].lower() for t in triples}

                if not expected_entities:
                    # Expect NONE — success if no triples found
                    if len(triples) == 0:
                        hit_count += 1
                    total_expected += 1
                else:
                    for exp in expected_entities:
                        total_expected += 1
                        if any(exp.lower() in ent for ent in found_entities):
                            hit_count += 1
            except Exception as e:
                logger.debug("entity extraction error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        quality = hit_count / total_expected if total_expected else 0

        return TaskScore(
            task_name="entity_extraction",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(ENTITY_TEST_SUMMARIES),
            errors=errors,
            detail={"hits": hit_count, "total_expected": total_expected},
        )

    async def _bench_contradiction(
        self, router: LLMRouter, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import detect_contradiction
        from blipshell.llm.router import TaskType

        times = []
        correct = 0
        errors = 0
        total = len(CONTRADICTION_PAIRS)

        for mem_a, mem_b, expected_yes in CONTRADICTION_PAIRS:
            sys_p, user_p = detect_contradiction(mem_a, mem_b)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                answer = raw.strip().split()[0].upper() if raw.strip() else ""
                is_yes = answer.startswith("YES")
                if is_yes == expected_yes:
                    correct += 1
            except Exception as e:
                logger.debug("contradiction error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        quality = correct / total if total else 0

        return TaskScore(
            task_name="contradiction",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=total,
            errors=errors,
            detail={"correct": correct, "total": total},
        )

    async def _bench_summarization(
        self, router: LLMRouter, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import summarize_memory
        from blipshell.llm.router import TaskType

        times = []
        good_summaries = 0
        skip_count = 0
        errors = 0
        total = len(TEST_MESSAGES)

        for msg in TEST_MESSAGES:
            sys_p, user_p = summarize_memory(msg["content"])
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
                    # SKIP is correct for filler/noise
                    if msg.get("expect_rank", 3) <= 1:
                        good_summaries += 1
                elif len(text) > 5:
                    # Non-SKIP with actual content
                    word_count = len(text.split())
                    if word_count <= 40:
                        good_summaries += 1
            except Exception as e:
                logger.debug("summarization error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        quality = good_summaries / total if total else 0

        return TaskScore(
            task_name="summarization",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=total,
            errors=errors,
            detail={"good": good_summaries, "skips": skip_count, "total": total},
        )
