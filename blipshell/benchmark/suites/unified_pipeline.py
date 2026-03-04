"""Unified pipeline suite — summarization, scoring, dedup, contradiction, lesson extraction.

Matches benchmark-spec.md Section 4.2 pipeline grouping.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Callable

from blipshell.benchmark.models import SuiteResult, TaskScore
from blipshell.benchmark.suites.base import BenchmarkSuite

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)

# Roles in this suite and their task_name keys
ROLES = ["summarization", "scoring", "dedup", "contradiction", "lesson"]

# Curated test conversations for lesson extraction.
# expect_skip=True for filler, False for substantive conversations.
LESSON_CURATED = [
    {
        "content": (
            "User: My ESP32 keeps crashing when I plug in the JST connector.\n"
            "Assistant: Let me help debug that. What voltage is your power supply?\n"
            "User: 5V from a USB-C cable. The serial monitor shows brownout resets.\n"
            "Assistant: That's likely a current issue. Try a powered USB hub or "
            "dedicated 5V 2A supply. The JST connector might be drawing too much."
        ),
        "expect_skip": False,
    },
    {
        "content": (
            "User: Can you review my Python code? It's running really slowly.\n"
            "Assistant: Sure, paste it and I'll take a look.\n"
            "User: [pastes 50 lines of nested loops]\n"
            "Assistant: I see the issue — you have O(n^3) nested loops. "
            "Let me show you how to optimize with a dict lookup instead.\n"
            "User: Oh that's way faster, thanks! I always forget about dicts."
        ),
        "expect_skip": False,
    },
    {
        "content": "User: hey\nAssistant: Hello! How can I help?\nUser: ok thanks",
        "expect_skip": True,
    },
    {
        "content": "User: hi there\nAssistant: Hi! What are you working on today?",
        "expect_skip": True,
    },
    {
        "content": (
            "User: I've been going back and forth on whether to use React or Vue.\n"
            "Assistant: Both are solid choices. What's the project scope?\n"
            "User: Small internal dashboard, maybe 10 pages. I know React already but "
            "Vue seems simpler for this.\n"
            "Assistant: For a small dashboard where you already know React, "
            "I'd stick with React. Less new tooling to learn."
        ),
        "expect_skip": False,
    },
]

# Curated test messages for summarization (from benchmark_models.py TEST_MESSAGES).
# Each has an expected_skip flag: True if the message is filler/greeting.
# Curated test cases for scoring (rank + importance + type).
# Expected values are ground-truth labels for the curated messages.
SCORING_CURATED = [
    {"content": "hey", "expect_rank": 1, "expect_importance": 0.1, "expect_type": "conversation"},
    {"content": "ok thanks", "expect_rank": 1, "expect_importance": 0.1, "expect_type": "conversation"},
    {
        "content": (
            "[System: <important_rules>\nYou are in agent mode.\n"
            "If you need to use multiple tools...</important_rules>]\ntest"
        ),
        "expect_rank": 1, "expect_importance": 0.1, "expect_type": "conversation",
    },
    {
        "content": (
            "I'm using a MAX98357 with the esp32 and the sound it produces "
            "is terrible. Mostly sounds like just noise/garbage."
        ),
        "expect_rank": 4, "expect_importance": 0.6, "expect_type": "fact",
    },
    {
        "content": (
            "for some reason my daughters laptop when we got in dec only "
            "wants to load 10 blocks in minecraft, thats pretty terrbile"
        ),
        "expect_rank": 4, "expect_importance": 0.7, "expect_type": "fact",
    },
    {
        "content": (
            "Now I can see the worker.py file! Let me provide a code review:\n\n"
            "## Code Review: worker.py\n\n"
            "### Issues Found:\n"
            "1. Missing stop() call on shutdown\n"
            "2. No retry logic for failed HTTP requests\n"
            "3. No model fallback mechanism"
        ),
        "expect_rank": 3, "expect_importance": 0.5, "expect_type": "skill",
    },
    {
        "content": (
            "yeah i think ill go with the two-module design for the desk robot. "
            "main board plus a sidecar for the sensors. JST connectors between them."
        ),
        "expect_rank": 5, "expect_importance": 0.8, "expect_type": "preference",
    },
    {"content": "sanding paint when its still tacky?", "expect_rank": 2, "expect_importance": 0.3, "expect_type": "conversation"},
]

SUMMARIZATION_CURATED = [
    {"content": "hey", "expect_skip": True},
    {"content": "ok thanks", "expect_skip": True},
    {
        "content": (
            "[System: <important_rules>\nYou are in agent mode.\n"
            "If you need to use multiple tools...</important_rules>]\ntest"
        ),
        "expect_skip": True,
    },
    {
        "content": (
            "I'm using a MAX98357 with the esp32 and the sound it produces "
            "is terrible. Mostly sounds like just noise/garbage."
        ),
        "expect_skip": False,
    },
    {
        "content": (
            "for some reason my daughters laptop when we got in dec only "
            "wants to load 10 blocks in minecraft, thats pretty terrbile"
        ),
        "expect_skip": False,
    },
    {
        "content": (
            "Now I can see the worker.py file! Let me provide a code review:\n\n"
            "## Code Review: worker.py\n\n"
            "### Issues Found:\n"
            "1. Missing stop() call on shutdown\n"
            "2. No retry logic for failed HTTP requests\n"
            "3. No model fallback mechanism"
        ),
        "expect_skip": False,
    },
    {
        "content": (
            "yeah i think ill go with the two-module design for the desk robot. "
            "main board plus a sidecar for the sensors. JST connectors between them."
        ),
        "expect_skip": False,
    },
    {"content": "sanding paint when its still tacky?", "expect_skip": True},
]


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


class UnifiedPipelineSuite(BenchmarkSuite):
    name = "pipeline"
    description = "Summarization, scoring (rank+importance+type), dedup, contradiction, lesson extraction"
    task_types = ["summarization", "ranking_importance", "reasoning"]
    needs_db = True
    needs_router = True
    quick_samples = 20
    thorough_samples = 50

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
                on_status(f"[pipeline] Testing {model}")
            router = router_factory(model) if router_factory else None
            if not router:
                continue

            scores = []
            total_start = time.monotonic()

            scores.append(await self._score_summarization(router, on_status))
            scores.append(await self._score_scoring(router, on_status))
            scores.append(await self._score_dedup(router, on_status))
            scores.append(await self._score_contradiction(router, on_status))
            scores.append(await self._score_lesson(router, on_status))

            elapsed = time.monotonic() - total_start
            results.append(SuiteResult(
                suite_name=self.name, model=model, scores=scores,
                elapsed_s=round(elapsed, 1),
            ))
        return results

    # ------------------------------------------------------------------
    # Role 1: Summarization
    # Spec metrics: parse_ok, skip_correctness, word_count, no_echo,
    #               third_person, compression
    # Composite: sum(checks) / 6 per case, averaged
    # ------------------------------------------------------------------
    async def _score_summarization(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import summarize_memory
        from blipshell.llm.router import TaskType

        cases = list(SUMMARIZATION_CURATED)
        # TODO: when db_path available, append real DB messages via load_diverse_memories

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        for case in cases:
            content = case["content"]
            expect_skip = case["expect_skip"]

            sys_p, user_p = summarize_memory(content)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.SUMMARIZATION, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("summarization error: %s", e)
                errors += 1
                continue

            text = raw.strip()
            checks = {}

            # 1. Parse OK: non-empty response
            checks["parse_ok"] = 1 if text else 0

            # 2. SKIP correctness
            is_skip = text.upper() == "SKIP"
            if expect_skip:
                checks["skip_correct"] = 1 if is_skip else 0
            else:
                checks["skip_correct"] = 1 if not is_skip else 0

            # 3. Word count ≤30 (only for non-SKIP)
            if is_skip:
                checks["word_count"] = 1  # SKIP is always fine
            else:
                checks["word_count"] = 1 if len(text.split()) <= 30 else 0

            # 4. No echo: Jaccard similarity < 0.7
            if is_skip:
                checks["no_echo"] = 1
            else:
                checks["no_echo"] = 1 if _jaccard_similarity(text, content) < 0.7 else 0

            # 5. Third person: no "I ", "my ", "you " at word boundaries
            if is_skip:
                checks["third_person"] = 1
            else:
                has_first_second = bool(re.search(r"\b(I|my|you)\b", text, re.IGNORECASE))
                checks["third_person"] = 0 if has_first_second else 1

            # 6. Compression: output chars / input chars < 0.5 (for inputs >100 chars)
            if is_skip or len(content) <= 100:
                checks["compression"] = 1  # N/A for short inputs or SKIP
            else:
                ratio = len(text) / len(content)
                checks["compression"] = 1 if ratio < 0.5 else 0

            score = sum(checks.values()) / 6
            case_scores.append(score)
            detail_rows.append({
                "input": content[:50],
                "output": text[:60],
                "checks": checks,
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  summarization: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="summarization",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )

    # ------------------------------------------------------------------
    # Role 2: Scoring (rank + importance + type classification)
    # Spec metrics: parse_ok, rank±1, importance±0.2, type_match
    # Composite: sum(checks) / 4 per case, averaged
    # ------------------------------------------------------------------
    async def _score_scoring(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import rank_importance_and_classify
        from blipshell.llm.router import TaskType
        from blipshell.memory.processor import MemoryProcessor

        cases = list(SCORING_CURATED)

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        for case in cases:
            content = case["content"]
            expect_rank = case["expect_rank"]
            expect_importance = case["expect_importance"]
            expect_type = case["expect_type"]

            sys_p, user_p = rank_importance_and_classify(content)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.RANKING_IMPORTANCE, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("scoring error: %s", e)
                errors += 1
                continue

            text = raw.strip()
            rank, importance, mem_type = MemoryProcessor._parse_rank_importance_type(text)
            checks = {}

            # 1. Parse OK: all 3 values extracted (non-default)
            # We consider it parsed if the raw text has at least one number
            checks["parse_ok"] = 1 if re.search(r"\d", text) else 0

            # 2. Rank accuracy: |predicted - expected| <= 1
            checks["rank_ok"] = 1 if abs(rank - expect_rank) <= 1 else 0

            # 3. Importance accuracy: |predicted - expected| <= 0.2
            checks["importance_ok"] = 1 if abs(importance - expect_importance) <= 0.2 else 0

            # 4. Type accuracy: exact match
            checks["type_ok"] = 1 if mem_type == expect_type else 0

            score = sum(checks.values()) / 4
            case_scores.append(score)
            detail_rows.append({
                "input": content[:50],
                "output": text[:60],
                "parsed": f"r={rank} i={importance} t={mem_type}",
                "expected": f"r={expect_rank} i={expect_importance} t={expect_type}",
                "checks": checks,
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  scoring: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="scoring",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )

    # ------------------------------------------------------------------
    # Role 3: Deduplication
    # Spec metrics: parse_ok, decision_correct
    # Composite: sum(checks) / 2 per case, averaged
    # ------------------------------------------------------------------
    async def _score_dedup(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import decide_memory_action
        from blipshell.llm.router import TaskType
        from blipshell.memory.processor import MemoryProcessor

        from tests.benchmark_test_data import DEDUP_CASES

        cases = list(DEDUP_CASES)

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        for case in cases:
            new_mem = case["new_memory"]
            existing = case["existing_memories"]
            expected_action = case["expected_action"]

            sys_p, user_p = decide_memory_action(new_mem, existing)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("dedup error: %s", e)
                errors += 1
                continue

            text = raw.strip()
            action, target = MemoryProcessor._parse_memory_action(text)
            checks = {}

            # 1. Parse OK: valid action extracted
            checks["parse_ok"] = 1 if action in ("ADD", "NONE", "UPDATE", "DELETE") else 0

            # 2. Decision correct: action matches expected
            checks["decision_ok"] = 1 if action == expected_action else 0

            score = sum(checks.values()) / 2
            case_scores.append(score)
            detail_rows.append({
                "id": case["id"],
                "new_memory": new_mem[:50],
                "output": text[:60],
                "parsed_action": action,
                "expected_action": expected_action,
                "checks": checks,
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  dedup: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="dedup",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )

    # ------------------------------------------------------------------
    # Role 4: Contradiction Detection
    # Spec metrics: parse_ok, accuracy
    # Composite: sum(checks) / 2 per case, averaged
    # ------------------------------------------------------------------
    async def _score_contradiction(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import detect_contradiction
        from blipshell.llm.router import TaskType
        from tests.benchmark_models import CONTRADICTION_PAIRS

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        for new_mem, existing_mem, expected_yes in CONTRADICTION_PAIRS:
            sys_p, user_p = detect_contradiction(new_mem, existing_mem)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("contradiction error: %s", e)
                errors += 1
                continue

            text = raw.strip().upper()
            # Parse: first word should be YES or NO
            first_word = text.split()[0] if text.split() else ""
            is_yes = first_word == "YES"
            is_no = first_word == "NO"
            is_valid = is_yes or is_no

            checks = {}

            # 1. Parse OK: response starts with YES or NO
            checks["parse_ok"] = 1 if is_valid else 0

            # 2. Accuracy: matches expected
            if expected_yes:
                checks["accuracy"] = 1 if is_yes else 0
            else:
                checks["accuracy"] = 1 if is_no else 0

            score = sum(checks.values()) / 2
            case_scores.append(score)
            detail_rows.append({
                "new": new_mem[:40],
                "existing": existing_mem[:40],
                "output": text[:30],
                "expected": "YES" if expected_yes else "NO",
                "checks": checks,
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  contradiction: {avg_quality:.2f} quality, "
                      f"{len(CONTRADICTION_PAIRS)} cases, {errors} errors")

        return TaskScore(
            task_name="contradiction",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(CONTRADICTION_PAIRS),
            errors=errors,
            detail={"cases": detail_rows},
        )

    # ------------------------------------------------------------------
    # Role 5: Lesson Extraction
    # Spec metrics: skip_correctness, non_empty (>20 chars), actionable (verb)
    # Composite: sum(checks) / 3 per case, averaged
    # ------------------------------------------------------------------
    async def _score_lesson(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import extract_lesson
        from blipshell.llm.router import TaskType

        cases = list(LESSON_CURATED)

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        _actionable_re = re.compile(
            r"\b(use|avoid|prefer|always|never|try|consider|remember|"
            r"acknowledge|clarify|suggest|recommend|proactively)\b",
            re.IGNORECASE,
        )

        for case in cases:
            content = case["content"]
            expect_skip = case["expect_skip"]

            sys_p, user_p = extract_lesson(content)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("lesson error: %s", e)
                errors += 1
                continue

            text = raw.strip()
            is_skip = text.upper() == "SKIP"
            checks = {}

            # 1. SKIP correctness
            if expect_skip:
                checks["skip_correct"] = 1 if is_skip else 0
            else:
                checks["skip_correct"] = 1 if not is_skip else 0

            # 2. Non-empty: for non-SKIP, response > 20 chars
            if is_skip:
                checks["non_empty"] = 1  # SKIP is fine
            else:
                checks["non_empty"] = 1 if len(text) > 20 else 0

            # 3. Actionable: contains a verb (heuristic)
            if is_skip:
                checks["actionable"] = 1  # SKIP is fine
            else:
                checks["actionable"] = 1 if _actionable_re.search(text) else 0

            score = sum(checks.values()) / 3
            case_scores.append(score)
            detail_rows.append({
                "input": content[:50],
                "output": text[:60],
                "checks": checks,
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  lesson: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="lesson",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )
