"""Synthesis suite — session reflection, titling, project digest, self-reflection, plan gen.

Matches benchmark-spec.md Section 4.2 synthesis grouping.
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
ROLES = ["reflection", "titling", "digest", "self_reflection", "plan_gen"]

# The 5 expected sections in session reflection output
REFLECTION_SECTIONS = ["EFFECTIVENESS", "WHAT_WORKED", "WHAT_DIDNT_WORK",
                       "TECHNICAL_INSIGHTS", "PROCESS_INSIGHTS"]
VALID_EFFECTIVENESS = {"effective", "partially_effective", "ineffective", "unclear"}

# Curated session reflection test cases
REFLECTION_CURATED = [
    {
        "session_summary": "Debugged asyncio.gather error in BlipShell memory worker",
        "conversation_text": (
            "User: The memory worker crashes with 'gather() got unexpected keyword'\n"
            "Assistant: That's a known issue — return_exceptions needs to be True.\n"
            "User: Fixed it! Also, the worker wasn't catching timeouts properly.\n"
            "Assistant: Add a try/except around each task with asyncio.wait_for()\n"
            "User: Works now, deployed and running stable."
        ),
        "expect_skip": False,
    },
    {
        "session_summary": "Migrated SQLite database schema and rebuilt indexes",
        "conversation_text": (
            "User: I need to add a 'tags' column to the memories table without losing data.\n"
            "Assistant: Use ALTER TABLE to add the column, then backfill from the tags table.\n"
            "User: The backfill query is slow on 30K rows.\n"
            "Assistant: Add a WHERE clause to skip rows that already have tags. "
            "Also run VACUUM after to reclaim space from the old schema.\n"
            "User: Done — migration took 4 seconds. Much better than the 90s full rebuild."
        ),
        "expect_skip": False,
    },
    {
        "session_summary": "Configured Nginx reverse proxy with SSL termination",
        "conversation_text": (
            "User: My Node app is on port 3000 but I need HTTPS on port 443.\n"
            "Assistant: Set up Nginx as a reverse proxy with Let's Encrypt certs.\n"
            "User: Getting 502 Bad Gateway after the proxy config.\n"
            "Assistant: Check that proxy_pass points to http://localhost:3000 "
            "and the Node app is actually running. Also add proxy_set_header Host $host.\n"
            "User: That was it — the Host header was missing. Works now with SSL."
        ),
        "expect_skip": False,
    },
    {
        "session_summary": "Quick hello, no real work done",
        "conversation_text": "User: hey\nAssistant: Hello! How can I help?\nUser: nothing, bye",
        "expect_skip": True,
    },
    {
        "session_summary": "User asked one factual question",
        "conversation_text": (
            "User: What port does Redis use by default?\n"
            "Assistant: Redis uses port 6379 by default."
        ),
        "expect_skip": True,
    },
]


class SynthesisSuite(BenchmarkSuite):
    name = "synthesis"
    description = "Session reflection, titling, project digest, self-reflection, plan generation"
    task_types = ["reasoning", "summarization"]
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
        results = []
        for model in models:
            if on_status:
                on_status(f"[synthesis] Testing {model}")
            router = router_factory(model) if router_factory else None
            if not router:
                continue

            scores = []
            total_start = time.monotonic()

            scores.append(await self._score_reflection(router, on_status))
            scores.append(await self._score_titling(router, on_status))
            scores.append(await self._score_digest(router, on_status))
            scores.append(await self._score_self_reflection(router, on_status))
            scores.append(await self._score_plan_gen(router, on_status))

            elapsed = time.monotonic() - total_start
            results.append(SuiteResult(
                suite_name=self.name, model=model, scores=scores,
                elapsed_s=round(elapsed, 1),
            ))
        return results

    # ------------------------------------------------------------------
    # Role 10: Session Reflection
    # Spec: sections_filled/5, valid_effectiveness, specificity
    # Composite: (sections + validity + specificity) / 3
    # ------------------------------------------------------------------
    async def _score_reflection(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import reflect_on_session
        from blipshell.llm.router import TaskType

        cases = list(REFLECTION_CURATED)

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        # Concrete detail markers: '.', '(', ':', '=', '/', '`'
        _detail_re = re.compile(r"[.(:=`/]")

        for case in cases:
            summary = case["session_summary"]
            conversation = case["conversation_text"]
            expect_skip = case["expect_skip"]

            sys_p, user_p = reflect_on_session(summary, conversation)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("reflection error: %s", e)
                errors += 1
                continue

            text = raw.strip()
            is_skip = text.upper() == "SKIP"

            if expect_skip:
                # For trivial sessions, SKIP is correct
                score = 1.0 if is_skip else 0.0
                case_scores.append(score)
                detail_rows.append({
                    "summary": summary[:50],
                    "output": text[:60],
                    "checks": {"skip_correct": score},
                    "score": round(score, 2),
                })
                continue

            if is_skip:
                # Shouldn't have skipped a substantive session
                case_scores.append(0.0)
                detail_rows.append({
                    "summary": summary[:50],
                    "output": "SKIP",
                    "checks": {"skip_correct": 0.0},
                    "score": 0.0,
                })
                continue

            checks = {}

            # 1. Sections filled: count of 5 sections present
            sections_found = sum(1 for s in REFLECTION_SECTIONS if s + ":" in text or s + "\n" in text)
            checks["sections"] = sections_found / 5

            # 2. Valid effectiveness
            effectiveness_match = re.search(r"EFFECTIVENESS:\s*(\w+)", text, re.IGNORECASE)
            if effectiveness_match:
                checks["effectiveness"] = 1.0 if effectiveness_match.group(1).lower() in VALID_EFFECTIVENESS else 0.0
            else:
                checks["effectiveness"] = 0.0

            # 3. Specificity: proportion of bullets with concrete detail markers
            bullets = [line.strip() for line in text.split("\n") if line.strip().startswith("-")]
            if bullets:
                specific = sum(1 for b in bullets if _detail_re.search(b))
                checks["specificity"] = specific / len(bullets)
            else:
                checks["specificity"] = 0.0

            score = sum(checks.values()) / 3
            case_scores.append(score)
            detail_rows.append({
                "summary": summary[:50],
                "output": text[:80],
                "sections_found": sections_found,
                "checks": {k: round(v, 2) for k, v in checks.items()},
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  reflection: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="reflection",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )

    # ------------------------------------------------------------------
    # Role 11: Session Titling
    # Spec: length_ok (<=10 words), no_filler, relevance (keyword)
    # Composite: sum(checks) / 3 per case, averaged
    # ------------------------------------------------------------------
    async def _score_titling(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import generate_session_title
        from blipshell.llm.router import TaskType

        from tests.benchmark_test_data import SESSION_TITLING_CASES

        cases = list(SESSION_TITLING_CASES)

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        for case in cases:
            summary = case["session_summary"]
            expected_keywords = [k.lower() for k in case["expected_keywords"]]

            prompt = generate_session_title(summary)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.SUMMARIZATION, prompt, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("titling error: %s", e)
                errors += 1
                continue

            title = raw.strip().strip('"').strip("'")
            checks = {}

            # 1. Length OK: <=10 words
            checks["length_ok"] = 1 if len(title.split()) <= 10 else 0

            # 2. No filler: doesn't start with "A ", "The ", "Session about"
            lower = title.lower()
            has_filler = lower.startswith("a ") or lower.startswith("the ") or lower.startswith("session about")
            checks["no_filler"] = 0 if has_filler else 1

            # 3. Relevance: contains >=1 expected keyword
            title_lower = title.lower()
            has_keyword = any(kw in title_lower for kw in expected_keywords)
            checks["relevance"] = 1 if has_keyword else 0

            score = sum(checks.values()) / 3
            case_scores.append(score)
            detail_rows.append({
                "id": case["id"],
                "title": title,
                "checks": checks,
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  titling: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="titling",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )

    # ------------------------------------------------------------------
    # Role 12: Project Digest
    # Spec: sections_present (/5), length_ok (300-500 words), non_generic
    # Composite: (sections + length + specificity) / 3
    # ------------------------------------------------------------------
    async def _score_digest(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import generate_initial_digest
        from blipshell.llm.router import TaskType

        from tests.benchmark_test_data import PROJECT_DIGEST_CASES

        cases = list(PROJECT_DIGEST_CASES)
        expected_sections = ["Overview", "Current Status", "Key Decisions",
                             "Recent Activity", "Open Issues"]

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        for case in cases:
            project = case["project_name"]
            sessions = case["session_summaries"]
            details = [d.lower() for d in case["expected_details"]]

            sys_p, user_p = generate_initial_digest(project, sessions)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("digest error: %s", e)
                errors += 1
                continue

            text = raw.strip()
            checks = {}

            # 1. Sections present
            text_lower = text.lower()
            found = sum(1 for s in expected_sections if s.lower() in text_lower or f"**{s.lower()}" in text_lower)
            checks["sections"] = found / len(expected_sections)

            # 2. Length OK: 300-500 words (lenient: 100-800 for mock)
            word_count = len(text.split())
            checks["length_ok"] = 1.0 if 100 <= word_count <= 800 else 0.0

            # 3. Non-generic: mentions project name + >=2 specific details
            detail_found = sum(1 for d in details if d in text_lower)
            checks["specificity"] = 1.0 if detail_found >= 2 else detail_found / 2

            score = sum(checks.values()) / 3
            case_scores.append(score)
            detail_rows.append({
                "id": case["id"],
                "project": project,
                "word_count": word_count,
                "checks": {k: round(v, 2) for k, v in checks.items()},
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  digest: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="digest",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )

    # ------------------------------------------------------------------
    # Role 13: Self-Reflection
    # Spec: correctness (NO_CHANGES when good, IMPROVED when flawed)
    # Composite: 1.0 if correct, 0.0 if not
    # ------------------------------------------------------------------
    async def _score_self_reflection(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import reflect_on_response
        from blipshell.llm.router import TaskType

        from tests.benchmark_test_data import SELF_REFLECTION_CASES

        cases = list(SELF_REFLECTION_CASES)

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        for case in cases:
            user_msg = case["user_message"]
            response = case["response"]
            expected = case["expected"]  # "NO_CHANGES" or "IMPROVED"

            prompt = reflect_on_response(user_msg, response)
            try:
                start = time.monotonic()
                # Self-reflection uses no specific task type — uses default
                raw = await router.generate(
                    TaskType.REASONING, prompt, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("self-reflection error: %s", e)
                errors += 1
                continue

            text = raw.strip()
            is_no_changes = text.upper() == "NO_CHANGES" or text.upper() == "NO_CHANGES."

            if expected == "NO_CHANGES":
                score = 1.0 if is_no_changes else 0.0
            else:  # IMPROVED
                # Should NOT be NO_CHANGES, and should be >20 chars
                score = 1.0 if (not is_no_changes and len(text) > 20) else 0.0

            case_scores.append(score)
            detail_rows.append({
                "id": case["id"],
                "expected": expected,
                "output": text[:80],
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  self_reflection: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="self_reflection",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )

    # ------------------------------------------------------------------
    # Role 14: Plan Generation
    # Spec: parse_ok, step_count, tool_hints, relevance
    # Composite: (parse + count + hints + relevance) / 4
    # ------------------------------------------------------------------
    async def _score_plan_gen(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import generate_plan
        from blipshell.llm.router import TaskType

        from tests.benchmark_test_data import PLAN_GENERATION_CASES

        cases = list(PLAN_GENERATION_CASES)

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        _step_re = re.compile(r"^\d+\.\s+(.+)", re.MULTILINE)
        _tool_re = re.compile(r"\((\w+)\)")

        for case in cases:
            request = case["user_request"]
            context = case.get("context", "")
            expected_range = case["expected_step_range"]
            expected_hints = case["expected_tool_hints"]
            expected_keywords = [k.lower() for k in case["expected_keywords"]]

            prompt = generate_plan(request, context)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, prompt, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("plan gen error: %s", e)
                errors += 1
                continue

            text = raw.strip()

            # Parse steps
            steps = _step_re.findall(text)
            checks = {}

            # 1. Parse OK: >= 1 step extracted
            checks["parse_ok"] = 1.0 if len(steps) >= 1 else 0.0

            # 2. Step count: within expected range
            min_s, max_s = expected_range
            checks["step_count"] = 1.0 if min_s <= len(steps) <= max_s else 0.0

            # 3. Tool hints: proportion of steps with valid tool_hint
            if steps:
                with_hints = sum(1 for s in steps if _tool_re.search(s))
                checks["tool_hints"] = with_hints / len(steps)
            else:
                checks["tool_hints"] = 0.0

            # 4. Relevance: steps reference the task (contains expected keyword)
            all_steps_text = " ".join(steps).lower()
            has_keyword = any(kw in all_steps_text for kw in expected_keywords)
            checks["relevance"] = 1.0 if has_keyword else 0.0

            score = sum(checks.values()) / 4
            case_scores.append(score)
            detail_rows.append({
                "id": case["id"],
                "output": text[:80],
                "steps": len(steps),
                "checks": {k: round(v, 2) for k, v in checks.items()},
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  plan_gen: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="plan_gen",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )
