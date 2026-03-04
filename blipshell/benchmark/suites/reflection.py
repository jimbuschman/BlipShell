"""Reflection suite — session reflection quality benchmark.

Tests how well each model produces structured reflections from real session data.
Measures: parse success, section completeness, specificity, and timing.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable

from blipshell.benchmark.models import SuiteResult, TaskScore
from blipshell.benchmark.shared import get_context_tokens
from blipshell.benchmark.suites.base import BenchmarkSuite

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)


def _score_reflection(raw: str, parsed: dict) -> dict:
    """Score a reflection for quality metrics."""
    sections = [
        "effectiveness", "what_worked", "what_didnt_work",
        "technical_insights", "process_insights",
    ]
    filled = sum(1 for s in sections if parsed.get(s))

    valid_effectiveness = parsed.get("effectiveness") in (
        "effective", "partially_effective", "ineffective", "unclear",
    )

    bullet_count = 0
    specific_count = 0
    for section in ["what_worked", "what_didnt_work", "technical_insights", "process_insights"]:
        text = parsed.get(section) or ""
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("•"):
                bullet_count += 1
                if any(c in line for c in [".", "(", ":", "=", "/", "`", '"']):
                    specific_count += 1

    return {
        "sections_filled": filled,
        "sections_total": len(sections),
        "valid_effectiveness": valid_effectiveness,
        "bullet_count": bullet_count,
        "specific_count": specific_count,
        "is_skip": raw.strip().upper() == "SKIP",
    }


class ReflectionSuite(BenchmarkSuite):
    name = "reflection"
    description = "Session reflection quality: completeness, specificity, effectiveness"
    task_types = ["reasoning"]
    needs_db = True
    needs_router = True
    quick_samples = 5
    thorough_samples = 12

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
            logger.warning("Reflection suite requires DB path, skipping")
            return []

        n = self.thorough_samples if thorough else self.quick_samples

        # Load sessions
        from blipshell.memory.sqlite_store import SQLiteStore
        store = SQLiteStore(db_path)
        await store.initialize()

        try:
            sessions = await self._get_sample_sessions(store, n)
        finally:
            await store.close()

        if not sessions:
            logger.warning("No sessions with summaries found in DB")
            return []

        if on_status:
            on_status(f"[reflection] Loaded {len(sessions)} sessions")

        results = []
        for model in models:
            if on_status:
                on_status(f"[reflection] Testing {model}")
            router = router_factory(model) if router_factory else None
            if not router:
                continue

            # Re-open store for each model (sessions need message loading)
            store = SQLiteStore(db_path)
            await store.initialize()
            try:
                sr = await self._benchmark_model(model, router, sessions, store, on_status)
            finally:
                await store.close()
            results.append(sr)
        return results

    async def _get_sample_sessions(self, store, n: int) -> list[dict]:
        """Get a mix of sessions for benchmarking."""
        cursor = await store._db.execute("""
            SELECT s.id, s.summary, s.project, s.title,
                   COALESCE(
                       (SELECT COUNT(*) FROM memories m
                        WHERE m.session_id = s.id AND m.is_archived = 0),
                       0
                   ) as msg_count
            FROM sessions s
            WHERE s.summary IS NOT NULL AND s.summary != ''
              AND s.is_archived = 0
            ORDER BY RANDOM()
            LIMIT ?
        """, (n * 3,))
        rows = [dict(r) for r in await cursor.fetchall()]

        if not rows:
            return []

        short = [r for r in rows if r["msg_count"] <= 10]
        medium = [r for r in rows if 10 < r["msg_count"] <= 30]
        long = [r for r in rows if r["msg_count"] > 30]

        per = max(1, n // 3)
        selected = short[:per] + medium[:per] + long[:per]

        remaining = n - len(selected)
        if remaining > 0:
            used = {s["id"] for s in selected}
            extras = [r for r in rows if r["id"] not in used]
            selected.extend(extras[:remaining])

        return selected[:n]

    async def _prepare_session_text(self, store, session_id: int) -> str:
        """Get conversation text for a session."""
        messages = await store.get_session_messages_for_lesson(session_id)
        if not messages:
            memories = await store.get_memories_by_session(session_id)
            if not memories:
                return ""
            return "\n".join(f"{m.role}: {m.content}" for m in memories)
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    async def _benchmark_model(
        self, model: str, router: LLMRouter, sessions: list[dict],
        store, on_status: Callable | None,
    ) -> SuiteResult:
        from blipshell.llm.prompts import reflect_on_session
        from blipshell.llm.router import TaskType
        from blipshell.memory.processor import MemoryProcessor

        ctx = get_context_tokens(model) or 32768
        max_input = ctx - 4096

        times = []
        quality_scores = []
        errors = 0
        skipped = 0

        for session in sessions:
            sid = session["id"]
            summary = session["summary"]
            conversation = await self._prepare_session_text(store, sid)
            if not conversation:
                skipped += 1
                continue

            # Estimate and truncate if needed
            from blipshell.memory.manager import estimate_tokens
            conv_tokens = estimate_tokens(conversation)
            if conv_tokens > max_input:
                target_chars = max_input * 4
                conversation = conversation[:target_chars] + "\n\n[Truncated for benchmark]"

            sys_p, user_p = reflect_on_session(summary, conversation, session.get("project"))

            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                parsed = MemoryProcessor._parse_reflection(raw)
                scores = _score_reflection(raw, parsed)

                if scores["is_skip"]:
                    skipped += 1
                else:
                    # Quality = (sections_filled/5 * 0.4) + (valid_effectiveness * 0.3) + (specificity * 0.3)
                    section_q = scores["sections_filled"] / 5
                    eff_q = 1.0 if scores["valid_effectiveness"] else 0.0
                    spec_q = min(scores["specific_count"] / max(scores["bullet_count"], 1), 1.0)
                    quality_scores.append(section_q * 0.4 + eff_q * 0.3 + spec_q * 0.3)
            except Exception as e:
                logger.debug("reflection error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        scores = [
            TaskScore(
                task_name="reflection_quality",
                quality=round(avg_quality, 3),
                speed_s=round(avg_speed, 2),
                samples=len(sessions),
                errors=errors,
                detail={
                    "completed": len(quality_scores),
                    "skipped": skipped,
                },
            ),
        ]

        return SuiteResult(
            suite_name=self.name, model=model,
            scores=scores, elapsed_s=round(sum(times), 1),
        )
