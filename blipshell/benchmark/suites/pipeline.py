"""Pipeline suite — all memory pipeline tasks on real DB data.

Tests rank+importance+type, dedup, summarization, entity extraction,
and contradiction detection using production prompts against real memories.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable

from blipshell.benchmark.models import SuiteResult, TaskScore
from blipshell.benchmark.shared import (
    load_contradiction_pairs,
    load_dedup_pairs,
    load_diverse_memories,
    load_entity_ground_truth,
    load_stratified_sample,
)
from blipshell.benchmark.suites.base import BenchmarkSuite

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)


def _parse_triples(response: str) -> list[tuple[str, str, str]]:
    """Parse 'subject | predicate | object' lines from entity extraction."""
    response = response.strip()
    if not response or response.upper() == "NONE":
        return []
    triples = []
    for line in response.splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        subj, pred, obj = parts[0], parts[1], parts[2]
        if not subj or not pred or not obj:
            continue
        if subj.lower() in ("something", "it", "this", "that"):
            continue
        triples.append((subj, pred, obj))
    return triples


class PipelineSuite(BenchmarkSuite):
    name = "pipeline"
    description = "Memory pipeline: rank, summarize, entities, contradiction, dedup (real DB data)"
    task_types = ["ranking_importance", "summarization", "reasoning"]
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
        if not db_path:
            logger.warning("Pipeline suite requires DB path, skipping")
            return []

        n = self.thorough_samples if thorough else self.quick_samples

        # Load all test data up front
        rank_messages = load_stratified_sample(db_path, n)
        summ_messages = load_diverse_memories(db_path, max(n * 3 // 4, 15))
        entity_samples = load_entity_ground_truth(db_path, max(n * 3 // 4, 15))
        contra_pairs = load_contradiction_pairs(db_path, max(n // 2, 10))
        dedup_pairs = load_dedup_pairs(db_path, max(n // 2, 10))

        if on_status:
            on_status(
                f"[pipeline] Loaded: {len(rank_messages)} rank, "
                f"{len(summ_messages)} summ, {len(entity_samples)} entity, "
                f"{len(contra_pairs)} contradiction, {len(dedup_pairs)} dedup"
            )

        if not rank_messages:
            logger.warning("No sample messages found in DB")
            return []

        results = []
        for model in models:
            if on_status:
                on_status(f"[pipeline] Testing {model}")
            router = router_factory(model) if router_factory else None
            if not router:
                continue
            sr = await self._benchmark_model(
                model, router, rank_messages, summ_messages,
                entity_samples, contra_pairs, dedup_pairs, on_status,
            )
            results.append(sr)
        return results

    async def _benchmark_model(
        self,
        model: str,
        router: LLMRouter,
        rank_messages: list[dict],
        summ_messages: list[dict],
        entity_samples: list[dict],
        contra_pairs: list[tuple[str, str, bool]],
        dedup_pairs: list[tuple[dict, list[str]]],
        on_status: Callable | None,
    ) -> SuiteResult:
        scores = []
        total_start = time.monotonic()

        scores.append(await self._bench_rank_importance(router, rank_messages, on_status))
        scores.append(await self._bench_summarize(router, summ_messages, on_status))
        scores.append(await self._bench_entity_extraction(router, entity_samples, on_status))
        scores.append(await self._bench_contradiction(router, contra_pairs, on_status))
        scores.append(await self._bench_dedup(router, dedup_pairs, on_status))

        elapsed = time.monotonic() - total_start
        return SuiteResult(
            suite_name=self.name, model=model,
            scores=scores, elapsed_s=round(elapsed, 1),
        )

    # ------------------------------------------------------------------
    # Rank + Importance + Type
    # ------------------------------------------------------------------
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
        # Weighted quality: rank accuracy most important, then importance, then type
        if total > 0:
            quality = (
                (rank_within_1 / total) * 0.4
                + (imp_within_02 / total) * 0.3
                + (type_match / total) * 0.3
            )
        else:
            quality = 0

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

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------
    async def _bench_summarize(
        self, router: LLMRouter, messages: list[dict], on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import summarize_memory
        from blipshell.llm.router import TaskType

        times = []
        good = 0
        keyword_hits = 0
        keyword_checks = 0
        skip_appropriate = 0  # SKIP on rank<=2 (correct)
        skip_inappropriate = 0  # SKIP on rank>=4 (wrong)
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
                rank = msg.get("rank", 3)

                if text.upper() == "SKIP":
                    if rank is not None and rank <= 2:
                        skip_appropriate += 1
                        good += 1  # Correct SKIP
                    elif rank is not None and rank >= 4:
                        skip_inappropriate += 1
                    # rank 3 SKIP is ambiguous, don't count either way
                elif 5 < len(text) and len(text.split()) <= 40:
                    good += 1
                    # Keyword overlap: check new summary covers same content
                    existing = msg.get("summary", "")
                    if existing and len(existing) > 10:
                        keyword_checks += 1
                        existing_words = {
                            w.lower() for w in existing.split()
                            if len(w) > 3  # skip short words
                        }
                        new_words = {
                            w.lower() for w in text.split()
                            if len(w) > 3
                        }
                        overlap = existing_words & new_words
                        if existing_words and len(overlap) / len(existing_words) >= 0.3:
                            keyword_hits += 1
            except Exception as e:
                logger.debug("summarize error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        # Quality: format correctness weighted with keyword overlap
        format_score = good / total if total else 0
        keyword_score = keyword_hits / keyword_checks if keyword_checks else 0
        quality = format_score * 0.6 + keyword_score * 0.4

        return TaskScore(
            task_name="summarization",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=total,
            errors=errors,
            detail={
                "good": good,
                "keyword_overlap": f"{keyword_hits}/{keyword_checks}",
                "skip_appropriate": skip_appropriate,
                "skip_inappropriate": skip_inappropriate,
                "total": total,
            },
        )

    # ------------------------------------------------------------------
    # Entity Extraction
    # ------------------------------------------------------------------
    async def _bench_entity_extraction(
        self, router: LLMRouter, samples: list[dict], on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import extract_entities
        from blipshell.llm.router import TaskType

        times = []
        total_expected = 0
        total_found = 0
        errors = 0

        for sample in samples:
            summary = sample.get("summary", "")
            expected_entities = {e["name"].lower() for e in sample.get("entities", [])}
            if not summary or not expected_entities:
                continue

            total_expected += len(expected_entities)
            sys_p, user_p = extract_entities(summary)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                triples = _parse_triples(raw)
                extracted = {t[0].lower() for t in triples} | {t[2].lower() for t in triples}

                # Count recall: how many expected entities appear in extracted
                for expected in expected_entities:
                    if any(expected in ext or ext in expected for ext in extracted):
                        total_found += 1
            except Exception as e:
                logger.debug("entity_extraction error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        quality = total_found / total_expected if total_expected else 0

        return TaskScore(
            task_name="entity_extraction",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(samples),
            errors=errors,
            detail={
                "recalled": total_found,
                "total_expected": total_expected,
            },
        )

    # ------------------------------------------------------------------
    # Contradiction Detection
    # ------------------------------------------------------------------
    async def _bench_contradiction(
        self, router: LLMRouter, pairs: list[tuple[str, str, bool]],
        on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import detect_contradiction
        from blipshell.llm.router import TaskType

        times = []
        parsed_ok = 0
        cross_category_no = 0
        cross_category_total = 0
        errors = 0
        total = len(pairs)

        for content_a, content_b, same_category in pairs:
            sys_p, user_p = detect_contradiction(content_a, content_b)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                answer = raw.strip().upper()
                is_yes = answer.startswith("YES")
                is_no = answer.startswith("NO")

                if is_yes or is_no:
                    parsed_ok += 1

                if not same_category:
                    cross_category_total += 1
                    if is_no:
                        cross_category_no += 1
            except Exception as e:
                logger.debug("contradiction error: %s", e)
                errors += 1

        avg_speed = sum(times) / len(times) if times else 0
        # Quality: parse compliance + cross-category specificity
        parse_rate = parsed_ok / total if total else 0
        specificity = cross_category_no / cross_category_total if cross_category_total else 1.0
        quality = 0.5 * parse_rate + 0.5 * specificity

        return TaskScore(
            task_name="contradiction",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=total,
            errors=errors,
            detail={
                "parsed_ok": f"{parsed_ok}/{total}",
                "cross_category_no_rate": f"{cross_category_no}/{cross_category_total}",
            },
        )

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------
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
