"""Extraction suite — entity extraction, entity resolution, tag discovery, batch tags.

Matches benchmark-spec.md Section 4.2 extraction grouping.
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
ROLES = ["entity_extraction", "entity_resolution", "tag_discovery", "batch_tags"]

# Valid entity types from the prompt
VALID_ENTITY_TYPES = {"person", "project", "technology", "concept", "preference", "place", "organization"}

# Curated entity extraction test cases with expected entities.
ENTITY_EXTRACTION_CURATED = [
    {
        "summary": "User asked about Python performance tuning for data analysis.",
        "expect_none": False,
        "expected_entities": ["python", "data analysis"],
    },
    {
        "summary": "User decided to use a two-module design for the desk robot with JST connectors.",
        "expect_none": False,
        "expected_entities": ["desk robot", "ESP32", "JST connectors"],
    },
    {
        "summary": "Assistant explained how to configure Ollama with GPU acceleration.",
        "expect_none": False,
        "expected_entities": ["ollama", "gpu"],
    },
    {
        "summary": "User's daughter has a Minecraft performance issue on her HP laptop.",
        "expect_none": False,
        "expected_entities": ["minecraft"],
    },
    {
        "summary": "User said hello.",
        "expect_none": True,
        "expected_entities": [],
    },
]

# Curated test data for batch tag assignment.
# Each "memory" gets sent to LLM with available_tags, and we check assigned tags
# against ground truth.
BATCH_TAGS_AVAILABLE = [
    "python", "machine-learning", "data-processing",
    "home-automation", "electronics", "gaming",
    "web-dev", "devops", "rust",
]
BATCH_TAGS_CURATED = [
    {
        "id": 1,
        "summary": "User ran cProfile on a pandas data pipeline to find bottlenecks.",
        "ground_truth": {"python", "machine-learning"},
    },
    {
        "id": 2,
        "summary": "User set up a Zigbee coordinator on a Raspberry Pi for smart lights.",
        "ground_truth": {"home-automation", "electronics"},
    },
    {
        "id": 3,
        "summary": "User's daughter complains about Minecraft lag on her laptop.",
        "ground_truth": {"gaming"},
    },
]


class ExtractionSuite(BenchmarkSuite):
    name = "extraction"
    description = "Entity extraction, entity resolution, tag discovery, batch tag assignment"
    task_types = ["reasoning"]
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
        config_path: str | None = None,
        on_model_done: Callable | None = None,
    ) -> list[SuiteResult]:
        results = []
        for model in models:
            if on_status:
                on_status(f"[extraction] Testing {model}")
            router = router_factory(model) if router_factory else None
            if not router:
                continue

            scores = []
            total_start = time.monotonic()

            scores.append(await self._score_entity_extraction(router, on_status))
            scores.append(await self._score_entity_resolution(router, on_status))
            scores.append(await self._score_tag_discovery(router, on_status))
            scores.append(await self._score_batch_tags(router, on_status))

            elapsed = time.monotonic() - total_start
            results.append(SuiteResult(
                suite_name=self.name, model=model, scores=scores,
                elapsed_s=round(elapsed, 1),
            ))
        return results

    # ------------------------------------------------------------------
    # Role 6: Entity Extraction
    # Spec: parse_ok, triple_validity, type_validity, coverage
    # Composite: average of the 4 metrics per case, averaged across cases
    # ------------------------------------------------------------------
    async def _score_entity_extraction(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import extract_entities
        from blipshell.llm.router import TaskType

        cases = list(ENTITY_EXTRACTION_CURATED)

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        for case in cases:
            summary = case["summary"]
            expect_none = case["expect_none"]
            expected_entities = [e.lower() for e in case["expected_entities"]]

            sys_p, user_p = extract_entities(summary)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("entity extraction error: %s", e)
                errors += 1
                continue

            text = raw.strip()
            is_none = text.upper() == "NONE"

            # Parse triples
            triples = []
            if not is_none:
                for line in text.split("\n"):
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) == 5 and all(parts):
                        triples.append(parts)

            checks = {}

            # 1. Parse OK: NONE for filler, >=1 triple for substantive
            if expect_none:
                checks["parse_ok"] = 1.0 if is_none else 0.0
            else:
                checks["parse_ok"] = 1.0 if len(triples) >= 1 else 0.0

            # 2. Triple validity: all triples have 5 non-empty fields
            if is_none or expect_none:
                checks["triple_validity"] = 1.0
            elif triples:
                # Already filtered to 5-field triples above
                # Check all lines were valid
                all_lines = [l.strip() for l in text.split("\n") if l.strip()]
                checks["triple_validity"] = len(triples) / len(all_lines) if all_lines else 0.0
            else:
                checks["triple_validity"] = 0.0

            # 3. Type validity: subject_type and object_type in known set
            if is_none or expect_none:
                checks["type_validity"] = 1.0
            elif triples:
                valid_count = sum(
                    1 for t in triples
                    if t[3].lower() in VALID_ENTITY_TYPES and t[4].lower() in VALID_ENTITY_TYPES
                )
                checks["type_validity"] = valid_count / len(triples)
            else:
                checks["type_validity"] = 0.0

            # 4. Coverage: expected entities found in subjects/objects
            if is_none or expect_none:
                checks["coverage"] = 1.0
            elif expected_entities:
                all_entities_text = " ".join(
                    f"{t[0]} {t[2]}" for t in triples
                ).lower()
                found = sum(1 for e in expected_entities if e in all_entities_text)
                checks["coverage"] = found / len(expected_entities)
            else:
                checks["coverage"] = 1.0

            score = sum(checks.values()) / 4
            case_scores.append(score)
            detail_rows.append({
                "summary": summary[:50],
                "output": text[:80],
                "triples": len(triples),
                "checks": {k: round(v, 2) for k, v in checks.items()},
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  entity_extraction: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="entity_extraction",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )

    # ------------------------------------------------------------------
    # Role 9: Batch Tag Assignment
    # Spec: precision, recall, F1
    # Composite: F1 score
    # ------------------------------------------------------------------
    async def _score_batch_tags(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import batch_assign_tags
        from blipshell.llm.router import TaskType

        cases = list(BATCH_TAGS_CURATED)
        summaries = [(c["id"], c["summary"]) for c in cases]
        ground_truth = {c["id"]: c["ground_truth"] for c in cases}

        sys_p, user_p = batch_assign_tags(summaries, BATCH_TAGS_AVAILABLE)
        try:
            start = time.monotonic()
            raw = await router.generate(
                TaskType.REASONING, user_p, system=sys_p, think=False,
            )
            elapsed = time.monotonic() - start
        except Exception as e:
            logger.debug("batch tags error: %s", e)
            if on_status:
                on_status(f"  batch_tags: ERROR — {e}")
            return TaskScore(
                task_name="batch_tags",
                quality=0.0, speed_s=0.0, samples=len(cases), errors=1,
            )

        # Parse response: "1: tag1, tag2\n2: tag3\n..."
        assigned: dict[int, set[str]] = {}
        for line in raw.strip().split("\n"):
            m = re.match(r"(\d+):\s*(.+)", line.strip())
            if m:
                idx = int(m.group(1))
                tags = {t.strip().lower() for t in m.group(2).split(",") if t.strip().upper() != "NONE"}
                assigned[idx] = tags

        # Calculate precision, recall, F1 per memory
        total_precision = 0.0
        total_recall = 0.0
        count = 0

        detail_rows: list[dict] = []
        for case in cases:
            cid = case["id"]
            pred = assigned.get(cid, set())
            truth = ground_truth[cid]

            tp = len(pred & truth)
            precision = tp / len(pred) if pred else 0.0
            recall = tp / len(truth) if truth else 0.0

            total_precision += precision
            total_recall += recall
            count += 1

            detail_rows.append({
                "id": cid,
                "summary": case["summary"][:50],
                "predicted": sorted(pred),
                "truth": sorted(truth),
                "precision": round(precision, 2),
                "recall": round(recall, 2),
            })

        avg_p = total_precision / count if count else 0
        avg_r = total_recall / count if count else 0
        f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0

        if on_status:
            on_status(f"  batch_tags: F1={f1:.2f} (P={avg_p:.2f} R={avg_r:.2f}), "
                      f"{len(cases)} cases")

        return TaskScore(
            task_name="batch_tags",
            quality=round(f1, 3),
            speed_s=round(elapsed, 2),
            samples=len(cases),
            errors=0,
            detail={"cases": detail_rows, "precision": round(avg_p, 3), "recall": round(avg_r, 3)},
        )

    # ------------------------------------------------------------------
    # Role 7: Entity Resolution
    # Spec: parse_ok (YES/NO), accuracy (matches expected)
    # Composite: sum(checks) / 2 per case, averaged
    # ------------------------------------------------------------------
    async def _score_entity_resolution(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import resolve_entity_duplicate
        from blipshell.llm.router import TaskType

        from tests.benchmark_test_data import ENTITY_RESOLUTION_CASES

        cases = list(ENTITY_RESOLUTION_CASES)

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        for case in cases:
            entity_a = case["entity_a"]
            entity_b = case["entity_b"]
            expected = case["expected"]  # "YES" or "NO"

            sys_p, user_p = resolve_entity_duplicate(entity_a, entity_b)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("entity resolution error: %s", e)
                errors += 1
                continue

            text = raw.strip().upper()
            first_word = text.split()[0] if text.split() else ""
            is_yes = first_word == "YES"
            is_no = first_word == "NO"

            checks = {}
            checks["parse_ok"] = 1 if (is_yes or is_no) else 0
            checks["accuracy"] = 1 if first_word == expected else 0

            score = sum(checks.values()) / 2
            case_scores.append(score)
            detail_rows.append({
                "id": case["id"],
                "pair": f"{entity_a} vs {entity_b}",
                "output": text[:30],
                "expected": expected,
                "checks": checks,
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  entity_resolution: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="entity_resolution",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )

    # ------------------------------------------------------------------
    # Role 8: Tag Discovery
    # Spec: parse_ok, regex_validity, tag_novelty
    # Composite: average of the 3 metrics per case, averaged
    # ------------------------------------------------------------------
    async def _score_tag_discovery(
        self, router, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.prompts import discover_tag_patterns
        from blipshell.llm.router import TaskType

        from tests.benchmark_test_data import TAG_DISCOVERY_CASES

        cases = list(TAG_DISCOVERY_CASES)

        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        for case in cases:
            summaries = case["summaries"]
            existing_tags = case["existing_tags"]
            expect_patterns = case["expect_patterns"]  # True or False

            sys_p, user_p = discover_tag_patterns(summaries, existing_tags)
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, user_p, system=sys_p, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("tag discovery error: %s", e)
                errors += 1
                continue

            text = raw.strip()
            is_none = text.upper() == "NONE"

            # Parse tag:pattern lines
            tag_patterns: list[tuple[str, str]] = []
            if not is_none:
                for line in text.split("\n"):
                    if ":" in line:
                        parts = line.split(":", 1)
                        tag_name = parts[0].strip().lower()
                        pattern = parts[1].strip()
                        if tag_name and pattern:
                            tag_patterns.append((tag_name, pattern))

            checks = {}

            # 1. Parse OK: patterns found when expected, NONE when expected
            if expect_patterns:
                checks["parse_ok"] = 1.0 if len(tag_patterns) >= 1 else 0.0
            else:
                checks["parse_ok"] = 1.0 if is_none else 0.0

            # 2. Regex validity: each pattern compiles
            if is_none or not tag_patterns:
                checks["regex_valid"] = 1.0 if not expect_patterns else 0.0
            else:
                valid = 0
                for _, pat in tag_patterns:
                    try:
                        re.compile(pat)
                        valid += 1
                    except re.error:
                        pass
                checks["regex_valid"] = valid / len(tag_patterns)

            # 3. Tag novelty: tags not in existing_tags
            if is_none or not tag_patterns:
                checks["novelty"] = 1.0 if not expect_patterns else 0.0
            else:
                existing_lower = {t.lower() for t in existing_tags}
                novel = sum(1 for name, _ in tag_patterns if name not in existing_lower)
                checks["novelty"] = novel / len(tag_patterns)

            score = sum(checks.values()) / 3
            case_scores.append(score)
            detail_rows.append({
                "id": case.get("id", ""),
                "output": text[:80],
                "tags_found": len(tag_patterns),
                "expect_patterns": expect_patterns,
                "checks": {k: round(v, 2) for k, v in checks.items()},
                "score": round(score, 2),
            })

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  tag_discovery: {avg_quality:.2f} quality, "
                      f"{len(cases)} cases, {errors} errors")

        return TaskScore(
            task_name="tag_discovery",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(cases),
            errors=errors,
            detail={"cases": detail_rows},
        )
