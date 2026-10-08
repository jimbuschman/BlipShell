"""Versioned contract for comparable BlipShell model benchmark runs.

The benchmark used to infer "complete" from whichever stored model happened
to have the widest (often historical) result column.  That made missing suites
invisible.  This module is the single explicit contract shared by the runner,
result files, reports, and external-review packets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path


BENCHMARK_VERSION = "2.0"
DATASET_VERSION = "2026-09-23"
SCORER_VERSION = "2"
REVIEW_SCHEMA_VERSION = 1

# Production-relevant scoring categories.  Experimental structured dedup and
# the pre-2026-08-03 ambiguous ``coding`` row are deliberately excluded.
REQUIRED_CATEGORIES = (
    "rank_importance",
    "ranking",
    "importance",
    "contradiction",
    "dedup",
    "entity",
    "summarization",
    "lessons",
    "reasoning",
    "code_gen",
    "coding_agentic",
    "tool_calling",
    "session_review",
    "session_review_chunked",
    "embedding",
)

OPEN_ENDED_CATEGORIES = frozenset({
    "summarization", "lessons", "reasoning", "code_gen", "session_review",
})


@dataclass(frozen=True)
class RunTier:
    jobs: frozenset[str]
    repeats: int
    description: str


RUN_TIERS = {
    "smoke": RunTier(
        jobs=frozenset({"dedup", "reasoning"}),
        repeats=1,
        description="Fast plumbing/capability check; not sufficient for routing decisions.",
    ),
    "compare": RunTier(
        jobs=frozenset({"pipeline", "reasoning", "session_review"}),
        repeats=3,
        description="Comparable routing-job run without the slow coding/embedding suites.",
    ),
    "decision": RunTier(
        jobs=frozenset({
            "pipeline", "reasoning", "session_review", "realdata", "embedding", "coding",
        }),
        repeats=5,
        description="Full evidence run for a production model decision.",
    ),
}


def manifest_dict() -> dict:
    root = Path(__file__).resolve().parents[2]
    fingerprint_files = [
        root / "blipshell" / "benchmark" / "harness.py",
        root / "blipshell" / "llm" / "prompts.py",
        root / "tests" / "benchmark_models.py",
        root / "tests" / "benchmark_reasoning.py",
        root / "tests" / "benchmark_coding.py",
        root / "tests" / "benchmark_continuity.py",
    ]
    digest = hashlib.sha256()
    for path in fingerprint_files:
        if path.exists():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset_version": DATASET_VERSION,
        "scorer_version": SCORER_VERSION,
        "source_fingerprint": digest.hexdigest()[:16],
        "required_categories": list(REQUIRED_CATEGORIES),
        "open_ended_categories": sorted(OPEN_ENDED_CATEGORIES),
        "tiers": {
            name: {**asdict(tier), "jobs": sorted(tier.jobs)}
            for name, tier in RUN_TIERS.items()
        },
    }


def resolve_tier(name: str, jobs=None, repeats=None) -> tuple[set[str], int]:
    if name not in RUN_TIERS:
        raise ValueError(f"Unknown benchmark tier {name!r}; choose {sorted(RUN_TIERS)}")
    tier = RUN_TIERS[name]
    return set(jobs) if jobs is not None else set(tier.jobs), (
        int(repeats) if repeats is not None else tier.repeats
    )
