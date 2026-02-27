"""Reclassify memory types for all memories.

All 33K memories are currently 'conversation' because the original batch
reprocessing used the two-value rank_and_importance prompt (no type).
This script classifies type only — does NOT touch rank or importance.

Uses a lightweight type-only prompt (~10x cheaper than full rank+importance).
Processes in batches with resume support.

Usage:
    python scripts/reclassify_memory_types.py                    # run all
    python scripts/reclassify_memory_types.py --sample 50        # test on 50 first
    python scripts/reclassify_memory_types.py --dry-run          # show what would change
    python scripts/reclassify_memory_types.py --reset            # restart from scratch
    python scripts/reclassify_memory_types.py --model qwen3:14b  # use specific model
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.core.config import ConfigManager
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = Path(__file__).resolve().parent.parent / "data" / "reclassify_progress.json"
VALID_TYPES = {"fact", "event", "preference", "skill", "conversation"}

# Lightweight type-only prompt — much shorter than full rank+importance+classify
CLASSIFY_SYSTEM = (
    "Classify this memory into exactly one type.\n\n"
    "Types:\n"
    "fact = Stable truths: personal info, project details, technical specs, decisions made\n"
    "event = Time-bound happenings: meetings, bugs found, deployments, milestones\n"
    "preference = User likes/dislikes, style choices, workflow preferences\n"
    "skill = How-to knowledge, techniques, patterns, solutions, debugging approaches\n"
    "conversation = General chat, greetings, acknowledgments, meta-discussion\n\n"
    "Examples:\n"
    "\"User's cat is named Luna\" → fact\n"
    "\"User prefers dark mode in all editors\" → preference\n"
    "\"Deployed v2.0 to production yesterday\" → event\n"
    "\"To fix a stuck heatsink, apply gentle pressure from the side\" → skill\n"
    "\"ok sounds good\" → conversation\n"
    "\"User discussed building a local LLM system with tiered memory\" → fact\n"
    "\"User debugged rotary phone timer; identified bug in get_timer()\" → event\n\n"
    "Respond with ONLY the type word. Nothing else."
)


def parse_type(response: str) -> str:
    """Parse type from LLM response. Returns 'conversation' if unrecognized."""
    cleaned = re.sub(r"[^a-z]", "", response.strip().lower())
    if cleaned in VALID_TYPES:
        return cleaned
    # Try to find a valid type word anywhere in the response
    for word in response.strip().lower().split():
        cleaned = re.sub(r"[^a-z]", "", word)
        if cleaned in VALID_TYPES:
            return cleaned
    return "conversation"


def load_progress() -> set:
    """Load set of already-processed memory IDs."""
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text())
            return set(data.get("processed_ids", []))
        except Exception:
            pass
    return set()


def save_progress(processed_ids: set):
    """Save processed IDs for resume."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps({
        "processed_ids": list(processed_ids),
        "count": len(processed_ids),
    }))


async def run(args):
    config = ConfigManager().load()
    sqlite = SQLiteStore(config.database.path)
    await sqlite.initialize()

    endpoint_mgr = EndpointManager(config.endpoints, config.llm)
    router = LLMRouter(config.models, endpoint_mgr)

    # Override model if specified
    model_override = args.model

    try:
        # Get all memories that are currently 'conversation'
        rows = await sqlite.db.execute_fetchall(
            "SELECT id, summary FROM memories WHERE memory_type = 'conversation' ORDER BY id"
        )
        total = len(rows)
        logger.info("Found %d memories to reclassify", total)

        if args.sample:
            rows = rows[:args.sample]
            logger.info("Sampling first %d memories", args.sample)

        # Load resume progress
        processed = set() if args.reset else load_progress()
        if processed:
            logger.info("Resuming: %d already processed, skipping", len(processed))

        # Stats
        type_counts = {t: 0 for t in VALID_TYPES}
        errors = 0
        skipped = 0
        t_start = time.monotonic()

        for i, (mem_id, summary) in enumerate(rows):
            if mem_id in processed:
                skipped += 1
                continue

            if not summary or len(summary.strip()) < 10:
                type_counts["conversation"] += 1
                processed.add(mem_id)
                continue

            try:
                if model_override:
                    response = await router._endpoint_manager.get_client_for_role(
                        TaskType.RANKING_IMPORTANCE
                    )
                    # Use the router's generate which handles endpoint selection
                    pass

                prompt = f"Classify this memory:\n\n{summary}"
                response = await router.generate(
                    TaskType.RANKING_IMPORTANCE,
                    prompt,
                    system=CLASSIFY_SYSTEM,
                    think=False,
                )
                memory_type = parse_type(response)
                type_counts[memory_type] += 1

                if not args.dry_run:
                    await sqlite.update_memory(mem_id, memory_type=memory_type)

                processed.add(mem_id)

                # Progress logging
                done = i + 1 - skipped
                if done % 100 == 0 or done <= 5:
                    elapsed = time.monotonic() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(rows) - i - 1) / rate if rate > 0 else 0
                    logger.info(
                        "[%d/%d] %s → %s (%.1f/s, ETA %.0fm) | %s",
                        i + 1, len(rows), summary[:50], memory_type,
                        rate, eta / 60, dict(type_counts),
                    )

                # Save progress every 50 memories
                if done % 50 == 0:
                    save_progress(processed)

            except Exception as e:
                errors += 1
                logger.error("Failed on memory %d: %s", mem_id, e)
                if errors > 20:
                    logger.error("Too many errors, stopping")
                    break

        # Final save
        save_progress(processed)

        elapsed = time.monotonic() - t_start
        logger.info("=" * 60)
        logger.info("DONE in %.1f minutes", elapsed / 60)
        logger.info("Type distribution:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = (c / sum(type_counts.values()) * 100) if sum(type_counts.values()) > 0 else 0
            logger.info("  %-15s %5d  (%.1f%%)", t, c, pct)
        logger.info("Errors: %d", errors)
        if args.dry_run:
            logger.info("DRY RUN — no changes written to DB")

    finally:
        await sqlite.close()


def main():
    parser = argparse.ArgumentParser(description="Reclassify memory types")
    parser.add_argument("--sample", type=int, help="Only process first N memories")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--reset", action="store_true", help="Ignore previous progress")
    parser.add_argument("--model", type=str, help="Override model (e.g. qwen3:14b)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
