"""One-time batch tag run using Ollama cloud models.

Clears the poorly-tagged backlog by routing through local-cloud endpoint
(Ollama paid tier) instead of slow local models. Much faster — cloud
models respond in <2s vs 10-30s locally.

Usage:
  python scripts/batch_tag_cloud.py                    # full run
  python scripts/batch_tag_cloud.py --max-batches 10   # limited run
  python scripts/batch_tag_cloud.py --dry-run           # show count only
  python scripts/batch_tag_cloud.py --model glm-5:cloud # specify model

Requires: Ollama cloud tier active, model available.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def run(max_batches: int, dry_run: bool, model: str):
    from blipshell.core.config import ConfigManager
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter, TaskType
    from blipshell.memory.batch_tagger import BatchTagger
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.llm.prompts import batch_assign_tags

    config_manager = ConfigManager(config_path=None)
    config = config_manager.load()

    sqlite = SQLiteStore(config.database.path)
    await sqlite.initialize()

    # Check backlog
    poorly_tagged = await sqlite.get_poorly_tagged_memory_ids(max_tags=1, limit=99999)
    print(f"Poorly-tagged memories: {len(poorly_tagged)}")
    batch_size = config.memory.batch_tag_batch_size
    batches_needed = (len(poorly_tagged) + batch_size - 1) // batch_size
    print(f"Batches needed: {batches_needed} (at {batch_size}/batch)")

    if dry_run:
        print("Dry run — exiting.")
        await sqlite.close()
        return

    if not poorly_tagged:
        print("Nothing to tag.")
        await sqlite.close()
        return

    # Create router, then patch generate to use Ollama cloud directly.
    # This avoids the normal routing logic and sends all calls to the
    # specified cloud model via the Ollama API.
    endpoint_manager = EndpointManager(config.endpoints, config.llm)
    router = LLMRouter(config.models, endpoint_manager, pii_enabled=False)

    import ollama as ollama_sdk

    ollama_client = ollama_sdk.AsyncClient(host="http://localhost:11434")

    original_generate = router.generate

    async def cloud_generate(task_type, prompt="", system=None, **kwargs):
        """Route all calls through Ollama cloud model."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = await ollama_client.chat(
                model=model,
                messages=messages,
            )
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Cloud call failed: {e}") from e

    router.generate = cloud_generate

    tagger = BatchTagger(sqlite, router, config.memory)

    actual_batches = min(max_batches, batches_needed)
    print(f"\nRunning {actual_batches} batches via {model}...\n")

    t0 = time.monotonic()
    total_tagged = 0
    total_assigned = 0
    errors = 0

    for i in range(actual_batches):
        bt0 = time.monotonic()
        stats = await tagger.tag_batch()
        elapsed = time.monotonic() - bt0

        if stats["memories_in_batch"] == 0:
            print(f"  Batch {i+1}: no more poorly-tagged memories. Done.")
            break

        total_tagged += stats["memories_tagged"]
        total_assigned += stats["tags_assigned"]
        if stats["error"]:
            errors += 1

        tagged = stats["memories_tagged"]
        assigned = stats["tags_assigned"]
        err = f" ERROR: {stats['error']}" if stats["error"] else ""
        print(f"  Batch {i+1}/{actual_batches}: {tagged} tagged, {assigned} tags ({elapsed:.1f}s){err}")

    total_elapsed = time.monotonic() - t0

    # Check remaining
    remaining = await sqlite.get_poorly_tagged_memory_ids(max_tags=1, limit=99999)

    print(f"\n{'='*50}")
    print(f"Complete in {total_elapsed:.1f}s")
    print(f"  Tagged: {total_tagged} memories")
    print(f"  Tags assigned: {total_assigned}")
    print(f"  Errors: {errors}")
    print(f"  Remaining: {len(remaining)} poorly-tagged")

    await sqlite.close()


def main():
    parser = argparse.ArgumentParser(description="One-time cloud batch tagging")
    parser.add_argument("--max-batches", type=int, default=99999, help="Max batches to run")
    parser.add_argument("--dry-run", action="store_true", help="Show count only")
    parser.add_argument("--model", default="glm-5:cloud", help="Cloud model to use")
    args = parser.parse_args()

    asyncio.run(run(args.max_batches, args.dry_run, args.model))


if __name__ == "__main__":
    main()
