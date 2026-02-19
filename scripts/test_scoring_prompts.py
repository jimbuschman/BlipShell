"""Test different scoring prompts and models on a sample of memories.

Runs 48 diverse memories through 3 strategies:
  A. Current prompt (qwen2.5:14b via RANKING_IMPORTANCE task type)
  B. Few-shot prompt (same model, better prompt with examples)
  C. Few-shot prompt on qwen3:14b (different model via IMPORTANCE task type)

Usage:
    python scripts/test_scoring_prompts.py [--db path/to/blipshell.db]
"""

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.core.config import ConfigManager
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.prompts import rank_and_importance
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.processor import MemoryProcessor

# --- Improved prompt with few-shot examples ---

FEWSHOT_SYSTEM = """\
You rate messages on two scales.

RANK (1-5) — how valuable is this to remember?
1 = Noise: greetings, filler, "ok", "thanks", "sure", short acknowledgments
2 = Minor: vague or context-dependent messages, simple yes/no, "still not working", "can you fix it?"
3 = Useful: contains a clear topic or question with enough context to be standalone
4 = Important: meaningful insight, decision, preference, technical detail, or personal fact
5 = Critical: core identity fact, major life decision, architectural choice, or turning point

IMPORTANCE (0.0-1.0) — how important to remember long-term?
0.1 = Throwaway: greetings, filler, system noise
0.3 = Low: casual chat, minor details, context-dependent fragments
0.5 = Medium: useful context, specific technical question
0.7 = High: user preference, project decision, recurring theme, personal detail
0.9 = Critical: core identity fact, major decision, key personal info

IMPORTANT: Use the FULL scale. Most conversations contain a mix of filler (1-2) and substance (3-5). Short context-dependent messages without standalone meaning should be 1-2, not 3.

Examples:
"ok sounds good" → 1 0.1
"can you fix that?" → 2 0.2
"still not working" → 1 0.1
"whichever you wanna do first" → 1 0.1
"I want to swing an x/y point in an arch, how can i do that with a c# function?" → 3 0.5
"I decided to switch from MongoDB to PostgreSQL for the project" → 4 0.7
"My cat's name is Luna and she's 3 years old" → 5 0.9
"Here's how to add a heatsink to the back of the PCB when the chip is blocked..." → 4 0.7
"I'm thinking about quitting my job to focus on this full-time" → 5 0.9

Respond with ONLY two numbers separated by a space: rank importance
Example: 4 0.7"""

FEWSHOT_USER = "Rate this message:\n\n{text}"


async def score_batch(router, task_type, samples, system_prompt, user_template, label):
    """Score a batch of samples and return results."""
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")
    results = []
    for i, (mid, content, old_rank, old_imp) in enumerate(samples):
        user_prompt = user_template.format(text=content)
        start = time.monotonic()
        try:
            resp = await router.generate(task_type, user_prompt, system=system_prompt)
            rank, importance = MemoryProcessor._parse_rank_and_importance(resp)
        except Exception as e:
            rank, importance = 3, 0.3
        elapsed = time.monotonic() - start

        preview = content[:90].replace("\n", " ")
        print(f"  [{i+1:2d}/{len(samples)}] id={mid} old={old_rank}/{old_imp:.1f} "
              f"new={rank}/{importance:.1f} ({elapsed:.1f}s) {preview}")
        results.append((mid, rank, importance))
    return results


def print_comparison(samples, *named_results):
    """Print distribution comparison and side-by-side."""
    print(f"\n{'=' * 70}")
    print("DISTRIBUTION COMPARISON")
    print(f"{'=' * 70}")

    all_sets = [("DB (current)", [(m, r, i) for m, _, r, i in samples])]
    all_sets.extend(named_results)

    for label, results in all_sets:
        ranks = [r for _, r, _ in results]
        imps = [i for _, _, i in results]
        print(f"\n  {label}:")
        for rv in range(1, 6):
            cnt = ranks.count(rv)
            bar = "#" * (cnt * 2)
            print(f"    Rank {rv}: {cnt:2d} ({cnt/len(ranks)*100:5.1f}%) {bar}")
        avg_rank = sum(ranks) / len(ranks)
        avg_imp = sum(imps) / len(imps)
        print(f"    Avg rank: {avg_rank:.2f}  Avg importance: {avg_imp:.2f}")

    # Side-by-side
    labels = [l for l, _ in named_results]
    header = f"  {'id':>6}  {'DB':>5}"
    for l in labels:
        header += f"  {l[:8]:>8}"
    header += "  content"

    print(f"\n{'=' * 70}")
    print("SIDE-BY-SIDE")
    print(f"{'=' * 70}")
    print(header)
    for j in range(len(samples)):
        mid, content, old_r, old_i = samples[j]
        line = f"  {mid:>6}  {old_r}/{old_i:.1f}"
        for _, results in named_results:
            _, r, i = results[j]
            line += f"  {r:>3}/{i:.1f}"
        preview = content[:70].replace("\n", " ")
        line += f"  {preview}"
        print(line)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="Path to blipshell.db (defaults to config)")
    args = parser.parse_args()

    config_mgr = ConfigManager()
    config = config_mgr.load()

    db_path = args.db or config.database.path
    sqlite_conn = sqlite3.connect(db_path)
    print(f"Using database: {db_path}")

    # Load sample IDs, or auto-generate them
    sample_file = Path("data/test_sample_ids.json")
    if not sample_file.exists():
        print("No sample file -- auto-selecting 48 diverse samples...")
        c = sqlite_conn.cursor()
        sample_ids = []
        for query in [
            "SELECT id FROM memories WHERE LENGTH(content) < 30 AND role='user' ORDER BY RANDOM() LIMIT 8",
            "SELECT id FROM memories WHERE LENGTH(content) BETWEEN 50 AND 150 AND role='user' ORDER BY RANDOM() LIMIT 8",
            "SELECT id FROM memories WHERE LENGTH(content) BETWEEN 200 AND 600 AND role='user' ORDER BY RANDOM() LIMIT 8",
            "SELECT id FROM memories WHERE LENGTH(content) > 500 AND role='assistant' ORDER BY RANDOM() LIMIT 8",
            "SELECT id FROM memories WHERE LENGTH(content) BETWEEN 100 AND 500 AND role='assistant' ORDER BY RANDOM() LIMIT 8",
            "SELECT id FROM memories WHERE LENGTH(content) < 80 AND role='assistant' ORDER BY RANDOM() LIMIT 8",
        ]:
            c.execute(query)
            sample_ids.extend(r[0] for r in c.fetchall())
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text(json.dumps(sample_ids))
        print(f"  Selected {len(sample_ids)} samples")
    else:
        sample_ids = json.loads(sample_file.read_text())

    # Load sample data
    c = sqlite_conn.cursor()
    samples = []
    for mid in sample_ids:
        c.execute("SELECT id, content, rank, importance FROM memories WHERE id=?", (mid,))
        row = c.fetchone()
        if row:
            samples.append(row)
    print(f"Loaded {len(samples)} samples")

    # Init router
    endpoint_mgr = EndpointManager(config.endpoints, llm_config=config.llm)
    router = LLMRouter(config.models, endpoint_mgr)

    ri_model = router.get_model(TaskType.RANKING_IMPORTANCE)
    imp_model = router.get_model(TaskType.IMPORTANCE)
    print(f"RANKING_IMPORTANCE model (A/B): {ri_model}")
    print(f"IMPORTANCE model (C):           {imp_model}")

    # Get the current prompt's system text for test A
    _, sample_user = rank_and_importance("test")
    current_system = rank_and_importance("test")[0]

    # --- Test A: Current prompt, current model (qwen2.5:14b) ---
    results_a = await score_batch(
        router, TaskType.RANKING_IMPORTANCE, samples,
        current_system, "Rate this message:\n\n{text}",
        f"A: Current prompt ({ri_model})",
    )

    # --- Test B: Few-shot prompt, same model (qwen2.5:14b) ---
    results_b = await score_batch(
        router, TaskType.RANKING_IMPORTANCE, samples,
        FEWSHOT_SYSTEM, FEWSHOT_USER,
        f"B: Few-shot prompt ({ri_model})",
    )

    # --- Test C: Few-shot prompt, different model (qwen3:14b via IMPORTANCE) ---
    results_c = await score_batch(
        router, TaskType.IMPORTANCE, samples,
        FEWSHOT_SYSTEM, FEWSHOT_USER,
        f"C: Few-shot prompt ({imp_model})",
    )

    # --- Compare all ---
    print_comparison(
        samples,
        ("A:current", results_a),
        ("B:fewshot", results_b),
        (f"C:{imp_model[:8]}", results_c),
    )

    # Save results for review
    output = {
        "samples": [(m, c[:200], r, i) for m, c, r, i in samples],
        "results_a": results_a,
        "results_b": results_b,
        "results_c": results_c,
    }
    out_path = Path("data/scoring_test_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
