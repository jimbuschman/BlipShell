"""Benchmark pipeline-critical LLM calls: speed vs accuracy.

Pulls real scored memories from the DB and tests the actual production prompts
(rank_importance_and_classify, decide_memory_action, summarize_memory) against
multiple models — both local and cloud.

The goal: find faster models that maintain acceptable quality for the memory
pipeline bottleneck (currently 13-36s per call on local qwen2.5:14b/qwen3:14b).

Usage:
    python scripts/benchmark_pipeline_speed.py
    python scripts/benchmark_pipeline_speed.py --models gemma3:4b qwen2.5:7b
    python scripts/benchmark_pipeline_speed.py --cloud          # include cloud models
    python scripts/benchmark_pipeline_speed.py --sample 10      # fewer samples (faster)
    python scripts/benchmark_pipeline_speed.py --task ranking    # only test ranking
    python scripts/benchmark_pipeline_speed.py --config path/to/config.yaml
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
from blipshell.llm.prompts import (
    decide_memory_action,
    rank_importance_and_classify,
    summarize_memory,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.processor import MemoryProcessor
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig

# Default small/fast local models to test
LOCAL_MODELS = [
    "qwen2.5:14b",     # current ranking model (baseline)
    "qwen2.5:7b",      # smaller variant
    "gemma3:4b",        # small and fast
    "phi4-mini:latest", # Microsoft small model
    "llama3.2:3b",      # Meta small model
]

# Cloud models to test (need --cloud flag and config with endpoints)
CLOUD_MODELS = [
    # Groq — fast inference on open-source models
    ("groq", "openai/gpt-oss-120b"),       # current summarization model
    ("groq", "openai/gpt-oss-20b"),         # 1000 t/s, fastest on Groq
    ("groq", "llama-3.3-70b-versatile"),    # strong general model
    ("groq", "llama-3.1-8b-instant"),       # small and fast
    ("groq", "qwen/qwen3-32b"),            # same family as local qwen3:14b
    # Gemini — Google free tier
    ("gemini", "gemini-2.5-flash"),         # current config
    ("gemini", "gemini-2.5-flash-lite"),    # faster, 4x daily quota (1000 RPD)
]


def load_sample(db_path: str, n: int = 20) -> list[dict]:
    """Pull a stratified sample of memories with known scores."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    messages = []
    # Get a mix of rank levels so we can check accuracy across the range
    for rank in [1, 2, 3, 4, 5]:
        per_rank = max(n // 5, 2)
        rows = c.execute(
            """SELECT id, content, summary, role, rank, importance, memory_type
               FROM memories
               WHERE rank = ? AND summary IS NOT NULL AND length(content) > 20
               ORDER BY RANDOM() LIMIT ?""",
            (rank, per_rank),
        ).fetchall()
        for row in rows:
            messages.append(dict(row))

    conn.close()

    # Deduplicate
    seen = set()
    unique = []
    for m in messages:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique.append(m)
    return unique[:n]


def load_dedup_pairs(db_path: str, n: int = 10) -> list[tuple[dict, list[str]]]:
    """Pull memories that have similar neighbors for dedup testing."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get some memories with summaries
    rows = c.execute(
        """SELECT id, content, summary, role, rank
           FROM memories
           WHERE summary IS NOT NULL AND length(summary) > 20
           ORDER BY RANDOM() LIMIT ?""",
        (n * 3,),
    ).fetchall()
    conn.close()

    # Pair them up as "new" + "existing" for dedup prompts
    pairs = []
    items = [dict(r) for r in rows]
    for i in range(0, len(items) - 1, 2):
        if len(pairs) >= n:
            break
        pairs.append((items[i], [items[i + 1]["summary"]]))

    return pairs


def make_local_router(model_name: str, ollama_url: str = "http://localhost:11434") -> LLMRouter:
    """Create a router that sends all tasks to a specific local model."""
    models = ModelsConfig(
        reasoning=model_name,
        summarization=model_name,
        ranking=model_name,
        importance=model_name,
        ranking_importance=model_name,
    )
    ep = EndpointConfig(
        name="bench-local",
        url=ollama_url,
        roles=["reasoning", "summarization", "ranking", "importance", "ranking_importance"],
        priority=1,
        max_concurrent=1,
    )
    return LLMRouter(models, EndpointManager([ep], LLMConfig()))


def make_cloud_router(endpoint_name: str, model_name: str, config) -> LLMRouter | None:
    """Create a router using a cloud endpoint from the config."""
    # Find the endpoint config
    ep_config = None
    for ep in config.endpoints:
        if ep.name == endpoint_name and ep.enabled:
            ep_config = ep
            break
    if not ep_config:
        print(f"  [SKIP] Endpoint '{endpoint_name}' not found or disabled")
        return None

    # Override all roles to use this endpoint
    ep_config_copy = ep_config.model_copy()
    ep_config_copy.roles = ["reasoning", "summarization", "ranking", "importance", "ranking_importance"]
    ep_config_copy.models = {
        "reasoning": model_name,
        "summarization": model_name,
        "ranking": model_name,
        "importance": model_name,
        "ranking_importance": model_name,
    }

    models = ModelsConfig(
        reasoning=model_name,
        summarization=model_name,
        ranking=model_name,
        importance=model_name,
        ranking_importance=model_name,
    )
    return LLMRouter(models, EndpointManager([ep_config_copy], config.llm))


async def bench_rank_importance(router: LLMRouter, messages: list[dict], is_cloud: bool = False) -> dict:
    """Benchmark rank_importance_and_classify (the production prompt)."""
    results = []
    delay = 3.0 if is_cloud else 0.05  # respect cloud rate limits
    for msg in messages:
        sys_prompt, user_prompt = rank_importance_and_classify(msg["content"])
        t0 = time.monotonic()
        try:
            raw = await router.generate(
                TaskType.RANKING_IMPORTANCE, user_prompt, system=sys_prompt, think=False,
            )
            elapsed = time.monotonic() - t0
            rank, importance, mem_type = MemoryProcessor._parse_rank_importance_type(raw)
            results.append({
                "id": msg["id"],
                "orig_rank": msg["rank"],
                "orig_importance": msg["importance"],
                "orig_type": msg.get("memory_type", "?"),
                "new_rank": rank,
                "new_importance": importance,
                "new_type": mem_type,
                "time": round(elapsed, 2),
                "raw": raw[:100],
                "ok": True,
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            results.append({
                "id": msg["id"],
                "time": round(elapsed, 2),
                "error": str(e)[:80],
                "ok": False,
            })
        await asyncio.sleep(delay)

    return _analyze_ranking(results)


def _analyze_ranking(results: list[dict]) -> dict:
    """Compute accuracy and speed stats for ranking results."""
    ok = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    times = [r["time"] for r in ok]

    if not ok:
        return {"results": results, "avg_time": 0, "errors": len(errors), "total": len(results)}

    # Rank accuracy: within ±1 of original
    rank_exact = sum(1 for r in ok if r["new_rank"] == r["orig_rank"])
    rank_close = sum(1 for r in ok if abs(r["new_rank"] - r["orig_rank"]) <= 1)

    # Importance accuracy: within ±0.2 of original
    imp_close = sum(1 for r in ok
                    if abs(r["new_importance"] - r["orig_importance"]) <= 0.2)

    # Type accuracy
    type_match = sum(1 for r in ok if r["new_type"] == r["orig_type"])

    # Rank distribution
    rank_dist = {}
    for r in ok:
        rank_dist[r["new_rank"]] = rank_dist.get(r["new_rank"], 0) + 1

    return {
        "results": results,
        "total": len(results),
        "errors": len(errors),
        "avg_time": round(sum(times) / len(times), 2),
        "min_time": round(min(times), 2),
        "max_time": round(max(times), 2),
        "total_time": round(sum(times), 1),
        "rank_exact": f"{rank_exact}/{len(ok)}",
        "rank_within_1": f"{rank_close}/{len(ok)}",
        "importance_within_0.2": f"{imp_close}/{len(ok)}",
        "type_match": f"{type_match}/{len(ok)}",
        "rank_distribution": rank_dist,
    }


async def bench_dedup(router: LLMRouter, pairs: list[tuple[dict, list[str]]], is_cloud: bool = False) -> dict:
    """Benchmark decide_memory_action (dedup reasoning call)."""
    results = []
    delay = 3.0 if is_cloud else 0.05
    for msg, existing in pairs:
        sys_prompt, user_prompt = decide_memory_action(msg["summary"], existing)
        t0 = time.monotonic()
        try:
            raw = await router.generate(
                TaskType.REASONING, user_prompt, system=sys_prompt, think=False,
            )
            elapsed = time.monotonic() - t0
            action, _ = MemoryProcessor._parse_memory_action(raw)
            results.append({
                "id": msg["id"],
                "action": action,
                "time": round(elapsed, 2),
                "raw": raw[:80],
                "ok": True,
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            results.append({
                "id": msg["id"],
                "time": round(elapsed, 2),
                "error": str(e)[:80],
                "ok": False,
            })
        await asyncio.sleep(delay)

    ok = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    times = [r["time"] for r in ok]
    actions = {}
    for r in ok:
        actions[r["action"]] = actions.get(r["action"], 0) + 1

    return {
        "results": results,
        "total": len(results),
        "errors": len(errors),
        "avg_time": round(sum(times) / len(times), 2) if times else 0,
        "min_time": round(min(times), 2) if times else 0,
        "max_time": round(max(times), 2) if times else 0,
        "total_time": round(sum(times), 1) if times else 0,
        "action_distribution": actions,
    }


async def bench_summarize(router: LLMRouter, messages: list[dict], is_cloud: bool = False) -> dict:
    """Benchmark summarize_memory."""
    results = []
    delay = 3.0 if is_cloud else 0.05
    for msg in messages[:10]:  # summarization is less critical, test fewer
        sys_prompt, user_prompt = summarize_memory(msg["content"])
        t0 = time.monotonic()
        try:
            raw = await router.generate(
                TaskType.SUMMARIZATION, user_prompt, system=sys_prompt, think=False,
            )
            elapsed = time.monotonic() - t0
            is_skip = raw.strip().upper() == "SKIP"
            results.append({
                "id": msg["id"],
                "time": round(elapsed, 2),
                "summary_len": len(raw),
                "is_skip": is_skip,
                "ok": True,
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            results.append({
                "id": msg["id"],
                "time": round(elapsed, 2),
                "error": str(e)[:80],
                "ok": False,
            })
        await asyncio.sleep(delay)

    ok = [r for r in results if r["ok"]]
    times = [r["time"] for r in ok]

    return {
        "results": results,
        "total": len(results),
        "errors": sum(1 for r in results if not r["ok"]),
        "avg_time": round(sum(times) / len(times), 2) if times else 0,
        "min_time": round(min(times), 2) if times else 0,
        "max_time": round(max(times), 2) if times else 0,
        "total_time": round(sum(times), 1) if times else 0,
        "skips": sum(1 for r in ok if r["is_skip"]),
    }


def print_comparison(all_results: dict, task: str):
    """Print a comparison table for a specific task across all models."""
    print(f"\n{'=' * 80}")
    print(f"  {task.upper()} — MODEL COMPARISON")
    print(f"{'=' * 80}")

    header = f"  {'Model':<30s} {'Avg':>6s} {'Min':>6s} {'Max':>6s} {'Total':>7s} {'Err':>4s}"
    if task == "ranking":
        header += f" {'Rank±1':>8s} {'Imp±.2':>8s} {'Type':>8s}"
    elif task == "dedup":
        header += f" {'Actions':>30s}"
    elif task == "summarize":
        header += f" {'Skips':>6s}"
    print(header)
    print("  " + "-" * 78)

    for model_name, data in all_results.items():
        if task not in data:
            continue
        stats = data[task]
        line = (f"  {model_name:<30s} {stats['avg_time']:>5.1f}s {stats['min_time']:>5.1f}s "
                f"{stats['max_time']:>5.1f}s {stats['total_time']:>6.1f}s {stats['errors']:>4d}")
        if task == "ranking":
            line += (f" {stats['rank_within_1']:>8s} "
                     f"{stats['importance_within_0.2']:>8s} "
                     f"{stats['type_match']:>8s}")
        elif task == "dedup":
            actions = stats.get("action_distribution", {})
            act_str = " ".join(f"{k}={v}" for k, v in sorted(actions.items()))
            line += f" {act_str:>30s}"
        elif task == "summarize":
            line += f" {stats.get('skips', 0):>6d}"
        print(line)


OUTPUT_PATH = Path("data") / "benchmark_pipeline_speed.json"


def _save_results(all_results: dict, messages: list[dict]):
    """Save results to JSON (called incrementally after each model)."""
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    save_data = {}
    for model_name, data in all_results.items():
        save_data[model_name] = {}
        for task, stats in data.items():
            # Include per-call results for later analysis
            save_data[model_name][task] = stats
    save_data["_meta"] = {
        "sample_size": len(messages),
        "sample_ids": [m["id"] for m in messages],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(save_data, f, indent=2, default=str)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline speed: find faster models")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Local models to test (default: auto-detect from LOCAL_MODELS)")
    parser.add_argument("--cloud", action="store_true",
                        help="Also test cloud models (Groq, Gemini)")
    parser.add_argument("--sample", type=int, default=20,
                        help="Number of sample memories (default: 20)")
    parser.add_argument("--task", choices=["ranking", "dedup", "summarize", "all"],
                        default="all", help="Which task to benchmark (default: all)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config.yaml (needed for --cloud)")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to blipshell.db (default: from config)")
    args = parser.parse_args()

    # Load config
    config_mgr = ConfigManager(args.config)
    config = config_mgr.load()
    db_path = args.db or config.database.path

    print("=" * 80)
    print("  PIPELINE SPEED BENCHMARK")
    print("=" * 80)
    print(f"  Database: {db_path}")
    print(f"  Sample size: {args.sample}")
    print(f"  Tasks: {args.task}")

    # Load test data
    print(f"\n  Loading sample data...")
    messages = load_sample(db_path, args.sample)
    print(f"  Loaded {len(messages)} memories with known scores")
    rank_dist = {}
    for m in messages:
        rank_dist[m["rank"]] = rank_dist.get(m["rank"], 0) + 1
    print(f"  Rank distribution: {dict(sorted(rank_dist.items()))}")

    dedup_pairs = []
    if args.task in ("dedup", "all"):
        dedup_pairs = load_dedup_pairs(db_path, min(args.sample, 10))
        print(f"  Loaded {len(dedup_pairs)} dedup test pairs")

    # Determine which models to test
    local_models = args.models or LOCAL_MODELS
    test_configs: list[tuple[str, LLMRouter, bool]] = []  # (name, router, is_cloud)

    # Check which local models are actually available
    print(f"\n  Checking model availability...")
    from blipshell.llm.client import LLMClient
    ollama_url = "http://localhost:11434"
    for ep in config.endpoints:
        if ep.provider == "ollama" and ep.enabled:
            ollama_url = ep.url
            break

    client = LLMClient(host=ollama_url)
    try:
        available = await client.list_models()
        available_names = {m.get("name", "") for m in available if isinstance(m, dict)}
        # Also match without tag
        available_base = {n.split(":")[0] for n in available_names}
    except Exception:
        available_names = set()
        available_base = set()
        print(f"  WARNING: Could not list models from {ollama_url}")

    for model in local_models:
        base = model.split(":")[0]
        if model in available_names or base in available_base:
            router = make_local_router(model, ollama_url)
            test_configs.append((f"{model} (local)", router, False))
            print(f"    {model}: available")
        else:
            print(f"    {model}: NOT FOUND, skipping")

    # Cloud models
    if args.cloud:
        for ep_name, model_name in CLOUD_MODELS:
            router = make_cloud_router(ep_name, model_name, config)
            if router:
                test_configs.append((f"{model_name} ({ep_name})", router, True))
                print(f"    {model_name} ({ep_name}): configured")

    if not test_configs:
        print("\n  ERROR: No models available to test!")
        return

    print(f"\n  Testing {len(test_configs)} model configs: "
          f"{', '.join(name for name, _, _ in test_configs)}")

    # Run benchmarks
    all_results: dict[str, dict] = {}

    for model_name, router, is_cloud in test_configs:
        print(f"\n{'─' * 80}")
        print(f"  MODEL: {model_name}{'  [cloud - 3s delay between calls]' if is_cloud else ''}")
        print(f"{'─' * 80}")

        model_data = {}

        if args.task in ("ranking", "all"):
            print(f"  Running rank_importance_and_classify ({len(messages)} messages)...")
            stats = await bench_rank_importance(router, messages, is_cloud)
            model_data["ranking"] = stats
            print(f"    avg={stats['avg_time']}s  "
                  f"rank±1={stats['rank_within_1']}  "
                  f"imp±.2={stats['importance_within_0.2']}  "
                  f"errors={stats['errors']}")

        if args.task in ("dedup", "all") and dedup_pairs:
            print(f"  Running decide_memory_action ({len(dedup_pairs)} pairs)...")
            stats = await bench_dedup(router, dedup_pairs, is_cloud)
            model_data["dedup"] = stats
            print(f"    avg={stats['avg_time']}s  "
                  f"actions={stats.get('action_distribution', {})}  "
                  f"errors={stats['errors']}")

        if args.task in ("summarize", "all"):
            print(f"  Running summarize_memory (10 messages)...")
            stats = await bench_summarize(router, messages, is_cloud)
            model_data["summarize"] = stats
            print(f"    avg={stats['avg_time']}s  "
                  f"skips={stats.get('skips', 0)}  "
                  f"errors={stats['errors']}")

        all_results[model_name] = model_data

        # Incremental save after each model (crash-safe)
        _save_results(all_results, messages)
        print(f"  [saved incrementally]")

    # Print comparison tables
    if args.task in ("ranking", "all"):
        print_comparison(all_results, "ranking")
    if args.task in ("dedup", "all"):
        print_comparison(all_results, "dedup")
    if args.task in ("summarize", "all"):
        print_comparison(all_results, "summarize")

    # Final save
    _save_results(all_results, messages)
    print(f"\n  Results saved to data/benchmark_pipeline_speed.json")

    # Recommendation
    print(f"\n{'=' * 80}")
    print("  RECOMMENDATION")
    print(f"{'=' * 80}")
    if "ranking" in next(iter(all_results.values()), {}):
        fastest = min(all_results.items(),
                      key=lambda x: x[1].get("ranking", {}).get("avg_time", 999))
        best_acc = max(all_results.items(),
                       key=lambda x: int(x[1].get("ranking", {}).get("rank_within_1", "0/0").split("/")[0]))
        print(f"  Fastest for ranking: {fastest[0]} ({fastest[1]['ranking']['avg_time']}s avg)")
        print(f"  Best accuracy: {best_acc[0]} (rank±1: {best_acc[1]['ranking']['rank_within_1']})")
        # Find the sweet spot
        for name, data in sorted(all_results.items(),
                                  key=lambda x: x[1].get("ranking", {}).get("avg_time", 999)):
            stats = data.get("ranking", {})
            within1 = stats.get("rank_within_1", "0/0")
            n_close, n_total = within1.split("/")
            if n_total != "0" and int(n_close) / int(n_total) >= 0.7:
                print(f"  Best speed/quality: {name} "
                      f"({stats['avg_time']}s, rank±1={within1})")
                break


if __name__ == "__main__":
    asyncio.run(main())
