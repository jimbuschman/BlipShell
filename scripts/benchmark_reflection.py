"""Benchmark models for session reflection quality.

Tests how well each model produces structured reflections from real session data.
Measures: parse success, section completeness, specificity, and timing.

Usage:
    python scripts/benchmark_reflection.py
    python scripts/benchmark_reflection.py --models qwen3:14b,qwen3:4b,glm4:latest
    python scripts/benchmark_reflection.py --sample 5 --db data/blipshell.db
    python scripts/benchmark_reflection.py --cloud  # include cloud models via Ollama
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from blipshell.core.config import ConfigManager
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.prompts import reflect_on_session
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig, get_ollama_url

console = Console()

DEFAULT_LOCAL_MODELS = ["qwen3.5:4b", "qwen3.5:9b", "qwen3:14b", "qwen3:4b"]
CLOUD_MODELS = ["glm-5:cloud", "qwen3.5:cloud"]


def make_router(model_name: str, ollama_url: str, context_tokens: int = 32768):
    """Create a router that routes REASONING to a specific model."""
    models = ModelsConfig(reasoning=model_name)
    ep = EndpointConfig(
        name="bench",
        url=ollama_url,
        roles=["reasoning"],
        priority=1,
        max_concurrent=1,
        context_tokens=context_tokens,
    )
    return LLMRouter(models, EndpointManager([ep], LLMConfig()))


def score_reflection(raw: str, parsed: dict) -> dict:
    """Score a reflection for quality metrics."""
    scores = {}

    # 1. Parse success — did the parser extract all 5 sections?
    sections = ["effectiveness", "what_worked", "what_didnt_work",
                "technical_insights", "process_insights"]
    filled = sum(1 for s in sections if parsed.get(s))
    scores["sections_filled"] = filled
    scores["sections_total"] = len(sections)

    # 2. Effectiveness is valid
    scores["valid_effectiveness"] = parsed.get("effectiveness") in (
        "effective", "partially_effective", "ineffective", "unclear",
    )

    # 3. Specificity — count bullet points with concrete details
    # (lines starting with - that contain specific terms like file names, numbers, etc.)
    bullet_count = 0
    specific_count = 0
    for section in ["what_worked", "what_didnt_work", "technical_insights", "process_insights"]:
        text = parsed.get(section) or ""
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("•"):
                bullet_count += 1
                # Check for specificity markers
                if any(c in line for c in [".", "(", ":", "=", "/", "`", '"']):
                    specific_count += 1
    scores["bullet_count"] = bullet_count
    scores["specific_count"] = specific_count

    # 4. Length — raw response length (too short = shallow, too long = rambling)
    scores["response_length"] = len(raw)

    # 5. SKIP detection
    scores["is_skip"] = raw.strip().upper() == "SKIP"

    return scores


async def get_sample_sessions(
    sqlite: SQLiteStore, sample_size: int,
) -> list[dict]:
    """Get a mix of sessions for benchmarking.

    Picks sessions with varying sizes to test context handling.
    """
    cursor = await sqlite._db.execute("""
        SELECT s.id, s.summary, s.project, s.title,
               COALESCE(
                   (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.id AND m.is_archived = 0),
                   0
               ) as msg_count
        FROM sessions s
        WHERE s.summary IS NOT NULL AND s.summary != ''
          AND s.is_archived = 0
        ORDER BY RANDOM()
        LIMIT ?
    """, (sample_size * 3,))  # fetch extra to pick a good mix
    rows = [dict(r) for r in await cursor.fetchall()]

    if not rows:
        return []

    # Pick a mix: some short, some medium, some long
    short = [r for r in rows if r["msg_count"] <= 10]
    medium = [r for r in rows if 10 < r["msg_count"] <= 30]
    long = [r for r in rows if r["msg_count"] > 30]

    per_bucket = max(1, sample_size // 3)
    selected = short[:per_bucket] + medium[:per_bucket] + long[:per_bucket]

    # Fill remaining from any bucket
    remaining = sample_size - len(selected)
    if remaining > 0:
        used_ids = {s["id"] for s in selected}
        extras = [r for r in rows if r["id"] not in used_ids]
        selected.extend(extras[:remaining])

    return selected[:sample_size]


async def prepare_session_text(sqlite: SQLiteStore, session_id: int) -> str:
    """Get full raw conversation text for a session."""
    messages = await sqlite.get_session_messages_for_lesson(session_id)
    if not messages:
        memories = await sqlite.get_memories_by_session(session_id)
        if not memories:
            return ""
        return "\n".join(f"{m.role}: {m.content}" for m in memories)
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


async def benchmark_model(
    model_name: str,
    sessions: list[dict],
    sqlite: SQLiteStore,
    ollama_url: str,
    context_tokens: int = 32768,
) -> dict:
    """Benchmark a single model on all sample sessions."""
    router = make_router(model_name, ollama_url, context_tokens)

    results = []
    total_time = 0
    skipped = 0
    errors = 0

    for i, session in enumerate(sessions):
        sid = session["id"]
        summary = session["summary"]
        msg_count = session["msg_count"]
        prefix = f"  [{i+1}/{len(sessions)}]"

        conversation = await prepare_session_text(sqlite, sid)
        if not conversation:
            console.print(f"{prefix} Session {sid}: no data, skipping")
            skipped += 1
            continue

        # Estimate tokens
        from blipshell.memory.manager import estimate_tokens
        conv_tokens = estimate_tokens(conversation)

        # If conversation exceeds context, truncate for this benchmark
        # (we're testing model quality, not chunking logic)
        max_input = context_tokens - 4096
        if conv_tokens > max_input:
            # Truncate to fit — take first N chars that fit
            target_chars = max_input * 4  # rough chars-per-token
            conversation = conversation[:target_chars] + "\n\n[Truncated for benchmark]"
            truncated = True
        else:
            truncated = False

        system, user_prompt = reflect_on_session(summary, conversation, session.get("project"))

        try:
            start = time.monotonic()
            raw = await router.generate(
                TaskType.REASONING, user_prompt, system=system,
            )
            elapsed = time.monotonic() - start
            total_time += elapsed

            try:
                parsed = MemoryProcessor._parse_reflection(raw)
            except Exception as parse_err:
                console.print(f"{prefix} Session {sid}: PARSE ERROR — {parse_err}")
                errors += 1
                results.append({
                    "session_id": sid,
                    "msg_count": msg_count,
                    "error": f"parse: {parse_err}",
                })
                continue
            scores = score_reflection(raw, parsed)

            status = "SKIP" if scores["is_skip"] else "OK"
            if scores["is_skip"]:
                skipped += 1

            console.print(
                f"{prefix} Session {sid} ({msg_count} msgs, {conv_tokens} tok"
                f"{', truncated' if truncated else ''}): "
                f"{status} — {scores['sections_filled']}/5 sections, "
                f"{scores['bullet_count']} bullets, {elapsed:.1f}s"
            )

            results.append({
                "session_id": sid,
                "msg_count": msg_count,
                "conv_tokens": conv_tokens,
                "truncated": truncated,
                "elapsed": round(elapsed, 2),
                "scores": scores,
                "raw_preview": raw[:300],
            })

        except Exception as e:
            console.print(f"{prefix} Session {sid}: ERROR — {e}")
            errors += 1
            results.append({
                "session_id": sid,
                "msg_count": msg_count,
                "error": str(e),
            })

    # Aggregate scores
    valid = [r for r in results if "scores" in r and not r["scores"]["is_skip"]]
    if valid:
        avg_sections = sum(r["scores"]["sections_filled"] for r in valid) / len(valid)
        avg_bullets = sum(r["scores"]["bullet_count"] for r in valid) / len(valid)
        avg_specific = sum(r["scores"]["specific_count"] for r in valid) / len(valid)
        avg_time = sum(r["elapsed"] for r in valid) / len(valid)
        valid_eff = sum(1 for r in valid if r["scores"]["valid_effectiveness"]) / len(valid)
    else:
        avg_sections = avg_bullets = avg_specific = avg_time = valid_eff = 0

    return {
        "model": model_name,
        "context_tokens": context_tokens,
        "total_sessions": len(sessions),
        "completed": len(valid),
        "skipped": skipped,
        "errors": errors,
        "avg_sections": round(avg_sections, 2),
        "avg_bullets": round(avg_bullets, 1),
        "avg_specific": round(avg_specific, 1),
        "avg_time": round(avg_time, 2),
        "valid_effectiveness_pct": round(valid_eff * 100, 1),
        "total_time": round(total_time, 1),
        "results": results,
    }


async def main(args):
    config_mgr = ConfigManager()
    config = config_mgr.load()
    db_path = args.db or config.database.path
    ollama_url = get_ollama_url(config.endpoints)

    console.print(f"\n[bold]Session Reflection Benchmark[/bold]")
    console.print(f"DB: {db_path}")
    console.print(f"Ollama: {ollama_url}")

    sqlite = SQLiteStore(db_path)
    await sqlite.initialize()

    # Get sample sessions
    sessions = await get_sample_sessions(sqlite, args.sample)
    if not sessions:
        console.print("[red]No sessions with summaries found![/red]")
        await sqlite.close()
        return

    console.print(f"Selected {len(sessions)} sessions "
                  f"(msgs: {min(s['msg_count'] for s in sessions)}-"
                  f"{max(s['msg_count'] for s in sessions)})\n")

    # Parse model list
    models = args.models.split(",") if args.models else DEFAULT_LOCAL_MODELS
    if args.cloud:
        models.extend(CLOUD_MODELS)

    # Context tokens per model (known values)
    context_map = {
        # qwen3.5 — all have 256K context
        "qwen3.5:0.8b": 262144,
        "qwen3.5:2b": 262144,
        "qwen3.5:4b": 262144,
        "qwen3.5:9b": 262144,
        "qwen3.5:27b": 262144,
        "qwen3.5:cloud": 262144,
        # qwen3
        "qwen3:4b": 40960,
        "qwen3:14b": 32768,
        # older models
        "qwen2.5:7b": 32768,
        "qwen2.5:14b": 32768,
        "glm4:latest": 32768,
        "gpt-oss:latest": 32768,
        "glm-5:cloud": 196608,
    }

    all_results = []
    for model in models:
        ctx = context_map.get(model, 32768)
        console.print(f"\n[bold cyan]Testing: {model}[/bold cyan] (context: {ctx})")
        console.print("─" * 60)
        result = await benchmark_model(model, sessions, sqlite, ollama_url, ctx)
        all_results.append(result)

    await sqlite.close()

    # Summary table
    console.print(f"\n{'═' * 70}")
    table = Table(title="Reflection Benchmark Results")
    table.add_column("Model", style="bold")
    table.add_column("Context", justify="right")
    table.add_column("OK/Skip/Err", justify="center")
    table.add_column("Sections", justify="right")
    table.add_column("Bullets", justify="right")
    table.add_column("Specific", justify="right")
    table.add_column("Valid Eff%", justify="right")
    table.add_column("Avg Time", justify="right")

    for r in all_results:
        table.add_row(
            r["model"],
            f"{r['context_tokens']//1024}K",
            f"{r['completed']}/{r['skipped']}/{r['errors']}",
            f"{r['avg_sections']:.1f}/5",
            f"{r['avg_bullets']:.0f}",
            f"{r['avg_specific']:.0f}",
            f"{r['valid_effectiveness_pct']:.0f}%",
            f"{r['avg_time']:.1f}s",
        )

    console.print(table)

    # Save detailed results
    output_path = Path("data/benchmark_reflection_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\nDetailed results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark models for session reflection quality",
    )
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: qwen3:14b,qwen3:4b,qwen2.5:7b,qwen2.5:14b)")
    parser.add_argument("--sample", type=int, default=5,
                        help="Number of sessions to test per model (default 5)")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to blipshell.db")
    parser.add_argument("--cloud", action="store_true",
                        help="Include cloud models (glm-5:cloud)")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
