"""Classify Ollama cloud models as free-tier accessible or subscription-gated.

Sends one tiny chat request per model and classifies the outcome:
- working:      request succeeded — model is accessible
- subscription: 403 with "requires a subscription" message
- not_found:    model doesn't exist on this endpoint
- rate_limited: 429 (retry later)
- other:        unclassified error

Saves a timestamped JSON to data/ollama_probe_YYYY-MM-DD_HHMMSS.json.

Usage:
    python scripts/probe_ollama_models.py
    python scripts/probe_ollama_models.py --models models.txt
    python scripts/probe_ollama_models.py --concurrency 3
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import ollama
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()

DEFAULT_CANDIDATES = [
    # Cloud — new since the Mar 2026 benchmark
    "kimi-k2.6:cloud",
    "deepseek-v4-flash:cloud",
    "gemma4:31b-cloud",
    "nemotron-3-super:cloud",
    "minimax-m2.7:cloud",
    "ministral-3:14b-cloud",
    "ministral-3:8b-cloud",
    # Cloud — known / previously benched
    "glm-5.1:cloud",
    "gpt-oss:120b-cloud",
    "qwen3.5:397b-cloud",
    "qwen3-coder:480b-cloud",
    "kimi-k2.5:cloud",
    "glm-4.7:cloud",
    "nemotron-3-nano:30b-cloud",
    "devstral-small-2:24b-cloud",
    "gemini-3-flash-preview:cloud",
    "minimax-m2.5:cloud",
    # Local — RTX 3060 12GB friendly (pull first with `ollama pull <name>`)
    "qwen3:14b",
    "gpt-oss:latest",
    "ministral-3:14b",
    "gemma4:e4b",
]

STATUS_COLORS = {
    "working": "green",
    "subscription": "yellow",
    "rate_limited": "magenta",
    "not_found": "red",
    "other": "red",
}


@dataclass
class ProbeResult:
    model: str
    status: str
    message: str
    status_code: int | None
    elapsed_seconds: float


def classify(err: Exception) -> tuple[str, int | None]:
    sc = getattr(err, "status_code", None)
    msg = str(err).lower()
    if sc == 403 or "subscription" in msg or "upgrade for access" in msg:
        return "subscription", sc
    if sc == 429 or "rate limit" in msg or "quota" in msg or "too many requests" in msg:
        return "rate_limited", sc
    if sc == 404 or "not found" in msg or "does not exist" in msg or "pull the model" in msg:
        return "not_found", sc
    return "other", sc


async def probe_one(client: ollama.AsyncClient, model: str, semaphore: asyncio.Semaphore) -> ProbeResult:
    async with semaphore:
        start = time.perf_counter()
        try:
            await client.chat(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                options={"num_predict": 4},
            )
            return ProbeResult(model, "working", "OK", 200, time.perf_counter() - start)
        except Exception as e:
            status, sc = classify(e)
            return ProbeResult(model, status, str(e)[:250], sc, time.perf_counter() - start)


async def run(host: str, models: list[str], concurrency: int) -> list[ProbeResult]:
    client = ollama.AsyncClient(host=host, timeout=httpx.Timeout(30.0, connect=10.0))
    semaphore = asyncio.Semaphore(concurrency)

    console.print(f"[cyan]Probing {len(models)} models at {host} (concurrency={concurrency})...[/cyan]\n")

    tasks = [probe_one(client, m, semaphore) for m in models]
    results: list[ProbeResult] = []

    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        color = STATUS_COLORS.get(r.status, "white")
        console.print(f"  [{color}]{r.status:13}[/{color}] {r.model:40} ({r.elapsed_seconds:.1f}s)")

    return results


def render_summary(results: list[ProbeResult]) -> None:
    results = sorted(results, key=lambda r: (r.status, r.model))

    table = Table(title="Probe Summary", show_lines=False)
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Time", justify="right")
    table.add_column("Code", justify="right")
    table.add_column("Message", overflow="fold", max_width=60)

    for r in results:
        color = STATUS_COLORS.get(r.status, "white")
        table.add_row(
            f"[{color}]{r.status}[/{color}]",
            r.model,
            f"{r.elapsed_seconds:.1f}s",
            str(r.status_code) if r.status_code else "-",
            r.message[:80],
        )
    console.print(table)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    console.print(f"\n[bold]Totals:[/bold] {', '.join(parts)}")

    working = [r.model for r in results if r.status == "working"]
    if working:
        console.print(f"\n[green]Use --models with these {len(working)} working models for benchmark:[/green]")
        for m in working:
            console.print(f"  {m}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Ollama models for free-tier access")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--models", help="Path to newline-separated model list file")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Concurrent probes (keep low to avoid tripping rate limits)")
    parser.add_argument("--output", default="data", help="Output directory for JSON results")
    args = parser.parse_args()

    if args.models:
        with open(args.models) as f:
            models = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        models = DEFAULT_CANDIDATES

    results = asyncio.run(run(args.host, models, args.concurrency))
    render_summary(results)

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / f"ollama_probe_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": args.host,
            "models_probed": len(models),
            "results": [asdict(r) for r in results],
        }, f, indent=2)

    console.print(f"\n[dim]Saved to {out_path}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
