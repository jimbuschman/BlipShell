"""Capture the exact shape of Ollama's 429 / quota-exhausted errors.

Sends concurrent requests to the configured Ollama cloud model until one
fails, then dumps the full exception structure (type, MRO, attributes,
headers, cause chain) so we can write a correct classifier for the router.

Usage:
    python scripts/capture_ollama_429.py
    python scripts/capture_ollama_429.py --concurrency 10 --max-requests 50
    python scripts/capture_ollama_429.py --host http://localhost:11434 --model glm-5:cloud
"""

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

import httpx
import ollama


def dump_exception(e: BaseException, label: str = "EXCEPTION") -> None:
    """Print everything useful about an exception."""
    print(f"\n{'=' * 70}")
    print(label)
    print("=" * 70)
    print(f"type:     {type(e).__name__}")
    print(f"module:   {type(e).__module__}")
    print(f"mro:      {[c.__name__ for c in type(e).__mro__]}")
    print(f"str():    {str(e)!r}")
    print(f"repr():   {repr(e)}")

    # Attributes commonly set by SDK error classes
    interesting = (
        "status_code", "error", "code", "message", "response",
        "headers", "body", "request", "args",
    )
    for attr in interesting:
        if hasattr(e, attr):
            try:
                val = getattr(e, attr)
                print(f"  .{attr}: {val!r}")
            except Exception as inner:
                print(f"  .{attr}: <read failed: {inner}>")

    # Instance __dict__
    try:
        d = vars(e)
        if d:
            print(f"  vars(): {d!r}")
    except TypeError:
        pass

    # If there's an httpx response buried in here, show its headers —
    # Retry-After is the key signal we need.
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            headers = getattr(resp, "headers", None)
            if headers:
                print("  response.headers:")
                for k, v in headers.items():
                    print(f"    {k}: {v}")
            sc = getattr(resp, "status_code", None)
            if sc is not None:
                print(f"  response.status_code: {sc}")
            text = getattr(resp, "text", None)
            if text:
                print(f"  response.text: {text[:500]!r}")
        except Exception as inner:
            print(f"  response inspection failed: {inner}")

    if e.__cause__ is not None:
        print(f"  __cause__: {type(e.__cause__).__name__}: {e.__cause__!r}")
    if e.__context__ is not None and e.__context__ is not e.__cause__:
        print(f"  __context__: {type(e.__context__).__name__}: {e.__context__!r}")

    print("=" * 70)


def load_from_config(config_path: Path) -> tuple[str, str]:
    """Pull host + cloud model from config.yaml."""
    try:
        import yaml
    except ImportError:
        return "http://localhost:11434", "glm-5:cloud"

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        return "http://localhost:11434", "glm-5:cloud"

    endpoints = (cfg.get("llm", {}) or {}).get("endpoints", []) or []
    for ep in endpoints:
        if ep.get("provider") != "ollama":
            continue
        host = ep.get("host", "http://localhost:11434")
        models = ep.get("models", {}) or {}
        # Prefer a cloud-tagged model
        for key in ("tool_calling", "coding", "reasoning", "summarization"):
            m = models.get(key)
            if m and ":cloud" in m:
                return host, m
        # Fall back to any tool_calling model on this endpoint
        m = models.get("tool_calling") or "glm-5:cloud"
        return host, m

    return "http://localhost:11434", "glm-5:cloud"


async def send_one(client: ollama.AsyncClient, model: str, i: int):
    try:
        await client.chat(
            model=model,
            messages=[{"role": "user", "content": f"Say only the number {i}."}],
            stream=False,
            options={"num_predict": 4},
        )
        return i, None
    except BaseException as e:
        return i, e


async def run(host: str, model: str, concurrency: int, max_requests: int) -> int:
    print(f"Host:        {host}")
    print(f"Model:       {model}")
    print(f"Concurrency: {concurrency}")
    print(f"Max reqs:    {max_requests}")
    print()

    client = ollama.AsyncClient(
        host=host,
        timeout=httpx.Timeout(30.0, connect=10.0),
    )

    sent = 0
    ok = 0
    errors: list[tuple[int, BaseException]] = []

    while sent < max_requests:
        batch = min(concurrency, max_requests - sent)
        tasks = [send_one(client, model, sent + j) for j in range(batch)]
        results = await asyncio.gather(*tasks)

        for idx, err in results:
            sent += 1
            if err is None:
                ok += 1
                print(f"  [{idx:03d}] OK")
            else:
                errors.append((idx, err))
                print(f"  [{idx:03d}] ERROR: {type(err).__name__}: {err}")

        if errors:
            break

    print(f"\nSummary: {ok} ok, {len(errors)} errors, {sent} sent")

    if not errors:
        print("\nNo errors captured. Suggestions:")
        print("  - re-run with --concurrency 10 --max-requests 100")
        print("  - run repeatedly — free tier quota resets daily")
        return 1

    for idx, err in errors[:3]:
        dump_exception(err, label=f"ERROR from request #{idx}")

    print("\nFull traceback of first error:")
    first = errors[0][1]
    traceback.print_exception(type(first), first, first.__traceback__)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Ollama 429 error shape")
    parser.add_argument("--host", default=None, help="Override host (default: from config.yaml)")
    parser.add_argument("--model", default=None, help="Override model (default: from config.yaml)")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-requests", type=int, default=30)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    host, model = args.host, args.model
    if host is None or model is None:
        cfg_host, cfg_model = load_from_config(Path(args.config))
        host = host or cfg_host
        model = model or cfg_model

    return asyncio.run(run(host, model, args.concurrency, args.max_requests))


if __name__ == "__main__":
    sys.exit(main())
