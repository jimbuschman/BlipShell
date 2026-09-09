"""D2a evaluation boundary CLI: build, label, freeze, run (see memory/attribution_eval.py).

Usage (from repo root):
    python -m scripts.attribution_eval build  --generation pre-D1 [--db PATH]
    python -m scripts.attribution_eval show   --generation pre-D1
    python -m scripts.attribution_eval label  --generation pre-D1 c12 lesson_wrong 34
    python -m scripts.attribution_eval freeze --generation pre-D1
    python -m scripts.attribution_eval run    --generation pre-D1 --url http://<ollama>:11434 \
        --model qwen3:14b --repeats 3

`build` reads the database read-only and refuses to overwrite a frozen set.
`run` needs a frozen set and a real model (Ollama PC, or over Tailscale -
ask before tying up the GPU); it writes the full record under
data/attribution_eval/ (gitignored) and a numbers-only summary under
benchmark_results/. The first run on a generation is its baseline; a run
with a changed judge is reported as a new judge version, never merged.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.memory import attribution_eval as ev  # noqa: E402

SELECTION_PRE_D1 = ("pre-D1: always-on top-30 lessons by importance in every request "
                    "plus per-query Recall lesson hits (>= 0.4 similarity)")


def _sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "nosha"
    except Exception:
        return "nosha"


async def _build(args) -> int:
    from blipshell.core.config import ConfigManager
    from blipshell.memory.sqlite_store import SQLiteStore

    db = args.db or ConfigManager().load().database.path
    p = ev.set_path(args.generation)
    if p.exists() and ev.load_set(args.generation).frozen:
        print(f"refusing: {p} is frozen; use a new --generation")
        return 2
    store = SQLiteStore(db)
    await store.initialize()
    try:
        s = await ev.build_set(store, generation=args.generation, selection_behavior=SELECTION_PRE_D1,
                               db_name=Path(db).name)
    finally:
        await store.close()
    out = ev.save_set(s)
    c = ev.census(s)
    print(f"built {c['items']} items -> {out}")
    print(f"  from corrections rows: {sum(1 for i in s.items if i.source == 'corrections_row')}, "
          f"from historical detector lessons: {sum(1 for i in s.items if i.source == 'historical_lesson')}")
    print("label with: python -m scripts.attribution_eval label --generation "
          f"{args.generation} <item_id> <attribution> [<lesson_id>]")
    return 0


def _show(args) -> int:
    s = ev.load_set(args.generation)
    print(ev.census(s))
    for i in s.items:
        lab = f" human={i.human_attribution}/{i.human_lesson_id}" if i.human_attribution else " (unlabelled)"
        print(f"{i.item_id} [{i.source}, lessons {i.lessons_present_source}]{lab}")
        print(f"    prev: {i.prev_assistant[:100]!r}")
        print(f"    user: {i.text[:160]!r}")
        for lid, txt in i.lessons:
            print(f"    [{lid}] {txt[:100]}")
    return 0


def _label(args) -> int:
    s = ev.load_set(args.generation)
    lid = int(args.rest[2]) if len(args.rest) > 2 else None
    ev.label_item(s, args.rest[0], args.rest[1], lid)
    ev.save_set(s)
    print(f"labelled {args.rest[0]}: {args.rest[1]} lesson={lid}; {ev.census(s)}")
    return 0


def _freeze(args) -> int:
    s = ev.load_set(args.generation)
    c = ev.freeze(s)
    ev.save_set(s)
    print(f"frozen: {c}")
    if not c["positives_ok"]:
        print(f"WARNING: only {c['positives_lesson_wrong']} lesson_wrong positives; "
              f"the thresholds are not meaningful below {ev.MIN_POSITIVES}")
    return 0


async def _run(args) -> int:
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig

    s = ev.load_set(args.generation)
    cfg = [EndpointConfig(name="eval", url=args.url, provider="ollama", roles=["reasoning"],
                          priority=1, max_concurrent=1, context_tokens=32768)]
    router = LLMRouter(ModelsConfig(reasoning=args.model, embedding="unused"),
                       EndpointManager(cfg, LLMConfig()), pii_enabled=False, disable_fallback=True)
    result = await ev.run_judge(s, router, repeats=args.repeats)
    full, summary = ev.write_run(result, summary_dir=Path("benchmark_results"), git_sha=_sha())
    print(ev.render(result))
    print(f"full record: {full}\nsummary: {summary}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--generation", required=True); b.add_argument("--db")
    sh = sub.add_parser("show"); sh.add_argument("--generation", required=True)
    la = sub.add_parser("label"); la.add_argument("--generation", required=True)
    la.add_argument("rest", nargs="+", metavar="ITEM ATTRIBUTION [LESSON_ID]")
    fr = sub.add_parser("freeze"); fr.add_argument("--generation", required=True)
    ru = sub.add_parser("run"); ru.add_argument("--generation", required=True)
    ru.add_argument("--url", required=True); ru.add_argument("--model", default="qwen3:14b")
    ru.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        return asyncio.run(_build(args))
    if args.cmd == "show":
        return _show(args)
    if args.cmd == "label":
        return _label(args)
    if args.cmd == "freeze":
        return _freeze(args)
    if args.cmd == "run":
        return asyncio.run(_run(args))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
