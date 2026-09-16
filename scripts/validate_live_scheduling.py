"""Live check: does a chat turn own the local model while background work is eligible?

Boots a REAL Agent against a COPY of a database (default: the configured one),
with only the local Ollama endpoint, and instruments the OllamaGate singleton
so every acquisition records its priority, its wait, and whether a chat turn
was open when it was requested. Then: let background entity extraction start,
hold two chat turns while it is eligible, and confirm background work resumes
afterwards. The report answers, per turn: were any background calls granted
after the turn began, how long did the turn's own calls wait, did retrieval
stay semantic, and did recall work.

Real local inference: run it only when that load is appropriate (the GPU is
shared). Everything is written under --out (gitignored `data/` by default);
the source database is opened read-only and copied with the backup API.

    python -m scripts.validate_live_scheduling                # localhost Ollama
    python -m scripts.validate_live_scheduling --url http://<ollama-host>:11434

Evidence 2026-09-16 (dev box over Tailscale, copy of the dev DB):
docs/ACCEPTANCE_2026_09_16.md, "Follow-up 2026-09-16: scheduling under load".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MARKER_RE = re.compile(r"\bMARIGOLD[-‐-―−]742\b", re.IGNORECASE)
T0 = time.monotonic()
EVENTS: list[dict] = []
MARKS: dict[str, float] = {}
_ev_lock = threading.Lock()


def now() -> float:
    return round(time.monotonic() - T0, 3)


def mark(name: str) -> None:
    MARKS[name] = now()
    print(f"[{now():8.1f}s] {name}", flush=True)


def copy_db(source: Path, destination: Path) -> None:
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(destination) as dst:
            src.backup(dst)


def unextracted(db_path: Path) -> int:
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        return db.execute(
            "SELECT count(*) FROM memories WHERE entities_extracted_at IS NULL "
            "AND is_archived = 0 AND summary IS NOT NULL").fetchone()[0]


def build_config(out: Path, url: str) -> Path:
    """The production config, reduced to the local endpoint at `url`."""
    import yaml
    from blipshell.core.config import ConfigManager
    original = ConfigManager().load()
    base = original.model_copy(deep=True)
    base.endpoints = [e for e in base.endpoints if e.provider == "ollama" and not e.pii_sanitize]
    assert base.endpoints, "no non-relaying local Ollama endpoint in config.yaml"
    for ep in base.endpoints:
        ep.url = url
        ep.api_key = None
        ep.roles = list(set(ep.roles) | {"tool_calling", "coding"})
    base.models.tool_calling = original.models.tool_calling_fallback
    base.models.coding = original.models.coding_fallback
    base.pii.local_mode_default = True
    base.database.path = str(out / "agent.db")
    base.database.backup_dir = str(out / "backups")
    base.benchmark.db_path = str(out / "benchmark.db")
    base.robotics.enabled = False
    base.telegram.enabled = False
    base.reflection.enabled = False
    raw = base.model_dump()
    for field in ("token", "bot_token", "api_key"):
        if field in raw.get("telegram", {}):
            raw["telegram"][field] = ""
    path = out / "agent.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def instrument_gate():
    from blipshell.llm.ollama_gate import get_gate
    gate = get_gate()
    orig_acquire, orig_async_acquire, orig_release = gate.acquire, gate.async_acquire, gate.release

    def record(kind, priority, t_req, turn_at_request):
        with _ev_lock:
            EVENTS.append({
                "kind": kind, "priority": priority, "t_req": t_req, "t_grant": now(),
                "wait_ms": round((now() - t_req) * 1000, 1),
                "turn_at_request": turn_at_request,
                "thread": threading.current_thread().name,
            })

    def acquire(priority=gate.BACKGROUND, timeout=None):
        t_req, turn = now(), gate.interactive_active
        ok = orig_acquire(priority, timeout)
        record("sync", priority, t_req, turn)
        return ok

    async def async_acquire(priority=gate.BACKGROUND, timeout=None):
        t_req, turn = now(), gate.interactive_active
        ok = await orig_async_acquire(priority, timeout)
        record("async", priority, t_req, turn)
        return ok

    def release():
        with _ev_lock:
            EVENTS.append({"kind": "release", "t_grant": now(),
                           "thread": threading.current_thread().name})
        orig_release()

    gate.acquire, gate.async_acquire, gate.release = acquire, async_acquire, release
    return gate


def grants(priority, *, requested_after=None, requested_before=None,
           granted_before=None, granted_after=None):
    out = []
    for e in EVENTS:
        if e["kind"] == "release" or e["priority"] != priority:
            continue
        if requested_after is not None and e["t_req"] < requested_after:
            continue
        if requested_before is not None and e["t_req"] > requested_before:
            continue
        if granted_before is not None and e["t_grant"] > granted_before:
            continue
        if granted_after is not None and e["t_grant"] < granted_after:
            continue
        out.append(e)
    return out


async def wait_until(cond, timeout, label):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(1.0)
    print(f"[{now():8.1f}s] WAIT TIMEOUT: {label}", flush=True)
    return False


async def run(out: Path, url: str, idle_interval: float) -> dict:
    from blipshell.core.agent import Agent
    from blipshell.core.config import ConfigManager
    from blipshell.memory import worker as worker_mod
    from blipshell.llm.ollama_gate import BACKGROUND, INTERACTIVE

    worker_mod._IDLE_EXTRACT_INTERVAL = idle_interval   # idle extraction eligible quickly
    gate = instrument_gate()
    db_path = out / "agent.db"
    report: dict = {"url": url, "started_at": datetime.now(timezone.utc).isoformat(),
                    "unextracted_at_start": unextracted(db_path)}

    cm = ConfigManager(build_config(out, url))
    agent = Agent(cm.load(), cm)
    try:
        mark("initialize_start")
        await asyncio.wait_for(agent.initialize(), 300)
        mark("initialize_done")
        sid = await asyncio.wait_for(agent.start_session(), 300)
        report["session_id"] = sid
        mark("session_started")

        # Background must genuinely be running before a turn proves anything:
        # at least two BACKGROUND grants, and startup's own main-thread model
        # work settled (no INTERACTIVE grant for 15 s).
        def background_live():
            bg = grants(BACKGROUND)
            fg = grants(INTERACTIVE)
            quiet = not fg or (now() - fg[-1]["t_grant"]) > 15.0
            return len(bg) >= 2 and quiet
        report["background_live_before_turns"] = await wait_until(
            background_live, 180, "background extraction to start")
        report["unextracted_before_turns"] = unextracted(db_path)
        report["background_grants_before_turns"] = len(grants(BACKGROUND))

        prompts = [
            "This is an isolated scheduling test. Remember this project fact: "
            "the Lumen Orchard project uses release code MARIGOLD-742. Briefly acknowledge.",
            "What is the release code for the Lumen Orchard project? Answer briefly.",
        ]
        turns = []
        for i, prompt in enumerate(prompts, 1):
            mark(f"turn{i}_start")
            t_start = now()
            reply = await asyncio.wait_for(agent.chat(prompt), 600)
            t_end = now()
            mark(f"turn{i}_end")
            stats = dict(agent.search.last_search_stats or {})
            bg_started_in_turn = grants(BACKGROUND, requested_after=t_start, granted_before=t_end)
            bg_inflight = grants(BACKGROUND, requested_before=t_start,
                                 granted_after=t_start, granted_before=t_end)
            fg_in_turn = grants(INTERACTIVE, requested_after=t_start, granted_before=t_end)
            turns.append({
                "prompt": prompt,
                "reply_preview": reply[:200],
                "elapsed_s": round(t_end - t_start, 1),
                "contains_marker": bool(MARKER_RE.search(reply)),
                "search_stats": stats,
                "background_grants_requested_during_turn": len(bg_started_in_turn),
                "background_grants_inflight_from_before_turn": len(bg_inflight),
                "interactive_grants": len(fg_in_turn),
                "interactive_max_wait_ms": max([e["wait_ms"] for e in fg_in_turn], default=0),
                "interactive_waits_ms": [e["wait_ms"] for e in fg_in_turn],
                "endpoint": getattr(agent, "last_endpoint_used", None),
            })
            print(json.dumps({k: v for k, v in turns[-1].items()
                              if k not in ("prompt", "reply_preview", "search_stats")}), flush=True)
        report["turns"] = turns

        # Deferred background work must resume once the turns are over.
        t_after = MARKS["turn2_end"]
        report["background_resumed_after_turns"] = await wait_until(
            lambda: len(grants(BACKGROUND, requested_after=t_after)) >= 2, 240,
            "background work to resume after the turns")
        report["background_grants_after_turns"] = len(grants(BACKGROUND, requested_after=t_after))
        report["unextracted_after"] = unextracted(db_path)
        mark("observation_done")
    finally:
        mark("cleanup_start")
        await agent.force_cleanup()
        mark("cleanup_done")
    report["marks"] = MARKS
    report["gate_stats"] = gate.get_stats()
    return report


def verdict(report: dict) -> list[str]:
    """Named misses, empty when the scheduling contract held."""
    misses = []
    if not report.get("background_live_before_turns"):
        misses.append("background work never started before the turns (nothing was tested)")
    for i, t in enumerate(report.get("turns", []), 1):
        if t["background_grants_requested_during_turn"]:
            misses.append(f"turn {i}: {t['background_grants_requested_during_turn']} background call(s) "
                          "granted after the turn began")
        if t["search_stats"].get("chroma_hits", 0) == 0:
            misses.append(f"turn {i}: no semantic hits (keyword fallback or empty vector search)")
        if i > 1 and not t["contains_marker"]:
            misses.append(f"turn {i}: planted fact not recalled")
    if not report.get("background_resumed_after_turns"):
        misses.append("background work did not resume after the turns")
    signals = report.get("log_signals", {})
    for key in ("semantic_search_unavailable", "core_memory_search_failed",
                "lesson_search_failed", "memory_search_failed"):
        if signals.get(key):
            misses.append(f"log: {key} x{signals[key]}")
    return misses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:11434",
                        help="Ollama base URL for the single local endpoint (default: localhost)")
    parser.add_argument("--source-db", default=None,
                        help="database to COPY for the run (default: the configured database)")
    parser.add_argument("--out", default=None,
                        help="run folder (default: data/live_scheduling_<timestamp>/)")
    parser.add_argument("--idle-interval", type=float, default=3.0,
                        help="worker idle-extraction interval in seconds for the run (default 3)")
    args = parser.parse_args(argv)

    if args.source_db:
        source = Path(args.source_db)
    else:
        from blipshell.core.config import ConfigManager
        source = Path(ConfigManager().load().database.path)
    out = Path(args.out) if args.out else REPO / "data" / ("live_scheduling_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    copy_db(source, out / "agent.db")
    with sqlite3.connect(out / "agent.db") as db:   # defer startup tag discovery, as the acceptance run did
        db.execute("INSERT OR REPLACE INTO app_metadata(key,value) VALUES (?,?)",
                   ("last_tag_discovery", datetime.now(timezone.utc).isoformat()))

    log_path = out / "run.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
                        encoding="utf-8")
    logging.getLogger("blipshell.llm.ollama_gate").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    report = asyncio.run(run(out, args.url, args.idle_interval))

    log = log_path.read_text(encoding="utf-8", errors="replace")
    report["log_signals"] = {
        "semantic_search_unavailable": log.count("Semantic search unavailable"),
        "core_memory_search_failed": log.count("Core memory search failed"),
        "lesson_search_failed": log.count("Lesson search failed"),
        "memory_search_failed": log.count("Memory search failed"),
        "timeout_mentions": len(re.findall(r"timed out|Timeout|timeout", log)),
        "gate_waits": log.count("OllamaGate: waiting"),
    }
    report["misses"] = verdict(report)
    (out / "events.json").write_text(json.dumps(EVENTS, indent=1), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== REPORT ===")
    print(json.dumps({k: v for k, v in report.items() if k not in ("turns", "url")}, indent=2, default=str))
    for i, t in enumerate(report.get("turns", []), 1):
        print(f"--- turn {i} ---")
        print(json.dumps({k: v for k, v in t.items() if k not in ("prompt", "reply_preview")},
                         indent=2, default=str))
    print("\nVERDICT:", "PASS" if not report["misses"] else "FAIL")
    for miss in report["misses"]:
        print(" -", miss)
    print("run folder:", out)
    return 0 if not report["misses"] else 1


if __name__ == "__main__":
    sys.exit(main())
