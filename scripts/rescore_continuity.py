"""Versioned rescoring of preserved gate runs (V3 Stage E).

Originals in benchmark_results/simulate_continuity__*.json are never
rewritten. This reads their stored replies and tool calls, applies the
CURRENT scorers (SCORER_VERSION), and writes ONE report:

    benchmark_results/rescore_continuity__v<N>__<ts>.json

per run and scenario: the original misses (as scored at the time), the
rescored misses, and an explicit validity classification -
`scored` | `timeout` | `error` | `blocked` | `infrastructure_invalid` -
so an infrastructure-invalid run, a model failure and a timeout are never
read as the same thing. Invalidated runs can be passed with --invalid
PATH=REASON to appear in the report under `infrastructure_invalid`.

    python -m scripts.rescore_continuity [--invalid PATH=REASON ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from blipshell.simulate.scenarios import continuity as sc

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmark_results"

_SCENARIOS = {s.name: s for s in sc.get_scenarios()}
SCORERS = {name: s.steps[0].response_validator for name, s in _SCENARIOS.items()}
DISCUSSION = {name: s.steps[0].expect_no_write_tools for name, s in _SCENARIOS.items()}


def _sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10, cwd=REPO).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def classify(step: dict) -> str:
    if step.get("outcome") in ("timeout", "error", "blocked"):
        return step["outcome"]
    err = step.get("error") or ""
    if err.startswith("Step timed out"):
        return "timeout"
    if err.startswith("blocked:"):
        return "blocked"
    if err:
        return "error"
    return "scored"


def rescore_step(scenario_name: str, step: dict) -> list[str]:
    scorer = SCORERS.get(scenario_name)
    misses = list(scorer(step.get("response") or "")) if scorer else []
    written = [t for t in step.get("tools_called", []) if t in sc.WRITE_TOOLS]
    if written and DISCUSSION.get(scenario_name, True):
        misses.append(f"acted during a discussion turn: {', '.join(written)}")
    return misses


def rescore_file(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {"file": path.name, "git_sha": d.get("git_sha"), "run_ts": d.get("run_ts"),
           "original_scorer_version": d.get("scorer_version", 1), "scenarios": []}
    for scn in d.get("scenarios", []):
        for st in scn.get("steps", []):
            validity = classify(st)
            entry = {
                "scenario": scn["name"], "validity": validity,
                "model_used": (st.get("model_used") or {}).get("model"),
                "original_misses": list(st.get("soft_failures", [])),
                "original_status": scn.get("status"),
            }
            if validity == "scored":
                entry["rescored_misses"] = rescore_step(scn["name"], st)
                entry["rescored_status"] = "warn" if entry["rescored_misses"] else "pass"
            else:
                entry["rescored_misses"] = None
                entry["rescored_status"] = validity
                entry["error"] = st.get("error")
            out["scenarios"].append(entry)
    return out


def summarize(runs: list[dict]) -> dict:
    per: dict[str, dict] = {}
    for r in runs:
        for e in r["scenarios"]:
            s = per.setdefault(e["scenario"], {"scored": 0, "original_pass": 0, "rescored_pass": 0,
                                                "not_scored": {}, "rescored_miss_counts": {}})
            if e["validity"] != "scored":
                s["not_scored"][e["validity"]] = s["not_scored"].get(e["validity"], 0) + 1
                continue
            s["scored"] += 1
            s["original_pass"] += int(e["original_status"] == "pass")
            s["rescored_pass"] += int(e["rescored_status"] == "pass")
            for m in e["rescored_misses"]:
                key = m.split(":")[0]
                s["rescored_miss_counts"][key] = s["rescored_miss_counts"].get(key, 0) + 1
    return per


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--invalid", action="append", default=[], metavar="PATH=REASON",
                    help="an invalidated run to list under infrastructure_invalid (not rescored)")
    ap.add_argument("--glob", default="simulate_continuity__*.json")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(RESULTS / args.glob)))
    runs = [rescore_file(Path(p)) for p in files]
    invalid = []
    for spec in args.invalid:
        path, _, reason = spec.partition("=")
        invalid.append({"file": os.path.basename(path), "validity": "infrastructure_invalid", "reason": reason})

    report = {
        "kind": "behavioural_rescore",
        "scorer_version": sc.SCORER_VERSION,
        "git_sha": _sha(), "host": platform.node(),
        "run_ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "note": "originals untouched; rescored with the current scorers over the stored replies and tool calls",
        "runs": runs,
        "infrastructure_invalid": invalid,
        "summary": summarize(runs),
    }
    out = RESULTS / f"rescore_continuity__v{sc.SCORER_VERSION}__{report['run_ts']}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"rescored {len(runs)} run file(s) with scorer v{sc.SCORER_VERSION} -> {out.name}")
    for name, s in report["summary"].items():
        print(f"  {name}: scored {s['scored']}, pass {s['original_pass']} -> {s['rescored_pass']} (v1 -> v{sc.SCORER_VERSION});"
              f" not scored {s['not_scored'] or '-'}")
        for k, c in sorted(s["rescored_miss_counts"].items(), key=lambda kv: -kv[1]):
            print(f"      {c}/{s['scored']}  {k}")
    for i in invalid:
        print(f"  infrastructure_invalid: {i['file']} - {i['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
