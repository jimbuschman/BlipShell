"""The predefined production-model validation batch for the V3 Stage E gate.

ONE batch, fixed in advance, so the result cannot be steered by rerunning:

    N=5 sequential runs of `blipshell simulate -c continuity`, each run its
    own JSON in benchmark_results/, `--require-model` set to the production
    chat model for project mode (config `models.coding`, minimax-m3 at the
    time of writing). Every chat step served by any other model is reported
    `blocked`, never scored. The batch stops early if the pre-flight finds
    the model cannot be served here (no enabled endpoint, no credentials).

What counts as DONE (decided before running):
  - 5 run files exist, every chat step `outcome == scored`, served model ==
    the required one, scorer_version == the current one;
  - the printed aggregate shows per scenario PASS count out of 5 and the
    named-miss frequencies. A scenario "holds" when it passes 4/5 or better.
What counts as BLOCKED: the pre-flight refuses (no credentials/endpoint) or
any chat step is `blocked`. What counts as FAILED: a scenario below 4/5, or
any `timeout`/`error` step. Report failures with the misses; do not extend
the batch or rerun to chase a pass.

    python -m scripts.run_gate_batch [--runs 5] [--model minimax/minimax-m3] [--config-path PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmark_results"
HOLD_THRESHOLD = 4  # of 5


def run_once(index: int, model: str, config_path: str | None) -> Path | None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RESULTS / f"simulate_continuity__batch{index}__{ts}.json"
    cmd = [sys.executable, "-m", "blipshell.ui.cli"]
    if config_path:
        cmd += ["--config-path", config_path]
    cmd += ["simulate", "-c", "continuity", "--require-model", model, "--output", str(out)]
    print(f"=== run {index}: {' '.join(cmd[3:])}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        print(f"    process exit {proc.returncode}")
    return out if out.exists() else None


def aggregate(files: list[Path]) -> dict:
    status = defaultdict(list)
    misses = defaultdict(Counter)
    outcomes = Counter()
    served = Counter()
    blocked_reason = None
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for scn in d["scenarios"]:
            if scn["name"] == "__require_model__":
                blocked_reason = scn.get("error")
                continue
            status[scn["name"]].append(scn["status"])
            for st in scn["steps"]:
                outcomes[st.get("outcome", "scored")] += 1
                served[(st.get("model_used") or {}).get("model")] += 1
                for m in st.get("soft_failures", []):
                    misses[scn["name"]][m.split(":")[0]] += 1
    return {"status": status, "misses": misses, "outcomes": outcomes, "served": served, "blocked": blocked_reason}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", default=None, help="default: config models.coding (project-mode chat)")
    ap.add_argument("--config-path", default=None)
    args = ap.parse_args(argv)

    model = args.model
    if model is None:
        from blipshell.core.config import ConfigManager
        model = ConfigManager(args.config_path).load().models.coding

    from blipshell.core.config import ConfigManager
    from blipshell.simulate.runner import required_model_blocker
    blocker = required_model_blocker(ConfigManager(args.config_path).load(), model)
    if blocker:
        print(f"BLOCKED before running: {blocker}")
        print("Nothing was run. Run this batch where the model is servable (the Ollama PC).")
        return 2

    files = []
    for i in range(1, args.runs + 1):
        out = run_once(i, model, args.config_path)
        if out is None:
            print(f"    run {i} produced no result file - stopping the batch (infrastructure)")
            break
        files.append(out)
        agg = aggregate([out])
        if agg["blocked"] or agg["outcomes"].get("blocked"):
            print(f"    BLOCKED: {agg['blocked'] or 'a chat step was served by another model'} - stopping the batch")
            break

    if not files:
        return 2
    agg = aggregate(files)
    print(f"\n=== batch of {len(files)} run(s), required model {model!r}")
    print("served by:", dict(agg["served"]), "| outcomes:", dict(agg["outcomes"]))
    verdict = "PASS"
    for name, statuses in agg["status"].items():
        passes = sum(1 for s in statuses if s == "pass")
        hold = passes >= HOLD_THRESHOLD if len(statuses) >= 5 else None
        print(f"  {name}: {passes}/{len(statuses)} pass  {'holds' if hold else ('FAILS' if hold is False else 'incomplete')}")
        for k, c in agg["misses"][name].most_common():
            print(f"      {c}/{len(statuses)}  {k}")
        if hold is False or hold is None:
            verdict = "FAIL" if hold is False else ("INCOMPLETE" if verdict == "PASS" else verdict)
    if agg["outcomes"].get("blocked") or agg["blocked"]:
        verdict = "BLOCKED"
    elif agg["outcomes"].get("timeout") or agg["outcomes"].get("error"):
        verdict = "FAIL"
    print(f"\nBATCH VERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
