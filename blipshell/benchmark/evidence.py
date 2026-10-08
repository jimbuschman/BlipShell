"""Load adjacent agent and continuity evidence into the unified report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_system_evidence(repo_root: str | Path) -> dict:
    repo_root = Path(repo_root)
    evidence = {"agent_eval": [], "continuity": []}

    summaries = sorted((repo_root / "agent_eval_results").glob("*/SUMMARY.json"))
    if summaries:
        data = _read(summaries[-1]) or {}
        for result in data.get("results") or []:
            if not isinstance(result, dict) or not result.get("model"):
                continue
            evidence["agent_eval"].append({
                "model": result["model"],
                "score": result.get("score"),
                "max": result.get("max"),
                "silent": result.get("silent"),
                "turn_limit": result.get("hit_turn_limit"),
                "api_errors": result.get("api_errors"),
                "seconds_per_episode": result.get("avg_s_per_episode"),
                "run_date": data.get("run_date"),
            })

    result_root = repo_root / "benchmark_results"
    files = list(result_root.glob("simulate_continuity__*.json"))
    files += list((result_root / "continuity" / "behavioural").glob(
        "simulate_continuity__*.json"))
    totals: dict[str, dict] = defaultdict(lambda: {"scored": 0, "passed": 0, "failed": 0})
    for path in sorted(set(files)):
        data = _read(path) or {}
        for scenario in data.get("scenarios") or []:
            if scenario.get("name") == "__require_model__":
                continue
            for step in scenario.get("steps") or []:
                used = step.get("model_used") or {}
                model = used.get("model") if isinstance(used, dict) else None
                if not model or step.get("outcome") not in (None, "scored"):
                    continue
                totals[model]["scored"] += 1
                passed = scenario.get("status") == "pass" and not step.get("soft_failures")
                totals[model]["passed" if passed else "failed"] += 1
    evidence["continuity"] = [
        {"model": model, **counts} for model, counts in sorted(totals.items())
    ]
    return evidence
