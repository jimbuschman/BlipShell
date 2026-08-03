"""One-shot: move benchmark results out of the gitignored DB into committed files.

Run this ONCE on whichever machine has real benchmark history (the Ollama PC —
the dev box's benchmark.db has zero result rows). It reads `benchmark_runs` from
`data/benchmark.db`, groups rows by run, and writes one JSON file per run into
`benchmark_results/`. Then commit that directory and both machines share the
history for good.

Existing files are never overwritten unless --force is passed, so re-running is
safe. The source DB is opened read-only and never modified — if the migration
looks wrong you can delete the output directory and start over.

Provenance note: migrated runs get `"git_sha": null` and `"migrated_from_db":
true`. We genuinely do not know which commit produced them, and guessing would
be worse than admitting it — the report's provenance table will show them as
unknown-code, which is the honest signal that they predate the current scorers.

Usage:
    python -m scripts.migrate_benchmark_results --dry-run
    python -m scripts.migrate_benchmark_results
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.benchmark.results import (  # noqa: E402
    SCHEMA_VERSION,
    _compact_ts,
    rows_from_legacy_db,
    slugify_model,
)

ROW_FIELDS = ("suite", "task_type", "metric", "value", "unit", "raw")


def group_runs(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group legacy rows into runs, keyed by (run_group, model).

    run_group already ties one `benchmark run` invocation together, so it is the
    natural unit. Model is included in the key because a malformed or
    hand-inserted row set could in principle share a group across models, and
    silently merging two models' scores into one file would be a data error we
    could never detect afterwards.
    """
    runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        group = r.get("run_group") or ""
        model = r.get("model") or ""
        if not model:
            continue
        runs[(group, model)].append(r)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default="data/benchmark.db")
    ap.add_argument("--out", default="benchmark_results")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be written, touch nothing")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite result files that already exist")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"ERROR: no benchmark DB at {db}")
        print("Nothing to migrate. If this is the dev box, that is expected --")
        print("run this on the machine that actually ran the benchmarks.")
        return 1

    rows = rows_from_legacy_db(db)
    if not rows:
        print(f"No benchmark_runs rows in {db} -- nothing to migrate.")
        print("(The table may already have been dropped, or this box never ran one.)")
        return 0

    runs = group_runs(rows)
    out_dir = Path(args.out)
    print(f"Found {len(rows)} rows across {len(runs)} run(s) in {db}")
    print(f"Target: {out_dir}{'  (DRY RUN)' if args.dry_run else ''}")
    print()

    written = skipped = 0
    for (group, model), grows in sorted(runs.items(), key=lambda kv: kv[0][0]):
        run_ts = next((r.get("run_ts") for r in grows if r.get("run_ts")), "") or ""
        tier = next((r.get("tier") for r in grows if r.get("tier")), "deep") or "deep"
        fname = f"{_compact_ts(run_ts)}__{slugify_model(model)}.json"
        path = out_dir / fname

        if path.exists() and not args.force:
            print(f"  SKIP  {fname}  (exists; --force to overwrite)")
            skipped += 1
            continue

        payload = {
            "schema": SCHEMA_VERSION,
            "model": model,
            "run_group": group,
            "run_ts": run_ts,
            "tier": tier,
            "judge_model": None,       # not recorded in the legacy schema
            "jobs": None,
            "git_sha": None,           # unknowable for migrated runs -- see docstring
            "host": None,
            "migrated_from_db": True,
            "rows": [{k: r.get(k) for k in ROW_FIELDS} for r in grows],
        }

        print(f"  WRITE {fname}  model={model} rows={len(grows)} ts={run_ts[:16]}")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        written += 1

    print()
    if args.dry_run:
        print(f"DRY RUN: would write {written} file(s), skip {skipped}.")
        print("Re-run without --dry-run to apply.")
    else:
        print(f"Wrote {written} file(s), skipped {skipped}.")
        print()
        print("Next:")
        print(f"  git add {out_dir}")
        print("  git commit -m \"Benchmark: migrate stored results to committed files\"")
        print("  blipshell benchmark report      # rebuild report.md from them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
