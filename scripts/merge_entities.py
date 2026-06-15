"""Standalone retroactive entity-merge runner (no nightly timeout).

The `merge_entities` nightly job scans the WHOLE entity graph (~36K entities,
each a vector similarity search plus possible edge-aware LLM arbitration in the
0.80-0.90 band). That scan can't finish inside the nightly per-job cap
(_JOB_TIMEOUT, 300s), so `blipshell nightly --job merge_entities` only ever
makes partial progress. This script runs it uncapped with a progress heartbeat.

Two modes:
  (default) DRY-RUN SCAN — full pass, writes the complete confidence-sorted
    plan to <db_dir>/entity_merge_preview.json. Changes nothing. Honors the
    config flags (entity_merge_enabled / entity_merge_dry_run / thresholds).
  --apply-preview — apply a reviewed preview file DIRECTLY, with no re-scan
    (minutes, not hours) and applying exactly what you reviewed. Merges are
    reversible (loser is soft-archived, never deleted) and idempotent (safe to
    re-run). The version/instance guard runs again at apply time, so a preview
    generated before the guard existed can't apply v1/v2-style false merges.

Usage:
    python -m scripts.merge_entities                  # dry-run scan -> preview
    python -m scripts.merge_entities --local          # force local Ollama
    python -m scripts.merge_entities --apply-preview  # apply the preview file
    python -m scripts.merge_entities --apply-preview --preview-file path.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def _default_preview_path(runner) -> Path:
    return Path(runner.config.database.path).parent / "entity_merge_preview.json"


async def run_scan(config_path: str | None = None, local: bool = False) -> dict:
    """Full uncapped dry-run/apply scan via the nightly handler. Returns stats."""
    from blipshell.core.nightly import NightlyRunner

    runner = await NightlyRunner.create_from_config(config_path, local_only=local)
    started = time.monotonic()

    def on_status(msg: str):
        click.echo(msg)
        logger.info(msg)

    def on_progress(scanned: int, total: int, stats: dict):
        elapsed = time.monotonic() - started
        click.echo(
            f"  scanned {scanned}/{total} ({scanned / max(total, 1) * 100:.0f}%) — "
            f"auto={stats['auto_merges']} llm={stats['llm_merges']} "
            f"rejected={stats['llm_rejects']} version_skips={stats['version_skips']} "
            f"— {elapsed:.0f}s elapsed"
        )

    try:
        # time_budget_seconds=None → uncapped full pass (complete preview),
        # unlike the scheduled nightly which budgets to fit _JOB_TIMEOUT.
        result = await runner._job_merge_entities(
            on_status, on_progress=on_progress, time_budget_seconds=None,
        )
    finally:
        await runner.close()

    result["elapsed_s"] = round(time.monotonic() - started, 1)
    return result


async def run_apply_preview(
    config_path: str | None = None, local: bool = False,
    preview_file: str | None = None,
) -> dict:
    """Apply a reviewed preview file directly — no re-scan."""
    from blipshell.core.nightly import NightlyRunner
    from blipshell.memory.entity_merger import EntityMerger

    runner = await NightlyRunner.create_from_config(config_path, local_only=local)
    started = time.monotonic()

    def on_status(msg: str):
        click.echo(msg)
        logger.info(msg)

    try:
        cfg = runner.config.memory
        path = Path(preview_file) if preview_file else _default_preview_path(runner)
        if not path.exists():
            return {"error": f"preview file not found: {path}"}
        plan = json.loads(path.read_text(encoding="utf-8"))
        click.echo(f"Applying {len(plan)} merges from {path} (no re-scan)...")

        merger = EntityMerger(
            runner.sqlite, runner.router, runner.vectors,
            auto_threshold=cfg.entity_merge_auto_threshold,
            llm_threshold=cfg.entity_merge_llm_threshold,
            max_candidates=cfg.entity_merge_max_candidates,
            edge_sample=cfg.entity_merge_edge_sample,
        )
        result = await merger.apply_plan(plan, on_status=on_status)
    finally:
        await runner.close()

    result["elapsed_s"] = round(time.monotonic() - started, 1)
    return result


@click.command()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.option("--local", is_flag=True, help="Force all LLM calls through local Ollama")
@click.option("--apply-preview", is_flag=True,
              help="Apply the reviewed preview file directly (no re-scan)")
@click.option("--preview-file", default=None,
              help="Path to the preview JSON (default: <db_dir>/entity_merge_preview.json)")
def main(config_path, local, apply_preview, preview_file):
    """Run the retroactive entity merge with no nightly timeout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if apply_preview:
        result = asyncio.run(run_apply_preview(config_path, local, preview_file))
        click.echo("\n=== Apply result ===")
        for k, v in result.items():
            click.echo(f"  {k}: {v}")
        if result.get("error"):
            return
        click.echo(
            f"\nApplied {result['merged']} merges "
            f"({result['guard_skipped']} blocked by version guard, "
            f"{result['skipped']} skipped as already-merged). "
            f"All losers soft-archived (is_archived=1) — recoverable, not deleted."
        )
        return

    result = asyncio.run(run_scan(config_path, local))
    click.echo("\n=== Merge pass result ===")
    for k, v in result.items():
        if k in ("sample", "plan"):
            continue  # too large to dump; full plan is in the preview file
        click.echo(f"  {k}: {v}")

    if result.get("skipped"):
        click.echo(
            "\nJob skipped. To enable, set entity_merge_enabled: true in config.yaml "
            "(keep entity_merge_dry_run: true for the first preview run)."
        )
    elif result.get("dry_run"):
        click.echo(
            "\nDry run — nothing changed. Review the preview file, then apply it with:\n"
            "    python -m scripts.merge_entities --apply-preview"
        )


if __name__ == "__main__":
    main()
