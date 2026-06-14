"""Standalone retroactive entity-merge runner (no nightly timeout).

The `merge_entities` nightly job scans the WHOLE entity graph (~31K entities,
each a vector similarity search plus possible edge-aware LLM arbitration in the
0.80-0.90 band). That structurally cannot finish inside the nightly per-job cap
(_JOB_TIMEOUT, 300s) — `blipshell nightly --job merge_entities` will always
time out on a real graph.

This script runs the exact same job logic via NightlyRunner.run_job's
underlying handler, but with NO timeout wrapper, and prints a progress
heartbeat so a multi-minute run is distinguishable from a hang. It honors the
config flags (entity_merge_enabled / entity_merge_dry_run / thresholds) — flip
entity_merge_dry_run to false in config.yaml only after reviewing the dry-run
preview file it writes.

Recommended order on the Ollama PC:
    cleanup_entities -> entity_extraction -> merge (dry-run) -> prune (dry-run)

Usage:
    python -m scripts.merge_entities                 # dry-run per config
    python -m scripts.merge_entities --local         # force local Ollama
    python -m scripts.merge_entities --config path/to/config.yaml
"""

from __future__ import annotations

import asyncio
import logging
import time

import click

logger = logging.getLogger(__name__)


async def run_merge(config_path: str | None = None, local: bool = False) -> dict:
    """Run one retroactive merge pass with no per-job timeout. Returns stats."""
    from blipshell.core.nightly import NightlyRunner

    runner = await NightlyRunner.create_from_config(config_path, local_only=local)
    started = time.monotonic()

    def on_status(msg: str):
        click.echo(msg)
        logger.info(msg)

    def on_progress(scanned: int, total: int, stats: dict):
        elapsed = time.monotonic() - started
        click.echo(
            f"  scanned {scanned}/{total} "
            f"({scanned / total * 100:.0f}%) — "
            f"auto={stats['auto_merges']} llm={stats['llm_merges']} "
            f"rejected={stats['llm_rejects']} — {elapsed:.0f}s elapsed"
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


@click.command()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.option("--local", is_flag=True, help="Force all LLM calls through local Ollama")
def main(config_path: str | None, local: bool):
    """Run the retroactive entity merge with no nightly timeout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_merge(config_path, local))

    click.echo("\n=== Merge pass result ===")
    for k, v in result.items():
        click.echo(f"  {k}: {v}")

    if result.get("skipped"):
        click.echo(
            "\nJob skipped. To enable, set entity_merge_enabled: true in config.yaml "
            "(keep entity_merge_dry_run: true for the first preview run)."
        )
    elif result.get("dry_run"):
        click.echo(
            "\nDry run — nothing changed. Review the preview file above, then set "
            "entity_merge_dry_run: false in config.yaml and re-run to apply."
        )


if __name__ == "__main__":
    main()
