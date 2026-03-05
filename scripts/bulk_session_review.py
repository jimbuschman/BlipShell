"""Bulk session review — process all sessions missing lessons/reflections.

Unlike the nightly job (capped at 50/20 per run), this processes ALL
unreviewed sessions in one shot. Useful for clearing the backlog after
import or after adding the session_review role.

Usage:
    python -m scripts.bulk_session_review [--lessons] [--reflections] [--limit N] [--config PATH]
    blipshell review [--lessons] [--reflections] [--limit N]
"""

from __future__ import annotations

import asyncio
import logging
import time

import click

from blipshell.memory.manager import estimate_tokens

logger = logging.getLogger(__name__)


async def run_bulk_review(
    config_path: str | None = None,
    do_lessons: bool = True,
    do_reflections: bool = True,
    limit: int | None = None,
    on_status=None,
) -> dict:
    """Process all sessions missing lessons and/or reflections.

    Returns stats dict with counts and timing.
    """
    from blipshell.core.nightly import NightlyRunner

    runner = await NightlyRunner.create_from_config(config_path)
    processor = runner.processor
    sqlite = runner.sqlite
    stats = {"lessons": None, "reflections": None, "elapsed_s": 0}
    started = time.monotonic()

    def _status(msg: str):
        if on_status:
            on_status(msg)
        logger.info(msg)

    try:
        if do_lessons:
            sessions = await sqlite.get_sessions_missing_lessons(limit=limit or 10000)
            _status(f"Lessons: {len(sessions)} sessions to process")

            processed = 0
            failed = 0
            skipped = 0
            cloud_routed = 0

            for i, session in enumerate(sessions):
                sid = session["id"]
                project = session.get("project")
                try:
                    messages = await sqlite.get_session_messages_for_lesson(sid)
                    if not messages:
                        skipped += 1
                        continue

                    conversation_lines = [f"{m['role']}: {m['content']}" for m in messages]
                    conversation_text = "\n".join(conversation_lines)
                    tokens = estimate_tokens(conversation_text)

                    # Route large sessions to bigger-context endpoint
                    min_ctx = tokens + 4096 if tokens > 28000 else None
                    if min_ctx:
                        cloud_routed += 1

                    await processor.process_lesson(
                        conversation_text, sid, project=project,
                        min_context_tokens=min_ctx,
                    )
                    processed += 1
                    if (i + 1) % 10 == 0:
                        _status(f"Lessons: {i+1}/{len(sessions)} ({processed} done, {failed} failed)")
                except Exception as e:
                    logger.error("Lesson extraction failed for session %d: %s", sid, e)
                    failed += 1

            stats["lessons"] = {
                "processed": processed,
                "failed": failed,
                "skipped": skipped,
                "cloud_routed": cloud_routed,
                "total": len(sessions),
            }

        if do_reflections:
            sessions = await sqlite.get_sessions_missing_reflections(limit=limit or 10000)
            _status(f"Reflections: {len(sessions)} sessions to process")

            processed = 0
            failed = 0
            skipped = 0
            cloud_routed = 0

            for i, session in enumerate(sessions):
                sid = session["id"]
                summary = session["summary"]
                project = session.get("project")
                try:
                    chunks, total_tokens = await processor.prepare_conversation_for_reflection(
                        sid, summary,
                    )
                    min_ctx = total_tokens + 4096 if total_tokens > 28000 else None
                    if min_ctx:
                        cloud_routed += 1

                    result = await processor.process_reflection(
                        session_id=sid,
                        session_summary=summary,
                        conversation_chunks=chunks,
                        project=project,
                        min_context_tokens=min_ctx,
                    )
                    if result is None:
                        skipped += 1
                    else:
                        processed += 1
                    if (i + 1) % 10 == 0:
                        _status(f"Reflections: {i+1}/{len(sessions)} ({processed} done, {skipped} skipped)")
                except Exception as e:
                    logger.error("Session reflection failed for session %d: %s", sid, e)
                    failed += 1

            stats["reflections"] = {
                "processed": processed,
                "failed": failed,
                "skipped": skipped,
                "cloud_routed": cloud_routed,
                "total": len(sessions),
            }

    finally:
        stats["elapsed_s"] = round(time.monotonic() - started, 1)
        await runner.close()

    return stats


@click.command()
@click.option("--lessons/--no-lessons", default=True, help="Extract lessons from sessions")
@click.option("--reflections/--no-reflections", default=True, help="Generate session reflections")
@click.option("--limit", type=int, default=None, help="Max sessions to process (default: all)")
@click.option("--config", "config_path", default=None, help="Config file path")
@click.option("--quiet", "-q", is_flag=True, help="JSON output only")
def main(lessons, reflections, limit, config_path, quiet):
    """Bulk process all sessions missing lessons and/or reflections."""
    import json

    async def _run():
        if quiet:
            result = await run_bulk_review(
                config_path=config_path,
                do_lessons=lessons,
                do_reflections=reflections,
                limit=limit,
            )
            print(json.dumps(result, indent=2, default=str))
        else:
            from rich.console import Console
            from rich.status import Status

            console = Console()
            with Status("[bold cyan]Processing sessions...", console=console) as status:
                def on_status(msg):
                    status.update(f"[bold cyan]{msg}")

                result = await run_bulk_review(
                    config_path=config_path,
                    do_lessons=lessons,
                    do_reflections=reflections,
                    limit=limit,
                    on_status=on_status,
                )

            for job_name in ["lessons", "reflections"]:
                job_stats = result.get(job_name)
                if job_stats is None:
                    continue
                console.print(f"\n[bold]{job_name.title()}[/bold]:")
                for k, v in job_stats.items():
                    console.print(f"  {k}: {v}")

            console.print(f"\n[dim]Total: {result['elapsed_s']}s[/dim]")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
