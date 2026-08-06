"""Shared conversation-import plumbing for the CLI.

Five near-identical ~70-line Click commands (ChatGPT, three Claude variants,
DeepSeek) differed only in which parser produced the conversation list. Beyond
the duplication, they had drifted apart in a way that mattered: only
`import-claude code` wrapped its run in `import_lock`, so the other four could
run concurrently with the nightly maintenance job — exactly the SQLite write
contention the lock exists to prevent (nightly checks `is_import_active` and
skips itself). Consolidating means every importer takes the lock.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from blipshell.ui.console import console


async def _run_import(
    config_path: str | None,
    parse: Callable[[str], list],
    source: str,
    operation: str,
    max_count: int | None,
    skip_lessons: bool,
    max_concurrent: int | None = None,
) -> None:
    from rich.progress import Progress

    from blipshell.core.config import ConfigManager
    from blipshell.core.import_lock import import_lock
    from blipshell.import_common import import_conversations
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.memory.vector_store import VectorStore
    from blipshell.models.config import get_ollama_url

    console.print(f"[cyan]Parsing {source}...[/cyan]")
    convs = parse(source)
    console.print(f"Found [bold]{len(convs)}[/bold] conversations.")

    if max_count:
        convs = convs[:max_count]
        console.print(f"Importing first [bold]{max_count}[/bold].")

    if not convs:
        console.print("[yellow]No conversations to import.[/yellow]")
        return

    config_manager = ConfigManager(config_path)
    cfg = config_manager.load()

    sqlite = SQLiteStore(cfg.database.path)
    await sqlite.initialize()

    vectors = VectorStore(
        db_path=cfg.database.path,
        embedding_model=cfg.models.embedding,
        ollama_url=get_ollama_url(cfg.endpoints),
        embedding_dim=cfg.database.embedding_dimensions,
    )
    vectors.initialize()

    endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
    router = LLMRouter(cfg.models, endpoint_manager)

    try:
        # Signals "something heavy is running" so nightly stands down.
        with import_lock(cfg.database.path, operation=operation):
            with Progress(console=console) as progress:
                task = progress.add_task("Importing...", total=len(convs))

                def on_progress(idx, total, title, stats):
                    label = f"[cyan]{title[:40]}[/cyan]"
                    i, s = stats.conversations_imported, stats.conversations_skipped
                    if i or s:
                        label += f"  [dim]({i} imported, {s} skipped)[/dim]"
                    progress.update(task, completed=idx, description=label)

                kwargs = {}
                if max_concurrent is not None:
                    kwargs["max_concurrent"] = max_concurrent

                stats = await import_conversations(
                    sqlite=sqlite,
                    vectors=vectors,
                    router=router,
                    config=cfg.memory,
                    conversations=convs,
                    on_progress=on_progress,
                    skip_lessons=skip_lessons,
                    **kwargs,
                )
                progress.update(task, completed=len(convs))
    finally:
        await sqlite.close()

    # Lives with cli.py's other Click-command helpers; imported here rather
    # than at module scope because cli.py imports this module.
    from blipshell.ui.cli import _print_import_summary
    _print_import_summary(stats)


def run_import(
    config_path: str | None,
    parse: Callable[[str], list],
    source: str,
    operation: str,
    max_count: int | None = None,
    skip_lessons: bool = False,
    max_concurrent: int | None = None,
) -> None:
    """Synchronous entry point for a Click command."""
    asyncio.run(_run_import(
        config_path, parse, source, operation,
        max_count, skip_lessons, max_concurrent,
    ))
