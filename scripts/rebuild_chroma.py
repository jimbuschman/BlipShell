"""Rebuild ChromaDB from SQLite summaries — no LLM calls needed.

Wipes the ChromaDB memories/core_memories/lessons collections and
re-embeds everything from the current SQLite data using nomic-embed-text.

Usage:
    python scripts/rebuild_chroma.py
    python scripts/rebuild_chroma.py --db data/blipshell.db --chroma data/chroma
    python scripts/rebuild_chroma.py --batch-size 100
"""

import argparse
import asyncio
import sys
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


async def main():
    parser = argparse.ArgumentParser(description="Rebuild ChromaDB from SQLite")
    parser.add_argument("--db", default="data/blipshell.db", help="SQLite DB path")
    parser.add_argument("--chroma", default="data/chroma", help="ChromaDB directory")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    # Pre-operation backup (ChromaDB rebuild is destructive)
    from scripts.backup_db import backup_before_destructive
    backup_before_destructive("rebuild_chroma", db_path=args.db, chroma_path=args.chroma)

    # Import after parsing so --help is fast
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore

    # Connect
    console.print(f"SQLite: {args.db}")
    console.print(f"ChromaDB: {args.chroma}")
    console.print(f"Embedding model: {args.model}")
    console.print()

    sqlite = SQLiteStore(args.db)
    await sqlite.initialize()

    chroma = ChromaStore(
        persist_dir=args.chroma,
        embedding_model=args.model,
        ollama_url=args.ollama_url,
    )
    chroma.initialize()

    counts = chroma.get_counts()
    console.print(f"Current ChromaDB: memories={counts['memories']}, core={counts['core_memories']}, lessons={counts['lessons']}")

    # Count what we need to embed
    cursor = await sqlite._db.execute(
        "SELECT COUNT(*) FROM memories WHERE summary IS NOT NULL AND is_archived = 0"
    )
    mem_count = (await cursor.fetchone())[0]

    cursor = await sqlite._db.execute(
        "SELECT COUNT(*) FROM core_memories WHERE is_active = 1"
    )
    core_count = (await cursor.fetchone())[0]

    cursor = await sqlite._db.execute("SELECT COUNT(*) FROM lessons")
    lesson_count = (await cursor.fetchone())[0]

    console.print(f"SQLite: {mem_count} memories, {core_count} core memories, {lesson_count} lessons")
    console.print()

    # --- Memories ---
    console.print("[bold]Rebuilding memories...[/bold]")
    start = time.perf_counter()

    cursor = await sqlite._db.execute(
        "SELECT id, summary, session_id, role FROM memories WHERE summary IS NOT NULL AND is_archived = 0 ORDER BY id"
    )
    rows = await cursor.fetchall()

    embedded = 0
    errors = 0
    with Progress(
        SpinnerColumn(), BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Embedding memories", total=len(rows))

        batch_ids = []
        batch_texts = []
        batch_metas = []

        for row in rows:
            batch_ids.append(row["id"])
            batch_texts.append(row["summary"])
            batch_metas.append({
                "session_id": str(row["session_id"] or ""),
                "role": row["role"] or "",
            })

            if len(batch_ids) >= args.batch_size:
                try:
                    chroma.add_memories_batch(batch_ids, batch_texts, batch_metas)
                    embedded += len(batch_ids)
                except Exception as e:
                    errors += len(batch_ids)
                    console.print(f"[red]Batch error: {e}[/red]")
                progress.advance(task, len(batch_ids))
                batch_ids, batch_texts, batch_metas = [], [], []

        # Final batch
        if batch_ids:
            try:
                chroma.add_memories_batch(batch_ids, batch_texts, batch_metas)
                embedded += len(batch_ids)
            except Exception as e:
                errors += len(batch_ids)
                console.print(f"[red]Batch error: {e}[/red]")
            progress.advance(task, len(batch_ids))

    elapsed = time.perf_counter() - start
    console.print(f"  Embedded {embedded} memories ({errors} errors) in {elapsed:.1f}s")

    # --- Core memories (batched) ---
    console.print("[bold]Rebuilding core memories...[/bold]")
    cursor = await sqlite._db.execute(
        "SELECT id, content FROM core_memories WHERE is_active = 1"
    )
    core_rows = await cursor.fetchall()
    core_errors = 0
    for i in range(0, len(core_rows), args.batch_size):
        batch = core_rows[i:i + args.batch_size]
        try:
            coll = chroma._core_memories
            coll.upsert(
                ids=[str(r["id"]) for r in batch],
                documents=[chroma._truncate(r["content"]) for r in batch],
                metadatas=[{"source": "core_memory"} for _ in batch],
            )
        except Exception as e:
            core_errors += len(batch)
            console.print(f"[red]Core memory batch error: {e}[/red]")
    console.print(f"  Embedded {len(core_rows)} core memories ({core_errors} errors)")

    # --- Lessons (batched) ---
    console.print("[bold]Rebuilding lessons...[/bold]")
    cursor = await sqlite._db.execute("SELECT id, content FROM lessons")
    lesson_rows = await cursor.fetchall()
    lesson_errors = 0
    for i in range(0, len(lesson_rows), args.batch_size):
        batch = lesson_rows[i:i + args.batch_size]
        try:
            coll = chroma._lessons
            coll.upsert(
                ids=[str(r["id"]) for r in batch],
                documents=[chroma._truncate(r["content"]) for r in batch],
                metadatas=[{"source": "lesson"} for _ in batch],
            )
        except Exception as e:
            lesson_errors += len(batch)
            console.print(f"[red]Lesson batch error: {e}[/red]")
    console.print(f"  Embedded {len(lesson_rows)} lessons ({lesson_errors} errors)")

    # Final counts
    console.print()
    final = chroma.get_counts()
    console.print(f"[bold green]Done![/bold green] ChromaDB: memories={final['memories']}, core={final['core_memories']}, lessons={final['lessons']}")

    await sqlite.close()


if __name__ == "__main__":
    asyncio.run(main())
