"""Rebuild vector embeddings from SQLite summaries — no LLM calls needed.

Drops the sqlite-vec vec0 tables and re-embeds everything from
the current SQLite data.

Usage:
    python scripts/rebuild_chroma.py
    python scripts/rebuild_chroma.py --db data/blipshell.db
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
    parser = argparse.ArgumentParser(description="Rebuild vector embeddings from SQLite")
    parser.add_argument("--db", default="data/blipshell.db", help="SQLite DB path")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    # Pre-operation backup (rebuild is destructive)
    from scripts.backup_db import backup_before_destructive
    backup_before_destructive("rebuild_vectors", db_path=args.db)

    # Import after parsing so --help is fast
    from blipshell.memory.vector_store import VectorStore
    from blipshell.memory.sqlite_store import SQLiteStore

    # Connect
    console.print(f"SQLite: {args.db}")
    console.print(f"Embedding model: {args.model}")
    console.print()

    sqlite = SQLiteStore(args.db)
    await sqlite.initialize()

    # Delete existing vec0 tables to rebuild from scratch
    import sqlite3
    conn = sqlite3.connect(args.db)
    for table in ["vec_memories", "vec_core_memories", "vec_lessons", "vec_entities"]:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            console.print(f"  Dropped table: {table}")
        except Exception:
            pass
    conn.commit()
    conn.close()

    vectors = VectorStore(
        db_path=args.db,
        embedding_model=args.model,
        ollama_url=args.ollama_url,
        embedding_dim=1024,
    )
    vectors.initialize()

    console.print(f"Embedding model: {args.model}")
    console.print(f"Vec tables recreated (fresh)")

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
                    vectors.add_memories_batch(batch_ids, batch_texts, batch_metas)
                    embedded += len(batch_ids)
                except Exception as e:
                    errors += len(batch_ids)
                    console.print(f"[red]Batch error: {e}[/red]")
                progress.advance(task, len(batch_ids))
                batch_ids, batch_texts, batch_metas = [], [], []

        # Final batch
        if batch_ids:
            try:
                vectors.add_memories_batch(batch_ids, batch_texts, batch_metas)
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
            for r in batch:
                vectors.add_core_memory(r["id"], r["content"])
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
            for r in batch:
                vectors.add_lesson(r["id"], r["content"])
        except Exception as e:
            lesson_errors += len(batch)
            console.print(f"[red]Lesson batch error: {e}[/red]")
    console.print(f"  Embedded {len(lesson_rows)} lessons ({lesson_errors} errors)")

    # --- Entities ---
    console.print("[bold]Rebuilding entity embeddings...[/bold]")
    cursor = await sqlite._db.execute(
        "SELECT id, name, entity_type FROM entities"
    )
    entity_rows = await cursor.fetchall()
    entity_errors = 0
    for i in range(0, len(entity_rows), args.batch_size):
        batch = entity_rows[i:i + args.batch_size]
        try:
            for r in batch:
                vectors.upsert_entity(r["id"], r["name"], r["entity_type"] or "concept")
        except Exception as e:
            entity_errors += len(batch)
            console.print(f"[red]Entity batch error: {e}[/red]")
    console.print(f"  Embedded {len(entity_rows)} entities ({entity_errors} errors)")

    # --- Verification ---
    console.print()
    console.print("[bold]Verifying...[/bold]")
    final = vectors.get_counts()
    ok = True

    checks = [
        ("memories", final["memories"], mem_count),
        ("core_memories", final["core_memories"], core_count),
        ("lessons", final["lessons"], lesson_count),
        ("entities", final.get("entities", 0), len(entity_rows)),
    ]
    for name, vec_count, sqlite_count in checks:
        if vec_count == 0 and sqlite_count > 0:
            console.print(f"  [bold red]FAIL: {name} — vec store has 0, SQLite has {sqlite_count}[/bold red]")
            ok = False
        elif vec_count < sqlite_count * 0.9:
            console.print(f"  [yellow]WARN: {name} — vec={vec_count}, SQLite={sqlite_count} (mismatch)[/yellow]")
        else:
            console.print(f"  [green]OK: {name} — {vec_count}[/green]")

    console.print()
    if ok:
        console.print(f"[bold green]Done![/bold green] All collections verified.")
    else:
        console.print(f"[bold red]FAILED — some collections are empty. Check errors above.[/bold red]")
        await sqlite.close()
        sys.exit(1)

    await sqlite.close()


if __name__ == "__main__":
    asyncio.run(main())
