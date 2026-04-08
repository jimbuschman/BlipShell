"""Rebuild sqlite-vec vector tables from SQLite summaries.

Populates vec_memories, vec_core_memories, vec_lessons, vec_entities
by embedding text from the main SQLite tables via Ollama.

Fast path: if data/chroma exists, extracts vectors directly from
ChromaDB without re-embedding (migration from ChromaDB).

Usage:
    python scripts/rebuild_vectors.py
    python scripts/rebuild_vectors.py --db data/blipshell.db
    python scripts/rebuild_vectors.py --batch-size 100
    python scripts/rebuild_vectors.py --from-chroma data/chroma
"""

import argparse
import asyncio
import sys
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


async def main():
    parser = argparse.ArgumentParser(description="Rebuild sqlite-vec vectors from SQLite")
    parser.add_argument("--db", default="data/blipshell.db", help="SQLite DB path")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dimensions")
    parser.add_argument("--from-chroma", default=None,
                        help="Extract vectors from existing ChromaDB directory (skip re-embedding)")
    args = parser.parse_args()

    # Pre-operation backup
    from scripts.backup_db import backup_before_destructive
    backup_before_destructive("rebuild_vectors", db_path=args.db)

    from blipshell.memory.vector_store import VectorStore
    from blipshell.memory.sqlite_store import SQLiteStore

    console.print(f"SQLite: {args.db}")
    console.print(f"Embedding model: {args.model}")
    console.print(f"Dimensions: {args.dim}")
    console.print()

    sqlite = SQLiteStore(args.db)
    await sqlite.initialize()

    vectors = VectorStore(
        db_path=args.db,
        embedding_model=args.model,
        ollama_url=args.ollama_url,
        embedding_dim=args.dim,
    )
    vectors.initialize()

    # Fast path: migrate from ChromaDB
    if args.from_chroma:
        await _migrate_from_chroma(args.from_chroma, vectors, sqlite, args)
    else:
        await _rebuild_from_scratch(vectors, sqlite, args)

    # Verification
    console.print()
    console.print("[bold]Verifying...[/bold]")
    final = vectors.get_counts()

    cursor = await sqlite._db.execute(
        "SELECT COUNT(*) FROM memories WHERE summary IS NOT NULL AND is_archived = 0"
    )
    mem_count = (await cursor.fetchone())[0]
    cursor = await sqlite._db.execute("SELECT COUNT(*) FROM core_memories WHERE is_active = 1")
    core_count = (await cursor.fetchone())[0]
    cursor = await sqlite._db.execute("SELECT COUNT(*) FROM lessons")
    lesson_count = (await cursor.fetchone())[0]
    cursor = await sqlite._db.execute("SELECT COUNT(*) FROM entities")
    entity_count = (await cursor.fetchone())[0]

    ok = True
    checks = [
        ("memories", final["memories"], mem_count),
        ("core_memories", final["core_memories"], core_count),
        ("lessons", final["lessons"], lesson_count),
        ("entities", final["entities"], entity_count),
    ]
    for name, vec_count, sql_count in checks:
        if vec_count == 0 and sql_count > 0:
            console.print(f"  [bold red]FAIL: {name} — vectors=0, SQLite={sql_count}[/bold red]")
            ok = False
        elif sql_count > 0 and vec_count < sql_count * 0.9:
            console.print(f"  [yellow]WARN: {name} — vectors={vec_count}, SQLite={sql_count}[/yellow]")
        else:
            console.print(f"  [green]OK: {name} — {vec_count}[/green]")

    console.print()
    if ok:
        console.print("[bold green]Done![/bold green] All collections verified.")
    else:
        console.print("[bold red]Some collections incomplete. Check errors above.[/bold red]")

    vectors.close()
    await sqlite.close()


async def _migrate_from_chroma(chroma_path, vectors, sqlite, args):
    """Extract vectors directly from ChromaDB (no re-embedding needed)."""
    import struct
    from pathlib import Path

    if not Path(chroma_path).exists():
        console.print(f"[red]ChromaDB directory not found: {chroma_path}[/red]")
        sys.exit(1)

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        console.print("[red]chromadb not installed — cannot migrate from ChromaDB.[/red]")
        console.print("Install it: pip install chromadb")
        console.print("Or rebuild without --from-chroma to re-embed from scratch.")
        sys.exit(1)

    console.print(f"[bold]Migrating from ChromaDB: {chroma_path}[/bold]")
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))

    collections = {
        "memories": "vec_memories",
        "core_memories": "vec_core_memories",
        "lessons": "vec_lessons",
        "entities": "vec_entities",
    }

    for coll_name, vec_table in collections.items():
        try:
            coll = client.get_collection(coll_name)
        except Exception:
            console.print(f"  [yellow]{coll_name}: collection not found in ChromaDB[/yellow]")
            continue

        count = coll.count()
        if count == 0:
            console.print(f"  [yellow]{coll_name}: empty[/yellow]")
            continue

        console.print(f"  Migrating {count} {coll_name}...")

        # Get all embeddings
        result = coll.get(include=["embeddings"])
        ids = result.get("ids", [])
        embeddings = result.get("embeddings", [])

        if not ids or not embeddings:
            console.print(f"  [yellow]{coll_name}: no embeddings retrieved[/yellow]")
            continue

        migrated = 0
        with vectors._lock:
            for str_id, emb in zip(ids, embeddings):
                try:
                    int_id = int(str_id)
                    blob = struct.pack(f"{len(emb)}f", *emb)
                    vectors._conn.execute(f"DELETE FROM {vec_table} WHERE rowid = ?", [int_id])
                    vectors._conn.execute(
                        f"INSERT INTO {vec_table}(rowid, embedding) VALUES (?, ?)",
                        [int_id, blob],
                    )
                    migrated += 1
                except Exception as e:
                    console.print(f"  [red]Error migrating {coll_name} id={str_id}: {e}[/red]")
            vectors._conn.commit()

        console.print(f"  [green]{coll_name}: migrated {migrated}/{count}[/green]")


async def _rebuild_from_scratch(vectors, sqlite, args):
    """Re-embed everything from SQLite summaries via Ollama."""
    console.print("[bold]Rebuilding from SQLite (re-embedding via Ollama)...[/bold]")

    # Memories
    console.print("[bold]Embedding memories...[/bold]")
    cursor = await sqlite._db.execute(
        "SELECT id, summary FROM memories WHERE summary IS NOT NULL AND is_archived = 0 ORDER BY id"
    )
    rows = await cursor.fetchall()
    _embed_batch(vectors, rows, "vec_memories", args.batch_size, "memories")

    # Core memories
    console.print("[bold]Embedding core memories...[/bold]")
    cursor = await sqlite._db.execute("SELECT id, content FROM core_memories WHERE is_active = 1")
    rows = await cursor.fetchall()
    _embed_batch(vectors, rows, "vec_core_memories", args.batch_size, "core memories")

    # Lessons
    console.print("[bold]Embedding lessons...[/bold]")
    cursor = await sqlite._db.execute("SELECT id, content FROM lessons")
    rows = await cursor.fetchall()
    _embed_batch(vectors, rows, "vec_lessons", args.batch_size, "lessons")

    # Entities
    console.print("[bold]Embedding entities...[/bold]")
    cursor = await sqlite._db.execute("SELECT id, name FROM entities")
    rows = await cursor.fetchall()
    _embed_batch(vectors, rows, "vec_entities", args.batch_size, "entities")


def _embed_batch(vectors, rows, vec_table, batch_size, label):
    """Embed and insert a batch of rows into a vec0 table."""
    from blipshell.memory.vector_store import _serialize_f32

    embedded = 0
    errors = 0
    start = time.perf_counter()

    with Progress(
        SpinnerColumn(), BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task(f"Embedding {label}", total=len(rows))

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            ids = [r[0] for r in batch]
            texts = [r[1] or "" for r in batch]

            try:
                vecs = vectors._embed_batch(texts)
                with vectors._lock:
                    for item_id, vec in zip(ids, vecs):
                        vectors._conn.execute(f"DELETE FROM {vec_table} WHERE rowid = ?", [item_id])
                        vectors._conn.execute(
                            f"INSERT INTO {vec_table}(rowid, embedding) VALUES (?, ?)",
                            [item_id, _serialize_f32(vec)],
                        )
                    vectors._conn.commit()
                embedded += len(batch)
            except Exception as e:
                errors += len(batch)
                console.print(f"[red]Batch error: {e}[/red]")

            progress.advance(task, len(batch))

    elapsed = time.perf_counter() - start
    console.print(f"  Embedded {embedded} {label} ({errors} errors) in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
