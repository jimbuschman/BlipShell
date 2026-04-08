"""Import core memories from a text file into BlipShell.

Inserts into SQLite and embeds via sqlite-vec for semantic search.

Formats supported:
  - Plain text (one memory per line, category defaults to "general")
  - Categorized: "category: content" (e.g., "preference: Jim likes dark mode")

Usage:
    python scripts/import_core_memories.py data/core.txt
    python scripts/import_core_memories.py data/core.txt --db data/blipshell.db
    python scripts/import_core_memories.py data/core.txt --dry-run
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.markup import escape

from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.memory import CoreMemory

console = Console()

VALID_CATEGORIES = {"general", "preference", "fact", "personality", "project", "system"}


def parse_line(line: str) -> tuple[str, str]:
    """Parse a line into (category, content). Returns ('general', line) if no category prefix."""
    line = line.strip()
    if not line or line.startswith("#"):
        return "", ""

    # Check for "category: content" format
    if ":" in line:
        prefix, _, rest = line.partition(":")
        prefix = prefix.strip().lower()
        rest = rest.strip()
        if prefix in VALID_CATEGORIES and rest:
            return prefix, rest

    return "general", line


async def main():
    parser = argparse.ArgumentParser(description="Import core memories from text file")
    parser.add_argument("file", help="Path to core memories text file")
    parser.add_argument("--db", default="data/blipshell.db", help="SQLite DB path")
    parser.add_argument("--importance", type=float, default=0.8,
                        help="Default importance for imported memories (default: 0.8)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and show without importing")
    args = parser.parse_args()

    input_path = Path(args.file)
    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        sys.exit(1)

    # Parse input file
    lines = input_path.read_text(encoding="utf-8").strip().splitlines()
    memories = []
    for line in lines:
        category, content = parse_line(line)
        if content:
            memories.append((category, content))

    if not memories:
        console.print("[yellow]No memories found in file.[/yellow]")
        sys.exit(0)

    console.print(f"[bold]Core Memory Import[/bold]")
    console.print(f"File: {input_path}")
    console.print(f"Memories: {len(memories)}")
    console.print(f"Importance: {args.importance}")
    console.print()

    for i, (cat, content) in enumerate(memories, 1):
        safe = content[:80].encode("ascii", "replace").decode()
        console.print(f"  {i}. \\[{cat}] {escape(safe)}{'...' if len(content) > 80 else ''}")

    if args.dry_run:
        console.print("\n[yellow]Dry run -- nothing imported.[/yellow]")
        return

    console.print()

    from blipshell.memory.vector_store import VectorStore

    # Initialize stores
    sqlite = SQLiteStore(args.db)
    await sqlite.initialize()

    vectors = VectorStore(db_path=args.db, embedding_dim=1024)
    vectors.initialize()

    # Check for duplicates against existing core memories
    existing = await sqlite.get_active_core_memories()
    existing_contents = {m.content.lower().strip() for m in existing}

    imported = 0
    skipped = 0
    for category, content in memories:
        if content.lower().strip() in existing_contents:
            safe = content[:60].encode("ascii", "replace").decode()
            console.print(f"  [dim]SKIP (duplicate): {escape(safe)}...[/dim]")
            skipped += 1
            continue

        core_mem = CoreMemory(
            content=content,
            category=category,
            importance=args.importance,
        )
        mem_id = await sqlite.create_core_memory(core_mem)

        # Embed in vector store
        try:
            vectors.add_core_memory(mem_id, content, {"category": category})
        except Exception as e:
            console.print(f"  [yellow]Warning: Vector embed failed for #{mem_id}: {e}[/yellow]")

        # Tag the core memory
        await sqlite.tag_core_memory(mem_id, [category])

        safe = content[:60].encode("ascii", "replace").decode()
        console.print(f"  [green]OK[/green] #{mem_id} \\[{category}] {escape(safe)}...")
        imported += 1

    await sqlite.close()

    console.print(f"\n[bold green]Done:[/bold green] {imported} imported, {skipped} skipped (duplicates)")


if __name__ == "__main__":
    asyncio.run(main())
