"""ChromaDB retry queue for failed write operations.

When a ChromaDB write (add/delete) fails — usually due to Ollama embedding
timeouts or index corruption — the operation is queued in SQLite and retried
later (on startup and during nightly runs).

This prevents SQLite/ChromaDB sync drift: SQLite has the record but ChromaDB
doesn't have the embedding (or vice versa for deletes).
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Operations that can be retried
OP_UPSERT = "upsert"
OP_DELETE = "delete"

# Collections matching ChromaDB collection names
COLLECTION_MEMORIES = "memories"
COLLECTION_CORE = "core_memories"
COLLECTION_LESSONS = "lessons"
COLLECTION_ENTITIES = "entities"


async def queue_failed_op(
    sqlite,
    operation: str,
    collection: str,
    item_id: int,
    document: Optional[str] = None,
    metadata: Optional[dict] = None,
    error: str = "",
):
    """Queue a failed ChromaDB operation for later retry.

    Args:
        sqlite: SQLiteStore instance.
        operation: "upsert" or "delete".
        collection: ChromaDB collection name.
        item_id: ID of the memory/core_memory/lesson/entity.
        document: Text to embed (for upserts). Not needed for deletes.
        metadata: ChromaDB metadata dict (for upserts).
        error: Error message from the failed attempt.
    """
    try:
        meta_json = json.dumps(metadata) if metadata else None
        await sqlite._db.execute(
            """INSERT OR REPLACE INTO chroma_retry_queue
               (operation, collection, item_id, document, metadata_json, error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (operation, collection, item_id, document, meta_json, error[:500]),
        )
        await sqlite._db.commit()
        logger.info(
            "Queued ChromaDB retry: %s %s id=%d (error: %s)",
            operation, collection, item_id, error[:100],
        )
    except Exception as e:
        # Last resort — if we can't even queue the retry, just log it
        logger.error(
            "Failed to queue ChromaDB retry (%s %s id=%d): %s",
            operation, collection, item_id, e,
        )


async def process_retry_queue(sqlite, chroma, limit: int = 200) -> dict:
    """Process pending ChromaDB retry operations.

    Returns stats dict with counts of processed, succeeded, failed items.
    """
    try:
        cursor = await sqlite._db.execute(
            """SELECT id, operation, collection, item_id, document, metadata_json, retry_count
               FROM chroma_retry_queue
               WHERE retry_count < 5
               ORDER BY created_at ASC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
    except Exception as e:
        logger.error("Failed to read retry queue: %s", e)
        return {"processed": 0, "succeeded": 0, "failed": 0}

    if not rows:
        return {"processed": 0, "succeeded": 0, "failed": 0}

    succeeded = 0
    failed = 0

    for row in rows:
        row_id = row["id"]
        operation = row["operation"]
        collection = row["collection"]
        item_id = row["item_id"]
        document = row["document"]
        meta_json = row["metadata_json"]
        metadata = json.loads(meta_json) if meta_json else None

        try:
            if operation == OP_UPSERT and document:
                _do_upsert(chroma, collection, item_id, document, metadata)
            elif operation == OP_DELETE:
                _do_delete(chroma, collection, item_id)
            else:
                logger.warning("Unknown retry op: %s (id=%d)", operation, row_id)
                # Remove invalid entries
                await sqlite._db.execute(
                    "DELETE FROM chroma_retry_queue WHERE id = ?", (row_id,),
                )
                continue

            # Success — remove from queue
            await sqlite._db.execute(
                "DELETE FROM chroma_retry_queue WHERE id = ?", (row_id,),
            )
            succeeded += 1
            logger.info("Retry succeeded: %s %s id=%d", operation, collection, item_id)

        except Exception as e:
            # Still failing — increment retry count
            failed += 1
            await sqlite._db.execute(
                """UPDATE chroma_retry_queue
                   SET retry_count = retry_count + 1, error = ?
                   WHERE id = ?""",
                (str(e)[:500], row_id),
            )
            logger.warning(
                "Retry still failing: %s %s id=%d (attempt %d): %s",
                operation, collection, item_id, row["retry_count"] + 1, e,
            )

    await sqlite._db.commit()

    # Clean up entries that have exceeded max retries
    await sqlite._db.execute(
        "DELETE FROM chroma_retry_queue WHERE retry_count >= 5"
    )
    await sqlite._db.commit()

    return {"processed": len(rows), "succeeded": succeeded, "failed": failed}


def _do_upsert(chroma, collection: str, item_id: int, document: str, metadata: dict | None):
    """Execute a ChromaDB upsert for the given collection."""
    if collection == COLLECTION_MEMORIES:
        chroma.add_memory(item_id, document, metadata)
    elif collection == COLLECTION_CORE:
        chroma.add_core_memory(item_id, document, metadata)
    elif collection == COLLECTION_LESSONS:
        chroma.add_lesson(item_id, document, metadata)
    elif collection == COLLECTION_ENTITIES:
        entity_type = (metadata or {}).get("entity_type", "concept")
        chroma.upsert_entity(item_id, document, entity_type)
    else:
        raise ValueError(f"Unknown collection: {collection}")


def _do_delete(chroma, collection: str, item_id: int):
    """Execute a ChromaDB delete for the given collection."""
    if collection == COLLECTION_MEMORIES:
        chroma.delete_memory(item_id)
    elif collection == COLLECTION_CORE:
        chroma.delete_core_memory(item_id)
    elif collection == COLLECTION_LESSONS:
        chroma.delete_lesson(item_id)
    else:
        raise ValueError(f"Unknown collection for delete: {collection}")


async def reconcile_stores(
    sqlite, chroma, max_actions: int = 500,
) -> dict:
    """Compare SQLite and ChromaDB, queue operations to fix drift.

    Checks three collections: memories, core_memories, lessons.
    - ChromaDB IDs not in SQLite → queue delete (orphaned embeddings)
    - SQLite IDs not in ChromaDB → queue upsert (missing embeddings)

    Args:
        sqlite: SQLiteStore instance.
        chroma: ChromaStore instance.
        max_actions: Safety cap — don't queue more than this many operations per run.

    Returns:
        Stats dict with orphans_found, missing_found, actions_queued per collection.
    """
    stats = {
        "orphans_deleted": 0,
        "missing_queued": 0,
        "collections_checked": 0,
        "errors": 0,
    }

    # --- Memories ---
    try:
        chroma_ids = chroma.get_all_ids("memories")
        cursor = await sqlite._db.execute(
            "SELECT id FROM memories WHERE is_archived = 0"
        )
        sqlite_ids = {row[0] for row in await cursor.fetchall()}
        stats["collections_checked"] += 1

        # Orphans: in ChromaDB but not in SQLite
        orphans = chroma_ids - sqlite_ids
        for oid in list(orphans)[:max_actions]:
            try:
                chroma.delete_memory(oid)
                stats["orphans_deleted"] += 1
            except Exception as e:
                logger.warning("Reconcile: failed to delete orphan memory %d: %s", oid, e)
                stats["errors"] += 1

        # Missing: in SQLite but not in ChromaDB
        missing = sqlite_ids - chroma_ids
        for mid in list(missing)[:max_actions - stats["orphans_deleted"]]:
            try:
                cursor = await sqlite._db.execute(
                    "SELECT summary, session_id, role FROM memories WHERE id = ?",
                    (mid,),
                )
                row = await cursor.fetchone()
                if row and row[0]:
                    await queue_failed_op(
                        sqlite, OP_UPSERT, COLLECTION_MEMORIES, mid,
                        document=row[0],
                        metadata={"session_id": str(row[1] or ""), "role": row[2] or ""},
                        error="reconcile: missing from ChromaDB",
                    )
                    stats["missing_queued"] += 1
            except Exception as e:
                logger.warning("Reconcile: failed to queue missing memory %d: %s", mid, e)
                stats["errors"] += 1

        if orphans or missing:
            logger.info(
                "Reconcile memories: %d orphans deleted, %d missing queued (of %d/%d)",
                min(len(orphans), max_actions), stats["missing_queued"],
                len(orphans), len(missing),
            )
    except Exception as e:
        logger.error("Reconcile memories failed: %s", e)
        stats["errors"] += 1

    # --- Core memories ---
    try:
        chroma_ids = chroma.get_all_ids("core_memories")
        cursor = await sqlite._db.execute(
            "SELECT id FROM core_memories WHERE is_active = 1"
        )
        sqlite_ids = {row[0] for row in await cursor.fetchall()}
        stats["collections_checked"] += 1

        orphans = chroma_ids - sqlite_ids
        for oid in list(orphans)[:max_actions]:
            try:
                chroma.delete_core_memory(oid)
                stats["orphans_deleted"] += 1
            except Exception as e:
                stats["errors"] += 1

        missing = sqlite_ids - chroma_ids
        for mid in list(missing)[:max_actions]:
            try:
                cursor = await sqlite._db.execute(
                    "SELECT content FROM core_memories WHERE id = ?", (mid,),
                )
                row = await cursor.fetchone()
                if row and row[0]:
                    await queue_failed_op(
                        sqlite, OP_UPSERT, COLLECTION_CORE, mid,
                        document=row[0],
                        error="reconcile: missing from ChromaDB",
                    )
                    stats["missing_queued"] += 1
            except Exception as e:
                stats["errors"] += 1

        if orphans or missing:
            logger.info(
                "Reconcile core_memories: %d orphans, %d missing",
                len(orphans), len(missing),
            )
    except Exception as e:
        logger.error("Reconcile core_memories failed: %s", e)
        stats["errors"] += 1

    # --- Lessons ---
    try:
        chroma_ids = chroma.get_all_ids("lessons")
        cursor = await sqlite._db.execute("SELECT id FROM lessons")
        sqlite_ids = {row[0] for row in await cursor.fetchall()}
        stats["collections_checked"] += 1

        orphans = chroma_ids - sqlite_ids
        for oid in list(orphans)[:max_actions]:
            try:
                chroma.delete_lesson(oid)
                stats["orphans_deleted"] += 1
            except Exception as e:
                stats["errors"] += 1

        missing = sqlite_ids - chroma_ids
        for mid in list(missing)[:max_actions]:
            try:
                cursor = await sqlite._db.execute(
                    "SELECT content FROM lessons WHERE id = ?", (mid,),
                )
                row = await cursor.fetchone()
                if row and row[0]:
                    await queue_failed_op(
                        sqlite, OP_UPSERT, COLLECTION_LESSONS, mid,
                        document=row[0],
                        error="reconcile: missing from ChromaDB",
                    )
                    stats["missing_queued"] += 1
            except Exception as e:
                stats["errors"] += 1

        if orphans or missing:
            logger.info(
                "Reconcile lessons: %d orphans, %d missing",
                len(orphans), len(missing),
            )
    except Exception as e:
        logger.error("Reconcile lessons failed: %s", e)
        stats["errors"] += 1

    return stats
