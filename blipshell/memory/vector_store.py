"""sqlite-vec vector storage for semantic search.

Replaces ChromaDB. Vectors stored in the same blipshell.db as structured data,
eliminating the dual-store sync drift that caused HNSW corruption, empty
collections, and entity FK errors.

Uses sqlite-vec's vec0 virtual tables for KNN search with cosine distance.
Embeddings generated via Ollama's /api/embed endpoint.
"""

import functools
import logging
import sqlite3
import struct
import threading
from typing import Optional

import httpx
import ollama
import sqlite_vec

logger = logging.getLogger(__name__)

# Max chars sent to embedding model. qwen3-embedding:0.6b has 32K token context.
MAX_EMBED_CHARS = 6000

# Timeout for Ollama embedding requests (seconds).
EMBED_TIMEOUT = 300.0

# Collection names (match ChromaDB collection names for migration compatibility)
MEMORIES_COLLECTION = "memories"
CORE_MEMORIES_COLLECTION = "core_memories"
LESSONS_COLLECTION = "lessons"
ENTITIES_COLLECTION = "entities"

# Map collection names to vec0 table names
_VEC_TABLES = {
    MEMORIES_COLLECTION: "vec_memories",
    CORE_MEMORIES_COLLECTION: "vec_core_memories",
    LESSONS_COLLECTION: "vec_lessons",
    ENTITIES_COLLECTION: "vec_entities",
}

# Map collection names to source tables and their text/metadata columns
_SOURCE_TABLES = {
    MEMORIES_COLLECTION: {
        "table": "memories",
        "text_col": "summary",
        "meta_cols": ["session_id", "role"],
        "active_filter": "is_archived = 0 AND summary IS NOT NULL",
    },
    CORE_MEMORIES_COLLECTION: {
        "table": "core_memories",
        "text_col": "content",
        "meta_cols": [],
        "active_filter": "is_active = 1",
    },
    LESSONS_COLLECTION: {
        "table": "lessons",
        "text_col": "content",
        "meta_cols": ["project"],
        "active_filter": None,
    },
    ENTITIES_COLLECTION: {
        "table": "entities",
        "text_col": "name",
        "meta_cols": ["entity_type"],
        "active_filter": None,
    },
}


def _ollama_gated(fn):
    """Wrap a VectorStore method with OllamaGate for embedding calls.

    Serializes embedding calls through the shared OllamaGate so they
    don't compete with interactive LLM calls on a single GPU.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        from blipshell.llm.ollama_gate import get_gate
        gate = get_gate()
        with gate.gate(gate.infer_priority()):
            return fn(self, *args, **kwargs)
    return wrapper


def _serialize_f32(vector: list[float]) -> bytes:
    """Pack a float list into a binary blob for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize_f32(blob: bytes) -> list[float]:
    """Unpack a sqlite-vec binary blob into a float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class VectorStore:
    """sqlite-vec vector storage for semantic memory search.

    Drop-in replacement for ChromaStore. Uses vec0 virtual tables
    in the same SQLite database as structured data.
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "qwen3-embedding:0.6b",
        ollama_url: str = "http://localhost:11434",
        embedding_dim: int = 1024,
    ):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.embedding_dim = embedding_dim
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._closed = False
        self._ollama_client: Optional[ollama.Client] = None

    def initialize(self):
        """Open connection, load sqlite-vec, create vec0 tables."""
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=60,
        )
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 60000")

        # Load sqlite-vec extension
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Create vec0 virtual tables
        for vec_table in _VEC_TABLES.values():
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {vec_table} USING vec0("
                f"embedding float[{self.embedding_dim}] distance_metric=cosine"
                f")"
            )
        self._conn.commit()

        # Initialize Ollama client (lazy — may not be available on dev machines)
        try:
            self._ollama_client = ollama.Client(
                host=self.ollama_url,
                timeout=httpx.Timeout(EMBED_TIMEOUT, connect=10.0),
            )
        except Exception as e:
            logger.warning("Could not create Ollama client: %s", e)

        # Log counts
        counts = self.get_counts()
        logger.info(
            "VectorStore initialized: memories=%d, core=%d, lessons=%d, entities=%d",
            counts["memories"], counts["core_memories"],
            counts["lessons"], counts["entities"],
        )

    def close(self):
        """Close the connection."""
        self._closed = True
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._ollama_client = None
        logger.info("VectorStore closed")

    def _require_open(self):
        """Raise if store is closed or not initialized."""
        if self._closed:
            raise RuntimeError("VectorStore is closed")
        if self._conn is None:
            raise RuntimeError("VectorStore not initialized — call initialize() first")

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate text to fit embedding model context."""
        if len(text) <= MAX_EMBED_CHARS:
            return text
        logger.debug("Truncating text from %d to %d chars", len(text), MAX_EMBED_CHARS)
        return text[:MAX_EMBED_CHARS]

    # --- Embedding generation ---

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for a single text via Ollama."""
        if self._ollama_client is None:
            raise RuntimeError("Ollama client not available — cannot generate embeddings")
        response = self._ollama_client.embed(
            model=self.embedding_model,
            input=self._truncate(text),
        )
        return response["embeddings"][0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in one Ollama call."""
        if self._ollama_client is None:
            raise RuntimeError("Ollama client not available — cannot generate embeddings")
        response = self._ollama_client.embed(
            model=self.embedding_model,
            input=[self._truncate(t) for t in texts],
        )
        return response["embeddings"]

    # --- Write methods (OllamaGate serialized) ---

    @_ollama_gated
    def add_memory(self, memory_id: int, text: str, metadata: Optional[dict] = None):
        """Add or update a memory embedding."""
        self._require_open()
        vec = self._embed(text)
        blob = _serialize_f32(vec)
        with self._lock:
            self._conn.execute(
                "DELETE FROM vec_memories WHERE rowid = ?", [memory_id]
            )
            self._conn.execute(
                "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                [memory_id, blob],
            )
            self._conn.commit()

    @_ollama_gated
    def add_memories_batch(
        self,
        memory_ids: list[int],
        texts: list[str],
        metadatas: list[dict],
    ):
        """Add multiple memory embeddings in one batch."""
        self._require_open()
        if not memory_ids:
            return
        vectors = self._embed_batch(texts)
        with self._lock:
            for mid, vec in zip(memory_ids, vectors):
                self._conn.execute(
                    "DELETE FROM vec_memories WHERE rowid = ?", [mid]
                )
                self._conn.execute(
                    "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                    [mid, _serialize_f32(vec)],
                )
            self._conn.commit()

    @_ollama_gated
    def add_core_memory(self, core_memory_id: int, text: str, metadata: Optional[dict] = None):
        """Add or update a core memory embedding."""
        self._require_open()
        vec = self._embed(text)
        blob = _serialize_f32(vec)
        with self._lock:
            self._conn.execute(
                "DELETE FROM vec_core_memories WHERE rowid = ?", [core_memory_id]
            )
            self._conn.execute(
                "INSERT INTO vec_core_memories(rowid, embedding) VALUES (?, ?)",
                [core_memory_id, blob],
            )
            self._conn.commit()

    @_ollama_gated
    def add_lesson(self, lesson_id: int, text: str, metadata: Optional[dict] = None):
        """Add or update a lesson embedding."""
        self._require_open()
        vec = self._embed(text)
        blob = _serialize_f32(vec)
        with self._lock:
            self._conn.execute(
                "DELETE FROM vec_lessons WHERE rowid = ?", [lesson_id]
            )
            self._conn.execute(
                "INSERT INTO vec_lessons(rowid, embedding) VALUES (?, ?)",
                [lesson_id, blob],
            )
            self._conn.commit()

    @_ollama_gated
    def upsert_entity(self, entity_id: int, name: str, entity_type: str = "concept"):
        """Add or update an entity embedding."""
        self._require_open()
        vec = self._embed(name)
        blob = _serialize_f32(vec)
        with self._lock:
            self._conn.execute(
                "DELETE FROM vec_entities WHERE rowid = ?", [entity_id]
            )
            self._conn.execute(
                "INSERT INTO vec_entities(rowid, embedding) VALUES (?, ?)",
                [entity_id, blob],
            )
            self._conn.commit()

    @_ollama_gated
    def upsert_entities_batch(
        self,
        entity_ids: list[int],
        names: list[str],
        entity_types: list[str] | None = None,
    ):
        """Batch upsert entity embeddings."""
        self._require_open()
        if not entity_ids:
            return
        vectors = self._embed_batch(names)
        with self._lock:
            for eid, vec in zip(entity_ids, vectors):
                self._conn.execute(
                    "DELETE FROM vec_entities WHERE rowid = ?", [eid]
                )
                self._conn.execute(
                    "INSERT INTO vec_entities(rowid, embedding) VALUES (?, ?)",
                    [eid, _serialize_f32(vec)],
                )
            self._conn.commit()

    # --- Search methods (NOT gated — small embedding model runs concurrently) ---

    def search_memories(
        self,
        query: str,
        n_results: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Search memories by semantic similarity.

        Returns list of {id, document, similarity, metadata} dicts,
        same format as ChromaStore for compatibility.

        The `where` parameter triggers overfetch + post-filter since vec0
        doesn't support metadata filtering during KNN.
        """
        self._require_open()
        vec = self._embed(query)
        blob = _serialize_f32(vec)

        # Overfetch when filtering is needed
        fetch_k = n_results * 3 if where else n_results

        with self._lock:
            rows = self._conn.execute(
                "SELECT rowid, distance FROM vec_memories "
                "WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                [blob, fetch_k],
            ).fetchall()

        if not rows:
            return []

        # Enrich from memories table
        row_ids = [r[0] for r in rows]
        dist_map = {r[0]: r[1] for r in rows}

        with self._lock:
            placeholders = ",".join("?" * len(row_ids))
            enriched = self._conn.execute(
                f"SELECT id, summary, session_id, role FROM memories "
                f"WHERE id IN ({placeholders})",
                row_ids,
            ).fetchall()

        results = []
        for row in enriched:
            mem_id, summary, session_id, role = row
            if mem_id not in dist_map:
                continue
            similarity = 1.0 - dist_map[mem_id]
            meta = {"session_id": str(session_id) if session_id else "", "role": role or "", "source": "memory"}
            results.append({
                "id": mem_id,
                "document": summary or "",
                "similarity": similarity,
                "metadata": meta,
            })

        # Post-filter by where clause if provided
        if where:
            results = self._apply_where_filter(results, where)

        # Sort by similarity descending, limit to n_results
        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:n_results]

    def search_core_memories(self, query: str, n_results: int = 10) -> list[dict]:
        """Search core memories by semantic similarity."""
        self._require_open()
        vec = self._embed(query)
        blob = _serialize_f32(vec)

        with self._lock:
            rows = self._conn.execute(
                "SELECT rowid, distance FROM vec_core_memories "
                "WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                [blob, n_results],
            ).fetchall()

        if not rows:
            return []

        row_ids = [r[0] for r in rows]
        dist_map = {r[0]: r[1] for r in rows}

        with self._lock:
            placeholders = ",".join("?" * len(row_ids))
            enriched = self._conn.execute(
                f"SELECT id, content FROM core_memories WHERE id IN ({placeholders})",
                row_ids,
            ).fetchall()

        results = []
        for row in enriched:
            mem_id, content = row
            if mem_id not in dist_map:
                continue
            results.append({
                "id": mem_id,
                "document": content or "",
                "similarity": 1.0 - dist_map[mem_id],
                "metadata": {"source": "core_memory"},
            })

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results

    def search_lessons(self, query: str, n_results: int = 10) -> list[dict]:
        """Search lessons by semantic similarity."""
        self._require_open()
        vec = self._embed(query)
        blob = _serialize_f32(vec)

        with self._lock:
            rows = self._conn.execute(
                "SELECT rowid, distance FROM vec_lessons "
                "WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                [blob, n_results],
            ).fetchall()

        if not rows:
            return []

        row_ids = [r[0] for r in rows]
        dist_map = {r[0]: r[1] for r in rows}

        with self._lock:
            placeholders = ",".join("?" * len(row_ids))
            enriched = self._conn.execute(
                f"SELECT id, content, project FROM lessons WHERE id IN ({placeholders})",
                row_ids,
            ).fetchall()

        results = []
        for row in enriched:
            lesson_id, content, project = row
            if lesson_id not in dist_map:
                continue
            meta = {"source": "lesson"}
            if project:
                meta["project"] = project
            results.append({
                "id": lesson_id,
                "document": content or "",
                "similarity": 1.0 - dist_map[lesson_id],
                "metadata": meta,
            })

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results

    def search_similar_entities(self, name: str, n_results: int = 5) -> list[dict]:
        """Search for similar entity names by embedding similarity.

        Returns list of {id, name, similarity, entity_type} dicts.
        """
        self._require_open()
        vec = self._embed(name)
        blob = _serialize_f32(vec)

        with self._lock:
            rows = self._conn.execute(
                "SELECT rowid, distance FROM vec_entities "
                "WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                [blob, n_results],
            ).fetchall()

        if not rows:
            return []

        row_ids = [r[0] for r in rows]
        dist_map = {r[0]: r[1] for r in rows}

        with self._lock:
            placeholders = ",".join("?" * len(row_ids))
            enriched = self._conn.execute(
                f"SELECT id, name, entity_type FROM entities WHERE id IN ({placeholders})",
                row_ids,
            ).fetchall()

        results = []
        for row in enriched:
            eid, ename, etype = row
            if eid not in dist_map:
                continue
            results.append({
                "id": eid,
                "name": ename or "",
                "similarity": 1.0 - dist_map[eid],
                "entity_type": etype or "concept",
            })

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results

    # --- Delete methods ---

    def delete_memory(self, memory_id: int):
        """Remove a memory vector."""
        with self._lock:
            self._conn.execute("DELETE FROM vec_memories WHERE rowid = ?", [memory_id])
            self._conn.commit()

    def delete_core_memory(self, core_memory_id: int):
        """Remove a core memory vector."""
        with self._lock:
            self._conn.execute("DELETE FROM vec_core_memories WHERE rowid = ?", [core_memory_id])
            self._conn.commit()

    def delete_lesson(self, lesson_id: int):
        """Remove a lesson vector."""
        with self._lock:
            self._conn.execute("DELETE FROM vec_lessons WHERE rowid = ?", [lesson_id])
            self._conn.commit()

    def delete_entity(self, entity_id: int):
        """Remove an entity vector."""
        with self._lock:
            self._conn.execute("DELETE FROM vec_entities WHERE rowid = ?", [entity_id])
            self._conn.commit()

    # --- Utility methods ---

    def get_all_ids(self, collection: str = "memories") -> set[int]:
        """Get all vector IDs for a collection."""
        vec_table = _VEC_TABLES.get(collection)
        if not vec_table:
            return set()
        try:
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT rowid FROM {vec_table}"
                ).fetchall()
            return {r[0] for r in rows}
        except Exception as e:
            logger.error("Failed to get IDs from %s: %s", vec_table, e)
            return set()

    def get_embeddings_by_ids(self, memory_ids: list[int]) -> dict[int, list[float]]:
        """Retrieve raw embedding vectors for memories by ID.

        Returns dict mapping memory_id to embedding vector.
        Missing IDs are silently skipped.
        """
        self._require_open()
        if not memory_ids:
            return {}

        try:
            with self._lock:
                placeholders = ",".join("?" * len(memory_ids))
                rows = self._conn.execute(
                    f"SELECT rowid, embedding FROM vec_memories "
                    f"WHERE rowid IN ({placeholders})",
                    memory_ids,
                ).fetchall()
        except Exception as e:
            logger.error("Failed to get embeddings by IDs: %s", e)
            return {}

        return {r[0]: _deserialize_f32(r[1]) for r in rows if r[1] is not None}

    def get_counts(self) -> dict[str, int]:
        """Get vector counts for all collections."""
        counts = {}
        for collection, vec_table in _VEC_TABLES.items():
            try:
                with self._lock:
                    row = self._conn.execute(
                        f"SELECT COUNT(*) FROM {vec_table}"
                    ).fetchone()
                counts[collection] = row[0] if row else 0
            except Exception:
                counts[collection] = 0
        return counts

    def backfill_missing_vectors(self, collection: str = "memories", limit: int = 200) -> dict:
        """Find items in SQLite without vectors and re-embed them.

        Replaces the chroma_retry_queue + reconcile_stores system.
        Returns stats dict with processed/succeeded/failed counts.
        """
        self._require_open()
        if self._ollama_client is None:
            return {"processed": 0, "succeeded": 0, "failed": 0, "error": "no ollama"}

        source = _SOURCE_TABLES.get(collection)
        vec_table = _VEC_TABLES.get(collection)
        if not source or not vec_table:
            return {"processed": 0, "succeeded": 0, "failed": 0, "error": "unknown collection"}

        # Find IDs in source table but not in vec table
        where = f"WHERE {source['active_filter']}" if source["active_filter"] else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT s.id, s.{source['text_col']} FROM {source['table']} s "
                f"LEFT JOIN {vec_table} v ON v.rowid = s.id "
                f"WHERE v.rowid IS NULL "
                + (f"AND s.{source['active_filter']} " if source["active_filter"] else "")
                + f"LIMIT ?",
                [limit],
            ).fetchall()

        if not rows:
            return {"processed": 0, "succeeded": 0, "failed": 0}

        stats = {"processed": len(rows), "succeeded": 0, "failed": 0}

        # Embed and insert in batches
        batch_size = 50
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            ids = [r[0] for r in batch]
            texts = [r[1] or "" for r in batch]

            try:
                from blipshell.llm.ollama_gate import get_gate
                gate = get_gate()
                with gate.gate(gate.infer_priority()):
                    vectors = self._embed_batch(texts)

                with self._lock:
                    for item_id, vec in zip(ids, vectors):
                        self._conn.execute(
                            f"DELETE FROM {vec_table} WHERE rowid = ?", [item_id]
                        )
                        self._conn.execute(
                            f"INSERT INTO {vec_table}(rowid, embedding) VALUES (?, ?)",
                            [item_id, _serialize_f32(vec)],
                        )
                    self._conn.commit()
                stats["succeeded"] += len(batch)
            except Exception as e:
                logger.error("Backfill batch failed for %s: %s", collection, e)
                stats["failed"] += len(batch)

        return stats

    # --- Internal helpers ---

    @staticmethod
    def _apply_where_filter(results: list[dict], where: dict) -> list[dict]:
        """Apply ChromaDB-style where filter as post-filter.

        Supports: {"field": {"$in": [values]}} and {"field": value}.
        """
        filtered = []
        for r in results:
            meta = r.get("metadata", {})
            match = True
            for key, condition in where.items():
                meta_val = meta.get(key, "")
                if isinstance(condition, dict):
                    if "$in" in condition:
                        if meta_val not in condition["$in"]:
                            match = False
                            break
                    elif "$eq" in condition:
                        if meta_val != condition["$eq"]:
                            match = False
                            break
                else:
                    if meta_val != condition:
                        match = False
                        break
            if match:
                filtered.append(r)
        return filtered
