"""ChromaDB vector storage for semantic search.

Replaces the manual OllamaEmbedder + MemoryBlobs + cosine similarity code from C#.
ChromaDB handles embedding generation, HNSW indexing, and similarity search.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# nomic-embed-text has 8192 token context but Ollama may default lower.
# 2000 chars (~500 tokens) is safe for any model and plenty for quality.
MAX_EMBED_CHARS = 2000

# Collection names
MEMORIES_COLLECTION = "memories"
CORE_MEMORIES_COLLECTION = "core_memories"
LESSONS_COLLECTION = "lessons"


class ChromaStore:
    """ChromaDB vector storage for semantic memory search."""

    def __init__(self, persist_dir: str, embedding_model: str = "nomic-embed-text",
                 ollama_url: str = "http://localhost:11434"):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self._client: Optional[chromadb.ClientAPI] = None
        self._memories: Optional[chromadb.Collection] = None
        self._core_memories: Optional[chromadb.Collection] = None
        self._lessons: Optional[chromadb.Collection] = None

    def initialize(self):
        """Initialize ChromaDB client and collections."""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Use Ollama for embedding generation
        embedding_fn = chromadb.utils.embedding_functions.OllamaEmbeddingFunction(
            url=self.ollama_url,
            model_name=self.embedding_model,
        )

        self._memories = self._client.get_or_create_collection(
            name=MEMORIES_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self._core_memories = self._client.get_or_create_collection(
            name=CORE_MEMORIES_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self._lessons = self._client.get_or_create_collection(
            name=LESSONS_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "ChromaDB initialized: memories=%d, core=%d, lessons=%d",
            self._memories.count(),
            self._core_memories.count(),
            self._lessons.count(),
        )

    def _require_collections(self):
        """Raise if collections haven't been initialized."""
        if self._memories is None or self._core_memories is None or self._lessons is None:
            raise RuntimeError("ChromaStore not initialized — call initialize() first")

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate text to fit within the embedding model's context window."""
        if len(text) <= MAX_EMBED_CHARS:
            return text
        logger.debug("Truncating text from %d to %d chars for embedding", len(text), MAX_EMBED_CHARS)
        return text[:MAX_EMBED_CHARS]

    def add_memory(self, memory_id: int, text: str, metadata: Optional[dict] = None):
        """Add a memory embedding to ChromaDB."""
        self._require_collections()
        meta = {**(metadata or {}), "source": "memory"}
        self._memories.upsert(
            ids=[str(memory_id)],
            documents=[self._truncate(text)],
            metadatas=[meta],
        )

    def add_memories_batch(
        self,
        memory_ids: list[int],
        texts: list[str],
        metadatas: list[dict],
    ):
        """Add multiple memory embeddings in a single ChromaDB call.

        Much faster than calling add_memory() in a loop because Ollama
        embeds all documents in one request instead of N separate ones.
        """
        self._require_collections()
        safe_metas = [{**m, "source": "memory"} for m in metadatas]
        self._memories.upsert(
            ids=[str(mid) for mid in memory_ids],
            documents=[self._truncate(t) for t in texts],
            metadatas=safe_metas,
        )

    def add_core_memory(self, core_memory_id: int, text: str, metadata: Optional[dict] = None):
        """Add a core memory embedding to ChromaDB."""
        self._require_collections()
        meta = {**(metadata or {}), "source": "core_memory"}
        self._core_memories.upsert(
            ids=[str(core_memory_id)],
            documents=[self._truncate(text)],
            metadatas=[meta],
        )

    def add_lesson(self, lesson_id: int, text: str, metadata: Optional[dict] = None):
        """Add a lesson embedding to ChromaDB."""
        self._require_collections()
        meta = {**(metadata or {}), "source": "lesson"}
        self._lessons.upsert(
            ids=[str(lesson_id)],
            documents=[self._truncate(text)],
            metadatas=[meta],
        )

    def search_memories(
        self,
        query: str,
        n_results: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Search memories by semantic similarity.

        Returns list of {id, document, distance, metadata} dicts.
        Distance is cosine distance (lower = more similar).
        Similarity = 1 - distance.
        """
        self._require_collections()
        kwargs = {"query_texts": [self._truncate(query)], "n_results": n_results}
        if where:
            kwargs["where"] = where

        try:
            results = self._memories.query(**kwargs)
        except Exception as e:
            logger.error("ChromaDB memory search failed: %s", e)
            return []

        return self._format_results(results)

    def search_core_memories(self, query: str, n_results: int = 10) -> list[dict]:
        """Search core memories by semantic similarity."""
        self._require_collections()
        try:
            results = self._core_memories.query(
                query_texts=[self._truncate(query)], n_results=n_results
            )
        except Exception as e:
            logger.error("ChromaDB core memory search failed: %s", e)
            return []

        return self._format_results(results)

    def search_lessons(self, query: str, n_results: int = 10) -> list[dict]:
        """Search lessons by semantic similarity."""
        self._require_collections()
        try:
            results = self._lessons.query(
                query_texts=[self._truncate(query)], n_results=n_results
            )
        except Exception as e:
            logger.error("ChromaDB lesson search failed: %s", e)
            return []

        return self._format_results(results)

    def _format_results(self, results: dict) -> list[dict]:
        """Format ChromaDB query results into a flat list."""
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        formatted = []
        for i, doc_id in enumerate(results["ids"][0]):
            similarity = 1.0 - (results["distances"][0][i] if results.get("distances") else 0.0)
            formatted.append({
                "id": int(doc_id),
                "document": results["documents"][0][i] if results.get("documents") else "",
                "similarity": similarity,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })
        return formatted

    def delete_memory(self, memory_id: int):
        """Remove a memory from ChromaDB."""
        self._memories.delete(ids=[str(memory_id)])

    def delete_core_memory(self, core_memory_id: int):
        """Remove a core memory from ChromaDB."""
        self._core_memories.delete(ids=[str(core_memory_id)])

    def delete_lesson(self, lesson_id: int):
        """Remove a lesson from ChromaDB."""
        self._lessons.delete(ids=[str(lesson_id)])

    def get_counts(self) -> dict[str, int]:
        """Get document counts for all collections."""
        return {
            "memories": self._memories.count(),
            "core_memories": self._core_memories.count(),
            "lessons": self._lessons.count(),
        }
