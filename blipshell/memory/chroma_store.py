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

    def add_memory(self, memory_id: int, text: str, metadata: Optional[dict] = None):
        """Add a memory embedding to ChromaDB."""
        meta = metadata or {}
        meta["source"] = "memory"
        self._memories.upsert(
            ids=[str(memory_id)],
            documents=[text],
            metadatas=[meta],
        )

    def add_core_memory(self, core_memory_id: int, text: str, metadata: Optional[dict] = None):
        """Add a core memory embedding to ChromaDB."""
        meta = metadata or {}
        meta["source"] = "core_memory"
        self._core_memories.upsert(
            ids=[str(core_memory_id)],
            documents=[text],
            metadatas=[meta],
        )

    def add_lesson(self, lesson_id: int, text: str, metadata: Optional[dict] = None):
        """Add a lesson embedding to ChromaDB."""
        meta = metadata or {}
        meta["source"] = "lesson"
        self._lessons.upsert(
            ids=[str(lesson_id)],
            documents=[text],
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
        kwargs = {"query_texts": [query], "n_results": n_results}
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
        try:
            results = self._core_memories.query(
                query_texts=[query], n_results=n_results
            )
        except Exception as e:
            logger.error("ChromaDB core memory search failed: %s", e)
            return []

        return self._format_results(results)

    def search_lessons(self, query: str, n_results: int = 10) -> list[dict]:
        """Search lessons by semantic similarity."""
        try:
            results = self._lessons.query(
                query_texts=[query], n_results=n_results
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

    def get_counts(self) -> dict[str, int]:
        """Get document counts for all collections."""
        return {
            "memories": self._memories.count(),
            "core_memories": self._core_memories.count(),
            "lessons": self._lessons.count(),
        }
