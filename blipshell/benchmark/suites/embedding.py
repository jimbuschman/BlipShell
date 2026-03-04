"""Embedding suite — embed speed + search quality benchmark.

No LLMRouter needed — uses OllamaEmbeddingFunction directly.
Tests embedding throughput and search relevance.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING, Callable

from blipshell.benchmark.models import SuiteResult, TaskScore
from blipshell.benchmark.shared import load_real_user_queries
from blipshell.benchmark.suites.base import BenchmarkSuite

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODELS = ["nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed:335m"]


class EmbeddingSuite(BenchmarkSuite):
    name = "embedding"
    description = "Embedding speed + search quality (cosine similarity)"
    task_types = ["embedding"]
    needs_db = True
    needs_router = False  # Uses ChromaDB embedding directly
    quick_samples = 100
    thorough_samples = 500

    async def run(
        self,
        models: list[str],
        *,
        router_factory: Callable[[str], LLMRouter] | None = None,
        config: BlipShellConfig | None = None,
        db_path: str | None = None,
        ollama_url: str = "http://localhost:11434",
        thorough: bool = False,
        on_status: Callable[[str], None] | None = None,
    ) -> list[SuiteResult]:
        if not db_path:
            logger.warning("Embedding suite requires DB path, skipping")
            return []

        n = self.thorough_samples if thorough else self.quick_samples
        memories = self._load_sample(db_path, n)
        if not memories:
            logger.warning("No memories found for embedding benchmark")
            return []

        # Load real user queries instead of hardcoded ones
        query_n = 50 if thorough else 20
        queries = load_real_user_queries(db_path, query_n)
        if not queries:
            logger.warning("No user queries found for search quality test")
            queries = ["test query"]  # minimal fallback

        if on_status:
            on_status(f"[embedding] Loaded {len(memories)} memories, {len(queries)} queries")

        # For embedding, model names are embedding model names, not LLM names
        embed_models = models if models else DEFAULT_EMBED_MODELS

        results = []
        for model in embed_models:
            if on_status:
                on_status(f"[embedding] Testing {model}")
            sr = self._benchmark_model(model, memories, queries, ollama_url, on_status)
            results.append(sr)
        return results

    def _load_sample(self, db_path: str, n: int) -> list[dict]:
        """Load memories with summaries for embedding."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, content, summary
            FROM memories
            WHERE summary IS NOT NULL AND length(summary) > 20
              AND is_archived = 0
            ORDER BY RANDOM()
            LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _benchmark_model(
        self, model: str, memories: list[dict], queries: list[str],
        ollama_url: str, on_status: Callable | None,
    ) -> SuiteResult:
        """Run embedding benchmark (synchronous — ChromaDB API is sync)."""
        import shutil
        import tempfile

        try:
            import chromadb
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        except ImportError:
            logger.warning("ChromaDB not installed, skipping embedding suite")
            return SuiteResult(suite_name=self.name, model=model)

        tmp_dir = tempfile.mkdtemp(prefix="bench_embed_")
        scores = []

        try:
            embed_fn = OllamaEmbeddingFunction(
                url=f"{ollama_url}/api/embed", model_name=model,
            )
            client = chromadb.PersistentClient(path=tmp_dir)
            collection = client.create_collection(
                name="bench", embedding_function=embed_fn,
            )

            # Phase 1: Embed all memories
            docs = [m["summary"] for m in memories]
            ids = [f"mem_{m['id']}" for m in memories]
            batch_size = 50

            embed_start = time.monotonic()
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                collection.add(documents=batch_docs, ids=batch_ids)
            embed_time = time.monotonic() - embed_start
            embed_rate = len(docs) / embed_time if embed_time > 0 else 0

            # Phase 2: Query with test queries
            query_times = []
            top1_sims = []
            top5_sims = []
            hits_05 = 0
            hits_03 = 0
            total_hits = 0

            for query in queries:
                qstart = time.monotonic()
                results = collection.query(query_texts=[query], n_results=5)
                qelapsed = time.monotonic() - qstart
                query_times.append(qelapsed)

                distances = results["distances"][0] if results["distances"] else []
                # ChromaDB returns L2 distances; convert to similarity
                sims = [1 / (1 + d) for d in distances]

                if sims:
                    top1_sims.append(sims[0])
                    top5_sims.extend(sims)
                    for s in sims:
                        total_hits += 1
                        if s >= 0.5:
                            hits_05 += 1
                        if s >= 0.3:
                            hits_03 += 1

            avg_query_time = sum(query_times) / len(query_times) if query_times else 0
            avg_top1 = sum(top1_sims) / len(top1_sims) if top1_sims else 0
            avg_top5 = sum(top5_sims) / len(top5_sims) if top5_sims else 0

            scores.append(TaskScore(
                task_name="embed_speed",
                quality=round(min(embed_rate / 100, 1.0), 3),  # normalize: 100/s = 1.0
                speed_s=round(embed_time, 2),
                samples=len(docs),
                detail={
                    "embed_time_s": round(embed_time, 2),
                    "embed_rate": round(embed_rate, 1),
                    "memories": len(docs),
                },
            ))

            scores.append(TaskScore(
                task_name="search_quality",
                quality=round(avg_top1, 3),
                speed_s=round(avg_query_time, 3),
                samples=len(queries),
                detail={
                    "avg_top1_sim": round(avg_top1, 3),
                    "avg_top5_sim": round(avg_top5, 3),
                    "hits_gte_05": f"{hits_05}/{total_hits}",
                    "hits_gte_03": f"{hits_03}/{total_hits}",
                },
            ))

        except Exception as e:
            logger.warning("Embedding benchmark error: %s", e)
            scores.append(TaskScore(
                task_name="embed_speed", quality=0, speed_s=0, errors=1,
                detail={"error": str(e)},
            ))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        total_time = sum(s.speed_s for s in scores)
        return SuiteResult(
            suite_name=self.name, model=model,
            scores=scores, elapsed_s=round(total_time, 1),
        )
