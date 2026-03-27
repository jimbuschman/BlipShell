"""Reranker for search results using Ollama logprobs.

Calls Ollama's /api/generate with a Qwen3-Reranker model to score
query-document pairs. Uses logprobs extraction (num_predict=1) to get
calibrated relevance scores instead of binary yes/no parsing.

Ollama has no native /api/rerank endpoint, so we use the generate API
with logprobs=true and extract P(yes) / (P(yes) + P(no)) as the score.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Default instruction for memory retrieval (Qwen recommends task-specific instructions)
DEFAULT_INSTRUCTION = (
    "Given a user's search query, determine if the document contains "
    "information relevant to answering the query"
)

# System prompt per Qwen3-Reranker model card
RERANKER_SYSTEM = (
    'Judge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only be '
    '"yes" or "no".'
)


@dataclass
class RerankResult:
    """A reranked document with its relevance score."""
    index: int          # Original index in the input list
    score: float        # Relevance score (0.0-1.0)
    memory_id: int      # Memory ID for correlation


class Reranker:
    """Scores query-document pairs using an Ollama reranker model.

    Uses logprobs extraction from /api/generate to get calibrated
    relevance scores. The model generates a single token (yes/no),
    and we extract P(yes) as the relevance score.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "dengcao/Qwen3-Reranker-0.6B:Q8_0",
        instruction: str | None = None,
        timeout: float = 30.0,
        max_concurrent: int = 8,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair.

        Returns a relevance score between 0.0 and 1.0.
        """
        prompt = (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

        client = await self._get_client()
        try:
            async with self._semaphore:
                resp = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": RERANKER_SYSTEM,
                        "stream": False,
                        "options": {
                            "num_predict": 1,
                            "temperature": 0.0,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning("Reranker call failed: %s", e)
            return 0.5  # Neutral score on failure

        return self._extract_score(data)

    def _extract_score(self, data: dict) -> float:
        """Extract relevance score from Ollama generate response.

        Tries logprobs first (calibrated probability), falls back to
        parsing the text response as yes/no (binary 1.0/0.0).
        """
        # Try logprobs extraction (preferred — calibrated scores)
        logprobs_list = data.get("logprobs")
        if logprobs_list and len(logprobs_list) > 0:
            top_logprobs = logprobs_list[0].get("top_logprobs", [])
            if top_logprobs:
                yes_logprob = -10.0
                no_logprob = -10.0
                for entry in top_logprobs:
                    token = entry.get("token", "").strip().lower()
                    logprob = entry.get("logprob", -10.0)
                    if token == "yes":
                        yes_logprob = logprob
                    elif token == "no":
                        no_logprob = logprob

                # Softmax over yes/no
                yes_score = math.exp(yes_logprob)
                no_score = math.exp(no_logprob)
                total = yes_score + no_score
                if total > 0:
                    return yes_score / total

        # Fallback: parse text response (binary only)
        response_text = data.get("response", "").strip().lower()
        if "yes" in response_text:
            return 1.0
        elif "no" in response_text:
            return 0.0
        return 0.5  # Can't determine

    async def rerank(
        self,
        query: str,
        documents: list[tuple[int, str]],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Rerank a list of documents by relevance to the query.

        Args:
            query: The search query
            documents: List of (memory_id, text) tuples to rerank
            top_n: Return only top N results (None = return all, sorted)

        Returns:
            List of RerankResult sorted by score descending
        """
        if not documents:
            return []

        start = time.monotonic()

        # Score all documents concurrently
        async def _score(idx: int, memory_id: int, text: str) -> RerankResult:
            score = await self.score_pair(query, text)
            return RerankResult(index=idx, score=score, memory_id=memory_id)

        tasks = [
            _score(i, mem_id, text)
            for i, (mem_id, text) in enumerate(documents)
        ]
        results = await asyncio.gather(*tasks)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        elapsed = time.monotonic() - start
        logger.debug(
            "Reranked %d documents in %.1fms (avg %.1fms/doc)",
            len(documents), elapsed * 1000,
            (elapsed * 1000) / len(documents) if documents else 0,
        )

        if top_n is not None:
            results = results[:top_n]

        return results
