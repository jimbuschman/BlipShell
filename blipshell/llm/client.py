"""Ollama client wrapper (replaces LLMUtility.cs HTTP calls).

Uses the official `ollama` Python package for native tool calling,
streaming, and structured responses.
"""

import asyncio
import logging
from collections import OrderedDict
from typing import Any, AsyncIterator, Optional

import ollama

logger = logging.getLogger(__name__)

# Simple LRU-style response cache
_response_cache: OrderedDict[str, str] = OrderedDict()
_CACHE_MAX_SIZE = 200


class LLMClient:
    """Async wrapper around ollama.AsyncClient with retry/backoff."""

    def __init__(self, host: str = "http://localhost:11434",
                 max_retries: int = 2, retry_base_delay: float = 1.0):
        self.host = host
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._client = ollama.AsyncClient(host=host)

    async def _retry_call(self, func, *args, **kwargs):
        """Retry an async call with exponential backoff.

        Retries up to max_retries times with delays of base*2^attempt seconds.
        """
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)
        raise last_error

    async def chat(
        self,
        messages: list[dict],
        model: str,
        tools: Optional[list[dict]] = None,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """Send a chat request (non-streaming). Returns full response dict."""
        params = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            params["tools"] = tools
        params.update(kwargs)

        async def _do_chat():
            return await self._client.chat(**params)

        try:
            return await self._retry_call(_do_chat)
        except Exception as e:
            logger.error("Chat request failed after retries: %s", e)
            raise

    async def chat_stream(
        self,
        messages: list[dict],
        model: str,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Send a streaming chat request. Yields response chunks.

        On failure, retries the whole stream (not mid-stream).
        """
        params = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            params["tools"] = tools
        params.update(kwargs)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async for chunk in await self._client.chat(**params):
                    yield chunk
                return  # success — stream completed
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Streaming chat failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)

        logger.error("Streaming chat failed after retries: %s", last_error)
        raise last_error

    async def generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> str:
        """Simple generate (non-chat) with optional caching.

        Used for utility tasks like summarization, ranking, etc.
        """
        cache_key = f"{model}:{system or ''}:{prompt}"

        if use_cache and cache_key in _response_cache:
            _response_cache.move_to_end(cache_key)
            return _response_cache[cache_key]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async def _do_generate():
            return await self._client.chat(
                model=model,
                messages=messages,
                stream=False,
                **kwargs,
            )

        try:
            response = await self._retry_call(_do_generate)
            # Handle both object (ollama 0.4+) and dict responses
            msg = getattr(response, "message", None)
            if msg is not None:
                result = getattr(msg, "content", "") or ""
            elif isinstance(response, dict):
                result = response.get("message", {}).get("content", "")
            else:
                result = ""

            if use_cache:
                _response_cache[cache_key] = result
                if len(_response_cache) > _CACHE_MAX_SIZE:
                    _response_cache.popitem(last=False)

            return result.strip()
        except Exception as e:
            logger.error("Generate request failed after retries: %s", e)
            raise

    async def check_health(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            await self._client.list()
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available models on the server."""
        try:
            response = await self._client.list()
            return [m.get("name", "") for m in response.get("models", [])]
        except Exception as e:
            logger.error("Failed to list models: %s", e)
            return []
