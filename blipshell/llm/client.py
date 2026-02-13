"""Ollama client wrapper (replaces LLMUtility.cs HTTP calls).

Uses the official `ollama` Python package for native tool calling,
streaming, and structured responses.
"""

import logging
from collections import OrderedDict
from typing import Any, AsyncIterator, Optional

import ollama

logger = logging.getLogger(__name__)

# Simple LRU-style response cache
_response_cache: OrderedDict[str, str] = OrderedDict()
_CACHE_MAX_SIZE = 200


class LLMClient:
    """Async wrapper around ollama.AsyncClient."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self._client = ollama.AsyncClient(host=host)

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

        try:
            response = await self._client.chat(**params)
            return response
        except Exception as e:
            logger.error("Chat request failed: %s", e)
            raise

    async def chat_stream(
        self,
        messages: list[dict],
        model: str,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Send a streaming chat request. Yields response chunks."""
        params = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            params["tools"] = tools
        params.update(kwargs)

        try:
            async for chunk in await self._client.chat(**params):
                yield chunk
        except Exception as e:
            logger.error("Streaming chat failed: %s", e)
            raise

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

        try:
            response = await self._client.chat(
                model=model,
                messages=messages,
                stream=False,
                **kwargs,
            )
            result = response.get("message", {}).get("content", "")

            if use_cache:
                _response_cache[cache_key] = result
                if len(_response_cache) > _CACHE_MAX_SIZE:
                    _response_cache.popitem(last=False)

            return result.strip()
        except Exception as e:
            logger.error("Generate request failed: %s", e)
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
