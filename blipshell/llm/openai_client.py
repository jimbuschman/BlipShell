"""OpenAI-compatible client for Groq, Gemini, and similar providers.

Uses the openai Python SDK to talk to any OpenAI-compatible API.
Same interface as LLMClient (duck-typed) so the router can use either.
"""

import asyncio
import json
import logging
from collections import OrderedDict
from typing import Any, AsyncIterator, Optional

import httpx
import openai

from blipshell.llm.exceptions import RateLimitExhaustedError

logger = logging.getLogger(__name__)

# Shared response cache (same as client.py)
_response_cache: OrderedDict[str, str] = OrderedDict()
_CACHE_MAX_SIZE = 200


class OpenAICompatClient:
    """Async client for OpenAI-compatible APIs with retry/backoff.

    Drop-in replacement for LLMClient — same method signatures,
    different underlying SDK.
    """

    def __init__(self, base_url: str, api_key: str = "",
                 max_retries: int = 2, retry_base_delay: float = 1.0,
                 timeout: float = 120.0):
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.timeout = timeout
        # Cumulative token usage tracking (for cost estimation)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_requests: int = 0
        # Short connect timeout (5s) so failed cloud endpoints fall back fast.
        # Read timeout stays long for slow model responses.
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "unused",
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def _retry_call(self, func, *args, **kwargs):
        """Retry an async call with exponential backoff.

        Rate limit errors (429) get extra retries with longer delays
        since they're temporary and shouldn't disable the endpoint.
        """
        last_error = None
        rate_limit_retries = 0
        max_rate_limit_retries = 2  # ~15s total (5+10) then let router try next endpoint
        for attempt in range(self.max_retries + 1 + max_rate_limit_retries):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"OpenAI-compatible call timed out after {self.timeout}s"
                )
                logger.warning(
                    "OpenAI call timed out (attempt %d/%d) after %.0fs",
                    attempt + 1, self.max_retries + 1, self.timeout,
                )
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    break
            except openai.RateLimitError as e:
                last_error = e
                rate_limit_retries += 1
                if rate_limit_retries > max_rate_limit_retries:
                    logger.warning(
                        "Rate limit retries exhausted (%d), signaling router to try next endpoint",
                        max_rate_limit_retries,
                    )
                    raise RateLimitExhaustedError(
                        endpoint_name=self.base_url,
                        message=f"Rate limit retries exhausted after {max_rate_limit_retries} attempts: {e}",
                    ) from e
                # Use Retry-After header if available, otherwise exponential backoff
                delay = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = min(float(retry_after), 120.0)
                        except (ValueError, TypeError):
                            pass
                if delay is None:
                    # Exponential: 10s, 30s — long enough for TPM recovery
                    delay = 10.0 * (3 ** (rate_limit_retries - 1))
                logger.warning(
                    "Rate limited (%d/%d), waiting %.0fs...",
                    rate_limit_retries, max_rate_limit_retries, delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "OpenAI call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)
                else:
                    break
        raise last_error

    async def chat(
        self,
        messages: list[dict],
        model: str,
        tools: Optional[list[dict]] = None,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """Send a chat completion request (non-streaming).

        Returns a dict with a 'message' key containing 'content' and
        optionally 'tool_calls', matching the shape callers expect.
        """
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            params["tools"] = self._convert_tools(tools)
        # Ignore Ollama-specific kwargs
        for key in ("options", "think", "format"):
            kwargs.pop(key, None)
        params.update(kwargs)

        async def _do_chat():
            return await self._client.chat.completions.create(**params)

        try:
            response = await self._retry_call(_do_chat)
            return self._normalize_response(response)
        except Exception as e:
            logger.error("OpenAI chat request failed after retries: %s", e)
            raise

    async def chat_stream(
        self,
        messages: list[dict],
        model: str,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Send a streaming chat request. Yields response chunks.

        Chunk format matches what callers expect from LLMClient:
        each chunk has a 'message' dict with 'content'.
        """
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            params["tools"] = self._convert_tools(tools)
        for key in ("options", "think", "format"):
            kwargs.pop(key, None)
        params.update(kwargs)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                stream = await self._client.chat.completions.create(**params)
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield {"message": {"content": delta.content}}
                return
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
        """Generate a response (non-chat). Used for summarization, ranking, etc."""
        cache_key = f"{model}:{system or ''}:{prompt}"

        if use_cache and cache_key in _response_cache:
            _response_cache.move_to_end(cache_key)
            return _response_cache[cache_key]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Strip Ollama-specific kwargs
        for key in ("options", "think", "format"):
            kwargs.pop(key, None)

        async def _do_generate():
            return await self._client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **kwargs,
            )

        try:
            response = await self._retry_call(_do_generate)
            # Accumulate token usage
            if hasattr(response, "usage") and response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens or 0
                self.total_completion_tokens += response.usage.completion_tokens or 0
                self.total_requests += 1
            result = response.choices[0].message.content or ""

            if use_cache:
                _response_cache[cache_key] = result
                if len(_response_cache) > _CACHE_MAX_SIZE:
                    _response_cache.popitem(last=False)

            return result.strip()
        except Exception as e:
            logger.error("OpenAI generate request failed after retries: %s", e)
            raise

    def reset_usage(self):
        """Reset cumulative token counters (e.g., between benchmark tasks)."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0

    def get_usage(self) -> dict:
        """Return current cumulative token usage."""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "requests": self.total_requests,
        }

    async def check_health(self) -> bool:
        """Check if the API is reachable by listing models."""
        try:
            await asyncio.wait_for(
                self._client.models.list(), timeout=10.0,
            )
            return True
        except Exception:
            return False

    def _normalize_response(self, response) -> dict:
        """Convert OpenAI response to the dict shape callers expect.

        Returns a dict-like object with .message.content (matching ollama
        response objects) for compatibility with existing code.
        Also accumulates token usage for cost tracking.
        """
        # Accumulate token usage (OpenAI/OpenRouter include this)
        if hasattr(response, "usage") and response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens or 0
            self.total_completion_tokens += response.usage.completion_tokens or 0
            self.total_requests += 1

        choice = response.choices[0] if response.choices else None
        if not choice:
            return {"message": {"content": "", "role": "assistant"}}

        msg = choice.message
        result = {
            "message": {
                "content": msg.content or "",
                "role": msg.role or "assistant",
            }
        }

        # Preserve tool calls in OpenAI wire format so they can be sent back
        # in conversation history without format errors. Arguments stay as
        # JSON strings (parsed to dicts only in _extract_tool_call_info).
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id or "",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,  # keep as JSON string
                    }
                })
            result["message"]["tool_calls"] = tool_calls

        return result

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert Ollama-style tool defs to OpenAI format if needed.

        Ollama tools have shape: {"type": "function", "function": {...}}
        OpenAI expects the same shape, so mostly pass-through.
        """
        converted = []
        for tool in tools:
            if "type" in tool and "function" in tool:
                # Already in OpenAI format
                converted.append(tool)
            elif "name" in tool:
                # Bare function dict — wrap it
                converted.append({"type": "function", "function": tool})
            else:
                converted.append(tool)
        return converted
