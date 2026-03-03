"""Tests for LLM routing fallback system (CLAUDE.md item 1).

Verifies:
- is_model_error() classifies errors correctly
- generate() cascades to fallback on model/rate-limit errors
- Endpoint not penalized for model-level errors
- Context tokens (num_ctx) passed through on both primary and fallback paths
- Think config respected on fallback
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from blipshell.llm.exceptions import RateLimitExhaustedError, is_model_error
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.models.config import ModelsConfig


# ── is_model_error() classification ──────────────────────────────────────────


class TestIsModelError:
    def test_rate_limit_exhausted(self):
        err = RateLimitExhaustedError("groq", "429 rate limit")
        assert is_model_error(err) is True

    def test_generic_exception_is_not_model_error(self):
        assert is_model_error(RuntimeError("something broke")) is False

    def test_connection_error_is_not_model_error(self):
        assert is_model_error(ConnectionError("refused")) is False

    def test_timeout_is_not_model_error(self):
        assert is_model_error(TimeoutError("timed out")) is False

    def test_ollama_model_not_found(self):
        """ollama.ResponseError with 'not found' is model-level."""
        try:
            import ollama
            err = ollama.ResponseError("model 'xyz' not found")
            assert is_model_error(err) is True
        except ImportError:
            pytest.skip("ollama not installed")

    def test_ollama_502_proxy_error(self):
        """ollama.ResponseError with 502/503/504 is model-level (cloud proxy)."""
        try:
            import ollama
            err = ollama.ResponseError("502 Bad Gateway")
            assert is_model_error(err) is True
        except ImportError:
            pytest.skip("ollama not installed")

    def test_openai_rate_limit(self):
        """openai.RateLimitError is model-level."""
        try:
            import openai
            err = openai.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
            assert is_model_error(err) is True
        except ImportError:
            pytest.skip("openai not installed")

    def test_openai_not_found(self):
        """openai.NotFoundError is model-level."""
        try:
            import openai
            err = openai.NotFoundError(
                message="Model not found",
                response=MagicMock(status_code=404),
                body=None,
            )
            assert is_model_error(err) is True
        except ImportError:
            pytest.skip("openai not installed")


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_endpoint(name="ep1", context_tokens=65536, models=None):
    """Create a mock endpoint with sensible defaults."""
    ep = MagicMock()
    ep.name = name
    ep.context_tokens = context_tokens
    ep.models = models or {}
    ep.client = AsyncMock()
    ep.client.generate = AsyncMock(return_value="response from " + name)
    ep.start_request = MagicMock()
    ep.complete_request = MagicMock()
    ep.record_success = MagicMock()
    ep.record_failure = MagicMock()
    ep.would_exceed_tpm = MagicMock(return_value=False)  # no TPM limit by default
    return ep


def _make_models_config(**overrides):
    """Create a ModelsConfig with defaults + overrides."""
    defaults = dict(
        reasoning="model-a",
        tool_calling="model-a",
        coding="model-a",
        summarization="model-a",
        ranking="model-a",
        importance="model-a",
        embedding="embed-model",
        # Fallbacks
        reasoning_fallback="model-b",
        tool_calling_fallback="model-b",
        coding_fallback="model-b",
        summarization_fallback="model-b",
        ranking_fallback="model-b",
        importance_fallback="model-b",
        ranking_importance_fallback="model-b",
        fallback_think=False,
    )
    defaults.update(overrides)
    return ModelsConfig(**defaults)


@pytest.fixture
def endpoint_manager():
    mgr = AsyncMock()
    return mgr


@pytest.fixture
def models_config():
    return _make_models_config()


@pytest.fixture
def router(models_config, endpoint_manager):
    return LLMRouter(models_config, endpoint_manager)


# ── generate() — primary succeeds ───────────────────────────────────────────


class TestGeneratePrimarySuccess:
    async def test_returns_result_on_success(self, router, endpoint_manager):
        ep = _make_endpoint()
        endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=ep)
        result = await router.generate("reasoning", "hello")
        assert result == "response from ep1"

    async def test_passes_context_tokens_as_num_ctx(self, router, endpoint_manager):
        ep = _make_endpoint(context_tokens=131072)
        endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=ep)
        await router.generate("reasoning", "hello")
        call_kwargs = ep.client.generate.call_args
        assert call_kwargs.kwargs.get("options") == {"num_ctx": 131072}

    async def test_no_num_ctx_when_context_tokens_is_none(self, router, endpoint_manager):
        ep = _make_endpoint(context_tokens=None)
        endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=ep)
        await router.generate("reasoning", "hello")
        call_kwargs = ep.client.generate.call_args
        assert "options" not in call_kwargs.kwargs

    async def test_records_success_not_failure(self, router, endpoint_manager):
        ep = _make_endpoint()
        endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=ep)
        await router.generate("reasoning", "hello")
        ep.record_success.assert_called_once()
        ep.record_failure.assert_not_called()


# ── generate() — fallback on model error ─────────────────────────────────────


class TestGenerateFallbackOnModelError:
    async def test_rate_limit_triggers_fallback(self, router, endpoint_manager):
        primary_ep = _make_endpoint("primary")
        primary_ep.client.generate = AsyncMock(
            side_effect=RateLimitExhaustedError("primary", "429")
        )
        fallback_ep = _make_endpoint("fallback", context_tokens=196608)
        # First call returns primary, second (for fallback) returns fallback_ep
        endpoint_manager.get_endpoint_for_role = AsyncMock(
            side_effect=[primary_ep, fallback_ep]
        )
        result = await router.generate("reasoning", "hello")
        assert result == "response from fallback"

    async def test_model_error_does_not_penalize_endpoint(self, router, endpoint_manager):
        primary_ep = _make_endpoint("primary")
        primary_ep.client.generate = AsyncMock(
            side_effect=RateLimitExhaustedError("primary", "429")
        )
        fallback_ep = _make_endpoint("fallback")
        endpoint_manager.get_endpoint_for_role = AsyncMock(
            side_effect=[primary_ep, fallback_ep]
        )
        await router.generate("reasoning", "hello")
        # Endpoint should NOT be penalized for rate limit errors
        primary_ep.record_failure.assert_not_called()

    async def test_model_marked_failed_after_error(self, router, endpoint_manager):
        primary_ep = _make_endpoint("primary")
        primary_ep.client.generate = AsyncMock(
            side_effect=RateLimitExhaustedError("primary", "429")
        )
        fallback_ep = _make_endpoint("fallback")
        endpoint_manager.get_endpoint_for_role = AsyncMock(
            side_effect=[primary_ep, fallback_ep]
        )
        await router.generate("reasoning", "hello")
        assert router.is_model_failed("model-a")


# ── generate() — fallback on endpoint error ──────────────────────────────────


class TestGenerateFallbackOnEndpointError:
    async def test_connection_error_penalizes_endpoint(self, router, endpoint_manager):
        primary_ep = _make_endpoint("primary")
        primary_ep.client.generate = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        fallback_ep = _make_endpoint("fallback")
        endpoint_manager.get_endpoint_for_role = AsyncMock(
            side_effect=[primary_ep, fallback_ep]
        )
        await router.generate("reasoning", "hello")
        primary_ep.record_failure.assert_called_once()


# ── generate() — no fallback configured ──────────────────────────────────────


class TestGenerateNoFallback:
    async def test_raises_when_fallback_same_as_primary(self, endpoint_manager):
        # Fallback is the same model as primary — should not retry
        models = _make_models_config(reasoning_fallback="model-a")
        router = LLMRouter(models, endpoint_manager)
        ep = _make_endpoint()
        ep.client.generate = AsyncMock(side_effect=RuntimeError("boom"))
        endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=ep)
        with pytest.raises(RuntimeError, match="boom"):
            await router.generate("reasoning", "hello")


# ── generate() — both primary and fallback fail ──────────────────────────────


class TestGenerateBothFail:
    async def test_raises_primary_error_when_fallback_also_fails(
        self, router, endpoint_manager,
    ):
        primary_ep = _make_endpoint("primary")
        primary_ep.client.generate = AsyncMock(
            side_effect=RateLimitExhaustedError("primary", "429")
        )
        fallback_ep = _make_endpoint("fallback")
        fallback_ep.client.generate = AsyncMock(
            side_effect=RuntimeError("fallback also dead")
        )
        endpoint_manager.get_endpoint_for_role = AsyncMock(
            side_effect=[primary_ep, fallback_ep]
        )
        with pytest.raises(RateLimitExhaustedError):
            await router.generate("reasoning", "hello")


# ── generate() — context tokens on fallback ──────────────────────────────────


class TestFallbackContextTokens:
    async def test_fallback_uses_its_own_context_tokens(
        self, router, endpoint_manager,
    ):
        primary_ep = _make_endpoint("primary", context_tokens=65536)
        primary_ep.client.generate = AsyncMock(
            side_effect=ConnectionError("down")
        )
        fallback_ep = _make_endpoint("fallback", context_tokens=196608)
        endpoint_manager.get_endpoint_for_role = AsyncMock(
            side_effect=[primary_ep, fallback_ep]
        )
        await router.generate("reasoning", "hello")
        # Fallback endpoint should have used its own context_tokens
        call_kwargs = fallback_ep.client.generate.call_args
        assert call_kwargs.kwargs.get("options") == {"num_ctx": 196608}


# ── generate() — think config on fallback ────────────────────────────────────


class TestFallbackThinkConfig:
    async def test_fallback_think_disabled(self, endpoint_manager):
        models = _make_models_config(fallback_think=False)
        router = LLMRouter(models, endpoint_manager)
        primary_ep = _make_endpoint("primary")
        primary_ep.client.generate = AsyncMock(side_effect=RuntimeError("down"))
        fallback_ep = _make_endpoint("fallback")
        endpoint_manager.get_endpoint_for_role = AsyncMock(
            side_effect=[primary_ep, fallback_ep]
        )
        await router.generate("reasoning", "hello", think=True)
        call_kwargs = fallback_ep.client.generate.call_args
        assert call_kwargs.kwargs.get("think") is False

    async def test_fallback_think_not_disabled_when_allowed(self, endpoint_manager):
        models = _make_models_config(fallback_think=True)
        router = LLMRouter(models, endpoint_manager)
        primary_ep = _make_endpoint("primary")
        primary_ep.client.generate = AsyncMock(side_effect=RuntimeError("down"))
        fallback_ep = _make_endpoint("fallback")
        endpoint_manager.get_endpoint_for_role = AsyncMock(
            side_effect=[primary_ep, fallback_ep]
        )
        await router.generate("reasoning", "hello", think=True)
        call_kwargs = fallback_ep.client.generate.call_args
        # think should NOT be overridden to False
        assert call_kwargs.kwargs.get("think") is not False


# ── generate() — skips to fallback when model already failed ─────────────────


class TestSkipToFallback:
    async def test_skips_primary_when_model_already_failed(
        self, router, endpoint_manager,
    ):
        router.mark_model_failed("model-a")
        ep = _make_endpoint()
        endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=ep)
        result = await router.generate("reasoning", "hello")
        # Should have called generate with fallback model, not primary
        call_kwargs = ep.client.generate.call_args
        assert call_kwargs.kwargs.get("model") == "model-b"


# ── Per-endpoint model overrides ─────────────────────────────────────────────


class TestPerEndpointModelOverrides:
    async def test_endpoint_model_override_used(self, router, endpoint_manager):
        ep = _make_endpoint(models={"reasoning": "custom-model"})
        endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=ep)
        await router.generate("reasoning", "hello")
        call_kwargs = ep.client.generate.call_args
        assert call_kwargs.kwargs.get("model") == "custom-model"


# ── No endpoint available ────────────────────────────────────────────────────


class TestNoEndpointAvailable:
    async def test_raises_when_no_endpoint(self, router, endpoint_manager):
        endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=None)
        with pytest.raises(RuntimeError, match="No available endpoint"):
            await router.generate("reasoning", "hello")
