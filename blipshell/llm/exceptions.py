"""LLM routing exceptions and error classification.

Distinguishes model-level errors (wrong model name, cloud proxy failures)
from endpoint-level errors (connection refused, timeouts) so the router
can make smart decisions about which endpoints to penalize.
"""

try:
    import openai as _openai
except ImportError:
    _openai = None

try:
    import ollama as _ollama
except ImportError:
    _ollama = None


class RateLimitExhaustedError(Exception):
    """Raised when all rate-limit retries are exhausted for an endpoint.

    Signals the router to try the next endpoint instead of penalizing
    the current one (rate limits are temporary, not endpoint failures).
    """

    def __init__(self, endpoint_name: str = "", message: str = ""):
        self.endpoint_name = endpoint_name
        super().__init__(message or f"Rate limit retries exhausted for {endpoint_name}")


def is_bad_request_error(error: Exception) -> bool:
    """Check if an error is a per-request 400 Bad Request.

    Bad requests are caused by the specific input (content filter, malformed
    content, context overflow) — NOT a model or endpoint failure. The same
    model/endpoint will likely succeed on the next request with different input.

    Should NOT penalize the endpoint or mark the model as permanently failed.
    """
    if _ollama is not None:
        if isinstance(error, _ollama.ResponseError):
            if hasattr(error, "status_code") and error.status_code == 400:
                return True
            msg = str(error).lower()
            if "bad request" in msg:
                return True

    if _openai is not None:
        if isinstance(error, _openai.BadRequestError):
            return True

    return False


def is_model_retired_error(error: Exception) -> bool:
    """Check if an error means the model is permanently gone (HTTP 410).

    A provider retiring a model is the most model-specific failure there is —
    the endpoint serving it is perfectly healthy. Detected by status code
    where available, and by message text because Ollama relays a cloud
    provider's 410 as a ResponseError whose only distinguishing feature is
    the wording (live 2026-07-31: "kimi-k2.5 was retired at ... (status
    code: 410)").
    """
    if getattr(error, "status_code", None) == 410:
        return True
    msg = str(error).lower()
    return "status code: 410" in msg or "was retired" in msg


def is_model_error(error: Exception) -> bool:
    """Check if an error is a model-level problem (not an endpoint failure).

    Model errors should NOT penalize the endpoint because the endpoint
    itself is healthy — the specific model just isn't available or failed.

    Returns True for:
    - ollama.ResponseError with "not found", "model" keywords
    - ollama.ResponseError with HTTP 502/503/504 (cloud proxy errors)
    - HTTP 410 / "was retired" (provider retired the model)
    - RateLimitExhaustedError (rate limit, not endpoint failure)
    - openai.NotFoundError (model not found on cloud provider)
    - openai.RateLimitError (should not disable endpoint)

    Returns False for:
    - Connection refused / timeouts (real endpoint failure)
    - Everything else (unknown — safer to penalize)
    """
    if isinstance(error, RateLimitExhaustedError):
        return True

    # Provider retirement — checked before the SDK-specific branches because
    # it arrives as a plain ResponseError whose text mentions neither "model"
    # nor "not found", so it used to fall through to the penalize-the-endpoint
    # branch. That took `local-cloud` down for a dead session_review model,
    # and `local-cloud` also serves interactive tool_calling.
    if is_model_retired_error(error):
        return True

    if _openai is not None:
        if isinstance(error, _openai.RateLimitError):
            return True
        if isinstance(error, _openai.NotFoundError):
            return True

    if _ollama is not None:
        if isinstance(error, _ollama.ResponseError):
            msg = str(error).lower()
            # Model not found or model-level failure
            if "not found" in msg or "model" in msg:
                return True
            # Cloud proxy errors routed through Ollama
            if any(code in msg for code in ("502", "503", "504")):
                return True

    return False
