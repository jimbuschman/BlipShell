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


def is_model_error(error: Exception) -> bool:
    """Check if an error is a model-level problem (not an endpoint failure).

    Model errors should NOT penalize the endpoint because the endpoint
    itself is healthy — the specific model just isn't available or failed.

    Returns True for:
    - ollama.ResponseError with "not found", "model" keywords
    - ollama.ResponseError with HTTP 502/503/504 (cloud proxy errors)
    - RateLimitExhaustedError (rate limit, not endpoint failure)
    - openai.NotFoundError (model not found on cloud provider)
    - openai.RateLimitError (should not disable endpoint)

    Returns False for:
    - Connection refused / timeouts (real endpoint failure)
    - Everything else (unknown — safer to penalize)
    """
    if isinstance(error, RateLimitExhaustedError):
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
