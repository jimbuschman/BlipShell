"""Multi-endpoint management (port of EndpointManager.cs).

Handles priority selection, health tracking, failure counting,
and load balancing by active requests.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from blipshell.llm.client import LLMClient
from blipshell.models.config import EndpointConfig, LLMConfig, resolve_env_vars

logger = logging.getLogger(__name__)


@dataclass
class Endpoint:
    """Runtime state for an LLM endpoint."""
    name: str
    url: str
    roles: list[str]
    priority: int
    max_concurrent: int
    enabled: bool = True
    failure_count: int = 0
    success_count: int = 0
    active_requests: int = 0
    last_used: float = field(default_factory=time.time)
    last_response_time: float = 1.0  # seconds
    context_tokens: Optional[int] = None  # per-endpoint context window
    provider: str = "ollama"
    models: dict[str, str] = field(default_factory=dict)  # per-endpoint model overrides
    pii_sanitize: Optional[bool] = None  # None = auto (true for openai, false for ollama)
    rate_limit_rpm: Optional[int] = None
    rate_limit_rpd: Optional[int] = None
    rate_limit_tpm: Optional[int] = None
    _minute_requests: list[float] = field(default_factory=list, repr=False)
    _minute_tokens: list[tuple[float, int]] = field(default_factory=list, repr=False)
    _day_requests: int = field(default=0, repr=False)
    _day_reset: float = field(default_factory=lambda: time.time() + 86400, repr=False)
    client: Optional[LLMClient] = field(default=None, repr=False)

    def _is_rate_limited(self) -> bool:
        """Check if this endpoint is currently rate-limited."""
        now = time.time()

        if self.rate_limit_rpm is not None:
            # Prune timestamps older than 60s
            self._minute_requests = [t for t in self._minute_requests if now - t < 60]
            if len(self._minute_requests) >= self.rate_limit_rpm:
                return True

        if self.rate_limit_tpm is not None:
            self._minute_tokens = [(t, n) for t, n in self._minute_tokens if now - t < 60]
            used = sum(n for _, n in self._minute_tokens)
            if used >= self.rate_limit_tpm:
                return True

        if self.rate_limit_rpd is not None:
            if now >= self._day_reset:
                self._day_requests = 0
                self._day_reset = now + 86400
            if self._day_requests >= self.rate_limit_rpd:
                return True

        return False

    def would_exceed_tpm(self, estimated_tokens: int) -> bool:
        """Check if a request of this size would exceed the TPM budget.

        Returns False if no TPM limit is configured.
        """
        if self.rate_limit_tpm is None:
            return False
        now = time.time()
        self._minute_tokens = [(t, n) for t, n in self._minute_tokens if now - t < 60]
        used = sum(n for _, n in self._minute_tokens)
        return (used + estimated_tokens) > self.rate_limit_tpm

    @property
    def should_sanitize_pii(self) -> bool:
        """Whether PII should be sanitized before sending to this endpoint.

        If pii_sanitize is explicitly set, use that. Otherwise auto-detect:
        true for cloud providers (openai), false for local (ollama).
        """
        if self.pii_sanitize is not None:
            return self.pii_sanitize
        return self.provider == "openai"

    @property
    def can_accept_request(self) -> bool:
        return (self.enabled
                and self.active_requests < self.max_concurrent
                and not self._is_rate_limited())

    def start_request(self, token_count: int = 0):
        self.active_requests += 1
        now = time.time()
        self.last_used = now
        # Track for rate limiting
        self._minute_requests.append(now)
        self._day_requests += 1
        if token_count > 0 and self.rate_limit_tpm is not None:
            self._minute_tokens.append((now, token_count))

    def complete_request(self):
        self.active_requests = max(0, self.active_requests - 1)

    def record_success(self, response_time: float):
        self.failure_count = 0
        self.success_count += 1
        self.last_response_time = response_time

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= 3:
            self.enabled = False
            logger.warning("Endpoint %s disabled after %d failures", self.name, self.failure_count)


class EndpointManager:
    """Manages multiple LLM endpoints with role-based routing.

    Supports both Ollama and OpenAI-compatible endpoints (Groq, Gemini, etc.).
    Config-driven endpoints with role-based selection and async health polling.
    """

    def __init__(self, configs: list[EndpointConfig], llm_config: LLMConfig | None = None):
        self._lock = asyncio.Lock()
        self._endpoints: list[Endpoint] = []
        self._last_routed: dict[str, str] = {}  # role → endpoint name
        llm_cfg = llm_config or LLMConfig()
        for cfg in configs:
            client = self._create_client(cfg, llm_cfg)
            ep = Endpoint(
                name=cfg.name,
                url=cfg.url,
                roles=cfg.roles,
                priority=cfg.priority,
                max_concurrent=cfg.max_concurrent,
                enabled=cfg.enabled,
                context_tokens=cfg.context_tokens,
                provider=cfg.provider,
                models=cfg.models,
                pii_sanitize=cfg.pii_sanitize,
                rate_limit_rpm=cfg.rate_limit_rpm,
                rate_limit_rpd=cfg.rate_limit_rpd,
                rate_limit_tpm=cfg.rate_limit_tpm,
                client=client,
            )
            self._endpoints.append(ep)

    @staticmethod
    def _create_client(cfg: EndpointConfig, llm_cfg: LLMConfig):
        """Create the appropriate client based on provider type."""
        if cfg.provider == "openai":
            from blipshell.llm.openai_client import OpenAICompatClient
            return OpenAICompatClient(
                base_url=cfg.url,
                api_key=resolve_env_vars(cfg.api_key) or "",
                max_retries=llm_cfg.max_retries,
                retry_base_delay=llm_cfg.retry_base_delay,
                timeout=llm_cfg.timeout,
            )
        return LLMClient(
            host=cfg.url,
            max_retries=llm_cfg.max_retries,
            retry_base_delay=llm_cfg.retry_base_delay,
            timeout=llm_cfg.timeout,
        )

    async def get_endpoint_for_role(self, role: str, exclude: str | None = None, min_context_tokens: int | None = None) -> Optional[Endpoint]:
        """Get the best available endpoint that supports the given role.

        Selection priority:
        1. Supports the requested role
        2. Enabled and can accept requests
        3. Has sufficient context window (if min_context_tokens specified)
        4. Highest priority value
        5. Fewest active requests (load balancing)

        Args:
            exclude: Endpoint name to skip (used for fallback to avoid retrying the same endpoint).
            min_context_tokens: Minimum context window required. Endpoints with smaller
                context are filtered out, preferring endpoints that can handle the request
                without chunking.
        """
        async with self._lock:
            candidates = [
                ep for ep in self._endpoints
                if role in ep.roles and ep.can_accept_request and ep.name != exclude
            ]
            # Filter by minimum context window if specified
            if min_context_tokens and candidates:
                big_enough = [
                    ep for ep in candidates
                    if ep.context_tokens and ep.context_tokens >= min_context_tokens
                ]
                if big_enough:
                    candidates = big_enough
                # If no endpoint is big enough, fall through to original candidates (will chunk)
            is_fallback = False
            if not candidates:
                # Fallback: any enabled endpoint (still excluding the failed one)
                candidates = [
                    ep for ep in self._endpoints
                    if ep.can_accept_request and ep.name != exclude
                ]
                is_fallback = True
            if not candidates:
                return None

            chosen = sorted(
                candidates,
                key=lambda e: (-e.priority, e.active_requests),
            )[0]

            if is_fallback:
                logger.info("No endpoint for role '%s', falling back to '%s'", role, chosen.name)
            else:
                logger.debug("Routing '%s' → endpoint '%s'", role, chosen.name)

            # Track last routing decision
            self._last_routed[role] = chosen.name
            return chosen

    async def start_request_atomic(self, endpoint: Endpoint, token_count: int = 0):
        """Atomically start a request under the manager lock.

        Ensures no race between rate limit check (can_accept_request) and
        request registration (start_request) when multiple coroutines
        are concurrently routing requests.
        """
        async with self._lock:
            endpoint.start_request(token_count=token_count)

    async def get_client_for_role(self, role: str) -> Optional[LLMClient]:
        """Get the LLMClient for the best endpoint matching a role."""
        ep = await self.get_endpoint_for_role(role)
        return ep.client if ep else None

    def get_client_by_name(self, name: str) -> Optional[LLMClient]:
        """Get the LLMClient for a specific endpoint by name."""
        for ep in self._endpoints:
            if ep.name == name and ep.enabled:
                return ep.client
        return None

    def get_first_remote_name(self) -> Optional[str]:
        """Get the name of the first available remote (non-local) endpoint.

        Returns None if no remote endpoint is enabled.
        """
        for ep in self._endpoints:
            if ep.name != "local" and ep.enabled:
                return ep.name
        return None

    def get_context_tokens_for_role(self, role: str, default: int = 32768) -> int:
        """Get the context window size for the endpoint that handles a role.

        Checks the last routed endpoint for the role, or finds the best
        candidate. Returns the endpoint's context_tokens if set, otherwise
        the provided default.
        """
        # Check last routed endpoint first
        last_name = self._last_routed.get(role)
        if last_name:
            for ep in self._endpoints:
                if ep.name == last_name and ep.context_tokens:
                    return ep.context_tokens

        # Fallback: find best endpoint for this role
        for ep in self._endpoints:
            if role in ep.roles and ep.enabled and ep.context_tokens:
                return ep.context_tokens

        return default

    async def mark_failed(self, endpoint_name: str):
        """Mark an endpoint as failed."""
        async with self._lock:
            for ep in self._endpoints:
                if ep.name == endpoint_name:
                    ep.record_failure()
                    break

    async def mark_success(self, endpoint_name: str, response_time: float):
        """Mark an endpoint request as successful."""
        async with self._lock:
            for ep in self._endpoints:
                if ep.name == endpoint_name:
                    ep.record_success(response_time)
                    break

    async def health_check_all(self):
        """Check health of all endpoints concurrently."""
        tasks = []
        for ep in self._endpoints:
            tasks.append(self._check_endpoint(ep))
        await asyncio.gather(*tasks)

    async def _check_endpoint(self, ep: Endpoint):
        """Check a single endpoint's health.

        Health checks can only RE-ENABLE disabled endpoints, never disable
        working ones. Only real request failures (in router/agent_chat)
        should count toward the 3-strike disable. This prevents flaky health
        checks (e.g. OpenRouter's huge /models response timing out) from
        killing a perfectly functional endpoint.
        """
        try:
            healthy = await ep.client.check_health()
            if healthy and not ep.enabled and ep.failure_count > 0:
                ep.enabled = True
                ep.failure_count = 0
                logger.info("Endpoint %s re-enabled after health check", ep.name)
            elif not healthy:
                logger.debug("Health check unhealthy for %s (not penalizing)", ep.name)
        except Exception as e:
            logger.debug("Health check failed for %s: %s (not penalizing)", ep.name, e)

    async def startup_health_check(self):
        """Run a health check on startup to detect available endpoints.

        Disables endpoints that aren't reachable so the router falls back
        to local automatically.
        """
        logger.info("Running startup health check on %d endpoints...", len(self._endpoints))
        await self.health_check_all()
        for ep in self._endpoints:
            status = "[green]available[/green]" if ep.enabled else "[red]unavailable[/red]"
            logger.info("Endpoint '%s' (%s): %s", ep.name, ep.url,
                       "available" if ep.enabled else "unavailable")

    def start_health_loop(self, interval: int = 60, on_check=None) -> asyncio.Task:
        """Start a background loop that periodically checks all endpoints.

        Re-enables endpoints that come back online, disables ones that go down.
        on_check is an optional callback called after each health check cycle
        (used to clear model failure caches so primaries get retried).
        Returns the asyncio.Task so the caller can cancel it on shutdown.
        """
        async def _loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.health_check_all()
                    if on_check:
                        on_check()
                except Exception as e:
                    logger.debug("Health check loop error: %s", e)

        task = asyncio.create_task(_loop())
        logger.info("Endpoint health check loop started (every %ds)", interval)
        return task

    def get_routing_summary(self) -> dict[str, str]:
        """Get a summary of which endpoint is handling which role.

        Returns dict like {"reasoning": "local", "summarization": "remote-pc"}.
        """
        summary = {}
        for ep in self._endpoints:
            if not ep.enabled:
                continue
            for role in ep.roles:
                if role not in summary:
                    summary[role] = ep.name
                else:
                    # Higher priority wins
                    current = next(
                        (e for e in self._endpoints if e.name == summary[role]), None
                    )
                    if current and ep.priority > current.priority:
                        summary[role] = ep.name
        return summary

    def get_status(self) -> list[dict]:
        """Get status of all endpoints for display."""
        statuses = []
        for ep in self._endpoints:
            status = {
                "name": ep.name,
                "url": ep.url,
                "provider": ep.provider,
                "enabled": ep.enabled,
                "roles": ep.roles,
                "active_requests": ep.active_requests,
                "max_concurrent": ep.max_concurrent,
                "failure_count": ep.failure_count,
                "success_count": ep.success_count,
            }
            if ep.rate_limit_rpm is not None:
                status["rate_limit_rpm"] = ep.rate_limit_rpm
            if ep.rate_limit_rpd is not None:
                status["rate_limit_rpd"] = ep.rate_limit_rpd
            if ep.models:
                status["models"] = ep.models
            statuses.append(status)
        return statuses
