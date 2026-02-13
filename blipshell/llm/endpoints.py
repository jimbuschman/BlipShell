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
from blipshell.models.config import EndpointConfig, LLMConfig

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
    client: Optional[LLMClient] = field(default=None, repr=False)

    @property
    def can_accept_request(self) -> bool:
        return self.enabled and self.active_requests < self.max_concurrent

    def start_request(self):
        self.active_requests += 1
        self.last_used = time.time()

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
    """Manages multiple Ollama endpoints with role-based routing.

    Port of EndpointManager.cs with enhancements:
    - Config-driven endpoints
    - Role-based selection (reasoning, summarization, etc.)
    - Async health polling
    """

    def __init__(self, configs: list[EndpointConfig], llm_config: LLMConfig | None = None):
        self._lock = asyncio.Lock()
        self._endpoints: list[Endpoint] = []
        self._last_routed: dict[str, str] = {}  # role → endpoint name
        llm_cfg = llm_config or LLMConfig()
        for cfg in configs:
            ep = Endpoint(
                name=cfg.name,
                url=cfg.url,
                roles=cfg.roles,
                priority=cfg.priority,
                max_concurrent=cfg.max_concurrent,
                enabled=cfg.enabled,
                client=LLMClient(
                    host=cfg.url,
                    max_retries=llm_cfg.max_retries,
                    retry_base_delay=llm_cfg.retry_base_delay,
                ),
            )
            self._endpoints.append(ep)

    async def get_endpoint_for_role(self, role: str) -> Optional[Endpoint]:
        """Get the best available endpoint that supports the given role.

        Selection priority:
        1. Supports the requested role
        2. Enabled and can accept requests
        3. Highest priority value
        4. Fewest active requests (load balancing)
        """
        async with self._lock:
            candidates = [
                ep for ep in self._endpoints
                if role in ep.roles and ep.can_accept_request
            ]
            is_fallback = False
            if not candidates:
                # Fallback: any enabled endpoint
                candidates = [ep for ep in self._endpoints if ep.can_accept_request]
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
        """Check a single endpoint's health."""
        try:
            healthy = await ep.client.check_health()
            if healthy and not ep.enabled and ep.failure_count > 0:
                ep.enabled = True
                ep.failure_count = 0
                logger.info("Endpoint %s re-enabled after health check", ep.name)
            elif not healthy and ep.enabled:
                ep.record_failure()
        except Exception as e:
            logger.debug("Health check failed for %s: %s", ep.name, e)
            ep.record_failure()

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

    def start_health_loop(self, interval: int = 60) -> asyncio.Task:
        """Start a background loop that periodically checks all endpoints.

        Re-enables endpoints that come back online, disables ones that go down.
        Returns the asyncio.Task so the caller can cancel it on shutdown.
        """
        async def _loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.health_check_all()
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
        return [
            {
                "name": ep.name,
                "url": ep.url,
                "enabled": ep.enabled,
                "roles": ep.roles,
                "active_requests": ep.active_requests,
                "max_concurrent": ep.max_concurrent,
                "failure_count": ep.failure_count,
                "success_count": ep.success_count,
            }
            for ep in self._endpoints
        ]
