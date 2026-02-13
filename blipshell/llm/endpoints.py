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
from blipshell.models.config import EndpointConfig

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

    def __init__(self, configs: list[EndpointConfig]):
        self._lock = asyncio.Lock()
        self._endpoints: list[Endpoint] = []
        for cfg in configs:
            ep = Endpoint(
                name=cfg.name,
                url=cfg.url,
                roles=cfg.roles,
                priority=cfg.priority,
                max_concurrent=cfg.max_concurrent,
                enabled=cfg.enabled,
                client=LLMClient(host=cfg.url),
            )
            self._endpoints.append(ep)

    def get_endpoint_for_role(self, role: str) -> Optional[Endpoint]:
        """Get the best available endpoint that supports the given role.

        Selection priority:
        1. Supports the requested role
        2. Enabled and can accept requests
        3. Highest priority value
        4. Fewest active requests (load balancing)
        """
        candidates = [
            ep for ep in self._endpoints
            if role in ep.roles and ep.can_accept_request
        ]
        if not candidates:
            # Fallback: any enabled endpoint
            candidates = [ep for ep in self._endpoints if ep.can_accept_request]
        if not candidates:
            return None

        return sorted(
            candidates,
            key=lambda e: (-e.priority, e.active_requests),
        )[0]

    def get_client_for_role(self, role: str) -> Optional[LLMClient]:
        """Get the LLMClient for the best endpoint matching a role."""
        ep = self.get_endpoint_for_role(role)
        return ep.client if ep else None

    def mark_failed(self, endpoint_name: str):
        """Mark an endpoint as failed."""
        for ep in self._endpoints:
            if ep.name == endpoint_name:
                ep.record_failure()
                break

    def mark_success(self, endpoint_name: str, response_time: float):
        """Mark an endpoint request as successful."""
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
