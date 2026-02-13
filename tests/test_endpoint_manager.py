"""Tests for the endpoint manager (llm/endpoints.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from blipshell.llm.endpoints import Endpoint, EndpointManager
from blipshell.models.config import EndpointConfig, LLMConfig


@pytest.fixture
def endpoint_configs():
    return [
        EndpointConfig(
            name="primary",
            url="http://localhost:11434",
            roles=["reasoning", "tool_calling"],
            priority=2,
            max_concurrent=3,
        ),
        EndpointConfig(
            name="secondary",
            url="http://localhost:11435",
            roles=["summarization", "ranking"],
            priority=1,
            max_concurrent=2,
        ),
    ]


@pytest.fixture
def endpoint_manager(endpoint_configs):
    return EndpointManager(endpoint_configs, LLMConfig(max_retries=1, retry_base_delay=0.1))


class TestEndpoint:
    def test_can_accept_request(self):
        ep = Endpoint(name="test", url="http://localhost", roles=["reasoning"],
                      priority=1, max_concurrent=2)
        assert ep.can_accept_request

    def test_cannot_accept_when_full(self):
        ep = Endpoint(name="test", url="http://localhost", roles=["reasoning"],
                      priority=1, max_concurrent=1)
        ep.start_request()
        assert not ep.can_accept_request

    def test_cannot_accept_when_disabled(self):
        ep = Endpoint(name="test", url="http://localhost", roles=["reasoning"],
                      priority=1, max_concurrent=2, enabled=False)
        assert not ep.can_accept_request

    def test_start_complete_request(self):
        ep = Endpoint(name="test", url="http://localhost", roles=["reasoning"],
                      priority=1, max_concurrent=2)
        ep.start_request()
        assert ep.active_requests == 1
        ep.complete_request()
        assert ep.active_requests == 0

    def test_complete_request_no_negative(self):
        ep = Endpoint(name="test", url="http://localhost", roles=["reasoning"],
                      priority=1, max_concurrent=2)
        ep.complete_request()
        assert ep.active_requests == 0

    def test_record_success(self):
        ep = Endpoint(name="test", url="http://localhost", roles=["reasoning"],
                      priority=1, max_concurrent=2)
        ep.failure_count = 2
        ep.record_success(0.5)
        assert ep.failure_count == 0
        assert ep.success_count == 1
        assert ep.last_response_time == 0.5

    def test_record_failure_disables_after_threshold(self):
        ep = Endpoint(name="test", url="http://localhost", roles=["reasoning"],
                      priority=1, max_concurrent=2)
        ep.record_failure()
        ep.record_failure()
        assert ep.enabled  # 2 failures, still enabled
        ep.record_failure()
        assert not ep.enabled  # 3 failures, disabled


class TestEndpointManager:
    async def test_get_endpoint_for_role(self, endpoint_manager):
        ep = await endpoint_manager.get_endpoint_for_role("reasoning")
        assert ep is not None
        assert ep.name == "primary"

    async def test_get_endpoint_for_different_role(self, endpoint_manager):
        ep = await endpoint_manager.get_endpoint_for_role("summarization")
        assert ep is not None
        assert ep.name == "secondary"

    async def test_fallback_to_any_endpoint(self, endpoint_manager):
        # "embedding" is not in any endpoint's roles, but should fallback
        ep = await endpoint_manager.get_endpoint_for_role("embedding")
        assert ep is not None

    async def test_no_endpoint_when_all_disabled(self, endpoint_configs):
        for cfg in endpoint_configs:
            cfg.enabled = False
        mgr = EndpointManager(endpoint_configs)
        ep = await mgr.get_endpoint_for_role("reasoning")
        assert ep is None

    async def test_get_client_for_role(self, endpoint_manager):
        client = await endpoint_manager.get_client_for_role("reasoning")
        assert client is not None

    async def test_mark_failed(self, endpoint_manager):
        await endpoint_manager.mark_failed("primary")
        status = endpoint_manager.get_status()
        primary = [s for s in status if s["name"] == "primary"][0]
        assert primary["failure_count"] == 1

    async def test_mark_success(self, endpoint_manager):
        await endpoint_manager.mark_success("primary", 0.5)
        status = endpoint_manager.get_status()
        primary = [s for s in status if s["name"] == "primary"][0]
        assert primary["success_count"] == 1

    async def test_priority_selection(self, endpoint_configs):
        # Both endpoints have "reasoning" role
        endpoint_configs[1].roles = ["reasoning"]
        endpoint_configs[1].priority = 10  # higher priority
        mgr = EndpointManager(endpoint_configs)
        ep = await mgr.get_endpoint_for_role("reasoning")
        assert ep.name == "secondary"  # higher priority

    async def test_load_balancing(self, endpoint_configs):
        # Both support reasoning, same priority
        endpoint_configs[1].roles = ["reasoning"]
        endpoint_configs[0].priority = 1
        endpoint_configs[1].priority = 1
        mgr = EndpointManager(endpoint_configs)

        ep1 = await mgr.get_endpoint_for_role("reasoning")
        ep1.start_request()

        # Second call should pick the less loaded one
        ep2 = await mgr.get_endpoint_for_role("reasoning")
        assert ep2.active_requests == 0

    def test_get_status(self, endpoint_manager):
        status = endpoint_manager.get_status()
        assert len(status) == 2
        assert status[0]["name"] in ("primary", "secondary")
        assert "enabled" in status[0]
        assert "active_requests" in status[0]

    async def test_lock_serializes_access(self, endpoint_manager):
        """Verify the lock doesn't deadlock on sequential calls."""
        ep1 = await endpoint_manager.get_endpoint_for_role("reasoning")
        ep2 = await endpoint_manager.get_endpoint_for_role("reasoning")
        assert ep1 is not None
        assert ep2 is not None
