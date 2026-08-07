"""Tests for the endpoint manager (llm/endpoints.py) and LLM router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from blipshell.llm.endpoints import Endpoint, EndpointManager
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig


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


class TestLLMRouter:
    """Tests for the LLM router — model selection, failure tracking, fallback."""

    @pytest.fixture
    def router(self, endpoint_configs):
        models = ModelsConfig(
            reasoning="qwen3:14b",
            tool_calling="glm-5:cloud",
            coding="qwen3-coder:480b-cloud",
            summarization="glm4:latest",
            ranking="qwen2.5:14b",
            importance="qwen3:14b",
            embedding="nomic-embed-text",
            tool_calling_fallback="gpt-oss:latest",
            coding_fallback="gpt-oss:latest",
            reasoning_fallback="gpt-oss:latest",
        )
        mgr = EndpointManager(endpoint_configs, LLMConfig())
        return LLMRouter(models, mgr)

    def test_get_model(self, router):
        assert router.get_model(TaskType.TOOL_CALLING) == "glm-5:cloud"
        assert router.get_model(TaskType.CODING) == "qwen3-coder:480b-cloud"
        assert router.get_model(TaskType.REASONING) == "qwen3:14b"
        assert router.get_model(TaskType.SUMMARIZATION) == "glm4:latest"

    def test_get_fallback_model(self, router):
        assert router.get_fallback_model(TaskType.TOOL_CALLING) == "gpt-oss:latest"
        assert router.get_fallback_model(TaskType.CODING) == "gpt-oss:latest"
        assert router.get_fallback_model(TaskType.EMBEDDING) is None  # no fallback

    def test_mark_model_failed(self, router):
        assert not router.is_model_failed("glm-5:cloud")
        router.mark_model_failed("glm-5:cloud")
        assert router.is_model_failed("glm-5:cloud")

    def test_clear_failed_models(self, router):
        router.mark_model_failed("glm-5:cloud")
        router.mark_model_failed("qwen3-coder:480b-cloud")
        assert router.is_model_failed("glm-5:cloud")
        assert router.is_model_failed("qwen3-coder:480b-cloud")

        router.clear_failed_models()
        assert not router.is_model_failed("glm-5:cloud")
        assert not router.is_model_failed("qwen3-coder:480b-cloud")

    def test_clear_empty_is_noop(self, router):
        """Clearing when nothing is failed should be silent."""
        router.clear_failed_models()  # should not raise

    def test_unknown_task_type_defaults_to_reasoning(self, router):
        model = router.get_model("unknown_task")
        assert model == "qwen3:14b"  # defaults to reasoning model

    async def test_get_model_and_client(self, router):
        model, client = await router.get_model_and_client(TaskType.TOOL_CALLING)
        assert model == "glm-5:cloud"
        assert client is not None


class TestLocalMode:
    """/local (V2_PLAN D1): endpoints that relay off the machine become
    invisible to routing. Mirrors the live config: cloud chat via OpenRouter
    AND via Ollama-cloud (same daemon URL as local, pii_sanitize: true), with
    a local endpoint that does NOT serve tool_calling — so chat only reaches
    it through the any-endpoint fallback branch."""

    def _manager(self):
        configs = [
            EndpointConfig(
                name="openrouter", url="https://openrouter.example/v1",
                provider="openai", roles=["coding", "tool_calling"],
                priority=1, max_concurrent=2, context_tokens=204800,
                pii_sanitize=True, api_key="k",
            ),
            # provider="openai" imports the openai package at client
            # construction, which this box doesn't have (the two-PC split) —
            # and these tests exercise ROUTING, not clients.
            EndpointConfig(
                name="local-cloud", url="http://localhost:11434",
                roles=["tool_calling", "session_review"], priority=2,
                max_concurrent=2, context_tokens=131072, pii_sanitize=True,
            ),
            EndpointConfig(
                name="local", url="http://localhost:11434",
                roles=["reasoning", "summarization"], priority=1,
                max_concurrent=2, context_tokens=32768,
            ),
        ]
        with patch.object(EndpointManager, "_create_client", return_value=MagicMock()):
            return EndpointManager(configs, LLMConfig(max_retries=1, retry_base_delay=0.1))

    async def test_cloud_serves_chat_when_off(self):
        em = self._manager()
        ep = await em.get_endpoint_for_role("tool_calling")
        assert ep.name == "local-cloud"      # priority 2 beats openrouter's 1

    async def test_local_mode_hides_every_offmachine_endpoint(self):
        """The role has NO local endpoint, so this exercises the any-endpoint
        fallback branch — the exact hole that would silently defeat the mode."""
        em = self._manager()
        em.local_only = True

        ep = await em.get_endpoint_for_role("tool_calling")

        assert ep is not None, "local mode left chat with no endpoint at all"
        assert ep.name == "local", (
            f"local mode routed chat to '{ep.name}', which relays off-machine"
        )

    async def test_local_mode_never_yields_a_sanitizing_endpoint(self):
        """Sweep every role either cloud endpoint serves."""
        em = self._manager()
        em.local_only = True
        for role in ("tool_calling", "coding", "session_review",
                     "reasoning", "summarization"):
            ep = await em.get_endpoint_for_role(role)
            assert ep is None or not ep.should_sanitize_pii, (
                f"role '{role}' escaped to '{ep.name}' in local mode"
            )

    async def test_context_sizing_follows_the_mode(self):
        """After /local, the last-routed shortcut still remembers the cloud
        endpoint — sizing the prompt at 204K for a 32K model would overflow
        the context on the first turn."""
        em = self._manager()
        await em.get_endpoint_for_role("tool_calling")     # routes to cloud
        assert em.get_context_tokens_for_role("tool_calling") == 131072

        em.local_only = True
        assert em.get_context_tokens_for_role("tool_calling") == 32768

    async def test_toggling_back_restores_cloud(self):
        em = self._manager()
        em.local_only = True
        assert (await em.get_endpoint_for_role("tool_calling")).name == "local"

        em.local_only = False
        assert (await em.get_endpoint_for_role("tool_calling")).name == "local-cloud"

    async def test_default_is_off(self):
        assert self._manager().local_only is False


class TestLocalCommand:
    def _ctx(self, name, arg=""):
        from blipshell.ui.commands import CommandContext

        agent = MagicMock()
        agent.endpoint_manager.local_only = False
        parts = [name] + ([arg] if arg else [])
        return CommandContext(
            agent=agent, config=MagicMock(), raw=f"/{name} {arg}".strip(),
            parts=parts, args=parts[1:], ui=MagicMock(), console=MagicMock(),
        )

    async def _run(self, ctx):
        from blipshell.ui.commands import registry
        await registry.dispatch(ctx)

    async def test_local_toggles_on(self):
        import blipshell.ui.command_handlers  # noqa: F401  (registers commands)

        ctx = self._ctx("local")
        await self._run(ctx)
        assert ctx.agent.endpoint_manager.local_only is True

    async def test_cloud_always_turns_off(self):
        import blipshell.ui.command_handlers  # noqa: F401

        ctx = self._ctx("cloud")
        ctx.agent.endpoint_manager.local_only = True
        await self._run(ctx)
        assert ctx.agent.endpoint_manager.local_only is False

    async def test_explicit_on_off(self):
        import blipshell.ui.command_handlers  # noqa: F401

        ctx = self._ctx("local", "on")
        await self._run(ctx)
        assert ctx.agent.endpoint_manager.local_only is True

        ctx2 = self._ctx("local", "off")
        ctx2.agent.endpoint_manager.local_only = True
        await self._run(ctx2)
        assert ctx2.agent.endpoint_manager.local_only is False
