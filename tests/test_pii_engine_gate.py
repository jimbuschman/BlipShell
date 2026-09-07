"""PII engine: visible, and enforceable for background cloud traffic.

Presidio + spaCy are an optional extra. When they are absent (or the spaCy
model is missing) the sanitizer silently became regex-only — credentials
redacted, names and places NOT — and said so once, at INFO. These tests pin
three things: the engine is reportable without loading it; the agent logs
it at WARNING when it matters; and with ``pii.require_ner`` the router keeps
fully-sanitized background text off sanitizing endpoints while NER is down.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.llm import pii
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig


@pytest.fixture
def presidio_missing(monkeypatch):
    monkeypatch.setattr(pii, "_presidio_available", False)
    monkeypatch.setattr(pii, "_presidio_analyzer", None)


@pytest.fixture
def presidio_present(monkeypatch):
    monkeypatch.setattr(pii, "_presidio_available", True)
    monkeypatch.setattr(pii, "_presidio_analyzer", object())


# --- reporting -------------------------------------------------------------------


def test_engine_status_does_not_trigger_the_load(monkeypatch):
    monkeypatch.setattr(pii, "_presidio_available", None)
    loads = []
    monkeypatch.setattr(pii, "_get_presidio_analyzer", lambda: loads.append(1))
    assert "not yet checked" in pii.engine_status()
    assert loads == []


def test_engine_status_and_description_agree_once_checked(presidio_missing):
    assert pii.engine_status() == pii.REGEX_ONLY_DESCRIPTION
    assert pii.engine_description() == pii.REGEX_ONLY_DESCRIPTION
    assert "names and places are NOT" in pii.REGEX_ONLY_DESCRIPTION


def test_engine_description_when_present(presidio_present):
    assert pii.engine_description() == pii.PRESIDIO_DESCRIPTION
    assert pii.engine_status() == pii.PRESIDIO_DESCRIPTION


def _agent_stub(*, cloud: bool, pii_enabled: bool = True):
    from blipshell.core.agent import Agent
    ep = SimpleNamespace(name="openrouter" if cloud else "local", should_sanitize_pii=cloud)
    stub = SimpleNamespace(
        endpoint_manager=SimpleNamespace(endpoints=[ep]),
        config=SimpleNamespace(pii=SimpleNamespace(enabled=pii_enabled)),
    )
    return lambda: Agent._report_pii_engine(stub)


def test_agent_warns_when_regex_only_and_an_endpoint_relays_offsite(presidio_missing, caplog):
    with caplog.at_level(logging.INFO, logger="blipshell.core.agent"):
        _agent_stub(cloud=True)()
    rec = [r for r in caplog.records if "PII engine" in r.getMessage()]
    assert rec and rec[0].levelno == logging.WARNING
    assert "openrouter" in rec[0].getMessage()


def test_agent_stays_at_info_when_nothing_leaves_the_machine(presidio_missing, caplog):
    with caplog.at_level(logging.INFO, logger="blipshell.core.agent"):
        _agent_stub(cloud=False)()
    rec = [r for r in caplog.records if "PII engine" in r.getMessage()]
    assert rec and rec[0].levelno == logging.INFO


def test_agent_stays_at_info_when_presidio_loads(presidio_present, caplog):
    with caplog.at_level(logging.INFO, logger="blipshell.core.agent"):
        _agent_stub(cloud=True)()
    rec = [r for r in caplog.records if "PII engine" in r.getMessage()]
    assert rec and rec[0].levelno == logging.INFO


# --- the routing gate -----------------------------------------------------------


def _manager(*, with_local=True):
    configs = [
        EndpointConfig(name="cloud", url="https://api.example", provider="openai",
                       roles=["summarization"], priority=10, max_concurrent=4),
    ]
    if with_local:
        configs.append(EndpointConfig(name="local", url="http://localhost:11434", provider="ollama",
                                      roles=["summarization"], priority=1, max_concurrent=4))
    mgr = EndpointManager(configs, LLMConfig(max_retries=1, retry_base_delay=0.01))
    for ep in mgr.endpoints:
        ep.client = MagicMock()
        ep.client.generate = AsyncMock(return_value=f"from {ep.name}")
    return mgr


def _client(mgr, name):
    return next(ep for ep in mgr.endpoints if ep.name == name).client


_MODELS = ModelsConfig(summarization="glm4:latest", summarization_fallback="qwen3:14b")


async def test_require_ner_routes_background_call_to_local_when_presidio_missing(presidio_missing, caplog):
    mgr = _manager()
    router = LLMRouter(_MODELS, mgr, require_ner=True)
    with caplog.at_level(logging.WARNING, logger="blipshell.llm.router"):
        out = await router.generate(TaskType.SUMMARIZATION, "Jim met Kortney in Denver")
    assert out == "from local"
    _client(mgr, "cloud").generate.assert_not_called()
    # the local hop gets the UNSANITIZED text (nothing left the machine)
    sent = _client(mgr, "local").generate.call_args.kwargs["prompt"]
    assert "Kortney" in sent and "Denver" in sent
    assert any("require_ner" in r.getMessage() for r in caplog.records)


async def test_gate_logs_once_not_per_call(presidio_missing, caplog):
    mgr = _manager()
    router = LLMRouter(_MODELS, mgr, require_ner=True)
    with caplog.at_level(logging.WARNING, logger="blipshell.llm.router"):
        await router.generate(TaskType.SUMMARIZATION, "a")
        await router.generate(TaskType.SUMMARIZATION, "b")
    assert sum("require_ner" in r.getMessage() for r in caplog.records) == 1


async def test_without_require_ner_cloud_is_used_with_regex_sanitization(presidio_missing):
    """Today's behaviour, unchanged: the flag is opt-in."""
    mgr = _manager()
    router = LLMRouter(_MODELS, mgr, require_ner=False)
    out = await router.generate(TaskType.SUMMARIZATION, "token ghp_" + "a" * 40 + " is mine")
    assert out == "from cloud"
    sent = _client(mgr, "cloud").generate.call_args.kwargs["prompt"]
    assert "ghp_" not in sent and "[API_KEY]" in sent  # regex engine still ran


async def test_require_ner_is_a_no_op_when_presidio_loads(presidio_present, monkeypatch):
    mgr = _manager()
    # Presidio "present" but we do not want a real analyzer run in a unit test.
    monkeypatch.setattr(pii, "_sanitize_with_presidio", lambda text: text)
    router = LLMRouter(_MODELS, mgr, require_ner=True)
    out = await router.generate(TaskType.SUMMARIZATION, "hello")
    assert out == "from cloud"


async def test_require_ner_refuses_rather_than_leaks_when_only_cloud_exists(presidio_missing):
    mgr = _manager(with_local=False)
    router = LLMRouter(_MODELS, mgr, require_ner=True)
    with pytest.raises(RuntimeError, match="require_ner"):
        await router.generate(TaskType.SUMMARIZATION, "Jim met Kortney")
    _client(mgr, "cloud").generate.assert_not_called()


async def test_gate_holds_on_the_error_fallback_hop(presidio_missing):
    """Local primary fails; the fallback search must not pick the cloud."""
    mgr = _manager()
    # make local the primary by outranking cloud, then make it fail
    for ep in mgr.endpoints:
        ep.priority = 20 if ep.name == "local" else 1
    _client(mgr, "local").generate = AsyncMock(side_effect=RuntimeError("ollama down"))
    router = LLMRouter(_MODELS, mgr, require_ner=True)
    with pytest.raises(Exception):
        await router.generate(TaskType.SUMMARIZATION, "Jim met Kortney")
    _client(mgr, "cloud").generate.assert_not_called()


async def test_pii_disabled_disables_the_gate_too(presidio_missing):
    """pii.enabled=false means 'do not sanitize at all'; require_ner is moot."""
    mgr = _manager()
    router = LLMRouter(_MODELS, mgr, pii_enabled=False, require_ner=True)
    out = await router.generate(TaskType.SUMMARIZATION, "hello")
    assert out == "from cloud"


def test_exclude_accepts_a_collection_and_a_string():
    """The router now passes a set; every older caller passes a str."""
    import asyncio
    mgr = _manager()

    async def pick(exclude):
        ep = await mgr.get_endpoint_for_role("summarization", exclude=exclude)
        return ep.name if ep else None

    assert asyncio.run(pick("cloud")) == "local"
    assert asyncio.run(pick({"cloud"})) == "local"
    assert asyncio.run(pick({"cloud", "local"})) is None
    assert asyncio.run(pick(None)) == "cloud"
