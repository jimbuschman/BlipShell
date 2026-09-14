"""Maintenance commands obey the same privacy settings as chat.

`Agent._build_subsystems` was the only place that applied all three of
`pii.local_mode_default`, `pii.enabled` and `pii.require_ner`. Every
maintenance path built a bare `LLMRouter(cfg.models, EndpointManager(...))`:

- `blipshell repair --blank-summaries` re-summarizes real memories, and on a
  machine configured for local-only operation it would still pick a cloud
  endpoint.
- the nightly runner, both importers and both reprocess commands are the
  FULL-sanitize background path `require_ner` exists for, and the constructor
  default is `False` — the gate was never armed on the jobs it was written
  for.

These tests drive the real callers (the Click command, `NightlyRunner.create_from_config`,
`MemoryWorker`) with mocked clients. No memory text leaves the process.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from click.testing import CliRunner

from blipshell.llm import pii
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.routing import build_endpoint_manager, build_router
from blipshell.models.config import BlipShellConfig, EndpointConfig, PIIConfig


# --- the factory itself ---------------------------------------------------------


def _config(**pii_kwargs) -> BlipShellConfig:
    cfg = BlipShellConfig()
    cfg.pii = PIIConfig(**pii_kwargs)
    cfg.endpoints = [
        EndpointConfig(name="cloud", url="https://api.example", provider="openai",
                       roles=["summarization"], priority=1, max_concurrent=4),
        EndpointConfig(name="local", url="http://localhost:11434", provider="ollama",
                       roles=["summarization"], priority=10, max_concurrent=4),
    ]
    return cfg


def test_factory_applies_local_mode_from_config():
    mgr = build_endpoint_manager(_config(local_mode_default=True))
    assert mgr.local_only is True


def test_factory_leaves_local_mode_off_when_config_says_so():
    assert build_endpoint_manager(_config(local_mode_default=False)).local_only is False


def test_caller_may_tighten_routing_but_never_loosen_it():
    """A local_only argument is an OR with the config, not an assignment."""
    assert build_endpoint_manager(
        _config(local_mode_default=False), local_only=True).local_only is True
    assert build_endpoint_manager(
        _config(local_mode_default=True), local_only=False).local_only is True


def test_factory_router_carries_the_pii_settings():
    cfg = _config(enabled=False, require_ner=True)
    router = build_router(cfg, build_endpoint_manager(cfg))
    assert router._pii_enabled is False
    assert router._require_ner is True
    assert router._disable_fallback is False


def test_factory_honours_an_intentional_model_and_fallback_override():
    """Specialized construction is preserved, not replaced."""
    from blipshell.models.config import ModelsConfig
    cfg = _config()
    models = ModelsConfig(summarization="override:latest", embedding="unused")
    router = build_router(cfg, build_endpoint_manager(cfg), models=models,
                          disable_fallback=True)
    assert router._models.summarization == "override:latest"
    assert router._disable_fallback is True


def test_factory_endpoint_override_is_used():
    """The nightly runner disables cloud endpoints before building."""
    cfg = _config()
    only_local = [e for e in cfg.endpoints if e.name == "local"]
    mgr = build_endpoint_manager(cfg, endpoints=only_local)
    assert [e.name for e in mgr.endpoints] == ["local"]


# --- the real callers -----------------------------------------------------------


@pytest.fixture
def fake_clients(monkeypatch):
    """Every endpoint gets a recording client; nothing opens a socket."""
    made: dict[str, MagicMock] = {}

    def _create(cfg, llm_cfg):
        client = MagicMock()
        client.generate = AsyncMock(return_value=f"summary from {cfg.name}")
        client.chat = AsyncMock(return_value=f"chat from {cfg.name}")
        made[cfg.name] = client
        return client

    monkeypatch.setattr(EndpointManager, "_create_client", staticmethod(_create))
    return made


@pytest.fixture
def presidio_missing(monkeypatch):
    monkeypatch.setattr(pii, "_presidio_available", False)
    monkeypatch.setattr(pii, "_presidio_analyzer", None)


ENDPOINTS_YAML = [
    {"name": "cloud", "url": "https://api.example", "provider": "openai",
     "api_key": "unused", "roles": ["summarization", "reasoning", "chat"],
     "priority": 10, "max_concurrent": 4, "context_tokens": 128000},
    {"name": "local", "url": "http://localhost:11434", "provider": "ollama",
     "roles": ["summarization", "reasoning", "chat"],
     "priority": 1, "max_concurrent": 2, "context_tokens": 32768},
]


async def _seed_blank_summary_async(db_path: str) -> int:
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.models.memory import Memory
    store = SQLiteStore(db_path)
    await store.initialize()
    session_id = await store.create_session(title="t")
    mem_id = await store.create_memory(Memory(
        session_id=session_id, role="user", summary="",
        content="A message whose summary was stored blank.",
    ))
    await store.close()
    return mem_id


def _seed_blank_summary(db_path: str) -> int:
    """Sync: the repair command runs its own asyncio.run, so these tests must
    not already be inside an event loop."""
    return asyncio.run(_seed_blank_summary_async(db_path))


def _write_config(tmp_path, db_path, **pii_kwargs) -> str:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "database": {"path": str(db_path)},
        "endpoints": ENDPOINTS_YAML,
        "models": {"summarization": "glm4:latest", "embedding": "qwen3-embedding:0.6b"},
        "pii": pii_kwargs,
    }), encoding="utf-8")
    return str(cfg_path)


def _run_repair(config_path):
    from blipshell.ui.cli import main
    return CliRunner().invoke(
        main, ["--config-path", config_path, "repair", "--blank-summaries"],
        catch_exceptions=False,
    )


def test_local_only_repair_cannot_select_a_cloud_endpoint(tmp_path, fake_clients):
    """THE regression: pii.local_mode_default was ignored by the repair path."""
    db = tmp_path / "t.db"
    _seed_blank_summary(str(db))
    cfg = _write_config(tmp_path, db, local_mode_default=True)

    result = _run_repair(cfg)

    assert result.exit_code == 0, result.output
    assert fake_clients["cloud"].generate.await_count == 0
    assert fake_clients["local"].generate.await_count >= 1


def test_repair_uses_cloud_when_local_mode_is_off(tmp_path, fake_clients):
    """Negative control: the gate is the config, not a blanket ban."""
    db = tmp_path / "t.db"
    _seed_blank_summary(str(db))
    cfg = _write_config(tmp_path, db, local_mode_default=False)

    result = _run_repair(cfg)

    assert result.exit_code == 0, result.output
    assert fake_clients["cloud"].generate.await_count >= 1


def test_repair_enforces_require_ner_when_presidio_is_unavailable(
    tmp_path, fake_clients, presidio_missing, caplog,
):
    """The NER gate reaches the maintenance path, not just Agent."""
    db = tmp_path / "t.db"
    _seed_blank_summary(str(db))
    cfg = _write_config(tmp_path, db, require_ner=True)

    with caplog.at_level(logging.WARNING, logger="blipshell.llm.router"):
        result = _run_repair(cfg)

    assert result.exit_code == 0, result.output
    assert fake_clients["cloud"].generate.await_count == 0
    assert fake_clients["local"].generate.await_count >= 1
    assert any("require_ner" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_nightly_runner_builds_a_privacy_configured_router(tmp_path, fake_clients):
    from blipshell.core.nightly import NightlyRunner
    db = tmp_path / "t.db"
    await _seed_blank_summary_async(str(db))
    cfg = _write_config(tmp_path, db, local_mode_default=True, require_ner=True)

    runner = await NightlyRunner.create_from_config(config_path=cfg)
    try:
        assert runner.router._require_ner is True
        assert runner.router._endpoint_manager.local_only is True
    finally:
        await runner.sqlite.close()
        runner.vectors.close()


@pytest.mark.asyncio
async def test_nightly_local_only_flag_still_disables_cloud_endpoints(tmp_path, fake_clients):
    """The pre-existing override is preserved, not replaced by the factory."""
    from blipshell.core.nightly import NightlyRunner
    db = tmp_path / "t.db"
    await _seed_blank_summary_async(str(db))
    cfg = _write_config(tmp_path, db)

    runner = await NightlyRunner.create_from_config(config_path=cfg, local_only=True)
    try:
        cloud = [e for e in runner.router._endpoint_manager.endpoints if e.name == "cloud"]
        assert cloud and cloud[0].enabled is False
        assert runner.router._endpoint_manager.local_only is True
    finally:
        await runner.sqlite.close()
        runner.vectors.close()


def test_memory_worker_builds_its_router_through_the_factory(monkeypatch, tmp_path):
    """The worker had the PII flags but not local mode."""
    import blipshell.llm.routing as routing

    seen = {}
    real = routing.build_routing

    def spy(config, **kwargs):
        mgr, router = real(config, **kwargs)
        seen["local_only"] = mgr.local_only
        seen["require_ner"] = router._require_ner
        return mgr, router

    monkeypatch.setattr(routing, "build_routing", spy)

    from blipshell.memory.worker import MemoryWorker
    config = BlipShellConfig()
    config.pii = PIIConfig(local_mode_default=True, require_ner=True)
    config.database.path = str(tmp_path / "w.db")
    worker = MemoryWorker(config=config, vectors=MagicMock())
    worker.start()
    try:
        assert seen == {"local_only": True, "require_ner": True}
    finally:
        worker.shutdown(timeout=5.0)


def test_no_production_caller_builds_a_bare_router():
    """Structural guard against the drift coming back.

    The benchmark harness is the one intentional exception — it measures a
    model, so it disables fallback and PII on purpose.
    """
    import ast
    import pathlib

    package = pathlib.Path(__file__).resolve().parent.parent / "blipshell"
    allowed = {"llm/routing.py", "benchmark/harness.py"}
    offenders = []
    for path in package.rglob("*.py"):
        rel = path.relative_to(package).as_posix()
        if rel in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "LLMRouter"):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "build a router through blipshell.llm.routing so the PII settings "
        f"cannot be dropped: {offenders}"
    )
