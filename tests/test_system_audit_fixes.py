"""Regression scenarios from the September 15 audit; no model/network IO."""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, WebSocketDisconnect

from blipshell.core.agent import Agent
from blipshell.core.agent_background import BackgroundMixin
from blipshell.core.background import BackgroundTaskManager
from blipshell.core.nightly_history import HISTORY_KEY, save_run
from blipshell.core.tools.base import Tool, ToolRegistry
from blipshell.core.tools.shell import ShellTool, _collect_output
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.routing import build_routing
from blipshell.memory.search import MemorySearch
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import AuthConfig, BlipShellConfig, EndpointConfig, WorkerConfig
from blipshell.models.memory import Memory
from blipshell.models.session import MessageRole, SessionMessage
from blipshell.models.task import BackgroundTask
from blipshell.models.tools import ToolCall, ToolDefinition
from blipshell.session.manager import SessionManager
from blipshell.ui.command_handlers import _local
from blipshell.ui.web import app as web


def session():
    sm = SessionManager(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    sm.session_id = 1
    sm._messages = [SessionMessage(role=MessageRole.USER, content=f'message{i}') for i in range(6)]
    sm._memory_db_ids = dict(enumerate(range(100, 106)))
    sm._dumped_indices = {0, 1, 2}
    return sm


async def test_compaction_preserves_ids_and_processing_state():
    sm = session()
    sm.history_summarized_upto = 4
    queued = []
    agent = NS(session_manager=sm, router=NS(generate=AsyncMock(return_value='summary')),
               _memory_worker=NS(is_alive=True, enqueue=queued.append))
    result = await Agent.compact_conversation(agent)
    assert result.startswith('Compacted')
    assert sm._memory_db_ids == {1: 102, 2: 103, 3: 104, 4: 105}
    assert sm._dumped_indices == {0, 1}
    assert sm.history_summarized_upto == 3
    await BackgroundMixin._enqueue_undumped_messages(agent)
    assert [(i.text, i.memory_id) for i in queued] == [
        ('message3', 103), ('message4', 104), ('message5', 105)]


@pytest.mark.parametrize('summary', ['', '\n\t'])
async def test_blank_compaction_keeps_history(summary):
    sm = session()
    before = sm.get_messages()
    await Agent.compact_conversation(NS(session_manager=sm, router=NS(generate=AsyncMock(return_value=summary))))
    assert sm.get_messages() == before


async def test_compaction_waits_for_persist_and_refuses_unprocessed_prefix():
    sm = session()
    sm._dumped_indices.clear()
    before = sm.get_messages()
    persisted = asyncio.Event()

    async def persist():
        sm._memory_db_ids[5] = 999
        persisted.set()

    sm._pending_persists = [asyncio.create_task(persist())]
    with pytest.raises(ValueError, match='awaiting'):
        await sm.compact_prefix(2, before[0], before)
    assert persisted.is_set()
    assert sm.get_messages() == before
    assert sm._memory_db_ids[5] == 999


async def test_compaction_rejects_a_changed_conversation():
    sm = session()
    before = sm.get_messages()
    sm._messages.append(SessionMessage(role=MessageRole.USER, content='new turn'))
    with pytest.raises(ValueError, match='changed'):
        await sm.compact_prefix(2, before[0], before)
    assert len(sm._messages) == 7


@pytest.mark.parametrize('code, output', [(7, b'tests failed'), (1, b''), (2, b'x' * 9000), (0, b'Error: quoted log')])
async def test_shell_result_tracks_exit_code(code, output):
    registry = ToolRegistry()
    registry.register(ShellTool())
    process = NS(returncode=code, communicate=AsyncMock(return_value=(output, b'')))
    with patch('blipshell.core.tools.shell.asyncio.create_subprocess_shell', AsyncMock(return_value=process)), \
         patch.object(ShellTool, '_save_full_output', return_value=None):
        result = await registry.execute_tool_call(ToolCall(name='run_command', arguments={'command': 'echo probe'}))
    assert result.success == (code == 0)


async def test_background_output_is_drained_and_bounded():
    process = NS(stdout=NS(read=AsyncMock(side_effect=[b'x' * 800000, b'y' * 800000, b''])),
                 stderr=NS(read=AsyncMock(side_effect=[b'err', b''])), wait=AsyncMock())
    stdout, stderr = await _collect_output(process)
    assert len(stdout) == 1048576
    assert stdout.endswith(b'y' * 800000)
    assert stderr == b'err'


def routing(monkeypatch):
    monkeypatch.setattr(EndpointManager, '_create_client', staticmethod(lambda *args: NS(generate=AsyncMock(return_value='ok'))))
    cfg = BlipShellConfig()
    cfg.pii.require_ner = False
    cfg.endpoints = [EndpointConfig(name='cloud', provider='openai', url='https://invalid.example',
                                   roles=['reasoning'], models={'reasoning': 'cloud-model'})]
    return cfg, build_routing(cfg)


async def test_local_toggle_is_shared_with_worker_manager(monkeypatch):
    cfg, (foreground, _) = routing(monkeypatch)
    worker, router = build_routing(cfg)
    worker.local_policy = foreground.local_policy
    _local(NS(agent=NS(endpoint_manager=foreground), name='local', arg=lambda _: 'on', console=MagicMock()))
    assert worker.local_only
    with pytest.raises(RuntimeError):
        await router.generate('reasoning', 'private')
    worker.endpoints[0].client.generate.assert_not_awaited()


async def test_targeted_background_obeys_local_mode(monkeypatch):
    cfg, (manager, router) = routing(monkeypatch)
    manager.local_only = True
    store = NS(claim_background_task=AsyncMock(return_value='token'), update_claimed_background_task=AsyncMock(return_value=True), get_background_task=AsyncMock(return_value=NS(task_type='research')))
    await BackgroundTaskManager(router, store, WorkerConfig())._run_task(1, 'private', 'cloud')
    manager.endpoints[0].client.generate.assert_not_awaited()


async def test_targeted_router_sanitizes_and_uses_endpoint_model(monkeypatch):
    cfg, (manager, router) = routing(monkeypatch)
    monkeypatch.setattr('blipshell.llm.pii.sanitize_text', lambda text: 'scrubbed')
    assert await router.generate('reasoning', 'private', target_endpoint='cloud') == 'ok'
    kwargs = manager.endpoints[0].client.generate.await_args.kwargs
    assert kwargs['prompt'] == 'scrubbed'
    assert kwargs['model'] == 'cloud-model'


async def test_targeted_router_cannot_bypass_ner(monkeypatch):
    cfg, (manager, router) = routing(monkeypatch)
    router._require_ner = True
    monkeypatch.setattr('blipshell.llm.pii.is_presidio_available', lambda: False)
    with pytest.raises(RuntimeError):
        await router.generate('reasoning', 'private', target_endpoint='cloud')
    manager.endpoints[0].client.generate.assert_not_awaited()


@pytest.mark.parametrize('cloud_model', ['gemma4:31b-cloud', 'vendor/cloud-model'])
async def test_local_endpoint_cannot_proxy_default_cloud_model(monkeypatch, cloud_model):
    monkeypatch.setattr(EndpointManager, '_create_client', staticmethod(
        lambda *args: NS(generate=AsyncMock(return_value='local response'))))
    cfg = BlipShellConfig()
    cfg.models.tool_calling = cloud_model
    cfg.models.tool_calling_fallback = 'gpt-oss:latest'
    cfg.endpoints = [
        EndpointConfig(name='local', provider='ollama', roles=['tool_calling']),
        EndpointConfig(name='cloud', provider='openai', roles=['tool_calling'],
                       models={'tool_calling': cloud_model}),
    ]
    manager, router = build_routing(cfg, local_only=True)
    model, _ = await router.get_model_and_client('tool_calling')
    assert model == 'gpt-oss:latest'
    await router.generate('tool_calling', 'synthetic prompt')
    manager.endpoints[0].client.generate.assert_awaited_once()
    assert manager.endpoints[0].client.generate.await_args.kwargs['model'] == 'gpt-oss:latest'
    manager.endpoints[1].client.generate.assert_not_awaited()


async def test_agent_enforces_approval_without_ui():
    class Dummy(Tool):
        def definition(self):
            return ToolDefinition(name='write_file', description='mock')

        async def execute(self, **kwargs):
            raise AssertionError('must not execute')

    agent = Agent(BlipShellConfig(), MagicMock())
    agent.tool_registry.register(Dummy())
    result = await agent.tool_registry.execute_tool_call(ToolCall(name='write_file'))
    assert not result.success
    assert 'Approval required' in result.result


async def test_embeddings_outage_still_returns_keyword_match(sqlite_store):
    mid = await sqlite_store.create_memory(Memory(role='user', content='quartz manual',
        summary='quartz manual', importance=0.9, rank=5))
    vectors = NS(search_memories=MagicMock(side_effect=RuntimeError('offline')))
    search = MemorySearch(sqlite_store, vectors, MagicMock())
    results = await search.search('quartz manual')
    assert mid in [r.memory_id for r in results]


async def test_history_and_claims_are_atomic_across_connections(sqlite_store, temp_db_path):
    other = SQLiteStore(temp_db_path)
    await other.initialize()
    try:
        await asyncio.gather(save_run(sqlite_store, {'started_at': 1}), save_run(other, {'started_at': 2}))
        history = json.loads(await sqlite_store.get_metadata(HISTORY_KEY))
        assert [r['started_at'] for r in history] == [1, 2]
        task_id = await sqlite_store.create_background_task(BackgroundTask(title='synthetic'))
        claims = await asyncio.gather(sqlite_store.claim_background_task(task_id), other.claim_background_task(task_id))
        assert sum(bool(c) for c in claims) == 1
        token = next(c for c in claims if c)
        assert not await other.update_claimed_background_task(task_id, 'wrong', result='bad')
        assert await other.update_claimed_background_task(task_id, token, status='completed', result='good')
        assert not await other.update_claimed_background_task(task_id, token, status='failed')
    finally:
        await other.close()


def route(app, path):
    return next(r.endpoint for r in app.routes if getattr(r, 'path', None) == path)


def web_agent():
    return NS(start_session=AsyncMock(return_value=1), chat=AsyncMock(return_value='answer'),
              _enqueue_undumped_messages=AsyncMock(), session_manager=NS(end_session=AsyncMock()))


async def test_enabled_auth_with_empty_key_fails_closed(monkeypatch):
    monkeypatch.setattr(web, '_auth_config', AuthConfig(enabled=True, api_key=''))
    with pytest.raises(HTTPException):
        await web.verify_auth(None)


async def test_websocket_lease_prevents_session_takeover(monkeypatch):
    agent = web_agent()
    monkeypatch.setattr(web, '_agent', agent)
    monkeypatch.setattr(web, '_auth_config', None)
    app = web.create_app()
    held, release = asyncio.Event(), asyncio.Event()

    class Socket:
        def __init__(self):
            self.calls, self.sent = 0, []
        async def accept(self): pass
        async def close(self, **kwargs): pass
        async def send_json(self, value): self.sent.append(value)
        async def receive_json(self):
            self.calls += 1
            if self.calls == 1: return {}
            held.set()
            await release.wait()
            raise WebSocketDisconnect()

    first, second = Socket(), Socket()
    task = asyncio.create_task(route(app, '/ws/chat')(first))
    await asyncio.wait_for(held.wait(), 2)
    try:
        await route(app, '/ws/chat')(second)
        assert 'Another conversation' in second.sent[0]['message']
        agent.start_session.assert_awaited_once()
        request = web.ChatCompletionRequest(messages=[web.ChatMessage(role='user', content='x')])
        with pytest.raises(HTTPException) as error:
            await route(app, '/v1/chat/completions')(request)
        assert error.value.status_code == 409
    finally:
        release.set()
        await task


async def test_stream_error_is_not_normal_completion(monkeypatch):
    agent = web_agent()
    agent.chat.side_effect = RuntimeError('offline')
    monkeypatch.setattr(web, '_agent', agent)
    request = web.ChatCompletionRequest(stream=True, messages=[web.ChatMessage(role='user', content='x')])
    response = await route(web.create_app(), '/v1/chat/completions')(request)
    output = ''.join([chunk async for chunk in response.body_iterator])
    assert '"error"' in output
    assert '[DONE]' not in output
    assert '"finish_reason": "stop"' not in output


async def test_stream_close_cancels_chat_and_releases_lease(monkeypatch):
    agent = web_agent()
    entered, cancelled = asyncio.Event(), asyncio.Event()
    async def chat(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    agent.chat.side_effect = chat
    monkeypatch.setattr(web, '_agent', agent)
    handler = route(web.create_app(), '/v1/chat/completions')
    request = web.ChatCompletionRequest(stream=True, messages=[web.ChatMessage(role='user', content='x')])
    response = await handler(request)
    await anext(response.body_iterator)
    await asyncio.wait_for(entered.wait(), 2)
    await response.body_iterator.aclose()
    assert cancelled.is_set()
    agent.chat.side_effect = None
    request.stream = False
    assert (await handler(request))['choices'][0]['message']['content'] == 'answer'


async def test_local_and_remote_runner_cannot_duplicate_work(sqlite_store):
    router = NS(generate=AsyncMock(return_value='result'))
    manager = BackgroundTaskManager(router, sqlite_store, WorkerConfig())
    task_id = await sqlite_store.create_background_task(BackgroundTask(title='synthetic'))
    token = await sqlite_store.claim_background_task(task_id)
    assert token
    await manager._run_task(task_id, 'prompt')
    router.generate.assert_not_awaited()
    task_id = await sqlite_store.create_background_task(BackgroundTask(title='second'))
    await manager._run_task(task_id, 'prompt')
    router.generate.assert_awaited_once()
    task = await sqlite_store.get_background_task(task_id)
    assert task.status == 'completed'
    assert task.result == 'result'


async def test_resuming_session_replaces_old_messages(sqlite_store):
    sm = SessionManager(sqlite_store, MagicMock(), MagicMock(), MagicMock())
    first = await sm.start_session()
    sm.add_message(MessageRole.USER, 'first session')
    await sm.flush_pending_persists()
    await sm.start_session()
    sm.add_message(MessageRole.USER, 'second session')
    sm.history_summarized_upto = 99
    await sm.start_session(resume_session_id=first)
    assert [m.content for m in sm.get_messages()] == ['first session']
    assert sm.history_summarized_upto == 0
    memory = await sqlite_store.get_memory(sm._memory_db_ids[0])
    assert memory.content == 'first session'


async def test_claim_api_rejects_foreign_and_late_results(sqlite_store, monkeypatch):
    monkeypatch.setattr(web, '_agent', NS(sqlite=sqlite_store))
    app = web.create_app()
    tid = await sqlite_store.create_background_task(BackgroundTask(title='synthetic'))
    claim = await route(app, '/api/worker/claim/{task_id}')(tid)
    complete = route(app, '/api/worker/complete/{task_id}')
    with pytest.raises(HTTPException) as error:
        await complete(tid, {'claim_token': 'foreign', 'result': 'bad'})
    assert error.value.status_code == 409
    await complete(tid, {'claim_token': claim['claim_token'], 'result': 'good'})
    with pytest.raises(HTTPException):
        await complete(tid, {'claim_token': claim['claim_token'], 'result': 'late'})
    assert (await sqlite_store.get_background_task(tid)).result == 'good'


async def test_api_requests_replay_only_supplied_history(monkeypatch):
    agent = web_agent()
    monkeypatch.setattr(web, '_agent', agent)
    handler = route(web.create_app(), '/v1/chat/completions')
    request = web.ChatCompletionRequest(messages=[web.ChatMessage(role='user', content='older'),
        web.ChatMessage(role='assistant', content='reply'), web.ChatMessage(role='user', content='new')])
    await handler(request)
    assert [m.content for m in agent.session_manager._messages] == ['older', 'reply']
    await handler(web.ChatCompletionRequest(messages=[web.ChatMessage(role='user', content='unrelated')]))
    assert agent.session_manager._messages == []
    assert agent.start_session.await_count == 2
