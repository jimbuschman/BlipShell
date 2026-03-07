"""FastAPI backend with WebSocket streaming for chat.

REST endpoints for sessions, memories, lessons, config, endpoint status,
memory browser, data export, and API key auth.
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from blipshell.core.agent import Agent
from blipshell.core.config import ConfigManager
from blipshell.models.config import AuthConfig

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Global agent instance (created on startup)
_agent: Optional[Agent] = None
_config_manager: Optional[ConfigManager] = None
_auth_config: Optional[AuthConfig] = None

# Per-WebSocket session tracking: {ws_id: session_id}
_ws_sessions: dict[str, int] = {}

# Lazy API session for /v1 endpoints (Continue.dev, etc.)
_api_session_id: Optional[int] = None

# HTTP Bearer scheme (auto_error=False so we can check auth_config.enabled)
_bearer = HTTPBearer(auto_error=False)


# --- OpenAI-compatible request models ---

class ChatMessage(BaseModel):
    role: str  # system, user, assistant
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "blipshell"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


async def verify_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
):
    """Dependency that checks API key auth when enabled."""
    if not _auth_config or not _auth_config.enabled:
        return  # auth disabled
    if not _auth_config.api_key:
        return  # no key configured = open
    if not credentials or credentials.credentials != _auth_config.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def create_app(config_path: str | None = None) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="BlipShell", version="0.1.0")

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    async def startup():
        global _agent, _config_manager, _auth_config
        _config_manager = ConfigManager(config_path)
        config = _config_manager.load()
        _auth_config = config.auth
        _agent = Agent(config, _config_manager)
        await _agent.initialize()
        logger.info("Web UI agent initialized")

    @app.on_event("shutdown")
    async def shutdown():
        if _agent:
            await _agent.end_session()

    # --- HTML ---

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return index_path.read_text()
        return _default_html()

    # --- WebSocket Chat (per-connection sessions) ---

    @app.websocket("/ws/chat")
    async def websocket_chat(ws: WebSocket):
        await ws.accept()
        ws_id = str(uuid.uuid4())

        try:
            # Receive initial config
            init = await ws.receive_json()

            # Auth check for WebSocket
            if _auth_config and _auth_config.enabled and _auth_config.api_key:
                token = init.get("token", "")
                if token != _auth_config.api_key:
                    await ws.send_json({"type": "error", "message": "Authentication failed"})
                    await ws.close(code=4001)
                    return

            project = init.get("project")
            session_id = init.get("session_id")
            resume = init.get("resume", False)

            # Each WebSocket gets its own session
            rid = session_id if resume else None
            sid = await _agent.start_session(project=project, resume_session_id=rid)
            _ws_sessions[ws_id] = sid

            await ws.send_json({"type": "session_started", "session_id": sid})

            # Chat loop
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type", "message")

                if msg_type == "message":
                    user_msg = data.get("content", "")
                    if not user_msg:
                        continue

                    await ws.send_json({"type": "thinking"})

                    # Stream tokens to client as they arrive
                    token_queue: asyncio.Queue[str] = asyncio.Queue()
                    chat_done = asyncio.Event()

                    def on_token(token: str):
                        token_queue.put_nowait(token)

                    async def run_chat():
                        try:
                            return await _agent.chat(user_msg, on_token=on_token)
                        finally:
                            chat_done.set()

                    chat_task = asyncio.create_task(run_chat())

                    # Forward tokens as they arrive
                    while not chat_done.is_set() or not token_queue.empty():
                        try:
                            token = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                            await ws.send_json({"type": "token", "content": token})
                        except asyncio.TimeoutError:
                            continue

                    response = await chat_task

                    await ws.send_json({
                        "type": "response_complete",
                        "content": response,
                    })

                elif msg_type == "status":
                    status = _agent.get_status()
                    await ws.send_json({"type": "status", "data": status})

        except WebSocketDisconnect:
            logger.info("WebSocket client %s disconnected", ws_id)
        except Exception as e:
            logger.error("WebSocket error: %s", e)
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            _ws_sessions.pop(ws_id, None)

    # --- REST Endpoints (all require auth when enabled) ---

    @app.get("/api/sessions", dependencies=[Depends(verify_auth)])
    async def list_sessions(limit: int = 50, project: Optional[str] = None):
        sessions = await _agent.sqlite.list_sessions(limit=limit, project=project)
        return [s.model_dump() for s in sessions]

    @app.get("/api/sessions/{session_id}", dependencies=[Depends(verify_auth)])
    async def get_session(session_id: int):
        session = await _agent.sqlite.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.model_dump()

    # --- Memory Browser ---

    @app.get("/api/memories", dependencies=[Depends(verify_auth)])
    async def list_memories(
        page: int = Query(1, ge=1),
        limit: int = Query(20, ge=1, le=100),
        sort: str = Query("recent"),
    ):
        memories, total = await _agent.sqlite.get_memories_paginated(
            page=page, limit=limit, sort=sort,
        )
        return {
            "memories": [m.model_dump() for m in memories],
            "total": total,
            "page": page,
            "pages": (total + limit - 1) // limit if limit else 1,
        }

    @app.get("/api/memories/search", dependencies=[Depends(verify_auth)])
    async def search_memories(query: str, limit: int = 10):
        results = await _agent.search.search(query=query, n_results=limit)
        return [
            {
                "memory_id": r.memory_id,
                "summary": r.summary,
                "similarity": r.similarity,
                "boosted_score": r.boosted_score,
                "rank": r.rank,
                "importance": r.importance,
            }
            for r in results
        ]

    @app.get("/api/memories/stats", dependencies=[Depends(verify_auth)])
    async def memory_stats():
        archive_stats = await _agent.sqlite.get_archive_stats()
        chroma_counts = _agent.chroma.get_counts()
        return {**archive_stats, "chroma": chroma_counts}

    @app.get("/api/memories/{memory_id}", dependencies=[Depends(verify_auth)])
    async def get_memory(memory_id: int):
        result = await _agent.sqlite.get_memory_with_tags(memory_id)
        if not result:
            raise HTTPException(status_code=404, detail="Memory not found")
        return result

    @app.put("/api/memories/{memory_id}", dependencies=[Depends(verify_auth)])
    async def update_memory(memory_id: int, updates: dict):
        allowed = {"rank", "importance", "is_archived"}
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        await _agent.sqlite.update_memory(memory_id, **filtered)
        return {"status": "ok"}

    @app.delete("/api/memories/{memory_id}", dependencies=[Depends(verify_auth)])
    async def archive_memory(memory_id: int):
        await _agent.sqlite.update_memory(memory_id, is_archived=True)
        try:
            _agent.chroma.delete_memory(memory_id)
        except Exception as e:
            from blipshell.memory.chroma_retry import queue_failed_op, OP_DELETE, COLLECTION_MEMORIES
            await queue_failed_op(_agent.sqlite, OP_DELETE, COLLECTION_MEMORIES, memory_id, error=str(e))
        return {"status": "archived"}

    # --- Core Memories ---

    @app.get("/api/core-memories", dependencies=[Depends(verify_auth)])
    async def list_core_memories():
        memories = await _agent.sqlite.get_active_core_memories()
        return [m.model_dump() for m in memories]

    @app.put("/api/core-memories/{core_memory_id}", dependencies=[Depends(verify_auth)])
    async def update_core_memory(core_memory_id: int, updates: dict):
        allowed = {"content", "category", "importance"}
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        await _agent.sqlite.update_core_memory(core_memory_id, **filtered)
        return {"status": "ok"}

    @app.delete("/api/core-memories/{core_memory_id}", dependencies=[Depends(verify_auth)])
    async def deactivate_core_memory(core_memory_id: int):
        await _agent.sqlite.deactivate_core_memory(core_memory_id)
        try:
            _agent.chroma.delete_core_memory(core_memory_id)
        except Exception as e:
            from blipshell.memory.chroma_retry import queue_failed_op, OP_DELETE, COLLECTION_CORE
            await queue_failed_op(_agent.sqlite, OP_DELETE, COLLECTION_CORE, core_memory_id, error=str(e))
        return {"status": "deactivated"}

    # --- Lessons ---

    @app.get("/api/lessons", dependencies=[Depends(verify_auth)])
    async def list_lessons():
        lessons = await _agent.sqlite.get_all_lessons()
        return [l.model_dump() for l in lessons]

    @app.delete("/api/lessons/{lesson_id}", dependencies=[Depends(verify_auth)])
    async def delete_lesson(lesson_id: int):
        await _agent.sqlite.delete_lesson(lesson_id)
        try:
            _agent.chroma.delete_lesson(lesson_id)
        except Exception as e:
            from blipshell.memory.chroma_retry import queue_failed_op, OP_DELETE, COLLECTION_LESSONS
            await queue_failed_op(_agent.sqlite, OP_DELETE, COLLECTION_LESSONS, lesson_id, error=str(e))
        return {"status": "deleted"}

    # --- Config ---

    @app.get("/api/config", dependencies=[Depends(verify_auth)])
    async def get_config():
        return _config_manager.to_dict()

    @app.put("/api/config", dependencies=[Depends(verify_auth)])
    async def update_config(updates: dict):
        for key, value in updates.items():
            _config_manager.set(key, value)
        _config_manager.save()
        return {"status": "ok"}

    # --- Status ---

    @app.get("/api/status", dependencies=[Depends(verify_auth)])
    async def get_status():
        return _agent.get_status()

    @app.get("/api/endpoints", dependencies=[Depends(verify_auth)])
    async def get_endpoints():
        return _agent.endpoint_manager.get_status()

    # --- Flow Observability ---

    @app.get("/api/flow", dependencies=[Depends(verify_auth)])
    async def get_flow(turn: int | None = None):
        session_id = _agent.session_manager.session_id if _agent.session_manager else None
        if not session_id:
            return {"events": []}
        if turn is not None:
            events = await _agent.sqlite.get_turn_events_for_turn(session_id, turn)
        else:
            events = await _agent.sqlite.get_turn_events(session_id, limit=50)
        return {"events": events}

    # --- Data Export ---

    @app.get("/api/export/sessions", dependencies=[Depends(verify_auth)])
    async def export_sessions(format: str = "json"):
        from blipshell.export import export_sessions_json
        data = await export_sessions_json(_agent.sqlite)
        return JSONResponse(
            content=data,
            headers={"Content-Disposition": "attachment; filename=blipshell_sessions.json"},
        )

    @app.get("/api/export/memories", dependencies=[Depends(verify_auth)])
    async def export_memories(format: str = "json", include_archived: bool = False):
        from blipshell.export import export_memories_json
        data = await export_memories_json(_agent.sqlite, include_archived=include_archived)
        return JSONResponse(
            content=data,
            headers={"Content-Disposition": "attachment; filename=blipshell_memories.json"},
        )

    @app.get("/api/export/all", dependencies=[Depends(verify_auth)])
    async def export_all(format: str = "json"):
        from blipshell.export import export_all_json
        data = await export_all_json(_agent.sqlite)
        return JSONResponse(
            content=data,
            headers={"Content-Disposition": "attachment; filename=blipshell_export.json"},
        )

    # --- Worker API Endpoints (for remote worker daemon) ---

    @app.get("/api/worker/poll", dependencies=[Depends(verify_auth)])
    async def worker_poll(endpoint_name: str):
        """Return pending background tasks for a specific endpoint."""
        tasks = await _agent.sqlite.get_pending_background_tasks(
            endpoint_name=endpoint_name,
        )
        return [
            {
                "id": t.id,
                "title": t.title,
                "task_type": t.task_type,
                "prompt": t.prompt,
                "priority": t.priority,
            }
            for t in tasks
        ]

    @app.post("/api/worker/claim/{task_id}", dependencies=[Depends(verify_auth)])
    async def worker_claim(task_id: int):
        """Mark a task as claimed/running by a worker."""
        task = await _agent.sqlite.get_background_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status != "pending":
            raise HTTPException(status_code=409, detail="Task is not pending")
        await _agent.sqlite.update_background_task(
            task_id, status="claimed", progress_message="Claimed by worker",
        )
        return {"status": "claimed"}

    @app.post("/api/worker/complete/{task_id}", dependencies=[Depends(verify_auth)])
    async def worker_complete(task_id: int, body: dict):
        """Report task completion or failure from a worker."""
        task = await _agent.sqlite.get_background_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        status = body.get("status", "completed")
        if status == "completed":
            await _agent.sqlite.update_background_task(
                task_id,
                status="completed",
                result=body.get("result", ""),
                progress_pct=1.0,
                progress_message="Done",
            )
        else:
            await _agent.sqlite.update_background_task(
                task_id,
                status="failed",
                error_message=body.get("error_message", "Unknown error"),
                progress_message="Failed",
            )
        return {"status": "ok"}

    @app.post("/api/worker/progress/{task_id}", dependencies=[Depends(verify_auth)])
    async def worker_progress(task_id: int, body: dict):
        """Report progress update from a worker."""
        task = await _agent.sqlite.get_background_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        await _agent.sqlite.update_background_task(
            task_id,
            progress_pct=body.get("progress_pct", task.progress_pct),
            progress_message=body.get("progress_message", task.progress_message),
        )
        return {"status": "ok"}

    # --- OpenAI-compatible API (for Continue.dev / VS Code) ---

    async def _ensure_api_session() -> int:
        """Lazily start a session for /v1 API requests."""
        global _api_session_id
        if _api_session_id is None:
            _api_session_id = await _agent.start_session(project="vscode-continue")
        return _api_session_id

    @app.get("/v1/models", dependencies=[Depends(verify_auth)])
    async def list_models():
        """Return available models in OpenAI format."""
        return {
            "object": "list",
            "data": [
                {"id": "blipshell", "object": "model", "owned_by": "blipshell"},
                {"id": "blipshell-code", "object": "model", "owned_by": "blipshell"},
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(verify_auth)])
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint."""
        await _ensure_api_session()

        # Extract content: prepend system message if present, use last user message
        system_parts = [
            m.content for m in request.messages if m.role == "system"
        ]
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message provided")

        content = user_messages[-1].content
        if system_parts:
            content = f"[System: {system_parts[-1]}]\n\n{content}"

        if request.stream:
            return StreamingResponse(
                _stream_chat(content),
                media_type="text/event-stream",
            )

        # Non-streaming: call agent and return full response
        response = await _agent.chat(content)
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def _stream_chat(content: str):
        """SSE generator that yields OpenAI-format chunks from Agent.chat()."""
        queue: asyncio.Queue[str] = asyncio.Queue()
        done = asyncio.Event()
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def on_token(token: str):
            queue.put_nowait(token)

        async def run_chat():
            try:
                await _agent.chat(content, on_token=on_token)
            finally:
                done.set()

        task = asyncio.create_task(run_chat())

        # First chunk: role declaration
        first = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n"

        # Stream content tokens
        while not done.is_set() or not queue.empty():
            try:
                token = await asyncio.wait_for(queue.get(), timeout=0.1)
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            except asyncio.TimeoutError:
                continue

        # Final chunk: stop signal
        final = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return app


def _default_html() -> str:
    """Mobile-first chat UI with memory search."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>BlipShell</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body { height: 100%; overflow: hidden; }
        body {
            font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
            background: #1a1a2e; color: #e0e0e0;
            display: flex; flex-direction: column; height: 100dvh;
        }
        /* Header */
        .header {
            padding: 10px 16px; background: #16213e;
            border-bottom: 1px solid #0f3460;
            display: flex; align-items: center; gap: 10px;
            flex-shrink: 0;
        }
        .header h1 { color: #00d9ff; font-size: 18px; }
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #ff9800;
        }
        .status-text { font-size: 11px; color: #888; }
        .header-spacer { flex: 1; }
        .header-btn {
            padding: 6px 10px; background: #0f3460; color: #aaa; border: none;
            border-radius: 6px; cursor: pointer; font-size: 12px;
        }
        .header-btn:hover { color: #00d9ff; }
        .header-btn.active { color: #00d9ff; background: #1a3a6e; }

        /* Views */
        .view { display: none; flex: 1; flex-direction: column; overflow: hidden; }
        .view.active { display: flex; }

        /* Chat */
        .chat-area {
            flex: 1; overflow-y: auto; padding: 12px 16px;
            -webkit-overflow-scrolling: touch;
        }
        .message {
            max-width: 85%; margin-bottom: 12px; padding: 10px 14px;
            border-radius: 16px; line-height: 1.5; white-space: pre-wrap;
            word-wrap: break-word; font-size: 15px;
        }
        .message.user {
            background: #0f3460; margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        .message.assistant {
            background: #1e2a4a;
            border-bottom-left-radius: 4px;
        }
        .message.system {
            background: transparent; font-size: 12px; color: #666;
            text-align: center; max-width: 100%;
        }
        .thinking { display: inline-flex; gap: 4px; padding: 4px 0; }
        .thinking span {
            width: 7px; height: 7px; border-radius: 50%;
            background: #00d9ff; opacity: 0.3;
            animation: pulse 1.4s infinite ease-in-out;
        }
        .thinking span:nth-child(2) { animation-delay: 0.2s; }
        .thinking span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes pulse {
            0%, 80%, 100% { opacity: 0.3; }
            40% { opacity: 1; }
        }

        /* Input */
        .input-area {
            padding: 8px 12px; background: #16213e;
            border-top: 1px solid #0f3460;
            display: flex; gap: 8px; align-items: flex-end;
            flex-shrink: 0;
            padding-bottom: max(8px, env(safe-area-inset-bottom));
        }
        #userInput {
            flex: 1; padding: 10px 14px; border: 1px solid #0f3460;
            border-radius: 20px; background: #1a1a2e; color: #e0e0e0;
            font-size: 16px; outline: none; resize: none;
            max-height: 120px; line-height: 1.4;
        }
        #userInput:focus { border-color: #00d9ff; }
        #sendBtn {
            width: 42px; height: 42px; background: #00d9ff; color: #1a1a2e;
            border: none; border-radius: 50%; cursor: pointer;
            font-size: 18px; display: flex; align-items: center;
            justify-content: center; flex-shrink: 0;
        }
        #sendBtn:disabled { opacity: 0.4; }

        /* Memory search view */
        .mem-view { padding: 12px 16px; overflow-y: auto; flex: 1; }
        .mem-search-bar {
            display: flex; gap: 8px; margin-bottom: 12px;
        }
        .mem-search-bar input {
            flex: 1; padding: 10px 14px; background: #1a1a2e;
            border: 1px solid #0f3460; border-radius: 20px;
            color: #e0e0e0; font-size: 15px; outline: none;
        }
        .mem-search-bar input:focus { border-color: #00d9ff; }
        .mem-search-bar button {
            padding: 10px 16px; background: #00d9ff; color: #1a1a2e;
            border: none; border-radius: 20px; cursor: pointer;
            font-weight: 600; font-size: 14px;
        }
        .mem-card {
            background: #1e2a4a; border-radius: 10px; padding: 12px;
            margin-bottom: 8px; font-size: 14px; line-height: 1.4;
        }
        .mem-card .meta {
            color: #888; font-size: 11px; margin-top: 6px;
        }
        .mem-section-title {
            color: #00d9ff; font-size: 13px; font-weight: 600;
            margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 0.5px;
        }
        .empty { color: #555; font-size: 13px; font-style: italic; }
    </style>
</head>
<body>
    <div class="header">
        <h1>BlipShell</h1>
        <span class="status-dot" id="statusDot"></span>
        <span class="status-text" id="statusText">Connecting...</span>
        <div class="header-spacer"></div>
        <button class="header-btn active" onclick="showView('chat')" id="btnChat">Chat</button>
        <button class="header-btn" onclick="showView('mem')" id="btnMem">Memory</button>
    </div>

    <!-- Chat View -->
    <div id="chatView" class="view active">
        <div class="chat-area" id="chatArea"></div>
        <div class="input-area">
            <textarea id="userInput" rows="1" placeholder="Message BlipShell..."
                oninput="autoGrow(this)"
                onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage()}"></textarea>
            <button id="sendBtn" onclick="sendMessage()">&#9654;</button>
        </div>
    </div>

    <!-- Memory View -->
    <div id="memView" class="view">
        <div class="mem-view">
            <div class="mem-search-bar">
                <input type="text" id="memQuery" placeholder="Search memories..."
                    onkeydown="if(event.key==='Enter')searchMem()">
                <button onclick="searchMem()">Search</button>
            </div>
            <div id="memResults"></div>
        </div>
    </div>

    <script>
        let ws = null, currentResponse = '', activeEl = null, sending = false;
        const API_KEY = '';

        function authHeaders() {
            const h = {'Content-Type': 'application/json'};
            if (API_KEY) h['Authorization'] = 'Bearer ' + API_KEY;
            return h;
        }

        // --- Views ---
        function showView(v) {
            document.getElementById('chatView').classList.toggle('active', v === 'chat');
            document.getElementById('memView').classList.toggle('active', v === 'mem');
            document.getElementById('btnChat').classList.toggle('active', v === 'chat');
            document.getElementById('btnMem').classList.toggle('active', v === 'mem');
            if (v === 'mem') loadMemOverview();
        }

        // --- Chat ---
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + location.host + '/ws/chat');
            ws.onopen = () => {
                setStatus('connected', 'Connected');
                ws.send(JSON.stringify({project: null, resume: false, token: API_KEY || undefined}));
            };
            ws.onmessage = (e) => handleMsg(JSON.parse(e.data));
            ws.onclose = () => { setStatus('off', 'Offline'); setTimeout(connect, 3000); };
            ws.onerror = () => setStatus('error', 'Error');
        }

        function handleMsg(d) {
            switch(d.type) {
                case 'session_started':
                    addSystem('Session #' + d.session_id);
                    break;
                case 'thinking':
                    currentResponse = '';
                    activeEl = addBubble('assistant');
                    activeEl.innerHTML = '<div class="thinking"><span></span><span></span><span></span></div>';
                    scroll();
                    break;
                case 'token':
                    currentResponse += d.content;
                    if (activeEl) activeEl.textContent = currentResponse;
                    scroll();
                    break;
                case 'response_complete':
                    if (activeEl) activeEl.textContent = d.content;
                    activeEl = null;
                    sending = false;
                    document.getElementById('sendBtn').disabled = false;
                    scroll();
                    break;
                case 'error':
                    if (activeEl) {
                        activeEl.textContent = 'Error: ' + d.message;
                        activeEl.style.borderLeft = '3px solid #f44336';
                        activeEl = null;
                    }
                    sending = false;
                    document.getElementById('sendBtn').disabled = false;
                    break;
            }
        }

        function sendMessage() {
            const input = document.getElementById('userInput');
            const msg = input.value.trim();
            if (!msg || !ws || ws.readyState !== 1 || sending) return;
            addBubble('user').textContent = msg;
            ws.send(JSON.stringify({type: 'message', content: msg}));
            input.value = '';
            input.style.height = 'auto';
            sending = true;
            document.getElementById('sendBtn').disabled = true;
            scroll();
        }

        function addBubble(cls) {
            const div = document.createElement('div');
            div.className = 'message ' + cls;
            document.getElementById('chatArea').appendChild(div);
            return div;
        }
        function addSystem(text) {
            const div = addBubble('system');
            div.textContent = text;
        }
        function scroll() {
            const a = document.getElementById('chatArea');
            a.scrollTop = a.scrollHeight;
        }
        function setStatus(s, t) {
            document.getElementById('statusDot').style.background =
                s === 'connected' ? '#4caf50' : s === 'error' ? '#f44336' : '#ff9800';
            document.getElementById('statusText').textContent = t;
        }
        function autoGrow(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 120) + 'px';
        }

        // --- Memory ---
        async function loadMemOverview() {
            const el = document.getElementById('memResults');
            el.innerHTML = '<div class="empty">Loading...</div>';
            try {
                const [coreRes, lessonRes] = await Promise.all([
                    fetch('/api/core-memories', {headers: authHeaders()}),
                    fetch('/api/lessons', {headers: authHeaders()}),
                ]);
                const core = await coreRes.json();
                const lessons = await lessonRes.json();
                let html = '<div class="mem-section-title">Core Memories (' + core.length + ')</div>';
                if (!core.length) html += '<div class="empty">None</div>';
                core.forEach(m => {
                    html += '<div class="mem-card">' + esc(m.content) +
                        '<div class="meta">' + m.category + '</div></div>';
                });
                html += '<div class="mem-section-title">Lessons (' + lessons.length + ')</div>';
                if (!lessons.length) html += '<div class="empty">None</div>';
                lessons.slice(0, 20).forEach(l => {
                    html += '<div class="mem-card">' + esc(l.content).substring(0, 200) +
                        '<div class="meta">Rank: ' + l.rank + '</div></div>';
                });
                if (lessons.length > 20) html += '<div class="empty">...and ' + (lessons.length - 20) + ' more</div>';
                el.innerHTML = html;
            } catch(e) { el.innerHTML = '<div class="empty">Failed to load</div>'; }
        }

        async function searchMem() {
            const q = document.getElementById('memQuery').value.trim();
            if (!q) { loadMemOverview(); return; }
            const el = document.getElementById('memResults');
            el.innerHTML = '<div class="empty">Searching...</div>';
            try {
                const resp = await fetch('/api/memories/search?query=' + encodeURIComponent(q) + '&limit=20', {headers: authHeaders()});
                const results = await resp.json();
                if (!results.length) { el.innerHTML = '<div class="empty">No results</div>'; return; }
                let html = '<div class="mem-section-title">Results (' + results.length + ')</div>';
                results.forEach(r => {
                    html += '<div class="mem-card">' + esc(r.summary).substring(0, 250) +
                        '<div class="meta">Score: ' + r.boosted_score.toFixed(3) + ' | Rank: ' + r.rank + '</div></div>';
                });
                el.innerHTML = html;
            } catch(e) { el.innerHTML = '<div class="empty">Search failed</div>'; }
        }

        function esc(s) {
            const d = document.createElement('div');
            d.textContent = s || '';
            return d.innerHTML;
        }

        connect();
    </script>
</body>
</html>"""
