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

                    tokens = []

                    def collect_token(token: str):
                        tokens.append(token)

                    response = await _agent.chat(user_msg, on_token=collect_token)

                    for token in tokens:
                        await ws.send_json({"type": "token", "content": token})

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
        except Exception:
            pass
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
        except Exception:
            pass
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
        except Exception:
            pass
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
    """Default HTML with chat, memory browser, and export."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BlipShell</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #1a1a2e; color: #e0e0e0;
            display: flex; height: 100vh;
        }
        .sidebar {
            width: 260px; background: #16213e; padding: 16px;
            border-right: 1px solid #0f3460; overflow-y: auto;
            display: flex; flex-direction: column;
        }
        .sidebar h2 { color: #00d9ff; margin-bottom: 12px; font-size: 18px; }
        .nav-tabs {
            display: flex; gap: 4px; margin-bottom: 12px;
        }
        .nav-tab {
            flex: 1; padding: 8px 4px; background: #1a1a2e; border: 1px solid #0f3460;
            border-radius: 6px; cursor: pointer; font-size: 11px; text-align: center;
            color: #888;
        }
        .nav-tab.active { background: #0f3460; color: #00d9ff; border-color: #00d9ff; }
        .session-item {
            padding: 10px; margin-bottom: 8px; background: #1a1a2e;
            border-radius: 6px; cursor: pointer; font-size: 13px;
        }
        .session-item:hover { background: #0f3460; }
        .main { flex: 1; display: flex; flex-direction: column; }
        .header {
            padding: 12px 20px; background: #16213e;
            border-bottom: 1px solid #0f3460;
            display: flex; align-items: center; gap: 12px;
        }
        .header h1 { color: #00d9ff; font-size: 20px; }
        .header-actions { margin-left: auto; display: flex; gap: 8px; }
        .header-btn {
            padding: 6px 12px; background: #0f3460; color: #00d9ff; border: 1px solid #0f3460;
            border-radius: 6px; cursor: pointer; font-size: 12px;
        }
        .header-btn:hover { background: #1a3a6e; }
        .status-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: #4caf50; display: inline-block;
        }
        .tab-content { display: none; flex: 1; overflow-y: auto; }
        .tab-content.active { display: flex; flex-direction: column; }
        .chat-area { flex: 1; overflow-y: auto; padding: 20px; }
        .message {
            max-width: 80%; margin-bottom: 16px; padding: 12px 16px;
            border-radius: 12px; line-height: 1.5; white-space: pre-wrap;
        }
        .message.user { background: #0f3460; margin-left: auto; }
        .message.assistant { background: #1e2a4a; }
        .message.system {
            background: #2a1a3e; font-size: 12px; color: #b0b0b0;
            text-align: center; max-width: 100%;
        }
        .thinking-indicator { display: inline-flex; gap: 4px; padding: 4px 0; }
        .thinking-indicator span {
            width: 8px; height: 8px; border-radius: 50%;
            background: #00d9ff; opacity: 0.3;
            animation: pulse 1.4s infinite ease-in-out;
        }
        .thinking-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .thinking-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes pulse {
            0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
            40% { opacity: 1; transform: scale(1); }
        }
        .input-area {
            padding: 16px 20px; background: #16213e;
            border-top: 1px solid #0f3460;
            display: flex; gap: 12px;
        }
        #userInput {
            flex: 1; padding: 12px 16px; border: 1px solid #0f3460;
            border-radius: 8px; background: #1a1a2e; color: #e0e0e0;
            font-size: 14px; outline: none; resize: none;
        }
        #userInput:focus { border-color: #00d9ff; }
        #sendBtn {
            padding: 12px 24px; background: #00d9ff; color: #1a1a2e;
            border: none; border-radius: 8px; cursor: pointer;
            font-weight: bold; font-size: 14px;
        }
        #sendBtn:hover { background: #00b8d4; }
        #sendBtn:disabled { opacity: 0.5; cursor: not-allowed; }
        /* Memory Browser */
        .mem-browser { padding: 20px; overflow-y: auto; flex: 1; }
        .mem-search {
            display: flex; gap: 8px; margin-bottom: 16px;
        }
        .mem-search input {
            flex: 1; padding: 10px 14px; background: #1a1a2e; border: 1px solid #0f3460;
            border-radius: 8px; color: #e0e0e0; font-size: 14px; outline: none;
        }
        .mem-search input:focus { border-color: #00d9ff; }
        .mem-search button {
            padding: 10px 16px; background: #00d9ff; color: #1a1a2e;
            border: none; border-radius: 8px; cursor: pointer; font-weight: bold;
        }
        .mem-section { margin-bottom: 24px; }
        .mem-section h3 { color: #00d9ff; margin-bottom: 8px; font-size: 15px; }
        .mem-card {
            background: #1e2a4a; border-radius: 8px; padding: 12px;
            margin-bottom: 8px; font-size: 13px; position: relative;
        }
        .mem-card .meta { color: #888; font-size: 11px; margin-top: 6px; }
        .mem-card .actions {
            position: absolute; top: 8px; right: 8px; display: flex; gap: 4px;
        }
        .mem-card .actions button {
            padding: 3px 8px; font-size: 11px; border: 1px solid #0f3460;
            background: #16213e; color: #888; border-radius: 4px; cursor: pointer;
        }
        .mem-card .actions button:hover { color: #00d9ff; border-color: #00d9ff; }
        .mem-card .actions button.delete:hover { color: #f44336; border-color: #f44336; }
        .mem-stats {
            display: flex; gap: 16px; margin-bottom: 16px; font-size: 13px; color: #888;
        }
        .mem-stats span { color: #00d9ff; font-weight: bold; }
        .pagination { display: flex; gap: 8px; justify-content: center; margin-top: 16px; }
        .pagination button {
            padding: 6px 14px; background: #0f3460; color: #e0e0e0;
            border: none; border-radius: 6px; cursor: pointer;
        }
        .pagination button:disabled { opacity: 0.4; cursor: not-allowed; }
        .pagination button.active { background: #00d9ff; color: #1a1a2e; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>BlipShell</h2>
        <div class="nav-tabs">
            <div class="nav-tab active" onclick="switchView('chat')">Chat</div>
            <div class="nav-tab" onclick="switchView('memories')">Memories</div>
        </div>
        <div id="sidebarContent">
            <div id="sessionList"></div>
        </div>
    </div>
    <div class="main">
        <div class="header">
            <h1 id="viewTitle">Chat</h1>
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText" style="font-size:12px;color:#888;">Connecting...</span>
            <div class="header-actions">
                <button class="header-btn" onclick="exportAll()">Export</button>
            </div>
        </div>
        <!-- Chat Tab -->
        <div id="chatTab" class="tab-content active">
            <div class="chat-area" id="chatArea"></div>
            <div class="input-area">
                <textarea id="userInput" rows="1" placeholder="Type a message..."
                    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage()}"></textarea>
                <button id="sendBtn" onclick="sendMessage()">Send</button>
            </div>
        </div>
        <!-- Memory Browser Tab -->
        <div id="memTab" class="tab-content">
            <div class="mem-browser">
                <div class="mem-stats" id="memStats"></div>
                <div class="mem-search">
                    <input type="text" id="memSearchInput" placeholder="Search memories..."
                        onkeydown="if(event.key==='Enter')searchMem()">
                    <button onclick="searchMem()">Search</button>
                </div>
                <div class="mem-section">
                    <h3>Core Memories</h3>
                    <div id="coreMemList"></div>
                </div>
                <div class="mem-section">
                    <h3>Lessons</h3>
                    <div id="lessonList"></div>
                </div>
                <div class="mem-section">
                    <h3>Memories</h3>
                    <div id="memList"></div>
                    <div class="pagination" id="memPagination"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let currentResponse = '';
        let activeResponseEl = null;
        let msgCounter = 0;
        let memPage = 1;
        const API_KEY = ''; // Set if auth enabled

        function authHeaders() {
            const h = {'Content-Type': 'application/json'};
            if (API_KEY) h['Authorization'] = 'Bearer ' + API_KEY;
            return h;
        }

        function switchView(view) {
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            if (view === 'chat') {
                document.querySelectorAll('.nav-tab')[0].classList.add('active');
                document.getElementById('chatTab').classList.add('active');
                document.getElementById('viewTitle').textContent = 'Chat';
            } else {
                document.querySelectorAll('.nav-tab')[1].classList.add('active');
                document.getElementById('memTab').classList.add('active');
                document.getElementById('viewTitle').textContent = 'Memory Browser';
                loadMemoryBrowser();
            }
        }

        // --- Chat ---
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws/chat`);
            ws.onopen = () => {
                setStatus('connected', 'Connected');
                const init = {project: null, resume: false};
                if (API_KEY) init.token = API_KEY;
                ws.send(JSON.stringify(init));
            };
            ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
            ws.onclose = () => { setStatus('disconnected', 'Disconnected'); setTimeout(connect, 3000); };
            ws.onerror = () => setStatus('error', 'Error');
        }

        function handleMessage(data) {
            switch(data.type) {
                case 'session_started':
                    addSystemMessage('Session #' + data.session_id + ' started');
                    loadSessions();
                    break;
                case 'thinking':
                    currentResponse = '';
                    activeResponseEl = createAssistantBubble();
                    activeResponseEl.innerHTML = '<div class="thinking-indicator"><span></span><span></span><span></span></div>';
                    scrollToBottom();
                    break;
                case 'token':
                    currentResponse += data.content;
                    if (activeResponseEl) activeResponseEl.textContent = currentResponse;
                    scrollToBottom();
                    break;
                case 'response_complete':
                    if (activeResponseEl) activeResponseEl.textContent = data.content;
                    activeResponseEl = null;
                    document.getElementById('sendBtn').disabled = false;
                    scrollToBottom();
                    break;
                case 'error':
                    if (activeResponseEl) {
                        activeResponseEl.textContent = 'Error: ' + data.message;
                        activeResponseEl.style.borderLeft = '3px solid #f44336';
                        activeResponseEl = null;
                    } else { addSystemMessage('Error: ' + data.message); }
                    document.getElementById('sendBtn').disabled = false;
                    break;
            }
        }

        function sendMessage() {
            const input = document.getElementById('userInput');
            const msg = input.value.trim();
            if (!msg || !ws || ws.readyState !== 1) return;
            addUserMessage(msg);
            ws.send(JSON.stringify({type: 'message', content: msg}));
            input.value = '';
            document.getElementById('sendBtn').disabled = true;
        }

        function addUserMessage(text) {
            const div = document.createElement('div');
            div.className = 'message user'; div.textContent = text;
            document.getElementById('chatArea').appendChild(div); scrollToBottom();
        }
        function createAssistantBubble() {
            const div = document.createElement('div');
            div.className = 'message assistant'; div.id = 'msg-' + (++msgCounter);
            document.getElementById('chatArea').appendChild(div); return div;
        }
        function addSystemMessage(text) {
            const div = document.createElement('div');
            div.className = 'message system'; div.textContent = text;
            document.getElementById('chatArea').appendChild(div);
        }
        function scrollToBottom() {
            const area = document.getElementById('chatArea');
            area.scrollTop = area.scrollHeight;
        }
        function setStatus(state, text) {
            document.getElementById('statusDot').style.background =
                state === 'connected' ? '#4caf50' : state === 'error' ? '#f44336' : '#ff9800';
            document.getElementById('statusText').textContent = text;
        }

        async function loadSessions() {
            try {
                const resp = await fetch('/api/sessions?limit=20', {headers: authHeaders()});
                const sessions = await resp.json();
                const list = document.getElementById('sessionList');
                list.innerHTML = '';
                sessions.forEach(s => {
                    const div = document.createElement('div');
                    div.className = 'session-item';
                    div.textContent = '#' + s.id + ' ' + (s.title || 'Untitled');
                    list.appendChild(div);
                });
            } catch(e) {}
        }

        // --- Memory Browser ---
        async function loadMemoryBrowser() {
            loadMemStats(); loadCoreMem(); loadLessons(); loadMemories(1);
        }

        async function loadMemStats() {
            try {
                const resp = await fetch('/api/memories/stats', {headers: authHeaders()});
                const s = await resp.json();
                document.getElementById('memStats').innerHTML =
                    'Active: <span>' + s.active + '</span> | ' +
                    'Archived: <span>' + s.archived + '</span> | ' +
                    'ChromaDB: <span>' + (s.chroma?.memories||0) + '</span>';
            } catch(e) {}
        }

        async function loadCoreMem() {
            try {
                const resp = await fetch('/api/core-memories', {headers: authHeaders()});
                const mems = await resp.json();
                const el = document.getElementById('coreMemList');
                el.innerHTML = mems.length ? '' : '<div style="color:#666;font-size:12px">No core memories</div>';
                mems.forEach(m => {
                    el.innerHTML += '<div class="mem-card">' + escHtml(m.content) +
                        '<div class="meta">Category: ' + m.category + ' | Importance: ' + (m.importance||0).toFixed(2) + '</div>' +
                        '<div class="actions"><button class="delete" onclick="deactivateCore(' + m.id + ')">Deactivate</button></div></div>';
                });
            } catch(e) {}
        }

        async function loadLessons() {
            try {
                const resp = await fetch('/api/lessons', {headers: authHeaders()});
                const lessons = await resp.json();
                const el = document.getElementById('lessonList');
                el.innerHTML = lessons.length ? '' : '<div style="color:#666;font-size:12px">No lessons</div>';
                lessons.forEach(l => {
                    el.innerHTML += '<div class="mem-card">' + escHtml(l.content).substring(0,300) +
                        '<div class="meta">Rank: ' + l.rank + ' | Importance: ' + (l.importance||0).toFixed(2) + '</div>' +
                        '<div class="actions"><button class="delete" onclick="deleteLesson(' + l.id + ')">Delete</button></div></div>';
                });
            } catch(e) {}
        }

        async function loadMemories(page) {
            memPage = page;
            try {
                const resp = await fetch('/api/memories?page=' + page + '&limit=20', {headers: authHeaders()});
                const data = await resp.json();
                const el = document.getElementById('memList');
                el.innerHTML = '';
                data.memories.forEach(m => {
                    el.innerHTML += '<div class="mem-card">' + escHtml(m.summary || m.content).substring(0,200) +
                        '<div class="meta">Rank: ' + m.rank + ' | Imp: ' + (m.importance||0).toFixed(2) +
                        ' | ' + (m.timestamp||'').substring(0,10) + '</div>' +
                        '<div class="actions">' +
                        '<button class="delete" onclick="archiveMem(' + m.id + ')">Archive</button>' +
                        '</div></div>';
                });
                // Pagination
                const pg = document.getElementById('memPagination');
                pg.innerHTML = '';
                for (let i = 1; i <= data.pages && i <= 10; i++) {
                    pg.innerHTML += '<button ' + (i===page?'class="active"':'') +
                        ' onclick="loadMemories(' + i + ')">' + i + '</button>';
                }
            } catch(e) {}
        }

        async function searchMem() {
            const q = document.getElementById('memSearchInput').value.trim();
            if (!q) { loadMemories(1); return; }
            try {
                const resp = await fetch('/api/memories/search?query=' + encodeURIComponent(q) + '&limit=20', {headers: authHeaders()});
                const results = await resp.json();
                const el = document.getElementById('memList');
                el.innerHTML = '';
                results.forEach(r => {
                    el.innerHTML += '<div class="mem-card">' + escHtml(r.summary).substring(0,200) +
                        '<div class="meta">Score: ' + r.boosted_score.toFixed(3) + ' | Rank: ' + r.rank + '</div></div>';
                });
                document.getElementById('memPagination').innerHTML = '';
            } catch(e) {}
        }

        async function archiveMem(id) {
            await fetch('/api/memories/' + id, {method:'DELETE', headers: authHeaders()});
            loadMemories(memPage); loadMemStats();
        }
        async function deactivateCore(id) {
            await fetch('/api/core-memories/' + id, {method:'DELETE', headers: authHeaders()});
            loadCoreMem();
        }
        async function deleteLesson(id) {
            await fetch('/api/lessons/' + id, {method:'DELETE', headers: authHeaders()});
            loadLessons();
        }
        async function exportAll() {
            window.open('/api/export/all?format=json', '_blank');
        }

        function escHtml(s) {
            const d = document.createElement('div');
            d.textContent = s || '';
            return d.innerHTML;
        }

        connect();
    </script>
</body>
</html>"""
