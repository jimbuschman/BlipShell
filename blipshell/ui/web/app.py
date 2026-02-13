"""FastAPI backend with WebSocket streaming for chat.

REST endpoints for sessions, memories, lessons, config, and endpoint status.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from blipshell.core.agent import Agent
from blipshell.core.config import ConfigManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Global agent instance (created on startup)
_agent: Optional[Agent] = None
_config_manager: Optional[ConfigManager] = None


def create_app(config_path: str | None = None) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="BlipShell", version="0.1.0")

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    async def startup():
        nonlocal _agent, _config_manager
        _config_manager = ConfigManager(config_path)
        config = _config_manager.load()
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

    # --- WebSocket Chat ---

    @app.websocket("/ws/chat")
    async def websocket_chat(ws: WebSocket):
        await ws.accept()

        try:
            # Receive initial config (optional)
            init = await ws.receive_json()
            project = init.get("project")
            session_id = init.get("session_id")
            resume = init.get("resume", False)

            # Start session
            rid = session_id if resume else None
            sid = await _agent.start_session(project=project, resume_session_id=rid)

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

                    # Stream response via WebSocket
                    async def send_token(token: str):
                        await ws.send_json({"type": "token", "content": token})

                    # Use a wrapper since on_token needs to be sync
                    tokens = []

                    def collect_token(token: str):
                        tokens.append(token)

                    response = await _agent.chat(user_msg, on_token=collect_token)

                    # Send collected tokens
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
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error("WebSocket error: %s", e)
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    # --- REST Endpoints ---

    @app.get("/api/sessions")
    async def list_sessions(limit: int = 50, project: Optional[str] = None):
        sessions = await _agent.sqlite.list_sessions(limit=limit, project=project)
        return [s.model_dump() for s in sessions]

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: int):
        session = await _agent.sqlite.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        return session.model_dump()

    @app.get("/api/memories/search")
    async def search_memories(query: str, limit: int = 10):
        results = await _agent.search.search(query=query, n_results=limit)
        return [
            {
                "memory_id": r.memory_id,
                "summary": r.summary,
                "similarity": r.similarity,
                "boosted_score": r.boosted_score,
                "rank": r.rank,
            }
            for r in results
        ]

    @app.get("/api/core-memories")
    async def list_core_memories():
        memories = await _agent.sqlite.get_active_core_memories()
        return [m.model_dump() for m in memories]

    @app.get("/api/lessons")
    async def list_lessons():
        lessons = await _agent.sqlite.get_all_lessons()
        return [l.model_dump() for l in lessons]

    @app.get("/api/config")
    async def get_config():
        return _config_manager.to_dict()

    @app.put("/api/config")
    async def update_config(updates: dict):
        for key, value in updates.items():
            _config_manager.set(key, value)
        _config_manager.save()
        return {"status": "ok"}

    @app.get("/api/status")
    async def get_status():
        return _agent.get_status()

    @app.get("/api/endpoints")
    async def get_endpoints():
        return _agent.endpoint_manager.get_status()

    return app


def _default_html() -> str:
    """Default HTML when no static files exist."""
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
        }
        .sidebar h2 { color: #00d9ff; margin-bottom: 16px; font-size: 18px; }
        .session-item {
            padding: 10px; margin-bottom: 8px; background: #1a1a2e;
            border-radius: 6px; cursor: pointer; font-size: 13px;
        }
        .session-item:hover { background: #0f3460; }
        .main {
            flex: 1; display: flex; flex-direction: column;
        }
        .header {
            padding: 12px 20px; background: #16213e;
            border-bottom: 1px solid #0f3460;
            display: flex; align-items: center; gap: 12px;
        }
        .header h1 { color: #00d9ff; font-size: 20px; }
        .status-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: #4caf50; display: inline-block;
        }
        .chat-area {
            flex: 1; overflow-y: auto; padding: 20px;
        }
        .message {
            max-width: 80%; margin-bottom: 16px; padding: 12px 16px;
            border-radius: 12px; line-height: 1.5; white-space: pre-wrap;
        }
        .message.user {
            background: #0f3460; margin-left: auto;
        }
        .message.assistant {
            background: #1e2a4a;
        }
        .message.system {
            background: #2a1a3e; font-size: 12px; color: #b0b0b0;
            text-align: center; max-width: 100%;
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
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Sessions</h2>
        <div id="sessionList"></div>
    </div>
    <div class="main">
        <div class="header">
            <h1>BlipShell</h1>
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText" style="font-size:12px;color:#888;">Connecting...</span>
        </div>
        <div class="chat-area" id="chatArea"></div>
        <div class="input-area">
            <textarea id="userInput" rows="1" placeholder="Type a message..."
                      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage()}"></textarea>
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let ws = null;
        let currentResponse = '';

        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws/chat`);

            ws.onopen = () => {
                setStatus('connected', 'Connected');
                ws.send(JSON.stringify({project: null, resume: false}));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onclose = () => {
                setStatus('disconnected', 'Disconnected');
                setTimeout(connect, 3000);
            };

            ws.onerror = () => setStatus('error', 'Error');
        }

        function handleMessage(data) {
            switch(data.type) {
                case 'session_started':
                    addSystemMessage(`Session #${data.session_id} started`);
                    loadSessions();
                    break;
                case 'thinking':
                    currentResponse = '';
                    addAssistantMessage('');
                    break;
                case 'token':
                    currentResponse += data.content;
                    updateLastAssistant(currentResponse);
                    break;
                case 'response_complete':
                    updateLastAssistant(data.content);
                    document.getElementById('sendBtn').disabled = false;
                    break;
                case 'error':
                    addSystemMessage('Error: ' + data.message);
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
            div.className = 'message user';
            div.textContent = text;
            document.getElementById('chatArea').appendChild(div);
            scrollToBottom();
        }

        function addAssistantMessage(text) {
            const div = document.createElement('div');
            div.className = 'message assistant';
            div.id = 'lastAssistant';
            div.textContent = text;
            document.getElementById('chatArea').appendChild(div);
            scrollToBottom();
        }

        function updateLastAssistant(text) {
            const el = document.getElementById('lastAssistant');
            if (el) el.textContent = text;
            scrollToBottom();
        }

        function addSystemMessage(text) {
            const div = document.createElement('div');
            div.className = 'message system';
            div.textContent = text;
            document.getElementById('chatArea').appendChild(div);
        }

        function scrollToBottom() {
            const area = document.getElementById('chatArea');
            area.scrollTop = area.scrollHeight;
        }

        function setStatus(state, text) {
            const dot = document.getElementById('statusDot');
            const txt = document.getElementById('statusText');
            dot.style.background = state === 'connected' ? '#4caf50' : state === 'error' ? '#f44336' : '#ff9800';
            txt.textContent = text;
        }

        async function loadSessions() {
            try {
                const resp = await fetch('/api/sessions?limit=20');
                const sessions = await resp.json();
                const list = document.getElementById('sessionList');
                list.innerHTML = '';
                sessions.forEach(s => {
                    const div = document.createElement('div');
                    div.className = 'session-item';
                    div.textContent = `#${s.id} ${s.title || 'Untitled'}`;
                    list.appendChild(div);
                });
            } catch(e) {}
        }

        connect();
    </script>
</body>
</html>"""
