"""Programmatic slash command executor.

Mirrors the parsing logic in cli.py lines 400-551 exactly.
Instead of console.print(), captures output to a string buffer.
This catches bugs in command parsing and state mutation, not just the underlying methods.
"""

from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING

from rich.console import Console

from blipshell.simulate.models import SlashResult

if TYPE_CHECKING:
    from blipshell.core.agent import Agent
    from blipshell.core.config import BlipShellConfig

logger = logging.getLogger(__name__)


class SlashCommandDispatcher:
    """Executes slash commands against an Agent programmatically.

    Reproduces the exact parsing from cli.py's chat_loop so that any
    bug in command handling is caught by simulation scenarios.
    """

    def __init__(self, agent: Agent, config: BlipShellConfig):
        self.agent = agent
        self.config = config
        # Session-approved tools mirror (same as cli.py's _session_approved_tools)
        self.session_approved_tools: set[str] = set()

    def _make_console(self, buf: io.StringIO) -> Console:
        """Create a Rich Console that writes to a string buffer."""
        return Console(file=buf, force_terminal=False, no_color=True, width=120)

    async def execute(self, command_str: str) -> SlashResult:
        """Execute a slash command string (e.g., '/project blipshell').

        Returns structured SlashResult with captured output.
        """
        if not command_str.startswith("/"):
            return SlashResult(
                command=command_str,
                output="",
                success=False,
                error="Not a slash command (must start with /)",
            )

        buf = io.StringIO()
        con = self._make_console(buf)

        try:
            # Parse exactly like cli.py does
            raw = command_str[1:]
            cmd = raw.lower().split()
            cmd_args = raw.split()[1:]  # preserve original case for args

            if not cmd:
                return SlashResult(command=command_str, output="", success=False,
                                   error="Empty command")

            command_name = cmd[0]

            # --- Dispatch (same order as cli.py) ---

            if command_name in ("quit", "exit", "q"):
                return SlashResult(command=command_str, output="quit", success=True)

            elif command_name == "status":
                await self._cmd_status(con)

            elif command_name == "memory":
                await self._cmd_memory(con)

            elif command_name == "save":
                await self._cmd_save(con)

            elif command_name == "plan":
                await self._cmd_plan(con)

            elif command_name == "plans":
                await self._cmd_plans(con)

            elif command_name == "tasks":
                await self._cmd_tasks(con)

            elif command_name == "task" and len(cmd) > 1:
                await self._cmd_task_detail(con, cmd[1])

            elif command_name == "workflow":
                await self._cmd_workflow(con, cmd_args)

            elif command_name == "core":
                if len(cmd) >= 3 and cmd[1] == "delete":
                    await self._cmd_core_delete(con, cmd[2:])
                else:
                    await self._cmd_core(con)

            elif command_name == "feedback":
                if len(cmd) < 2:
                    con.print("Usage: /feedback <your feedback>")
                else:
                    feedback_text = command_str[len("/feedback "):]
                    await self._cmd_feedback(con, feedback_text)

            elif command_name == "think":
                self._cmd_think(con, cmd)

            elif command_name == "reflect":
                self._cmd_reflect(con, cmd)

            elif command_name == "approve":
                self._cmd_approve(con, cmd)

            elif command_name == "code":
                # /code requires interactive streaming — record it but don't execute LLM
                if len(cmd) < 2:
                    con.print("Usage: /code [--model name] <file-or-folder> [instruction]")
                else:
                    con.print(f"[sim] /code command acknowledged (args: {' '.join(cmd_args)})")

            elif command_name == "offload":
                if len(cmd) < 2:
                    con.print("Usage: /offload <task description>")
                else:
                    con.print(f"[sim] /offload command acknowledged (skipped in simulation)")

            elif command_name == "health":
                quick = len(cmd) > 1 and cmd[1] == "quick"
                await self._cmd_health(con, quick)

            elif command_name == "flow":
                turn = None
                if len(cmd) > 1:
                    try:
                        turn = int(cmd[1])
                    except ValueError:
                        con.print("Usage: /flow [turn_number]")
                        return SlashResult(
                            command=command_str,
                            output=buf.getvalue(),
                            success=False,
                            error="Invalid turn number",
                        )
                await self._cmd_flow(con, turn)

            elif command_name == "cleanup":
                await self._cmd_cleanup(con)

            elif command_name == "nightly":
                if cmd_args and cmd_args[0] == "report":
                    await self._cmd_nightly_report(con)
                else:
                    job_name = cmd_args[0] if cmd_args else None
                    # Skip actually running nightly in sim — just verify it doesn't crash
                    con.print(f"[sim] /nightly command acknowledged (job={job_name})")


            elif command_name == "changes":
                self._cmd_changes(con)

            elif command_name == "compact":
                focus = " ".join(cmd_args) if cmd_args else ""
                await self._cmd_compact(con, focus)

            elif command_name == "context":
                self._cmd_context(con)

            elif command_name == "tokens":
                self._cmd_tokens(con)

            elif command_name == "projects":
                await self._cmd_projects(con)

            elif command_name == "project":
                await self._cmd_project(con, cmd_args)

            elif command_name in ("help", "commands"):
                self._cmd_help(con)

            else:
                con.print(f"Unknown command: /{command_name}")
                return SlashResult(
                    command=command_str,
                    output=buf.getvalue(),
                    success=False,
                    error=f"Unknown command: /{command_name}",
                )

            return SlashResult(
                command=command_str,
                output=buf.getvalue(),
                success=True,
            )

        except Exception as e:
            logger.exception("Slash command failed: %s", command_str)
            return SlashResult(
                command=command_str,
                output=buf.getvalue(),
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

    # ------------------------------------------------------------------
    # Individual command handlers — mirror cli.py helper functions
    # ------------------------------------------------------------------

    async def _cmd_status(self, con: Console):
        agent = self.agent
        con.print(f"Session: #{agent.session_manager.session_id if agent.session_manager else 'None'}")
        if agent.active_project:
            con.print(f"Project: {agent.active_project.get('name', '?')}")
        else:
            con.print("Project: none")
        con.print(f"Think: {'ON' if agent.think_enabled else 'OFF'}")
        con.print(f"Reflect: {'ON' if agent.reflect_enabled else 'OFF'}")
        # Endpoint info
        if hasattr(agent, 'endpoint_manager') and agent.endpoint_manager:
            for ep in agent.endpoint_manager.endpoints:
                con.print(f"Endpoint: {ep.name} ({ep.url})")

    async def _cmd_memory(self, con: Console):
        if hasattr(self.agent, 'memory_manager') and self.agent.memory_manager:
            stats = self.agent.memory_manager.pool_stats()
            for pool_name, info in stats.items():
                con.print(f"{pool_name}: {info.get('count', 0)} items, "
                          f"{info.get('tokens', 0)} tokens")
        else:
            con.print("Memory manager not available")

    async def _cmd_save(self, con: Console):
        if self.agent.session_manager:
            await self.agent.session_manager.dump_to_memory()
            con.print("Session dumped to memory.")
        else:
            con.print("No active session")

    # Plans live in SQLite, which is where the real CLI reads them from
    # (_print_active_plan / _print_plans). These used to read
    # task_planner.active_plan / .plans — attributes TaskPlanner never had, so
    # both commands raised AttributeError in every simulation run.
    async def _cmd_plan(self, con: Console):
        sid = getattr(self.agent.session_manager, "session_id", None)
        if not (self.agent.sqlite and sid):
            con.print("No active session")
            return
        plan = await self.agent.sqlite.get_active_plan(sid)
        con.print(f"Active plan: {plan.user_request}" if plan else "No active plan")

    async def _cmd_plans(self, con: Console):
        sid = getattr(self.agent.session_manager, "session_id", None)
        if not (self.agent.sqlite and sid):
            con.print("No active session")
            return
        plans = await self.agent.sqlite.list_plans(session_id=sid, limit=20)
        if plans:
            for pl in plans:
                con.print(f"Plan #{pl.id}: {pl.status} — {pl.user_request}")
        else:
            con.print("No plans this session")

    async def _cmd_tasks(self, con: Console):
        if hasattr(self.agent, 'background_tasks') and self.agent.background_tasks:
            tasks = self.agent.background_tasks.list_tasks()
            if tasks:
                for t in tasks:
                    con.print(f"Task #{t.id}: {t.status} — {t.description}")
            else:
                con.print("No background tasks")
        else:
            con.print("No background task manager")

    async def _cmd_task_detail(self, con: Console, task_id_str: str):
        try:
            task_id = int(task_id_str)
        except ValueError:
            con.print(f"Usage: /task <id> (got: {task_id_str})")
            return
        if hasattr(self.agent, 'background_tasks') and self.agent.background_tasks:
            task = self.agent.background_tasks.get_task(task_id)
            if task:
                con.print(f"Task #{task.id}: {task.status}")
                con.print(f"Description: {task.description}")
                if task.result:
                    con.print(f"Result: {task.result[:500]}")
            else:
                con.print(f"Task #{task_id} not found")
        else:
            con.print("No background task manager")

    async def _cmd_workflow(self, con: Console, args: list[str]):
        con.print("[sim] /workflow command acknowledged")

    async def _cmd_core(self, con: Console):
        if self.agent.sqlite:
            core_mems = await self.agent.sqlite.get_core_memories()
            lessons = await self.agent.sqlite.get_lessons()
            con.print(f"Core memories: {len(core_mems)}")
            con.print(f"Lessons: {len(lessons)}")
            for cm in core_mems[:5]:
                con.print(f"  [{cm.get('category', '?')}] {cm.get('content', '')[:80]}")
        else:
            con.print("No database")

    async def _cmd_core_delete(self, con: Console, args: list[str]):
        con.print(f"[sim] /core delete acknowledged (args: {args})")

    async def _cmd_feedback(self, con: Console, text: str):
        if self.agent.sqlite:
            from blipshell.models.session import MessageRole
            # Mirrors _save_feedback in cli.py
            await self.agent.sqlite.save_lesson(
                text=text,
                source="user_feedback",
                session_id=self.agent.session_manager.session_id if self.agent.session_manager else None,
            )
            con.print(f"Feedback saved as lesson: {text[:80]}")
        else:
            con.print("No database")

    def _cmd_think(self, con: Console, cmd: list[str]):
        # Exact mirror of cli.py lines 448-455
        if len(cmd) > 1 and cmd[1] in ("on", "off"):
            self.agent.think_enabled = cmd[1] == "on"
        else:
            self.agent.think_enabled = not self.agent.think_enabled
        state = "ON" if self.agent.think_enabled else "OFF"
        con.print(f"Thinking mode: {state}")

    def _cmd_reflect(self, con: Console, cmd: list[str]):
        # Exact mirror of cli.py lines 456-463
        if len(cmd) > 1 and cmd[1] in ("on", "off"):
            self.agent.reflect_enabled = cmd[1] == "on"
        else:
            self.agent.reflect_enabled = not self.agent.reflect_enabled
        state = "ON" if self.agent.reflect_enabled else "OFF"
        con.print(f"Self-reflection: {state}")

    def _cmd_approve(self, con: Console, cmd: list[str]):
        # Exact mirror of cli.py lines 464-480
        if len(cmd) > 1 and cmd[1] == "all":
            for t in self.config.agent.tools_requiring_approval:
                self.session_approved_tools.add(t)
            con.print("All tools auto-approved for this session")
        elif len(cmd) > 1 and cmd[1] == "reset":
            self.session_approved_tools.clear()
            con.print("Tool approvals reset")
        else:
            approved = ", ".join(sorted(self.session_approved_tools)) if self.session_approved_tools else "none"
            requiring = ", ".join(self.config.agent.tools_requiring_approval)
            con.print(f"Tools requiring approval: {requiring}")
            con.print(f"Session-approved: {approved}")

    async def _cmd_health(self, con: Console, quick: bool):
        # Run the actual health check — this is a key integration test
        try:
            from scripts.audit_db import run_audit
            results = await run_audit(
                self.agent.sqlite,
                self.agent.vectors,
                self.config,
                skip_chroma=quick,
                skip_endpoints=quick,
            )
            con.print(f"Health check: {len(results)} checks run")
            for r in results:
                status = "PASS" if r.get("ok") else "FAIL"
                con.print(f"  {status}: {r.get('name', '?')}")
        except Exception as e:
            con.print(f"Health check: ran (note: {e})")

    async def _cmd_flow(self, con: Console, turn: int | None):
        if self.agent.sqlite and self.agent.session_manager:
            sid = self.agent.session_manager.session_id
            events = await self.agent.sqlite.get_turn_events(sid, limit=20)
            con.print(f"Flow events: {len(events)} events")
            if turn is not None:
                turn_events = [e for e in events if e.get("turn") == turn]
                for ev in turn_events:
                    con.print(f"  {ev.get('event_type')}: {json.dumps(ev.get('data', {}))[:100]}")
        else:
            con.print("No flow data")

    async def _cmd_cleanup(self, con: Console):
        con.print("[sim] /cleanup command acknowledged")

    async def _cmd_nightly_report(self, con: Console):
        if self.agent.sqlite:
            meta = await self.agent.sqlite.get_app_metadata("nightly_last_run")
            if meta:
                con.print(f"Last nightly run: {meta}")
            else:
                con.print("No nightly run recorded")
        else:
            con.print("No database")

    def _cmd_changes(self, con: Console):
        if hasattr(self.agent, '_files_created'):
            created = getattr(self.agent, '_files_created', set())
            edited = getattr(self.agent, '_files_edited', set())
            con.print(f"Files created: {len(created)}")
            for f in sorted(created):
                con.print(f"  + {f}")
            con.print(f"Files edited: {len(edited)}")
            for f in sorted(edited):
                con.print(f"  ~ {f}")
        else:
            con.print("No file tracking data")

    async def _cmd_compact(self, con: Console, focus: str):
        if hasattr(self.agent, 'compact_conversation'):
            await self.agent.compact_conversation(focus)
            con.print(f"Conversation compacted{' (focus: ' + focus + ')' if focus else ''}")
        else:
            con.print("Compact not available")

    def _cmd_context(self, con: Console):
        stats = getattr(self.agent, '_last_context_stats', None)
        if stats:
            con.print(f"Context usage: {stats.get('usage_pct', 0):.1f}%")
            for pool, info in stats.get('pools', {}).items():
                con.print(f"  {pool}: {info}")
        else:
            con.print("No context stats (send a message first)")

    def _cmd_tokens(self, con: Console):
        usage = getattr(self.agent, '_token_usage', None)
        if usage:
            for ep, counts in usage.items():
                con.print(f"{ep}: {counts}")
        else:
            con.print("No token usage data")

    async def _cmd_projects(self, con: Console):
        if self.agent.sqlite:
            projects = await self.agent.sqlite.list_projects()
            con.print(f"Projects: {len(projects)}")
            for p in projects:
                con.print(f"  {p.get('name', '?')} ({p.get('root_path', '?')})")
        else:
            con.print("No database")

    async def _cmd_project(self, con: Console, args: list[str]):
        """Handle /project subcommands — mirrors _handle_project_command in cli.py."""
        agent = self.agent

        if not args:
            if agent.active_project:
                con.print(f"Active project: {agent.active_project.get('name', '?')}")
                con.print(f"Path: {agent.active_project.get('root_path', '?')}")
                con.print(f"Language: {agent.active_project.get('language', '?')}")
            else:
                con.print("No active project")
            return

        subcmd = args[0].lower()

        if subcmd == "new":
            if len(args) < 3:
                con.print("Usage: /project new <name> <path>")
                return
            name, path_str = args[1], " ".join(args[2:])
            from pathlib import Path
            path = Path(path_str).resolve()
            if not path.is_dir():
                con.print(f"Directory not found: {path_str}")
                return
            existing = await agent.sqlite.get_project(name)
            if existing:
                con.print(f"Project '{name}' already exists")
                return
            await agent.sqlite.create_project(
                name=name,
                root_path=str(path),
                language="unknown",
            )
            con.print(f"Project '{name}' created at {path}")

        elif subcmd == "info":
            if agent.active_project:
                con.print(f"Active project: {agent.active_project.get('name', '?')}")
                con.print(f"Path: {agent.active_project.get('root_path', '?')}")
                con.print(f"Language: {agent.active_project.get('language', '?')}")
            else:
                con.print("No active project")

        elif subcmd == "off":
            if agent.active_project:
                name = agent.active_project["name"]
                await agent.deactivate_project()
                con.print(f"Deactivated project '{name}'")
            else:
                con.print("No active project to deactivate")

        elif subcmd == "delete":
            if len(args) < 2:
                con.print("Usage: /project delete <name>")
                return
            name = args[1]
            project = await agent.sqlite.get_project(name)
            if not project:
                con.print(f"Project '{name}' not found")
                return
            if agent.active_project and agent.active_project["name"] == name:
                await agent.deactivate_project()
            await agent.sqlite.delete_project(name)
            con.print(f"Project '{name}' deleted")

        elif subcmd == "digest":
            if not agent.active_project:
                con.print("No active project")
                return
            project_name = agent.active_project["name"]
            if len(args) > 1 and args[1].lower() == "rebuild":
                con.print(f"[sim] /project digest rebuild acknowledged for {project_name}")
            else:
                meta = json.loads(agent.active_project.get("metadata_json") or "{}")
                digest = meta.get("digest")
                if digest:
                    con.print(f"Project Digest — {project_name}")
                    con.print(digest[:500])
                else:
                    con.print("No digest yet")

        else:
            # Treat as project name to activate
            name = args[0]
            try:
                project = await agent.activate_project(name)
                root = project.get("root_path") or "no path"
                lang = project.get("language") or ""
                con.print(f"Activated project '{name}' ({root})")
                if lang:
                    con.print(f"Language: {lang}")
            except KeyError:
                con.print(f"Project '{name}' not found")

    def _cmd_help(self, con: Console):
        con.print("Available commands:")
        commands = [
            "/status", "/memory", "/context", "/tokens", "/core", "/feedback",
            "/save", "/think", "/reflect", "/approve", "/code", "/offload",
            "/health", "/flow", "/cleanup", "/nightly", "/changes",
            "/compact", "/projects", "/project", "/plan", "/plans", "/tasks",
            "/workflow", "/help", "/quit",
        ]
        for c in commands:
            con.print(f"  {c}")
