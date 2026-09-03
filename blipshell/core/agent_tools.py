"""Tool registration mixin for Agent.

Extracts tool setup methods so agent.py stays focused on orchestration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    pass  # All types accessed via self

from blipshell.core.tools.code_tools import GlobTool, GrepTool
from blipshell.core.tools.filesystem import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from blipshell.core.tools.plan_tools import EnterPlanModeTool, ExitPlanModeTool
from blipshell.core.tools.followup_tools import (
    AddFollowUpTool,
    ListFollowUpsTool,
    ResolveFollowUpTool,
)
from blipshell.core.tools.memory_fs import (
    MemoryCreateTool,
    MemoryDeleteTool,
    MemoryStrReplaceTool,
    MemoryViewTool,
)
from blipshell.core.tools.memory_tools import (
    ListSessionsTool,
    PromoteToCoreMemoryTool,
    SaveCoreMemoryTool,
    SearchMemoriesTool,
)
from blipshell.core.tools.project_tools import (
    ActivateProjectTool,
    CreateProjectTool,
    DeactivateProjectTool,
    ListProjectsTool,
)
from blipshell.core.tools.shell import CheckProcessTool, ShellTool
from blipshell.core.tools.time_tools import GetCurrentTimeTool
from blipshell.core.tools.task_tools import (
    CheckBackgroundTaskTool,
    ListBackgroundTasksTool,
    RunWorkflowTool,
    StartBackgroundTaskTool,
)
from blipshell.core.tools.web import WebFetchTool, WebSearchTool

logger = logging.getLogger(__name__)


class ToolsMixin:
    """Tool registration methods mixed into Agent."""

    def _register_tools(self):
        """Register all tools with their group for selective inclusion."""
        cfg = self.config.tools

        # Filesystem group
        self.tool_registry.register(ReadFileTool(
            max_file_size=cfg.filesystem.max_file_size,
            blocked_paths=cfg.filesystem.blocked_paths,
            files_read=self._files_read,
        ), group="filesystem")
        self.tool_registry.register(WriteFileTool(
            blocked_paths=cfg.filesystem.blocked_paths,
        ), group="filesystem")
        self.tool_registry.register(EditFileTool(), group="filesystem")
        self.tool_registry.register(ListDirectoryTool(), group="filesystem")

        # Code search group (available in all modes)
        self.tool_registry.register(GrepTool(), group="coding")
        self.tool_registry.register(GlobTool(), group="coding")

        # Shell group
        self.tool_registry.register(ShellTool(
            timeout=cfg.shell.timeout,
            allowed_commands=cfg.shell.allowed_commands,
        ), group="shell")
        self.tool_registry.register(CheckProcessTool(), group="shell")

        # Web group
        from blipshell.models.config import resolve_env_vars
        tavily_key = resolve_env_vars(cfg.web.tavily_api_key) if cfg.web.tavily_api_key else None
        self.tool_registry.register(WebSearchTool(tavily_api_key=tavily_key), group="web")
        self.tool_registry.register(WebFetchTool(
            max_size=cfg.web.max_fetch_size,
            timeout=cfg.web.timeout,
            router=self.router,
        ), group="web")

        # Time awareness (read-only, available in all modes)
        self.tool_registry.register(GetCurrentTimeTool(), group="general")

        # Self-transparency (read-only): the architecture card — how its own
        # memory/thought/continuity mechanisms work, consultable on demand.
        from blipshell.core.tools.architecture_tools import DescribeArchitectureTool
        self.tool_registry.register(
            DescribeArchitectureTool(self.config), group="general")

        # Plan mode tools
        self.tool_registry.register(
            EnterPlanModeTool(self.tool_registry), group="general")
        self.tool_registry.register(
            ExitPlanModeTool(self.tool_registry), group="general")

    def _register_memory_tools(self):
        """Register memory tools (needs session_id, so called after session start)."""
        session_id = self.session_manager.session_id if self.session_manager else None

        self.tool_registry.register(SearchMemoriesTool(self.search, session_id), group="memory")
        self.tool_registry.register(SaveCoreMemoryTool(self.processor, session_id), group="memory")
        self.tool_registry.register(PromoteToCoreMemoryTool(
            self.sqlite, self.processor, session_id,
        ), group="memory")
        self.tool_registry.register(ListSessionsTool(self.sqlite), group="memory")
        self.tool_registry.register(CreateProjectTool(self.sqlite), group="general")
        self.tool_registry.register(ListProjectsTool(self.sqlite), group="general")
        self.tool_registry.register(ActivateProjectTool(
            callback=self._activate_project_callback,
        ), group="general")
        self.tool_registry.register(DeactivateProjectTool(
            callback=self._deactivate_project_callback,
        ), group="general")

        # Follow-up queue tools (always available)
        project_name = self.active_project["name"] if self.active_project else None
        self.tool_registry.register(AddFollowUpTool(
            self.sqlite, session_id, project_name,
        ), group="memory")
        self.tool_registry.register(ListFollowUpsTool(
            self.sqlite, project_name,
        ), group="memory")
        self.tool_registry.register(ResolveFollowUpTool(
            self.sqlite, session_id,
        ), group="memory")

        # Memory filesystem tools — exposes lessons/core/digests/sessions/
        # friction/notes as a navigable /memories/... tree. Only core memories
        # and notes are writable; core writes go through the agent's approval
        # callback (path-based, not name-based). The notes tier shares the same
        # session_notes store as the save_note/get_notes tools.
        if self._memory_fs_backend is not None:
            from blipshell.memory.fs_notes import NotesBackend

            def _session_id_provider():
                return self.session_manager.session_id if self.session_manager else None

            self._notes_backend = NotesBackend(
                self.sqlite, self._session_notes, _session_id_provider,
                max_notes=self.config.notes.max_notes,
            )

            mfs_approval = getattr(self, "_ask_user_callback", None)

            async def _memory_approval(name, args):
                """Wrap ask_user as a yes/no approval for memory_fs core writes."""
                if mfs_approval is None:
                    return False
                op = args.get("_operation", "edit")
                prompt = (
                    f"The agent wants to {op} a core memory at "
                    f"{args.get('path', '?')}. Approve? (yes/no)"
                )
                response = await mfs_approval(prompt)
                if isinstance(response, str):
                    return response.strip().lower().startswith("y")
                return bool(response)

            self.tool_registry.register(MemoryViewTool(
                self._memory_fs_backend, self._notes_backend,
            ), group="memory_fs")
            self.tool_registry.register(MemoryCreateTool(
                self._memory_fs_backend, self._notes_backend, _memory_approval,
            ), group="memory_fs")
            self.tool_registry.register(MemoryStrReplaceTool(
                self._memory_fs_backend, self._notes_backend, _memory_approval,
            ), group="memory_fs")
            self.tool_registry.register(MemoryDeleteTool(
                self._memory_fs_backend, self._notes_backend, _memory_approval,
            ), group="memory_fs")

        # Session notes tools (persistent state surviving compaction)
        if self.config.notes.enabled and session_id:
            from blipshell.core.tools.note_tools import (
                DeleteNoteTool,
                GetNotesTool,
                SaveNoteTool,
            )
            # Shared notes dict — all three tools see the same state
            notes = getattr(self, "_session_notes", {})
            self.tool_registry.register(SaveNoteTool(
                self.sqlite, session_id, self.config.notes, notes,
            ), group="memory")
            self.tool_registry.register(GetNotesTool(
                self.sqlite, session_id, notes,
            ), group="memory")
            self.tool_registry.register(DeleteNoteTool(
                self.sqlite, session_id, notes,
            ), group="memory")

    def _register_task_tools(self):
        """Register background task and workflow tools (needs session_id)."""
        session_id = self.session_manager.session_id if self.session_manager else None

        self.tool_registry.register(StartBackgroundTaskTool(
            self.background_manager, session_id,
        ), group="tasks")
        self.tool_registry.register(CheckBackgroundTaskTool(
            self.background_manager,
        ), group="tasks")
        self.tool_registry.register(ListBackgroundTasksTool(
            self.background_manager, session_id,
        ), group="tasks")

        if self.workflow_executor:
            self.tool_registry.register(RunWorkflowTool(
                self.workflow_executor, session_id,
            ), group="tasks")

    async def _activate_project_callback(self, name: str) -> dict:
        """Callback for ActivateProjectTool — delegates to ProjectMixin."""
        return await self.activate_project(name)

    async def _deactivate_project_callback(self) -> str | None:
        """Callback for DeactivateProjectTool — returns deactivated project name."""
        if not self.active_project:
            return None
        name = self.active_project["name"]
        await self.deactivate_project()
        return name

    def set_ask_user_callback(self, callback):
        """Set the callback for ask_user tool (wired by CLI)."""
        self._ask_user_callback = callback

    def set_pause_check_callback(self, callback):
        """Set the callback for mid-task pause checking (wired by CLI)."""
        self._pause_check_callback = callback
