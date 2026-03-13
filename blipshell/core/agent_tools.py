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
from blipshell.core.tools.memory_tools import (
    ListSessionsTool,
    PromoteToCoreMemoryTool,
    SaveCoreMemoryTool,
    SearchMemoriesTool,
)
from blipshell.core.tools.project_tools import CreateProjectTool
from blipshell.core.tools.shell import CheckProcessTool, ShellTool
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

    async def _connect_mcp_servers(self):
        """Connect to configured MCP servers and register their tools."""
        from blipshell.mcp.manager import MCPManager
        from blipshell.mcp.tools import MCPTool

        self.mcp_manager = MCPManager()

        for server_config in self.config.mcp_servers:
            if not server_config.enabled:
                continue
            try:
                definitions = await self.mcp_manager.connect_server(server_config)
                for defn in definitions:
                    # Strip prefix to get original name for MCP calls
                    prefix = f"mcp_{server_config.name}_"
                    original_name = defn.name[len(prefix):]
                    tool = MCPTool(
                        tool_def=defn,
                        server_name=server_config.name,
                        original_name=original_name,
                        manager=self.mcp_manager,
                        timeout=server_config.timeout,
                    )
                    self.tool_registry.register(tool, group=f"mcp_{server_config.name}")

                    # Require approval unless auto_approve
                    if not server_config.auto_approve:
                        self.tool_registry._tools_requiring_approval.add(defn.name)

                logger.info(
                    "MCP server '%s': %d tools registered",
                    server_config.name, len(definitions),
                )
            except Exception as e:
                logger.error("Failed to connect MCP server '%s': %s", server_config.name, e)

    def set_ask_user_callback(self, callback):
        """Set the callback for ask_user tool (wired by CLI)."""
        self._ask_user_callback = callback

    def set_pause_check_callback(self, callback):
        """Set the callback for mid-task pause checking (wired by CLI)."""
        self._pause_check_callback = callback
