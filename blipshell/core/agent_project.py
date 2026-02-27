"""Project mode mixin for Agent.

Extracts project activation/deactivation and context scanning.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # All types accessed via self

from blipshell.core.tools.code_tools import GlobTool, GrepTool
from blipshell.core.tools.git_tools import (
    GitAddTool, GitCommitTool, GitDiffTool, GitStatusTool,
)
from blipshell.core.tools.interaction_tools import AskUserTool, TaskCompleteTool
from blipshell.core.tools.filesystem import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from blipshell.core.tools.shell import ShellTool
from blipshell.core.repo_map import RepoMap

logger = logging.getLogger(__name__)


class ProjectMixin:
    """Project mode methods mixed into Agent."""

    async def activate_project(self, name: str) -> dict:
        """Activate a project by name. Loads context and re-registers tools.

        Returns the project dict from the DB.
        Raises KeyError if project not found.
        """
        project = await self.sqlite.get_project(name)
        if not project:
            raise KeyError(f"Project '{name}' not found")

        # Dump memory if switching from another project (preserve conversation)
        if self.active_project and self.active_project["name"] != name:
            if self.session_manager:
                await self.session_manager.dump_to_memory()

        self.active_project = project
        root = project.get("root_path")

        # Re-register file tools with project root
        self._register_tools_with_root(root)

        # Register coding tools
        self.tool_registry.register(GrepTool(root_path=root), group="coding")
        self.tool_registry.register(GlobTool(root_path=root), group="coding")

        # Register git tools
        self.tool_registry.register(GitStatusTool(root_path=root), group="coding")
        self.tool_registry.register(GitDiffTool(root_path=root), group="coding")
        self.tool_registry.register(GitAddTool(root_path=root), group="coding")
        self.tool_registry.register(GitCommitTool(root_path=root), group="coding")

        # Register interaction tools for execution
        self.tool_registry.register(
            AskUserTool(callback=self._ask_user_callback), group="general",
        )
        self.tool_registry.register(TaskCompleteTool(), group="general")

        # Initialize repo map for code structure context
        self._repo_map = RepoMap(root)

        # Tag current session with this project
        if self.session_manager and self.session_manager.session_id:
            await self.sqlite.update_session_project(
                self.session_manager.session_id, name,
            )
            self.session_manager.project = name

        # Touch last_active
        await self.sqlite.touch_project(name)

        # Load project context — use cache if fresh (< 1 hour)
        settings = json.loads(project.get("settings_json") or "{}")
        cached = settings.get("project_context")
        cached_at = settings.get("project_context_cached_at", 0)

        if cached and (time.time() - cached_at) < 3600:
            self._project_context = cached
            logger.info("Using cached project context for '%s'", name)
        else:
            self._project_context = await self._scan_project_context(project)
            settings["project_context"] = self._project_context
            settings["project_context_cached_at"] = time.time()
            await self.sqlite.update_project(
                name, settings_json=json.dumps(settings),
            )

        # Sync executor with project state
        if self.task_executor:
            self.task_executor.active_project = self.active_project
            self.task_executor.project_context = self._project_context
            self.task_executor.files_read = self._files_read

        logger.info("Activated project '%s' at %s", name, root)
        return project

    async def deactivate_project(self):
        """Deactivate the current project, reset tools to defaults."""
        if not self.active_project:
            return

        self.active_project = None
        self._project_context = ""
        self._repo_map = None
        # Re-register file tools without root
        self._register_tools_with_root(None)

        # Remove coding and git tools
        self.tool_registry.unregister("grep_files")
        self.tool_registry.unregister("glob_files")
        self.tool_registry.unregister("git_status")
        self.tool_registry.unregister("git_diff")
        self.tool_registry.unregister("git_add")
        self.tool_registry.unregister("git_commit")
        self.tool_registry.unregister("ask_user")
        self.tool_registry.unregister("task_complete")

        # Clear executor project state
        if self.task_executor:
            self.task_executor.active_project = None
            self.task_executor.project_context = ""

        logger.info("Deactivated project")

    def _register_tools_with_root(self, root_path: str | None):
        """Re-register file and shell tools with a root_path (or None to reset)."""
        cfg = self.config.tools

        # Unregister existing file/shell tools and re-register with root_path
        for name in ("read_file", "write_file", "edit_file", "list_directory", "run_command"):
            self.tool_registry.unregister(name)

        self.tool_registry.register(ReadFileTool(
            max_file_size=cfg.filesystem.max_file_size,
            blocked_paths=cfg.filesystem.blocked_paths,
            root_path=root_path,
            files_read=self._files_read,
        ), group="filesystem")
        self.tool_registry.register(WriteFileTool(
            blocked_paths=cfg.filesystem.blocked_paths,
            root_path=root_path,
        ), group="filesystem")
        self.tool_registry.register(EditFileTool(root_path=root_path), group="filesystem")
        self.tool_registry.register(ListDirectoryTool(root_path=root_path), group="filesystem")
        self.tool_registry.register(ShellTool(
            timeout=cfg.shell.timeout,
            allowed_commands=cfg.shell.allowed_commands,
            cwd=root_path,
        ), group="shell")

    async def _scan_project_context(self, project: dict) -> str:
        """Scan a project directory and build a context string for the LLM."""
        import subprocess
        from pathlib import Path

        root = project.get("root_path")
        if not root or not Path(root).is_dir():
            return f"Project: {project['name']}\nRoot path not accessible."

        root_path = Path(root)
        parts = [
            f"Project: {project['name']}",
            f"Root: {root}",
        ]
        if project.get("description"):
            parts.append(f"Description: {project['description']}")
        if project.get("language"):
            parts.append(f"Language: {project['language']}")
        if project.get("git_url"):
            parts.append(f"Git: {project['git_url']}")

        # Git info
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root, capture_output=True, text=True, timeout=5,
            )
            if branch.returncode == 0:
                parts.append(f"Branch: {branch.stdout.strip()}")

            log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=root, capture_output=True, text=True, timeout=5,
            )
            if log.returncode == 0 and log.stdout.strip():
                parts.append(f"\nRecent commits:\n{log.stdout.strip()}")
        except Exception:
            pass

        # Code map: AST-based structure of Python files (replaces file tree)
        if self._repo_map:
            code_map = self._repo_map.build(max_lines=120)
            if code_map:
                parts.append(f"\nCode structure (classes, functions):\n{code_map}")

        # Compact file tree (top level only, for non-Python files/dirs)
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                     ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
                     ".vs", ".idea", ".vscode", "backups"}
        tree_lines = []
        for entry in sorted(root_path.iterdir()):
            if entry.name in skip_dirs:
                continue
            prefix = "[DIR] " if entry.is_dir() else "      "
            tree_lines.append(f"  {prefix}{entry.name}")
        if tree_lines:
            parts.append(f"\nTop-level layout:\n" + "\n".join(tree_lines[:40]))

        # BLIPSHELL.md — project-level instructions (loaded in full, like CLAUDE.md)
        blipshell_md = root_path / "BLIPSHELL.md"
        if blipshell_md.is_file():
            try:
                content = blipshell_md.read_text(encoding="utf-8", errors="replace")
                parts.append(f"\n=== BLIPSHELL.md (project instructions) ===\n{content}")
                logger.info("Loaded BLIPSHELL.md from %s (%d chars)", root, len(content))
            except Exception:
                pass

        # Key files
        key_files = ["README.md", "README.rst", "README.txt", "readme.md",
                     "pyproject.toml", "setup.py", "setup.cfg",
                     "package.json", "Cargo.toml", "go.mod",
                     "requirements.txt", "Makefile", "CLAUDE.md"]
        for fname in key_files:
            fpath = root_path / fname
            if fpath.is_file():
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    lines = content.splitlines()[:60]
                    truncated = "\n".join(lines)
                    if len(content.splitlines()) > 60:
                        truncated += "\n... (truncated)"
                    parts.append(f"\n=== {fname} ===\n{truncated}")
                except Exception:
                    pass

        return "\n".join(parts)
