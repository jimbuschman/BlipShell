"""Git tools — dedicated git operations for project-aware coding."""

import asyncio
import logging

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


async def _run_git(args: list[str], cwd: str | None = None,
                   timeout: int = 15) -> tuple[str, str, int]:
    """Run a git command and return (stdout, stderr, returncode)."""
    try:
        process = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except FileNotFoundError:
        return "", "git is not installed or not on PATH", -1

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        return "", "Command timed out", -1

    return (
        stdout.decode("utf-8", errors="replace").strip(),
        stderr.decode("utf-8", errors="replace").strip(),
        process.returncode,
    )


class GitStatusTool(Tool):
    """Show git status: changed, staged, and untracked files."""

    def __init__(self, root_path: str | None = None):
        self.root_path = root_path

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="git_status",
            description="Show git status: changed, staged, and untracked files.",
            parameters=[],
        )

    async def execute(self, **kwargs) -> str:
        stdout, stderr, rc = await _run_git(
            ["status", "--porcelain=v1"], cwd=self.root_path,
        )
        if rc != 0:
            if "not a git repository" in stderr.lower():
                return "Error: Not a git repository."
            return f"Error: {stderr}"
        if not stdout:
            return "Working tree clean — nothing to commit."

        # Parse porcelain output into sections
        staged, modified, untracked = [], [], []
        for line in stdout.splitlines():
            if len(line) < 3:
                continue
            index, work = line[0], line[1]
            path = line[3:]
            if index in ("A", "M", "D", "R"):
                staged.append(f"  {index} {path}")
            if work in ("M", "D"):
                modified.append(f"  {work} {path}")
            if index == "?" and work == "?":
                untracked.append(f"  {path}")

        parts = []
        if staged:
            parts.append("Staged:\n" + "\n".join(staged))
        if modified:
            parts.append("Modified:\n" + "\n".join(modified))
        if untracked:
            parts.append("Untracked:\n" + "\n".join(untracked))
        return "\n\n".join(parts) if parts else stdout


class GitDiffTool(Tool):
    """Show git diff for working tree changes."""

    def __init__(self, root_path: str | None = None):
        self.root_path = root_path

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="git_diff",
            description=(
                "Show git diff for working tree changes or specific files. "
                "Use staged=true for staged (--cached) changes."
            ),
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="Specific file to diff (optional)",
                              required=False),
                ToolParameter(name="staged", type=ToolParameterType.BOOLEAN,
                              description="Show staged changes (--cached)",
                              required=False),
            ],
        )

    async def execute(self, path: str = "", staged: bool = False, **kwargs) -> str:
        args = ["diff"]
        if staged:
            args.append("--cached")
        if path:
            args.extend(["--", path])

        stdout, stderr, rc = await _run_git(args, cwd=self.root_path)
        if rc != 0:
            return f"Error: {stderr}"
        if not stdout:
            return "No differences found."
        # Truncate very long diffs
        if len(stdout) > 10000:
            return stdout[:10000] + "\n... (truncated)"
        return stdout


class GitAddTool(Tool):
    """Stage files for commit."""

    def __init__(self, root_path: str | None = None):
        self.root_path = root_path

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="git_add",
            description="Stage files for commit. Use '.' for all changes.",
            parameters=[
                ToolParameter(name="paths", type=ToolParameterType.STRING,
                              description="Space-separated file paths to stage, or '.' for all"),
            ],
        )

    async def execute(self, paths: str, **kwargs) -> str:
        path_list = paths.strip().split()
        if not path_list:
            return "Error: No paths specified."

        stdout, stderr, rc = await _run_git(["add"] + path_list, cwd=self.root_path)
        if rc != 0:
            return f"Error: {stderr}"
        return f"Staged: {paths}"


class GitCommitTool(Tool):
    """Create a git commit."""

    def __init__(self, root_path: str | None = None):
        self.root_path = root_path

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="git_commit",
            description="Create a git commit with the given message.",
            parameters=[
                ToolParameter(name="message", type=ToolParameterType.STRING,
                              description="Commit message"),
            ],
        )

    async def execute(self, message: str, **kwargs) -> str:
        if not message.strip():
            return "Error: Commit message cannot be empty."

        stdout, stderr, rc = await _run_git(
            ["commit", "-m", message], cwd=self.root_path,
        )
        if rc != 0:
            if "nothing to commit" in stderr.lower() or "nothing to commit" in stdout.lower():
                return "Nothing to commit — working tree clean."
            return f"Error: {stderr or stdout}"
        return stdout or "Commit created."
