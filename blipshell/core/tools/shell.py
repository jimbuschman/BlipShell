"""Shell command execution tool with timeout, allowlist, and background support."""

import asyncio
import logging
import platform
import re
import shlex
import sys

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)

# Track background processes across all ShellTool instances
_background_processes: dict[int, asyncio.subprocess.Process] = {}

# Patterns that indicate destructive or dangerous commands.
# When matched, approval is forced even if session-auto-approved.
DESTRUCTIVE_PATTERNS = [
    (r'\brm\s+(-[rf]+\s+)*/', "rm with absolute path"),
    (r'\brm\s+-[rf]*r', "recursive delete"),
    (r'\bgit\s+reset\s+--hard\b', "git reset --hard (discards changes)"),
    (r'\bgit\s+push\s+.*--force\b', "git force push"),
    (r'\bgit\s+push\s+.*-f\b', "git force push"),
    (r'\bgit\s+clean\s+-[fd]', "git clean (removes untracked files)"),
    (r'\bgit\s+checkout\s+\.\s*$', "git checkout . (discards all changes)"),
    (r'\bdrop\s+table\b', "SQL drop table"),
    (r'\btruncate\s+table\b', "SQL truncate table"),
    (r'\brmdir\s+/s\b', "recursive directory delete"),
    (r'\bdel\s+/[sq]\b', "recursive file delete"),
    (r'\bformat\s+[a-z]:', "format drive"),
]


def check_destructive(command: str) -> str | None:
    """Return warning if command matches a destructive pattern, else None."""
    for pattern, description in DESTRUCTIVE_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return description
    return None


class ShellTool(Tool):
    def __init__(self, timeout: int = 30, allowed_commands: list[str] | None = None,
                 cwd: str | None = None):
        self.timeout = timeout
        self.allowed_commands = allowed_commands
        self.cwd = cwd

    def definition(self) -> ToolDefinition:
        os_name = platform.system()  # "Windows", "Linux", "Darwin"
        os_hint = ""
        if os_name == "Windows":
            os_hint = (
                " This is Windows. Do NOT use Unix commands (ls, cat, grep, head, wc). "
                "Use the dedicated tools instead: list_directory, read_file, grep_files, glob_files."
            )
        elif os_name == "Darwin":
            os_hint = " This is macOS — use Unix commands."
        else:
            os_hint = f" This is {os_name} — use Unix commands."

        return ToolDefinition(
            name="run_command",
            description=(
                f"Run a shell command and return its output.{os_hint}\n\n"
                "IMPORTANT:\n"
                "- Do NOT use this for file searching — use grep_files or glob_files instead.\n"
                "- Do NOT use this for reading files — use read_file instead.\n"
                "- Foreground commands time out after 30 seconds. Do NOT run interactive programs.\n"
                "- Use run_in_background=true for long-running processes (tests, servers, builds). "
                "This returns a PID immediately — use check_process to see output later.\n"
                "- Use this for: syntax validation, pip install, git, python -c '...', pytest, etc."
            ),
            parameters=[
                ToolParameter(name="command", type=ToolParameterType.STRING,
                              description="The shell command to execute"),
                ToolParameter(name="timeout", type=ToolParameterType.INTEGER,
                              description="Timeout in seconds for foreground commands (default 30, max 600)",
                              required=False),
                ToolParameter(name="run_in_background", type=ToolParameterType.BOOLEAN,
                              description="Start the process in background and return immediately with a PID (default: false)",
                              required=False),
            ],
        )

    async def execute(self, command: str, timeout: int | None = None,
                      run_in_background: bool = False, **kwargs) -> str:
        if timeout is None:
            timeout = self.timeout
        timeout = min(timeout, 600)  # hard cap at 10 minutes

        # Validate command against allowlist
        if self.allowed_commands:
            base_cmd = self._extract_base_command(command)
            if base_cmd not in self.allowed_commands:
                # Guide model to the right tool instead of another shell command
                alt = {
                    "ls": "Use list_directory tool instead.",
                    "cat": "Use read_file tool instead.",
                    "head": "Use read_file with max_lines parameter instead.",
                    "tail": "Use read_file with start_line parameter instead.",
                    "grep": "Use grep_files tool instead.",
                    "find": "Use glob_files tool instead.",
                    "wc": "Use read_file to check file contents.",
                    "findstr": "Use grep_files tool instead.",
                }
                hint = alt.get(base_cmd, f"Allowed: {', '.join(self.allowed_commands)}")
                return f"Error: '{base_cmd}' is not allowed. {hint}"

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )

            if run_in_background:
                _background_processes[process.pid] = process
                return (
                    f"Started in background (PID: {process.pid}).\n"
                    f"Use check_process(pid={process.pid}) to see output and status."
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return (
                    f"Error: Command timed out after {timeout} seconds. "
                    "The command likely started an interactive process or server. "
                    "Consider using run_in_background=true for long-running commands."
                )

            output = stdout.decode("utf-8", errors="replace").strip()
            errors = stderr.decode("utf-8", errors="replace").strip()

            result_parts = []
            if output:
                result_parts.append(output)
            if errors:
                result_parts.append(f"STDERR: {errors}")
            if process.returncode != 0:
                result_parts.append(f"Exit code: {process.returncode}")

            result = "\n".join(result_parts) if result_parts else "(no output)"

            # Cap output to prevent context blowup from large stack traces etc.
            max_chars = 8000
            if len(result) > max_chars:
                result = result[:max_chars] + f"\n... [truncated — {len(result)} total chars]"

            return result

        except Exception as e:
            return f"Error executing command: {e}"

    @staticmethod
    def _extract_base_command(command: str) -> str:
        """Extract the base command name from a full command string."""
        try:
            parts = shlex.split(command)
            if parts:
                return parts[0].split("/")[-1].split("\\")[-1]
        except ValueError:
            pass
        # Fallback: split on spaces
        return command.strip().split()[0].split("/")[-1].split("\\")[-1] if command.strip() else ""


class CheckProcessTool(Tool):
    """Check the status and output of a background process."""
    read_only = True

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="check_process",
            description=(
                "Check the status and output of a background process started with run_command.\n\n"
                "Returns the process status (running/finished/failed) and any available output.\n"
                "Use this after starting a process with run_in_background=true."
            ),
            parameters=[
                ToolParameter(name="pid", type=ToolParameterType.INTEGER,
                              description="Process ID returned by run_command with run_in_background=true"),
            ],
        )

    async def execute(self, pid: int, **kwargs) -> str:
        process = _background_processes.get(pid)
        if process is None:
            active = list(_background_processes.keys())
            if active:
                return f"Error: No background process with PID {pid}. Active PIDs: {active}"
            return f"Error: No background process with PID {pid}. No background processes are running."

        if process.returncode is None:
            # Still running — try to read available output without blocking
            return f"Process {pid} is still running."

        # Process has finished — collect output and clean up
        try:
            stdout = await process.stdout.read() if process.stdout else b""
            stderr = await process.stderr.read() if process.stderr else b""
        except Exception:
            stdout = stderr = b""

        output = stdout.decode("utf-8", errors="replace").strip()
        errors = stderr.decode("utf-8", errors="replace").strip()

        result_parts = []
        status = "finished" if process.returncode == 0 else "failed"
        result_parts.append(f"Process {pid} {status} (exit code: {process.returncode})")
        if output:
            result_parts.append(f"\nOutput:\n{output}")
        if errors:
            result_parts.append(f"\nSTDERR:\n{errors}")

        result = "\n".join(result_parts)

        # Cap output
        max_chars = 8000
        if len(result) > max_chars:
            result = result[:max_chars] + f"\n... [truncated — {len(result)} total chars]"

        # Clean up
        del _background_processes[pid]
        return result


async def cleanup_background_processes():
    """Kill all remaining background processes. Called on agent shutdown."""
    for pid, process in list(_background_processes.items()):
        try:
            if process.returncode is None:
                process.kill()
                logger.info("Killed background process PID %d", pid)
        except Exception:
            pass
    _background_processes.clear()
