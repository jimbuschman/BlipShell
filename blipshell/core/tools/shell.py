"""Shell command execution tool with timeout and allowlist."""

import asyncio
import platform
import shlex
import sys

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType


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
            description=f"Run a shell command and return its output. Commands are validated against an allowlist.{os_hint}",
            parameters=[
                ToolParameter(name="command", type=ToolParameterType.STRING,
                              description="The shell command to execute"),
                ToolParameter(name="timeout", type=ToolParameterType.INTEGER,
                              description="Timeout in seconds (default 30)", required=False),
            ],
        )

    async def execute(self, command: str, timeout: int | None = None, **kwargs) -> str:
        if timeout is None:
            timeout = self.timeout

        # Validate command against allowlist
        if self.allowed_commands:
            base_cmd = self._extract_base_command(command)
            if base_cmd not in self.allowed_commands:
                return (
                    f"Error: Command '{base_cmd}' is not in the allowed list. "
                    f"Allowed: {', '.join(self.allowed_commands)}"
                )

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return f"Error: Command timed out after {timeout} seconds."

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
