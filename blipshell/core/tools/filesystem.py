"""Filesystem tools: read, write, edit, list files."""

import os
from pathlib import Path

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType


class ReadFileTool(Tool):
    def __init__(self, max_file_size: int = 1048576, blocked_paths: list[str] | None = None):
        self.max_file_size = max_file_size
        self.blocked_paths = blocked_paths or []

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read the contents of a file at the given path.",
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="Absolute or relative file path to read"),
                ToolParameter(name="max_lines", type=ToolParameterType.INTEGER,
                              description="Maximum number of lines to return", required=False),
            ],
        )

    async def execute(self, path: str, max_lines: int = 0, **kwargs) -> str:
        resolved = Path(path).resolve()
        if self._is_blocked(str(resolved)):
            return f"Error: Access to '{path}' is blocked."
        if not resolved.is_file():
            return f"Error: File '{path}' not found."
        if resolved.stat().st_size > self.max_file_size:
            return f"Error: File '{path}' exceeds max size ({self.max_file_size} bytes)."

        content = resolved.read_text(encoding="utf-8", errors="replace")
        if max_lines > 0:
            lines = content.splitlines()[:max_lines]
            content = "\n".join(lines)
        return content

    def _is_blocked(self, path: str) -> bool:
        return any(blocked in path for blocked in self.blocked_paths)


class WriteFileTool(Tool):
    def __init__(self, blocked_paths: list[str] | None = None):
        self.blocked_paths = blocked_paths or []

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="File path to write to"),
                ToolParameter(name="content", type=ToolParameterType.STRING,
                              description="Content to write"),
            ],
        )

    async def execute(self, path: str, content: str, **kwargs) -> str:
        resolved = Path(path).resolve()
        if any(blocked in str(resolved) for blocked in self.blocked_paths):
            return f"Error: Access to '{path}' is blocked."

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"


class EditFileTool(Tool):
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="edit_file",
            description="Replace a specific string in a file with new content.",
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="File path to edit"),
                ToolParameter(name="old_text", type=ToolParameterType.STRING,
                              description="Text to find and replace"),
                ToolParameter(name="new_text", type=ToolParameterType.STRING,
                              description="Replacement text"),
            ],
        )

    async def execute(self, path: str, old_text: str, new_text: str, **kwargs) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            return f"Error: File '{path}' not found."

        content = resolved.read_text(encoding="utf-8")
        if old_text not in content:
            return f"Error: Text to replace not found in '{path}'."

        new_content = content.replace(old_text, new_text, 1)
        resolved.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {path}"


class ListDirectoryTool(Tool):
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_directory",
            description="List files and directories at the given path.",
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="Directory path to list", required=False),
            ],
        )

    async def execute(self, path: str = ".", **kwargs) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            return f"Error: '{path}' is not a directory."

        entries = []
        try:
            for entry in sorted(resolved.iterdir()):
                prefix = "[DIR] " if entry.is_dir() else "      "
                entries.append(f"{prefix}{entry.name}")
        except PermissionError:
            return f"Error: Permission denied for '{path}'."

        if not entries:
            return f"Directory '{path}' is empty."
        return "\n".join(entries)
