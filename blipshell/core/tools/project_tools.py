"""Project management tool — allows the LLM to create projects from conversation."""

import os
import subprocess
from pathlib import Path

from blipshell.core.tools.base import Tool
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType


class CreateProjectTool(Tool):
    """Allows the LLM to create a new coding project from a directory path."""

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="create_project",
            description=(
                "Create a new coding project from an existing directory. "
                "Auto-detects language and git URL. Use this when the user "
                "wants to turn a directory into a tracked project."
            ),
            parameters=[
                ToolParameter(name="name", type=ToolParameterType.STRING,
                              description="Short project name (e.g. 'blipshell', 'my-api')"),
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="Absolute path to the project directory"),
                ToolParameter(name="description", type=ToolParameterType.STRING,
                              description="Brief project description",
                              required=False),
            ],
        )

    async def execute(self, name: str, path: str, description: str = "", **kwargs) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            return f"Error: Directory not found: {path}"

        existing = await self.sqlite.get_project(name)
        if existing:
            return f"Error: Project '{name}' already exists."

        # Auto-detect language
        language = _detect_language(resolved)

        # Auto-detect git URL
        git_url = None
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(resolved), capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                git_url = result.stdout.strip()
        except Exception:
            pass

        # Auto-detect description from README if not provided
        if not description:
            for readme_name in ("README.md", "README.txt", "README.rst", "README"):
                readme = resolved / readme_name
                if readme.is_file():
                    try:
                        for line in readme.read_text(encoding="utf-8").splitlines():
                            stripped = line.strip().lstrip("#").strip()
                            if stripped:
                                description = stripped[:200]
                                break
                    except Exception:
                        pass
                    break

        await self.sqlite.create_project(
            name=name,
            description=description,
            root_path=str(resolved),
            git_url=git_url,
            language=language,
        )

        parts = [f"Project '{name}' created at {resolved}"]
        if language:
            parts.append(f"Language: {language}")
        if git_url:
            parts.append(f"Git: {git_url}")
        parts.append(f"Activate with: /project {name}")
        return ". ".join(parts)


def _detect_language(path: Path) -> str:
    """Detect the primary language of a project directory from file extensions."""
    ext_counts: dict[str, int] = {}
    lang_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".jsx": "JavaScript", ".tsx": "TypeScript",
        ".rs": "Rust", ".go": "Go", ".java": "Java",
        ".cs": "C#", ".cpp": "C++", ".c": "C",
        ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
        ".kt": "Kotlin", ".scala": "Scala",
    }
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                 "dist", "build", ".tox", ".eggs"}

    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in lang_map:
                lang = lang_map[ext]
                ext_counts[lang] = ext_counts.get(lang, 0) + 1

    if not ext_counts:
        return ""
    return max(ext_counts, key=ext_counts.get)
