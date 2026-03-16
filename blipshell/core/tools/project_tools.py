"""Project management tools — create, list, activate, deactivate projects."""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Awaitable, Callable, Optional

from blipshell.core.tools.base import Tool
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


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


class ListProjectsTool(Tool):
    """Lists all projects so the LLM knows what's available to activate."""

    read_only = True

    def __init__(self, sqlite: SQLiteStore):
        self.sqlite = sqlite

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_projects",
            description=(
                "List all registered projects with their name, path, and language. "
                "Use this to find available projects before calling activate_project."
            ),
            parameters=[],
        )

    async def execute(self, **kwargs) -> str:
        projects = await self.sqlite.list_projects()
        if not projects:
            return "No projects found. Use create_project to create one."

        lines = []
        for p in projects:
            name = p["name"]
            root = p.get("root_path", "?")
            lang = p.get("language", "")
            desc = p.get("description", "")
            parts = [f"  {name} — {root}"]
            if lang:
                parts[0] += f" ({lang})"
            if desc:
                parts.append(f"    {desc[:100]}")
            lines.extend(parts)

        return f"Projects ({len(projects)}):\n" + "\n".join(lines)


class ActivateProjectTool(Tool):
    """Allows the LLM to activate a project, scoping file tools to its directory."""

    read_only = False

    def __init__(
        self,
        callback: Optional[Callable[[str], Awaitable[dict]]] = None,
    ):
        self._callback = callback

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="activate_project",
            description=(
                "Activate a project by name. This scopes file and git tools to the "
                "project directory, loads project context and digest, and switches "
                "to the coding model. Use list_projects first to see available projects. "
                "Call this when the conversation shifts from discussion to implementation."
            ),
            parameters=[
                ToolParameter(
                    name="name",
                    type=ToolParameterType.STRING,
                    description="Project name to activate (use list_projects to see options)",
                ),
            ],
        )

    async def execute(self, name: str, **kwargs) -> str:
        if not self._callback:
            return "Error: Project activation not available in this context."

        try:
            project = await self._callback(name)
        except KeyError:
            return (
                f"Error: Project '{name}' not found. "
                "Use list_projects to see available projects."
            )

        root = project.get("root_path", "")
        lang = project.get("language", "")
        desc = project.get("description", "")

        # Check for digest
        metadata = json.loads(project.get("metadata_json") or "{}")
        digest = metadata.get("digest", {}).get("content", "")

        parts = [f"Activated project '{name}' at {root}."]
        if lang:
            parts.append(f"Language: {lang}.")
        if desc:
            parts.append(desc[:200])
        if digest:
            parts.append(f"\nDigest:\n{digest[:500]}")

        return " ".join(parts) if not digest else parts[0] + " " + " ".join(parts[1:-1]) + parts[-1]


class DeactivateProjectTool(Tool):
    """Allows the LLM to deactivate the current project."""

    read_only = False

    def __init__(
        self,
        callback: Optional[Callable[[], Awaitable[Optional[str]]]] = None,
    ):
        self._callback = callback

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="deactivate_project",
            description=(
                "Deactivate the current project. Removes project root scoping "
                "from file tools and returns to general conversation mode. "
                "Use when done working on a project."
            ),
            parameters=[],
        )

    async def execute(self, **kwargs) -> str:
        if not self._callback:
            return "Error: Project deactivation not available in this context."

        name = await self._callback()
        if name:
            return f"Deactivated project '{name}'. File tools no longer scoped to a project directory."
        return "No active project to deactivate."


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
