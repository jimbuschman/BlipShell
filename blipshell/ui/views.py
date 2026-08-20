"""Rendering and command-action functions for the CLI.

Extracted from cli.py 2026-08-05: ~1,850 lines of presentation that had no
coupling to the chat loop beyond the shared Console. Everything here is
either a `_print_*` renderer or the action behind a slash command.

These are moved verbatim — same names, same signatures — so the extraction
is reviewable as a move rather than a rewrite.
"""

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from blipshell.core.agent import Agent
from blipshell.models.session import MessageRole
from blipshell.ui.console import console

logger = logging.getLogger(__name__)


async def _print_thoughts(agent: Agent):
    """Show the self-thought store — the observability surface for the
    lingering-thoughts layer and the self-gravity step-1 readout."""
    from datetime import datetime, timezone

    store = getattr(agent, "_self_thoughts", None)
    if store is None:
        console.print("[yellow]Self-reflection layer not initialized.[/yellow]")
        return

    rows = await store.snapshot()
    refl = agent.config.reflection
    gravity_on = getattr(refl, "gravity_enabled", False)
    marker_w = getattr(refl, "gravity_marker_weight", 1.5)

    state = "[green]ON[/green]" if gravity_on else "[yellow]OFF[/yellow]"
    console.print(f"[dim]Self-gravity: {state}"
                  + (f"  (recur ≥{refl.gravity_recur_threshold} → +{refl.gravity_recur_boost},"
                     f" fatigue ×{refl.gravity_fatigue},"
                     f" half-life {refl.gravity_half_life_days:g}d,"
                     f" recurring marker ≥{marker_w:g})" if gravity_on else "")
                  + "[/dim]")

    if not rows:
        console.print("[dim]No lingering thoughts stored yet — they form after "
                      f"~{refl.idle_seconds / 3600:.1f}h of quiet.[/dim]")
        return

    now = datetime.now(timezone.utc)

    def _fmt_age(iso) -> str:
        """Compact relative age: 12m, 5h, 3d, 2w. '?' when undated."""
        if not iso:
            return "?"
        try:
            secs = (now - datetime.fromisoformat(iso)).total_seconds()
        except (TypeError, ValueError):
            return "?"
        if secs < 60:
            return "now"
        if secs < 3600:
            return f"{int(secs // 60)}m"
        if secs < 86400:
            return f"{int(secs // 3600)}h"
        if secs < 7 * 86400:
            return f"{int(secs // 86400)}d"
        return f"{int(secs // (7 * 86400))}w"

    table = Table(title=f"Lingering Thoughts ({len(rows)})")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Age", style="dim", width=5)
    table.add_column("Status", width=9)
    table.add_column("Weight", justify="right", width=12)
    # The two gravity channels, side by side. Echoes reinforce, surfacings
    # fatigue — seeing the ratio is the whole point of the readout: a thought
    # with echoes is one it returns to on its own; one with only surfacings is
    # just being indexed well.
    table.add_column("Echo", justify="right", width=4)
    table.add_column("Surf", justify="right", width=4)
    table.add_column("Thought")

    no_embedding = 0
    total_echoes = 0
    # Thoughts carrying weight above the 1.0 baseline. `echo_count` only exists
    # from 2026-07-30, so a thought boosted before that has the reinforcement
    # in its WEIGHT and nothing in its counter — see the footer below.
    legacy_boosted = 0
    for i, r in enumerate(rows, 1):
        age = _fmt_age(r["created_at"])
        status = "[dim]surfaced[/dim]" if r["surfaced"] else "[yellow]pending[/yellow]"
        eff = r["effective_weight"]
        if eff is None:
            weight = "—"
        else:
            weight = f"{r['weight']:.2f} → {eff:.2f}"
        echoes = r.get("echo_count", 0)
        total_echoes += echoes
        if not echoes and r["weight"] > 1.0:
            legacy_boosted += 1
        echo_cell = f"[magenta]{echoes}[/magenta]" if echoes else "[dim]0[/dim]"
        surfs = r.get("surface_count", 0)
        surf_cell = str(surfs) if surfs else "[dim]0[/dim]"
        text = r["text"][:90] + ("..." if len(r["text"]) > 90 else "")
        if eff is not None and eff >= marker_w:
            text = f"[bold]{text}[/bold] [magenta]· recurring[/magenta]"
        if not r["has_embedding"]:
            no_embedding += 1
            text += " [red](no embedding — can't resurface)[/red]"
        table.add_row(str(i), age, status, weight, echo_cell, surf_cell, text)
    console.print(table)

    if gravity_on:
        console.print("[dim]Weight is stored → effective (age-decayed). Recurrence "
                      "reinforces, surfacing fatigues. A thought marked "
                      "'recurring' keeps coming back on its own — that's the "
                      "gravity signal.[/dim]")
        console.print(
            f"[dim]Echo = times a thought recurred (reinforcement); "
            f"Surf = times it reached the prompt (fatigue). "
            f"{total_echoes} echo(es) across {len(rows)} thought(s).[/dim]"
        )
        if not total_echoes and legacy_boosted:
            # Do NOT say "nothing has recurred" here. On 2026-08-20 that
            # sentence was read as evidence the layer had never worked, while
            # weights up to 3.50 (= 1.0 + 5 boosts) proved recurrence had
            # fired repeatedly — before `echo_count` existed to count it.
            # The counter being empty and the history being empty are
            # different claims, and only one of them is checkable here.
            console.print(
                f"[dim yellow]Counter reads 0, but {legacy_boosted} thought(s) "
                "carry weight above the 1.0 baseline — recurrence DID fire, "
                "before `echo_count` was added (2026-07-30) to record it. So "
                "this is 'nothing counted yet', NOT 'nothing ever recurred'. "
                "Only echoes on thoughts newer than that date are "
                "measurable.[/dim yellow]"
            )
        elif not total_echoes:
            console.print(
                "[dim yellow]No echoes recorded yet — nothing has recurred, so "
                "gravity is only decaying. Re-read this after ≥10 NEW thoughts "
                "before judging the layer.[/dim yellow]"
            )
    if no_embedding:
        console.print(f"[dim yellow]{no_embedding} thought(s) lack embeddings and "
                      "will be backfilled on next relevance check.[/dim yellow]")


async def _print_core(agent: Agent):
    """Print everything in the Core memory pool — core memories and lessons."""
    if not agent.sqlite:
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    core_memories = await agent.sqlite.get_active_core_memories()
    lessons = await agent.sqlite.get_all_lessons()

    if not core_memories and not lessons:
        console.print("[dim]Core memory is empty.[/dim]")
        return

    if core_memories:
        table = Table(title="Core Memories")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Content")
        table.add_column("Category", style="dim")
        table.add_column("Importance", justify="right")

        for cm in core_memories:
            table.add_row(
                str(cm.id),
                cm.content[:80] + ("..." if len(cm.content) > 80 else ""),
                cm.category or "-",
                f"{cm.importance:.1f}",
            )
        console.print(table)

    if lessons:
        table = Table(title=f"Lessons ({len(lessons)})")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Content")
        table.add_column("Rank", justify="right")
        table.add_column("Importance", justify="right")
        table.add_column("Source", style="dim")

        for lesson in lessons:
            source = f"Session #{lesson.source_session_id}" if lesson.source_session_id else "-"
            table.add_row(
                str(lesson.id),
                lesson.content[:80] + ("..." if len(lesson.content) > 80 else ""),
                str(lesson.rank),
                f"{lesson.importance:.1f}",
                source,
            )
        console.print(table)

    total_tokens = sum(
        len(cm.content.split()) * 2 for cm in core_memories
    ) + sum(
        len(l.content.split()) * 2 for l in lessons
    )
    console.print(f"\n[dim]{len(core_memories)} core memories + {len(lessons)} lessons (~{total_tokens} tokens)[/dim]")


async def _delete_core_item(agent: Agent, args: list[str]):
    """Delete a core memory or lesson by type and ID.

    Usage: /core delete lesson <id> | /core delete memory <id>
    """
    if len(args) < 2 or args[0] not in ("lesson", "memory"):
        console.print("[yellow]Usage: /core delete lesson <id> | /core delete memory <id>[/yellow]")
        return

    item_type = args[0]
    try:
        item_id = int(args[1])
    except ValueError:
        console.print("[yellow]ID must be a number.[/yellow]")
        return

    if item_type == "lesson":
        lesson = await agent.sqlite.get_lesson(item_id)
        if not lesson:
            console.print(f"[yellow]Lesson #{item_id} not found.[/yellow]")
            return
        await agent.sqlite.delete_lesson(item_id)
        try:
            agent.vectors.delete_lesson(item_id)
        except Exception as e:
            logging.getLogger(__name__).debug("Lesson vector delete failed: %s", e)
        console.print(f"[green]Lesson #{item_id} deleted.[/green]")
    else:
        cm = await agent.sqlite.get_core_memory(item_id)
        if not cm:
            console.print(f"[yellow]Core memory #{item_id} not found.[/yellow]")
            return
        await agent.sqlite.deactivate_core_memory(item_id)
        try:
            agent.vectors.delete_core_memory(item_id)
        except Exception as e:
            logging.getLogger(__name__).debug("Core memory vector delete failed: %s", e)
        console.print(f"[green]Core memory #{item_id} deactivated.[/green]")


async def _save_feedback(agent: Agent, feedback: str):
    """Save user feedback as a lesson so the LLM learns from it."""
    if not agent.processor:
        console.print("[yellow]Memory processor not initialized.[/yellow]")
        return

    from blipshell.models.memory import Lesson

    session_id = agent.session_manager.session_id if agent.session_manager else None

    lesson = Lesson(
        content=f"User feedback: {feedback}",
        summary=feedback,
        rank=4,  # high — explicit user feedback
        importance=0.8,
        source_session_id=session_id,
        tags=["feedback"],
    )

    lesson_id = await agent.sqlite.create_lesson(lesson)

    # Embed so it surfaces in semantic search
    try:
        agent.vectors.add_lesson(lesson_id, lesson.content)
    except Exception as e:
        logging.getLogger(__name__).debug("Feedback embed failed: %s", e)

    # Tag it
    try:
        await agent.sqlite.tag_lesson(lesson_id, ["feedback", "user-preference"])
    except Exception:
        pass

    console.print(f"[green]Feedback saved as lesson #{lesson_id}.[/green]")


async def _list_projects(agent: Agent):
    """List all registered projects."""
    if not agent.sqlite:
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    projects = await agent.sqlite.list_projects()
    if not projects:
        console.print("[dim]No projects. Create one with /project new <name> <path>[/dim]")
        return

    table = Table(title="Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Path")
    table.add_column("Language", style="dim")
    table.add_column("Last Active")
    table.add_column("", justify="center")

    active_name = agent.active_project.get("name") if agent.active_project else None

    for p in projects:
        marker = "[green]>>>[/green]" if p["name"] == active_name else ""
        last_active = (p.get("last_active") or "")[:19]
        table.add_row(
            p["name"],
            p.get("root_path") or "-",
            p.get("language") or "-",
            last_active,
            marker,
        )

    console.print(table)


async def _handle_project_command(agent: Agent, args: list[str]):
    """Handle /project subcommands."""
    if not args:
        if agent.active_project:
            _print_project_info(agent)
        else:
            console.print(
                "[dim]No active project. Use /project <name> to activate, "
                "or /project new <name> <path> to create.[/dim]"
            )
        return

    subcmd = args[0].lower()

    if subcmd == "new":
        if len(args) < 3:
            console.print("[yellow]Usage: /project new <name> <path>[/yellow]")
            return
        await _create_project(agent, args[1], " ".join(args[2:]))

    elif subcmd == "info":
        if agent.active_project:
            _print_project_info(agent)
        else:
            console.print("[dim]No active project.[/dim]")

    elif subcmd == "off":
        if agent.active_project:
            name = agent.active_project["name"]
            await agent.deactivate_project()
            console.print(f"[dim]Deactivated project '{name}'.[/dim]")
        else:
            console.print("[dim]No active project to deactivate.[/dim]")

    elif subcmd == "delete":
        if len(args) < 2:
            console.print("[yellow]Usage: /project delete <name>[/yellow]")
            return
        name = args[1]
        project = await agent.sqlite.get_project(name)
        if not project:
            console.print(f"[yellow]Project '{name}' not found.[/yellow]")
            return
        if agent.active_project and agent.active_project["name"] == name:
            await agent.deactivate_project()
        await agent.sqlite.delete_project(name)
        console.print(f"[green]Project '{name}' deleted (files on disk untouched).[/green]")

    elif subcmd == "digest":
        if not agent.active_project:
            console.print("[dim]No active project. Activate one first.[/dim]")
            return
        project_name = agent.active_project["name"]
        if len(args) > 1 and args[1].lower() == "rebuild":
            from blipshell.memory.project_digest import ProjectDigestManager
            digest_mgr = ProjectDigestManager(agent.sqlite, agent.router, agent.vectors)
            with console.status("[dim]Rebuilding project digest...[/dim]", spinner="dots"):
                digest = await digest_mgr.bootstrap_digest(project_name)
            if digest:
                console.print(Panel(digest, title=f"Project Digest — {project_name} (rebuilt)"))
            else:
                console.print("[dim]No data found for this project (no sessions or memories mention it).[/dim]")
        else:
            import json
            project = agent.active_project
            meta = json.loads(project.get("metadata_json") or "{}")
            digest = meta.get("digest")
            if digest:
                updated_at = meta.get("digest_updated_at", "unknown")
                session_count = len(meta.get("digest_session_ids", []))
                console.print(Panel(
                    digest,
                    title=f"Project Digest — {project_name}",
                    subtitle=f"Updated: {updated_at[:19]} | Sessions: {session_count}",
                ))
            else:
                console.print(
                    "[dim]No digest yet. Use /project digest rebuild to generate one.[/dim]"
                )

    else:
        # Treat as project name to activate
        name = args[0]
        try:
            with console.status("[dim]Loading project...[/dim]", spinner="dots"):
                project = await agent.activate_project(name)
            root = project.get("root_path") or "no path"
            lang = project.get("language") or ""
            console.print(f"[green]Activated project '{name}'[/green] ({root})")
            if lang:
                console.print(f"[dim]Language: {lang}[/dim]")
        except KeyError:
            console.print(
                f"[yellow]Project '{name}' not found. "
                f"Use /projects to list or /project new to create.[/yellow]"
            )


async def _create_project(agent: Agent, name: str, path_str: str):
    """Create a new project from an existing directory."""
    import subprocess
    from pathlib import Path

    path = Path(path_str).resolve()
    if not path.is_dir():
        console.print(f"[yellow]Directory not found: {path_str}[/yellow]")
        return

    existing = await agent.sqlite.get_project(name)
    if existing:
        console.print(
            f"[yellow]Project '{name}' already exists. "
            f"Use /project delete {name} first.[/yellow]"
        )
        return

    # Auto-detect language from file extensions
    language = _detect_language(path)

    # Auto-detect git URL
    git_url = None
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(path), capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_url = result.stdout.strip()
    except Exception:
        pass

    # Auto-detect description from README first line
    description = ""
    for readme_name in ("README.md", "README.txt", "README.rst", "README"):
        readme = path / readme_name
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

    await agent.sqlite.create_project(
        name=name,
        description=description,
        root_path=str(path),
        git_url=git_url,
        language=language,
    )

    console.print(f"[green]Created project '{name}'[/green]")
    console.print(f"  Path: {path}")
    if language:
        console.print(f"  Language: {language}")
    if git_url:
        console.print(f"  Git: {git_url}")
    if description:
        console.print(f"  Description: {description}")
    console.print(f"\n[dim]Activate with: /project {name}[/dim]")


def _detect_language(path) -> str:
    """Detect the primary language of a project directory from file extensions."""
    from pathlib import Path

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


def _print_project_info(agent: Agent):
    """Print detailed information about the active project."""
    proj = agent.active_project
    if not proj:
        console.print("[dim]No active project.[/dim]")
        return

    table = Table(title=f"Project: {proj['name']}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Name", proj["name"])
    table.add_row("Path", proj.get("root_path") or "-")
    table.add_row("Language", proj.get("language") or "-")
    table.add_row("Git URL", proj.get("git_url") or "-")
    table.add_row("Description", proj.get("description") or "-")
    table.add_row("Created", (proj.get("created_at") or "")[:19])
    table.add_row("Last Active", (proj.get("last_active") or "")[:19])

    console.print(table)

    if agent._project_context:
        lines = agent._project_context.splitlines()
        console.print(f"\n[dim]Project context loaded: {len(lines)} lines[/dim]")


async def _handle_code_command(agent: Agent, args_str: str):
    """Handle /code [--model name] <path> [instruction] — send code to LLM for review."""
    from pathlib import Path

    from blipshell.llm.router import TaskType

    # Parse optional --model flag
    model_override = None
    remaining = args_str.strip()
    if remaining.startswith("--model "):
        parts = remaining.split(None, 2)  # --model, modelname, rest
        if len(parts) >= 2:
            model_override = parts[1]
            remaining = parts[2] if len(parts) > 2 else ""

    if not remaining.strip():
        console.print("[yellow]Usage: /code [--model name] <file-or-folder> [instruction][/yellow]")
        return

    # Parse: first token is the path, rest is instruction
    parts = remaining.strip().split(None, 1)
    path_str = parts[0]
    instruction = parts[1] if len(parts) > 1 else (
        "Review this code for issues, bugs, and potential improvements. "
        "Be specific and actionable."
    )

    path = Path(path_str)
    if not path.exists():
        # Try relative to cwd
        path = Path.cwd() / path_str
    if not path.exists():
        console.print(f"[yellow]Path not found: {path_str}[/yellow]")
        return

    # Collect files
    files_content = {}
    if path.is_file():
        try:
            files_content[str(path)] = path.read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[red]Error reading {path}: {e}[/red]")
            return
    elif path.is_dir():
        # Common code extensions
        extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".yaml", ".yml",
                      ".json", ".toml", ".cfg", ".rs", ".go", ".java", ".cs",
                      ".c", ".cpp", ".h", ".hpp", ".rb", ".sh", ".bat"}
        code_files = sorted(
            f for f in path.rglob("*")
            if f.is_file()
            and f.suffix in extensions
            and "__pycache__" not in f.parts
            and ".git" not in f.parts
            and "node_modules" not in f.parts
        )
        if not code_files:
            console.print(f"[yellow]No code files found in {path_str}[/yellow]")
            return
        console.print(f"[dim]Found {len(code_files)} code files in {path_str}[/dim]")
        for f in code_files:
            try:
                rel = f.relative_to(Path.cwd()) if f.is_relative_to(Path.cwd()) else f
                files_content[str(rel)] = f.read_text(encoding="utf-8")
            except Exception as e:
                console.print(f"[dim]Skipping {f.name}: {e}[/dim]")

    if not files_content:
        console.print("[yellow]No files could be read.[/yellow]")
        return

    # Build prompt
    code_sections = []
    total_chars = 0
    for filepath, content in files_content.items():
        code_sections.append(f"=== {filepath} ===\n{content}")
        total_chars += len(content)

    all_code = "\n\n".join(code_sections)

    prompt = f"**Instruction**: {instruction}\n\n**Code to review**:\n\n{all_code}"

    system_prompt = (
        "You are a code review assistant. Analyze the provided code carefully. "
        "Be specific — reference file names and line numbers. "
        "Focus on: bugs, logic errors, security issues, performance problems, and code quality. "
        "Suggest concrete fixes when possible. Be concise but thorough."
    )

    # Use specified model or default coding model (qwen3-coder:480b-cloud)
    if model_override:
        model = model_override
        client = await agent.router.get_client(TaskType.CODING)
    else:
        model = agent.router.get_model(TaskType.CODING)
        # Skip straight to fallback if primary model is known to be down
        if agent.router.is_model_failed(model):
            fallback = agent.router.get_fallback_model(TaskType.CODING)
            if fallback:
                console.print(f"[yellow]Model {model} is down, using {fallback}[/yellow]")
                model = fallback
        client = await agent.router.get_client(TaskType.CODING)

    if not client:
        # Try fallback model (coding_fallback from config)
        fallback = agent.router.get_fallback_model(TaskType.CODING)
        if fallback:
            console.print(f"[yellow]Cloud unavailable, falling back to {fallback}[/yellow]")
            model = fallback
            client = await agent.router.get_client(TaskType.CODING)
        if not client:
            console.print("[red]No LLM endpoint available.[/red]")
            return

    # Get context window and provider for the endpoint
    ctx_tokens = None
    endpoint_is_ollama = False
    if agent.endpoint_manager:
        ctx_tokens = agent.endpoint_manager.get_context_tokens_for_role(TaskType.CODING)
        ep = await agent.endpoint_manager.get_endpoint_for_role(TaskType.CODING)
        if ep:
            endpoint_is_ollama = ep.provider == "ollama"
    stream_kwargs = {}
    if ctx_tokens:
        stream_kwargs["options"] = {"num_ctx": ctx_tokens}

    console.print(
        f"[cyan]Sending {len(files_content)} file(s) ({total_chars:,} chars) to {model}...[/cyan]"
    )

    # Stream response with thinking spinner
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    thinking_status = console.status("[dim]Thinking...[/dim]", spinner="dots")
    thinking_active = True
    thinking_status.start()
    full_response = []
    code_cancelled = False

    # Gate helper — acquire OllamaGate for local Ollama streaming
    _gate_ctx = None
    if endpoint_is_ollama:
        from blipshell.llm.ollama_gate import get_gate
        _code_gate = get_gate()

    async def _stream_code():
        nonlocal thinking_active
        # Acquire gate for local Ollama to avoid concurrent access
        if endpoint_is_ollama:
            await _code_gate.async_acquire(_code_gate.INTERACTIVE)
        try:
            async for chunk in client.chat_stream(messages=messages, model=model, **stream_kwargs):
                msg = getattr(chunk, "message", None)
                if msg:
                    content = getattr(msg, "content", "")
                elif isinstance(chunk, dict):
                    content = chunk.get("message", {}).get("content", "")
                else:
                    content = ""

                if content:
                    if thinking_active:
                        thinking_status.stop()
                        thinking_active = False
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    full_response.append(content)
        finally:
            if endpoint_is_ollama:
                _code_gate.release()

    # Imported here, not at module scope: these two live in cli.py's terminal
    # plumbing (Windows msvcrt / VT handling), and cli.py imports this module —
    # a top-level import would be circular. That plumbing was deliberately left
    # in place; only presentation moved.
    from blipshell.ui.cli import _drain_keyboard, _poll_for_escape

    code_task = asyncio.create_task(_stream_code())
    esc_task = asyncio.create_task(_poll_for_escape())

    try:
        done, pending = await asyncio.wait(
            {code_task, esc_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        if code_task in done and not code_task.cancelled():
            try:
                code_task.result()  # re-raise any exception
            except Exception as e:
                console.print(f"\n[red]Code review failed: {e}[/red]")
                return
        else:
            code_cancelled = True
    except Exception as e:
        console.print(f"\n[red]Code review failed: {e}[/red]")
        return
    finally:
        if thinking_active:
            thinking_status.stop()
        _drain_keyboard()

    if code_cancelled:
        console.print("\n[dim][Cancelled][/dim]")
    else:
        console.print()  # newline after streaming

    # Inject result into session context so the main LLM knows about it
    if agent.session_manager and full_response:
        result_text = "".join(full_response)
        file_names = ", ".join(files_content.keys())
        suffix = " [cancelled]" if code_cancelled else ""
        context_msg = (
            f"[Code review completed] The user ran /code on {file_names}.\n"
            f"Instruction: {instruction}\n"
            f"Result:\n{result_text[:2000]}{suffix}"
        )
        agent.session_manager.add_message(MessageRole.SYSTEM, context_msg)


async def _submit_offload(agent: Agent, message: str):
    """Submit a task to run on a remote endpoint in the background.

    Detects file paths in the message and injects their contents into the prompt.
    """
    from pathlib import Path

    if not agent.background_manager:
        console.print("[yellow]Background task manager not initialized.[/yellow]")
        return

    if not agent.endpoint_manager:
        console.print("[yellow]Endpoint manager not initialized.[/yellow]")
        return

    # Find a remote endpoint
    remote_name = agent.endpoint_manager.get_first_remote_name()
    if not remote_name:
        console.print(
            "[yellow]No remote endpoints available.[/yellow]\n"
            "[dim]Check /status to see endpoint health.[/dim]"
        )
        return

    # Detect file paths in the message and inject their contents
    prompt = message
    words = message.split()
    files_injected = []
    for word in words:
        p = Path(word)
        if not p.exists():
            p = Path.cwd() / word
        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8")
                prompt += f"\n\n=== File: {word} ===\n{content}"
                files_injected.append(word)
            except Exception:
                pass

    if files_injected:
        console.print(f"[dim]Attached {len(files_injected)} file(s): {', '.join(files_injected)}[/dim]")

    session_id = agent.session_manager.session_id if agent.session_manager else None

    # Truncate title for display
    title = message[:80] + ("..." if len(message) > 80 else "")

    task_id = await agent.background_manager.submit_task(
        title=title,
        task_type="custom",
        prompt=prompt,
        session_id=session_id,
        target_endpoint=remote_name,
    )

    console.print(
        f"[cyan]Task #{task_id} offloaded to {remote_name}[/cyan]\n"
        f"[dim]Check progress: /tasks | View result: /task {task_id}[/dim]"
    )


async def _check_completed_tasks(agent: Agent):
    """Check for background tasks that completed, show results, and inject into LLM context."""
    if not agent.background_manager:
        return

    completed_ids = agent.background_manager.pop_completed()
    for task_id in completed_ids:
        task = await agent.background_manager.get_status(task_id)
        if not task:
            continue

        status_label = task.status.value
        if task.result:
            # Show result to user
            preview = task.result[:500]
            console.print(
                f"\n[bold green]Background task #{task_id} finished:[/bold green] "
                f"{task.title}"
            )
            console.print(Panel(preview, border_style="green", title=f"Task #{task_id} Result"))
            if len(task.result) > 500:
                console.print(f"[dim]Result truncated. Full result: /task {task_id}[/dim]")

            # Inject into LLM context so it knows the result
            if agent.session_manager:
                context_msg = (
                    f"[Background task completed] The user previously offloaded this task: "
                    f"\"{task.title}\"\n\nResult:\n{task.result[:2000]}"
                )
                agent.session_manager.add_message(MessageRole.SYSTEM, context_msg)
        elif task.error_message:
            console.print(
                f"\n[bold red]Background task #{task_id} failed:[/bold red] "
                f"{task.title}\n[red]{task.error_message}[/red]"
            )
        else:
            console.print(
                f"\n[bold green]Background task #{task_id} finished![/bold green] "
                f"[dim]View with /task {task_id}[/dim]"
            )


def _print_status(agent: Agent):
    """Print agent status."""
    status = agent.get_status()

    table = Table(title="Agent Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Session ID", str(status["session_id"]))
    table.add_row("Project", status["project"] or "None")
    table.add_row("Messages", str(status["message_count"]))
    table.add_row("Planner", "[green]Enabled[/green]" if status.get("planner_enabled") else "[dim]Disabled[/dim]")
    table.add_row("Workflows", str(status.get("workflows_loaded", 0)))

    # Show active background tasks count
    bg_running = len(agent.background_manager._running_tasks) if agent.background_manager else 0
    if bg_running:
        table.add_row("Background Tasks", f"[yellow]{bg_running} running[/yellow]")

    console.print(table)

    # Endpoint status
    if status["endpoints"]:
        ep_table = Table(title="Endpoints")
        ep_table.add_column("Name", style="cyan")
        ep_table.add_column("URL", style="dim")
        ep_table.add_column("Status")
        ep_table.add_column("Roles")
        ep_table.add_column("Load", justify="right")
        ep_table.add_column("Success", justify="right", style="green")
        ep_table.add_column("Failures", justify="right")

        for ep in status["endpoints"]:
            if ep["enabled"]:
                status_str = "[green]Online[/green]"
            elif ep["failure_count"] > 0:
                status_str = "[red]Down[/red]"
            else:
                status_str = "[dim]Disabled[/dim]"
            fail_str = f"[red]{ep['failure_count']}[/red]" if ep["failure_count"] else "0"
            ep_table.add_row(
                ep["name"],
                ep["url"],
                status_str,
                ", ".join(ep["roles"]),
                f"{ep['active_requests']}/{ep['max_concurrent']}",
                str(ep["success_count"]),
                fail_str,
            )
        console.print(ep_table)

    # Routing summary — show which PC handles what
    if agent.endpoint_manager:
        routing = agent.endpoint_manager.get_routing_summary()
        if routing:
            rt_table = Table(title="Routing")
            rt_table.add_column("Task Type", style="cyan")
            rt_table.add_column("Endpoint")
            for role, ep_name in sorted(routing.items()):
                ep_style = "[green]" if ep_name != "local" else ""
                ep_end = "[/green]" if ep_name != "local" else ""
                rt_table.add_row(role, f"{ep_style}{ep_name}{ep_end}")
            console.print(rt_table)


def _print_memory_usage(agent: Agent):
    """Print memory pool usage."""
    if not agent.memory_manager:
        console.print("[yellow]Memory manager not initialized.[/yellow]")
        return

    usage = agent.memory_manager.get_usage()
    table = Table(title="Memory Pools")
    table.add_column("Pool", style="cyan")
    table.add_column("Used", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Items", justify="right")
    table.add_column("Usage", justify="right")

    for name, stats in usage.items():
        pct = (stats["used"] / stats["max"] * 100) if stats["max"] > 0 else 0
        color = "green" if pct < 70 else "yellow" if pct < 90 else "red"
        table.add_row(
            name,
            str(stats["used"]),
            str(stats["max"]),
            str(stats["items"]),
            f"[{color}]{pct:.0f}%[/{color}]",
        )

    console.print(table)


async def _print_health(agent: Agent, config, quick: bool = False):
    """Run database audit and display results inline."""
    from scripts.audit_db import run_audit, severity_color

    with console.status("[dim]Running health checks...[/dim]", spinner="dots"):
        result = run_audit(
            db_path=config.database.path,
            skip_vectors=quick,
            skip_endpoints=False,
        )

    # Display as compact table
    table = Table(title="Health Check", show_lines=False, expand=False)
    table.add_column("Category", style="bold", no_wrap=True)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Message")

    for f in result.findings:
        sev = f["severity"]
        table.add_row(
            f["category"],
            f["check"],
            f"[{severity_color(sev)}]{sev.upper()}[/{severity_color(sev)}]",
            f["message"],
        )

    console.print(table)

    # Summary line
    counts = {}
    for f in result.findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    parts = []
    for sev in ["error", "warn", "info", "ok"]:
        if sev in counts:
            parts.append(f"[{severity_color(sev)}]{counts[sev]} {sev}[/{severity_color(sev)}]")
    console.print(f"\n{', '.join(parts)}")
    if quick:
        console.print("[dim](quick mode — ChromaDB sync skipped, use /health for full)[/dim]")


async def _run_cleanup(agent: Agent):
    """Reprocess failed messages with relaxed timeouts."""
    from rich.status import Status

    with Status("[bold cyan]Running cleanup...", console=console) as status:
        def on_status(msg: str):
            status.update(f"[bold cyan]{msg}")

        result = await agent.night_cleanup(on_status=on_status)

    console.print(
        f"\n[bold green]Cleanup complete:[/bold green] "
        f"{result['processed']}/{result['total']} processed, "
        f"{result['failed']} failed"
    )


async def _handle_cube_command(agent: Agent, args: list[str]):
    """Show connected cubes, inspect one, or disconnect one. Connection is automatic.

    /cube              — list connected cubes + server status
    /cube <cube_id>    — show what BlipShell authored for that cube (its behaviors)
    /cube disconnect <cube_id>
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    if agent.robotics is None:
        console.print("[dim]Robotics not initialized.[/dim]")
        return

    if not agent.config.robotics.enabled:
        console.print("[yellow]Robotics is disabled. Set robotics.enabled: true in "
                      "config.yaml so cubes can connect.[/yellow]")
        return

    registry = agent.robotics.registry

    # /cube <cube_id> — show the profile/behaviors BlipShell authored for it.
    if args and args[0] != "disconnect" and registry.is_connected(args[0]):
        cube_id = args[0]
        profile = agent.robotics.get_profile(cube_id)
        if profile is None:
            console.print(f"[dim]'{cube_id}' connected, but no profile authored "
                          "(LLM authoring may have failed or is disabled).[/dim]")
            return
        console.print(f"\n[bold cyan]{cube_id}[/bold cyan] — what BlipShell decided")
        if profile.semantic_role:
            console.print(f"  [bold]role:[/bold] {profile.semantic_role}")
        if profile.intended_uses:
            console.print(f"  [bold]uses:[/bold] {', '.join(profile.intended_uses)}")
        if profile.usage_guidance:
            console.print(f"  [bold]note:[/bold] {profile.usage_guidance}")
        console.print(f"  [bold]behaviors ({len(profile.behaviors)}):[/bold]")
        for b in profile.behaviors:
            acts = "; ".join(
                f"{a.action}({', '.join(f'{k}={v!r}' for k, v in a.args.items())})"
                for a in b.actions
            )
            console.print(f"    • on [yellow]{b.trigger}[/yellow] → {acts or '(nothing)'}")
        if profile.unresolved_issues:
            console.print(f"  [red]unresolved: {profile.unresolved_issues[0]}[/red]")
        console.print()
        return

    # /cube disconnect <cube_id>
    if args and args[0] == "disconnect":
        if len(args) < 2:
            console.print("[yellow]Usage: /cube disconnect <cube_id>[/yellow]")
            return
        cube_id = args[1]
        ok = await agent.robotics.disconnect(cube_id)
        console.print(f"[green]Disconnected '{cube_id}'.[/green]" if ok
                      else f"[red]No cube '{cube_id}' connected.[/red]")
        return

    # /cube reauthor <cube_id> — force a fresh profile (for tuning).
    if args and args[0] == "reauthor":
        if len(args) < 2:
            console.print("[yellow]Usage: /cube reauthor <cube_id>[/yellow]")
            return
        cube_id = args[1]
        console.print(f"[cyan]Re-authoring profile for '{cube_id}' (live LLM)...[/cyan]")
        profile = await agent.robotics.reauthor(cube_id)
        if profile is None:
            console.print(f"[red]Could not re-author '{cube_id}' (connected? LLM available?).[/red]")
        else:
            console.print(f"[green]Re-authored: {len(profile.behaviors)} behavior(s). "
                          f"Run /cube {cube_id} to see them.[/green]")
        return

    # /cube — status + connected cubes
    server = agent._cube_server
    if server is not None:
        console.print(f"[dim]Cube server listening on "
                      f"{agent.config.robotics.host}:{server.port}[/dim]")
    cubes = registry.list_cubes()
    if not cubes:
        console.print("[dim]No cubes connected. Launch one: "
                      "python -m scripts.cube_window[/dim]")
        return

    table = Table(title="Connected Cubes")
    table.add_column("Cube ID", style="cyan")
    table.add_column("Type")
    table.add_column("Actions", justify="right")
    table.add_column("Behaviors", justify="right")
    for meta in cubes:
        profile = agent.robotics.get_profile(meta.cube_id)
        n_behaviors = len(profile.behaviors) if profile else 0
        table.add_row(meta.cube_id, meta.module_type,
                      str(len(meta.actions)), str(n_behaviors))
    console.print(table)


async def _run_nightly(agent: Agent, job_name: str | None = None):
    """Run nightly maintenance jobs."""
    from rich.status import Status
    from rich.table import Table

    from blipshell.core.nightly import NightlyRunner

    runner = NightlyRunner(
        agent.config, agent.sqlite, agent.vectors,
        agent.router, agent.processor,
    )

    jobs = [job_name] if job_name else None
    label = f"job: {job_name}" if job_name else "all jobs"

    with Status(f"[bold cyan]Running nightly ({label})...", console=console) as status:
        def on_status(msg: str):
            status.update(f"[bold cyan]{msg}")

        result = await runner.run(on_status=on_status, jobs=jobs)

    # Print results table
    table = Table(title="Nightly Run Results")
    table.add_column("Job", style="cyan")
    table.add_column("Status")
    table.add_column("Time", justify="right")
    table.add_column("Details")

    for name, stats in result.get("jobs", {}).items():
        status_str = stats.get("status", "?")
        style = "green" if status_str == "ok" else "red"
        elapsed = f"{stats.get('elapsed_s', 0):.1f}s"

        # Build details string from non-meta keys
        detail_parts = []
        for k, v in stats.items():
            if k not in ("status", "elapsed_s", "error"):
                detail_parts.append(f"{k}={v}")
        details = ", ".join(detail_parts) if detail_parts else ""

        if stats.get("error"):
            details = f"[red]{stats['error']}[/red]"

        table.add_row(name, f"[{style}]{status_str}[/{style}]", elapsed, details)

    console.print()
    console.print(table)
    console.print(f"\n[dim]Total: {result.get('elapsed_s', 0):.0f}s[/dim]")


async def _print_nightly_report(agent: Agent):
    """Display the stored nightly report."""
    import json
    from datetime import datetime, timezone
    from rich.table import Table

    raw = await agent.sqlite.get_metadata("nightly_report")
    if not raw:
        console.print("[dim]No nightly report found. Run /nightly first.[/dim]")
        return

    report = json.loads(raw)

    # Header with timestamp
    ts = report.get("timestamp")
    if ts:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        console.print(f"\n[bold]Last nightly run:[/bold] {dt:%Y-%m-%d %H:%M} UTC ({report.get('elapsed_s', 0):.0f}s)")
    else:
        console.print("\n[bold]Last nightly run:[/bold] unknown time")

    # Errors
    errors = report.get("errors", [])
    if errors:
        console.print(f"\n[bold red]Errors ({len(errors)}):[/bold red]")
        for e in errors:
            console.print(f"  [red]{e}[/red]")

    # Warnings
    warnings = report.get("warnings", [])
    if warnings:
        console.print(f"\n[bold yellow]Warnings ({len(warnings)}):[/bold yellow]")
        for w in warnings:
            console.print(f"  [yellow]{w}[/yellow]")

    if not errors and not warnings:
        console.print("[green]  All clear — no warnings or errors.[/green]")

    # Job summary table
    summary = report.get("summary", {})
    if summary:
        table = Table(title="Job Summary")
        table.add_column("Job", style="cyan")
        table.add_column("Status")
        table.add_column("Time", justify="right")
        table.add_column("Details")

        for job, data in summary.items():
            status = data.get("status", "?")
            style = "green" if status == "ok" else "red"
            elapsed = f"{data.get('elapsed_s', 0):.1f}s"
            detail_parts = [
                f"{k}={v}" for k, v in data.items()
                if k not in ("status", "elapsed_s", "error")
            ]
            details = ", ".join(detail_parts) if detail_parts else ""
            if data.get("error"):
                details = f"[red]{data['error']}[/red]"
            table.add_row(job, f"[{style}]{status}[/{style}]", elapsed, details)

        console.print()
        console.print(table)
    console.print()


async def _print_flow(agent: Agent, turn: int | None = None):
    """Print conversation flow events for observability."""
    if not agent.sqlite or not agent.session_manager:
        console.print("[yellow]No active session.[/yellow]")
        return

    session_id = agent.session_manager.session_id

    if turn is not None:
        # Detailed view for a specific turn
        events = await agent.sqlite.get_turn_events_for_turn(session_id, turn)
        if not events:
            console.print(f"[yellow]No events for turn {turn}.[/yellow]")
            return

        console.print(f"\n[bold]Turn {turn} — Detailed Flow[/bold]")
        for evt in events:
            data = evt["data"]
            etype = evt["event_type"]
            ts = evt["timestamp"]

            if etype == "turn_start":
                console.print(f"\n  [cyan]turn_start[/cyan] ({ts})")
                console.print(f"    Route: {data.get('route', '?')}")
                console.print(f"    Query length: {data.get('query_length', '?')} chars")

            elif etype == "search_complete":
                console.print(f"\n  [cyan]search_complete[/cyan]")
                console.print(f"    ChromaDB hits: {data.get('chroma_hits', '?')}")
                console.print(f"    FTS5 hits: {data.get('fts_hits', '?')}")
                console.print(f"    Entity hits: {data.get('entity_hits', '?')}")
                entity_names = data.get("entity_names", [])
                if entity_names:
                    console.print(f"    Entity names matched: {', '.join(entity_names[:10])}")
                    connected = data.get("connected_entities", 0)
                    if connected:
                        console.print(f"    Connected entities: {connected}")
                # Filtering breakdown
                f_sim = data.get("filtered_by_similarity", 0)
                f_imp = data.get("filtered_by_importance", data.get("filtered_by_rank", 0))
                f_sess = data.get("filtered_by_session", 0)
                if f_sim or f_imp or f_sess:
                    console.print(f"    Filtered: {f_sim} by similarity, {f_imp} by importance, {f_sess} by session")
                console.print(f"    Post-filter: {data.get('post_filter', '?')}")
                console.print(f"    Final returned: {data.get('final_returned', '?')}")
                console.print(f"    Memories used: {data.get('memory_results', '?')}")
                console.print(f"    Lessons used: {data.get('lesson_results', '?')}")
                if data.get("skipped"):
                    console.print(f"    [dim]Skipped: {data['skipped']}[/dim]")

            elif etype == "context_built":
                console.print(f"\n  [cyan]context_built[/cyan]")
                console.print(f"    Query profile: {data.get('query_profile', '?')}")
                console.print(f"    Context limit: {data.get('context_limit', '?'):,} tokens")
                console.print(f"    Available: {data.get('available_tokens', '?'):,} tokens")
                console.print(f"    Total items: {data.get('total_context_items', '?')}")
                pool_budgets = data.get("pool_budgets", {})
                pool_usage = data.get("pool_usage", {})
                all_pools = sorted(set(list(pool_budgets.keys()) + list(pool_usage.keys())))
                if all_pools:
                    console.print("    [bold]Pool breakdown:[/bold]")
                    for pool in all_pools:
                        budget = pool_budgets.get(pool, "?")
                        usage = pool_usage.get(pool, {})
                        items = usage.get("items", 0)
                        tokens = usage.get("tokens", 0)
                        budget_str = f"{budget:,}" if isinstance(budget, int) else str(budget)
                        console.print(f"      {pool}: {items} items, {tokens} tokens (budget: {budget_str})")

            elif etype == "llm_complete":
                console.print(f"\n  [cyan]llm_complete[/cyan]")
                console.print(f"    Endpoint: {data.get('endpoint', '?')}")
                console.print(f"    Model: {data.get('model', '?')}")
                console.print(f"    Fallback: {data.get('fallback', False)}")
                tools = data.get("tool_calls", [])
                if tools:
                    console.print(f"    Tools: {', '.join(tools)}")
                console.print(f"    Response: {data.get('response_length', '?')} chars")

            else:
                console.print(f"\n  [cyan]{etype}[/cyan]")
                for k, v in data.items():
                    console.print(f"    {k}: {v}")
    else:
        # Summary view — last 5 turns
        events = await agent.sqlite.get_turn_events(session_id, limit=100)
        if not events:
            console.print("[dim]No flow events yet. Send a message first.[/dim]")
            return

        # Group events by turn
        turns: dict[int, dict] = {}
        for evt in events:
            tn = evt["turn_number"]
            if tn not in turns:
                turns[tn] = {}
            turns[tn][evt["event_type"]] = evt["data"]

        # Show last 5 turns
        table = Table(title="Conversation Flow (recent turns)")
        table.add_column("Turn", style="cyan", justify="right")
        table.add_column("Route")
        table.add_column("Profile")
        table.add_column("Search", justify="right")
        table.add_column("Sources")
        table.add_column("Context", justify="right")
        table.add_column("Model")
        table.add_column("Endpoint")
        table.add_column("Tools")
        table.add_column("Resp", justify="right")

        recent = sorted(turns.items())[-5:]
        for tn, evts in recent:
            start = evts.get("turn_start", {})
            search = evts.get("search_complete", {})
            ctx = evts.get("context_built", {})
            llm = evts.get("llm_complete", {})

            search_str = f"{search.get('final_returned', '?')}m/{search.get('lesson_results', '?')}l"
            # Search source breakdown
            if search.get("skipped"):
                sources_str = f"[dim]{search['skipped']}[/dim]"
            else:
                chroma = search.get("chroma_hits", 0)
                fts = search.get("fts_hits", 0)
                entity = search.get("entity_hits", 0)
                sources_str = f"c:{chroma} f:{fts} e:{entity}"
            ctx_str = str(ctx.get("total_context_items", "?"))
            tools = llm.get("tool_calls", [])
            tools_str = ", ".join(tools) if tools else "-"

            table.add_row(
                str(tn),
                start.get("route", "?"),
                ctx.get("query_profile", "?"),
                search_str,
                sources_str,
                ctx_str,
                llm.get("model", "?"),
                llm.get("endpoint", "?"),
                tools_str,
                str(llm.get("response_length", "?")),
            )

        console.print(table)
        console.print("[dim]Use /flow <turn_number> for details[/dim]")


async def _print_active_plan(agent: Agent):
    """Print the current active plan and step statuses."""
    if not agent.sqlite or not agent.session_manager:
        console.print("[yellow]No active session.[/yellow]")
        return

    plan = await agent.sqlite.get_active_plan(agent.session_manager.session_id)
    if not plan:
        console.print("[dim]No active plan for this session.[/dim]")
        return

    _render_plan(plan)


async def _print_plans(agent: Agent):
    """List all plans for the current session."""
    if not agent.sqlite or not agent.session_manager:
        console.print("[yellow]No active session.[/yellow]")
        return

    plans = await agent.sqlite.list_plans(
        session_id=agent.session_manager.session_id, limit=20,
    )
    if not plans:
        console.print("[dim]No plans found for this session.[/dim]")
        return

    table = Table(title="Task Plans")
    table.add_column("ID", style="cyan")
    table.add_column("Request")
    table.add_column("Status")
    table.add_column("Steps", justify="right")
    table.add_column("Created")

    for p in plans:
        status_color = {
            "completed": "green", "running": "yellow",
            "failed": "red", "cancelled": "dim",
        }.get(p.status.value, "white")
        table.add_row(
            str(p.id),
            (p.user_request or "")[:50],
            f"[{status_color}]{p.status.value}[/{status_color}]",
            str(len(p.steps)),
            str(p.created_at)[:19] if p.created_at else "",
        )

    console.print(table)


def _render_plan(plan):
    """Render a single plan with step details."""
    from blipshell.models.task import PlanStatus, StepStatus

    status_color = {
        "completed": "green", "running": "yellow",
        "failed": "red", "cancelled": "dim",
    }.get(plan.status.value, "white")

    console.print(f"\n[bold]Plan #{plan.id}[/bold] [{status_color}]{plan.status.value}[/{status_color}]")
    console.print(f"[dim]{plan.user_request}[/dim]\n")

    table = Table()
    table.add_column("#", style="cyan", width=3)
    table.add_column("Step")
    table.add_column("Status")
    table.add_column("Tool Hint", style="dim")

    for step in plan.steps:
        step_icon = {
            "pending": "[dim]...[/dim]",
            "running": "[yellow]>>>[/yellow]",
            "completed": "[green]OK[/green]",
            "failed": "[red]!![/red]",
            "skipped": "[dim]--[/dim]",
        }.get(step.status.value, "?")
        table.add_row(
            str(step.step_number),
            step.description[:60],
            step_icon,
            step.tool_hint or "",
        )

    console.print(table)

    if plan.result_summary:
        console.print(f"\n[bold]Summary:[/bold] {plan.result_summary[:500]}")


async def _print_background_tasks(agent: Agent):
    """Show background tasks in a Rich table."""
    if not agent.background_manager:
        console.print("[yellow]Background task manager not initialized.[/yellow]")
        return

    tasks = await agent.background_manager.list_all(
        session_id=agent.session_manager.session_id if agent.session_manager else None,
    )
    if not tasks:
        console.print("[dim]No background tasks.[/dim]")
        return

    table = Table(title="Background Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Type", style="dim")
    table.add_column("Status")
    table.add_column("Progress", justify="right")
    table.add_column("Target", style="dim")

    for t in tasks:
        status_color = {
            "completed": "green", "running": "yellow",
            "failed": "red", "pending": "dim", "cancelled": "dim",
        }.get(t.status.value, "white")
        table.add_row(
            str(t.id),
            (t.title or "")[:40],
            t.task_type,
            f"[{status_color}]{t.status.value}[/{status_color}]",
            f"{t.progress_pct:.0%}",
            t.target_endpoint or "local",
        )

    console.print(table)


async def _print_task_detail(agent: Agent, task_id: int):
    """Show full result of a background task."""
    if not agent.background_manager:
        console.print("[yellow]Background task manager not initialized.[/yellow]")
        return

    task = await agent.background_manager.get_status(task_id)
    if not task:
        console.print(f"[yellow]Task #{task_id} not found.[/yellow]")
        return

    console.print(f"\n[bold]Task #{task.id}:[/bold] {task.title}")
    console.print(f"Type: {task.task_type} | Status: {task.status.value} | Progress: {task.progress_pct:.0%}")
    if task.target_endpoint:
        console.print(f"Target: {task.target_endpoint}")
    if task.result:
        console.print(Panel(task.result[:2000], title="Result", border_style="green"))
    if task.error_message:
        console.print(Panel(task.error_message, title="Error", border_style="red"))


async def _handle_workflow_command(agent: Agent, args: list[str]):
    """Handle /workflow subcommands."""
    if not agent.workflow_registry:
        console.print("[yellow]Workflow system not initialized.[/yellow]")
        return

    if not args or args[0].lower() == "list":
        workflows = agent.workflow_registry.list_all()
        if not workflows:
            console.print("[dim]No workflows found. Add .yaml files to workflows/ directory.[/dim]")
            return
        table = Table(title="Available Workflows")
        table.add_column("Name", style="cyan")
        table.add_column("Description")
        table.add_column("Steps", justify="right")
        table.add_column("Parameters")
        for wf in workflows:
            param_names = ", ".join(p["name"] for p in wf.parameters)
            table.add_row(wf.name, wf.description[:60], str(len(wf.steps)), param_names)
        console.print(table)

    elif args[0].lower() == "show" and len(args) > 1:
        wf = agent.workflow_registry.get(args[1])
        if not wf:
            console.print(f"[yellow]Workflow '{args[1]}' not found.[/yellow]")
            return
        console.print(f"\n[bold]{wf.name}[/bold]: {wf.description}")
        console.print("\n[bold]Parameters:[/bold]")
        for p in wf.parameters:
            default = f" (default: {p.get('default', '')})" if p.get("default") else ""
            console.print(f"  {p['name']}: {p.get('description', '')}{default}")
        console.print("\n[bold]Steps:[/bold]")
        for i, step in enumerate(wf.steps, 1):
            hint = f" [{step.tool_hint}]" if step.tool_hint else ""
            cond = f" (if {step.condition})" if step.condition else ""
            console.print(f"  {i}. {step.description}{hint}{cond}")

    elif args[0].lower() == "run" and len(args) > 1:
        wf_name = args[1]
        # Parse param=value pairs
        params = {}
        for arg in args[2:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                params[key] = value

        console.print(f"[cyan]Running workflow '{wf_name}'...[/cyan]")

        def on_token(token: str):
            sys.stdout.write(token)
            sys.stdout.flush()

        try:
            result = await agent.workflow_executor.run_workflow(
                wf_name, params,
                session_id=agent.session_manager.session_id if agent.session_manager else None,
                on_token=on_token,
            )
            console.print(f"\n\n[green]Workflow complete.[/green]")
        except KeyError:
            console.print(f"[yellow]Workflow '{wf_name}' not found.[/yellow]")
        except Exception as e:
            console.print(f"[red]Workflow failed: {e}[/red]")

    else:
        console.print(
            "[dim]Usage: /workflow list | /workflow show <name> | "
            "/workflow run <name> param=value[/dim]"
        )


def _print_changes(agent):
    """Print files modified during this session."""
    changes = agent.file_changes
    if not changes:
        console.print("[dim]No files modified this session.[/dim]")
        return

    table = Table(title="Modified Files")
    table.add_column("Turn", style="cyan", width=5)
    table.add_column("Tool", style="yellow", width=12)
    table.add_column("Path")

    seen = set()
    for change in changes:
        key = (change["turn_number"], change["path"])
        if key not in seen:
            seen.add(key)
            table.add_row(str(change["turn_number"]), change["tool"], change["path"])
    console.print(table)


async def _print_followups(agent: Agent):
    """Print pending follow-up items."""
    items = await agent.sqlite.get_pending_follow_ups(
        project=agent.active_project["name"] if agent.active_project else None,
        limit=20,
    )
    if not items:
        console.print("[dim]No pending follow-ups.[/dim]")
        return

    table = Table(title=f"Pending Follow-ups ({len(items)})")
    table.add_column("#", style="cyan", width=5)
    table.add_column("Content")
    table.add_column("Due", style="yellow", width=14)
    table.add_column("Added", style="dim", width=12)

    for item in items:
        table.add_row(
            str(item["id"]),
            item["content"],
            item.get("due_hint") or "",
            item.get("created_at", "")[:10],
        )
    console.print(table)


async def _print_friction(agent: Agent, show_all: bool = False):
    """Print friction log entries."""
    items = await agent.sqlite.get_friction_entries(
        unreviewed_only=not show_all, limit=30,
    )
    # Filter out NONE sentinel entries
    items = [i for i in items if i["category"] != "NONE"]
    if not items:
        console.print("[dim]No friction entries found.[/dim]")
        return

    # Category styling
    cat_styles = {
        "TOOL_FAILURE": "red", "TOOL_ISSUE": "red",
        "REPEATED_RETRY": "yellow", "WORKFLOW_FRICTION": "yellow",
        "WORKFLOW_ISSUE": "yellow",
        "MISSING_CAPABILITY": "cyan", "MISSING_FEATURE": "cyan",
        "CONTEXT_ISSUE": "magenta", "CONTEXT_PROBLEM": "magenta",
    }

    title = f"Friction Log ({len(items)} {'total' if show_all else 'unreviewed'})"
    table = Table(title=title)
    table.add_column("#", style="dim", width=4)
    table.add_column("Src", style="dim", width=7)
    table.add_column("Category", width=20)
    table.add_column("Description")
    table.add_column("Session", style="dim", width=7)
    table.add_column("Date", style="dim", width=10)

    for item in items:
        cat = item["category"]
        style = cat_styles.get(cat, "white")
        table.add_row(
            str(item["id"]),
            item["source"][:7],
            f"[{style}]{cat}[/{style}]",
            item["description"],
            str(item["session_id"] or ""),
            item.get("created_at", "")[:10],
        )
    console.print(table)

    # Offer to mark as reviewed
    unreviewed_ids = [i["id"] for i in items if not i.get("is_reviewed")]
    if unreviewed_ids and not show_all:
        console.print(
            f"[dim]{len(unreviewed_ids)} unreviewed entries. "
            "They'll auto-clear after next /friction view.[/dim]"
        )
        await agent.sqlite.mark_friction_reviewed(unreviewed_ids)


async def _handle_notes_command(agent: Agent, args: list[str]):
    """Handle /notes commands: list, get, save, delete, clear."""
    if not args:
        # /notes — list all
        notes = await agent.get_session_notes()
        if not notes:
            console.print("[dim]No session notes.[/dim]")
            return
        from blipshell.memory.manager import estimate_tokens
        total_tokens = sum(estimate_tokens(v) for v in notes.values())
        console.print(f"[bold]Session Notes[/bold] ({len(notes)} notes, ~{total_tokens} tokens)")
        for name, content in notes.items():
            preview = content[:200].replace("\n", " ")
            if len(content) > 200:
                preview += "..."
            console.print(f"  [cyan]{name}[/cyan]: {preview}")
    elif args[0] == "get" and len(args) > 1:
        name = args[1]
        notes = await agent.get_session_notes()
        if name in notes:
            console.print(f"[bold cyan]{name}[/bold cyan]")
            console.print(notes[name])
        else:
            available = ", ".join(sorted(notes.keys())) if notes else "none"
            console.print(f"[dim]Note '{name}' not found. Available: {available}[/dim]")
    elif args[0] == "save" and len(args) > 2:
        name = args[1]
        content = " ".join(args[2:])
        result = await agent.save_session_note(name, content)
        console.print(f"[dim]{result}[/dim]")
    elif args[0] == "clear":
        result = await agent.clear_session_notes()
        console.print(f"[dim]{result}[/dim]")
    elif args[0] == "delete" and len(args) > 1:
        name = args[1]
        notes = await agent.get_session_notes()
        if name in notes:
            del agent._session_notes[name]
            await agent.sqlite.save_session_notes(
                agent.session_manager.session_id, agent._session_notes,
            )
            console.print(f"[dim]Note '{name}' deleted.[/dim]")
        else:
            console.print(f"[dim]Note '{name}' not found.[/dim]")
    else:
        console.print(
            "[dim]Usage: /notes, /notes get <name>, /notes save <name> <content>, "
            "/notes delete <name>, /notes clear[/dim]"
        )


async def _handle_compact(agent: Agent, focus: str):
    """Compact older conversation messages to free context space."""
    with console.status("[dim]Compacting conversation...[/dim]", spinner="dots"):
        result = await agent.compact_conversation(focus)
    console.print(f"[dim]{result}[/dim]")


def _print_context(agent: Agent):
    """Print context window usage breakdown."""
    info = agent.get_context_info()

    table = Table(title="Context Window")
    table.add_column("Property", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Context limit", f"{info['context_limit']:,} tokens")
    table.add_row("Overhead reserve", f"{info['overhead_reserve']:,} tokens")
    available = info["context_limit"] - info["overhead_reserve"]
    table.add_row("Available for content", f"{available:,} tokens")
    table.add_row("Messages this session", str(info["message_count"]))
    table.add_row("Session message tokens", f"{info['session_tokens']:,}")
    table.add_row("Turn number", str(info["turn_number"]))
    console.print(table)

    # Pool usage
    pool_usage = info.get("pool_usage", {})
    if pool_usage:
        pool_table = Table(title="Memory Pools (current allocations)")
        pool_table.add_column("Pool", style="cyan")
        pool_table.add_column("Items", justify="right")
        pool_table.add_column("Used", justify="right")
        pool_table.add_column("Max", justify="right")
        pool_table.add_column("Hard Cap", justify="right")
        pool_table.add_column("Usage", justify="right")

        for name, stats in pool_usage.items():
            used = stats.get("used", 0)
            mx = stats.get("max", 0)
            pct = (used / mx * 100) if mx > 0 else 0
            color = "green" if pct < 70 else "yellow" if pct < 90 else "red"
            cap = stats.get("hard_cap")
            cap_str = f"{cap:,}" if cap else "-"
            pool_table.add_row(
                name,
                str(stats.get("items", 0)),
                f"{used:,}",
                f"{mx:,}",
                cap_str,
                f"[{color}]{pct:.0f}%[/{color}]",
            )
        console.print(pool_table)

    # Last context build stats (from most recent _build_messages call)
    last = info.get("last_context_stats")
    if last:
        console.print(f"\n[dim]Last context build: profile={last.get('query_profile', '?')}, "
                       f"items={last.get('total_context_items', '?')}, "
                       f"available={last.get('available_tokens', 0):,} tokens[/dim]")
        pool_budgets = last.get("pool_budgets", {})
        pool_actual = last.get("pool_usage", {})
        if pool_budgets:
            budget_parts = []
            for pool, budget in sorted(pool_budgets.items()):
                actual = pool_actual.get(pool, {}).get("tokens", 0)
                budget_parts.append(f"{pool}: {actual:,}/{budget:,}")
            console.print(f"[dim]  Budgets: {', '.join(budget_parts)}[/dim]")


def _print_tokens(agent: Agent):
    """Print token usage per endpoint for this session, with cost if configured."""
    usage = agent.get_token_usage()
    if not usage:
        console.print("[dim]No token usage recorded this session.[/dim]")
        return

    # Build endpoint cost rates from config
    cost_rates = {}
    for ep in agent.config.endpoints:
        if ep.cost_per_1m_prompt > 0 or ep.cost_per_1m_completion > 0:
            cost_rates[ep.name] = (ep.cost_per_1m_prompt, ep.cost_per_1m_completion)

    has_costs = bool(cost_rates)

    table = Table(title="Token Usage (this session)")
    table.add_column("Endpoint", style="cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Completion Tokens", justify="right")
    table.add_column("Total", justify="right", style="bold")
    if has_costs:
        table.add_column("Cost", justify="right", style="green")

    grand_prompt = 0
    grand_completion = 0
    grand_requests = 0
    grand_cost = 0.0

    for endpoint, stats in sorted(usage.items()):
        prompt = stats.get("prompt_tokens", 0)
        completion = stats.get("completion_tokens", 0)
        total = prompt + completion
        requests = stats.get("requests", 0)
        grand_prompt += prompt
        grand_completion += completion
        grand_requests += requests

        row = [
            endpoint,
            str(requests),
            f"{prompt:,}",
            f"{completion:,}",
            f"{total:,}",
        ]

        if has_costs:
            rates = cost_rates.get(endpoint)
            if rates:
                cost = (prompt / 1_000_000 * rates[0]) + (completion / 1_000_000 * rates[1])
                grand_cost += cost
                row.append(f"${cost:.4f}")
            else:
                row.append("free")

        table.add_row(*row)

    # Totals row
    if len(usage) > 1:
        row = [
            "[bold]Total[/bold]",
            f"[bold]{grand_requests}[/bold]",
            f"[bold]{grand_prompt:,}[/bold]",
            f"[bold]{grand_completion:,}[/bold]",
            f"[bold]{grand_prompt + grand_completion:,}[/bold]",
        ]
        if has_costs:
            row.append(f"[bold]${grand_cost:.4f}[/bold]")
        table.add_row(*row)

    console.print(table)
