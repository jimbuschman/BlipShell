"""Rich CLI interface with Click commands.

Usage:
    blipshell                        # fresh session with memory
    blipshell --continue             # resume last session
    blipshell --session 46           # resume specific session
    blipshell --project blip-robot   # named project context
    blipshell config                 # view/edit config
    blipshell memories search "query"  # search memories
    blipshell sessions               # list sessions
    blipshell web                    # launch web UI
"""

import asyncio
import difflib
import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from blipshell.core.agent import Agent
from blipshell.core.config import ConfigManager
from blipshell.models.session import MessageRole
from blipshell.ui.input import (
    APPROVAL_PROMPT, SIMPLE_PROMPT,
    async_prompt, create_chat_session, create_simple_session, format_chat_prompt,
)

console = Console()

# Session-level auto-approve set: tools the user has approved for the rest of the session
_session_approved_tools: set[str] = set()

# prompt_toolkit session for tool approval / ask_user (no history)
_simple_session = None


def _generate_colored_diff(old_lines: list[str], new_lines: list[str],
                           filename: str, max_lines: int = 50) -> str:
    """Generate a colored unified diff string using ANSI codes."""
    diff = list(difflib.unified_diff(old_lines, new_lines,
                                     fromfile=f"a/{filename}", tofile=f"b/{filename}",
                                     lineterm=""))
    if not diff:
        return ""

    colored = []
    for i, line in enumerate(diff):
        if i >= max_lines:
            colored.append(f"\x1b[2m  ... ({len(diff) - max_lines} more lines)\x1b[0m")
            break
        if line.startswith("---") or line.startswith("+++"):
            colored.append(f"\x1b[1m{line}\x1b[0m")
        elif line.startswith("@@"):
            colored.append(f"\x1b[36m{line}\x1b[0m")
        elif line.startswith("-"):
            colored.append(f"\x1b[31m{line}\x1b[0m")
        elif line.startswith("+"):
            colored.append(f"\x1b[32m{line}\x1b[0m")
        else:
            colored.append(line)
    return "\n".join(colored)


async def _tool_approval_prompt(tool_name: str, arguments: dict) -> bool:
    """Prompt the user before executing a dangerous tool.

    Shows a colored diff for file edit/write operations.
    Returns True to allow, False to deny.
    """
    # If user already approved this tool for the session, skip the prompt
    if tool_name in _session_approved_tools:
        return True

    # Build a readable summary of what the tool wants to do
    arg_summary = ""
    diff_output = ""

    if tool_name == "run_command":
        arg_summary = arguments.get("command", "")
    elif tool_name == "edit_file":
        arg_summary = arguments.get("path", "")
        old_text = arguments.get("old_text", "")
        new_text = arguments.get("new_text", "")
        if old_text or new_text:
            diff_output = _generate_colored_diff(
                old_text.splitlines(), new_text.splitlines(), arg_summary,
            )
    elif tool_name == "write_file":
        arg_summary = arguments.get("path", "")
        new_content = arguments.get("content", "")
        resolved = Path(arg_summary) if arg_summary else None
        if resolved and resolved.is_file():
            try:
                old_content = resolved.read_text(encoding="utf-8", errors="replace")
                diff_output = _generate_colored_diff(
                    old_content.splitlines(), new_content.splitlines(), arg_summary,
                )
            except Exception:
                pass
        elif new_content:
            line_count = new_content.count("\n") + (1 if new_content else 0)
            diff_output = f"\x1b[32m  Creating new file ({line_count} lines)\x1b[0m"
    elif tool_name == "git_add":
        arg_summary = arguments.get("paths", "")
    elif tool_name == "git_commit":
        arg_summary = arguments.get("message", "")[:80]
    elif tool_name == "create_project":
        arg_summary = f"{arguments.get('name', '')} at {arguments.get('path', '')}"
    else:
        for v in arguments.values():
            arg_summary = str(v)[:80]
            break

    console.print(
        f"\n\x1b[33m[Approval required]\x1b[0m "
        f"\x1b[1m{tool_name}\x1b[0m: {arg_summary}"
    )
    if diff_output:
        console.print(diff_output)

    try:
        choice = (await async_prompt(_simple_session, APPROVAL_PROMPT)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    if choice in ("a", "allow", "y", "yes"):
        return True
    elif choice in ("s", "session"):
        _session_approved_tools.add(tool_name)
        console.print(f"[dim]{tool_name} auto-approved for this session[/dim]")
        return True
    else:
        return False


async def _ask_user_input(question: str) -> str:
    """Prompt the user with a question from the LLM during execution."""
    console.print(f"\n[bold yellow][LLM Question][/bold yellow] {question}")
    try:
        answer = (await async_prompt(_simple_session, SIMPLE_PROMPT)).strip()
        return answer if answer else "No answer provided."
    except (EOFError, KeyboardInterrupt):
        return "User cancelled. Proceed with your best judgment."


async def _poll_for_escape():
    """Poll for Esc keypress. Returns when Esc is detected.

    Uses msvcrt on Windows. On other platforms, blocks forever (no-op).
    """
    try:
        import msvcrt
    except ImportError:
        await asyncio.Event().wait()
        return

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':
                return
            # Discard other keypresses so they don't bleed into next input
        await asyncio.sleep(0.05)


def _drain_keyboard():
    """Drain any buffered keypresses to prevent bleed-through."""
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        pass


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


@click.group(invoke_without_command=True)
@click.option("--continue", "resume_last", is_flag=True, help="Resume last session")
@click.option("--session", "session_id", type=int, help="Resume specific session ID")
@click.option("--project", type=str, help="Named project context")
@click.option("--config-path", type=click.Path(), help="Path to config.yaml")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.pass_context
def main(ctx, resume_last, session_id, project, config_path, verbose):
    """BlipShell - Local LLM personal assistant with persistent memory."""
    setup_logging(verbose)

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path

    if ctx.invoked_subcommand is None:
        # Default: start chat — catch KeyboardInterrupt so Click
        # doesn't print "Aborted!" and skip our cleanup.
        try:
            asyncio.run(chat_loop(
                config_path=config_path,
                resume_last=resume_last,
                session_id=session_id,
                project=project,
            ))
        except KeyboardInterrupt:
            pass  # cleanup already handled in chat_loop's finally block


async def chat_loop(
    config_path: str | None = None,
    resume_last: bool = False,
    session_id: int | None = None,
    project: str | None = None,
):
    """Main interactive chat loop."""
    import signal

    # Custom Ctrl+C handler: first press sets _exit_requested to break the
    # loop cleanly; during cleanup it's suppressed; second press force-quits.
    _exit_requested = False
    _in_cleanup = False

    def _sigint_handler(sig, frame):
        nonlocal _exit_requested
        if _in_cleanup:
            # During cleanup, second Ctrl+C force-quits
            raise KeyboardInterrupt
        _exit_requested = True

    signal.signal(signal.SIGINT, _sigint_handler)

    # Load config
    config_manager = ConfigManager(config_path)
    config = config_manager.load()

    # Create agent with startup progress indicator
    agent = Agent(config, config_manager)
    with console.status("[dim]Starting up...[/dim]", spinner="dots") as status:
        def _on_status(msg: str):
            status.update(f"[dim]{msg}[/dim]")
        await agent.initialize(on_status=_on_status)

    # Determine session to start/resume
    resume_id = session_id
    if resume_last and not session_id:
        latest = await agent.sqlite.get_latest_session()
        if latest:
            resume_id = latest.id
            console.print(f"[dim]Resuming session #{latest.id}: {latest.title}[/dim]")

    # Set up tool approval callback (prompts user before dangerous tool calls)
    if not config.agent.auto_approve_tools and config.agent.tools_requiring_approval:
        agent.tool_registry.set_approval_callback(
            callback=_tool_approval_prompt,
            tools_requiring_approval=set(config.agent.tools_requiring_approval),
        )

    # Wire ask_user callback so the LLM can ask questions during execution
    agent.set_ask_user_callback(_ask_user_input)

    # Create prompt_toolkit sessions for input (history, bracketed paste)
    global _simple_session
    chat_session = create_chat_session()
    _simple_session = create_simple_session()

    sid = await agent.start_session(project=project, resume_session_id=resume_id)

    # Auto-activate project if specified via --project flag
    if project:
        try:
            with console.status("[dim]Loading project...[/dim]", spinner="dots"):
                await agent.activate_project(project)
        except KeyError:
            console.print(f"[yellow]Project '{project}' not found in DB. Use /project new to create it.[/yellow]")

    # Header
    proj_display = agent.active_project["name"] if agent.active_project else None
    console.print(Panel.fit(
        f"[bold cyan]BlipShell[/bold cyan] v0.1.0\n"
        f"Session #{sid}"
        + (f" | Project: [bold]{proj_display}[/bold]" if proj_display else "")
        + f"\nType [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit, [bold]Esc[/bold] to cancel"
        + f"\nThinking: [bold]{'ON' if agent.think_enabled else 'OFF'}[/bold]",
        border_style="cyan",
    ))

    try:
        while True:
            if _exit_requested:
                break

            # Notify about background tasks that finished
            await _check_completed_tasks(agent)

            try:
                prompt = format_chat_prompt(
                    agent.active_project["name"] if agent.active_project else None
                )
                user_input = (await async_prompt(chat_session, prompt)).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input[1:].lower().split()
                cmd_args = user_input[1:].split()[1:]  # preserve original case for args
                if cmd[0] in ("quit", "exit", "q"):
                    break
                elif cmd[0] == "status":
                    _print_status(agent)
                    continue
                elif cmd[0] == "memory":
                    _print_memory_usage(agent)
                    continue
                elif cmd[0] == "save":
                    await agent.session_manager.dump_to_memory()
                    console.print("[dim]Session dumped to memory.[/dim]")
                    continue
                elif cmd[0] == "plan":
                    await _print_active_plan(agent)
                    continue
                elif cmd[0] == "plans":
                    await _print_plans(agent)
                    continue
                elif cmd[0] == "tasks":
                    await _print_background_tasks(agent)
                    continue
                elif cmd[0] == "task" and len(cmd) > 1:
                    try:
                        await _print_task_detail(agent, int(cmd[1]))
                    except ValueError:
                        console.print("[yellow]Usage: /task <id> (e.g. /task 1)[/yellow]")
                    continue
                elif cmd[0] == "workflow":
                    await _handle_workflow_command(agent, cmd_args)
                    continue
                elif cmd[0] == "core":
                    if len(cmd) >= 3 and cmd[1] == "delete":
                        await _delete_core_item(agent, cmd[2:])
                    else:
                        await _print_core(agent)
                    continue
                elif cmd[0] == "feedback":
                    if len(cmd) < 2:
                        console.print("[yellow]Usage: /feedback <your feedback>[/yellow]")
                        console.print("[dim]Example: /feedback too verbose, keep answers shorter[/dim]")
                    else:
                        feedback_text = user_input[len("/feedback "):]
                        await _save_feedback(agent, feedback_text)
                    continue
                elif cmd[0] == "think":
                    if len(cmd) > 1 and cmd[1] in ("on", "off"):
                        agent.think_enabled = cmd[1] == "on"
                    else:
                        agent.think_enabled = not agent.think_enabled
                    state = "[green]ON[/green]" if agent.think_enabled else "[yellow]OFF[/yellow]"
                    console.print(f"[dim]Thinking mode: {state}[/dim]")
                    continue
                elif cmd[0] == "reflect":
                    if len(cmd) > 1 and cmd[1] in ("on", "off"):
                        agent.reflect_enabled = cmd[1] == "on"
                    else:
                        agent.reflect_enabled = not agent.reflect_enabled
                    state = "[green]ON[/green]" if agent.reflect_enabled else "[yellow]OFF[/yellow]"
                    console.print(f"[dim]Self-reflection: {state}[/dim]")
                    continue
                elif cmd[0] == "approve":
                    if len(cmd) > 1 and cmd[1] == "all":
                        # Auto-approve everything for this session
                        for t in config.agent.tools_requiring_approval:
                            _session_approved_tools.add(t)
                        console.print("[dim]All tools auto-approved for this session[/dim]")
                    elif len(cmd) > 1 and cmd[1] == "reset":
                        _session_approved_tools.clear()
                        console.print("[dim]Tool approvals reset — will prompt again[/dim]")
                    else:
                        approved = ", ".join(sorted(_session_approved_tools)) if _session_approved_tools else "none"
                        requiring = ", ".join(config.agent.tools_requiring_approval)
                        console.print(f"[dim]Tools requiring approval: {requiring}[/dim]")
                        console.print(f"[dim]Session-approved: {approved}[/dim]")
                        console.print("[dim]  /approve all   — auto-approve all for this session[/dim]")
                        console.print("[dim]  /approve reset — reset all approvals[/dim]")
                    continue
                elif cmd[0] == "code":
                    if len(cmd) < 2:
                        console.print("[yellow]Usage: /code [--model name] <file-or-folder> [instruction][/yellow]")
                        console.print("[dim]Examples:[/dim]")
                        console.print("[dim]  /code blipshell/core/agent.py[/dim]")
                        console.print("[dim]  /code blipshell/core/ find potential bugs[/dim]")
                        console.print("[dim]  /code --model gemma3:4b tests/benchmark_agent_buggy.py[/dim]")
                    else:
                        code_args = user_input[len("/code "):]
                        await _handle_code_command(agent, code_args)
                    continue
                elif cmd[0] == "offload":
                    if len(cmd) < 2:
                        console.print("[yellow]Usage: /offload <task description>[/yellow]")
                        console.print("[dim]Example: /offload review this code for errors: ...[/dim]")
                    else:
                        offload_msg = user_input[len("/offload "):]
                        await _submit_offload(agent, offload_msg)
                    continue
                elif cmd[0] == "health":
                    quick = len(cmd) > 1 and cmd[1] == "quick"
                    await _print_health(agent, config, quick=quick)
                    continue
                elif cmd[0] == "flow":
                    turn = None
                    if len(cmd) > 1:
                        try:
                            turn = int(cmd[1])
                        except ValueError:
                            console.print("[yellow]Usage: /flow [turn_number][/yellow]")
                            continue
                    await _print_flow(agent, turn)
                    continue
                elif cmd[0] == "cleanup":
                    await _run_cleanup(agent)
                    continue
                elif cmd[0] == "nightly":
                    job_name = cmd_args[0] if cmd_args else None
                    await _run_nightly(agent, job_name)
                    continue
                elif cmd[0] == "mcp":
                    _print_mcp_status(agent, cmd_args)
                    continue
                elif cmd[0] == "changes":
                    _print_changes(agent)
                    continue
                elif cmd[0] == "compact":
                    focus = " ".join(cmd_args) if cmd_args else ""
                    await _handle_compact(agent, focus)
                    continue
                elif cmd[0] == "context":
                    _print_context(agent)
                    continue
                elif cmd[0] == "tokens":
                    _print_tokens(agent)
                    continue
                elif cmd[0] == "projects":
                    await _list_projects(agent)
                    continue
                elif cmd[0] == "project":
                    await _handle_project_command(agent, cmd_args)
                    continue
                elif cmd[0] in ("help", "commands"):
                    _print_help()
                    continue
                else:
                    console.print(f"[yellow]Unknown command: /{cmd[0]}[/yellow]")
                    continue

            # Check for force-plan prefix
            force_plan = False
            message = user_input
            if user_input.startswith("!plan "):
                force_plan = True
                message = user_input[6:]

            # Stream response with thinking spinner (Esc to cancel)
            response_parts = []
            thinking_status = console.status("[dim]Thinking...[/dim]", spinner="dots")
            thinking_active = True
            cancelled = False

            def on_token(token: str):
                nonlocal thinking_active
                if thinking_active:
                    thinking_status.stop()
                    thinking_active = False
                response_parts.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()

            console.print()  # blank line before response
            thinking_status.start()

            chat_task = asyncio.create_task(
                agent.chat(message, on_token=on_token, force_plan=force_plan)
            )
            esc_task = asyncio.create_task(_poll_for_escape())

            try:
                done, pending = await asyncio.wait(
                    {chat_task, esc_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

                if chat_task in done and not chat_task.cancelled():
                    try:
                        response = chat_task.result()
                    except Exception as e:
                        response = f"Error: {e}"
                else:
                    cancelled = True
                    response = "".join(response_parts)
            except Exception as e:
                response = f"Error: {e}"
            finally:
                if thinking_active:
                    thinking_status.stop()
                _drain_keyboard()

            if cancelled:
                console.print("\n[dim][Cancelled][/dim]")
                # Save partial response so conversation history stays coherent
                if response_parts:
                    agent.session_manager.add_message(
                        MessageRole.ASSISTANT,
                        "".join(response_parts) + " [cancelled]",
                    )
            elif not response_parts:
                # Response wasn't streamed (e.g., tool calls happened)
                console.print(Markdown(response))
            else:
                console.print()  # newline after streaming
                # For planned execution (!plan), streamed content is tool call
                # progress — show the final result separately so the user sees it.
                # For simple chat, the response was already streamed via on_token.
                if force_plan and response and response.strip():
                    console.print(Panel(
                        Markdown(response),
                        title="Result",
                        border_style="green",
                    ))

            # Show which endpoint handled the request
            ep = agent.last_endpoint_used
            if ep and ep != "local":
                console.print(f"[dim]via {ep}[/dim]")

            console.print()  # spacing

    finally:
        console.print()
        _in_cleanup = True
        try:
            with console.status("[dim]Ending session...[/dim]", spinner="dots") as status:
                def _on_status(msg: str):
                    status.update(f"[dim]{msg}[/dim]")
                await agent.end_session(on_status=_on_status)
        except KeyboardInterrupt:
            console.print("[yellow]Force quit.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Session cleanup error: {e}[/yellow]")
        await agent.force_cleanup()
        console.print("[dim]Session saved. Goodbye![/dim]")


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
            agent.chroma.delete_lesson(item_id)
        except Exception:
            pass
        console.print(f"[green]Lesson #{item_id} deleted.[/green]")
    else:
        cm = await agent.sqlite.get_core_memory(item_id)
        if not cm:
            console.print(f"[yellow]Core memory #{item_id} not found.[/yellow]")
            return
        await agent.sqlite.deactivate_core_memory(item_id)
        try:
            agent.chroma.delete_core_memory(item_id)
        except Exception:
            pass
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
        agent.chroma.add_lesson(lesson_id, lesson.content)
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
            digest_mgr = ProjectDigestManager(agent.sqlite, agent.router)
            with console.status("[dim]Rebuilding project digest...[/dim]", spinner="dots"):
                digest = await digest_mgr.bootstrap_digest(project_name)
            if digest:
                console.print(Panel(digest, title=f"Project Digest — {project_name} (rebuilt)"))
            else:
                console.print("[dim]No sessions with summaries found for this project.[/dim]")
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

    # Get context window for the endpoint
    ctx_tokens = None
    if agent.endpoint_manager:
        ctx_tokens = agent.endpoint_manager.get_context_tokens_for_role(TaskType.CODING)
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

    async def _stream_code():
        nonlocal thinking_active
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
    table.add_row("Queue Pending", str(status["job_queue_pending"]))

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
            chroma_path=config.database.chroma_path,
            skip_chroma=quick,
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


def _print_mcp_status(agent: Agent, args: list[str]):
    """Show MCP server status and tools."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    if not agent.mcp_manager:
        console.print("[dim]No MCP servers configured.[/dim]")
        return

    servers = agent.mcp_manager.get_connected_servers()
    if not servers:
        console.print("[dim]No MCP servers connected.[/dim]")
        return

    # /mcp tools [server] — list tools for a server
    if args and args[0] == "tools":
        server_name = args[1] if len(args) > 1 else servers[0]
        if server_name not in servers:
            console.print(f"[red]Server '{server_name}' not connected.[/red]")
            return
        tool_names = agent.mcp_manager.get_server_tool_names(server_name)
        console.print(f"\n[bold]MCP Server: {server_name}[/bold] ({len(tool_names)} tools)")
        for name in sorted(tool_names):
            console.print(f"  - mcp_{server_name}_{name}")
        console.print()
        return

    # /mcp — list connected servers
    table = Table(title="MCP Servers")
    table.add_column("Server", style="cyan")
    table.add_column("Tools", justify="right")
    table.add_column("Status")

    for name in servers:
        count = agent.mcp_manager.get_server_tool_count(name)
        table.add_row(name, str(count), "[green]connected[/green]")

    console.print(table)


async def _run_nightly(agent: Agent, job_name: str | None = None):
    """Run nightly maintenance jobs."""
    from rich.status import Status
    from rich.table import Table

    from blipshell.core.nightly import NightlyRunner

    runner = NightlyRunner(
        agent.config, agent.sqlite, agent.chroma,
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
                f_rank = data.get("filtered_by_rank", 0)
                f_sess = data.get("filtered_by_session", 0)
                if f_sim or f_rank or f_sess:
                    console.print(f"    Filtered: {f_sim} by similarity, {f_rank} by rank, {f_sess} by session")
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
    """Print token usage per endpoint for this session."""
    usage = agent.get_token_usage()
    if not usage:
        console.print("[dim]No token usage recorded this session.[/dim]")
        return

    table = Table(title="Token Usage (this session)")
    table.add_column("Endpoint", style="cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Completion Tokens", justify="right")
    table.add_column("Total", justify="right", style="bold")

    grand_prompt = 0
    grand_completion = 0
    grand_requests = 0

    for endpoint, stats in sorted(usage.items()):
        prompt = stats.get("prompt_tokens", 0)
        completion = stats.get("completion_tokens", 0)
        total = prompt + completion
        requests = stats.get("requests", 0)
        grand_prompt += prompt
        grand_completion += completion
        grand_requests += requests
        table.add_row(
            endpoint,
            str(requests),
            f"{prompt:,}",
            f"{completion:,}",
            f"{total:,}",
        )

    # Totals row
    if len(usage) > 1:
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{grand_requests}[/bold]",
            f"[bold]{grand_prompt:,}[/bold]",
            f"[bold]{grand_completion:,}[/bold]",
            f"[bold]{grand_prompt + grand_completion:,}[/bold]",
        )

    console.print(table)


def _print_help():
    """Print help for CLI commands."""
    console.print(Panel(
        "[bold]/quit[/bold]                  - Exit BlipShell\n"
        "[bold]/status[/bold]                - Show agent status, endpoints, routing\n"
        "[bold]/memory[/bold]                - Show memory pool usage\n"
        "[bold]/context[/bold]               - Show context window usage breakdown\n"
        "[bold]/tokens[/bold]                - Show token usage per endpoint this session\n"
        "[bold]/compact[/bold] [dim][focus][/dim]        - Compact older messages to free context\n"
        "[bold]/save[/bold]                  - Force save session to memory\n"
        "[bold]/core[/bold]                  - Show core memories and lessons\n"
        "[bold]/think[/bold]                 - Toggle LLM thinking mode on/off\n"
        "[bold]/reflect[/bold]               - Toggle self-reflection on/off\n"
        "[bold]/approve[/bold] [dim]all|reset[/dim]     - Manage tool approval (write/edit/run)\n"
        "[bold]/changes[/bold]               - Show files modified this session\n"
        "[bold]/code <path> [msg][/bold]     - Send code to LLM for review\n"
        "[bold]/feedback <msg>[/bold]        - Save feedback as a lesson\n"
        "[bold]/offload <msg>[/bold]         - Run a task on remote PC in background\n"
        "[bold]/health[/bold] [dim][quick][/dim]          - Database + endpoint health check\n"
        "[bold]/cleanup[/bold]               - Reprocess failed messages (relaxed timeouts)\n"
        "[bold]/nightly[/bold] [dim][job][/dim]          - Run nightly maintenance (tagging, pruning, etc.)\n"
        "[bold]/mcp[/bold] [dim][tools [server]][/dim]  - Show MCP server status and tools\n"
        "[bold]/flow[/bold] [dim][n][/dim]              - Show conversation flow events\n"
        "[bold]/plan[/bold]                  - Show current active plan\n"
        "[bold]/plans[/bold]                 - List all plans for this session\n"
        "[bold]/tasks[/bold]                 - Show background tasks\n"
        "[bold]/task <id>[/bold]             - Show background task detail\n"
        "[bold]/workflow[/bold] [dim]list|show|run[/dim] - Workflow management\n\n"
        "[bold cyan]Project Commands[/bold cyan]\n"
        "[bold]/projects[/bold]              - List all projects\n"
        "[bold]/project <name>[/bold]        - Activate a project\n"
        "[bold]/project new <n> <path>[/bold] - Create project from directory\n"
        "[bold]/project info[/bold]          - Show active project details\n"
        "[bold]/project off[/bold]           - Deactivate current project\n"
        "[bold]/project delete <name>[/bold] - Remove project from DB\n"
        "[bold]/project digest[/bold]        - Show project status digest\n"
        "[bold]/project digest rebuild[/bold] - Regenerate digest from scratch\n\n"
        "[bold]/help[/bold]                  - Show this help\n\n"
        "[dim]Press [bold]Esc[/bold] during a response to cancel the LLM call[/dim]\n"
        "[dim]Prefix with !plan to force planning: !plan <message>[/dim]",
        title="Commands",
        border_style="blue",
    ))


# --- Subcommands ---

@main.command()
@click.pass_context
def config(ctx):
    """View current configuration."""
    config_manager = ConfigManager(ctx.obj.get("config_path"))
    cfg = config_manager.load()

    import yaml
    console.print(Panel(
        yaml.dump(cfg.model_dump(), default_flow_style=False, sort_keys=False),
        title="BlipShell Config",
        border_style="blue",
    ))


@main.group()
def memories():
    """Memory management commands."""
    pass


@memories.command()
@click.argument("query")
@click.option("--limit", default=10, help="Max results")
@click.pass_context
def search(ctx, query, limit):
    """Search memories by semantic similarity."""
    async def _search():
        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()
        agent = Agent(cfg, config_manager)
        await agent.initialize()

        results = await agent.search.search(query=query, n_results=limit)
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        for r in results:
            console.print(Panel(
                f"[bold]Score: {r.boosted_score:.3f}[/bold] | Rank: {r.rank} | Importance: {r.importance:.2f}\n\n"
                f"{r.summary}",
                border_style="green" if r.boosted_score > 0.8 else "yellow",
            ))

    asyncio.run(_search())


@main.command()
@click.option("--limit", default=20, help="Max sessions to show")
@click.option("--project", type=str, help="Filter by project")
@click.pass_context
def sessions(ctx, limit, project):
    """List recent sessions."""
    async def _list():
        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        session_list = await sqlite.list_sessions(limit=limit, project=project)
        await sqlite.close()

        if not session_list:
            console.print("[yellow]No sessions found.[/yellow]")
            return

        table = Table(title="Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Project")
        table.add_column("Messages", justify="right")
        table.add_column("Last Active")

        for s in session_list:
            table.add_row(
                str(s.id),
                (s.title or "Untitled")[:50],
                s.project or "-",
                str(s.message_count),
                str(s.last_active)[:19],
            )

        console.print(table)

    from blipshell.memory.sqlite_store import SQLiteStore
    asyncio.run(_list())


@main.command()
@click.option("--format", "fmt", type=click.Choice(["json", "markdown"]), default="json",
              help="Export format")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
@click.pass_context
def export(ctx, fmt, output):
    """Export all data (sessions, memories, core memories, lessons)."""
    async def _export():
        from blipshell.export import export_all_json, export_all_markdown

        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        if fmt == "markdown":
            data = await export_all_markdown(sqlite)
        else:
            import json
            raw = await export_all_json(sqlite)
            data = json.dumps(raw, indent=2, default=str)

        await sqlite.close()

        if output:
            Path(output).write_text(data, encoding="utf-8")
            console.print(f"[green]Exported to {output}[/green]")
        else:
            console.print(data)

    from pathlib import Path
    from blipshell.memory.sqlite_store import SQLiteStore
    asyncio.run(_export())


@main.command()
@click.pass_context
def web(ctx):
    """Launch the web UI."""
    import uvicorn
    from blipshell.core.config import ConfigManager

    config_manager = ConfigManager(ctx.obj.get("config_path"))
    cfg = config_manager.load()

    console.print(f"[cyan]Starting web UI at http://{cfg.web_ui.host}:{cfg.web_ui.port}[/cyan]")

    uvicorn.run(
        "blipshell.ui.web.app:create_app",
        host=cfg.web_ui.host,
        port=cfg.web_ui.port,
        factory=True,
    )


# --- Headless Test ---

@main.command("test")
@click.argument("task", required=False)
@click.option("--project", "-p", default=None, help="Project to activate")
@click.option("--output", "-o", default=None, help="Write JSON report to file")
@click.option("--canned", is_flag=True, help="Quick test suite (~5 min)")
@click.option("--stress", is_flag=True, help="Full stress suite (~1-2 hours)")
@click.option("--category", "-c", default=None, help="Only run tests in this category (e.g. multi_step, real_world)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress streaming output")
@click.pass_context
def test_cmd(ctx, task, project, output, canned, stress, quiet, category):
    """Run a headless test task and output JSON results."""
    from scripts.test_executor import run_test, run_canned_tests, run_stress_tests

    config_path = ctx.obj.get("config_path")

    if stress:
        asyncio.run(run_stress_tests(
            project=project, config_path=config_path,
            output_path=output, quiet=quiet,
            category=category,
        ))
    elif canned:
        asyncio.run(run_canned_tests(
            project=project, config_path=config_path,
            output_path=output, quiet=quiet,
        ))
    elif task:
        asyncio.run(run_test(
            task=task, project=project, output_path=output,
            config_path=config_path, quiet=quiet,
        ))
    else:
        console.print("[yellow]Provide a task, or use --canned or --stress[/yellow]")


# --- Nightly Jobs ---

@main.command("nightly")
@click.option("--job", default=None, help="Run a specific job only (e.g. centroid_tag, batch_tag)")
@click.option("--quiet", "-q", is_flag=True, help="JSON output only (for scheduled runs)")
@click.pass_context
def nightly_cmd(ctx, job, quiet):
    """Run nightly maintenance jobs (backup, tagging, pruning, etc.)."""
    import json as _json

    async def _run():
        from blipshell.core.nightly import NightlyRunner

        runner = await NightlyRunner.create_from_config(ctx.obj.get("config_path"))
        try:
            jobs = [job] if job else None

            if quiet:
                result = await runner.run(jobs=jobs)
                print(_json.dumps(result, indent=2, default=str))
            else:
                from rich.status import Status
                from rich.table import Table

                label = f"job: {job}" if job else "all jobs"
                with Status(f"[bold cyan]Running nightly ({label})...", console=console) as status:
                    def on_status(msg: str):
                        status.update(f"[bold cyan]{msg}")

                    result = await runner.run(on_status=on_status, jobs=jobs)

                table = Table(title="Nightly Run Results")
                table.add_column("Job", style="cyan")
                table.add_column("Status")
                table.add_column("Time", justify="right")
                table.add_column("Details")

                for name, stats in result.get("jobs", {}).items():
                    status_str = stats.get("status", "?")
                    style = "green" if status_str == "ok" else "red"
                    elapsed = f"{stats.get('elapsed_s', 0):.1f}s"
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
        finally:
            await runner.close()

    asyncio.run(_run())


# --- ChatGPT Import ---

@main.group("import-chatgpt")
def import_chatgpt():
    """Import data from a ChatGPT export."""
    pass


@import_chatgpt.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--max", "max_count", type=int, default=None,
              help="Only import the first N conversations (for testing)")
@click.option("--skip-lessons", is_flag=True, help="Skip lesson extraction (faster)")
@click.pass_context
def conversations(ctx, file, max_count, skip_lessons):
    """Import conversations from a ChatGPT conversations.json export."""
    from rich.progress import Progress

    from blipshell.import_chatgpt import parse_conversations
    from blipshell.import_common import import_conversations
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import get_ollama_url
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore

    async def _import():
        # Parse
        console.print(f"[cyan]Parsing {file}...[/cyan]")
        convs = parse_conversations(file)
        console.print(f"Found [bold]{len(convs)}[/bold] conversations.")

        if max_count:
            convs = convs[:max_count]
            console.print(f"Importing first [bold]{max_count}[/bold].")

        if not convs:
            console.print("[yellow]No conversations to import.[/yellow]")
            return

        # Initialize infrastructure (same pattern as export command)
        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = ChromaStore(
            persist_dir=cfg.database.chroma_path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        # Import with progress bar
        with Progress(console=console) as progress:
            task = progress.add_task("Importing...", total=len(convs))

            def on_progress(idx, total, title, stats):
                label = f"[cyan]{title[:40]}[/cyan]"
                i, s = stats.conversations_imported, stats.conversations_skipped
                if i or s:
                    label += f"  [dim]({i} imported, {s} skipped)[/dim]"
                progress.update(task, completed=idx, description=label)

            stats = await import_conversations(
                sqlite=sqlite,
                chroma=chroma,
                router=router,
                config=cfg.memory,
                conversations=convs,
                on_progress=on_progress,
                skip_lessons=skip_lessons,
            )
            progress.update(task, completed=len(convs))

        await sqlite.close()
        _print_import_summary(stats)

    asyncio.run(_import())


@import_chatgpt.command()
@click.argument("file", type=click.Path(exists=True))
@click.pass_context
def personality(ctx, file):
    """Import a personality/system prompt from a text file."""
    from pathlib import Path

    from blipshell.import_chatgpt import import_personality

    text = Path(file).read_text(encoding="utf-8").strip()
    if not text:
        console.print("[yellow]File is empty, nothing to import.[/yellow]")
        return

    config_manager = ConfigManager(ctx.obj.get("config_path"))
    config_manager.load()

    import_personality(config_manager, text)
    console.print(f"[green]System prompt updated with personality from {file}[/green]")


@import_chatgpt.command("memories")
@click.argument("file", type=click.Path(exists=True))
@click.pass_context
def import_memories_cmd(ctx, file):
    """Import ChatGPT memories as core memories (one per line)."""
    from pathlib import Path

    from blipshell.import_chatgpt import import_memories_as_core
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import get_ollama_url
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore

    text = Path(file).read_text(encoding="utf-8")
    line_count = len([l for l in text.splitlines() if l.strip()])
    if not line_count:
        console.print("[yellow]No memories found in file.[/yellow]")
        return

    console.print(f"Found [bold]{line_count}[/bold] memories in {file}.")

    async def _import():
        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = ChromaStore(
            persist_dir=cfg.database.chroma_path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        count = await import_memories_as_core(
            sqlite=sqlite,
            chroma=chroma,
            router=router,
            config=cfg.memory,
            memories_text=text,
        )

        await sqlite.close()
        console.print(f"[green]Imported {count} core memories.[/green]")

    asyncio.run(_import())


# --- Claude Import ---

@main.group("import-claude")
def import_claude_group():
    """Import data from Claude exports."""
    pass


@import_claude_group.command("conversations")
@click.argument("file", type=click.Path(exists=True))
@click.option("--max", "max_count", type=int, default=None,
              help="Only import the first N conversations (for testing)")
@click.option("--skip-lessons", is_flag=True, help="Skip lesson extraction (faster)")
@click.pass_context
def claude_conversations(ctx, file, max_count, skip_lessons):
    """Import conversations from an official Claude conversations.json export."""
    from rich.progress import Progress

    from blipshell.import_claude import parse_conversations
    from blipshell.import_common import import_conversations
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import get_ollama_url
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore

    async def _import():
        console.print(f"[cyan]Parsing {file}...[/cyan]")
        convs = parse_conversations(file)
        console.print(f"Found [bold]{len(convs)}[/bold] conversations.")

        if max_count:
            convs = convs[:max_count]
            console.print(f"Importing first [bold]{max_count}[/bold].")

        if not convs:
            console.print("[yellow]No conversations to import.[/yellow]")
            return

        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = ChromaStore(
            persist_dir=cfg.database.chroma_path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        with Progress(console=console) as progress:
            task = progress.add_task("Importing...", total=len(convs))

            def on_progress(idx, total, title, stats):
                label = f"[cyan]{title[:40]}[/cyan]"
                i, s = stats.conversations_imported, stats.conversations_skipped
                if i or s:
                    label += f"  [dim]({i} imported, {s} skipped)[/dim]"
                progress.update(task, completed=idx, description=label)

            stats = await import_conversations(
                sqlite=sqlite,
                chroma=chroma,
                router=router,
                config=cfg.memory,
                conversations=convs,
                on_progress=on_progress,
                skip_lessons=skip_lessons,
            )
            progress.update(task, completed=len(convs))

        await sqlite.close()
        _print_import_summary(stats)

    asyncio.run(_import())


@import_claude_group.command("scraped")
@click.argument("file", type=click.Path(exists=True))
@click.option("--max", "max_count", type=int, default=None,
              help="Only import the first N conversations (for testing)")
@click.option("--skip-lessons", is_flag=True, help="Skip lesson extraction (faster)")
@click.pass_context
def claude_scraped(ctx, file, max_count, skip_lessons):
    """Import conversations from a scraped Claude conversations_export.json."""
    from rich.progress import Progress

    from blipshell.import_claude import parse_scraped_conversations
    from blipshell.import_common import import_conversations
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import get_ollama_url
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore

    async def _import():
        console.print(f"[cyan]Parsing {file}...[/cyan]")
        convs = parse_scraped_conversations(file)
        console.print(f"Found [bold]{len(convs)}[/bold] conversations.")

        if max_count:
            convs = convs[:max_count]
            console.print(f"Importing first [bold]{max_count}[/bold].")

        if not convs:
            console.print("[yellow]No conversations to import.[/yellow]")
            return

        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = ChromaStore(
            persist_dir=cfg.database.chroma_path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        with Progress(console=console) as progress:
            task = progress.add_task("Importing...", total=len(convs))

            def on_progress(idx, total, title, stats):
                label = f"[cyan]{title[:40]}[/cyan]"
                i, s = stats.conversations_imported, stats.conversations_skipped
                if i or s:
                    label += f"  [dim]({i} imported, {s} skipped)[/dim]"
                progress.update(task, completed=idx, description=label)

            stats = await import_conversations(
                sqlite=sqlite,
                chroma=chroma,
                router=router,
                config=cfg.memory,
                conversations=convs,
                on_progress=on_progress,
                skip_lessons=skip_lessons,
            )
            progress.update(task, completed=len(convs))

        await sqlite.close()
        _print_import_summary(stats)

    asyncio.run(_import())


# --- DeepSeek Import ---

@main.group("import-deepseek")
def import_deepseek_group():
    """Import data from a DeepSeek export."""
    pass


@import_deepseek_group.command("conversations")
@click.argument("file", type=click.Path(exists=True))
@click.option("--max", "max_count", type=int, default=None,
              help="Only import the first N conversations (for testing)")
@click.option("--skip-lessons", is_flag=True, help="Skip lesson extraction (faster)")
@click.pass_context
def deepseek_conversations(ctx, file, max_count, skip_lessons):
    """Import conversations from a DeepSeek conversations_deepseek.json export."""
    from rich.progress import Progress

    from blipshell.import_common import import_conversations
    from blipshell.import_deepseek import parse_conversations
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import get_ollama_url
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore

    async def _import():
        console.print(f"[cyan]Parsing {file}...[/cyan]")
        convs = parse_conversations(file)
        console.print(f"Found [bold]{len(convs)}[/bold] conversations.")

        if max_count:
            convs = convs[:max_count]
            console.print(f"Importing first [bold]{max_count}[/bold].")

        if not convs:
            console.print("[yellow]No conversations to import.[/yellow]")
            return

        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = ChromaStore(
            persist_dir=cfg.database.chroma_path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        with Progress(console=console) as progress:
            task = progress.add_task("Importing...", total=len(convs))

            def on_progress(idx, total, title, stats):
                label = f"[cyan]{title[:40]}[/cyan]"
                i, s = stats.conversations_imported, stats.conversations_skipped
                if i or s:
                    label += f"  [dim]({i} imported, {s} skipped)[/dim]"
                progress.update(task, completed=idx, description=label)

            stats = await import_conversations(
                sqlite=sqlite,
                chroma=chroma,
                router=router,
                config=cfg.memory,
                conversations=convs,
                on_progress=on_progress,
                skip_lessons=skip_lessons,
            )
            progress.update(task, completed=len(convs))

        await sqlite.close()
        _print_import_summary(stats)

    asyncio.run(_import())


# --- Import summary helper ---

def _print_import_summary(stats):
    """Print the standard import summary table."""
    console.print()
    summary = Table(title="Import Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Count", justify="right")
    summary.add_row("Conversations imported", str(stats.conversations_imported))
    summary.add_row("Conversations skipped (resume)", str(stats.conversations_skipped))
    summary.add_row("Conversations re-imported (incomplete)", str(stats.conversations_reimported))
    summary.add_row("Messages processed", str(stats.messages_processed))
    summary.add_row("Messages skipped (noise)", str(stats.messages_skipped_noise))
    summary.add_row("Lessons extracted", str(stats.lessons_extracted))
    console.print(summary)


# --- Reprocess ---

@main.group()
def reprocess():
    """Reprocess imported data with a better model."""
    pass


@reprocess.command("memories")
@click.option("--model", type=str, default=None, help="Override model for summarization/ranking")
@click.option("--batch-size", default=50, help="Memories per batch")
@click.option("--skip-embed", is_flag=True, help="Skip re-embedding (faster if you only want new scores)")
@click.option("--no-think", is_flag=True, help="Disable model thinking/reasoning (faster for simple tasks)")
@click.pass_context
def reprocess_memories_cmd(ctx, model, batch_size, skip_embed, no_think):
    """Re-summarize, re-rank, re-score, and re-embed all memories."""
    from rich.progress import Progress

    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import get_ollama_url
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.reprocess import reprocess_memories

    async def _run():
        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = ChromaStore(
            persist_dir=cfg.database.chroma_path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        # Override models if --model provided
        original_models = None
        if model:
            original_models = (
                cfg.models.summarization,
                cfg.models.ranking,
            )
            router._models.summarization = model
            router._models.ranking = model
            console.print(f"[cyan]Using model override: {model}[/cyan]")

        console.print(f"[cyan]Reprocessing memories (batch_size={batch_size}, skip_embed={skip_embed})...[/cyan]")

        with Progress(console=console) as progress:
            task = progress.add_task("Reprocessing memories...", total=1)

            def on_progress(done, total):
                progress.update(task, completed=done, total=total,
                                description=f"[cyan]Reprocessing {done}/{total}[/cyan]")

            stats = await reprocess_memories(
                sqlite=sqlite,
                chroma=chroma,
                router=router,
                batch_size=batch_size,
                skip_embed=skip_embed,
                no_think=no_think,
                on_progress=on_progress,
            )

        # Restore original models
        if original_models:
            router._models.summarization = original_models[0]
            router._models.ranking = original_models[1]

        await sqlite.close()

        # Print summary
        summary = Table(title="Reprocess Memories Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Count", justify="right")
        summary.add_row("Total memories", str(stats["total"]))
        summary.add_row("Processed", f"[green]{stats['processed']}[/green]")
        summary.add_row("Skipped (empty)", str(stats["skipped"]))
        summary.add_row("Errors", f"[red]{stats['errors']}[/red]" if stats["errors"] else "0")
        console.print(summary)

    asyncio.run(_run())


@reprocess.command("lessons")
@click.option("--model", type=str, default=None, help="Override model for lesson extraction")
@click.option("--min-messages", default=4, help="Minimum messages in a session to extract lessons")
@click.option("--no-think", is_flag=True, help="Disable model thinking/reasoning (faster for simple tasks)")
@click.pass_context
def reprocess_lessons_cmd(ctx, model, min_messages, no_think):
    """Delete bad lessons and re-extract from conversations."""
    from rich.progress import Progress

    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import get_ollama_url
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.reprocess import reprocess_lessons

    async def _run():
        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = ChromaStore(
            persist_dir=cfg.database.chroma_path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        # Override models if --model provided
        original_reasoning = None
        if model:
            original_reasoning = cfg.models.reasoning
            router._models.reasoning = model
            console.print(f"[cyan]Using model override: {model}[/cyan]")

        console.print(f"[cyan]Reprocessing lessons (min_messages={min_messages})...[/cyan]")

        with Progress(console=console) as progress:
            task = progress.add_task("Reprocessing lessons...", total=1)

            def on_progress(done, total):
                progress.update(task, completed=done, total=total,
                                description=f"[cyan]Processing sessions {done}/{total}[/cyan]")

            stats = await reprocess_lessons(
                sqlite=sqlite,
                chroma=chroma,
                router=router,
                min_messages=min_messages,
                no_think=no_think,
                on_progress=on_progress,
            )

        # Restore original model
        if original_reasoning:
            router._models.reasoning = original_reasoning

        await sqlite.close()

        # Print summary
        summary = Table(title="Reprocess Lessons Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Count", justify="right")
        summary.add_row("Old lessons deleted", str(stats["old_lessons_deleted"]))
        summary.add_row("Sessions processed", f"[green]{stats['sessions_processed']}[/green]")
        summary.add_row("Lessons extracted", f"[green]{stats['lessons_extracted']}[/green]")
        summary.add_row("Errors", f"[red]{stats['errors']}[/red]" if stats["errors"] else "0")
        console.print(summary)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
