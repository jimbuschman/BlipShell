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
import logging
import os
import re
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from blipshell.core.agent import Agent
from blipshell.core.config import ConfigManager
from blipshell.models.session import MessageRole
from prompt_toolkit.formatted_text import ANSI

from blipshell.ui.input import (
    APPROVAL_PROMPT, SIMPLE_PROMPT,
    async_prompt, create_chat_session, create_simple_session, format_chat_prompt,
)

console = Console()


def _ensure_vt_processing():
    """Enable ANSI/VT escape sequence processing on Windows.

    prompt_toolkit restores the original console mode (VT OFF) after each
    prompt. This must be called before writing raw ANSI to stdout.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            if not (mode.value & 0x4):  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
                kernel32.SetConsoleMode(handle, mode.value | 0x4)
    except Exception:
        pass


# Session-level auto-approve set: tools the user has approved for the rest of the session
_session_approved_tools: set[str] = set()

# prompt_toolkit session for tool approval / ask_user (no history)
_simple_session = None

# Tool display settings
_verbose_tools: bool = False  # When True, show full tool results (like /verbose toggle)
_tool_batch_history: list = []  # Stores recent tool batches for /expand


from blipshell.ui.diff import generate_colored_diff as _generate_colored_diff
from blipshell.ui.markdown_stream import MarkdownStreamer


async def _tool_approval_prompt(tool_name: str, arguments: dict, force: bool = False) -> bool:
    """Prompt the user before executing a dangerous tool.

    Shows a colored diff for file edit/write operations.
    When force=True, bypasses session auto-approve (used for destructive commands).
    Returns True to allow, False to deny.
    """
    # If user already approved this tool for the session, skip the prompt
    # UNLESS force=True (destructive command detected — always ask)
    if tool_name in _session_approved_tools and not force:
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

    # Show destructive command warning in bold red
    destructive_warning = arguments.get("_destructive_warning")
    if destructive_warning:
        console.print(
            f"\n\x1b[1;31m⚠ DESTRUCTIVE: {destructive_warning}\x1b[0m"
        )
        console.print(
            f"\x1b[33m[Approval required]\x1b[0m "
            f"\x1b[1m{tool_name}\x1b[0m: {arg_summary}"
        )
    else:
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
    Mouse clicks produce VT escape sequences (e.g. \\x1b[M... or \\x1b[<...)
    that must be consumed entirely to prevent character bleed-through.
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
                # Could be bare Esc or start of a VT sequence — peek ahead
                await asyncio.sleep(0.01)  # brief wait for rest of sequence
                if msvcrt.kbhit():
                    _consume_vt_sequence(msvcrt)
                    continue  # was a mouse/VT sequence, not Esc
                return  # bare Esc keypress
            # Discard other keypresses so they don't bleed into next input
        await asyncio.sleep(0.05)


def _consume_vt_sequence(msvcrt_mod):
    """Consume a VT/CSI escape sequence from the keyboard buffer.

    Called after \\x1b has been read and at least one more byte is available.
    Handles: CSI sequences (\\x1b[...), SS2/SS3 (\\x1b N/O ...), and
    mouse sequences (\\x1b[M + 3 bytes, \\x1b[< + SGR until M/m).
    """
    next_byte = msvcrt_mod.getch()
    if next_byte != b'[':
        # SS2/SS3 or unknown — consume one more byte and done
        return

    # CSI sequence: \x1b[ ... (ends at 0x40-0x7E)
    # Check for mouse protocols first
    if msvcrt_mod.kbhit():
        param = msvcrt_mod.getch()
        if param == b'M':
            # Basic mouse: \x1b[M + 3 raw bytes (button, x, y)
            for _ in range(3):
                if msvcrt_mod.kbhit():
                    msvcrt_mod.getch()
            return
        if param == b'<':
            # SGR mouse: \x1b[< params ; params ; params M/m
            while msvcrt_mod.kbhit():
                ch = msvcrt_mod.getch()
                if ch in (b'M', b'm'):
                    return
            return
        # Regular CSI: consume until final byte (0x40-0x7E)
        if 0x40 <= ord(param) <= 0x7E:
            return  # single-char CSI (e.g. \x1b[A for arrow keys)
        # Multi-byte CSI params — keep consuming
        while msvcrt_mod.kbhit():
            ch = msvcrt_mod.getch()
            if 0x40 <= ord(ch) <= 0x7E:
                return


def _drain_keyboard():
    """Drain any buffered keypresses to prevent bleed-through."""
    try:
        import msvcrt
        while msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch == b'\x1b' and msvcrt.kbhit():
                _consume_vt_sequence(msvcrt)
    except ImportError:
        pass


# Strong research signals — explicit intent to research/explore
_RESEARCH_STRONG = [
    re.compile(r'\b(investigate|explore|research|deep dive|dig into|look into)\b', re.I),
    re.compile(r'\b(compare|difference between|pros and cons|tradeoffs?)\b', re.I),
    re.compile(r'\b(best practice|state of the art|alternatives to)\b', re.I),
    re.compile(r'\b(find out|figure out|understand)\b.*\b(how|why|what)\b', re.I),
]

# Weak research signals — questions that MIGHT want research but could be conversational
_RESEARCH_WEAK = [
    re.compile(r'\b(how does|how do|how is|how are|how can|how would)\b', re.I),
    re.compile(r'\b(what is|what are|what\'s the|whats the)\b', re.I),
    re.compile(r'\b(why does|why do|why is|why are|why would)\b', re.I),
    re.compile(r'\bexplain\b', re.I),
]

# Patterns that indicate the user wants action or a quick answer, not research
_ACTION_PATTERNS = [
    re.compile(r'\b(fix|add|create|build|implement|write|change|update|modify|remove|delete|refactor)\b', re.I),
    re.compile(r'^!plan\b', re.I),
    # Conversational / status queries — not research
    re.compile(r'\b(show me|status|right now|currently|look at|check|run|list)\b', re.I),
    re.compile(r'\b(can you|could you|please)\b.*\b(do|make|set|give|tell)\b', re.I),
]


def _detect_research_intent(message: str) -> bool:
    """Detect if a message likely wants deep research, not a quick answer.

    Conservative: only triggers on strong signals or multiple weak signals
    in longer messages. Action verbs and conversational phrases suppress it.
    """
    if len(message) < 20:
        return False
    for p in _ACTION_PATTERNS:
        if p.search(message):
            return False

    # Strong signals — one is enough
    for p in _RESEARCH_STRONG:
        if p.search(message):
            return True

    # Weak signals — need the message to be longer (50+ chars) to avoid
    # triggering on casual questions like "what is the status"
    if len(message) >= 50:
        weak_hits = sum(1 for p in _RESEARCH_WEAK if p.search(message))
        if weak_hits >= 2:
            return True
        # Single weak hit + question mark on a long message
        if weak_hits == 1 and message.strip().endswith("?"):
            return True

    return False




def _format_tool_arg_summary(name: str, args: dict) -> str:
    """Format a brief argument summary for tool display."""
    if not args:
        return ""
    if name in ("read_file", "write_file", "edit_file"):
        return args.get("path", args.get("file_path", ""))
    if name in ("grep_files", "glob_files"):
        pattern = args.get("pattern", "")
        path = args.get("path", args.get("directory", ""))
        return f'"{pattern}" {path}'.strip()
    if name == "run_command":
        return args.get("command", "")[:60]
    if name == "list_directory":
        return args.get("path", args.get("directory", ""))
    if name == "web_search":
        return args.get("query", "")[:50]
    if name == "web_fetch":
        return args.get("url", "")[:60]
    # Generic: show first string arg
    for v in args.values():
        if isinstance(v, str) and v:
            return v[:50]
    return ""


def _tool_result_summary(name: str, args: dict, result, blocked: bool) -> str:
    """One-line result summary for a tool call."""
    if blocked:
        return "duplicate — skipped"
    if not result.success:
        err = (result.result or "unknown error")[:80].replace("\n", " ")
        return err
    if not result.result:
        return ""
    if name == "task_complete":
        return args.get("summary", result.result[:80] if result.result else "")[:80]
    if name == "edit_file":
        first_line = result.result.split("\n", 1)[0]
        return first_line[:80]
    if name == "run_command":
        lines = result.result.strip().split("\n")
        preview = lines[0][:80]
        if len(lines) > 1:
            preview += f" (+{len(lines) - 1} lines)"
        return preview
    if name == "read_file":
        return f"{result.result.count(chr(10)) + 1} lines"
    if name in ("grep_files", "glob_files"):
        stripped = result.result.strip()
        hits = stripped.split("\n") if stripped else []
        return f"{len(hits)} results"
    if name == "list_directory":
        stripped = result.result.strip()
        items = stripped.split("\n") if stripped else []
        return f"{len(items)} items"
    if name == "search_memories":
        stripped = result.result.strip()
        if not stripped:
            return "no results"
        # Count memory entries (each starts with a memory ID pattern or separator)
        lines = [l for l in stripped.split("\n") if l.strip() and not l.strip().startswith("---")]
        return f"{len(lines)} results" if lines else "no results"
    # Generic
    char_count = len(result.result)
    if char_count < 60:
        return result.result.strip()[:60]
    return f"{char_count} chars"


def _display_tool_batch(
    calls: list[tuple[str, dict]],
    results: list[tuple],
):
    """Render a tool batch — compact single-line per tool, Claude Code style.

    calls: list of (name, arguments) for each tool in the batch.
    results: list of (ToolResult, is_dedup_blocked) for each tool.
    """
    global _tool_batch_history
    _tool_batch_history.append((calls, results))
    if len(_tool_batch_history) > 50:
        _tool_batch_history = _tool_batch_history[-50:]

    for (name, args), (result, blocked) in zip(calls, results):
        arg_summary = _format_tool_arg_summary(name, args)
        summary = _tool_result_summary(name, args, result, blocked)

        # Icon by status
        if blocked:
            icon, style = "⎯", "dim"
        elif not result.success:
            icon, style = "✗", "red"
        elif name == "task_complete":
            icon, style = "●", "green"
        else:
            icon, style = "●", "dim"

        line = Text()
        line.append(f"  {icon} ", style=style)
        line.append(name, style="bold" if not blocked else "dim")
        if arg_summary:
            line.append(f" {arg_summary}", style="dim")
        if summary:
            line.append(f" — {summary}", style=style)
        console.print(line)

        # Edit diffs always shown (valuable context)
        if name == "edit_file" and result.success and result.result and "\x1b[" in result.result:
            diff_lines = result.result.split("\n", 1)
            if len(diff_lines) > 1 and diff_lines[1].strip():
                diff_text = diff_lines[1].replace("\n", "\n     ")
                sys.stdout.write(f"     {diff_text}\n")
                sys.stdout.flush()

        # Verbose mode: show extra detail for run_command
        if _verbose_tools and name == "run_command" and result.success and result.result:
            out_lines = result.result.strip().split("\n")
            for extra in out_lines[1:5]:
                console.print(f"     [dim]{extra[:120]}[/dim]")
            if len(out_lines) > 5:
                console.print(f"     [dim]... {len(out_lines) - 5} more lines[/dim]")


async def _pause_check() -> "PauseResult | None":
    """Non-blocking check for Space/p keypress to pause executor.

    Called between tool batches. If Space/p detected, shows interactive
    pause menu. Returns PauseResult or None (no pause requested).
    """
    from blipshell.core.chat_loop import PauseAction, PauseResult

    try:
        import msvcrt
    except ImportError:
        return None

    # Non-blocking peek — check if Space or 'p' was pressed
    paused = False
    while msvcrt.kbhit():
        key = msvcrt.getch()
        if key == b'\x1b' and msvcrt.kbhit():
            _consume_vt_sequence(msvcrt)
            continue  # mouse click — ignore
        if key in (b' ', b'p', b'P'):
            paused = True
        # Drain other buffered keys
    if not paused:
        return None

    # Show pause menu
    console.print("\n\x1b[1;33m--- PAUSED ---\x1b[0m")
    console.print("  (c) Continue   (r) Redirect   (s) Stop")

    try:
        choice = (await async_prompt(_simple_session, "  > ")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return PauseResult(action=PauseAction.STOP)

    if choice in ("c", "continue", ""):
        return PauseResult(action=PauseAction.CONTINUE)
    elif choice in ("s", "stop"):
        return PauseResult(action=PauseAction.STOP)
    elif choice in ("r", "redirect"):
        try:
            redirect = (await async_prompt(_simple_session, "  New instructions: ")).strip()
        except (EOFError, KeyboardInterrupt):
            return PauseResult(action=PauseAction.CONTINUE)
        if redirect:
            return PauseResult(action=PauseAction.REDIRECT, message=redirect)
        return PauseResult(action=PauseAction.CONTINUE)
    else:
        # Treat unknown input as redirect instructions
        return PauseResult(action=PauseAction.REDIRECT, message=choice)


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
    _active_chat_task: asyncio.Task | None = None

    def _sigint_handler(sig, frame):
        nonlocal _exit_requested, _active_chat_task
        if _in_cleanup:
            # During cleanup, second Ctrl+C force-quits
            raise KeyboardInterrupt
        _exit_requested = True
        if _active_chat_task is not None and not _active_chat_task.done():
            # Cancel the active chat task (including executor loops in project mode)
            # so asyncio.wait() returns immediately instead of waiting for completion
            _active_chat_task.cancel()
        else:
            # No active task — we're at the input prompt. Raise KeyboardInterrupt
            # so prompt_toolkit's prompt_async() breaks out immediately.
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    # Load config
    config_manager = ConfigManager(config_path)
    config = config_manager.load()

    # Create agent with startup progress
    agent = Agent(config, config_manager)
    console.print("[dim]Starting up...[/dim]")

    def _on_status(msg: str):
        console.print(f"[dim]  {msg}[/dim]")

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

    # Wire audit callback for tool approval logging
    async def _audit_tool_approval(tool_name: str, arguments: dict, approved: bool):
        sid = agent.session_manager.session_id if agent.session_manager else None
        await agent.sqlite.log_tool_approval(sid, tool_name, arguments, approved)

    agent.tool_registry.set_audit_callback(_audit_tool_approval)

    # Wire ask_user callback so the LLM can ask questions during execution
    agent.set_ask_user_callback(_ask_user_input)

    # Wire pause check callback for mid-task steering
    agent.set_pause_check_callback(_pause_check)

    # Create prompt_toolkit sessions for input (history, bracketed paste)
    global _simple_session

    def _build_toolbar():
        """Dynamic bottom toolbar showing session state."""
        parts = []
        if agent.session_manager and agent.session_manager.session_id:
            parts.append(f"Session #{agent.session_manager.session_id}")
        if agent.active_project:
            parts.append(f"Project: {agent.active_project['name']}")
        ep = agent._last_endpoint_used
        if ep:
            parts.append(f"Endpoint: {ep}")
        if agent._last_context_stats:
            pct = agent._last_context_stats.get("usage_pct", 0)
            parts.append(f"Context: {pct:.0f}%")
        if agent.think_enabled:
            parts.append("Think: ON")
        return ANSI(f"\x1b[2m {' | '.join(parts)} \x1b[0m") if parts else ""

    chat_session = create_chat_session(bottom_toolbar=_build_toolbar)
    _simple_session = create_simple_session()

    sid = await agent.start_session(project=project, resume_session_id=resume_id)

    # Auto-activate project if specified via --project flag
    if project:
        try:
            with console.status("[dim]Loading project...[/dim]", spinner="dots"):
                await agent.activate_project(project)
        except KeyError:
            console.print(f"[yellow]Project '{project}' not found in DB. Use /project new to create it.[/yellow]")

    # Disable terminal focus reporting — prevents \x1b[I / \x1b[O sequences
    # from being injected as literal [O[I characters when clicking in the window.
    sys.stdout.write("\x1b[?1004l")
    sys.stdout.flush()

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

    # Show nightly run status (always shown when nightly has run)
    if hasattr(agent, '_nightly_notification') and agent._nightly_notification:
        style = "dim yellow" if "error" in agent._nightly_notification or "warning" in agent._nightly_notification else "dim green"
        console.print(f"  {agent._nightly_notification}", style=style)
        agent._nightly_notification = None

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
                raw_input = (await async_prompt(chat_session, prompt)).strip()
                # Sanitize surrogates — Windows clipboard can produce unpaired
                # UTF-16 surrogates (e.g. from emoji) that crash UTF-8 encoding
                # in prompt_toolkit history, SQLite, ChromaDB, and LLM clients.
                user_input = "".join(
                    c if ord(c) < 0xD800 or ord(c) > 0xDFFF else "\ufffd"
                    for c in raw_input
                )
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
                elif cmd[0] == "guardrails":
                    if len(cmd) > 1 and cmd[1] in ("on", "off"):
                        config.guardrails.enabled = cmd[1] == "on"
                    else:
                        config.guardrails.enabled = not config.guardrails.enabled
                    state = "[green]ON[/green]" if config.guardrails.enabled else "[yellow]OFF[/yellow]"
                    console.print(f"[dim]Guardrails: {state}[/dim]")
                    if config.guardrails.enabled:
                        features = []
                        if config.guardrails.completion_audit:
                            features.append("completion audit")
                        if config.guardrails.correction_detector:
                            features.append("correction detector")
                        if config.guardrails.trajectory_monitor:
                            features.append("trajectory monitor")
                        if config.guardrails.context_pinning:
                            features.append("context pinning")
                        if config.guardrails.requirement_checklist:
                            features.append("requirement checklist")
                        console.print(f"[dim]  Active: {', '.join(features)}[/dim]")
                    continue
                elif cmd[0] == "verbose":
                    global _verbose_tools
                    _verbose_tools = not _verbose_tools
                    state = "[green]ON[/green]" if _verbose_tools else "[yellow]OFF[/yellow]"
                    console.print(f"[dim]Verbose tool output: {state}[/dim]")
                    continue
                elif cmd[0] == "expand":
                    if not _tool_batch_history:
                        console.print("[dim]No tool batches to show.[/dim]")
                    else:
                        # Show last N batches (default 1)
                        n = 1
                        if cmd_args:
                            try:
                                n = int(cmd_args[0])
                            except ValueError:
                                pass
                        batches = _tool_batch_history[-n:]
                        for batch_idx, (calls, results) in enumerate(batches):
                            for (name, args), (result, blocked) in zip(calls, results):
                                if blocked:
                                    console.print(f"[dim]  {name}: [duplicate blocked][/dim]")
                                    continue
                                arg_summary = _format_tool_arg_summary(name, args)
                                style = "red" if not result.success else "bold"
                                console.print(f"  [{style}]{name}[/{style}] {arg_summary}", highlight=False)
                                # Show full result
                                if result.result:
                                    text = result.result[:2000]
                                    console.print(Panel(text, border_style="dim", expand=False))
                            if batch_idx < len(batches) - 1:
                                console.print("[dim]---[/dim]")
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
                elif cmd[0] == "research":
                    query = " ".join(cmd_args) if cmd_args else ""
                    if not query:
                        console.print("[yellow]Usage: /research <question or topic>[/yellow]")
                        console.print("[dim]Triggers deep research with web search and thorough exploration.[/dim]")
                        continue
                    # Rewrite user_input so it's handled like !plan below
                    user_input = "!research " + query
                    # Fall through (no continue) — handled alongside !plan
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
                    if cmd_args and cmd_args[0] == "report":
                        await _print_nightly_report(agent)
                    else:
                        job_name = cmd_args[0] if cmd_args else None
                        await _run_nightly(agent, job_name)
                    continue
                elif cmd[0] == "mcp":
                    _print_mcp_status(agent, cmd_args)
                    continue
                elif cmd[0] == "changes":
                    _print_changes(agent)
                    continue
                elif cmd[0] in ("followups", "followup"):
                    await _print_followups(agent)
                    continue
                elif cmd[0] == "friction":
                    reviewed = len(cmd_args) > 0 and cmd_args[0] == "all"
                    await _print_friction(agent, show_all=reviewed)
                    continue
                elif cmd[0] == "compact":
                    focus = " ".join(cmd_args) if cmd_args else ""
                    await _handle_compact(agent, focus)
                    continue
                elif cmd[0] == "notes":
                    await _handle_notes_command(agent, cmd_args)
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

            # Check for force-plan and research prefixes
            force_plan = False
            research_mode = False
            message = user_input
            if user_input.startswith("!plan "):
                force_plan = True
                message = user_input[6:]
            elif user_input.startswith("!research "):
                research_mode = True
                message = user_input[10:]
                console.print(f"[dim italic]Researching: {message}[/dim italic]")

            # Auto-detect research intent (only for normal messages, not commands/plan)
            if not force_plan and not research_mode and _detect_research_intent(message):
                try:
                    answer = (await async_prompt(
                        _simple_session,
                        "  This looks like a research question. Use /research mode? (y/n) ",
                    )).strip().lower()
                    if answer in ("y", "yes"):
                        research_mode = True
                        console.print("[dim italic]Research mode activated[/dim italic]")
                except (EOFError, KeyboardInterrupt):
                    pass

            # Stream response with thinking spinner (Esc to cancel)
            _ensure_vt_processing()  # prompt_toolkit may have disabled VT after input
            response_parts = []
            thinking_status = console.status("[dim]Thinking...[/dim]", spinner="dots")
            thinking_active = True
            cancelled = False
            md_streamer = MarkdownStreamer()

            def on_token(token: str):
                nonlocal thinking_active
                if thinking_active:
                    thinking_status.stop()
                    thinking_active = False
                response_parts.append(token)
                # ANSI escape sequences (tool status, cursor control) pass through raw.
                # Reset the markdown streamer first — tool displays move the cursor,
                # invalidating the erase-and-replace mechanism for partial lines.
                if "\x1b[" in token:
                    md_streamer.reset_line()
                    sys.stdout.write(token)
                else:
                    sys.stdout.write(md_streamer.feed(token))
                sys.stdout.flush()

            console.print()  # blank line before response
            thinking_status.start()

            _active_chat_task = asyncio.create_task(
                agent.chat(message, on_token=on_token, force_plan=force_plan, on_tool_display=_display_tool_batch, research_mode=research_mode)
            )
            chat_task = _active_chat_task
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
            except (asyncio.CancelledError, KeyboardInterrupt):
                cancelled = True
                response = "".join(response_parts)
            except Exception as e:
                response = f"Error: {e}"
            finally:
                _active_chat_task = None
                if thinking_active:
                    thinking_status.stop()
                # Flush any remaining markdown formatting
                remaining = md_streamer.flush()
                if remaining:
                    sys.stdout.write(remaining)
                    sys.stdout.flush()
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


def _print_help():
    """Print help for CLI commands."""
    console.print(Panel(
        "[bold]/quit[/bold]                  - Exit BlipShell\n"
        "[bold]/status[/bold]                - Show agent status, endpoints, routing\n"
        "[bold]/memory[/bold]                - Show memory pool usage\n"
        "[bold]/context[/bold]               - Show context window usage breakdown\n"
        "[bold]/tokens[/bold]                - Show token usage per endpoint this session\n"
        "[bold]/compact[/bold] [dim][focus][/dim]        - Compact older messages to free context\n"
        "[bold]/notes[/bold]                 - Show session notes (survive compaction)\n"
        "[bold]/notes save[/bold] [dim]<name> <text>[/dim] - Save a session note\n"
        "[bold]/notes delete[/bold] [dim]<name>[/dim]   - Delete a note\n"
        "[bold]/save[/bold]                  - Force save session to memory\n"
        "[bold]/core[/bold]                  - Show core memories and lessons\n"
        "[bold]/think[/bold]                 - Toggle LLM thinking mode on/off\n"
        "[bold]/reflect[/bold]               - Toggle self-reflection on/off\n"
        "[bold]/guardrails[/bold] [dim][on|off][/dim]   - Toggle guardrails (completion audit, drift monitor)\n"
        "[bold]/verbose[/bold]               - Toggle verbose tool output on/off\n"
        "[bold]/expand[/bold] [dim][n][/dim]             - Show full output of last n tool batches\n"
        "[bold]/approve[/bold] [dim]all|reset[/dim]     - Manage tool approval (write/edit/run)\n"
        "[bold]/changes[/bold]               - Show files modified this session\n"
        "[bold]/followups[/bold]             - Show pending follow-up items\n"
        "[bold]/friction[/bold] [dim][all][/dim]        - Show system friction log (unreviewed, or all)\n"
        "[bold]/research <query>[/bold]       - Deep research with web + code exploration\n"
        "[bold]/code <path> [msg][/bold]     - Send code to LLM for review\n"
        "[bold]/feedback <msg>[/bold]        - Save feedback as a lesson\n"
        "[bold]/offload <msg>[/bold]         - Run a task on remote PC in background\n"
        "[bold]/health[/bold] [dim][quick][/dim]          - Database + endpoint health check\n"
        "[bold]/cleanup[/bold]               - Reprocess failed messages (relaxed timeouts)\n"
        "[bold]/nightly[/bold] [dim][job|report][/dim]    - Run nightly maintenance or show last report\n"
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


@main.command("telegram")
@click.pass_context
def telegram_cmd(ctx):
    """Run the Telegram bot — chat with BlipShell from your phone."""
    from blipshell.ui.telegram import run_telegram_bot

    console.print("[cyan]Starting Telegram bot... (Ctrl+C to stop)[/cyan]")
    asyncio.run(run_telegram_bot(config_path=ctx.obj.get("config_path")))


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


# --- Simulation ---

@main.command("simulate")
@click.option("--scenario", "-s", default=None, help="Run a single scenario by name")
@click.option("--category", "-c", default=None, help="Run scenarios in a category (e.g. regression, tool_registration)")
@click.option("--quiet", "-q", is_flag=True, help="JSON output only")
@click.option("--output", "-o", default=None, help="Write JSON report to file")
@click.option("--list", "list_scenarios", is_flag=True, help="List available scenarios")
@click.pass_context
def simulate_cmd(ctx, scenario, category, quiet, output, list_scenarios):
    """Run automated user simulation — exercises BlipShell like a real user.

    Boots the real Agent, runs multi-turn scenarios (slash commands, mode
    transitions, conversations, project workflows, regressions), reports
    what passed and what didn't.

    By default runs ALL scenarios. Use -s or -c to narrow down.
    """
    from blipshell.simulate import (
        SimRunner,
        collect_all_scenarios,
        filter_by_category,
        filter_by_name,
    )
    from blipshell.simulate.reporting import (
        export_json,
        print_scenario_result,
        print_suite_summary,
    )

    all_scenarios = collect_all_scenarios()

    if list_scenarios:
        from rich.table import Table
        table = Table(title=f"Available Scenarios ({len(all_scenarios)})")
        table.add_column("Name", style="cyan")
        table.add_column("Category")
        table.add_column("Steps", justify="right")
        table.add_column("Description")
        for s in all_scenarios:
            table.add_row(
                s.name, s.category,
                str(len(s.steps)),
                s.description[:70],
            )
        console.print(table)
        return

    # Filter scenarios
    scenarios = all_scenarios
    if scenario:
        scenarios = filter_by_name(scenarios, scenario)
        if not scenarios:
            console.print(f"[yellow]Scenario '{scenario}' not found[/yellow]")
            return
    if category:
        scenarios = filter_by_category(scenarios, category)

    if not scenarios:
        console.print("[yellow]No scenarios matched the filters[/yellow]")
        return

    config_path = ctx.obj.get("config_path")

    async def _run():
        def on_status(msg: str):
            if not quiet:
                console.print(f"[dim]{msg}[/dim]")

        runner = SimRunner(
            config_path=config_path,
            quiet=quiet,
            on_status=on_status,
        )

        if not quiet:
            console.print(f"[bold cyan]Running {len(scenarios)} simulation scenarios...[/bold cyan]")

        suite_result = await runner.run_suite(scenarios)

        if not quiet:
            for sr in suite_result.scenario_results:
                print_scenario_result(console, sr)
            print_suite_summary(console, suite_result)

        if output or quiet:
            json_str = export_json(suite_result)
            if output:
                with open(output, "w") as f:
                    f.write(json_str)
                if not quiet:
                    console.print(f"\n[dim]Report written to {output}[/dim]")
            if quiet:
                print(json_str)

    asyncio.run(_run())


# --- Nightly Jobs ---

@main.command("nightly")
@click.option("--job", default=None, help="Run a specific job only (e.g. centroid_tag, batch_tag)")
@click.option("--quiet", "-q", is_flag=True, help="JSON output only (for scheduled runs)")
@click.option("--loop", is_flag=True, help="Repeat until nothing left to process (use with --job)")
@click.option("--local", is_flag=True, help="Force all LLM calls through local Ollama (avoids cloud rate limits)")
@click.pass_context
def nightly_cmd(ctx, job, quiet, loop, local):
    """Run nightly maintenance jobs (backup, tagging, pruning, etc.)."""
    import json as _json

    async def _run():
        from blipshell.core.nightly import NightlyRunner

        runner = await NightlyRunner.create_from_config(
            ctx.obj.get("config_path"), local_only=local,
        )
        try:
            jobs = [job] if job else None
            iteration = 0

            while True:
                iteration += 1

                if quiet:
                    result = await runner.run(jobs=jobs)
                    print(_json.dumps(result, indent=2, default=str))
                else:
                    from rich.status import Status
                    from rich.table import Table

                    label = f"job: {job}" if job else "all jobs"
                    loop_label = f" (pass {iteration})" if loop else ""
                    with Status(f"[bold cyan]Running nightly ({label}{loop_label})...", console=console) as status:
                        def on_status(msg: str):
                            status.update(f"[bold cyan]{msg}")

                        result = await runner.run(on_status=on_status, jobs=jobs)

                    table = Table(title=f"Nightly Run Results{loop_label}")
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

                if not loop:
                    break

                # Check if the job actually did work — stop if nothing left
                job_stats = result.get("jobs", {})
                did_work = False
                for stats in job_stats.values():
                    for key in ("resummarized", "scored", "processed", "merged",
                                "deleted_junk", "deleted_dupes", "pruned", "rebuilt"):
                        if stats.get(key, 0) > 0:
                            did_work = True
                            break
                if not did_work:
                    if not quiet:
                        console.print("[green]Nothing left to process — done.[/green]")
                    break

        finally:
            await runner.close()

    asyncio.run(_run())


# --- Bulk Session Review ---

@main.command("review")
@click.option("--lessons/--no-lessons", default=True, help="Extract lessons from sessions")
@click.option("--reflections/--no-reflections", default=True, help="Generate session reflections")
@click.option("--limit", type=int, default=None, help="Max sessions to process (default: all)")
@click.option("--quiet", "-q", is_flag=True, help="JSON output only")
@click.pass_context
def review_cmd(ctx, lessons, reflections, limit, quiet):
    """Bulk process all sessions missing lessons and/or reflections."""
    import json as _json

    async def _run():
        from scripts.bulk_session_review import run_bulk_review

        config_path = ctx.obj.get("config_path")

        if quiet:
            result = await run_bulk_review(
                config_path=config_path,
                do_lessons=lessons,
                do_reflections=reflections,
                limit=limit,
            )
            print(_json.dumps(result, indent=2, default=str))
        else:
            from rich.status import Status

            with Status("[bold cyan]Processing sessions...", console=console) as status:
                def on_status(msg):
                    status.update(f"[bold cyan]{msg}")

                result = await run_bulk_review(
                    config_path=config_path,
                    do_lessons=lessons,
                    do_reflections=reflections,
                    limit=limit,
                    on_status=on_status,
                )

            for job_name in ["lessons", "reflections"]:
                job_stats = result.get(job_name)
                if job_stats is None:
                    continue
                console.print(f"\n[bold]{job_name.title()}[/bold]:")
                for k, v in job_stats.items():
                    console.print(f"  {k}: {v}")

            console.print(f"\n[dim]Total: {result['elapsed_s']}s[/dim]")

    asyncio.run(_run())


@main.command("repair")
@click.option("--restore-imports", is_flag=True,
              help="Unarchive memories from imported sessions ([project]-prefixed titles).")
@click.option("--sweep-orphans", is_flag=True,
              help="Delete vectors whose memories are archived or missing.")
@click.option("--fix-sessions", is_flag=True,
              help="Fix sessions with message_count=0 and title='New Session' that have memories.")
@click.option("--dry-run", is_flag=True, help="Show counts without making changes.")
@click.option("--all", "do_all", is_flag=True,
              help="Run all repairs.")
@click.pass_context
def repair_cmd(ctx, restore_imports, sweep_orphans, fix_sessions, dry_run, do_all):
    """Repair common DB issues.

    --restore-imports unarchives memories from imported sessions.
    --sweep-orphans removes orphan vector rows.
    --fix-sessions fixes sessions where end_session() failed (count=0, no title).
    """
    if not (restore_imports or sweep_orphans or fix_sessions or do_all):
        console.print("[yellow]Nothing to do. Pass --restore-imports, --sweep-orphans, --fix-sessions, or --all.[/yellow]")
        return
    if do_all:
        restore_imports = True
        sweep_orphans = True
        fix_sessions = True

    from blipshell.memory.vector_store import VectorStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.models.config import get_ollama_url

    async def _run():
        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        vectors = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
        )
        vectors.initialize()

        try:
            if restore_imports:
                cursor = await sqlite._db.execute(
                    """
                    SELECT COUNT(*) FROM memories m
                     JOIN sessions s ON s.id = m.session_id
                     WHERE m.is_archived = 1
                       AND (s.title LIKE '[%' OR s.title LIKE '[Claude Code Memory]%')
                    """
                )
                row = await cursor.fetchone()
                count = row[0] if row else 0
                console.print(f"[cyan]Imported memories archived:[/cyan] [bold]{count}[/bold]")

                if count and not dry_run:
                    await sqlite._db.execute(
                        """
                        UPDATE memories
                           SET is_archived = 0
                         WHERE is_archived = 1
                           AND session_id IN (
                                 SELECT id FROM sessions
                                  WHERE title LIKE '[%'
                                     OR title LIKE '[Claude Code Memory]%'
                           )
                        """
                    )
                    await sqlite._db.commit()
                    console.print(f"[green]Restored {count} memories.[/green]")
                elif dry_run:
                    console.print("[dim](dry-run; no changes)[/dim]")

            if fix_sessions:
                cursor = await sqlite._db.execute(
                    """
                    SELECT s.id,
                           (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.id) as mem_count,
                           (SELECT substr(m.content, 1, 80)
                              FROM memories m
                             WHERE m.session_id = s.id AND m.role = 'user'
                             ORDER BY m.id LIMIT 1) as first_user
                      FROM sessions s
                     WHERE s.message_count = 0
                       AND EXISTS (SELECT 1 FROM memories m WHERE m.session_id = s.id)
                    """
                )
                broken = await cursor.fetchall()
                console.print(f"[cyan]Broken sessions (count=0 but have memories):[/cyan] [bold]{len(broken)}[/bold]")

                if broken and not dry_run:
                    fixed = 0
                    for sid, mem_count, first_user in broken:
                        title = first_user.replace("\n", " ").strip()[:80] if first_user else f"Session {sid}"
                        if len(title) > 77:
                            title = title[:77] + "..."
                        await sqlite._db.execute(
                            "UPDATE sessions SET message_count = ?, title = CASE WHEN title = 'New Session' THEN ? ELSE title END WHERE id = ?",
                            (mem_count, title, sid),
                        )
                        fixed += 1
                    await sqlite._db.commit()
                    console.print(f"[green]Fixed {fixed} sessions (set message_count + fallback title).[/green]")
                elif dry_run and broken:
                    for sid, mem_count, first_user in broken[:5]:
                        title = (first_user or "").replace("\n", " ").strip()[:60]
                        console.print(f"  #{sid} mems={mem_count} [dim]{title}[/dim]")
                    if len(broken) > 5:
                        console.print(f"  [dim]...and {len(broken) - 5} more[/dim]")
                    console.print("[dim](dry-run; no changes)[/dim]")

            if sweep_orphans:
                if dry_run:
                    # Inspect counts without deleting
                    cur = vectors._conn.execute(
                        """
                        SELECT COUNT(*) FROM vec_memories vm
                         JOIN memories m ON m.id = vm.rowid
                         WHERE m.is_archived = 1
                        """
                    )
                    arch = cur.fetchone()[0]
                    cur = vectors._conn.execute(
                        """
                        SELECT COUNT(*) FROM vec_memories vm
                         WHERE vm.rowid NOT IN (SELECT id FROM memories)
                        """
                    )
                    miss = cur.fetchone()[0]
                    console.print(
                        f"[cyan]Orphan vectors:[/cyan] "
                        f"archived=[bold]{arch}[/bold] missing=[bold]{miss}[/bold] "
                        "[dim](dry-run; no changes)[/dim]"
                    )
                else:
                    result = vectors.cleanup_orphan_vectors()
                    console.print(
                        f"[green]Orphan vectors swept:[/green] "
                        f"archived={result['archived']} missing={result['missing']}"
                    )
        finally:
            await sqlite.close()

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
    from blipshell.memory.vector_store import VectorStore
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

        chroma = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
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
                vectors=chroma,
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
    from blipshell.memory.vector_store import VectorStore
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

        chroma = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        count = await import_memories_as_core(
            sqlite=sqlite,
            vectors=chroma,
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
    from blipshell.memory.vector_store import VectorStore
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

        chroma = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
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
                vectors=chroma,
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
    from blipshell.memory.vector_store import VectorStore
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

        chroma = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
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
                vectors=chroma,
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


@import_claude_group.command("code")
@click.argument("path", type=click.Path(exists=True))
@click.option("--max", "max_count", type=int, default=None,
              help="Only import the first N sessions (for testing)")
@click.option("--skip-lessons", is_flag=True, help="Skip lesson extraction (faster)")
@click.option("--concurrent", "max_concurrent", type=int, default=3,
              help="Number of conversations to process in parallel (default 3)")
@click.pass_context
def claude_code(ctx, path, max_count, skip_lessons, max_concurrent):
    """Import conversations from Claude Code JSONL session logs.

    PATH can be a single .jsonl file, a project directory, or the top-level
    ~/.claude/projects directory to import all sessions at once.
    """
    from rich.progress import Progress

    from blipshell.core.import_lock import import_lock
    from blipshell.import_claude_code import parse_claude_code_sessions
    from blipshell.import_common import import_conversations
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import get_ollama_url
    from blipshell.memory.vector_store import VectorStore
    from blipshell.memory.sqlite_store import SQLiteStore

    async def _import():
        console.print(f"[cyan]Scanning {path} for Claude Code sessions...[/cyan]")
        convs = parse_claude_code_sessions(path)
        console.print(f"Found [bold]{len(convs)}[/bold] sessions.")

        if max_count:
            convs = convs[:max_count]
            console.print(f"Importing first [bold]{max_count}[/bold].")

        if not convs:
            console.print("[yellow]No sessions to import.[/yellow]")
            return

        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        with import_lock(cfg.database.path, operation="import-claude-code"):
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
                    vectors=chroma,
                    router=router,
                    config=cfg.memory,
                    conversations=convs,
                    on_progress=on_progress,
                    skip_lessons=skip_lessons,
                    max_concurrent=max_concurrent,
                )
                progress.update(task, completed=len(convs))

        await sqlite.close()
        _print_import_summary(stats)

    asyncio.run(_import())


@import_claude_group.command("memories")
@click.argument("path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Show what would be imported without writing")
@click.pass_context
def claude_memories(ctx, path, dry_run):
    """Import memory files from Claude Code's memory system.

    PATH can be ~/.claude/projects (all projects), a single project dir,
    or a project's memory/ subdirectory.
    """
    from blipshell.core.import_lock import import_lock
    from blipshell.import_claude_code_memories import (
        MemoryImportStats, import_memories, parse_claude_code_memories,
    )
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.models.config import get_ollama_url
    from blipshell.memory.vector_store import VectorStore
    from blipshell.memory.sqlite_store import SQLiteStore

    async def _import():
        console.print(f"[cyan]Scanning {path} for Claude Code memory files...[/cyan]")
        memories = parse_claude_code_memories(path)
        console.print(f"Found [bold]{len(memories)}[/bold] memory files.")

        if not memories:
            console.print("[yellow]No memory files to import.[/yellow]")
            return

        if dry_run:
            console.print("\n[bold]Dry run — would import:[/bold]")
            for mem in memories:
                console.print(
                    f"  {mem.memory_type.value:12s} "
                    f"[cyan]{mem.name}[/cyan]"
                    f"  [dim]({mem.project_name or 'global'})[/dim]"
                )

        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        vectors = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
        )
        vectors.initialize()

        stats = MemoryImportStats(projects_scanned=len(set(
            m.project_name for m in memories if m.project_name
        )))

        with import_lock(cfg.database.path, operation="import-claude-memories"):
            await import_memories(sqlite, vectors, memories, stats, dry_run=dry_run)

        await sqlite.close()

        # Summary
        console.print(f"\n[bold]{'Dry run results' if dry_run else 'Import complete'}:[/bold]")
        console.print(f"  Projects scanned: {stats.projects_scanned}")
        console.print(f"  Files scanned:    {stats.files_scanned}")
        console.print(f"  Imported:         [green]{stats.memories_imported}[/green]")
        console.print(f"  Skipped (dupes):  [yellow]{stats.memories_skipped}[/yellow]")
        if stats.parse_errors:
            console.print(f"  Parse errors:     [red]{stats.parse_errors}[/red]")

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
    from blipshell.memory.vector_store import VectorStore
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

        chroma = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
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
                vectors=chroma,
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


# --- Generic Conversation Import ---

@main.command("import-conversation")
@click.argument("file", type=click.Path(exists=True))
@click.option("--skip-lessons", is_flag=True, help="Skip lesson extraction (faster)")
@click.option("--title", default=None, help="Override conversation title from JSON")
@click.pass_context
def import_conversation(ctx, file, skip_lessons, title):
    """Import a conversation from a JSON file through the memory pipeline.

    JSON format: {"conversation_title": "...", "messages": [{"role": "user"|"assistant", "content": "..."}]}

    Messages go through the full pipeline: summarization, ranking, importance,
    entity extraction, embedding, and lesson extraction — same as regular sessions.
    """
    import json
    from pathlib import Path

    from blipshell.import_common import ParsedConversation, ParsedMessage, import_conversations
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.memory.vector_store import VectorStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.models.config import get_ollama_url

    # Parse JSON
    file_path = Path(file)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        return
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        return

    # Support both single conversation and array of conversations
    if isinstance(data, list):
        conversations_data = data
    else:
        conversations_data = [data]

    parsed = []
    for conv_data in conversations_data:
        conv_title = title or conv_data.get("conversation_title") or conv_data.get("title") or file_path.stem
        raw_msgs = conv_data.get("messages", [])
        if not raw_msgs:
            console.print(f"[yellow]Skipping '{conv_title}' — no messages.[/yellow]")
            continue

        messages = []
        for msg in raw_msgs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "").lower()
            content = msg.get("content", "").strip()
            if role in ("user", "assistant") and content:
                messages.append(ParsedMessage(role=role, content=content))

        if messages:
            parsed.append(ParsedConversation(title=conv_title, messages=messages))
            console.print(f"[cyan]{conv_title}[/cyan]: {len(messages)} messages")

    if not parsed:
        console.print("[yellow]No conversations to import.[/yellow]")
        return

    console.print(f"\nImporting [bold]{len(parsed)}[/bold] conversation(s) through the memory pipeline...")

    async def _import():
        from rich.progress import Progress

        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        with Progress(console=console) as progress:
            task = progress.add_task("Importing...", total=len(parsed))

            def on_progress(idx, total, conv_title, stats):
                label = f"[cyan]{conv_title[:40]}[/cyan]"
                i, s = stats.conversations_imported, stats.conversations_skipped
                if i or s:
                    label += f"  [dim]({i} imported, {s} skipped)[/dim]"
                progress.update(task, completed=idx, description=label)

            stats = await import_conversations(
                sqlite=sqlite,
                vectors=chroma,
                router=router,
                config=cfg.memory,
                conversations=parsed,
                on_progress=on_progress,
                skip_lessons=skip_lessons,
            )
            progress.update(task, completed=len(parsed))

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
    from blipshell.memory.vector_store import VectorStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.reprocess import reprocess_memories

    async def _run():
        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
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
                vectors=chroma,
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
    from blipshell.memory.vector_store import VectorStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.reprocess import reprocess_lessons

    async def _run():
        config_manager = ConfigManager(ctx.obj.get("config_path"))
        cfg = config_manager.load()

        sqlite = SQLiteStore(cfg.database.path)
        await sqlite.initialize()

        chroma = VectorStore(
            db_path=cfg.database.path,
            embedding_model=cfg.models.embedding,
            ollama_url=get_ollama_url(cfg.endpoints),
            embedding_dim=cfg.database.embedding_dimensions,
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
                vectors=chroma,
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
