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
from blipshell.core.vision import extract_image_paths
from blipshell.models.session import MessageRole
from prompt_toolkit.formatted_text import ANSI

from blipshell.ui.input import (
    APPROVAL_PROMPT, SIMPLE_PROMPT,
    async_prompt, create_chat_session, create_simple_session, format_chat_prompt,
)

# The Console instance is shared with ui/views.py — same object, so Rich
# state (width, live displays) stays consistent across the split.
from blipshell.ui.console import console


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


# Session/UI state lives on ui_state so command handlers can mutate it from
# another module (a `global` statement can't reach across modules).
from blipshell.ui.state import ui_state
from blipshell.ui.commands import (
    QUIT, CommandContext, Rewrite, registry as command_registry,
)
import blipshell.ui.command_handlers  # noqa: F401  (registers the commands)

# prompt_toolkit session for tool approval / ask_user (no history)
_simple_session = None


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
    if tool_name in ui_state.session_approved_tools and not force:
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
        ui_state.session_approved_tools.add(tool_name)
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
    ui_state.record_tool_batch(calls, results)

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
        if ui_state.verbose_tools and name == "run_command" and result.success and result.result:
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
                parts = user_input[1:].lower().split()
                if not parts:
                    continue
                outcome = await command_registry.dispatch(CommandContext(
                    agent=agent,
                    config=config,
                    raw=user_input,
                    parts=parts,
                    args=user_input[1:].split()[1:],   # original case preserved
                    ui=ui_state,
                    console=console,
                ))
                if outcome is QUIT:
                    break
                if isinstance(outcome, Rewrite):
                    # e.g. /research -> "!research ..."; fall through to the
                    # normal message path below rather than looping.
                    user_input = outcome.text
                else:
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

            # Auto-detect inline image file paths (vision input). Strips the
            # paths from the text and attaches them to the turn.
            images = None
            cleaned_msg, img_paths = extract_image_paths(message)
            if img_paths:
                images = img_paths
                message = cleaned_msg
                for p in img_paths:
                    console.print(f"[dim italic]  attaching image: {Path(p).name}[/dim italic]")

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
                agent.chat(message, on_token=on_token, force_plan=force_plan, on_tool_display=_display_tool_batch, research_mode=research_mode, images=images)
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


# Renderers and command actions live in ui/views.py (extracted 2026-08-05).
from blipshell.ui.views import (  # noqa: F401
    _check_completed_tasks,
    _create_project,
    _delete_core_item,
    _detect_language,
    _handle_code_command,
    _handle_compact,
    _handle_cube_command,
    _handle_notes_command,
    _handle_project_command,
    _handle_workflow_command,
    _list_projects,
    _print_active_plan,
    _print_background_tasks,
    _print_changes,
    _print_context,
    _print_core,
    _print_flow,
    _print_followups,
    _print_friction,
    _print_health,
    _print_memory_usage,
    _print_nightly_report,
    _print_plans,
    _print_project_info,
    _print_status,
    _print_task_detail,
    _print_thoughts,
    _print_tokens,
    _render_plan,
    _run_cleanup,
    _run_nightly,
    _save_feedback,
    _submit_offload,
)

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
    from scripts.run_executor import run_test, run_canned_tests, run_stress_tests

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
@click.option("--db", "db_path", default=None,
              help="Database to run against (default: a throwaway temp DB)")
@click.option("--real-db", is_flag=True,
              help="Run against the CONFIGURED production database. Scenarios create "
                   "real sessions, lessons and digest updates in your live corpus.")
@click.pass_context
def simulate_cmd(ctx, scenario, category, quiet, output, list_scenarios, db_path, real_db):
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
            db_path=db_path,
            use_real_db=real_db,
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

        return suite_result

    suite_result = asyncio.run(_run())

    # Exit nonzero on hard failures so a simulate run can gate anything.
    # WARN is deliberately not a failure: response-content assertions are soft
    # because LLM phrasing is nondeterministic.
    if suite_result and suite_result.failed:
        if not quiet:
            console.print(
                f"[red]{suite_result.failed} of {suite_result.total} scenarios FAILED[/red]"
            )
        raise SystemExit(1)


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

            return result
        finally:
            await runner.close()

    result = asyncio.run(_run())

    # Exit nonzero when jobs didn't complete, so a scheduled run can be
    # monitored. Without this, a night where every job failed was
    # indistinguishable from success to Task Scheduler / cron.
    bad = {
        name: stats.get("status")
        for name, stats in (result or {}).get("jobs", {}).items()
        if stats.get("status") in ("error", "timeout")
    }
    if bad:
        if not quiet:
            console.print(
                f"[red]{len(bad)} job(s) did not complete: "
                f"{', '.join(f'{n} ({s})' for n, s in sorted(bad.items()))}[/red]"
            )
        raise SystemExit(1)


# --- Unified model benchmark harness ---

@main.group("benchmark")
def benchmark_grp():
    """Benchmark local/cloud models for the jobs BlipShell routes per task.

    run <model> — ONE deep test across every job (ability + speed); writes a
                  shareable report you can hand to a stronger LLM.
    report      — regenerate that report from stored runs (no re-run).
    discover    — pull a candidate shortlist from OpenRouter + Artificial Analysis.
    """


@benchmark_grp.command("run")
@click.argument("model")
@click.option("--judge/--no-judge", default=True, help="Grade open-ended jobs with the configured neutral judge")
@click.option("--provider", default="ollama", type=click.Choice(["ollama", "openai"]), help="Candidate endpoint provider")
@click.option("--url", default=None, help="Candidate endpoint URL (default: first local Ollama endpoint)")
@click.option("--api-key-env", default=None, help="Env var holding the API key (for --provider openai)")
@click.option("--coding-timeout", default=300.0, type=float, help="Per-task timeout for the agentic coding executor (seconds)")
@click.option("--jobs", default=None, help="Comma-separated subset to run: pipeline,reasoning,session_review,realdata,embedding,coding (default: all). Scope local-background comparisons by dropping the slow cloud-routed 'coding' suite.")
@click.option("--timeout", "timeout_override", default=None, type=float, help="Per-LLM-call timeout in seconds (default: config llm.timeout). Raise it for a slow local model — a timed-out case is DROPPED from the judged score, which biases the result upward.")
@click.option("--context-tokens", default=None, type=int, help="num_ctx for the candidate (default: the configured endpoint's window). Lower it if generation is slow: a large KV cache can spill to CPU and make every call many times slower.")
@click.pass_context
def benchmark_run_cmd(ctx, model, judge, provider, url, api_key_env, coding_timeout, jobs,
                      timeout_override, context_tokens):
    """Run the deep test of MODEL (e.g. qwen3:14b, minimax/minimax-m3).

    Tests every job (ranking, importance, contradiction, entity, summarization,
    lessons, reasoning, coding-gen, agentic coding, tool-calling, session review,
    embedding) for ability and speed, then updates data/benchmark/report.md.
    Full run is intentionally heavy (~30-90 min); use --jobs to scope. Cloud
    candidate: --provider openai --url <api-base> --api-key-env <ENV_VAR>.
    """
    from blipshell.benchmark.runner import run_benchmark
    job_set = {j.strip() for j in jobs.split(",") if j.strip()} if jobs else None
    asyncio.run(run_benchmark(
        model, config_path=ctx.obj.get("config_path"),
        judge_enabled=judge, provider=provider, url=url,
        api_key_env=api_key_env, coding_timeout=coding_timeout,
        jobs=job_set, timeout_override=timeout_override,
        context_tokens=context_tokens,
    ))


@benchmark_grp.command("report")
@click.pass_context
def benchmark_report_cmd(ctx):
    """Regenerate the shareable report (data/benchmark/report.md + .json) from all
    stored runs, without re-running any model."""
    from blipshell.benchmark.runner import run_report
    asyncio.run(run_report(config_path=ctx.obj.get("config_path")))


@benchmark_grp.command("discover")
@click.option("--min-context", type=int, default=None, help="Drop models below this context window")
@click.option("--max-price", type=float, default=None, help="Drop models above this $/1M prompt tokens")
@click.option("--vision", is_flag=True, help="Only vision-capable models")
@click.pass_context
def benchmark_discover_cmd(ctx, min_context, max_price, vision):
    """Pull the candidate shortlist from OpenRouter + Artificial Analysis."""
    from blipshell.benchmark.runner import run_discover
    asyncio.run(run_discover(
        config_path=ctx.obj.get("config_path"),
        min_context=min_context, max_price=max_price, vision=vision,
    ))


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
@click.option("--fix-pii-embeds", is_flag=True,
              help="Re-embed memories whose summaries were PII-sanitized ([PERSON]/[PII]).")
@click.option("--dry-run", is_flag=True, help="Show counts without making changes.")
@click.option("--all", "do_all", is_flag=True,
              help="Run all repairs.")
@click.pass_context
def repair_cmd(ctx, restore_imports, sweep_orphans, fix_sessions, fix_pii_embeds, dry_run, do_all):
    """Repair common DB issues.

    --restore-imports unarchives memories from imported sessions.
    --sweep-orphans removes orphan vector rows.
    --fix-sessions fixes sessions where end_session() failed (count=0, no title).
    --fix-pii-embeds re-embeds memories with PII-sanitized summaries.
    """
    if not (restore_imports or sweep_orphans or fix_sessions or fix_pii_embeds or do_all):
        console.print("[yellow]Nothing to do. Pass --restore-imports, --sweep-orphans, --fix-sessions, --fix-pii-embeds, or --all.[/yellow]")
        return
    if do_all:
        restore_imports = True
        sweep_orphans = True
        fix_sessions = True
        fix_pii_embeds = True

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

            if fix_pii_embeds:
                # Count affected
                cursor = await sqlite._db.execute(
                    "SELECT COUNT(*) FROM memories WHERE is_archived=0 AND content IS NOT NULL AND (summary LIKE '%[PERSON]%' OR summary LIKE '%[PII]%')"
                )
                row = await cursor.fetchone()
                count = row[0] if row else 0
                console.print(f"[cyan]Memories with PII-sanitized summaries:[/cyan] [bold]{count}[/bold]")

                if count and not dry_run:
                    console.print("[cyan]Re-embedding from raw content (this may take a minute)...[/cyan]")
                    result = vectors.re_embed_pii_damaged()
                    console.print(
                        f"[green]Re-embedded:[/green] "
                        f"{result['succeeded']} succeeded, {result['failed']} failed"
                    )
                elif dry_run:
                    console.print("[dim](dry-run; no changes)[/dim]")
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
    from blipshell.import_chatgpt import parse_conversations as _parse
    from blipshell.ui.importers import run_import

    run_import(
        config_path=ctx.obj.get("config_path"),
        parse=_parse,
        source=file,
        operation="import-chatgpt",
        max_count=max_count,
        skip_lessons=skip_lessons,
    )


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
    from blipshell.import_claude import parse_conversations as _parse
    from blipshell.ui.importers import run_import

    run_import(
        config_path=ctx.obj.get("config_path"),
        parse=_parse,
        source=file,
        operation="import-claude",
        max_count=max_count,
        skip_lessons=skip_lessons,
    )


@import_claude_group.command("scraped")
@click.argument("file", type=click.Path(exists=True))
@click.option("--max", "max_count", type=int, default=None,
              help="Only import the first N conversations (for testing)")
@click.option("--skip-lessons", is_flag=True, help="Skip lesson extraction (faster)")
@click.pass_context
def claude_scraped(ctx, file, max_count, skip_lessons):
    """Import conversations from a scraped Claude conversations_export.json."""
    from blipshell.import_claude import parse_scraped_conversations as _parse
    from blipshell.ui.importers import run_import

    run_import(
        config_path=ctx.obj.get("config_path"),
        parse=_parse,
        source=file,
        operation="import-claude-scraped",
        max_count=max_count,
        skip_lessons=skip_lessons,
    )


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
    from blipshell.import_claude_code import parse_claude_code_sessions as _parse
    from blipshell.ui.importers import run_import

    run_import(
        config_path=ctx.obj.get("config_path"),
        parse=_parse,
        source=path,
        operation="import-claude-code",
        max_count=max_count,
        skip_lessons=skip_lessons,
        max_concurrent=max_concurrent,
    )


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
    from blipshell.import_deepseek import parse_conversations as _parse
    from blipshell.ui.importers import run_import

    run_import(
        config_path=ctx.obj.get("config_path"),
        parse=_parse,
        source=file,
        operation="import-deepseek",
        max_count=max_count,
        skip_lessons=skip_lessons,
    )


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
