"""The slash commands themselves.

Each handler is the body of one branch of cli.py's old if/elif ladder, moved
verbatim apart from reading its inputs off CommandContext instead of the
enclosing scope. Registration order is display order in /help.
"""

from __future__ import annotations

from rich.markup import escape
from rich.panel import Panel

from blipshell.ui.commands import QUIT, CommandContext, Rewrite, registry
from blipshell.ui.views import (
    _delete_core_item,
    _handle_compact,
    _handle_cube_command,
    _handle_code_command,
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
    _print_status,
    _print_task_detail,
    _print_thoughts,
    _print_tokens,
    _run_cleanup,
    _run_nightly,
    _save_feedback,
    _submit_offload,
)

cmd = registry.command


# ── Session ────────────────────────────────────────────────────────────────

@cmd("quit", "exit", "q", help="Exit BlipShell", section="Session")
def _quit(ctx: CommandContext):
    return QUIT


@cmd("status", help="Show agent status, endpoints, routing", section="Session")
def _status(ctx: CommandContext):
    _print_status(ctx.agent)


@cmd("memory", help="Show memory pool usage", section="Session")
def _memory(ctx: CommandContext):
    _print_memory_usage(ctx.agent)


@cmd("context", help="Show context window usage breakdown", section="Session")
def _context(ctx: CommandContext):
    _print_context(ctx.agent)


@cmd("tokens", help="Show token usage per endpoint this session", section="Session")
def _tokens(ctx: CommandContext):
    _print_tokens(ctx.agent)


@cmd("compact", usage="[focus]", help="Compact older messages to free context",
     section="Session")
async def _compact(ctx: CommandContext):
    await _handle_compact(ctx.agent, ctx.rest)


@cmd("notes", usage="[save|delete]",
     help="Session notes (survive compaction)", section="Session")
async def _notes(ctx: CommandContext):
    await _handle_notes_command(ctx.agent, ctx.args)


@cmd("save", help="Force save session to memory", section="Session")
async def _save(ctx: CommandContext):
    await ctx.agent.session_manager.dump_to_memory()
    ctx.console.print("[dim]Session dumped to memory.[/dim]")


@cmd("changes", help="Show files modified this session", section="Session")
def _changes(ctx: CommandContext):
    _print_changes(ctx.agent)


# ── Memory ─────────────────────────────────────────────────────────────────

@cmd("core", usage="[delete <id>]", help="Show core memories and lessons",
     section="Memory")
async def _core(ctx: CommandContext):
    if len(ctx.parts) >= 3 and ctx.parts[1] == "delete":
        await _delete_core_item(ctx.agent, ctx.parts[2:])
    else:
        await _print_core(ctx.agent)


@cmd("thoughts", help="Show lingering thoughts + gravity weights", section="Memory")
async def _thoughts(ctx: CommandContext):
    await _print_thoughts(ctx.agent)


@cmd("feedback", usage="<msg>", help="Save feedback as a lesson", section="Memory")
async def _feedback(ctx: CommandContext):
    if not ctx.rest:
        ctx.console.print("[yellow]Usage: /feedback <your feedback>[/yellow]")
        ctx.console.print("[dim]Example: /feedback too verbose, keep answers shorter[/dim]")
        return
    await _save_feedback(ctx.agent, ctx.rest)


@cmd("followups", "followup", help="Show pending follow-up items", section="Memory")
async def _followups(ctx: CommandContext):
    await _print_followups(ctx.agent)


@cmd("friction", usage="[all]", help="Show system friction log", section="Memory")
async def _friction(ctx: CommandContext):
    await _print_friction(ctx.agent, show_all=ctx.arg(0) == "all")


# ── Behavior toggles ───────────────────────────────────────────────────────

def _toggle(current: bool, arg: str) -> bool:
    """`on`/`off` set explicitly; anything else flips."""
    if arg in ("on", "off"):
        return arg == "on"
    return not current


@cmd("think", usage="[on|off]", help="Toggle LLM thinking mode", section="Toggles")
def _think(ctx: CommandContext):
    ctx.agent.think_enabled = _toggle(ctx.agent.think_enabled, ctx.arg(0))
    state = "[green]ON[/green]" if ctx.agent.think_enabled else "[yellow]OFF[/yellow]"
    ctx.console.print(f"[dim]Thinking mode: {state}[/dim]")


@cmd("reflect", usage="[on|off]", help="Toggle self-reflection", section="Toggles")
def _reflect(ctx: CommandContext):
    ctx.agent.reflect_enabled = _toggle(ctx.agent.reflect_enabled, ctx.arg(0))
    state = "[green]ON[/green]" if ctx.agent.reflect_enabled else "[yellow]OFF[/yellow]"
    ctx.console.print(f"[dim]Self-reflection: {state}[/dim]")


@cmd("guardrails", usage="[on|off]", help="Toggle guardrails", section="Toggles")
def _guardrails(ctx: CommandContext):
    gr = ctx.config.guardrails
    gr.enabled = _toggle(gr.enabled, ctx.arg(0))
    state = "[green]ON[/green]" if gr.enabled else "[yellow]OFF[/yellow]"
    ctx.console.print(f"[dim]Guardrails: {state}[/dim]")
    if gr.enabled:
        features = [
            label for flag, label in (
                (gr.completion_audit, "completion audit"),
                (gr.correction_detector, "correction detector"),
                (gr.trajectory_monitor, "trajectory monitor"),
                (gr.context_pinning, "context pinning"),
                (gr.requirement_checklist, "requirement checklist"),
            ) if flag
        ]
        ctx.console.print(f"[dim]  Active: {', '.join(features)}[/dim]")


@cmd("why", help="Why did it bring that up? The last turn's retrieval trace",
     section="Memory")
async def _why(ctx: CommandContext):
    """Anti-confabulation: the ACTUAL trace of what memory search injected
    last turn — not the model's guess about its own attention."""
    trace = getattr(ctx.agent, "_last_retrieval_trace", None)
    if not trace:
        ctx.console.print("[dim]No retrieval trace yet — ask something first.[/dim]")
        return

    ctx.console.print(f"[bold]Retrieval trace[/bold] [dim]for:[/dim] {trace['query']}")
    injected = trace.get("injected") or []
    if not injected:
        ctx.console.print(
            "[dim]Nothing was injected — the model answered from "
            "conversation context alone (or confabulated; now you know).[/dim]"
        )
    for item in injected:
        label = {"memory": "mem", "core": "CORE", "lesson": "lesson"}.get(
            item["source"], item["source"])
        ctx.console.print(
            f"  [cyan]{item['score']:<6}[/cyan] [dim]{label:<7}[/dim] "
            f"{item['preview']}"
        )

    stats = trace.get("stats") or {}
    parts = []
    if stats.get("entity_names"):
        parts.append(f"entities matched: {', '.join(stats['entity_names'])}")
    for key, label in (("chroma_hits", "vector"), ("fts_hits", "keyword"),
                       ("floor_dropped", "dropped at floor"),
                       ("dedup_dropped", "deduped")):
        if stats.get(key):
            parts.append(f"{label} {stats[key]}")
    if parts:
        ctx.console.print(f"[dim]  {' | '.join(parts)}[/dim]")


@cmd("usermodel", help="Show BlipShell's working model of you", section="Memory")
async def _usermodel(ctx: CommandContext):
    from blipshell.memory.user_model import UserModel

    um = UserModel(ctx.agent.sqlite, ctx.agent.router)
    doc = await um.get()
    if not doc:
        ctx.console.print(
            "[dim]No user model yet — the nightly `update_user_model` job "
            "builds it from session reflections.[/dim]"
        )
        return
    updated = await um.updated_at()
    ctx.console.print("[bold]Working model of you[/bold]"
                      + (f" [dim](evidence through {updated[:10]})[/dim]" if updated else ""))
    ctx.console.print(doc)
    ctx.console.print(
        "[dim]Revised nightly from session reflections, local model only. "
        "Conclusions, not facts — tell me where it's wrong.[/dim]"
    )


@cmd("local", "cloud", usage="[on|off]", section="Toggles",
     help="Local mode: no call leaves this machine (/cloud = /local off)")
def _local(ctx: CommandContext):
    em = ctx.agent.endpoint_manager
    if ctx.name == "cloud":
        em.local_only = False
    else:
        em.local_only = _toggle(em.local_only, ctx.arg(0))
    if em.local_only:
        ctx.console.print(
            "[green]Local mode ON[/green] [dim]— cloud endpoints are invisible "
            "to routing; every call (chat and background) runs on this "
            "machine. Expect the local model, not minimax.[/dim]"
        )
        ctx.console.print(
            "[dim]Boundary to know: if you switch back with /local off before "
            "this session closes, the session-close review of the WHOLE "
            "session (including this part) may run on a cloud model, "
            "identity-scrubbed. Stay local until exit for a fully local "
            "session.[/dim]"
        )
    else:
        ctx.console.print(
            "[yellow]Local mode OFF[/yellow] [dim]— cloud endpoints are back "
            "in the rotation.[/dim]"
        )


@cmd("verbose", help="Toggle verbose tool output", section="Toggles")
def _verbose(ctx: CommandContext):
    ctx.ui.verbose_tools = not ctx.ui.verbose_tools
    state = "[green]ON[/green]" if ctx.ui.verbose_tools else "[yellow]OFF[/yellow]"
    ctx.console.print(f"[dim]Verbose tool output: {state}[/dim]")


@cmd("expand", usage="[n]", help="Show full output of last n tool batches",
     section="Toggles")
def _expand(ctx: CommandContext):
    # Formatter lives with cli.py's tool-display code, which stayed put; a
    # top-level import would be circular (cli imports this module).
    from blipshell.ui.cli import _format_tool_arg_summary

    history = ctx.ui.tool_batch_history
    if not history:
        ctx.console.print("[dim]No tool batches to show.[/dim]")
        return
    n = 1
    if ctx.args:
        try:
            n = int(ctx.args[0])
        except ValueError:
            pass
    batches = history[-n:]
    for batch_idx, (calls, results) in enumerate(batches):
        for (name, args), (result, blocked) in zip(calls, results):
            if blocked:
                ctx.console.print(f"[dim]  {name}: [duplicate blocked][/dim]")
                continue
            arg_summary = _format_tool_arg_summary(name, args)
            style = "red" if not result.success else "bold"
            ctx.console.print(f"  [{style}]{name}[/{style}] {arg_summary}", highlight=False)
            if result.result:
                ctx.console.print(Panel(result.result[:2000], border_style="dim", expand=False))
        if batch_idx < len(batches) - 1:
            ctx.console.print("[dim]---[/dim]")


@cmd("approve", usage="[all|reset]", help="Manage tool approval", section="Toggles")
def _approve(ctx: CommandContext):
    approved = ctx.ui.session_approved_tools
    sub = ctx.arg(0)
    if sub == "all":
        for t in ctx.config.agent.tools_requiring_approval:
            approved.add(t)
        ctx.console.print("[dim]All tools auto-approved for this session[/dim]")
    elif sub == "reset":
        approved.clear()
        ctx.console.print("[dim]Tool approvals reset — will prompt again[/dim]")
    else:
        listed = ", ".join(sorted(approved)) if approved else "none"
        requiring = ", ".join(ctx.config.agent.tools_requiring_approval)
        ctx.console.print(f"[dim]Tools requiring approval: {requiring}[/dim]")
        ctx.console.print(f"[dim]Session-approved: {listed}[/dim]")
        ctx.console.print("[dim]  /approve all   — auto-approve all for this session[/dim]")
        ctx.console.print("[dim]  /approve reset — reset all approvals[/dim]")


# ── Work ───────────────────────────────────────────────────────────────────

@cmd("research", usage="<query>", help="Deep research with web + code exploration",
     section="Work")
def _research(ctx: CommandContext):
    if not ctx.rest:
        ctx.console.print("[yellow]Usage: /research <question or topic>[/yellow]")
        ctx.console.print("[dim]Triggers deep research with web search and thorough exploration.[/dim]")
        return
    # Handled by the normal message path, same as typing "!research ..."
    return Rewrite("!research " + ctx.rest)


@cmd("code", usage="<path> [msg]", help="Send code to LLM for review", section="Work")
async def _code(ctx: CommandContext):
    if not ctx.rest:
        ctx.console.print("[yellow]Usage: /code [--model name] <file-or-folder> [instruction][/yellow]")
        ctx.console.print("[dim]Examples:[/dim]")
        ctx.console.print("[dim]  /code blipshell/core/agent.py[/dim]")
        ctx.console.print("[dim]  /code blipshell/core/ find potential bugs[/dim]")
        ctx.console.print("[dim]  /code --model gemma3:4b tests/benchmark_agent_buggy.py[/dim]")
        return
    await _handle_code_command(ctx.agent, ctx.rest)


@cmd("offload", usage="<msg>", help="Run a task on remote PC in background",
     section="Work")
async def _offload(ctx: CommandContext):
    if not ctx.rest:
        ctx.console.print("[yellow]Usage: /offload <task description>[/yellow]")
        ctx.console.print("[dim]Example: /offload review this code for errors: ...[/dim]")
        return
    await _submit_offload(ctx.agent, ctx.rest)


@cmd("plan", help="Show current active plan", section="Work")
async def _plan(ctx: CommandContext):
    await _print_active_plan(ctx.agent)


@cmd("plans", help="List all plans for this session", section="Work")
async def _plans(ctx: CommandContext):
    await _print_plans(ctx.agent)


@cmd("tasks", help="Show background tasks", section="Work")
async def _tasks(ctx: CommandContext):
    await _print_background_tasks(ctx.agent)


@cmd("task", usage="<id>", help="Show background task detail", section="Work")
async def _task(ctx: CommandContext):
    if len(ctx.parts) < 2:
        ctx.console.print("[yellow]Usage: /task <id> (e.g. /task 1)[/yellow]")
        return
    try:
        task_id = int(ctx.parts[1])
    except ValueError:
        ctx.console.print("[yellow]Usage: /task <id> (e.g. /task 1)[/yellow]")
        return
    await _print_task_detail(ctx.agent, task_id)


@cmd("workflow", usage="list|show|run", help="Workflow management", section="Work")
async def _workflow(ctx: CommandContext):
    await _handle_workflow_command(ctx.agent, ctx.args)


# ── Maintenance ────────────────────────────────────────────────────────────

@cmd("health", usage="[quick]", help="Database + endpoint health check",
     section="Maintenance")
async def _health(ctx: CommandContext):
    await _print_health(ctx.agent, ctx.config, quick=ctx.arg(0) == "quick")


@cmd("cleanup", help="Reprocess failed messages (relaxed timeouts)",
     section="Maintenance")
async def _cleanup(ctx: CommandContext):
    await _run_cleanup(ctx.agent)


@cmd("nightly", usage="[job|report]", help="Run nightly maintenance or show last report",
     section="Maintenance")
async def _nightly(ctx: CommandContext):
    if ctx.args and ctx.args[0] == "report":
        await _print_nightly_report(ctx.agent)
    else:
        await _run_nightly(ctx.agent, ctx.args[0] if ctx.args else None)


@cmd("flow", usage="[n]", help="Show conversation flow events", section="Maintenance")
async def _flow(ctx: CommandContext):
    turn = None
    if len(ctx.parts) > 1:
        try:
            turn = int(ctx.parts[1])
        except ValueError:
            ctx.console.print("[yellow]Usage: /flow [turn_number][/yellow]")
            return
    await _print_flow(ctx.agent, turn)


@cmd("cube", usage="[id|reauthor|disconnect]",
     help="Cubes: list, inspect, re-author, disconnect", section="Maintenance")
async def _cube(ctx: CommandContext):
    await _handle_cube_command(ctx.agent, ctx.args)


# ── Projects ───────────────────────────────────────────────────────────────

@cmd("projects", help="List all projects", section="Projects")
async def _projects(ctx: CommandContext):
    await _list_projects(ctx.agent)


@cmd("project", usage="<name>|new|info|off|delete|digest",
     help="Activate, create, inspect or deactivate a project", section="Projects")
async def _project(ctx: CommandContext):
    await _handle_project_command(ctx.agent, ctx.args)


# ── Help ───────────────────────────────────────────────────────────────────

@cmd("help", "commands", help="Show this help", section="Help")
def _help(ctx: CommandContext):
    ctx.console.print(render_help())


def render_help() -> Panel:
    """Build the help panel FROM the registry.

    The old version was a hand-written string listing every command, which
    could (and did) drift out of step with what was actually dispatchable.
    """
    sections: dict[str, list] = {}
    for command in registry.all():
        if not command.hidden:
            sections.setdefault(command.section, []).append(command)

    # Cap the label column: one long usage string shouldn't push every
    # description off the right edge of the panel.
    MAX_LABEL = 26
    width = min(
        MAX_LABEL,
        max((len(c.render_label()) for cmds in sections.values() for c in cmds),
            default=20),
    )
    lines: list[str] = []
    order = ["Session", "Memory", "Toggles", "Work", "Maintenance", "Projects", "Help"]
    for name in order + [s for s in sections if s not in order]:
        if name not in sections:
            continue
        if lines:
            lines.append("")
        lines.append(f"[bold cyan]{name}[/bold cyan]")
        for command in sections[name]:
            label = command.render_label()
            # Usage strings like "[on|off]" are literal text, but Rich reads
            # square brackets as markup tags — unescaped, they vanish and the
            # column alignment goes with them.
            shown = escape(label)
            alias = ""
            if len(command.names) > 1:
                alias = "  [dim](" + ", ".join(f"/{n}" for n in command.names[1:]) + ")[/dim]"
            pad = " " * max(1, width - len(label))
            lines.append(f"[bold]{shown}[/bold]{pad}  {command.help}{alias}")

    lines.append("")
    lines.append("[dim]Press [bold]Esc[/bold] during a response to cancel the LLM call[/dim]")
    lines.append("[dim]Prefix with !plan to force planning: !plan <message>[/dim]")
    return Panel("\n".join(lines), title="Commands", border_style="blue")
