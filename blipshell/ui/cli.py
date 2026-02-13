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
import sys

import click
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from blipshell.core.agent import Agent
from blipshell.core.config import ConfigManager
from blipshell.models.session import MessageRole

console = Console()


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
        # Default: start chat
        asyncio.run(chat_loop(
            config_path=config_path,
            resume_last=resume_last,
            session_id=session_id,
            project=project,
        ))


async def chat_loop(
    config_path: str | None = None,
    resume_last: bool = False,
    session_id: int | None = None,
    project: str | None = None,
):
    """Main interactive chat loop."""
    # Load config
    config_manager = ConfigManager(config_path)
    config = config_manager.load()

    # Create agent
    agent = Agent(config, config_manager)
    await agent.initialize()

    # Determine session to start/resume
    resume_id = session_id
    if resume_last and not session_id:
        latest = await agent.sqlite.get_latest_session()
        if latest:
            resume_id = latest.id
            console.print(f"[dim]Resuming session #{latest.id}: {latest.title}[/dim]")

    sid = await agent.start_session(project=project, resume_session_id=resume_id)

    # Header
    console.print(Panel.fit(
        f"[bold cyan]BlipShell[/bold cyan] v0.1.0\n"
        f"Session #{sid}"
        + (f" | Project: {project}" if project else "")
        + f"\nType [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit",
        border_style="cyan",
    ))

    try:
        while True:
            # Notify about background tasks that finished
            await _check_completed_tasks(agent)

            try:
                user_input = console.input("[bold green]> [/bold green]").strip()
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
                    await _print_task_detail(agent, int(cmd[1]))
                    continue
                elif cmd[0] == "workflow":
                    await _handle_workflow_command(agent, cmd_args)
                    continue
                elif cmd[0] == "offload":
                    if len(cmd) < 2:
                        console.print("[yellow]Usage: /offload <task description>[/yellow]")
                        console.print("[dim]Example: /offload review this code for errors: ...[/dim]")
                    else:
                        offload_msg = user_input[len("/offload "):]
                        await _submit_offload(agent, offload_msg)
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

            # Stream response
            response_parts = []

            def on_token(token: str):
                response_parts.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()

            console.print()  # blank line before response
            response = await agent.chat(message, on_token=on_token, force_plan=force_plan)

            if not response_parts:
                # Response wasn't streamed (e.g., tool calls happened)
                console.print(Markdown(response))
            else:
                console.print()  # newline after streaming

            # Show which endpoint handled the request
            ep = agent.last_endpoint_used
            if ep and ep != "local":
                console.print(f"[dim]via {ep}[/dim]")

            console.print()  # spacing

    finally:
        console.print("\n[dim]Ending session...[/dim]")
        await agent.end_session()
        console.print("[dim]Session saved. Goodbye![/dim]")


async def _submit_offload(agent: Agent, message: str):
    """Submit a task to run on a remote endpoint in the background."""
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

    session_id = agent.session_manager.session_id if agent.session_manager else None

    # Truncate title for display
    title = message[:80] + ("..." if len(message) > 80 else "")

    task_id = await agent.background_manager.submit_task(
        title=title,
        task_type="custom",
        prompt=message,
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


def _print_help():
    """Print help for CLI commands."""
    console.print(Panel(
        "[bold]/quit[/bold]              - Exit BlipShell\n"
        "[bold]/status[/bold]            - Show agent status, endpoints, routing\n"
        "[bold]/memory[/bold]            - Show memory pool usage\n"
        "[bold]/save[/bold]              - Force save session to memory\n"
        "[bold]/offload <msg>[/bold]     - Run a task on the remote PC in the background\n"
        "[bold]/plan[/bold]              - Show current active plan\n"
        "[bold]/plans[/bold]             - List all plans for this session\n"
        "[bold]/tasks[/bold]             - Show background tasks\n"
        "[bold]/task <id>[/bold]         - Show background task detail\n"
        "[bold]/workflow list[/bold]     - List available workflows\n"
        "[bold]/workflow show <n>[/bold] - Show workflow steps\n"
        "[bold]/workflow run <n>[/bold]  - Run a workflow\n"
        "[bold]/help[/bold]              - Show this help\n\n"
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

    from blipshell.import_chatgpt import import_conversations, parse_conversations
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
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
            ollama_url=cfg.endpoints[0].url if cfg.endpoints else "http://localhost:11434",
        )
        chroma.initialize()

        endpoint_manager = EndpointManager(cfg.endpoints, cfg.llm)
        router = LLMRouter(cfg.models, endpoint_manager)

        # Import with progress bar
        with Progress(console=console) as progress:
            task = progress.add_task("Importing...", total=len(convs))

            def on_progress(idx, total, title):
                progress.update(task, completed=idx,
                                description=f"[cyan]{title[:40]}[/cyan]")

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

        # Print summary
        console.print()
        summary = Table(title="Import Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Count", justify="right")
        summary.add_row("Conversations imported", str(stats.conversations_imported))
        summary.add_row("Messages processed", str(stats.messages_processed))
        summary.add_row("Messages skipped (noise)", str(stats.messages_skipped_noise))
        summary.add_row("Lessons extracted", str(stats.lessons_extracted))
        console.print(summary)

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
            ollama_url=cfg.endpoints[0].url if cfg.endpoints else "http://localhost:11434",
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


if __name__ == "__main__":
    main()
