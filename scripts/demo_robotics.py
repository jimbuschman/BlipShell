"""Live demo of the cube robotics reactive-provisioning loop.

Run on the Ollama PC (needs a working REASONING model). Boots a real Agent,
connects a virtual LED-matrix cube, and shows:

    1. the CapabilityProfile the LLM authored for the new cube,
    2. the behaviors that compiled into the rules engine,
    3. the cube tools that auto-registered,
    4. each authored behavior firing when its trigger event is published,
    5. everything unwinding on disconnect.

This is the validation gate the unit tests can't cover: whether a real model
authors sensible behaviors. The unit tests prove the plumbing; this proves the
model cooperates.

    python -m scripts.demo_robotics                 # default config
    python -m scripts.demo_robotics --config x.yaml
"""

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from blipshell.core.agent import Agent
from blipshell.core.config import ConfigManager
from blipshell.llm.router import TaskType
from blipshell.robotics import CapabilityRegistry
from blipshell.robotics.cubes import VirtualLEDMatrix
from blipshell.robotics.profile import CapabilityProfile, ProfileGenerator
from blipshell.robotics.rules import Behavior, BehaviorAction
from blipshell.robotics.trace import trace_behaviors

console = Console()


def _render_matrix(cube: VirtualLEDMatrix) -> str:
    """ASCII view of the matrix state — text cue or lit pixels."""
    if cube.last_text is not None:
        return f'text: "{cube.last_text}"'
    rows = ["".join("█" if v else "·" for v in row) for row in cube.frame]
    return "\n".join(rows)


async def run_inject_flaw_demo(agent: Agent) -> int:
    """Seed a known flash bug and show the live model catch and fix it.

    Bypasses normal authoring: hands the generator a deliberately broken profile
    (display "HI" then immediately clear — HI flashes for 0ms) so the trace ->
    review -> revise loop is exercised against the real model every run.
    """
    cube = VirtualLEDMatrix()
    registry = CapabilityRegistry()
    await registry.connect(cube)  # bare registry — no auto-authoring listener

    async def generate_fn(system: str, user: str) -> str:
        return await agent.router.generate(TaskType.TOOL_CALLING, user, system=system)

    generator = ProfileGenerator(generate_fn)
    meta = registry.get_metadata(cube.cube_id)

    flawed = CapabilityProfile(
        cube_id=cube.cube_id,
        semantic_role="status display (seeded with a deliberate flaw)",
        behaviors=[Behavior(
            trigger="user_present",
            intent="briefly greet the user, then return to idle",
            actions=[
                BehaviorAction(target=cube.cube_id, action="display_text", args={"text": "HI"}),
                BehaviorAction(target=cube.cube_id, action="clear", args={}),
            ],
        )],
    )

    console.print(Panel(
        'Seeded behavior: user_present → display_text "HI", then clear.\n'
        'Intent: "briefly greet the user". The clear runs instantly, so HI is '
        "never seen.",
        title="Inject-flaw — seeded bug", border_style="red",
    ))

    issues = await trace_behaviors(flawed.behaviors, registry)
    console.print(Panel(
        "\n".join(f"• {i.problem}" for i in issues) or "(tracer found nothing — unexpected)",
        title="Tracer — what it observed", border_style="yellow",
    ))

    console.print("[cyan]Asking the model to revise (live)...[/cyan]")
    fixed = await generator.revise_until_clean(flawed, meta, registry)

    console.print("[dim]revised behavior:[/dim]")
    console.print(JSON(json.dumps([b.model_dump() for b in fixed.behaviors])))

    status = ("[green]fixed — no observed problems remain[/green]"
              if not fixed.unresolved_issues
              else f"[red]still flawed after {fixed.revision_count} revision(s)[/red]")
    console.print(Panel(
        f"revisions: {fixed.revision_count}\n{status}",
        title="Self-correction result", border_style="magenta",
    ))

    # Fire the corrected behavior so you can see the greeting actually persist.
    await cube.invoke("clear", {})
    await registry.event_bus.publish("user_present", {})
    console.print(f"user_present → {_render_matrix(cube)}")
    return 0


async def run_demo(config_path: str, inject_flaw: bool = False) -> int:
    console.print("[bold cyan]Bootstrapping agent...[/bold cyan]")
    config_manager = ConfigManager(config_path)
    config = config_manager.load()
    agent = Agent(config, config_manager)
    await agent.initialize(on_status=lambda m: console.print(f"  [dim]{m}[/dim]"))

    # Memory processing is irrelevant to this demo — silence the worker so the
    # process exits cleanly afterward.
    if agent._memory_worker:
        agent._memory_worker = None

    try:
        if agent.robotics is None:
            console.print("[red]agent.robotics is None — robotics not wired.[/red]")
            return 1

        if inject_flaw:
            return await run_inject_flaw_demo(agent)

        core = agent.robotics
        cube = VirtualLEDMatrix()

        console.print()
        console.print(Panel(
            f"Connecting cube '{cube.cube_id}' ({cube.width}x{cube.height} LED matrix).\n"
            "The LLM will now be asked what it's for and to author behaviors...",
            title="Step 1 — Connect", border_style="cyan",
        ))

        await core.connect(cube)  # fires tool registration + LLM profile authoring

        # 1. The authored profile.
        profile = core.get_profile(cube.cube_id)
        if profile is None:
            console.print("[yellow]No profile authored (LLM call failed — check the "
                          "endpoint/model). Tools still registered.[/yellow]")
        else:
            if profile.revision_count:
                review = (f"[yellow]self-reviewed: {profile.revision_count} revision(s) "
                          f"after observing problems[/yellow]")
                if profile.unresolved_issues:
                    review += (f"\n[red]unresolved: {len(profile.unresolved_issues)} "
                               f"issue(s) — {profile.unresolved_issues[0]}[/red]")
                else:
                    review += "\n[green]all observed problems fixed[/green]"
            else:
                review = "[dim]no problems observed — no revision needed[/dim]"
            console.print(Panel(
                f"[bold]role:[/bold] {profile.semantic_role}\n"
                f"[bold]uses:[/bold] {', '.join(profile.intended_uses) or '(none)'}\n"
                f"[bold]guidance:[/bold] {profile.usage_guidance}\n"
                f"{review}",
                title="Step 2 — LLM-authored profile", border_style="green",
            ))
            console.print("[dim]behaviors:[/dim]")
            console.print(JSON(json.dumps([b.model_dump() for b in profile.behaviors])))

        # 2. Tools that registered.
        cube_tools = [n for n in agent.tool_registry.get_tool_names() if n.startswith("cube_")]
        console.print(Panel(
            "\n".join(cube_tools) or "(none)",
            title="Step 3 — Auto-registered LLM tools", border_style="blue",
        ))

        # 3. Fire each trigger the engine reacts to, show the matrix respond.
        triggers = sorted(core.rules.triggers)
        if not triggers:
            console.print("[yellow]No behaviors compiled — nothing to fire. "
                          "(Model authored no valid behaviors.)[/yellow]")
        else:
            console.print(Panel(
                f"Publishing each authored trigger: {', '.join(triggers)}",
                title="Step 4 — Fire events", border_style="cyan",
            ))
            for trigger in triggers:
                await cube.invoke("clear", {})  # reset between events
                await core.registry.event_bus.publish(trigger, {})
                table = Table(show_header=False, box=None)
                table.add_row(f"[bold]{trigger}[/bold] →", _render_matrix(cube))
                console.print(table)

        # 4. Disconnect and show teardown.
        await core.disconnect(cube.cube_id)
        remaining = [n for n in agent.tool_registry.get_tool_names() if n.startswith("cube_")]
        console.print(Panel(
            f"Disconnected. cube tools remaining: {len(remaining)} | "
            f"engine triggers: {len(core.rules.triggers)} | "
            f"profile: {core.get_profile(cube.cube_id)}",
            title="Step 5 — Disconnect unwinds everything", border_style="magenta",
        ))
        return 0
    finally:
        console.print("[dim]Cleaning up...[/dim]")
        try:
            await agent.end_session()
        except Exception as e:
            console.print(f"[dim]cleanup note: {e}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Live cube robotics demo")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--inject-flaw", action="store_true",
                        help="seed a known flash bug and show the model self-correct it")
    args = parser.parse_args()
    sys.exit(asyncio.run(run_demo(args.config, inject_flaw=args.inject_flaw)))


if __name__ == "__main__":
    main()
