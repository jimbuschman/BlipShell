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
from blipshell.robotics.cubes import VirtualLEDMatrix

console = Console()


def _render_matrix(cube: VirtualLEDMatrix) -> str:
    """ASCII view of the matrix state — text cue or lit pixels."""
    if cube.last_text is not None:
        return f'text: "{cube.last_text}"'
    rows = ["".join("█" if v else "·" for v in row) for row in cube.frame]
    return "\n".join(rows)


async def run_demo(config_path: str) -> int:
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
            console.print(Panel(
                f"[bold]role:[/bold] {profile.semantic_role}\n"
                f"[bold]uses:[/bold] {', '.join(profile.intended_uses) or '(none)'}\n"
                f"[bold]guidance:[/bold] {profile.usage_guidance}",
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
    args = parser.parse_args()
    sys.exit(asyncio.run(run_demo(args.config)))


if __name__ == "__main__":
    main()
