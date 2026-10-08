"""The self-architecture card (tools/architecture_tools.py) and prompt rule 8.

Together these are the 'guessing mitigation' completed: rule 8 makes
answers-from-context flag themselves and 'I don't know' first-class; the card
gives the model real access to its own scaffolding so it consults instead of
theorizing (its own diagnosis: 'the limitation isn't insight — it's access').
"""

import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from blipshell.core.tools.architecture_tools import (
    DescribeArchitectureTool,
    build_card,
)
from blipshell.core.tools.runtime_tools import InspectRuntimeTool
from blipshell.models.config import AgentConfig, BlipShellConfig


def test_rule8_uncertainty_discipline_in_system_prompt():
    prompt = AgentConfig().system_prompt
    assert "'I don't know' is a complete answer" in prompt
    assert "never construct a plausible-sounding one" in prompt
    assert "from the digest, unverified" in prompt


def test_card_reflects_live_config():
    config = BlipShellConfig()
    card = build_card(config)
    # Core mechanisms named, in plain language.
    for phrase in ("lingering thought", "NOT labeled", "display-only",
                   "user model", "WHAT YOU CANNOT SEE"):
        assert phrase in card, phrase
    # Config-derived facts: gravity is on in production config defaults? The
    # DEFAULT is off — the card must say so rather than tell a stale story.
    assert "gravity" in card.lower()
    config.reflection.gravity_enabled = True
    assert "recurrence reinforces" in build_card(config)
    config.reflection.gravity_enabled = False
    assert "currently disabled" in build_card(config)
    # Handoff line tracks its toggle.
    config.handoff.enabled = True
    assert "handoff note" in build_card(config)
    config.handoff.enabled = False
    assert "disabled" in build_card(config)


@pytest.mark.asyncio
async def test_tool_is_read_only_and_returns_card():
    config = BlipShellConfig()
    tool = DescribeArchitectureTool(config)
    assert tool.read_only is True
    assert tool.definition().name == "describe_architecture"
    result = await tool.execute()
    assert "YOUR ARCHITECTURE" in result


def test_runtime_questions_are_routed_to_live_inspection():
    prompt = AgentConfig().system_prompt
    assert "use inspect_runtime first" in prompt
    assert "Do not guess a path" in prompt


@pytest.mark.asyncio
async def test_runtime_tool_reports_authoritative_paths_scheduler_and_history():
    config = BlipShellConfig()
    existing_path = Path(__file__).resolve()
    config.database.path = str(existing_path)
    now = 1_800_000_000.0
    metadata = {
        "nightly_last_run": json.dumps({
            "completed_at": now, "status": "completed", "elapsed_s": 42.5,
        }),
        "nightly_report": json.dumps({
            "job_statuses": {"ok": 2, "error": 0, "timeout": 0},
            "warnings": [], "errors": [],
        }),
        "nightly_run_history": json.dumps([{
            "completed_at": now, "status": "completed",
            "tagging": {"before": {"pending": 5}, "after": {"pending": 0}},
        }]),
    }
    sqlite = NS(get_metadata=AsyncMock(side_effect=lambda key: metadata.get(key)))
    manager = NS(config_path=existing_path)
    tool = InspectRuntimeTool(
        config, manager, sqlite,
        lambda: {"nightly_scheduler": "running", "active_project": None},
    )

    assert tool.read_only is True
    assert tool.definition().name == "inspect_runtime"
    result = await tool.execute()
    assert str(existing_path) in result
    assert "Nightly scheduler: running" in result
    assert "2/2 jobs ok" in result
    assert "tagging pending 5->0" in result
