"""The self-architecture card (tools/architecture_tools.py) and prompt rule 8.

Together these are the 'guessing mitigation' completed: rule 8 makes
answers-from-context flag themselves and 'I don't know' first-class; the card
gives the model real access to its own scaffolding so it consults instead of
theorizing (its own diagnosis: 'the limitation isn't insight — it's access').
"""

import pytest

from blipshell.core.tools.architecture_tools import (
    DescribeArchitectureTool,
    build_card,
)
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
