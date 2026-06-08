"""Tests for the derived per-turn capability block (self-model anti-drift).

The block states image-input support from what's actually true this turn (the
active model's vision capability) instead of a hand-written prompt claim — the
fix for the class of bug where the model denied it could see images right after
using vision. Pure/deterministic; behavioral effect validated on the Ollama PC.
"""

import types

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.llm.model_settings import ModelSettingsRegistry
from blipshell.llm.router import TaskType


def _fake(active_project, chat_model="minimax-m3:cloud", coding_model="minimax/minimax-m3"):
    reg = ModelSettingsRegistry()
    reg.load({
        "minimax-m3": {"vision_capable": True},
        "minimax/minimax-m3": {"vision_capable": True},
        "gpt-oss": {"think": False},  # not vision
    })

    def get_model(task_type):
        return coding_model if task_type == TaskType.CODING else chat_model

    return types.SimpleNamespace(
        active_project=active_project,
        router=types.SimpleNamespace(get_model=get_model),
        model_settings=reg,
    )


def test_vision_model_states_supported():
    block = ChatMixin._build_capability_block(_fake(None))
    assert "SUPPORTED" in block
    assert "minimax-m3:cloud" in block
    assert "Never claim you can't receive images" in block


def test_text_only_model_states_not_available():
    block = ChatMixin._build_capability_block(_fake(None, chat_model="gpt-oss:latest"))
    assert "not available this turn" in block
    assert "text-only" in block
    assert "gpt-oss:latest" in block


def test_active_project_uses_coding_model():
    # active_project routes to the CODING model, which is vision-capable here.
    block = ChatMixin._build_capability_block(_fake({"name": "blipshell"}))
    assert "SUPPORTED" in block
    assert "minimax/minimax-m3" in block


def test_active_project_text_coding_model_not_available():
    block = ChatMixin._build_capability_block(
        _fake({"name": "x"}, coding_model="gpt-oss:latest")
    )
    assert "not available this turn" in block
