"""Tests for per-model behavioral settings."""

import pytest

from blipshell.llm.model_settings import ModelSettings, ModelSettingsRegistry


@pytest.fixture
def registry():
    """Registry with test settings loaded."""
    reg = ModelSettingsRegistry()
    reg.load({
        "qwen3-coder": {
            "max_tool_calls": 20,
            "use_repo_map": True,
            "extra_instructions": "Be concise.",
        },
        "gpt-oss": {
            "max_tool_calls": 10,
            "think": False,
        },
        "glm4": {
            "max_tool_calls": 5,
            "use_repo_map": False,
        },
    })
    return reg


def test_exact_match(registry):
    ms = registry.get("qwen3-coder")
    assert ms.max_tool_calls == 20
    assert ms.use_repo_map is True
    assert ms.extra_instructions == "Be concise."


def test_base_name_match(registry):
    """Model with tag/variant matches base name."""
    ms = registry.get("qwen3-coder:480b-cloud")
    assert ms.max_tool_calls == 20


def test_prefix_match(registry):
    """Model name starting with a known key matches."""
    ms = registry.get("gpt-oss:latest")
    assert ms.max_tool_calls == 10
    assert ms.think is False


def test_defaults_for_unknown(registry):
    """Unknown model gets sensible defaults."""
    ms = registry.get("some-unknown-model:v2")
    assert ms.max_tool_calls == 15  # default
    assert ms.use_repo_map is True
    assert ms.think is None
    assert ms.extra_instructions == ""


def test_has_settings(registry):
    assert registry.has_settings("qwen3-coder") is True
    assert registry.has_settings("qwen3-coder:480b-cloud") is True
    assert registry.has_settings("unknown-model") is False


def test_from_dict_ignores_unknown_keys():
    """Unknown keys in config are silently ignored."""
    ms = ModelSettings.from_dict({
        "max_tool_calls": 25,
        "unknown_key": "ignored",
        "another_unknown": 42,
    })
    assert ms.max_tool_calls == 25
    assert ms.use_repo_map is True  # default preserved


def test_empty_load():
    """Loading empty config produces empty registry."""
    reg = ModelSettingsRegistry()
    reg.load({})
    ms = reg.get("any-model")
    assert ms.max_tool_calls == 15  # default
