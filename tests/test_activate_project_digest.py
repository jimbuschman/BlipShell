"""Regression tests for the activate_project digest-shape bug.

ProjectDigestManager._save_digest stores the digest as a plain string, but
ActivateProjectTool.execute used to assume a {"content": ...} dict and called
.get("content") on it — raising "'str' object has no attribute 'get'" for any
project that actually had a digest. These tests pin down that activation works
for string digests (the real shape), dict digests (legacy/edge shape), and the
no-digest case.
"""

import json

import pytest

from blipshell.core.tools.project_tools import ActivateProjectTool


def _make_tool(project: dict) -> ActivateProjectTool:
    async def callback(name: str) -> dict:
        return project

    return ActivateProjectTool(callback=callback)


@pytest.mark.asyncio
async def test_activate_with_string_digest():
    """The real storage shape: digest is a plain string. Must not crash."""
    project = {
        "name": "blipshell",
        "root_path": "/repo/blipshell",
        "language": "python",
        "description": "Local LLM assistant",
        "metadata_json": json.dumps({"digest": "Project does X, Y, Z."}),
    }
    tool = _make_tool(project)

    result = await tool.execute(name="blipshell")

    assert "Activated project 'blipshell'" in result
    assert "Project does X, Y, Z." in result


@pytest.mark.asyncio
async def test_activate_with_dict_digest_legacy():
    """Legacy/edge shape: digest stored as {"content": ...}. Still supported."""
    project = {
        "name": "blipshell",
        "root_path": "/repo/blipshell",
        "language": "python",
        "description": "",
        "metadata_json": json.dumps({"digest": {"content": "Legacy digest text."}}),
    }
    tool = _make_tool(project)

    result = await tool.execute(name="blipshell")

    assert "Activated project 'blipshell'" in result
    assert "Legacy digest text." in result


@pytest.mark.asyncio
async def test_activate_with_no_digest():
    """No digest present: activation succeeds with no digest section."""
    project = {
        "name": "blipshell",
        "root_path": "/repo/blipshell",
        "language": "python",
        "description": "desc",
        "metadata_json": json.dumps({}),
    }
    tool = _make_tool(project)

    result = await tool.execute(name="blipshell")

    assert "Activated project 'blipshell'" in result
    assert "Digest:" not in result


@pytest.mark.asyncio
async def test_activate_with_empty_metadata():
    """metadata_json is None/empty: must not crash on json.loads or .get."""
    project = {
        "name": "blipshell",
        "root_path": "/repo/blipshell",
        "language": "",
        "description": "",
        "metadata_json": None,
    }
    tool = _make_tool(project)

    result = await tool.execute(name="blipshell")

    assert "Activated project 'blipshell'" in result


@pytest.mark.asyncio
async def test_activate_unknown_project_returns_error():
    """A KeyError from the callback becomes a friendly not-found message."""
    async def callback(name: str) -> dict:
        raise KeyError(name)

    tool = ActivateProjectTool(callback=callback)

    result = await tool.execute(name="ghost")

    assert "not found" in result.lower()
