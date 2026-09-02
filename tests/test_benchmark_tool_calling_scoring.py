"""Scoring logic for the tool-calling suite — logic only, no Ollama.

These guard two defects found on 2026-08-18 while auditing why minimax-m3
scored 0.667 against local qwen3:14b's 0.933:

  1. Single-turn exact-match scoring punished agentic behaviour. Asked to read
     worker.py, the model ran `find . -name "worker.py"` first — a correct
     opening move — and scored a miss. That mis-scoring WAS the metric's noise
     (sd ~0.09), large enough to drive a routing decision off a coin flip.
  2. Substring matching on shell commands inflated scores the other way:
     `find . -name "pytest.ini"` counted as "ran the test suite".
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_suite():
    spec = importlib.util.spec_from_file_location(
        "benchmark_reasoning_suite", ROOT / "tests" / "benchmark_reasoning.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


br = _load_suite()


# ---------------------------------------------------------------------------
# _command_invokes — mention vs invocation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command,expected", [
    ("pytest", "pytest"),
    ("python -m pytest", "pytest"),
    ("ls && pytest -q", "pytest"),
    ("cd repo && git status", "git status"),
    ("git status --short", "git status"),
])
def test_command_invokes_accepts_real_invocations(command, expected):
    assert br._command_invokes(command, expected) is True


@pytest.mark.parametrize("command,expected", [
    ('find . -name "pytest.ini"', "pytest"),
    ('ls -la && find . -name "pytest.ini" -o -name "tox.ini"', "pytest"),
    ('echo "git status"', "git status"),
    ("grep -r pytest.ini .", "pytest"),
])
def test_command_invokes_rejects_mere_mentions(command, expected):
    assert br._command_invokes(command, expected) is False


# ---------------------------------------------------------------------------
# _tool_call_matches
# ---------------------------------------------------------------------------

def test_tool_name_must_match():
    call = {"name": "list_directory", "args": {"path": "worker.py"}}
    assert br._tool_call_matches(call, "read_file", {"path": "worker.py"}) is False


def test_tuple_expected_tool_accepts_any_listed_name():
    """Read-before-edit is obedient to system prompt rule 5, not a failure."""
    read = {"name": "read_file", "args": {"path": "config.yaml"}}
    edit = {"name": "edit_file", "args": {"path": "config.yaml"}}
    other = {"name": "write_file", "args": {"path": "config.yaml"}}
    accepted = ("edit_file", "read_file")
    assert br._tool_call_matches(read, accepted, {"path": "config.yaml"}) is True
    assert br._tool_call_matches(edit, accepted, {"path": "config.yaml"}) is True
    assert br._tool_call_matches(other, accepted, {"path": "config.yaml"}) is False


def test_wrong_path_still_fails_under_tuple():
    """Accepting two tools must not weaken the argument check."""
    call = {"name": "read_file", "args": {"path": "somewhere_else.txt"}}
    assert br._tool_call_matches(call, ("edit_file", "read_file"), {"path": "config.yaml"}) is False


def test_loose_key_match_for_paths():
    call = {"name": "read_file", "args": {"file_path": "src/worker.py"}}
    assert br._tool_call_matches(call, "read_file", {"path": "worker.py"}) is True


def test_command_arg_uses_invocation_matching_not_substring():
    mention = {"name": "run_command", "args": {"command": 'find . -name "pytest.ini"'}}
    real = {"name": "run_command", "args": {"command": "python -m pytest"}}
    assert br._tool_call_matches(mention, "run_command", {"command": "pytest"}) is False
    assert br._tool_call_matches(real, "run_command", {"command": "pytest"}) is True


def test_missing_args_dict_fails():
    assert br._tool_call_matches({"name": "read_file", "args": None},
                                 "read_file", {"path": "worker.py"}) is False


def test_no_expected_args_only_checks_name():
    assert br._tool_call_matches({"name": "web_search", "args": {}}, "web_search", None) is True


# ---------------------------------------------------------------------------
# Fake workspace — the multi-turn loop depends on these being consistent
# ---------------------------------------------------------------------------

def test_fake_listing_contains_files_the_cases_ask_for():
    listing = br._fake_tool_result("list_directory", {"path": "."})
    assert "README.md" in listing and "worker.py" in listing


def test_fake_read_returns_content_and_reports_missing_files():
    assert "MemoryWorker" in br._fake_tool_result("read_file", {"path": "worker.py"})
    assert "no such file" in br._fake_tool_result("read_file", {"path": "absent.py"})


def test_fake_read_is_case_and_prefix_insensitive():
    """Models ask for ./README.md, README.md and readme.md interchangeably."""
    for path in ("README.md", "./README.md", "readme.md"):
        assert "BlipShell" in br._fake_tool_result("read_file", {"path": path})


def test_fake_find_surfaces_paths_so_the_model_can_proceed():
    out = br._fake_tool_result("run_command", {"command": 'find . -name "worker.py"'})
    assert "worker.py" in out


def test_every_mock_tool_has_a_fake_result():
    for tool in br.MOCK_TOOLS:
        name = tool["function"]["name"]
        assert br._fake_tool_result(name, {}) != ""


# ---------------------------------------------------------------------------
# Suite data integrity
# ---------------------------------------------------------------------------

def test_expected_tools_all_exist_in_mock_tools():
    available = {t["function"]["name"] for t in br.MOCK_TOOLS}
    for case in br.TOOL_CALLING_TESTS:
        expected = case["expected_tool"]
        names = (expected,) if isinstance(expected, str) else tuple(expected)
        assert set(names) <= available, f"{case['name']} expects unknown tool {names}"


def test_suite_sends_the_production_system_prompt():
    """A bare user turn measures unprompted eagerness, which production never does."""
    from blipshell.models.config import AgentConfig
    assert br.TOOL_CALLING_SYSTEM == AgentConfig().system_prompt
    assert br.TOOL_CALLING_MAX_TURNS > 1


# ---------------------------------------------------------------------------
# Cross-stack response parsing (2026-09-02 regression)
#
# The suite used to carry its own tool-call extractor that did not parse
# OpenAI-style arguments-as-JSON-string. Every openai-provider candidate
# scored exactly 0.000 on tool_calling (deepseek-v4-flash, minimax-m3:free)
# while tool-calling fine through the same endpoint in production. The suite
# must parse responses with PRODUCTION's extractor, or it measures the
# parser, not the model.
# ---------------------------------------------------------------------------

def test_extractor_is_productions():
    from blipshell.core.chat_loop import extract_tool_call_info
    assert br.extract_tool_call_info is extract_tool_call_info


def test_openai_json_string_args_parse_to_dict():
    tc = {"id": "call_1", "type": "function",
          "function": {"name": "read_file", "arguments": '{"path": "worker.py"}'}}
    name, args, tc_id = br.extract_tool_call_info(tc)
    assert (name, args, tc_id) == ("read_file", {"path": "worker.py"}, "call_1")


class _ScriptedChatClient:
    """Returns openai_client._normalize_response-shaped dicts, one per turn."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.seen_messages = []

    async def chat(self, messages, model, tools=None, **kwargs):
        self.seen_messages.append([dict(m) for m in messages])
        return self._turns.pop(0)


class _FakeRouter:
    def __init__(self, client):
        self._client = client

    async def get_model_and_client(self, task_type):
        return "fake-model", self._client


def _openai_turn(name, arguments_json, tc_id):
    return {"message": {"content": "", "role": "assistant", "tool_calls": [
        {"id": tc_id, "type": "function",
         "function": {"name": name, "arguments": arguments_json}}]}}


_CASE = {
    "name": "read_worker",
    "message": "Read the contents of worker.py",
    "expected_tool": "read_file",
    "expected_args": {"path": "worker.py"},
}


async def test_openai_shaped_candidate_scores(monkeypatch):
    """The 0.000 bug, end to end: orient on turn 1, correct call on turn 2,
    args as JSON strings throughout — must score correct, and the fed-back
    tool message must carry tool_call_id (OpenAI-compat endpoints require it)."""
    monkeypatch.setattr(br, "TOOL_CALLING_TESTS", [_CASE])
    client = _ScriptedChatClient([
        _openai_turn("list_directory", '{"path": "."}', "call_a"),
        _openai_turn("read_file", '{"path": "worker.py"}', "call_b"),
    ])
    results = await br.benchmark_tool_calling(_FakeRouter(client))
    assert results[0]["correct"] is True
    assert results[0]["turns"] == 2
    # Turn 2's history must include the turn-1 tool result with its id.
    turn2_messages = client.seen_messages[1]
    tool_msgs = [m for m in turn2_messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_a"


async def test_ollama_shaped_candidate_still_scores(monkeypatch):
    """No regression on the Ollama path: dict args, no tool_call ids."""
    monkeypatch.setattr(br, "TOOL_CALLING_TESTS", [_CASE])
    client = _ScriptedChatClient([
        {"message": {"content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "worker.py"}}}]}},
    ])
    results = await br.benchmark_tool_calling(_FakeRouter(client))
    assert results[0]["correct"] is True
    assert results[0]["turns"] == 1
