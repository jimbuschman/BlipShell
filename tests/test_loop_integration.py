"""End-to-end ChatLoop wiring tests — real loop, real ToolRegistry, real
GuardrailsEngine, scripted model (no Ollama).

These close the gap between unit tests (logic in isolation) and Ollama-PC
behavioral checks (model quality): they prove the *plumbing* works — completion
detection, the difficulty-gated audit, and the look-before-review gate all
behave correctly when driven through the actual loop.
"""

import pytest

from blipshell.core.chat_loop import ChatLoop, LoopConfig
from blipshell.core.guardrails import GuardrailsEngine
from blipshell.models.config import GuardrailsConfig
from tests.fakes import FakeTool, RecordingRouter, ScriptedLLMClient, make_registry


def _messages(system="You are a test assistant.", user="do the task"):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def _run(script, registry, guardrails=None, budget=20):
    client = ScriptedLLMClient(script)
    loop = ChatLoop(registry)
    config = LoopConfig(
        budget=budget,
        completion_tool="task_complete",
        enable_compaction=False,
        guardrails=guardrails,
    )
    result = await loop.run(
        client=client, messages=_messages(), model="fake",
        tools=None, chat_kwargs={}, config=config,
    )
    return result, client


@pytest.mark.asyncio
async def test_task_complete_terminates_loop():
    """A task_complete tool call ends the loop via the completion path."""
    reg = make_registry(FakeTool("task_complete", result="all done"))
    result, client = await _run(
        [{"tools": [("task_complete", {"summary": "all done", "files_modified": ""})]}],
        reg,
    )
    assert result.completion_method == "tool"
    assert "task_complete" in result.tool_call_names
    assert result.response == "all done"


@pytest.mark.asyncio
async def test_trivial_completion_skips_llm_audit():
    """THE core guardrails win, end-to-end: a trivial task (few tools, one file,
    no checklist) completes WITHOUT the audit ever calling the LLM."""
    router = RecordingRouter(result="PASS")
    engine = GuardrailsEngine(GuardrailsConfig(enabled=True), router=router)
    engine.original_request = "rename a variable"
    reg = make_registry(
        FakeTool("edit_file", result="edited"),
        FakeTool("task_complete", result="renamed it"),
    )
    result, _ = await _run(
        [
            {"tools": [("edit_file", {"path": "a.py"})]},
            {"tools": [("task_complete", {"summary": "renamed it", "files_modified": "a.py"})]},
        ],
        reg, guardrails=engine,
    )
    assert result.completion_method == "tool"
    assert router.calls == 0  # difficulty gate skipped the LLM audit


@pytest.mark.asyncio
async def test_nontrivial_completion_runs_llm_audit_once():
    """A non-trivial task (many tool calls) reaches the single LLM audit."""
    router = RecordingRouter(result="PASS")
    engine = GuardrailsEngine(GuardrailsConfig(enabled=True, completion_audit_min_tool_calls=3),
                              router=router)
    engine.original_request = "implement a feature across the module"
    reg = make_registry(
        FakeTool("read_file", read_only=True),
        FakeTool("edit_file", result="edited"),
        FakeTool("task_complete", result="done"),
    )
    result, _ = await _run(
        [
            {"tools": [("read_file", {"path": "a.py"})]},
            {"tools": [("edit_file", {"path": "a.py"})]},
            {"tools": [("read_file", {"path": "b.py"})]},
            {"tools": [("task_complete", {"summary": "done", "files_modified": "a.py"})]},
        ],
        reg, guardrails=engine,
    )
    assert result.completion_method == "tool"
    assert router.calls == 1  # crossed the difficulty threshold → one audit call


@pytest.mark.asyncio
async def test_look_before_review_gate_blocks_then_passes():
    """A review request that calls task_complete without reading is rejected by
    the deterministic gate; after a read, the second task_complete is accepted."""
    router = RecordingRouter(result="PASS")
    engine = GuardrailsEngine(GuardrailsConfig(enabled=True), router=router)
    engine.original_request = "review cli.py and tell me what could be improved"
    reg = make_registry(
        FakeTool("read_file", read_only=True),
        FakeTool("task_complete", result="here are the findings"),
    )
    result, _ = await _run(
        [
            # 1st: complete with no read → look-before-review gate fires, loop continues
            {"tools": [("task_complete", {"summary": "found issues X and Y", "files_modified": ""})]},
            # 2nd: actually read the code
            {"tools": [("read_file", {"path": "cli.py"})]},
            # 3rd: now complete — grounded
            {"tools": [("task_complete", {"summary": "after reading, issues are...", "files_modified": ""})]},
        ],
        reg, guardrails=engine,
    )
    assert result.completion_method == "tool"
    # The first task_complete must NOT have ended the loop — proves the gate fired.
    assert result.tool_call_names == ["task_complete", "read_file", "task_complete"]
