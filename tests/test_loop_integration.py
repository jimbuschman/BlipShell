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


# --- guardrails on the SIMPLE chat path -----------------------------------


async def _run_no_completion_tool(script, registry, guardrails=None, budget=20):
    """Drive the loop the way _chat_simple does: no completion tool, so the
    text response terminates and the completion audit never applies."""
    client = ScriptedLLMClient(script)
    loop = ChatLoop(registry)
    config = LoopConfig(
        budget=budget,
        enable_compaction=False,
        guardrails=guardrails,
    )
    result = await loop.run(
        client=client, messages=_messages(), model="fake",
        tools=None, chat_kwargs={}, config=config,
    )
    return result, client


@pytest.mark.asyncio
async def test_doom_loop_fires_on_the_simple_chat_path():
    """Until 2026-08 only the executor attached a GuardrailsEngine, so the
    doom-loop detector — pure counters, zero LLM cost — was off for ~all real
    traffic. A model re-reading the same file got no correction."""
    router = RecordingRouter()
    engine = GuardrailsEngine(
        GuardrailsConfig(doom_loop_detector=True, doom_loop_read_threshold=3,
                         trajectory_monitor=False),
        router,
    )
    engine.original_request = "look at config.yaml"
    reg = make_registry(FakeTool("read_file", read_only=True, result="file body"))

    same_read = {"tools": [("read_file", {"path": "config.yaml"})]}
    result, _ = await _run_no_completion_tool(
        [same_read, same_read, same_read, same_read, {"text": "done looking"}],
        reg, guardrails=engine,
    )

    assert result.response == "done looking"
    assert engine._file_read_counts["config.yaml"] >= 3
    assert engine._doom_warnings_sent, (
        "doom-loop detector never fired on the simple chat path"
    )
    assert router.calls == 0, "the doom-loop check must cost no LLM calls"


@pytest.mark.asyncio
async def test_no_guardrails_means_no_doom_loop_correction():
    """Pins what the bug was: with guardrails absent the same behavior passes
    unremarked."""
    reg = make_registry(FakeTool("read_file", read_only=True, result="file body"))
    same_read = {"tools": [("read_file", {"path": "config.yaml"})]}
    result, _ = await _run_no_completion_tool(
        [same_read, same_read, same_read, same_read, {"text": "done"}],
        reg, guardrails=None,
    )
    assert result.response == "done"     # no correction, no error


@pytest.mark.asyncio
async def test_chat_guardrails_disable_trajectory_injection():
    """Chat gets the deterministic guardrails but NOT the synthetic
    "[CHECKPOINT n/N]" injection, which is built for long unsupervised runs."""
    router = RecordingRouter()
    engine = GuardrailsEngine(
        GuardrailsConfig(trajectory_monitor=False, monitor_interval=2), router,
    )
    engine.original_request = "do a few things"
    assert engine.build_trajectory_injection(2, 20, ["read_file", "read_file"]) is None


@pytest.mark.asyncio
async def test_chat_guardrails_builder_respects_master_toggle():
    """_build_chat_guardrails returns None when guardrails are off entirely,
    and when there are no tools (budget 0) there is nothing to guard."""
    from blipshell.core.agent_chat import ChatMixin

    class _Host(ChatMixin):
        def __init__(self, cfg):
            self.config = cfg
            self.router = RecordingRouter()

    class _Cfg:
        def __init__(self, **kw):
            self.guardrails = GuardrailsConfig(**kw)

    assert _Host(_Cfg(enabled=False))._build_chat_guardrails("hi", 20) is None
    assert _Host(_Cfg())._build_chat_guardrails("hi", 0) is None

    engine = _Host(_Cfg())._build_chat_guardrails("fix the bug", 20)
    assert engine is not None
    assert engine.original_request == "fix the bug"
    assert engine.config.trajectory_monitor is False   # chat override
    assert engine.config.doom_loop_detector is True    # kept


# --- outbound transform (credential stripping at the wire) -----------------


@pytest.mark.asyncio
async def test_outbound_transform_scrubs_the_wire_not_the_history():
    """The interactive path bypasses router.generate(), so until 2026-08
    nothing on it was sanitized despite pii_sanitize: true on the cloud
    endpoints. The transform must reach the wire while leaving the loop's own
    messages — which become conversation history and stored memory — intact."""
    from blipshell.llm.pii import sanitize_messages_secrets

    secret = "ghp_" + "z" * 36
    reg = make_registry(FakeTool("read_file", read_only=True, result="ok"))
    client = ScriptedLLMClient([{"text": "understood"}])
    loop = ChatLoop(reg)
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": f"deploy using {secret}"},
    ]
    config = LoopConfig(
        budget=5, enable_compaction=False,
        outbound_transform=sanitize_messages_secrets,
    )

    result = await loop.run(
        client=client, messages=messages, model="fake",
        tools=None, chat_kwargs={}, config=config,
    )

    assert result.response == "understood"
    # What the model received: scrubbed
    wire = client.sent_messages[0]
    assert secret not in wire[1]["content"]
    assert "[API_KEY]" in wire[1]["content"]
    # What the session keeps: intact
    assert messages[1]["content"] == f"deploy using {secret}"
    assert secret in result.messages[1]["content"]


@pytest.mark.asyncio
async def test_no_transform_sends_text_unchanged():
    """Local endpoints get the raw text — that's the point of gating on
    should_sanitize_pii."""
    secret = "ghp_" + "y" * 36
    reg = make_registry(FakeTool("read_file", read_only=True, result="ok"))
    client = ScriptedLLMClient([{"text": "done"}])
    loop = ChatLoop(reg)
    messages = [{"role": "user", "content": f"token {secret}"}]

    await loop.run(
        client=client, messages=messages, model="fake",
        tools=None, chat_kwargs={}, config=LoopConfig(
            budget=5, enable_compaction=False, outbound_transform=None,
        ),
    )

    assert secret in client.sent_messages[0][0]["content"]


@pytest.mark.asyncio
async def test_transform_applies_to_every_turn_including_tool_results():
    """A credential can surface mid-loop in a tool result, so the transform
    has to run on each call, not just the first."""
    from blipshell.llm.pii import sanitize_messages_secrets

    secret = "AKIAABCDEFGHIJKLMNOP"
    reg = make_registry(
        FakeTool("read_file", read_only=True, result=f"config: KEY={secret}")
    )
    client = ScriptedLLMClient([
        {"tools": [("read_file", {"path": ".env"})]},
        {"text": "found it"},
    ])
    loop = ChatLoop(reg)

    await loop.run(
        client=client, messages=_messages(), model="fake",
        tools=None, chat_kwargs={}, config=LoopConfig(
            budget=5, enable_compaction=False,
            outbound_transform=sanitize_messages_secrets,
        ),
    )

    # Second call carries the tool result — it must be scrubbed on the wire
    assert len(client.sent_messages) >= 2
    second_turn = "\n".join(
        m.get("content") or "" for m in client.sent_messages[1]
    )
    assert secret not in second_turn
