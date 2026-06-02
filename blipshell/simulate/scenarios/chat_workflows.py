"""Scenarios: Multi-turn conversation workflows.

Tests real conversations that exercise the chat pipeline end-to-end:
tool use, context carry-over, slash commands mid-conversation, memory search.
"""

from blipshell.simulate.models import SimScenario, SimStep, StepAction


def get_scenarios() -> list[SimScenario]:
    return [
        _basic_conversation(),
        _multi_tool_chat(),
        _slash_commands_mid_conversation(),
        _multi_turn_context(),
        _chat_after_mode_switch(),
        _memory_search_in_chat(),
        _force_plan_via_prefix(),
        _self_thought_resurfaces(),
    ]


# A distinctive self-thought to seed; the positive turn echoes it near-verbatim
# so the reranker reliably clears its floor, the negative turn is unrelated.
_SEED_THOUGHT = (
    "I keep wondering whether the modular cubes should express emotion "
    "through motion rather than through color."
)


def _thought_injected(agent) -> bool:
    return _SEED_THOUGHT in getattr(agent, "_relevance_injected_thoughts", set())


def _reranker_on(agent) -> bool:
    # The standing-injection gate is fail-closed: with no reranker, nothing
    # surfaces, so the positive check is not applicable (treated as pass).
    return bool(getattr(getattr(agent, "search", None), "reranker_enabled", False))


def _self_thought_resurfaces() -> SimScenario:
    """End-to-end: a self-originated lingering thought resurfaces as context
    when the conversation is near it, and stays silent when it isn't.

    Exercises the real path: store embedding -> cosine prefilter -> reranker
    gate -> [Thought] injection. The reranker score is the tuning knob, so the
    positive turn echoes the thought near-verbatim to keep the gate reliable;
    if no reranker is configured the positive check is skipped (fail-closed).
    """
    return SimScenario(
        name="self_thought_resurfaces_on_relevance",
        description="Lingering thought resurfaces when relevant, silent when not",
        category="chat_workflow",
        steps=[
            SimStep(
                action=StepAction.AGENT_METHOD,
                method="_self_thoughts.add",
                method_args={"text": _SEED_THOUGHT},
                description="Seed a self-originated lingering thought (embeds it)",
            ),
            SimStep(
                action=StepAction.CHAT,
                input="Should the modular cubes express emotion through motion rather than color?",
                description="On-topic turn — the thought should resurface (if reranker on)",
                expect_no_error=True,
                custom_validator=lambda agent: (
                    []
                    if (not _reranker_on(agent) or _thought_injected(agent))
                    else ["on-topic turn did not resurface the seeded self-thought "
                          f"(injected={getattr(agent, '_relevance_injected_thoughts', set())!r})"]
                ),
            ),
            SimStep(
                action=StepAction.CHAT,
                input="What's a good recipe for sourdough bread?",
                description="Off-topic turn — the thought must NOT leak in",
                expect_no_error=True,
                custom_validator=lambda agent: (
                    []
                    if not _thought_injected(agent)
                    else ["off-topic turn leaked an unrelated self-thought into context"]
                ),
            ),
        ],
    )


def _basic_conversation() -> SimScenario:
    """Multi-turn conversation — no tools, verify no crash or error."""
    return SimScenario(
        name="basic_conversation",
        description="Multi-turn chat with no tool use",
        category="chat_workflow",
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input="Hello, how are you?",
                description="Greeting",
                expect_no_tools=True,
                expect_no_error=True,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="What can you help me with?",
                description="Follow-up question",
                expect_no_tools=True,
                expect_no_error=True,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="Explain what a decorator is in Python, briefly.",
                description="Technical question",
                expect_no_tools=True,
                expect_no_error=True,
            ),
        ],
    )


def _multi_tool_chat() -> SimScenario:
    """Chat that triggers multiple different tools across turns."""
    return SimScenario(
        name="multi_tool_chat",
        description="Chat triggers read_file, grep_files, glob_files, list_directory across turns",
        category="chat_workflow",
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input="Read the file blipshell/__init__.py and tell me what's in it.",
                description="Trigger read_file",
                expect_tools=["read_file"],
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="Search for 'def chat' in the blipshell/core/ directory.",
                description="Trigger grep_files",
                expect_tools=["grep_files"],
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="Find all Python files in blipshell/simulate/scenarios/",
                description="Trigger glob_files",
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="List what's in the blipshell/core/tools/ directory.",
                description="Trigger list_directory",
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            # After 4 tool-using turns, verify state is still clean
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Tools still registered after heavy tool use",
                expect_tools_registered=["read_file", "grep_files", "glob_files", "list_directory"],
            ),
        ],
    )


def _slash_commands_mid_conversation() -> SimScenario:
    """Interleave chat and slash commands — verify nothing breaks."""
    return SimScenario(
        name="slash_commands_mid_conversation",
        description="Slash commands between chat messages don't break state",
        category="chat_workflow",
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input="Tell me about Python's asyncio module in 2 sentences.",
                description="Chat message",
                expect_no_error=True,
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/status",
                description="/status between messages",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/memory",
                description="/memory between messages",
            ),
            SimStep(
                action=StepAction.CHAT,
                input="Can you elaborate on the event loop part?",
                description="Follow-up should still have context",
                expect_no_error=True,
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/save",
                description="/save mid-conversation",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/flow",
                description="/flow should show recent turns",
            ),
            SimStep(
                action=StepAction.CHAT,
                input="Thanks, that makes sense.",
                description="Conversation continues after slash commands",
                expect_no_error=True,
            ),
        ],
    )


def _multi_turn_context() -> SimScenario:
    """Verify conversation context carries across turns."""
    return SimScenario(
        name="multi_turn_context_carry",
        description="Information from earlier turns is available in later turns",
        category="chat_workflow",
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input="My favorite color is blue. Remember that.",
                description="Tell a fact",
                expect_no_error=True,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="I also prefer dark mode in all my apps.",
                description="Tell another fact",
                expect_no_error=True,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="What's my favorite color?",
                description="Recall from context",
                expect_response_contains=["blue"],
                expect_no_error=True,
            ),
            # Also verify session has the expected message count
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Session should have 6+ messages (3 user + 3 assistant)",
                custom_validator=lambda agent: (
                    []
                    if agent.session_manager and agent.session_manager.message_count >= 6
                    else [f"Expected 6+ messages, got {agent.session_manager.message_count if agent.session_manager else 0}"]
                ),
            ),
        ],
    )


def _chat_after_mode_switch() -> SimScenario:
    """Chat, switch to project mode, chat, switch back, chat — all should work."""
    return SimScenario(
        name="chat_after_mode_switch",
        description="Chat works correctly across project mode transitions",
        category="chat_workflow",
        steps=[
            # Chat in simple mode
            SimStep(
                action=StepAction.CHAT,
                input="What is BlipShell?",
                description="Chat in simple mode",
                expect_no_error=True,
            ),
            # Switch to project mode
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Activate project",
                expect_project_active="blipshell",
            ),
            # Chat in project mode
            SimStep(
                action=StepAction.CHAT,
                input="Read blipshell/__init__.py",
                description="Chat with tool in project mode",
                expect_tools=["read_file"],
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            # Switch back
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Deactivate project",
                expect_project_inactive=True,
            ),
            # Chat in simple mode again
            SimStep(
                action=StepAction.CHAT,
                input="What did you just read?",
                description="Chat should still work after deactivation",
                expect_no_error=True,
            ),
            # Verify tools are correct
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Base tools intact after mode switch cycle",
                expect_tools_registered=["read_file", "grep_files", "glob_files"],
                expect_tools_not_registered=["git_status", "task_complete"],
            ),
        ],
    )


def _memory_search_in_chat() -> SimScenario:
    """Verify the search_memories tool works in simple chat."""
    return SimScenario(
        name="memory_search_in_chat",
        description="search_memories tool is callable from simple chat",
        category="chat_workflow",
        steps=[
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="search_memories tool registered",
                expect_tools_registered=["search_memories"],
            ),
            SimStep(
                action=StepAction.CHAT,
                input="Search your memories for anything related to 'Python programming'.",
                description="Trigger memory search via chat",
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
        ],
    )


def _force_plan_via_prefix() -> SimScenario:
    """Test the !plan prefix triggers executor path."""
    return SimScenario(
        name="force_plan_prefix",
        description="!plan prefix forces executor path and completes",
        category="chat_workflow",
        requires_project="blipshell",
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input="Read the file blipshell/simulate/models.py and tell me how many dataclasses it defines.",
                description="Force plan execution",
                force_plan=True,
                expect_tools=["read_file"],
                expect_no_error=True,
                timeout_seconds=180.0,
            ),
        ],
    )
