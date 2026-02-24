# BlipShell Executor Research Prompt

Paste this into Claude Code (running on your dev machine with real Claude) from the BlipShell project directory.

---

## The Prompt

I need you to do a deep research analysis of this project's executor/agent architecture. This is a local LLM coding agent (like a Claude Code clone) that uses Ollama with glm-5 for tool calling. Read CLAUDE.md first for full project context.

**Do NOT make any code changes.** This is research only. Write your findings to a file called `research_findings.md` when done.

### What to research:

**1. Executor Loop Architecture**
Read `blipshell/core/executor.py` (the main execution loop) and `blipshell/core/agent.py` (the agent/chat loop). Compare this architecture against how successful coding agents work (Aider, SWE-agent, Cline, Claude Code, Codex CLI, OpenHands). Specifically:
- How does the message flow work? Is the conversation history structured correctly for tool-calling models?
- Is the system prompt / user message split correct?
- How does context grow over multiple tool calls? Is there any summarization or trimming?
- What's the stop condition logic? Is it robust?
- Are there any race conditions, edge cases, or anti-patterns in the loop?

**2. Context Management**
The executor accumulates tool results in conversation history. We use a 65K context window (cloud glm-5 via Ollama proxy) and 300-line file reads. Research:
- How do other agents handle context growth? (Aider's repo-map, SWE-agent's sliding window, Claude Code's compaction)
- When should we summarize or trim old tool results?
- What's the optimal balance between keeping context (so the model remembers what it read) and trimming (so inference stays fast)?
- How do other cloud-proxied agents handle latency vs context size tradeoffs?

**2b. Memory/Context Budget Architecture**
Read `blipshell/memory/manager.py`, `blipshell/memory/query_profiles.py`, and the memory assembly in `blipshell/core/agent.py` (around line 1520-1570). The memory system has 5 pools (Core, ActiveSession, RecentHistory, Recall, Buffer) sized by percentages of `total_context_tokens`. The current chat history (last 20 messages) is appended separately with NO budget or overflow protection. Research:
- Is the pool percentage split reasonable (10/35/15/30/10)?
- Should `total_context_tokens` even exist as a separate config, or should it always derive from the endpoint?
- The last 20 chat messages are uncapped — what happens when they exceed the remaining context budget?
- How do other agents guard against context overflow from accumulated conversation history?
- Should the conversation history have its own pool/budget?

**3. Tool Design Audit**
Read all tools in `blipshell/core/tools/`. For each tool:
- Is the description clear enough for an LLM to use correctly?
- Are the parameter descriptions precise?
- Are error messages actionable (do they guide the model to the right next step)?
- Are there missing tools that would reduce wasted tool calls?
- Compare our tool set against Claude Code's and SWE-agent's tool sets.
- Is 300 lines the right default for read_file? Claude Code sends entire files, SWE-agent uses 100-line windows.

**4. Prompt Engineering for Smaller Models**
Read `blipshell/llm/prompts.py`, focusing on `executor_system_prompt()` and `dynamic_execution_prompt()`. The executor prompt needs to work with glm-5, not Claude. Research:
- How should prompts differ for smaller/open-source models vs Claude/GPT-4?
- Are there too many rules? Too few? Research shows smaller models can't track 10+ constraints.
- Is the example workflow helpful or confusing?
- How do SWE-agent and Aider structure their prompts for open-source models?
- Should we use few-shot examples of complete tool-calling sequences?

**5. Tool Calling Format & Reliability**
The model uses Ollama's tool calling API (OpenAI-compatible format). Research:
- How reliable is tool calling with different open-source models?
- Are there known issues with specific models and Ollama's tool calling?
- Would a different approach (e.g., text-based tool calling like SWE-agent uses) be more reliable?
- How do we handle the model returning malformed tool calls?

**6. Performance & Timeout Issues**
We're hitting 300-second timeouts during executor runs. The model runs on a consumer GPU. Research:
- What causes inference slowdown as context grows?
- Is there a context size sweet spot for local models?
- Would streaming responses help detect stuck models earlier?
- How do other local-model agents handle slow inference?

**7. Completion Signaling**
We have a `task_complete` tool (in `blipshell/core/tools/interaction_tools.py`) plus "no tool calls = done" as fallback. The model often doesn't call task_complete. Research:
- How do other agents handle completion reliably?
- Is task_complete the right approach, or should we rely purely on natural completion?
- How does Cline's `attempt_completion` work in detail? How about OpenHands' finish action?
- Should we force completion when budget exhausts (inject a "you must call task_complete NOW" message)?

### Known bugs we've already fixed (for context):
- Pagination cache corruption: executor was caching paginated output and re-paginating it
- Disconnected file tracking: executor's files_read set wasn't wired to ReadFileTool
- Tool gating: keyword-based filtering removed tools incorrectly
- TASK_COMPLETE magic string: replaced with task_complete tool
- Chat-path iteration cap: "did you finish?" triggered 30+ tool calls
- Missing TerminalRule for task_complete: tool_rules.py now stops execution after task_complete
- Memory pool hard caps removed: Core and Recall had artificial max_tokens caps fighting the percentages
- total_context_tokens bumped from 32K to 65K to match endpoint context window

### Recent test results (for context):
- 65K context: 34 tool calls, task_complete fires, zero wasted re-reads, felt faster than 196K
- ask_user tool works: model asks clarifying questions with concrete options
- Budget wind-down respected: model wraps up when budget warning injected
- Timeout bumped to 1200s for large generation tasks (glm-5:cloud can take 19min for big files)

### Output format:
Write findings to `research_findings.md` with:
1. Executive summary (top 5 most impactful findings)
2. Detailed findings per section (1-7 above)
3. Recommended changes, prioritized by impact
4. Comparisons with specific agents where relevant (cite what they do differently)
5. Any bugs or anti-patterns you find that we haven't caught yet
