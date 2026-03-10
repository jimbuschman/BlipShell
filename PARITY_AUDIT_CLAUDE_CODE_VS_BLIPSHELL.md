# STRICT PARITY AUDIT: Claude Code vs BlipShell

**Date**: 2026-03-02
**Auditor**: Claude Code (Opus 4.6) — self-auditing against BlipShell codebase
**Scope**: Functional, architectural, behavioral, and UX parity

---

## SECTION 1: High-Level Architectural Comparison

### 1.1 Execution Model

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Core loop** | Single agent loop: LLM → tool calls → results → repeat until done or budget | Unified `ChatLoop.run()`: LLM → tool calls → results → repeat until done or budget |
| **Multi-turn** | Yes — loops until text response with no tool calls, or explicit plan mode exit | Yes — loops until text response, `task_complete` tool, or budget exhaustion |
| **Completion signal** | Implicit: text response with no tool calls = done | Hybrid: `task_complete` tool (primary) + implicit text (fallback) + inline text capture |
| **Budget** | Not explicitly exposed to user; internal turn limits | Explicit: 5 for simple chat, 50 for executor; configurable |
| **Subagents** | `Agent` tool spawns parallel/background subagents with isolated context | No subagent system — single execution thread only |
| **Execution paths** | Single unified path (all conversations go through same loop) | Two paths: `_chat_simple()` (conversational, budget=5) and `_chat_planned()` (executor, budget=50) |

**Verdict**: DIVERGENT. Claude Code's subagent architecture is a fundamental capability gap. BlipShell's dual-path execution is an architectural choice that adds complexity without clear benefit over a unified loop.

### 1.2 Context Handling

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Context window** | Managed by Anthropic infrastructure; automatic compression of prior messages | Managed locally: `compact_messages()` at 85% threshold, heuristic token estimation (`len/4`) |
| **Token counting** | Server-side accurate token counting | Client-side heuristic: `len(content) // 4` — inaccurate for code, JSON, non-English text |
| **Compression** | Automatic system-level compression (transparent to agent) | Manual compaction: keeps system msgs, first user msg, last 5 tool results, summarizes older |
| **Memory injection** | CLAUDE.md + auto-memory files loaded at conversation start | 5-pool system (Core, ActiveSession, RecentHistory, Recall, Buffer) with dynamic budget allocation |
| **Lost-in-middle** | Not explicitly handled (relies on model architecture) | Explicitly mitigated: Core first, history middle, Recall last |

**Verdict**: DIVERGENT. BlipShell's pool-based memory system is more sophisticated than Claude Code's flat file approach, but its token estimation is crude compared to server-side counting. The compaction strategy differs fundamentally.

### 1.3 Tool Invocation

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Tool format** | Anthropic tool_use blocks (native API) | Ollama/OpenAI function calling format |
| **Parallel calls** | Model decides; multiple tool_use blocks in single response | Model decides; partitioned into sequential (approval) + parallel (reads) via `asyncio.gather()` |
| **Tool results** | `tool_result` content blocks, fed back to model | Dict messages appended to conversation, fed back to model |
| **Deduplication** | Not mentioned in public docs; model responsibility | Explicit: cross-batch + within-batch dedup by (name, args) hash |
| **Output bounding** | Per-tool limits (Read: 2000 lines, Bash: timeout) | Per-tool limits (read_file: 300 lines, grep: 50 results, shell: 8000 chars) |
| **Tool availability** | Static per agent type; dynamic in plan mode | Dynamic via `tool_provider` callback; plan mode filters to read-only |

**Verdict**: PARTIALLY EQUIVALENT. Core mechanics similar. BlipShell's explicit dedup and partitioning are additions (not gaps). Output bounds differ in specifics.

### 1.4 Memory System

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Persistence** | CLAUDE.md (project, checked into repo) + auto-memory directory (~/.claude/projects/) | SQLite + ChromaDB dual store; entity graph; 5-pool context injection |
| **Scope** | File-based: MEMORY.md auto-loaded, topic files manually referenced | Database-backed: semantic search, keyword search, entity expansion, temporal decay |
| **Cross-session** | Auto-memory files persist; CLAUDE.md persists in repo | Full persistence: memories, entities, lessons, core memories, session summaries |
| **Retrieval** | Read tool on memory files; no semantic search | 9-step search pipeline: semantic + FTS5 + entity graph + multi-factor scoring |
| **Injection** | CLAUDE.md injected as system context; memory files read on demand | Automatic injection into 5 pools every turn; query-profile-based budget allocation |

**Verdict**: BlipShell EXCEEDS Claude Code in memory capabilities. This is intentional — BlipShell's memory system is its primary differentiator.

### 1.5 Streaming Behavior

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Token streaming** | Real-time SSE from Anthropic API → terminal | Real-time: `client.chat_stream()` → `on_token()` callback → `sys.stdout.write()` + flush |
| **Thinking indicator** | Shows "Thinking..." or extended thinking content | Shows "Thinking..." spinner, stops on first token |
| **Tool call display** | Shows tool name, parameters, inline in conversation | Compact inline: `▸ tool_name arg_hint` (single), `▸ Running N tools: ...` (batch) |
| **Extended thinking** | Supported natively (thinking blocks visible) | `/think on|off` toggle; model-dependent (qwen3 thinking mode) |

**Verdict**: FUNCTIONALLY EQUIVALENT. Both stream tokens in real-time. Display format differs but serves same purpose.

### 1.6 Error Handling

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Tool errors** | Error returned in tool_result; model self-corrects | Error returned in ToolResult; actionable error messages guide next action |
| **LLM errors** | Retry with backoff (Anthropic SDK handles) | 3-layer: client retries (exponential backoff) → router fallback (endpoint switch) → model switch |
| **Rate limits** | Anthropic handles; user sees "rate limited" | OpenAI client: 2 extra retries → `RateLimitExhaustedError` → router tries next endpoint |
| **Context overflow** | Automatic compression | Manual compaction at 85%; forced completion at 95% |
| **Model unavailable** | Falls back gracefully (API-side) | `is_model_error()` classification; endpoint vs model failure distinction; fallback routing |

**Verdict**: BlipShell has MORE SOPHISTICATED error handling due to multi-endpoint routing. Claude Code benefits from single-provider simplicity.

### 1.7 State Management

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Working directory** | Persists across commands; git-aware | Per-project root_path; tools scoped to project directory |
| **Git awareness** | Git status snapshot at conversation start | Live git status/diff tools; branch detection in project context |
| **File tracking** | Internal (not exposed to model) | `[STATE]` block injection: files read/created/edited, last 5 actions |
| **Session state** | Conversation history (compressed as needed) | SQLite-backed: session_messages, turn_events, token usage |
| **Worktrees** | `EnterWorktree` tool for isolated git worktrees | Not implemented |

**Verdict**: PARTIALLY EQUIVALENT with gaps. BlipShell's state injection is more explicit. Claude Code's worktree support is missing from BlipShell.

---

## SECTION 2: Feature Parity Checklist

| # | Feature | Claude Code Behavior | BlipShell Behavior | Match? | Required Action |
|---|---------|---------------------|-------------------|--------|-----------------|
| 1 | **File reading** | `Read` tool: 2000-line default, offset/limit, images, PDFs, notebooks | `read_file`: 300-line default, start_line/max_lines, no image/PDF/notebook support | Partial | Add image/PDF reading; increase default line limit |
| 2 | **File writing** | `Write` tool: creates new files, overwrites existing | `write_file`: creates/overwrites, parent dir creation, symlink check | Yes | — |
| 3 | **File editing** | `Edit` tool: exact string replacement, unique match required | `edit_file`: 3-layer matching (exact → whitespace-normalized → fuzzy 0.6), Python linting | Yes+ | BlipShell exceeds (fuzzy matching + linting) |
| 4 | **Shell execution** | `Bash` tool: timeout, background, description required | `run_command`: 30s timeout, 8000 char cap, destructive detection, OS-aware guidance | Partial | No background execution support |
| 5 | **File search (glob)** | `Glob` tool: glob patterns, sorted by mtime | `glob_files`: glob patterns, sorted by mtime, 100-file cap, size display | Yes | — |
| 6 | **Content search (grep)** | `Grep` tool: regex, file type filter, output modes, context lines, multiline | `grep_files`: regex, include glob, 50-result cap, 200-char line cap | Partial | Missing: output modes, context lines (-A/-B/-C), multiline, type filter, offset/head_limit |
| 7 | **Web search** | `WebSearch`: Anthropic-provided, sources required in response | `web_search`: Tavily primary, DuckDuckGo fallback | Yes | Different backends, same functionality |
| 8 | **Web fetch** | `WebFetch`: URL + prompt, AI processing of content, cache, redirect handling | `web_fetch`: URL only, HTML→text extraction, SSRF protection, no AI processing | Partial | Missing: AI-powered content extraction with prompt parameter |
| 9 | **Subagents** | `Agent` tool: 6+ agent types (Explore, Plan, general, etc.), parallel, background, worktree isolation | **Not implemented** | **No** | **CRITICAL GAP**: No subagent system |
| 10 | **Plan mode** | `EnterPlanMode`/`ExitPlanMode`: user approval required to enter, restricts to read-only | `enter_plan_mode`/`exit_plan_mode`: LLM-controlled (no user approval), filters to read-only tools | Partial | Claude Code requires user consent; BlipShell lets LLM decide autonomously |
| 11 | **Task tracking** | `TaskCreate`/`TaskUpdate`/`TaskList`/`TaskGet`: structured todo list with dependencies, blocking | `start_background_task`/`check_background_task`/`list_background_tasks`: background job execution | **No** | Different concept: Claude Code tracks work items; BlipShell executes background jobs |
| 12 | **User questions** | `AskUserQuestion`: structured questions with options, multi-select, markdown previews | `ask_user`: free-text question, free-text response | Partial | Missing: structured options, multi-select, preview markdown |
| 13 | **Git worktrees** | `EnterWorktree`: creates isolated worktree, auto-cleanup | **Not implemented** | **No** | Missing feature |
| 14 | **Skills/slash commands** | `Skill` tool: /commit, /review-pr, etc. via expandable skill system | Hardcoded slash commands in CLI if/elif chain | Partial | No extensible skill system; commands are hardcoded |
| 15 | **Notebook editing** | `NotebookEdit`: cell-level editing of .ipynb files | **Not implemented** | **No** | Missing feature |
| 16 | **Image reading** | `Read` tool handles PNG, JPG natively (multimodal) | **Not implemented** | **No** | Missing feature (requires multimodal model) |
| 17 | **PDF reading** | `Read` tool handles PDFs with page ranges | **Not implemented** | **No** | Missing feature |
| 18 | **Permission modes** | Configurable permission modes; per-tool approval with persistent settings | `tools_requiring_approval` config list; session-wide `/approve all`; destructive detection | Partial | Less granular than Claude Code's permission system |
| 19 | **Hooks system** | User-configurable shell hooks on events (tool calls, prompt submit) | **Not implemented** (on wish list) | **No** | Missing feature |
| 20 | **IDE integration** | VS Code extension, JetBrains plugin | **Not implemented** (on wish list) | **No** | Missing feature |
| 21 | **Cost tracking** | Shows API cost per session (dollar amounts) | `/tokens` shows token counts per endpoint; no cost calculation | Partial | Missing: dollar-amount cost tracking |
| 22 | **Git commit workflow** | Structured commit flow: status → diff → log → stage → commit with Co-Authored-By | `git_add` + `git_commit` tools; no structured workflow | Partial | Missing: structured commit protocol with conventions |
| 23 | **PR creation** | `gh pr create` with structured body template | `run_command` with manual gh commands | Partial | No structured PR workflow |
| 24 | **Context compression** | Automatic system-level compression (transparent) | `compact_messages()` at 85% threshold; heuristic token counting | Partial | Crude token estimation; manual compaction |
| 25 | **Streaming display** | Markdown rendering in terminal (CommonMark) | Raw text output via `sys.stdout.write()` | Partial | No markdown rendering in terminal |
| 26 | **Extended thinking** | Native support (thinking blocks in API) | `/think` toggle; model-dependent | Partial | Not natively supported; depends on model capability |
| 27 | **MCP support** | Connects to MCP servers; tool discovery and execution | Phase 1 implemented: stdio transport, tool discovery, execution | Yes | Core parity achieved |
| 28 | **Background execution** | `Agent` tool with `run_in_background`; `Bash` with `run_in_background` | `start_background_task` for LLM jobs; no background shell | Partial | Missing: background shell execution |
| 29 | **Conversation memory** | Auto-memory files (MEMORY.md + topic files) persisted across sessions | Full memory system: SQLite + ChromaDB + entity graph + lessons + consolidation | Yes+ | BlipShell far exceeds |
| 30 | **Project instructions** | CLAUDE.md loaded automatically from repo root | BLIPSHELL.md support in project context; project digest system | Yes | Equivalent concept |
| 31 | **Multi-model routing** | Single model (Claude) per conversation | Multi-endpoint routing: Ollama + Groq + Gemini with fallback cascade | Yes+ | BlipShell exceeds |
| 32 | **PII sanitization** | Not a feature (Anthropic handles data privacy) | Presidio + regex PII detection; per-endpoint sanitization | Yes+ | BlipShell addition (needed for multi-provider) |
| 33 | **Escape/cancel** | Ctrl+C to cancel | Esc key to cancel mid-response; partial response saved with `[cancelled]` | Yes | — |
| 34 | **Status bar** | Shows model, token count, mode | Bottom toolbar: session #, project, endpoint, context %, think mode | Yes | — |
| 35 | **Keyboard shortcuts** | Customizable via keybindings.json | Not implemented | **No** | Missing feature |
| 36 | **Pause/redirect** | Not a feature (user waits for completion or cancels) | Space/p between tool batches: continue, redirect, stop | Yes+ | BlipShell addition |
| 37 | **Session management** | Single conversation per invocation | Multi-session: create, resume, close with summary + lesson extraction | Yes+ | BlipShell addition |
| 38 | **Health/audit** | Not a feature | `/health`: DB integrity, ChromaDB sync, endpoint health, entity quality | Yes+ | BlipShell addition |
| 39 | **Nightly jobs** | Not a feature | 7-job nightly runner: backup, cleanup, centroid_tag, batch_tag, prune, consolidate, tag_discovery | Yes+ | BlipShell addition |
| 40 | **Observability** | Not exposed (internal Anthropic telemetry) | `/flow`: turn events, search stats, context build details, endpoint routing | Yes+ | BlipShell addition |

---

## SECTION 3: Behavioral Drift Analysis

### 3.1 Behaviors BlipShell ADDS That Claude Code Does Not Have

1. **Persistent memory system** — SQLite + ChromaDB dual store, entity graph, lessons, consolidation. Claude Code uses flat files (CLAUDE.md + auto-memory). BlipShell's is dramatically more capable.

2. **Multi-endpoint LLM routing** — Automatic fallback across Ollama, Groq, Gemini. Claude Code uses a single provider (Anthropic).

3. **PII sanitization** — Presidio NER + regex before cloud endpoints. Not needed in Claude Code (single trusted provider).

4. **Background memory worker** — Dedicated thread processes memories asynchronously. Claude Code has no background processing.

5. **Entity graph** — 31K entities, 56K relationships, predicate contradiction handling. No equivalent in Claude Code.

6. **Session lifecycle** — Start/end with summary generation, lesson extraction, digest updates. Claude Code conversations are ephemeral (auto-memory is the only persistence).

7. **Mid-task steering** — Pause/redirect between tool batches via Space/p key. Claude Code requires full cancellation to change direction.

8. **Query profile classification** — Dynamic pool budget allocation based on query type (recall/session/coding/balanced). Claude Code uses static context injection.

9. **OllamaGate** — Priority-based serialization of local LLM calls (interactive > background). Not needed in Claude Code (cloud API handles concurrency).

10. **Nightly maintenance** — Automated backup, cleanup, tagging, consolidation. No equivalent in Claude Code.

### 3.2 Behaviors Claude Code HAS That BlipShell OMITS

1. **Subagent system** — Claude Code can spawn parallel specialized agents (Explore, Plan, general-purpose) with isolated contexts. BlipShell has no subagent capability. This is the single largest functional gap.

2. **Git worktree isolation** — Claude Code can create isolated worktrees for safe experimentation. BlipShell has no worktree support.

3. **Multimodal input** — Claude Code reads images (PNG, JPG), PDFs (with page ranges), and Jupyter notebooks natively. BlipShell is text-only.

4. **Structured user questions** — Claude Code's `AskUserQuestion` supports options, multi-select, markdown previews, headers. BlipShell's `ask_user` is free-text only.

5. **Extensible skill system** — Claude Code skills are pluggable (/commit, /review-pr, etc.). BlipShell commands are hardcoded in an if/elif chain.

6. **Hooks system** — Claude Code supports user-configured shell hooks on lifecycle events. BlipShell has no hooks.

7. **IDE integration** — Claude Code has VS Code and JetBrains plugins. BlipShell is CLI-only.

8. **Customizable keybindings** — Claude Code allows keybinding customization. BlipShell has fixed key mappings.

9. **Background shell execution** — Claude Code's Bash tool supports `run_in_background`. BlipShell's `run_command` is synchronous-only.

10. **Notebook editing** — Claude Code can edit individual cells in .ipynb files. Not supported in BlipShell.

11. **AI-powered web content extraction** — Claude Code's `WebFetch` takes a prompt and processes content with an AI model. BlipShell's `web_fetch` only does HTML→text conversion.

12. **Cost tracking** — Claude Code displays dollar-amount API costs. BlipShell shows token counts only.

### 3.3 Edge Case Handling Differences

| Scenario | Claude Code | BlipShell |
|----------|------------|-----------|
| **Empty tool response** | Returns to model, model decides next action | Returns to model, may trigger inline text fallback |
| **Model refuses tool** | Error in tool_result, model adapts | Error in ToolResult with actionable guidance message |
| **Context overflow mid-conversation** | System compression (transparent) | Compaction at 85% + forced completion at 95% |
| **User cancels mid-tool** | Ctrl+C kills current operation | Esc cancels LLM call only; in-flight tool calls not cancellable |
| **Tool output too large** | Per-tool limits enforced | Per-tool limits enforced (different thresholds) |
| **Rate limit during conversation** | Anthropic SDK retries transparently | Multi-layer: client retry → endpoint fallback → model switch |
| **Model hallucinating tool calls** | Returns error, model adjusts | Dedup blocks repeated identical calls; errors guide to correct tool |

### 3.4 Context Management Differences

| Scenario | Claude Code | BlipShell |
|----------|------------|-----------|
| **Initial context** | CLAUDE.md + auto-memory MEMORY.md | 5-pool system: Core memories + Lessons + Recent history + Active session + Recall |
| **Mid-conversation** | Automatic compression by system | Manual compaction at 85% threshold |
| **Token counting** | Server-side (accurate) | `len(content) // 4` heuristic (can be 30-50% off for code/JSON) |
| **Cross-session** | Auto-memory files (manual management by model) | Automatic: semantic search retrieves past memories every turn |

### 3.5 Retry/Error Recovery Differences

| Scenario | Claude Code | BlipShell |
|----------|------------|-----------|
| **LLM request fails** | SDK retries with backoff | Client retries (3 attempts) → router tries fallback endpoint → different model |
| **Tool execution fails** | Error returned to model | Error + actionable guidance returned to model |
| **Rate limited** | SDK handles transparently | 2 extra retries (5s, 10s) → `RateLimitExhaustedError` → next endpoint |
| **Model not found** | N/A (single provider) | `is_model_error()` classification; don't penalize endpoint; try fallback model |

### 3.6 Long-Running Plan/Task Differences

| Scenario | Claude Code | BlipShell |
|----------|------------|-----------|
| **Complex multi-file task** | Subagents for parallel research; main agent coordinates | Single executor thread; 50-call budget; no parallelism at task level |
| **Plan mode** | User must approve entering plan mode | LLM enters autonomously via `enter_plan_mode` tool |
| **Budget exhaustion** | Internal limits (not exposed) | Explicit: 80% wind-down message → nudge → forced completion |
| **Task tracking** | `TaskCreate`/`TaskUpdate` for structured progress | No equivalent work-item tracking |

### 3.7 Tool Loop Differences

| Scenario | Claude Code | BlipShell |
|----------|------------|-----------|
| **Loop termination** | Text response with no tool calls | 5 methods: task_complete tool, text response, budget, nudge, empty |
| **Consecutive identical calls** | Model responsibility to avoid | Explicit dedup blocks identical consecutive calls |
| **Max iterations** | Internal limit | `budget + 10` max rounds; configurable per path |
| **Parallel tool calls** | Model decides; all execute concurrently | Partitioned: approval tools sequential, then parallel with semaphore(8) |

---

## SECTION 4: Bug Recurrence Analysis

### 4.1 Structural Reasons for Repeated Bugs

1. **Dual execution paths** (`_chat_simple` vs `_chat_planned`):
   - Any fix applied to one path may not be applied to the other
   - Tool registration, context building, error handling, and memory injection all have path-specific code
   - Claude Code avoids this by having a single unified loop
   - **Risk**: High. Every new feature must be tested on both paths.

2. **Mixin architecture** (5 mixins on Agent):
   - Shared state via `self.*` across 6 files
   - No interface contracts between mixins
   - Initialization order dependencies (e.g., tools must register before session starts)
   - **Risk**: Medium. Hard to reason about state mutations across files.

3. **Token estimation heuristic** (`len/4`):
   - Consistently underestimates for code (special characters, indentation)
   - Consistently overestimates for short English text
   - Causes premature compaction or late compaction depending on content type
   - **Risk**: Medium. Silent errors — context may be too full or too sparse.

4. **SQLite/ChromaDB dual-store without transactions**:
   - Deletes/archives happen as separate calls
   - Power loss or crash between SQLite update and ChromaDB update causes permanent drift
   - `chroma_retry_queue` mitigates but doesn't prevent all drift scenarios
   - **Risk**: Medium-High. Known issue (Section 33 in CLAUDE.md, NOT STARTED).

5. **Tool re-registration on project activate/deactivate**:
   - Known bug: `deactivate_project()` unregisters `grep_files`/`glob_files` but doesn't re-register them for non-project mode
   - Fixed at init time but not after deactivation
   - Regression scenario exists but fix is incomplete
   - **Risk**: High. Actively tracked bug.

### 4.2 Subtle Implementation Differences That May Cause Issues

1. **Plan mode consent model**:
   - Claude Code: User must approve entering plan mode (safety gate)
   - BlipShell: LLM enters autonomously
   - Risk: LLM may enter plan mode unnecessarily, wasting budget on exploration

2. **Completion detection**:
   - Claude Code: Implicit (no tool calls in response = done)
   - BlipShell: 5 methods including fallback to inline text capture
   - The fallback chain adds complexity; `capture_inline_text` was added to fix false negatives in stress tests
   - Risk: Edge cases where completion is detected incorrectly

3. **Streaming + tool calls in same response**:
   - When LLM returns text AND tool calls in the same response chunk
   - BlipShell: `capture_inline_text` saves text alongside tool calls; may cause double-counting
   - Risk: Low but subtle

4. **File cache in executor**:
   - `executor._file_cache` stores file contents after read
   - `ReadFileTool.file_cache` is a separate reference
   - If cache becomes stale (file modified externally), model works with stale data
   - No cache invalidation on external modification
   - Risk: Medium for long-running executor sessions

### 4.3 Async/Streaming Mismatches

1. **OllamaGate serialization**:
   - All local Ollama calls serialized through OllamaGate (single GPU)
   - Background memory worker and interactive chat compete for the gate
   - During heavy memory processing, interactive chat may have noticeable latency
   - Claude Code: No equivalent issue (cloud API handles concurrency)

2. **Escape cancellation scope**:
   - Esc cancels the LLM call but not in-flight tool executions
   - If a tool (e.g., `run_command` with 30s timeout) is executing, Esc won't interrupt it
   - Claude Code: Ctrl+C can interrupt at any point

3. **Background task completion**:
   - Background tasks run on the memory worker thread with its own event loop
   - Results must be polled via `check_background_task`
   - No push notification to the user when a background task completes
   - Claude Code: Background agents notify automatically on completion

---

## SECTION 5: Memory System Comparison

### 5.1 Injection Timing

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **When injected** | At conversation start: CLAUDE.md + MEMORY.md loaded | Every turn: `_build_messages()` reconstructs full context with memory pools |
| **Dynamic retrieval** | Manual: model must use Read tool to access topic memory files | Automatic: semantic search runs every turn, results injected into Recall pool |
| **Refresh** | Static for conversation duration (unless model re-reads) | Dynamic: each turn gets fresh search results based on current query |

**Verdict**: DIVERGENT. BlipShell's per-turn injection is more responsive but more expensive. Claude Code's is cheaper but potentially stale within a conversation.

### 5.2 Persistence Strategy

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Storage** | Flat files (MEMORY.md, topic .md files) in `~/.claude/projects/` | SQLite (structured) + ChromaDB (vectors) + entity graph |
| **Write trigger** | Model decides when to write via Write/Edit tools | Automatic: every message persisted immediately; background worker processes |
| **Granularity** | Arbitrary (model chooses what to save) | Per-message: each message summarized, ranked, embedded, tagged |
| **Deduplication** | Model responsibility (instructions say "check before writing") | Automated: ChromaDB similarity search + LLM dedup decision |

**Verdict**: ARCHITECTURALLY INCONSISTENT. Fundamentally different approaches. BlipShell's is systematic; Claude Code's is opportunistic.

### 5.3 Token Budgeting

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Budget source** | Server-managed context window | Endpoint-specific `context_tokens` config |
| **Allocation** | Flat: CLAUDE.md + MEMORY.md loaded in full | Pool-based: 5 pools with percentage allocations, hard caps, dynamic redistribution |
| **Overflow** | System compression (transparent) | Pool overflow → archive to RecentHistory; compaction at 85% |
| **Priority** | No explicit priority (order in system prompt) | Priority-weighted: Core pool has hard cap (2048), Recall capped at 8192 |

**Verdict**: DIVERGENT. BlipShell's is more sophisticated but adds complexity. Claude Code's simplicity may be more robust.

### 5.4 Pruning Behavior

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Auto-pruning** | Model must manually edit/delete memory files | Automatic: nightly prune job archives old low-value memories |
| **Consolidation** | Model merges entries in MEMORY.md | Automated: ChromaDB similarity search, LLM dedup, merge access counts |
| **Temporal decay** | None (files persist indefinitely) | Exponential decay per memory type (facts persist longer than events) |
| **Stale detection** | Instructions say "remove outdated memories" | Consolidation bonus based on access count; nightly pruning by age + rank |

**Verdict**: BlipShell EXCEEDS. Automated maintenance vs manual curation.

### 5.5 Session Boundary Behavior

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Session start** | Load CLAUDE.md + MEMORY.md; git status snapshot | Load core memories + lessons → Core pool; search recent sessions → RecentHistory; retry failed ChromaDB ops |
| **Session end** | Auto-memory update (if model decides to save) | Full pipeline: dump messages → generate summary → extract lessons → update project digest |
| **Cross-session** | Auto-memory files persist; topic files persist | Full persistence: all memories searchable; lessons boosted; session summaries preserved |
| **Session resume** | New conversation starts fresh (with memory files) | Can resume session by ID; loads previous messages into ActiveSession pool |

**Verdict**: BlipShell EXCEEDS. Rich session lifecycle vs stateless conversations.

### 5.6 Context Reset Behavior

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Within conversation** | System auto-compresses when approaching limits | `compact_messages()` at 85%: keeps system + first user + last 5 tool results |
| **Across conversations** | Full reset; memory files re-loaded | Session restart with fresh pool allocation; memories re-retrieved via search |
| **Forced reset** | N/A | `/compact` command; `/save` forces memory dump |

**Verdict**: PARTIALLY EQUIVALENT. Different mechanisms, similar outcomes.

### 5.7 Plan Handling

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| **Plan mode memory** | Read-only tools available; memory files readable | Read-only tools available; memory search still works; no write restrictions on memory system itself |
| **Plan persistence** | Plan exists in conversation context only | Plan exists in conversation context; session summary captures plan decisions |
| **Post-plan** | Full tools restored after user approval | Full tools restored after LLM calls `exit_plan_mode` |

**Verdict**: FUNCTIONALLY EQUIVALENT. Memory access patterns same in plan mode.

### 5.8 Overall Memory Assessment

BlipShell's memory system is:
- **Equivalent**: For cross-session persistence (both persist across sessions)
- **Exceeds**: For retrieval (semantic search vs file reads), maintenance (automated vs manual), richness (entity graph, lessons, consolidation)
- **Divergent**: In architecture (database vs files), injection timing (per-turn vs at-start), and pruning (automated vs manual)
- **Not missing functionality**: BlipShell's memory is strictly a superset of Claude Code's
- **Architecturally inconsistent**: Different paradigm entirely (systematic vs opportunistic)

---

## SECTION 6: Final Verdict

### **Functionally Equivalent With Significant Gaps**

BlipShell achieves functional parity with Claude Code's core loop mechanics (tool calling, streaming, error handling, plan mode) and EXCEEDS it in memory, multi-model routing, observability, and maintenance automation.

However, BlipShell is **missing critical components** that define Claude Code's power:

### Critical Gaps (Required for Parity)

| Gap | Impact | Effort |
|-----|--------|--------|
| **No subagent system** | Cannot parallelize research, cannot delegate to specialized agents, cannot run isolated tasks. This is Claude Code's primary scaling mechanism. | Very High |
| **No multimodal support** | Cannot read images, PDFs, or notebooks. Limits usefulness for visual debugging, documentation review, and data science. | High (requires multimodal model) |
| **No git worktree isolation** | Cannot safely experiment with code changes in isolation. | Medium |
| **No background shell execution** | Cannot run long build/test processes in background while continuing conversation. | Medium |
| **No structured user questions** | Cannot present choices with descriptions, previews, multi-select. Reduces UX quality for decision points. | Medium |

### Moderate Gaps

| Gap | Impact | Effort |
|-----|--------|--------|
| No extensible skill system | Commands are hardcoded; no plugin architecture | Medium |
| No hooks system | Cannot automate lifecycle events | Medium |
| No IDE integration | CLI-only limits adoption | Very High |
| No cost tracking | Users can't monitor spend | Low |
| Crude token estimation | Context management imprecise | Medium |
| Grep tool missing features | No context lines, output modes, multiline | Low-Medium |
| WebFetch lacks AI processing | Raw text only, no prompt-based extraction | Medium |
| No customizable keybindings | Fixed key mappings | Low |
| No notebook editing | Cannot edit .ipynb cells | Medium |

### Areas Where BlipShell Exceeds Claude Code

| Area | Description |
|------|-------------|
| Memory system | 9-step semantic search, entity graph, lessons, consolidation, temporal decay |
| Multi-model routing | Automatic fallback across providers with rate limit handling |
| PII sanitization | Presidio + regex before cloud endpoints |
| Observability | `/flow` turn events, `/health` audit, `/context` pool usage |
| Session management | Multi-session with summaries, lessons, resume |
| Nightly maintenance | Automated backup, cleanup, tagging, consolidation |
| Mid-task steering | Pause/redirect between tool batches |
| Edit quality | 3-layer matching + Python syntax linting |
| Error guidance | Actionable error messages that guide next tool call |
| Deduplication | Explicit tool call dedup prevents wasted iterations |
| Project digest | Auto-maintained cross-session project summary |

### Bottom Line

BlipShell is **not a clone of Claude Code** — it's a different system that shares the same general architecture (agent loop + tools + streaming) but makes fundamentally different choices about memory, routing, and state management. The subagent gap is the most significant architectural difference. Everything else is either at parity, a minor gap, or an area where BlipShell exceeds.

**To achieve true parity, the minimum required work is:**
1. Subagent system (parallel agent spawning with isolated contexts)
2. Multimodal input support (images, PDFs, notebooks)
3. Background shell execution
4. Structured user questions with options/previews
5. Git worktree isolation

**To maintain BlipShell's advantages, preserve:**
1. The memory system (far superior to Claude Code's flat files)
2. Multi-endpoint routing (Claude Code has no equivalent)
3. Observability infrastructure (/flow, /health, /context)
4. Nightly maintenance automation
5. Mid-task steering (pause/redirect)

---

*This audit was performed by Claude Code (Opus 4.6) examining BlipShell's source code at commit f49a89b on branch main, 2026-03-02.*
