# Deep Dive: Claude Code vs BlipShell — Complete Technical Comparison

**Date**: 2026-03-05
**Claude Code version**: Latest (Opus 4.6, March 2026)
**BlipShell version**: 0.1.0 (main branch, commit b91dfa6)

This document is an exhaustive, line-level comparison between Anthropic's Claude Code and BlipShell. It covers architecture, prompts, tools, context management, memory, UX, and everything in between — down to specific text, thresholds, and behavioral details.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Agent Loop](#2-core-agent-loop)
3. [System Prompts & Behavioral Rules](#3-system-prompts--behavioral-rules)
4. [Tool System — Complete Inventory](#4-tool-system--complete-inventory)
5. [Context Window & Compaction](#5-context-window--compaction)
6. [Memory & Persistence](#6-memory--persistence)
7. [Model Routing & Selection](#7-model-routing--selection)
8. [File Editing & Code Intelligence](#8-file-editing--code-intelligence)
9. [Permissions & Safety](#9-permissions--safety)
10. [Research Mode & Web Integration](#10-research-mode--web-integration)
11. [Sub-Agents & Delegation](#11-sub-agents--delegation)
12. [Git Integration](#12-git-integration)
13. [MCP Integration](#13-mcp-integration)
14. [Hooks & Lifecycle Events](#14-hooks--lifecycle-events)
15. [CLI Commands & UX](#15-cli-commands--ux)
16. [Streaming & Real-Time Feedback](#16-streaming--real-time-feedback)
17. [Error Handling & Recovery](#17-error-handling--recovery)
18. [Configuration System](#18-configuration-system)
19. [Cost & Token Management](#19-cost--token-management)
20. [Testing & Quality](#20-testing--quality)
21. [Privacy & PII](#21-privacy--pii)
22. [What Claude Code Has That BlipShell Doesn't](#22-what-claude-code-has-that-blipshell-doesnt)
23. [What BlipShell Has That Claude Code Doesn't](#23-what-blipshell-has-that-blipshell-doesnt)
24. [Prompt Text Comparison](#24-prompt-text-comparison)
25. [Specific Thresholds & Constants](#25-specific-thresholds--constants)
26. [Recommendations: Closing the Gap](#26-recommendations-closing-the-gap)

---

## 1. Executive Summary

**Claude Code** is a thin harness around an extremely capable model (Claude Opus/Sonnet). Its philosophy: the model is smart enough that you don't need much scaffolding. Simple loop, simple tools, CLAUDE.md for memory, sub-agents for parallelism. The system compensates for a minimal architecture with a $15/M-token model that follows instructions reliably.

**BlipShell** is heavy scaffolding around weaker models (7B-14B local, Groq/Gemini cloud). Its philosophy: the model isn't smart enough to self-manage, so the system must manage it. State injection, budget tracking, deduplication, guardrails, multi-pool memory, entity graphs, PII filtering, multi-endpoint failover. The system compensates for model weakness with engineering.

**Fundamental tradeoff**: Claude Code trusts the model. BlipShell doesn't (and can't).

### Key Gaps (BlipShell → Claude Code parity)

| Gap | Impact | Difficulty |
|-----|--------|------------|
| No sub-agent system | Can't parallelize research, isolate contexts | HIGH |
| No worktree isolation | Can't do parallel branch work | MEDIUM |
| No prompt caching | Every turn re-sends full context (costly for cloud) | MEDIUM |
| No code intelligence / LSP | Relies entirely on grep/glob for code nav | HIGH |
| No checkpoint/rewind | Can't undo to a previous conversation state | MEDIUM |
| No Chrome/browser tools | No web automation or visual testing | LOW-MED |
| Crude token counting (len/4) | Over/under-estimates context usage | LOW |
| No fork/branch conversations | Single conversation thread only | LOW |
| No desktop app / mobile handoff | CLI only | LOW |
| No diff viewer | No interactive diff review | LOW |
| No plugin system | Can't extend via third-party plugins | MEDIUM |

### Key Advantages (BlipShell over Claude Code)

| Advantage | Description |
|-----------|-------------|
| Persistent semantic memory | 5-pool system with summarization, ranking, entity graphs, temporal decay |
| Multi-endpoint failover | Automatic cascade: Cloud → Cloud2 → Local, per-task routing |
| PII sanitization | Presidio NER + regex, per-endpoint toggle |
| Guardrails system | Completion audit, trajectory monitoring, correction detection |
| Nightly maintenance | Automated backup, cleanup, tagging, consolidation |
| Zero cloud dependency option | Can run 100% locally via Ollama |
| Model-agnostic | Works with any Ollama-compatible model |
| Background memory processing | Every message goes through summarize → rank → embed → tag pipeline |
| Entity relationship graph | 31K entities, 56K relationships, used in search expansion |
| Project digests | Auto-maintained project summaries across sessions |

---

## 2. Core Agent Loop

### Claude Code

**Architecture**: Single unified loop. Every interaction — chat, code editing, research — goes through the same path.

```
while true:
    response = claude.call(messages, tools)
    if response has tool_calls:
        for each tool_call:
            result = execute(tool_call)
            messages.append(tool_result)
    else:
        # Text-only response = done
        break
```

**Completion**: Implicit — when the model returns text without any tool calls, the turn is done. No explicit completion tool needed because Claude is reliable enough to simply stop calling tools when done.

**Budget**: Not explicitly exposed. Internal turn limits exist but aren't shown to the model. The model self-regulates because it's instruction-following enough.

**Streaming**: Real-time token streaming via SSE. Tokens appear as they're generated.

**Key property**: The loop is trivially simple because the model doesn't need help.

### BlipShell

**Architecture**: Dual-path loop. Simple chat (`_chat_simple`, budget=5) and executor (`_chat_planned`/`execute_dynamic`, budget=50) are separate code paths that share a common `ChatLoop.run()` engine.

```python
# ChatLoop.run() — 7-phase tool execution per iteration
for _round in range(max_rounds):
    # Phase 1: Context compaction (if >85% of context_limit)
    # Phase 2: Tool availability (dynamic via tool_provider)
    # Phase 3: LLM call with OllamaGate serialization
    # Phase 4: Parse tool calls
    #   4a: Budget check (trim to remaining)
    #   4b: Dedup check (block identical consecutive calls)
    #   4c: Pre-execution display (⏳ icon)
    #   4d: Partition (sequential vs parallel)
    # Phase 5: Execute sequential tools (approval, ask_user)
    # Phase 6: Execute parallel tools (asyncio.gather + Semaphore(8))
    # Phase 7: Results, callbacks, display
    # Check: task_complete tool → guardrails audit → break
    # Check: Guardrails trajectory injection (every N calls)
    # Check: Pause/redirect callback
```

**Completion**: Three signals:
1. `task_complete` tool call (primary, preferred) — guardrails can reject and force retry
2. Text-only response with no tool calls (implicit, like Claude Code)
3. Substantial inline text (≥200 chars) returned alongside tool calls (`capture_inline_text`)

**Budget**: Explicit, visible to model via `[STATE]` injection:
- Simple chat: 5 tool calls (configurable)
- Executor: 50 tool calls (configurable, min 50 in project mode)
- Research mode: 3x normal budget (min 30)
- At 80% budget: wind-down message injected
- At 95% budget: all tools stripped except `task_complete`

**Streaming**: Token-by-token via Ollama streaming API. Tokens appear as generated. Thinking spinner shown until first token arrives.

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Loop complexity | ~50 lines of core logic | ~400 lines (7 phases) |
| Completion signal | Implicit (no tool calls = done) | Hybrid (task_complete tool + implicit + inline capture) |
| Budget management | Hidden, model self-regulates | Explicit, injected into context, wind-down mechanism |
| Deduplication | Model responsibility | System enforced (cross-batch + within-batch) |
| Tool partitioning | Model decides parallel vs sequential | System partitions: approval tools sequential, rest parallel |
| State injection | None needed (model tracks state) | `[STATE]` block every turn: budget, files read, recent actions |
| Context compaction | Automatic at ~95% | Manual at 85%, preserves system+first_user+last_5_tools |
| Dual execution path | No (single unified loop) | Yes (`_chat_simple` + `_chat_planned`) |

**Why BlipShell is more complex**: Weaker models need explicit scaffolding. A 14B model won't self-regulate budget, won't avoid re-reading files, won't stop when done without a task_complete tool. Claude Code's simplicity is a luxury of model quality.

**Gap**: BlipShell's dual-path architecture adds complexity. Consider unifying to a single loop with variable budget (like Claude Code) rather than two separate entry points.

---

## 3. System Prompts & Behavioral Rules

### Claude Code

**System prompt structure** (inferred from behavior and documentation):
1. Core role definition (brief)
2. Tool descriptions with usage guidance
3. 3-5 critical behavioral rules
4. Few-shot examples of good tool sequences
5. Error recovery guidance
6. Permission mode context

**Known enforced rules**:
- Explore before implementing complex changes
- Read files before making changes
- Use grep/search before reading (efficiency)
- Verify changes after making them (run tests)
- Ask for clarification on ambiguous requirements
- Prefer simple solutions, don't over-engineer
- Avoid re-reading files already in context

**Customization**:
- `--system-prompt "text"` — Replace entire prompt
- `--append-system-prompt "text"` — Append to default
- `--system-prompt-file path` — Load from file

**Key insight**: Claude Code's rules are terse and high-level because the model can infer specifics. "Read before editing" is sufficient — no need to explain why or how.

### BlipShell

**System prompt structure** (`prompts.py:executor_system_prompt()`, ~30 lines):
```
You are a coding agent. You complete tasks autonomously using tools.

[Platform: Windows — use dedicated tools, not shell equivalents]

# Rules
1. PLAN first — state your approach in 1-3 sentences before writing code.
2. Read before editing. NEVER re-read a file already in [STATE].
3. Make MINIMAL changes — no refactoring or extras beyond the task.
4. If something fails twice, use ask_user instead of retrying blindly.
5. Call task_complete when DONE. Do NOT just stop responding.
6. Each tool's description explains when/how to use it — follow that guidance.
7. Do NOT narrate your thinking — just call tools or answer directly.
8. For complex multi-file tasks, use enter_plan_mode to investigate first.
```

**Dynamic execution prompt** (`prompts.py:dynamic_execution_prompt()`):
```
Task: {user_request}
```

**Context assembly** (`agent_chat.py:_build_messages()`, ~150 lines):
1. Base system prompt from config
2. If project active: PROJECT CONTEXT section with tool limits + examples
3. Model-specific instructions from ModelSettingsRegistry
4. Project context from `_scan_project_context()` (README, recent files)
5. Scratchpad (general + project-specific `.md` files)
6. Memory text (6 pools, ordered: Core → Lessons → RecentHistory → Buffer → ActiveSession → Recall)
7. FILES ALREADY READ THIS SESSION list

**Guardrails additions** (when enabled):
- Confirm plan guidance
- Completion audit warning
- Context pinning (original task always visible)
- Trajectory checkpoints every N tool calls

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Rule count | 3-5 high-level rules | 8 explicit rules + tool descriptions |
| Rule specificity | High-level ("read before editing") | Prescriptive ("NEVER re-read a file already in [STATE]") |
| Prompt size | ~1-2K tokens (estimated) | ~2.3K tokens (executor system prompt alone) |
| Customization | Full replacement or append | Config-driven, no runtime override |
| Few-shot examples | Yes (built-in) | Yes (in executor prompt) |
| Platform awareness | Yes (detected automatically) | Yes (Windows-specific tool redirects) |
| Dynamic content | Permission context, tools | STATE block, memory pools, project context, guardrails |
| User message content | User's actual text | `Task: {user_request}` wrapper |

**Gap**: BlipShell wraps the user's message in `Task: {text}` in executor mode. Claude Code passes the raw user message. The wrapping is unnecessary — the system prompt already establishes the role.

**Gap**: BlipShell has no `--system-prompt` or `--append-system-prompt` CLI flags. Users can't customize the system prompt at runtime.

**Gap**: BlipShell's rules are more prescriptive because weaker models need it. But rule #7 ("Do NOT narrate your thinking") fights against models that are trained to think out loud. Consider removing this for thinking-capable models.

---

## 4. Tool System — Complete Inventory

### Side-by-Side Tool Comparison

| Category | Claude Code | BlipShell | Notes |
|----------|------------|-----------|-------|
| **Read file** | `Read` (2000 lines default) | `read_file` (300 lines default, 1MB max) | CC reads more by default |
| **Write file** | `Write` (full file) | `write_file` (full file) | Equivalent |
| **Edit file** | `Edit` (old_code/new_code match) | `edit_file` (old_text/new_text, fuzzy matching) | BS has fuzzy match fallback |
| **List directory** | `List` (recursive optional) | `list_directory` (max depth 3) | CC unlimited depth |
| **Glob search** | `Glob` (pattern, directory) | `glob_files` (pattern, path, max 50) | BS caps at 50 results |
| **Grep search** | `Grep` (regex, type filter, context) | `grep_files` (regex, type_filter, multiline, max 50) | Both comparable |
| **Shell/Bash** | `Bash` (command, background, timeout) | `run_command` (command, timeout=30, background) | CC: 60s default, BS: 30s |
| **Web search** | `WebSearch` (query, max_results) | `web_search` (query, max_results=5) | Equivalent |
| **Web fetch** | `WebFetch` (url, headers) | `web_fetch` (url, prompt) | BS has LLM extraction prompt |
| **Ask user** | `AskUserQuestion` (question, choices) | `ask_user` (question, options, allow_free_text) | Equivalent |
| **Git status** | Via `Bash` | `git_status` (dedicated tool) | BS has structured output |
| **Git diff** | Via `Bash` | `git_diff` (dedicated tool) | BS has structured output |
| **Git add** | Via `Bash` | `git_add` (dedicated tool) | BS has structured output |
| **Git commit** | Via `Bash` | `git_commit` (dedicated tool) | BS has structured output |
| **Agent/subagent** | `Agent` (type, task, working_dir) | **MISSING** | Major gap |
| **Pin context** | `PinToContext` | **MISSING** (guardrails context_pinning is partial) | Gap |
| **Chrome DOM** | `ChromeGetDOM` | **MISSING** | Gap |
| **Chrome click** | `ChromeClick` | **MISSING** | Gap |
| **Chrome fill** | `ChromeFill` | **MISSING** | Gap |
| **Chrome screenshot** | `ChromeScreenshot` | **MISSING** | Gap |
| **Chrome eval** | `ChromeEval` | **MISSING** | Gap |
| **Code intelligence** | Via plugins (LSP) | **MISSING** | Gap |
| **Notebook edit** | `NotebookEdit` | **MISSING** | Minor gap |
| **Task management** | `TaskCreate/Get/List/Update/Stop/Output` | `start_background_task`, `check_background_task`, `list_background_tasks` | BS has background tasks, CC has richer task model |
| **Plan mode** | Permission mode toggle | `enter_plan_mode` / `exit_plan_mode` (tools) | Different mechanism |
| **Completion** | Implicit (no tool calls) | `task_complete` (explicit tool) | Architecturally different |
| **Confirm plan** | **MISSING** | `confirm_plan` (guardrails) | BS advantage |
| **Memory search** | **MISSING** | `search_memories` | BS advantage |
| **Save core memory** | **MISSING** | `save_core_memory` | BS advantage |
| **Session listing** | **MISSING** | `list_sessions` | BS advantage |
| **Session summary** | **MISSING** | `get_session_summary` | BS advantage |
| **Memory promotion** | **MISSING** | `promote_to_core_memory` | BS advantage |
| **Background check** | Via `TaskOutput` | `check_process` | Similar |
| **Workflow** | **MISSING** | `run_workflow` | BS advantage |
| **Project creation** | **MISSING** | `create_project` | BS advantage |

### Tool Counts

| | Claude Code | BlipShell |
|-|------------|-----------|
| Core tools | ~15 | ~30 |
| Chrome tools | 5 | 0 |
| Task tools | 6 | 3 |
| MCP tools | Dynamic | Dynamic |
| **Total (without MCP)** | **~26** | **~30** |

### Tool Description Quality

**Claude Code** tool descriptions (from system prompt):
- Brief, focused on *when* to use
- Includes anti-patterns ("Use Read instead of cat")
- References other tools ("Use Grep instead of Bash grep")
- Efficiency guidance built in

**BlipShell** tool descriptions:
- More verbose, includes parameter details
- Actionable error messages built into tool execution
- References `[STATE]` block for re-read prevention
- Windows-specific guidance embedded

**Example — Read file**:

*Claude Code*:
> Reads a file from the local filesystem. By default reads up to 2000 lines. You can optionally specify a line offset and limit.

*BlipShell*:
> Read a file from the project. Shows content with line numbers. Use start_line/max_lines for large files. Check [STATE] first — if you already read this file, don't read it again. Max file size: 1MB.

**Gap**: BlipShell's descriptions are more defensive (re-read warnings, size limits) because weaker models need guardrails. Claude Code's are cleaner.

---

## 5. Context Window & Compaction

### Claude Code

| Property | Value |
|----------|-------|
| Context window | 200K tokens (Sonnet/Opus), 100K (Haiku), 1M (Vertex/Bedrock) |
| Token counting | Server-side, accurate |
| Compaction trigger | ~95% capacity (configurable via `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`) |
| Manual compaction | `/compact [focus]` — with optional focus guidance |
| Compaction strategy | Clear older tool outputs → summarize conversation → re-inject CLAUDE.md |
| Task list preservation | Yes — tasks survive compaction |
| Pin to context | `PinToContext` tool — survives compaction |
| CLAUDE.md re-injection | Yes — full file re-loaded after compaction |

### BlipShell

| Property | Value |
|----------|-------|
| Context window | 131,072 tokens (local Ollama), varies by cloud endpoint |
| Token counting | Client-side heuristic: `len(text) // 4` (or tiktoken if available) |
| Compaction trigger | 85% of context_limit |
| Manual compaction | No manual command |
| Compaction strategy | Keep system + first user + last 5 tool result pairs + all assistant text |
| Forced completion | At 95%: strip all tools except `task_complete` |
| Context pinning | Guardrails `pinned_context` (original task + confirmed plan) |
| Memory re-injection | Not after compaction (memory is in initial system prompt) |

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Window size | 200K (2-8x larger) | 131K |
| Token accuracy | Exact (server-side) | Heuristic (~25% error) |
| Compaction trigger | 95% | 85% |
| Manual trigger | `/compact [focus]` | None |
| Focus-guided compaction | Yes ("compact, focus on the API work") | No |
| Content preservation | Older tool outputs cleared first | System + first user + last 5 tools |
| Pin mechanism | `PinToContext` tool | Guardrails context_pinning |
| Memory persistence | CLAUDE.md re-injected | Not re-injected after compaction |

**Critical gaps**:

1. **Token counting accuracy**: `len/4` is notoriously inaccurate for code (operators, punctuation count as tokens but are single chars) and non-English text. This means BlipShell's compaction triggers at the wrong time. Consider using tiktoken unconditionally or calling Ollama's tokenizer.

2. **No manual `/compact` command**: Users can't force compaction or guide it. Easy to add.

3. **No focus-guided compaction**: When compacting, there's no way to say "keep the API discussion, discard the debugging." Claude Code's `/compact focus on X` is elegant.

4. **Memory not re-injected**: After compaction, the memory context from the system prompt is preserved (it's in the system message which is kept), but if the system prompt was truncated, memory is lost. Claude Code explicitly re-injects CLAUDE.md.

5. **No PinToContext equivalent**: The guardrails `pinned_context` only works within executor mode with guardrails enabled. Claude Code's `PinToContext` works everywhere, anytime.

---

## 6. Memory & Persistence

This is where the systems diverge most dramatically.

### Claude Code

**Memory model**: File-based, user-curated + auto-curated.

| Component | Description |
|-----------|-------------|
| CLAUDE.md | User-written project instructions. Loaded at session start. |
| .claude/rules/*.md | Modular rule files with path-matching frontmatter |
| Auto memory (`~/.claude/projects/<project>/memory/`) | Claude-maintained. MEMORY.md (first 200 lines loaded) + topic files |
| Session memory | Conversation history for resume (`/resume`) |
| Worktree memory | Per-worktree context |

**Memory operations**:
- Read: Automatic at session start (CLAUDE.md + MEMORY.md first 200 lines)
- Write: Claude updates MEMORY.md with patterns, decisions, preferences
- Search: No semantic search. Linear scan of loaded files.
- Forget: Manual file editing
- Cross-session: MEMORY.md persists across sessions

**What Claude remembers**: Build commands, debugging insights, architecture notes, code style, workflow habits. All stored as plain text in markdown files.

### BlipShell

**Memory model**: Database-backed, multi-pipeline, semantic.

| Component | Description |
|-----------|-------------|
| SQLite `memories` table | All processed messages with rank, importance, type, tags |
| ChromaDB | Vector embeddings for semantic search (nomic-embed-text) |
| FTS5 index | Full-text keyword search |
| Entity graph | 31K entities, 56K relationships, 95K mentions |
| Core memories | User-curated permanent facts (preferences, identity) |
| Lessons | LLM-discovered behavioral advice + user feedback |
| Session summaries | Auto-generated per session |
| Project digests | Auto-maintained project status summaries |
| Tags | 16K+ tags from centroid + LLM + discovery |

**5-Pool Architecture**:
```
┌─────────────────────────────────────────────┐
│  Context Window (131K tokens)               │
│  ┌─────────┐ ┌──────────┐ ┌─────────┐      │
│  │  Core   │ │ Lessons  │ │ Recent  │      │
│  │  (10%)  │ │  (varies)│ │ History │      │
│  │  2048cap│ │          │ │  (15%)  │      │
│  └─────────┘ └──────────┘ └─────────┘      │
│  ┌──────────┐ ┌───────────┐ ┌────────┐     │
│  │  Buffer  │ │  Active   │ │ Recall │     │
│  │  (10%)  │ │  Session  │ │  (30%) │     │
│  │         │ │  (35%)    │ │ 8192cap│     │
│  └──────────┘ └───────────┘ └────────┘     │
│  [Ordered: Core→Lessons→History→Buffer→     │
│   Session→Recall to mitigate lost-in-middle]│
└─────────────────────────────────────────────┘
```

**Memory processing pipeline** (every message):
1. Noise filter (skip trivial messages)
2. LLM summarization (cloud: Groq gpt-oss-120b)
3. SQLite insert
4. ChromaDB embedding (nomic-embed-text, local)
5. Dedup check (similar memories → merge/archive/keep)
6. Tag assignment (centroid + LLM)
7. LLM rank + importance scoring (Groq llama-3.3-70b)

**Search pipeline** (8 steps):
1. Noise filter
2. Query rephrasing (optional)
3. ChromaDB semantic search (2-pass if project active)
4. FTS5 keyword search
5. RRF (Reciprocal Rank Fusion) scoring
6. Entity expansion (find related entities)
7. Post-filters (rank threshold, importance boost, tag overlap, project proximity, decay)
8. Sort by boosted score

**Entity graph**:
- Triples: (entity, relationship, entity) extracted from every memory
- Used in search expansion: query mentions "Python" → find related entities → expand results
- 31,469 entities, 56K relationships, 95K mentions
- Entity matching: substring pre-filter before regex (340x speedup)

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Storage | Markdown files | SQLite + ChromaDB + FTS5 |
| Semantic search | None (linear file scan) | ChromaDB + RRF fusion |
| Entity relationships | None | Full graph (31K entities) |
| Memory processing | None (raw text saved) | 7-step pipeline per message |
| Cross-session | MEMORY.md persists | Full database persists |
| Memory capacity | ~200 lines loaded per session | Unlimited (pool-budgeted per query) |
| Ranking/importance | None | Per-memory scores (0-5 rank, 0-1 importance) |
| Temporal decay | None | Exponential decay for old memories |
| Contradiction detection | None | LLM-based pairwise check |
| Consolidation | None | Automated merge of near-duplicates |
| Tags | None | 16K+ tags, centroid + LLM assigned |
| Nightly maintenance | None | 7-job pipeline (backup, cleanup, tag, prune, consolidate) |
| User curation | Edit MEMORY.md directly | `/core`, `/feedback`, `save_core_memory` tool |

**Assessment**: BlipShell's memory system is dramatically more sophisticated. This is its single biggest advantage. Claude Code's memory is essentially "a text file that Claude edits." BlipShell has a full knowledge management system with semantic search, entity graphs, temporal decay, and automatic maintenance.

**However**: Claude Code's simplicity means CLAUDE.md is always loaded, always accurate (because the user wrote it), and always relevant (because it's project-specific instructions). BlipShell's complexity means memories can be noisy, incorrectly ranked, or irrelevant — the pipeline quality depends on the weaker models running it.

---

## 7. Model Routing & Selection

### Claude Code

**Model selection**: User chooses (Opus, Sonnet, Haiku). One model per session.

| Control | Method |
|---------|--------|
| Default model | Config setting or env var |
| Switch model | `/model [name]` command |
| Fast mode | `/fast` toggle (same model, faster output) |
| Subagent model | Can differ per subagent type (Haiku for Explore) |
| Extended thinking | `Alt+T` toggle or config |
| Model override | `ANTHROPIC_MODEL` env var |

**Routing**: None. One model does everything — chat, code, search, summarization.

### BlipShell

**Task-type routing**: Different models for different operations.

| Task Type | Primary Model | Fallback |
|-----------|--------------|----------|
| tool_calling | glm-5:cloud (Gemini via Ollama) | gpt-oss:latest (local) |
| coding | glm-5:cloud | gpt-oss:latest |
| summarization | Groq gpt-oss-120b | glm4:latest (local) |
| ranking_importance | Groq llama-3.3-70b | qwen2.5:7b (local) |
| reasoning | qwen3:14b (local) | gpt-oss:latest |
| importance | qwen3:14b (local) | — |
| embedding | nomic-embed-text (local) | — |

**Endpoint cascade** (per task type):
1. Cloud endpoint (Groq, Gemini) — check rate limits
2. If rate limited or failed: next cloud endpoint
3. If all cloud failed: local Ollama fallback
4. Per-endpoint model overrides

**Rate limiting**: Per-endpoint RPM/RPD tracking. When limit hit → automatic failover.

**OllamaGate**: Serializes local Ollama calls (single GPU can only run one inference at a time). Priority levels: INTERACTIVE=0, BACKGROUND=2.

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Models used | 1 per session (Opus/Sonnet/Haiku) | 6+ models simultaneously |
| Task routing | None (one model does all) | Per-task-type routing |
| Fallback | None (one endpoint) | Multi-endpoint cascade |
| Rate limiting | Anthropic-managed | Client-side tracking + auto-failover |
| Cost | $3-75/M tokens (Anthropic pricing) | Free (local) or Groq free tier |
| Quality ceiling | Very high (Opus) | Low-medium (7B-14B) |
| Latency | ~1-3s first token | ~2-8s first token (varies by model/endpoint) |

**Assessment**: Claude Code's single-model approach works because the model is excellent. BlipShell's multi-model routing is necessary because no single small model is good at everything — summarization needs a different model than tool-calling. The complexity is justified.

---

## 8. File Editing & Code Intelligence

### Claude Code

**Edit mechanism**: `old_code` / `new_code` exact match.
- Character-range matching (not line-based)
- Old code must match exactly (whitespace matters)
- Creates checkpoint before edit (for rewind)
- Syntax validation via hooks (PostToolUse)
- Code intelligence via LSP plugins (type errors, definitions, references)

**Read pagination**: 2000 lines default, configurable start_line/max_lines.

**Code intelligence** (via plugins):
- TypeScript, Python, Java, C#, Go, Rust, etc.
- Jump to definition, find references
- Inline type errors and warnings
- Requires language server installation

### BlipShell

**Edit mechanism**: `old_text` / `new_text` with layered matching.
1. Exact match (try first)
2. Whitespace-normalized match (collapse whitespace differences)
3. Fuzzy match (difflib, threshold 0.6)

Plus:
- **Syntax linting**: `ast.parse()` validation for Python files after edit
- If syntax error → reject edit, return error with line number
- Path traversal protection: `_validate_within_root()`
- Symlink check: `_check_symlink()` rejects symlink targets

**Read pagination**: 300 lines default, configurable `start_line`/`max_lines`, 1MB max.

**Code intelligence**: None. Relies entirely on grep and glob for code navigation.

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Edit matching | Exact only | Exact → whitespace-normalized → fuzzy |
| Syntax validation | Via hooks (PostToolUse) | Built-in ast.parse() for Python |
| Checkpoint/rewind | Yes (undo to previous state) | No |
| Read default lines | 2000 | 300 |
| File size limit | No explicit limit | 1MB |
| Code intelligence | LSP via plugins | None |
| Path traversal protection | Sandboxed | `_validate_within_root()` |
| Symlink protection | Sandboxed | `_check_symlink()` |
| File cache | Tracks files read in [STATE] | Full content cache + files_read set |

**Gaps**:

1. **No code intelligence**: BlipShell has no LSP integration. The model must grep for definitions, type signatures, etc. This is a significant productivity gap for large codebases.

2. **No checkpoint/rewind**: Claude Code can rewind to any previous conversation state and undo all file changes. BlipShell has no equivalent.

3. **Read pagination default**: 300 lines vs 2000. BlipShell is more conservative, which means more tool calls to read a file. Consider increasing to 500-1000.

**Advantage**: BlipShell's fuzzy edit matching is genuinely useful for weaker models that can't reproduce exact whitespace. Claude Code's exact matching works because Claude is precise enough.

---

## 9. Permissions & Safety

### Claude Code

**Permission modes** (cycle with `Shift+Tab`):
1. **Default**: Ask before file edits and shell commands
2. **Auto-Accept Edits**: Auto-approve Edit/Write, still ask for Bash
3. **Plan Mode**: Read-only tools only

**Permission rules** (JSON syntax):
```json
{
  "permissions": {
    "allow": ["Bash(npm run test)", "Read(~/.zshrc)"],
    "ask": ["Bash(git push *)"],
    "deny": ["Bash(rm -rf *)", "Read(./.env)"]
  }
}
```

**Rule evaluation**: Deny → Ask → Allow (first match wins).

**Pattern matching**: Exact, wildcard (`*`), glob (`src/**`), domain (`domain:example.com`).

**Sandbox**: Bash commands can be sandboxed (configurable).

### BlipShell

**Approval system**:
- Per-tool approval: configurable set (default: write_file, edit_file, run_command, git_add, git_commit)
- Session-scoped: `/approve all` auto-approves for session, `/approve reset` reverts
- Destructive pattern detection in `run_command`: rm -rf, git reset --hard, SQL DROP, format drive

**Plan mode**: `enter_plan_mode` / `exit_plan_mode` tools — LLM self-restricts to read-only tools.

**Path traversal**: `_validate_within_root()` in all filesystem tools.

**Symlink check**: `_check_symlink()` blocks symlink targets in write/edit.

**SSRF protection**: `_is_ssrf_target()` blocks private IPs, localhost, cloud metadata in web_fetch.

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Permission modes | 3 modes (Default, Auto-Accept, Plan) | 2 modes (approval required, session-auto-approved) |
| Rule syntax | JSON with patterns (allow/ask/deny) | Config-driven tool set only |
| Pattern granularity | Per-command pattern matching | Per-tool-name only |
| Deny rules | Yes (block specific commands/paths) | No deny rules |
| Sandbox | Yes (Bash sandboxing) | No sandbox |
| Plan mode | Permission mode toggle | LLM-controlled via tools |
| Path traversal | Sandboxed | Explicit validation |
| SSRF protection | Yes | Yes |
| Destructive detection | Via deny rules | Regex pattern matching in run_command |

**Gaps**:

1. **No per-command permission rules**: Claude Code can say "allow `npm test` but deny `rm -rf`." BlipShell can only approve/deny entire tools.

2. **No deny rules**: Can't block specific file reads (e.g., `.env`) or specific commands. Only whole-tool approval.

3. **No sandbox**: Claude Code can sandbox Bash execution. BlipShell runs commands directly.

---

## 10. Research Mode & Web Integration

### Claude Code

**Research mode**: Not a separate explicit mode in the same way. Claude naturally uses web tools when needed. The `/research` capability is more about the model's built-in behavior.

**Web tools**:
- `WebSearch`: Query → results with titles, URLs, snippets
- `WebFetch`: URL → markdown content

**Auto-detection**: Not explicit. Claude decides when to search based on the question.

### BlipShell

**Research mode**: Explicit, with multiple trigger paths.

**Activation**:
1. `/research <query>` command — forces research mode
2. `!research <query>` prefix — forces research mode
3. Auto-detection with user confirmation:
   - Strong signals (immediate): investigate, explore, research, deep dive, compare, alternatives to, etc.
   - Weak signals (need 50+ chars + 2 matches or `?`): how does, what is, why does, explain
   - Anti-patterns (suppress): fix, add, create, build, implement, show me, status, check

**Behavior in research mode**:
- 3x tool budget (min 30 iterations)
- System prompt injection (not user message — avoids session history pollution):
  ```
  [RESEARCH MODE]
  - Use web_search for current information, best practices, and alternatives
  - Use web_fetch to read multiple sources and cross-reference
  - If exploring code, read multiple files and trace through the architecture
  - Synthesize findings into a structured, detailed response
  - Cite sources when using web information
  - Don't stop at the first answer — explore thoroughly
  ```

**Web tools**:
- `web_search`: Tavily (priority) → DuckDuckGo (fallback)
- `web_fetch`: HTML → text extraction, 512KB limit, optional LLM extraction prompt

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Explicit research mode | No | Yes, with 3x budget |
| Auto-detection | Model decides | Signal-based with user confirmation |
| Budget increase | None | 3x (min 30 iterations) |
| Prompt injection | None | System message with research guidance |
| Web search backend | Anthropic integration | Tavily → DuckDuckGo fallback |
| Web fetch LLM extraction | No | Yes (optional `prompt` parameter) |
| Citation guidance | Model decides | Explicit in prompt |

**Assessment**: BlipShell's research mode is more structured, which helps weaker models stay on task. The auto-detection with user confirmation is a nice UX touch. The 3x budget is important because web search + fetch takes many tool calls.

**Advantage**: BlipShell's `web_fetch` with LLM extraction prompt is unique — you can ask "extract the API rate limits from this page" instead of getting the whole page.

---

## 11. Sub-Agents & Delegation

### Claude Code

**Sub-agent system**: First-class, with typed agents and isolation.

| Agent Type | Model | Tools | Use Case |
|------------|-------|-------|----------|
| Explore | Haiku (fast) | Read-only | Codebase search |
| Plan | Inherit | Read-only | Research in plan mode |
| General-purpose | Inherit | All | Multi-step tasks |
| Bash | Inherit | Bash only | Terminal operations |
| statusline-setup | Sonnet | Config | Status line setup |
| Claude Code Guide | Haiku | Help | Feature questions |

**Custom sub-agents**: File-based (`.claude/agents/*.md`) or CLI-based.
- Configurable: tools, model, permissions, max turns, skills, memory, hooks
- Frontmatter YAML configuration
- Background execution support
- Worktree isolation (`isolation: worktree`)
- Persistent memory per agent

**Delegation**: Claude auto-delegates based on task + agent descriptions.

**Parallelism**: Multiple subagents can run concurrently with isolated contexts.

### BlipShell

**Sub-agent system**: None.

**Workaround**: Background tasks (`start_background_task` tool) run prompts asynchronously, but:
- No isolated context
- No dedicated tool set per task
- No agent specialization
- No parallel execution with context isolation
- No auto-delegation

### Gap Analysis

This is **the largest architectural gap** between the two systems.

| Feature | Claude Code | BlipShell |
|---------|------------|-----------|
| Agent spawning | Yes | No |
| Context isolation | Yes (each agent gets fresh context) | No |
| Parallel research | Yes (multiple agents concurrently) | No |
| Specialized agents | Yes (typed, custom) | No |
| Agent memory | Yes (per-agent persistent) | No |
| Worktree isolation | Yes (git worktree per agent) | No |
| Auto-delegation | Yes | No |
| Custom agent definition | Yes (.claude/agents/ files) | No |

**Impact**: Without sub-agents, BlipShell cannot:
1. Research multiple topics in parallel
2. Isolate large tool outputs from the main conversation
3. Give different tasks different tool sets
4. Run background verification while continuing conversation
5. Use a cheaper/faster model for exploration while keeping the main model for editing

**Recommendation**: This is the highest-impact gap to close. Consider implementing:
1. A simple `spawn_agent` tool that creates a new ChatLoop with isolated messages
2. Agent types defined via config (like BlipShell's existing model routing)
3. Results returned as a summary to the main conversation
4. Background execution via existing job queue

---

## 12. Git Integration

### Claude Code

**Approach**: Native git commands via `Bash` tool. No dedicated git tools.

**Git-aware features**:
- Current branch shown in metadata
- Recent commit history accessible
- `.gitignore` respected in file search
- `/diff` interactive diff viewer
- `/review` for PR review
- Worktree support for parallel branches
- Attribution: `Co-Authored-By: Claude <noreply@anthropic.com>`
- GitHub integration via `gh` CLI

### BlipShell

**Approach**: Dedicated git tools with structured output.

| Tool | Description |
|------|-------------|
| `git_status` | Parses porcelain output into Staged/Modified/Untracked sections |
| `git_diff` | Shows diff with optional file filter, staged flag |
| `git_add` | Stage specific files |
| `git_commit` | Commit with message |

Plus `run_command` for anything else.

**Git-aware features**:
- `.gitignore` respected in grep/glob
- File changes tracked per session (`/changes` command)
- No interactive diff viewer
- No PR review
- No worktree support
- No attribution convention

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Git tools | Via Bash | Dedicated tools |
| Interactive diff | `/diff` viewer | No |
| PR review | `/review` | No |
| Worktrees | Yes | No |
| Attribution | Built-in | None |
| Branch display | In metadata | In status toolbar |
| GitHub integration | `gh` CLI | Via MCP (planned) |

**Assessment**: Claude Code's git integration is more complete (diff viewer, PR review, worktrees). BlipShell's dedicated git tools give structured output, which is better for weaker models that can't parse raw `git status` output reliably.

---

## 13. MCP Integration

### Claude Code

**Transport**: HTTP (recommended), SSE (legacy), stdio.
**Scopes**: Local (project), Project (team), User (global).
**Authentication**: OAuth 2.0 support for remote servers.
**Management**: `claude mcp add/remove/list/get` CLI commands + `/mcp` interactive.
**Tool naming**: `mcp__<server>__<tool>` (double underscore).
**Dynamic updates**: Servers can send `list_changed` notifications.
**Tool search**: Auto-enabled when MCP tools > 10% of context.
**Resources**: `@server:protocol://path` syntax for referencing.
**Prompts**: `/mcp__server__prompt` — MCP prompts as slash commands.

### BlipShell

**Transport**: stdio only (Phase 1).
**Scopes**: Config-driven (`mcp_servers` list in config.yaml).
**Authentication**: Environment variables only.
**Management**: `/mcp` and `/mcp tools [server]` commands.
**Tool naming**: `mcp_{server}_{tool}` (single underscore).
**Dynamic updates**: None.
**Tool search**: None.
**Resources**: None.
**Prompts**: None.

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Transports | HTTP, SSE, stdio | stdio only |
| Authentication | OAuth 2.0 | Env vars only |
| Dynamic tool updates | Yes | No |
| Tool search (for many tools) | Auto-enabled | No |
| MCP resources | Yes (`@` syntax) | No |
| MCP prompts | Yes (as slash commands) | No |
| Scopes | Local, Project, User | Config-only |
| Management CLI | Full (add/remove/list/get) | List + show only |

**Gaps**: HTTP transport and OAuth are important for cloud MCP servers (GitHub, Asana, etc.). Tool search is needed when many MCP tools are connected (prevents context bloat). Resources and prompts are nice-to-have.

---

## 14. Hooks & Lifecycle Events

### Claude Code

**17+ hook events**:
- SessionStart, UserPromptSubmit, InstructionsLoaded
- PreToolUse, PermissionRequest, PostToolUse, PostToolUseFailure
- Notification, SubagentStart, SubagentStop, Stop
- TeammateIdle, TaskCompleted, ConfigChange
- WorktreeCreate, WorktreeRemove, PreCompact, SessionEnd

**Hook types**: Command (shell), HTTP (webhook), Prompt (LLM), Agent (multi-turn).

**Configuration**: Per-project, per-user, per-subagent (frontmatter), managed policy.

**Decision output**: Allow, Deny, Ask (with reasons).

**Use cases**: Auto-linting, CI triggers, security scanning, logging, custom validation.

### BlipShell

**Hook events**: None.

The closest equivalents:
- `on_tool_executed` callback in ChatLoop (internal, not user-configurable)
- `on_tool_display` callback for Rich rendering
- `on_pause_check` callback for user interruption
- GuardrailsEngine's correction detector (regex on user messages)

### Assessment

**This is a significant gap**. Hooks are how power users customize Claude Code's behavior without modifying source code. Without hooks, BlipShell users must modify Python code to add custom behavior.

Key hooks BlipShell should add:
1. **PreToolUse**: Allow users to block/modify tool calls (security, linting)
2. **PostToolUse**: Allow users to run post-actions (e.g., auto-format after edit)
3. **SessionStart/End**: Custom setup/teardown
4. **UserPromptSubmit**: Input preprocessing

---

## 15. CLI Commands & UX

### Command Count Comparison

| Category | Claude Code | BlipShell |
|----------|------------|-----------|
| Session management | ~10 (/clear, /resume, /fork, /rename, /export, /rewind, /compact) | ~5 (/save, /status, /quit, /cleanup, /changes) |
| Memory | /memory | /core, /memory, /context, /feedback |
| Model control | /model, /fast | /think, /reflect, /guardrails, /verbose, /expand |
| Development | /diff, /review, /pr-comments, /security-review | /code, /plan, /research, /offload |
| Observability | /cost, /context, /stats, /insights | /flow, /health, /tokens, /nightly, /status |
| Configuration | /config, /permissions, /hooks, /keybindings | /approve |
| Project | /add-dir, /init | /project, /projects |
| Integration | /mcp, /chrome, /ide, /plugin | /mcp, /workflow |
| Account | /login, /logout, /usage, /upgrade, /passes | — |
| Misc | /help, /doctor, /release-notes, /stickers, /mobile, /desktop, /terminal-setup | /help |
| **Total** | **~50+** | **~30** |

### Unique Claude Code Commands

| Command | What it does | BlipShell equivalent |
|---------|-------------|---------------------|
| `/compact [focus]` | Guided context compaction | None |
| `/fork [name]` | Branch conversation | None |
| `/rewind` | Undo to previous checkpoint | None |
| `/diff` | Interactive diff viewer | None |
| `/review` | PR review | None |
| `/pr-comments` | Fetch PR comments | None |
| `/security-review` | Security analysis | None |
| `/insights` | Session analysis report | None |
| `/chrome` | Browser integration | None |
| `/plugin` | Plugin management | None |
| `/hooks` | Hook management | None |
| `/agents` | Subagent management | None |
| `/desktop` | Switch to desktop app | None |
| `/mobile` | Mobile handoff | None |
| `/copy` | Copy response to clipboard | None |
| `/output-style` | Change output verbosity | None |
| `/vim` | Vim editing mode | None |
| `/stats` | Usage statistics + streaks | None |

### Unique BlipShell Commands

| Command | What it does | Claude Code equivalent |
|---------|-------------|----------------------|
| `/core` | View core memories + lessons | `/memory` (partial) |
| `/feedback <msg>` | Save feedback as lesson | None |
| `/flow [n]` | Conversation flow events | None |
| `/health` | Database audit | `/doctor` (partial) |
| `/nightly` | Maintenance job runner | None |
| `/guardrails` | Toggle guardrails | None |
| `/research <query>` | Explicit research mode | None |
| `/offload` | Submit to remote endpoint | None |
| `/think`, `/reflect` | Toggle LLM modes | Extended thinking toggle (`Alt+T`) |
| `/verbose`, `/expand` | Tool output control | None |
| `/workflow` | Workflow management | None |
| `/project digest` | Project status digest | None |
| `/tasks` | Background task management | `/tasks` (similar) |
| `/tokens` | Per-endpoint token/cost tracking | `/cost` (similar) |

---

## 16. Streaming & Real-Time Feedback

### Claude Code

- Real-time token streaming via SSE
- Thinking spinner (customizable via `spinnerVerbs`)
- Turn duration shown (configurable)
- Tool call results shown inline
- Diff preview for edits
- Status line (customizable)

### BlipShell

- Token-by-token streaming via Ollama API
- Thinking spinner ("[dim]Thinking...[/dim]" until first token)
- Rich tool call display with icons (✎ edit, + write, $ command, ● complete, ✘ error)
- Pre-execution indicator (⏳ tool_name)
- Edit diffs always shown (colored)
- Session/project/endpoint/context in bottom toolbar
- Escape to cancel mid-generation
- `/verbose` toggle for full tool output
- `/expand` to review tool batch history (last 50 batches)

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Token streaming | Yes | Yes |
| Cancel mid-response | Yes (Esc) | Yes (Esc) |
| Thinking spinner | Yes (customizable) | Yes (fixed text) |
| Tool result display | Inline | Rich icons + condensed/verbose toggle |
| Edit diffs | Yes | Yes (always shown) |
| Status bar | Customizable | Fixed toolbar (session/project/endpoint/context) |
| Tool batch history | No | Yes (`/expand` for last 50) |
| Turn duration | Configurable | Not shown |
| Pre-execution indicator | No | Yes (⏳ icon) |

**Advantage**: BlipShell's tool display is richer — icons per tool type, pre-execution indicator, verbose toggle, expandable history. Claude Code's is simpler but sufficient.

---

## 17. Error Handling & Recovery

### Claude Code

**Tool errors**: Return error messages to the model. Model decides next action.

**Endpoint errors**: Managed by Anthropic infrastructure. User sees rate limit messages.

**Edit failures**: Checkpoint allows rewind. Model retries with corrected content.

**Session recovery**: `/resume` to continue interrupted sessions.

### BlipShell

**Tool errors**: Actionable error messages that guide next action:
- "File not found. Use list_directory to see available files."
- "text not found in file.py (350 lines). Try read_file to see current contents."
- "grep_files expects a directory, not a file. Use read_file for specific files."
- "Command timed out after 30s. Use background=true for long-running commands."

**Endpoint errors**: Multi-level recovery:
1. Model error → mark_model_failed(), try fallback model
2. Endpoint error → record_failure(), try fallback endpoint
3. Rate limit → auto-failover to next endpoint
4. Two attempts (primary + fallback)

**Edit failures**: Syntax lint (Python) → reject edit, return error with line number.

**Session recovery**: `--continue` or `--session N` to resume. Startup sweep reprocesses failed messages.

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Error message quality | Basic | Actionable (guide next action) |
| Multi-endpoint fallback | None | Yes (cloud → cloud2 → local) |
| Edit syntax validation | Via hooks | Built-in (ast.parse) |
| Edit rewind | Yes (checkpoint) | No |
| Session recovery | `/resume` | `--continue` / startup sweep |
| Rate limit handling | Anthropic-managed | Client-side auto-failover |

**Advantage**: BlipShell's actionable error messages are a genuine improvement. Every error tells the model what to do next. This is based on Anthropic's own research ("error messages are prompts").

---

## 18. Configuration System

### Claude Code

**Hierarchy** (highest to lowest precedence):
1. Managed policy (IT-deployed, cannot override)
2. CLI arguments
3. `.claude/settings.local.json` (project, gitignored)
4. `.claude/settings.json` (project, team-shared)
5. `~/.claude/settings.json` (user global)

**100+ environment variables** for fine-grained control.

**Schema validation**: JSON Schema published at schemastore.org.

### BlipShell

**Single config file**: `config.yaml`

**Config model**: `BlipShellConfig` in `blipshell/models/config.py` — dataclass-based.

**Key sections**: models, endpoints, memory, guardrails, mcp_servers, pii, llm.

**Per-model settings**: `ModelSettingsRegistry` for per-model overrides.

**No environment variable system** (except for API keys in endpoint config via `${ENV_VAR}` syntax).

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Config files | 5+ levels of precedence | 1 YAML file |
| Team sharing | `.claude/settings.json` | Config committed to git |
| Schema validation | Published JSON Schema | Dataclass validation |
| Environment variables | 100+ | Only for API keys |
| Runtime overrides | CLI flags | None |
| Per-project config | Yes (local + shared) | Single global config |
| Managed policies | Yes (IT can lock settings) | No |

**Gap**: BlipShell has no per-project settings override. A `.blipshell.yaml` per-project that merges with global config would be valuable.

---

## 19. Cost & Token Management

### Claude Code

- `/cost` shows input/output tokens, total cost, model
- `--max-budget-usd 5.00` for budget caps
- Prompt caching reduces input cost by ~90%
- Per-model pricing visible
- `/usage` shows plan tier and limits
- `/stats` shows historical usage
- Fast mode for cost reduction

### BlipShell

- `/tokens` shows per-endpoint token usage with cost breakdown
- No budget cap
- No prompt caching (Ollama doesn't support it; cloud via OpenAI SDK may)
- Per-endpoint cost rates in config (but mostly free tier / local)
- No usage statistics across sessions

### Comparison

| Aspect | Claude Code | BlipShell |
|--------|------------|-----------|
| Cost tracking | Yes (`/cost`) | Yes (`/tokens`) |
| Budget caps | Yes (`--max-budget-usd`) | No |
| Prompt caching | Yes (90% savings) | No |
| Historical stats | Yes (`/stats`) | No |
| Cost model | Pay per token | Mostly free (local + free tiers) |

**Note**: Cost management is less critical for BlipShell because local inference is free and cloud free tiers are used. But if BlipShell adds paid cloud endpoints, prompt caching and budget caps become important.

---

## 20. Testing & Quality

### Claude Code

- Internal testing (not publicly documented)
- Hooks system allows custom test integration
- `/security-review` for security scanning
- PostToolUse hooks for post-edit validation

### BlipShell

- 278+ unit tests (18s, all passing)
- Headless test harness (`blipshell test`): --canned (3 tests), --stress (65 tests), --simple-chat (14 tests)
- Multi-turn simulation system (26 scenarios, 7 categories)
- Pipeline benchmarks (`benchmark_all.py`, `benchmark_pipeline_speed.py`)
- Model quality benchmarks (reasoning, entity extraction, contradiction detection)
- Database audit (`scripts/audit_db.py`)
- Nightly job for automated testing (`simulate_quick`)

### Assessment

BlipShell has significantly more visible testing infrastructure. This makes sense — it's compensating for model unreliability. Claude Code doesn't need the same level of testing because the model is more predictable.

---

## 21. Privacy & PII

### Claude Code

- Data handling per Anthropic's privacy policy
- No built-in PII filtering
- `.env` and credential files excluded from commits
- Permission rules can deny reading sensitive files

### BlipShell

- **Presidio NER + regex fallback** for PII detection:
  - Names, addresses, locations (via spaCy NER)
  - Email, phone, SSN, credit card, API keys, AWS keys (regex)
- **Per-endpoint toggle**: `pii_sanitize` flag
- **Auto-detect**: Enabled for `openai` provider, configurable for local-cloud
- **Local bypass**: Local LLM calls NOT sanitized (preserves search quality)
- **Config toggle**: `pii.enabled` in config.yaml

### Assessment

BlipShell's PII sanitization is a significant advantage for users who send data to cloud endpoints. Claude Code relies on Anthropic's server-side data handling policies but doesn't filter PII client-side.

---

## 22. What Claude Code Has That BlipShell Doesn't

### Critical Gaps (should implement)

1. **Sub-agents with context isolation** — The biggest gap. Enables parallel research, specialized roles, context budget management.

2. **Worktree isolation** — Parallel branch work without switching. Essential for multi-task workflows.

3. **Prompt caching** — When using cloud endpoints, every turn re-sends full context. Caching could save 90% of input tokens.

4. **Code intelligence / LSP** — Jump to definition, find references, type errors. Massive productivity gap for large codebases.

5. **Checkpoint / rewind** — Undo conversation + file changes to any point. Safety net for experimentation.

6. **Hooks system** — User-configurable lifecycle hooks. Power-user extensibility without code changes.

### Important Gaps (should consider)

7. **Interactive diff viewer** (`/diff`) — Visual review of all changes.

8. **Focus-guided compaction** (`/compact focus on X`) — Users can guide what to keep.

9. **Per-command permission rules** — Allow `npm test`, deny `rm -rf /`. Not just per-tool.

10. **Sandbox mode** — Bash command sandboxing.

11. **Plugin system** — Third-party extensions.

12. **Fork/branch conversations** — Explore alternatives without losing original thread.

13. **Copy to clipboard** (`/copy`) — Quick access to response content.

### Nice-to-Have Gaps

14. **Desktop app / mobile handoff** — Cross-platform UX.
15. **PR review** (`/review`) — Built-in PR analysis.
16. **Security review** (`/security-review`) — Automated security scan.
17. **Output style modes** (Default/Explanatory/Learning) — Adjustable verbosity.
18. **Vim editing mode** — Power-user input.
19. **Custom spinner text** — Personalization.
20. **Usage statistics / streaks** (`/stats`) — Engagement tracking.

---

## 23. What BlipShell Has That Claude Code Doesn't

### Significant Advantages

1. **Persistent semantic memory** — 5-pool system with summarization, ranking, entity graphs, temporal decay, contradiction detection, consolidation. Claude Code has MEMORY.md (a text file).

2. **Multi-endpoint failover** — Automatic cascade across cloud and local endpoints. Claude Code has one endpoint.

3. **PII sanitization** — Client-side Presidio NER + regex. Claude Code has none.

4. **Guardrails system** — Completion audit, trajectory monitoring, correction detection, context pinning, requirement checklist. Claude Code has none.

5. **Model-agnostic design** — Works with any Ollama/OpenAI-compatible model. Claude Code works only with Claude.

6. **Zero-cloud option** — Can run 100% locally. Claude Code requires Anthropic API.

7. **Entity relationship graph** — 31K entities, 56K relationships, used in search expansion.

8. **Nightly maintenance** — Automated backup, cleanup, tagging, consolidation, prune.

9. **Project digests** — Auto-maintained project summaries across sessions.

10. **Background memory processing** — Every message goes through a 7-step pipeline.

11. **Research mode** — Explicit mode with 3x budget, auto-detection, research-specific prompts.

12. **Conversation flow observability** (`/flow`) — Turn-by-turn event breakdown.

13. **Actionable error messages** — Every tool error guides the model's next action.

14. **Tool call deduplication** — System-enforced, prevents wasted calls.

15. **Fuzzy edit matching** — Whitespace-normalized + fuzzy fallback for imprecise models.

16. **Task-type model routing** — Different models for summarization vs coding vs reasoning.

17. **OllamaGate serialization** — Prevents GPU contention on local inference.

18. **State injection** — `[STATE]` block gives model explicit awareness of budget, files, actions.

19. **Database audit** (`/health`) — 8-category integrity check.

20. **Per-endpoint cost tracking** (`/tokens`) — Granular cost by endpoint.

---

## 24. Prompt Text Comparison

### Executor System Prompt

**Claude Code** (inferred structure):
```
You are Claude Code, an AI assistant made by Anthropic...

# Tools
[Tool descriptions with usage guidance]

# Critical Rules
1. Explore before implementing
2. Read files before editing
3. Verify changes after making them
4. Prefer simple solutions
5. Ask when uncertain

# Error Recovery
[Guidance for when things go wrong]

# Context
[Permission mode, user preferences]
```

**BlipShell** (`prompts.py:executor_system_prompt()`):
```
You are a coding agent. You complete tasks autonomously using tools.

[Platform: Windows] Use dedicated tools...

# Rules
1. PLAN first — state your approach in 1-3 sentences...
2. Read before editing. NEVER re-read a file already in [STATE].
3. Make MINIMAL changes...
4. If something fails twice, use ask_user...
5. Call task_complete when DONE...
6. Each tool's description explains when/how to use it...
7. Do NOT narrate your thinking...
8. For complex multi-file tasks, use enter_plan_mode...
```

**Key differences**:
- Claude Code: 3-5 high-level rules, trusts model to infer details
- BlipShell: 8 prescriptive rules, explicit about what NOT to do
- Claude Code: "verify changes" (implies running tests). BlipShell: no equivalent.
- BlipShell: "NEVER re-read" (explicit). Claude Code: implicit via file tracking.
- BlipShell: "call task_complete when DONE" (explicit tool). Claude Code: no completion tool needed.
- BlipShell: "Do NOT narrate" (fights model training). Claude Code: no such rule.

### Research Mode Prompt

**Claude Code**: No explicit research prompt. Model uses web tools naturally.

**BlipShell** (injected as system message):
```
[RESEARCH MODE]
- Use web_search for current information, best practices, and alternatives
- Use web_fetch to read multiple sources and cross-reference
- If exploring code, read multiple files and trace through the architecture
- Synthesize findings into a structured, detailed response
- Cite sources when using web information
- Don't stop at the first answer — explore thoroughly
```

### Memory Processing Prompts

**Claude Code**: No memory processing prompts (memories are raw text in MEMORY.md).

**BlipShell** (key prompts):
- `summarize_memory()`: "Summarize this message in one sentence..."
- `rank_and_importance()`: "Rate this memory: rank 1-5, importance 0.0-1.0..."
- `extract_entities()`: "Extract entity relationship triples..."
- `extract_lesson()`: "Extract behavioral advice (NOT facts)..."
- `detect_contradiction()`: "Do these two memories contradict each other?"
- `validate_task_completion()`: "Does this summary address all requirements?"

---

## 25. Specific Thresholds & Constants

### File Operations

| Constant | Claude Code | BlipShell |
|----------|------------|-----------|
| Read default lines | 2000 | 300 |
| Read max lines | No explicit limit | Configurable (start_line/max_lines) |
| Max file size | No explicit limit | 1MB (1,048,576 bytes) |
| Grep max results | Not documented | 50 |
| Glob max results | Not documented | 50 |
| Shell timeout | 60s default | 30s default |
| Shell max output | Configurable (env var) | 8000 chars |
| Web fetch max size | Not documented | 512KB (524,288 bytes) |
| Web fetch timeout | Not documented | 15s |

### Context Management

| Constant | Claude Code | BlipShell |
|----------|------------|-----------|
| Context window | 200K tokens | 131K tokens |
| Compaction trigger | ~95% | 85% |
| Token counting | Server-side (exact) | `len/4` heuristic |
| Forced completion | Not documented | 95% (strip all tools except task_complete) |
| History preserved | Conversation summary | System + first user + last 5 tool pairs |
| MEMORY.md lines loaded | 200 | N/A (pool-budgeted) |

### Agent Loop

| Constant | Claude Code | BlipShell |
|----------|------------|-----------|
| Simple chat budget | Not explicitly limited | 5 tool calls |
| Executor budget | Not explicitly limited | 50 tool calls |
| Research budget | N/A | 3x (min 30) |
| Max parallel tools | 8 (configurable) | 8 (Semaphore) |
| Budget wind-down | Not documented | 80% budget: inject warning |
| Dedup check | Not documented | Cross-batch + within-batch |
| Inline text threshold | N/A | 200 chars |
| Guardrails interval | N/A | Every 5 tool calls |
| Max audit retries | N/A | 2 |

### Memory System (BlipShell only)

| Constant | Value |
|----------|-------|
| Core pool % | 10% (hard cap 2048 tokens) |
| Recall pool % | 30% (hard cap 8192 tokens) |
| ActiveSession pool % | 35% |
| RecentHistory pool % | 15% |
| Buffer pool % | 10% |
| Search n_results | 20 |
| Min rank threshold | 3 |
| Similarity threshold | 0.5 |
| Importance boost weight | 0.2 |
| Tag overlap boost | 0.1 |
| Project boost | 0.15 |
| Dedup similarity | 0.7 |
| RRF k parameter | 60 |
| Entity regex speedup | 340x (substring pre-filter) |

---

## 26. Recommendations: Closing the Gap

### Tier 1: High Impact, Achievable

1. **Sub-agent system** (Priority #1)
   - Spawn isolated ChatLoop instances with dedicated tools/budget
   - Agent types: Explore (read-only, smaller model), General (all tools), Research (web tools + 3x budget)
   - Background execution via existing job queue
   - Return summary to main conversation
   - Estimated complexity: HIGH but highest impact

2. **Hooks system** (Priority #2)
   - PreToolUse, PostToolUse, SessionStart, SessionEnd events
   - Command hooks (shell script execution)
   - Config-driven (add `hooks` section to config.yaml)
   - Estimated complexity: MEDIUM

3. **`/compact [focus]` command** (Priority #3)
   - Manual compaction trigger with optional focus guidance
   - Re-inject memory context after compaction
   - Estimated complexity: LOW

4. **Accurate token counting** (Priority #4)
   - Always use tiktoken (already optional dependency)
   - Or call Ollama's `/api/tokenize` endpoint
   - Estimated complexity: LOW

### Tier 2: Important, Medium Effort

5. **Checkpoint / rewind system**
   - Snapshot conversation state + file state at each turn
   - `/rewind [n]` to restore to turn N
   - Git stash or shadow copies for file state

6. **Per-command permission rules**
   - Pattern syntax: `run_command(npm test)`, `edit_file(src/**)`, `run_command(!rm *)`
   - Deny/Ask/Allow evaluation order
   - Config-driven

7. **Interactive diff viewer**
   - `/diff` shows all uncommitted changes
   - Per-turn diff tracking (already have `/changes`)
   - Rich formatted inline diff

8. **PinToContext tool**
   - Explicit pinning that survives compaction
   - Separate from guardrails (always available)

9. **System prompt customization flags**
   - `--system-prompt`, `--append-system-prompt`
   - Per-project prompt override in config

### Tier 3: Nice-to-Have

10. **Fork/branch conversations** — `/fork [name]`
11. **Copy to clipboard** — `/copy`
12. **Output style modes** — Default/Verbose/Learning
13. **Per-project config** — `.blipshell.yaml` that merges with global
14. **Usage statistics** — `/stats` with historical tracking
15. **Plugin system** — Third-party tool registration
16. **Code intelligence** — LSP integration (very high effort)

### What NOT to Copy

Some Claude Code features don't make sense for BlipShell:

- **Desktop app / mobile handoff** — Different target audience
- **Account management (/login, /logout, /upgrade)** — BlipShell is self-hosted
- **Stickers** — Marketing, not functionality
- **Chrome integration** — Low-priority, high-complexity
- **Managed policies** — No enterprise deployment scenario

---

## Summary Scorecard

| Dimension | Claude Code | BlipShell | Winner |
|-----------|:---------:|:---------:|:------:|
| Model quality | ★★★★★ | ★★☆☆☆ | CC |
| Agent loop simplicity | ★★★★★ | ★★☆☆☆ | CC |
| Scaffolding for weak models | ★☆☆☆☆ | ★★★★★ | BS |
| Memory system | ★★☆☆☆ | ★★★★★ | BS |
| Tool count | ★★★★☆ | ★★★★☆ | Tie |
| Context management | ★★★★★ | ★★★☆☆ | CC |
| Sub-agents | ★★★★★ | ☆☆☆☆☆ | CC |
| MCP integration | ★★★★★ | ★★☆☆☆ | CC |
| Hooks / extensibility | ★★★★★ | ☆☆☆☆☆ | CC |
| Permissions / safety | ★★★★★ | ★★★☆☆ | CC |
| Multi-model routing | ☆☆☆☆☆ | ★★★★★ | BS |
| PII sanitization | ★☆☆☆☆ | ★★★★★ | BS |
| Guardrails | ☆☆☆☆☆ | ★★★★☆ | BS |
| Error messages | ★★★☆☆ | ★★★★★ | BS |
| Git integration | ★★★★☆ | ★★★☆☆ | CC |
| Research mode | ★★★☆☆ | ★★★★☆ | BS |
| Testing infrastructure | ★★★☆☆ | ★★★★★ | BS |
| CLI UX polish | ★★★★★ | ★★★★☆ | CC |
| Streaming / real-time | ★★★★★ | ★★★★☆ | CC |
| Configuration flexibility | ★★★★★ | ★★★☆☆ | CC |
| Privacy / data control | ★★★☆☆ | ★★★★★ | BS |
| Cost (to user) | ★★☆☆☆ | ★★★★★ | BS |
| Nightly maintenance | ☆☆☆☆☆ | ★★★★★ | BS |
| Entity relationships | ☆☆☆☆☆ | ★★★★★ | BS |

**Overall**: Claude Code wins on model quality, simplicity, and ecosystem (sub-agents, hooks, MCP, permissions). BlipShell wins on memory, privacy, cost, resilience (multi-model routing, fallbacks), and scaffolding for weaker models. The systems serve different audiences and philosophies.

---

*Generated by Claude Code (Opus 4.6), 2026-03-05*
