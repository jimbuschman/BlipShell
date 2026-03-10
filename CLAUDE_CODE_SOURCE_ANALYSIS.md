# Claude Code Source Code Analysis (v2.1.72)

**Date**: 2026-03-10
**Source**: `@anthropic-ai/claude-code@2.1.72` npm package, `cli.js` (12MB minified)
**Method**: 4 parallel analysis agents examining minified source for string literals, constants, and patterns
**Related**: See `COMPARISON_CLAUDE_CODE_VS_BLIPSHELL.md` (blog/docs-based, 2026-02-27) and `PARITY_AUDIT_CLAUDE_CODE_VS_BLIPSHELL.md` (codebase audit, 2026-03-02)

This document captures findings from reverse-engineering Claude Code's actual source code.
Unlike the earlier comparison docs (based on blogs and public docs), these findings are
verified against the real implementation.

---

## Table of Contents

1. [UI & Display Layer](#1-ui--display-layer)
2. [Context & Compaction](#2-context--compaction)
3. [Tool Result Handling](#3-tool-result-handling)
4. [State Injection & Attachments](#4-state-injection--attachments)
5. [Budget & Wind-Down](#5-budget--wind-down)
6. [Hooks System](#6-hooks-system)
7. [Memory & Session Notes](#7-memory--session-notes)
8. [MCP Integration](#8-mcp-integration)
9. [Sub-Agents & Forking](#9-sub-agents--forking)
10. [Git Integration & Safety](#10-git-integration--safety)
11. [Permissions & Safety Classifier](#11-permissions--safety-classifier)
12. [Telemetry](#12-telemetry)
13. [Better of the Two — Per System](#13-better-of-the-two--per-system)
14. [Implementation Priority List](#14-implementation-priority-list)

---

## 1. UI & Display Layer

Claude Code uses **React/Ink** (terminal React renderer) for all UI. The entire terminal output
is a React component tree with flexbox layout.

### Tool Call Display

Each tool definition has **6 render methods**:
- `renderToolUseMessage(input, options)` — header (filename, command)
- `renderToolUseProgressMessage(progressMessages, options)` — progress during execution
- `renderToolResultMessage(result, progressMessages, options)` — result after completion
- `renderToolUseErrorMessage(content, options)` — error display
- `renderToolUseRejectedMessage(input, options)` — permission denied
- `renderToolUseQueuedMessage()` — queued/pending

**Status dots** (`F56` component):
- Blinking dot (600ms) while in-progress
- Green checkmark on success (`color:"success"`)
- Red X on error (`color:"error"`)
- Dim dot while unresolved
- Followed by **bold tool name** + parameters in parentheses

**Result truncation**: Results over **10 lines** truncated with dim `"... +N lines (/verbose to see all)"`.

**Parallel calls**: Rendered as individual items with separate status dots, not collapsed.

**Pending approval**: Shows dim "Waiting for permission..." text.

### Markdown Rendering

- Full **marked** library (markedjs) bundled — lexer, parser, tokenizer, renderer
- **highlight.js** v10.7.3 for code block syntax highlighting (full language detection)
- **Streaming-aware**: `SX4` component buffers incomplete markdown, only finalizes complete token sequences
- All standard markdown: headings, lists, tables, code blocks (fenced + indented), blockquotes, inline formatting, links, horizontal rules
- Tables rendered via dedicated `RX4` component
- `syntaxHighlightingDisabled` setting to turn off

### Spinner / Progress

**Two modes**:
1. **Full** (`Jt9`): animated dots + tool count + token count + duration + cost + teammate status
2. **Brief** (`Mt9`): "Thinking..." with 3-frame animation at 400ms

**Details**:
- Random verb pool ("Pondering", "Thinking", etc.) from `c56()` — customizable via `spinnerVerbs` setting (append or replace modes)
- Braille-like dot animation (`wQ6()`) at 120ms intervals
- Reduced motion preference falls back to static `"..."`
- Thinking duration badge when extended thinking completes
- Tips and task queue previews in full mode

### Cost Display

- `gI6(JD())` formats total cost in USD
- `"Total cost: {amount}"` shown dimly at session end
- Per-model breakdown, padded to 21 chars
- Warning: `"(costs may be inaccurate due to usage of unknown models)"`

### Semantic Color System

Named colors applied consistently (not raw ANSI codes):
- `"error"` — red
- `"success"` — green
- `"warning"` — yellow
- `"subtle"` — dim/muted
- `"text"` — default
- `"claude"` / `"claudeShimmer"` — brand colors for spinner
- `"suggestion"` — highlight for selected items
- `"planMode"` — special border color for plan mode
- `"inverseText"` — for colored backgrounds
- `"cyan_FOR_SUBAGENTS_ONLY"` — teammate display

### Layout Architecture

- **Ink primitives**: `Box` (flexbox), `Text` (styled text), height-constrained boxes
- Messages flow vertically (`flexDirection:"column"`)
- Tool calls indented with `marginTop:1`
- Borders: `borderStyle:"round"` with semantic `borderColor`
- **Two view modes**: expanded (full tool calls, streaming markdown, results) vs compact (only checkpoints, spinner for everything else)
- **Error boundaries** (`m56`): wraps tool result rendering, crashed render returns `null` not broken UI
- **Teammate tree**: Unicode box-drawing characters (`╘═`, `╞═`, `└─`, `├─`, `│`)

### Input Handling

- **Framework**: `@inquirer/core` prompt factory + Node.js `readline`
- **Multi-line**: Shift+Enter for newlines (needs terminal setup via `/terminal-setup`)
- **Paste detection**: Bracketed paste mode (`\x1b[200~` / `\x1b[201~`)
- **Direct bash**: `$` prefix enters bash mode
- **Slash commands**: `/` prefix routes through `processSlashCommand()`
- **Key bindings**: Enter (submit), Up/Down (navigate), Tab (autocomplete), Escape (cancel), number keys (quick-select)

---

## 2. Context & Compaction

Claude Code has a multi-layer compaction system — significantly more sophisticated than
BlipShell's `_compact_messages()`.

### Auto-Compact Trigger

Uses **actual API token counts** (not estimation):
- `autoCompactThreshold = contextWindow - maxOutputTokens - 13000`
- Constants: `_g9 = 20000` (max output reserve), `UN8 = 13000` (autocompact buffer), `dN8 = 3000` (blocking margin)
- Checked after every LLM turn via `$g9()`
- Overridable via `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` env var
- Disableable via `DISABLE_COMPACT` or `DISABLE_AUTO_COMPACT`

### Warning & Blocking Thresholds

```
isAboveWarningThreshold:      tokens >= effectiveWindow - 20000
isAboveErrorThreshold:        tokens >= effectiveWindow - 20000
isAboveAutoCompactThreshold:  tokens >= autoCompactThreshold (triggers compact)
isAtBlockingLimit:            tokens >= modelContextWindow - 3000 (BLOCKS next query)
```

UI shows "Context low (X% remaining)" in status bar.

### Compaction Prompt (9-Section Summary)

The compaction LLM call (`Hf9`) produces a `<summary>` block inside `<analysis>` tags:

1. **Primary Request and Intent** — all user requests/intents in detail
2. **Key Technical Concepts** — technologies, frameworks discussed
3. **Files and Code Sections** — files examined/modified/created, with code snippets and rationale
4. **Errors and Fixes** — errors encountered and how fixed, user feedback
5. **Problem Solving** — solved problems and ongoing troubleshooting
6. **All User Messages** — ALL non-tool-result user messages (critical, verbatim)
7. **Pending Tasks** — outstanding explicit tasks
8. **Current Work** — precise description of work immediately before compaction, with file names and code
9. **Optional Next Step** — next step with "direct quotes from the most recent conversation" to prevent drift

Three prompt variants:
- `wf9` — full analysis (chronological, thorough)
- `Of9` — partial/incremental (recent messages only, older summary kept)
- `$f9` — lightweight ("coverage, not detail" — planning scratchpad)

Includes: `"IMPORTANT: Do NOT use any tools. You MUST respond with ONLY the <summary>...</summary> block."`

### Partial (Incremental) Compaction

`rB9()` finds a split point in the conversation:
- Accumulates token counts from recent messages
- Stops when hitting `minTokens: 10000` AND `minTextBlockMessages: 5` OR `maxTokens: 40000`
- Only the older portion gets summarized; recent messages kept verbatim

### Post-Compaction File Restoration

`zY4()` restores recently-read files as attachments after compaction:
- `e94 = 5` — max files restored
- `hB9 = 50000` — max total tokens for restored files
- `SB9 = 5000` — max tokens per restored file
- Files sorted by recency (most recent first)

### Continuation Prompt

After compaction: `"This session is being continued from a previous conversation that ran out of context..."` followed by summary, transcript path, and `"Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with 'I'll continue' or similar. Pick up the last task as if the break never happened."`

### PreCompact Hook

`PreCompact` hook can:
- Exit 0: append custom compact instructions (stdout)
- Exit 2: block compaction entirely
- Triggered with `manual` or `auto` tag

---

## 3. Tool Result Handling

### Bash Output

- **Default**: 30,000 chars (`Hn1 = 30000`)
- **Max**: 150,000 chars (`$n1 = 150000`), configurable via `BASH_MAX_OUTPUT_LENGTH`
- **Truncation**: First N chars only + `"... [N lines truncated] ..."` (NOT first/last split)
- **Full output saved to temp file**: path given to model for re-reading
- **Streaming buffer**: `On1` class caps at 32MB, appends `"... [output truncated - ${KB}KB removed]"`

### Background Task Output

- **Default**: 32,000 chars (`rm8 = 32000`)
- **Max**: 160,000 chars (`nm8 = 160000`), configurable via `TASK_MAX_OUTPUT_LENGTH`
- **Truncation preserves tail** (last N chars), prepends: `"[Truncated. Full output: ${path}]"`

### Read File

- **Default**: 2,000 lines (`Xh6 = 2000`) — **10x BlipShell's 200-line default**
- **Line truncation**: 2,000 chars per line (`uQK = 2000`)
- **Max result**: 100,000 chars (`maxResultSizeChars: 1e5`)
- **Pagination**: `offset` and `limit` parameters
- **Overflow message** (hidden from user): `"Note: The file ${filename} was too large and has been truncated to the first ${Xh6} lines. Don't tell the user about this truncation. Use ${readTool} to read more."`
- **Image/PDF support**: Images returned as base64, PDFs with page-range support

### Git Status

Truncated at 40K chars: `"(truncated because it exceeds 40k characters. If you need more information, run 'git status' using BashTool)"`

---

## 4. State Injection & Attachments

### No Explicit `[STATE]` Block

Claude Code does **NOT** inject a centralized state block like BlipShell's `_build_state_block()`.
Instead, it computes up to **31 attachment types** per turn, injected as hidden `isMeta: true`
messages before each LLM call.

### Attachment Types

**Always computed**:
- `date_change` — "The date has changed to X. DO NOT mention this to the user."
- `ultrathink_effort` — reasoning effort level
- `deferred_tools_delta` — list of deferred tools available
- `mcp_instructions_delta` — MCP server tool descriptions
- `changed_files` — **key state injection**: re-reads previously read files, computes diffs, injects as `edited_text_file` attachments
- `nested_memory` — MEMORY.md content
- `dynamic_skill` / `skill_listing` — available skills
- `plan_mode` / `plan_mode_exit` / `auto_mode` / `auto_mode_exit`
- `todo_reminders` — pending TODO items
- `teammate_mailbox` / `team_context` — multi-agent state
- `critical_system_reminder` — user-configured reminders

**IDE-only**:
- `ide_selection`, `ide_opened_file`, `output_style`, `diagnostics`, `lsp_diagnostics`
- `unified_tasks`, `async_hook_responses`
- `token_usage` — behind feature flag `CLAUDE_CODE_ENABLE_TOKEN_USAGE_ATTACHMENT`
- `budget_usd`, `verify_plan_reminder`, `queued_commands`

### File Change Detection (`readFileState`)

LRU cache (`Yd7` class using `lru-cache`):
- **Max entries**: 100 (`gK6 = 100`)
- **Max size**: 25MB (`xO9 = 26,214,400`)
- **Key**: normalized file path
- **Value**: `{content, timestamp, offset, limit, isPartialView}`
- On each turn, `ok9()` polls all entries — if file mtime > cached timestamp, generates diff attachment
- Diff message: `"Note: ${filename} was modified, either by the user or by a linter. This change was intentional, so make sure to take it into account."`

### Plan Mode Reminders

Injected every `TURNS_BETWEEN_ATTACHMENTS = 5` turns, full reminder every 5th attachment.
Includes plan file path and whether a plan exists.

---

## 5. Budget & Wind-Down

**Claude Code has NO tool-call budget or wind-down mechanism.**

### What It Has Instead

1. **Context window management** (token-based):
   - Warning at `effectiveWindow - 20000` tokens (UI indicator only)
   - Auto-compact at threshold (transparent to model)
   - Hard block at `modelContextWindow - 3000` — refuses next query
   - No "wrap up" or "use task_complete" injection

2. **USD budget** (cost-based, for API/CI mode):
   - `maxBudgetUsd` parameter
   - Injects: `"USD budget: $${used}/$${total}; $${remaining} remaining"`

3. **maxTurns** (sub-agents only, not main loop):
   - `maxTurns` config on agent definitions
   - SDK `BetaToolRunner` checks `max_iterations`
   - Main interactive loop has **no turn limit** — runs until context fills or user stops

4. **Max output token recovery**:
   - When model hits `max_output_tokens`, retries up to 3 times
   - Injection: `"Output token limit hit. Resume directly — no apology, no recap. Break remaining work into smaller pieces."`

**Philosophy**: Keep going until context is full, then compact and continue. No gradual wind-down.

**BlipShell's approach is better for local models** — weaker models can't self-regulate as
effectively, so explicit budget + wind-down (80% warning, 95% tool stripping) prevents wasted iterations.

---

## 6. Hooks System

### 21 Hook Events

| Event | Matcher Field | Can Block (exit 2)? |
|-------|--------------|-------------------|
| `PreToolUse` | `tool_name` | Yes — blocks tool call |
| `PostToolUse` | `tool_name` | Yes — injects feedback to model |
| `PostToolUseFailure` | `tool_name` | No |
| `Notification` | `notification_type` | No |
| `UserPromptSubmit` | — | Yes — blocks prompt |
| `SessionStart` | `source` (startup/resume/clear/compact) | No |
| `SessionEnd` | `reason` | No |
| `Stop` | — | Yes — continues conversation |
| `SubagentStart` | `agent_type` | No |
| `SubagentStop` | `agent_type` | Yes — continues agent |
| `PreCompact` | `trigger` (manual/auto) | Yes — blocks compaction |
| `PermissionRequest` | `tool_name` | Yes — auto-approve/deny |
| `Setup` | `trigger` (init/maintenance) | No |
| `TeammateIdle` | — | Yes — prevents idle |
| `TaskCompleted` | — | Yes — prevents completion |
| `Elicitation` | `mcp_server_name` | Yes |
| `ElicitationResult` | `mcp_server_name` | Yes |
| `ConfigChange` | `source` | Yes — blocks change |
| `InstructionsLoaded` | `load_reason` | No |
| `WorktreeCreate` | — | Via stdout path |
| `WorktreeRemove` | — | No |

### Hook Configuration

- **Sources**: `~/.claude/settings.json`, `.claude/settings.json`, `.claude/settings.local.json`, managed/policy, plugins
- **Matcher system**: regex patterns on the `matcherMetadata.fieldToMatch` field (e.g., `Write|Edit` matches those tools)
- **Three hook types**:
  - `command` — shell command, receives JSON on stdin
  - `prompt` — LLM evaluates a condition (PreToolUse/PostToolUse/PermissionRequest only)
  - `agent` — runs an agent with tools (same events as prompt)

### Exit Code Semantics

- **Exit 0**: Success. Stdout behavior varies per event (some show to model, some silent).
- **Exit 2**: Block/intervene. Blocks tool call, shows stderr to model, blocks compaction, etc.
- **Other**: Show stderr to user only, continue.

### Hook Output Structure

`hookSpecificOutput` can contain: `permissionDecision`, `updatedInput`, `additionalContext`.

---

## 7. Memory & Session Notes

### MEMORY.md (Auto-Memory)

- Located at `~/.claude/MEMORY.md` (user-level) and `.claude/MEMORY.md` (project-level)
- Injected into system prompt each session
- **Line limit** (`RM` variable) — files exceeding it get truncated with warning to "move detailed content into separate topic files"
- Setting: `autoMemoryEnabled` (boolean)
- Types: "auto" (auto-generated) and "agent" (agent-specific)

### Session Notes (Newer System)

Structured notes file per session at `~/.claude/session-memory/`:

**Template** (10 sections, `VY4 = 12000` token total budget, `uP1 = 2000` per section):
1. `# Session Title`
2. `# Current State`
3. `# Task specification`
4. `# Files and Functions`
5. `# Workflow`
6. `# Errors & Corrections`
7. `# Codebase and System Documentation`
8. `# Learnings`
9. `# Key results`
10. `# Worklog`

**Update triggers**:
- `minimumMessageTokensToInit: 10000` — don't start until 10K tokens of conversation
- `minimumTokensBetweenUpdate: 5000` — at least 5K tokens between updates
- `toolCallsBetweenUpdates: 3` — or every 3 tool calls
- Updated via Edit tool calls (model makes parallel edits to multiple sections)

**Key property**: Session notes **survive compaction** — when context is compacted,
session notes replace the conversation summary as persistent state.

Custom template at `~/.claude/session-memory/config/template.md`.
Custom prompt at `~/.claude/session-memory/config/prompt.md`.

### Team Memory Sync

- Files in `.claude/memory/` synced to server via REST API
- ETag-based conflict resolution
- Bidirectional sync for multi-agent teams

---

## 8. MCP Integration

### Three Transport Types

- `stdio` — subprocess-based (default): `claude mcp add my-server -- npx my-mcp-server`
- `http` — HTTP/Streamable HTTP: `claude mcp add --transport http sentry https://mcp.sentry.dev/mcp`
- `sse` — Server-Sent Events: `claude mcp add --transport sse server url`

### Configuration

- Stored in settings files (global, project, local)
- `.mcp.json` files in project root (auto-discovered, requires user approval)
- CLI: `claude mcp add`, `claude mcp remove`, `claude mcp list`
- Env vars: `-e API_KEY=xxx`
- HTTP headers: `--header "Authorization: Bearer ..."`
- OAuth: `--client-id`, `--client-secret`, `--callback-port`

### Tool Naming & Deferral

- Tools prefixed as `mcp__{server_name}__{tool_name}` (double underscore)
- **Tool deferral**: MCP tools use `shouldDefer: true` — not loaded until explicitly requested via `ToolSearch`
- Keeps initial tool schema overhead low (important for context budget)
- `listChanged` notifications support dynamic tool updates

### Elicitation

MCP servers can request user input via `Elicitation`/`ElicitationResult` hook events.

---

## 9. Sub-Agents & Forking

### Two Invocation Modes

**Fork** (omit `subagent_type`):
- Inherits full conversation context from parent
- Shares prompt cache (cheap to spawn)
- Prompt is a directive, not a briefing — agent already knows everything
- Rules: "You ARE the fork. Do NOT spawn sub-agents; execute directly."
- Output format: `Scope:`, `Result:`, `Key files:`, `Files changed:`, `Issues:`
- Parent told: "Don't peek" (don't read fork's transcript) and "Don't race" (don't fabricate results)
- 200 max turns per fork

**Typed subagent** (specify `subagent_type`):
- Starts fresh with zero context
- Needs full briefing ("like a smart colleague who just walked into the room")
- Custom types defined by users

### Agent Definition Files

`.claude/agents/` directory, Markdown with YAML frontmatter:
- Fields: `agentType`, `whenToUse`, `tools`, `maxTurns`, `model`, `permissionMode`, `source`, `memory`
- Sources: `built-in`, `userSettings`, `projectSettings`, `policySettings`, `plugin`

### Built-in Agent Types

- `fork` — implicit, inherits context, `tools: ["*"]`, `maxTurns: 200`, `permissionMode: "bubble"`
- `Explore` — read-only exploration
- `code-reviewer` — independent code review
- Verification agent — auto-spawns after 3+ tasks to verify work
- `magic-docs` — updates documentation files

### Worktree Isolation

- `isolation: "worktree"` runs agent in temporary git worktree (`.claude/worktrees/`)
- New branch based on HEAD
- Auto-cleanup if no changes; keeps worktree if changes made
- `ExitWorktree` tool: `"keep"` or `"remove"` action
- `discard_changes` flag for forced removal

---

## 10. Git Integration & Safety

### Git Status at Session Start

- Runs `git status --porcelain` and `git log`
- Injected as snapshot: "This is the git status at the start of the conversation. Note that this status is a snapshot in time, and will not update during the conversation."
- Truncated at 40K chars
- Main branch auto-detected (used for PR base)

### Git Safety Protocol

- Never update git config
- Never skip hooks (`--no-verify`) unless user explicitly requests
- Always create NEW commits (never amend unless asked)
- Never use `-i` flag (interactive)
- Force push, hard reset, branch deletion: BLOCK-listed in safety classifier
- Pushing to default branch: BLOCKED (must use feature branch)
- Co-authored-by attribution automatically appended
- PR attribution includes coding percentage, prompt count, memory recall count

### File Checkpointing

`fileCheckpointingEnabled` setting — saves file state before edits for rewind capability.

---

## 11. Permissions & Safety Classifier

### Five Permission Modes

- `default` — standard approval prompts
- `plan` — read-only tools only + `ExitPlanMode`
- `acceptEdits` — auto-approve file edits
- `dontAsk` — approve all non-destructive operations
- `auto` — safety classifier auto-approves/denies

### Auto Mode Safety Classifier

Detailed BLOCK/ALLOW rules:
- **BLOCK**: force push, credential leakage, data exfiltration, self-modification, production deploys, creating unsafe agents
- **ALLOW**: test artifacts, local operations, read-only operations, toolchain bootstrap

### Settings Architecture

Three levels: global (`~/.claude/settings.json`), project (`.claude/settings.json`), local (`.claude/settings.local.json`).
Managed settings can be policy-enforced. Plugin hooks are read-only.

---

## 12. Telemetry

### OpenTelemetry-Based

- Enabled via `CLAUDE_CODE_ENABLE_TELEMETRY` env var
- Prometheus exporter format
- Metric types: SUM, GAUGE, HISTOGRAM
- Attributes: `user.id`, `session.id`, `app.version`, `organization.id`
- `OTEL_LOG_USER_PROMPTS` controls prompt redaction

### Internal Event Tracking

`d("tengu_*")` calls throughout codebase:
- `tengu_ripgrep_eagain_retry`, `tengu_bash_tool_reset_to_original_dir`
- `tengu_hook_created`, `tengu_worktree_created/kept/removed`
- `tengu_compact`, `tengu_compact_failed`
- `tengu_session_memory_loaded`, `tengu_memdir_accessed`
- Cost: `Total cost`, `Total duration (API)`, `Total duration (wall)`, `Total code changes`

### Privacy Controls

- `grove_enabled` — opt-in data sharing
- Domain exclusion support
- Policy-level controls for managed environments

### Session Metrics

- `total_input_tokens`, `total_output_tokens`
- `cache_creation_input_tokens`, `cache_read_input_tokens`
- `context_window_size`, `used_percentage`, `remaining_percentage`
- Frame duration, FPS (p50/p95/p99)

---

## 13. Better of the Two — Per System

### Claude Code Wins (Adopt for BlipShell)

| System | What CC Does Better | Effort | Priority |
|--------|-------------------|--------|----------|
| **Markdown rendering in terminal** | Full marked + highlight.js. BlipShell outputs raw text. Rich has `Markdown` + `Syntax` — straightforward to add. | Medium | **High** |
| **Tool display (status dots)** | Animated dot -> checkmark/X, bold name, 10-line truncation + `/verbose`. Much cleaner than BlipShell's current output. | Low | **High** |
| **Compaction quality** | 9-section structured LLM summary vs BlipShell's crude tool-result compression. Directly impacts long-session quality. | Medium | **High** |
| **Post-compaction file restoration** | Top 5 recently read files re-attached (5K tokens each). BlipShell loses all file context on compaction. | Low | **High** |
| **Session notes** | 10-section structured notes surviving compaction. BlipShell has nothing equivalent for within-session continuity. | Medium | **Medium** |
| **Semantic color theme** | Named colors (error/success/warning/subtle) applied consistently. BlipShell styling is ad-hoc. | Low | **Medium** |
| **Cost tracking** | USD per session. BlipShell has token counts + $/M rates — just needs multiplication. | Low | **Medium** |
| **MCP tool deferral** | Lazy-load MCP tools to save context. BlipShell loads all upfront. | Low | **Medium** |
| **Bump read_file default** | CC default is 2000 lines. BlipShell's 200 is too aggressive — files get paginated unnecessarily. | Low | **Medium** |
| **File change detection** | LRU cache detects external edits via mtime, generates diff attachments. | Medium | **Low** |
| **Partial compaction** | Summarize old messages, keep recent verbatim. More nuanced than compress-everything. | Medium | **Low** |
| **Bash output to temp file** | Full output saved, path given to model. Enables re-reading truncated output. | Low | **Low** |

### BlipShell Wins (Keep As-Is)

| System | Why BlipShell Is Better |
|--------|------------------------|
| **Memory system** | 9-step search pipeline, entity graph, lessons, consolidation, temporal decay, contradiction detection. CC uses flat files with zero semantic search. BlipShell's core differentiator. |
| **State injection `[STATE]` block** | CC trusts model to self-track (reasonable for Opus/Sonnet). Weaker local models need explicit state showing tool count, budget, files read/created/edited, recent actions. |
| **Budget wind-down** | CC has none — relies on autocompact for infinite context. BlipShell's 80%/95% thresholds prevent wasted iterations with local models. |
| **Multi-endpoint routing** | CC uses single provider. BlipShell's fallback cascade (Ollama -> Groq -> Gemini -> local fallback) is essential for reliability. |
| **Edit fuzzy matching + linting** | CC requires exact string match. BlipShell's 3-layer matching (exact -> whitespace-normalized -> fuzzy 0.6) + Python AST linting catches more issues with weaker models. |
| **Actionable error messages** | CC returns basic errors. BlipShell's "try read_file to see current contents" guidance actively helps weaker models recover. |
| **Mid-task steering (Space/p)** | CC has no equivalent — cancel or wait. BlipShell's pause/redirect between tool batches is unique. |
| **Tool call deduplication** | CC trusts model not to repeat. BlipShell's cross-batch dedup prevents wasted iterations (local models repeat more). |
| **Guardrails system** | CC has no trajectory monitor, doom loop detector, completion audit, or correction detector. |
| **Observability** | CC has internal telemetry but nothing user-facing. BlipShell's `/flow`, `/health`, `/context` are debugging superpowers. |
| **Session management** | CC conversations are ephemeral (auto-memory is the only persistence). BlipShell has full session lifecycle with summaries, lessons, digests. |
| **Nightly maintenance** | CC has no maintenance automation. BlipShell's 7-job nightly runner (backup, cleanup, tagging, consolidation) keeps the system healthy. |

### Neither Wins / Skip for Now

| System | Verdict |
|--------|---------|
| **Sub-agents/forking** | CC's fork model requires prompt cache sharing (Anthropic infrastructure). Can't replicate cheaply with local LLMs. Skip until practical local approach exists. |
| **Hooks system** | Powerful (21 events, 3 types) but complex. Not worth implementation cost now — guardrails covers the most important cases. Revisit when user extensibility becomes a priority. |
| **Git worktrees** | Niche use case. Project mode works fine. |
| **React/Ink rewrite** | Massive effort. Rich covers 90% of the same output. |
| **Multimodal (images/PDFs/notebooks)** | Requires multimodal model support from Ollama. Not actionable yet. |
| **IDE integration** | Big build. Not prioritized. |
| **Team memory sync** | Server-side multi-agent coordination. Out of scope. |

---

## 14. Implementation Priority List

### Tier 1 — Highest Impact, Do First

1. **Markdown rendering in terminal**
   - Use Rich's `Markdown` class + `Syntax` for code blocks
   - Apply to all LLM text output in both CLI and streaming
   - Reference: CC uses marked + highlight.js

2. **Cleaner tool display**
   - Status dots: spinning -> checkmark/X
   - Bold tool name + parameter summary
   - Default truncation to 10 lines + `/verbose` for full
   - Already have `/verbose` toggle — make truncation the default

3. **Structured compaction (9-section LLM summary)**
   - Replace `_compact_messages()` crude compression with LLM-driven structured summary
   - Use REASONING model for the compaction call
   - Keep BlipShell's budget wind-down (CC doesn't have this, but we need it)
   - Save transcript to file for post-compaction reference

4. **Post-compaction file restoration**
   - After compaction, re-inject top 5 recently read files (5K tokens each, 25K total cap)
   - Use executor's `files_read` tracking to identify recent files

### Tier 2 — High Value, Medium Effort

5. **Session notes** (10-section template surviving compaction)
   - Complement existing memory system, not replace it
   - Fixed sections: Task, Current State, Files, Decisions, Errors, Learnings
   - Update every 3 tool calls or 5K tokens
   - Token budget: 12K total, 2K per section

6. **Semantic color theme**
   - Create `theme.py` with named colors: error, success, warning, subtle, tool, etc.
   - Apply consistently across all CLI output
   - Replace ad-hoc Rich styling

7. **Cost tracking**
   - Multiply token counts by configured `cost_per_1m_prompt` / `cost_per_1m_completion` per endpoint
   - Show at session end: `"Total cost: $X.XX"`
   - Add to `/tokens` command output

8. **MCP tool deferral**
   - Don't include MCP tool schemas in every LLM call
   - Register as available but don't pass to model until requested
   - Saves significant context tokens with many MCP tools

9. **Bump read_file default to 500 lines**
   - CC uses 2000, BlipShell's 200 is too aggressive
   - 500 is a reasonable middle ground for local models with smaller context

### Tier 3 — Nice to Have

10. **File change detection via mtime**
    - Extend executor's file tracking with timestamps
    - On each turn, check if tracked files were modified externally
    - Inject diff attachment if changed

11. **Partial compaction**
    - Find split point in conversation
    - Summarize only older portion, keep recent messages verbatim
    - More nuanced than current all-or-nothing approach

12. **Bash output to temp file**
    - When shell output exceeds limit, save full output to temp file
    - Give model the path so it can re-read specific sections

---

## Appendix: Key Constants from Claude Code Source

```
Context/Compaction:
  _g9 = 20000    — max output tokens reserve
  UN8 = 13000    — autocompact buffer
  dN8 = 3000     — blocking limit margin
  wg9 = 20000    — warning threshold
  pSA = 100000   — SDK compaction token threshold

Tool Output:
  Hn1 = 30000    — default bash output chars
  $n1 = 150000   — max bash output chars
  rm8 = 32000    — default task output chars
  nm8 = 160000   — max task output chars
  Xh6 = 2000     — default read file lines
  uQK = 2000     — max chars per line in read

File Tracking:
  gK6 = 100      — LRU cache max entries
  xO9 = 26214400 — LRU cache max size (25MB)
  e94 = 5        — post-compaction files restored
  hB9 = 50000    — max tokens for restored files
  SB9 = 5000     — max tokens per restored file

Session Notes:
  VY4 = 12000    — total session notes budget (tokens)
  uP1 = 2000     — per-section budget (tokens)
  RM  = 200      — MEMORY.md line limit

UI:
  600ms          — status dot blink interval
  120ms          — spinner animation interval
  400ms          — brief mode animation interval
  10 lines       — tool result truncation default
```
