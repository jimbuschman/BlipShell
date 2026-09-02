# Agent evaluation — picking BlipShell's main (tool-calling) model

**Run date:** 2026-08-19 · **Models:** 24 · **Episodes:** 720 · **Harness:** `scripts/agent_eval/`
**Raw data:** `agent_eval_results/2026-08-19/` (full transcripts + `SUMMARY.json`)

## Why this exists separately from `blipshell benchmark`

The benchmark harness measures nine job types against ground-truth scorers. It could not
answer "which model should be my main," for a reason worth remembering: its `tool_calling`
suite sent a **single turn with no system prompt**, which is a condition production never
runs. Fixed (see CLAUDE.md), but even fixed it scores one turn against an expected call.
That cannot evaluate an agent, because the correct first move is often to orient —
`glob_files`, then read, then edit.

So this is a separate thing: real multi-turn episodes against a simulated repo, using
**BlipShell's real tool schemas** (11, exported from `blipshell/core/tools/`) and the
**real `AgentConfig.system_prompt`**, graded by reading transcripts.

## Method

Ten scenarios × 3 episodes × up to 20 turns. Tool calls execute against a consistent fake
workspace (`workspace.py`) and results feed back each turn, as `ChatLoop` does.

| Scenario | Probes |
|---|---|
| `locate_and_explain` | Find a file whose path was never given, then explain it |
| `precise_edit` | Read-before-edit; a guessed `old_text` genuinely fails |
| `restraint_no_tool` | A general question needs NO tool |
| `memory_recall` | Routes "what did we decide" to memory, not files/web |
| `nonexistent_file` | Honesty under a false premise — or invent an AWS region? |
| `diagnose_and_fix` | Run tests → read failure → implement → re-verify |
| `ambiguous_request` | "it's broken, fix it" — ask, or guess and edit? |
| `destructive_caution` | "clean up the junk" — does it reach for `rm -rf`? |
| `targeted_read` | Extract one section, not the whole file |
| `multi_fact` | Two facts from one file, both must be right |

`regrade.py` holds first-pass detectors. **They are not the verdict** — every 0/3 cell was
read by hand, and doing so corrected the grader four times (see Lessons).

## Results

Full table in `SUMMARY.json`. Headlines:

| Tier | Best | Score |
|---|---|---|
| Paid cloud | **glm-5.2** | 30/30 |
| Free cloud | **gemma4:cloud** | 29/30 |
| Free cloud (incumbent) | minimax-m3 | 26/30 |
| Local / Pi-sized | qwen3:4b, qwen3.5:4b | 21/30 |

### Main-model recommendation

**Keep `minimax-m3:cloud`** among free-tier options — not because it topped the table, but
because of *how* the others fail:

- **`gemma4` (both sizes) systematically misreads the domain.** Asked whether memory dedup
  is enabled — three lines of `config.yaml` — it shells out to Linux KSM and VMware
  commands (`/proc/sys/vm/ksm_run`, `esxcli`). `gemma4:cloud` recovers 2/3; `gemma4:31b`
  never does. Wrong instinct for a project assistant, and consistent rather than random.
- **`nemotron-3-ultra` doom-loops.** All 3 `destructive_caution` episodes burned 20 turns
  cycling `list_directory` and ended saying nothing. Also 40.1 s/episode.
- **`gpt-oss:120b` (24/30) goes silent.** Its 4 turn-limit episodes and 4 silent episodes
  are *the same four*: when it cannot resolve something it searches to the budget then
  returns nothing. m3 answers the same case in 2–3 turns. It also does 2.2× the tool calls
  (222 vs 102).
- **`minimax-m3`'s 4 failures are scattered singletons**, zero silent episodes, fastest
  strong model at 7.3 s.

If the paid plan continues, **`glm-5.2` is the clear answer** — the only model to pass
everything.

## Raspberry Pi feasibility

**No Pi-sized model can run the full agent loop.** Best local is 21/30 versus 18/30 for the
*worst* cloud model. `diagnose_and_fix` — the core executor loop — is 1/3 at best, 0/3 for
most, against 3/3 for nearly every cloud model.

Hard blockers:

- **The entire gemma3 line cannot tool-call**: `gemma3:1b`, `gemma3:4b`, `gemma3n:e4b` all
  return `does not support tools (400)`. Architectural, not tunable.
- **`xLAM-2-3b-fc-r` is unusable here** despite leading BFCL for small models (65.74%
  overall, 55.62% multi-turn). Both the Salesforce and Mungert GGUFs fail Ollama's tool
  parser: `The model produced output that does not match the expected peg-native format`.
  It chats fine; only tool output breaks. Would need a custom Modelfile template or vLLM.
- **`llama3.2:3b` contradicts published guidance.** Recommended for "broadest tool-calling
  compatibility"; scored **9/30**. On a plain Python question it called `ask_user` to echo
  the question back *and* emitted `{"name": "explain", "parameters": {...}}` as prose — it
  leaks tool-call structure into the content field.
- **`qwen3:4b` is disqualified by latency**, not score. Joint-best at 21/30 but **67.6
  s/episode on a GPU**; Pi CPU is 10–20× worse. Thinking models pay this tax twice, since
  they also need more turns.

### What does work: the split

`lfm2.5` is the cleanest signal — **3/3** on `locate_and_explain`, memory recall, and
honesty about missing files; **0/3** on `precise_edit` and `diagnose_and_fix`; 5.4 s,
zero silent episodes. Retrieval and explanation yes, file mutation and long chains no.

Every small model shows that shape. All are 3/3 on restraint and ambiguity handling — the
conversational judgment is fine. They fail on *precise-edit mechanics* (`qwen2.5:3b`
correctly diagnosed the bug, then tried to edit `utils.py` using the import line from the
*test* file) and on chains longer than ~4 turns.

This matches published data independently: LFM2.5-230M does 42 tok/s on a Pi 5 under 1GB
and is documented as having a "low knowledge ceiling, unsuitable for code generation,
multi-step reasoning, or general chat" — optimized for structured extraction and routing.
Our own numbers for `lfm2.5` (tool_calling 0.933, reasoning 0.450, entity 0.356 in the
older benchmark corpus) say the same thing from a different direction.

**Proposed Pi shape — a memory appliance, not a coding agent.** The Pi runs capture,
embedding, sqlite-vec retrieval, recall and chat on `lfm2.5` or `qwen2.5:3b`, with **no
file-mutation tools and no write-side memory pipeline**. Summarization, entity extraction
and ranking defer to HPBENDERTWO or cloud. The router already does per-task-type routing
with fallback, so this is configuration, not new code.

Measured Pi 5 throughput for sizing (Q4_K_M, active cooler + NVMe): Llama 3.2 1B 17.2 t/s,
Llama 3.2 3B 8.8 t/s, Phi-3.5 Mini 3.8B 7.4 t/s; ~5–15 t/s at 1.5B, 2–5 t/s at 3B.

## Lessons about the harness itself

Four measurement bugs were found by checking results rather than trusting them. **Every one
of them penalized the stronger models**, so each would have inverted part of the ranking:

1. **Missing `grep_files`/`glob_files`.** Production registers both
   (`agent_tools.py:78-79`); the first harness omitted them, so models that correctly
   reached for grep got "unknown tool." Fixing it moved minimax-m3 from 1/3 to 3/3 on two
   scenarios.
2. **Grep was substring-only.** Real `GrepTool` is ripgrep-backed. 48 of 218 grep calls
   wrongly returned "(no matches)" — worst for the models sophisticated enough to write
   `dedup|similarity|threshold`.
3. **The directory listing advertised unreadable files.** `DIRS` listed
   `tests/test_worker.py`, `blipshell/core/agent.py` and others absent from `FILES`,
   producing ~90 spurious errors that punished thorough exploration. `_fill_stubs()` now
   makes the workspace self-consistent by construction.
4. **Grader phrase lists were too narrow, four times.** "There's no `deploy_config.yaml`
   file in the project" and "The repository doesn't contain a `deploy_config.yaml`" are
   both correct answers that scored as failures; `gemma4`'s ideal clarifying reply was
   marked wrong for lacking a question mark. **A 0/3 cell was wrong more often than the
   model was.**

Rule this establishes: **read the transcript before believing the score**, and treat any
0/3 as a claim about the harness until proven otherwise.

## Caveats

- 3 episodes per scenario is coarse. Anything between 25 and 29 is within a couple of
  episodes. Trustworthy gaps: `gpt-oss:20b` being bad, `glm-5.2` being best, the `gemma4`
  domain blind spot, and the local-vs-cloud cliff.
- Ten scenarios weighted toward file/code work, because that is what the tool set is. This
  does **not** test long-conversation memory behaviour, which is BlipShell's actual moat.
- Local models ran on HPBENDERTWO's GPU. That measures **capability, not Pi speed** —
  capability is hardware-independent, latency is not.
- The paid-tier models (`glm-5.2`, `deepseek-v4-flash`, `qwen3.5:397b`,
  `mistral-large-3:675b`, `minimax-m2.7`, `kimi-k2.7-code`) require a subscription. Free
  tier is the 8 `cloud_free` rows.

## Reproducing

```bash
cd scripts/agent_eval
python run.py <model> 3                      # localhost:11434
DEEPTEST_URL=http://[tailscale-ip]:11434 python run.py <model> 3   # Ollama PC
python regrade.py                            # re-grade saved transcripts, no model calls
```

Transcripts are saved in full, so re-grading never needs another model call — which is how
all four grader bugs got fixed without re-running anything.
