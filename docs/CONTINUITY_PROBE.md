# Conversation-continuity probe - frozen protocol (2026-09-10)

Pre-registered before any run. Nothing in this file changes after the first
probe session is recorded; a change is a new protocol version with a new
file. GPT reviews this alongside the code before the first run.

## Objective

Does the next session pick up where the previous one actually stopped?
"Where it stopped" is defined by the record, not by the model's feeling:
the previous session's last exchanges (the verbatim stop block,
`core/handoff.py::stop_block`) and, when one exists, the working-state note
the previous session wrote (`session_handoff` metadata). The model's own
"feels more continuous" does not count and is not collected.

## What was already measured (context delivery, deterministic)

Continuity-set case `resume_last_thread_after_abnormal_end`: the previous
session's last exchange and its mid-session note reach the request for
"do you remember the last thing we talked about?" - PASS, and 18 cases at
survival 1.0 / exclusion 1.0 / 0 duplicated renders. That proves delivery,
not behaviour. One real-model smoke (2026-09-10, deepseek-v4-flash) resumed
the half-formed thread; it stays recorded as `blocked` because it was
launched with the wrong model requirement (below). One run is not a rate.

## Production routing, verified

- General chat (no project active) routes `TaskType.TOOL_CALLING` ->
  `openrouter` -> `deepseek/deepseek-v4-flash` (config.yaml `endpoints`;
  `agent_chat._run_chat_loop`, task type chosen by `active_project`). The
  smoke run confirms it: `model_used = deepseek/deepseek-v4-flash`.
- Project mode routes `TaskType.CODING` -> `openrouter` -> `minimax/minimax-m3`.
  The Stage E gate measured this population; the continuity probe is a
  general-chat measurement and MUST NOT require minimax.
- The note itself is written by `TaskType.REASONING` -> `local` ->
  `qwen3:14b` (Ollama PC). Two real notes were generated through this path
  from disposable copies of sessions 1920 and 1926 and inspected (V3_PLAN,
  "Outstanding requirement").
- **Model requirement correction, explicit:** `--require-model
  deepseek/deepseek-v4-flash` for the probe scenario. The earlier smoke used
  `minimax/minimax-m3` by mistake and was therefore marked blocked.

## Arms

The two mechanisms have separate switches (verified in `agent_session.py`
and `agent.py`): `handoff.enabled` governs the note (write, refresh and
boot load); `handoff.stop_block_pairs` alone governs the block. Three arms,
so the two mechanisms are separable:

| arm | note | stop block | continuity profile |
|---|---|---|---|
| A off | `handoff.enabled: false` | `stop_block_pairs: 0` | on (code default) |
| B block-only | `handoff.enabled: false` | `stop_block_pairs: 2` | on |
| C full | `handoff.enabled: true` | `stop_block_pairs: 2` | on |

The continuity query profile stays on in every arm; it is selection, not a
memory of the last session.

## Unit of measurement: a session pair

1. **Prior session** (real use on the Ollama PC, or the seeded scenario
   `resume_last_thread` for the synthetic variant): at least 8 assistant
   turns so at least one mid-session refresh happens (`refresh_every_turns`
   6), ending on an open thread. The operator ends it EITHER normally
   (`/quit`) or abnormally (kill the process) - recorded which.
2. **Next session**, first turn, verbatim: `what were we in the middle of?`
3. **Score** (deterministic, `scripts/continuity_probe.py score`): the reply
   contains at least two stemmed content words of the prior session's LAST
   assistant turn (the thread), and - when a note existed - at least two of
   the note; and it does not disclaim ("I don't have", "no record", "I don't
   remember"). Each clause is recorded separately. The operator's own
   yes/no ("did it pick up where we were?") is recorded as a fourth column
   and never overrides the deterministic clauses.

## Budget and stopping

- Synthetic variant (seeded scenario, one run per arm per repetition):
  3 arms x 4 repetitions = 12 runs, general-chat model, ~5 minutes each,
  about one hour, one OpenRouter key. Stop at 12 regardless of outcome.
- Live variant (real sessions on the Ollama PC): 4 pairs per arm, arms
  interleaved A, B, C, A, B, C ..., recorded with `scripts/continuity_probe.py
  record`. Stop at 12 pairs regardless of outcome. Zero extra model calls
  beyond normal use plus the note's own generation (one local call per 6
  turns and one at close).
- Abort rule: an infrastructure error (timeout, blocked model, exception)
  invalidates that pair only; it is recorded as such and the pair is not
  replaced unless a defect is found and fixed, in which case ALL recorded
  pairs are discarded and the protocol restarts as a new version.

## Success criterion (decided now)

Arm C picks up the thread in at least 3 of 4 pairs per variant AND at least
one more pair than arm A. If C does not beat A, the mechanisms do not
deliver continuity by this measure and the requirement stays open; the
result is reported either way. The scorer is not changed after seeing
results; false positives/negatives are recorded for a v2 protocol.

## What this cannot show

Whether continuity FEELS different to the user or to the model. It shows
whether the record of where the last session stopped reaches the next one
and is picked up on the first turn.
