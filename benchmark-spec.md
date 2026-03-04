# BENCHMARK_SPEC.md — Unified LLM Benchmark for BlipShell

**Purpose**: ONE benchmark run producing ONE results table: model × role × score.
**Goal**: Systematically evaluate every LLM functional role to select optimal models per task.

---

## 1. Every Functional Role That Uses an LLM

Sixteen distinct roles, grouped by execution pattern. Dead code (`generate_memory_name`) and utility calls (web page extraction) excluded.

### Pipeline Roles (single-prompt, background, high-volume)

| # | Role | Prompt Function(s) | Task Type | Current Model | Call Site |
|---|------|-------------------|-----------|---------------|-----------|
| 1 | **Summarization** | `summarize_memory`, `summarize_session_chunk`, `summarize_session_conversation`, `summarize_session_summaries` | SUMMARIZATION | Groq gpt-oss-120b → local glm4 | processor.py:91, manager.py:324-356, agent_background.py:58 |
| 2 | **Scoring** | `rank_importance_and_classify` | RANKING_IMPORTANCE | Groq llama-3.3-70b → local qwen2.5:7b | processor.py:165 |
| 3 | **Deduplication** | `decide_memory_action` | REASONING | qwen3:14b | processor.py:353 |
| 4 | **Contradiction Detection** | `detect_contradiction` | REASONING | qwen3:14b | processor.py:299 |
| 5 | **Lesson Extraction** | `extract_lesson` | REASONING | qwen3:14b | processor.py:242 |
| 6 | **Entity Extraction** | `extract_entities` | REASONING | qwen3:14b | entity_extractor.py:166 |
| 7 | **Entity Resolution** | `resolve_entity_duplicate` | REASONING | qwen3:14b | entity_extractor.py:311 |

### Tagging Roles (single-prompt, nightly batch)

| # | Role | Prompt Function(s) | Task Type | Current Model | Call Site |
|---|------|-------------------|-----------|---------------|-----------|
| 8 | **Tag Discovery** | `discover_tag_patterns` | REASONING | qwen3:14b | tag_discovery.py:75 |
| 9 | **Batch Tag Assignment** | `batch_assign_tags` | RANKING | qwen2.5:14b | batch_tagger.py:130 |

### Synthesis Roles (single-prompt, session-end or nightly)

| # | Role | Prompt Function(s) | Task Type | Current Model | Call Site |
|---|------|-------------------|-----------|---------------|-----------|
| 10 | **Session Reflection** | `reflect_on_session`, `merge_chunk_reflections` | REASONING | qwen3:14b | processor.py:511-562 |
| 11 | **Session Titling** | `generate_session_title` | SUMMARIZATION | glm4 | manager.py:356 |
| 12 | **Project Digest** | `generate_initial_digest`, `update_digest_incremental`, `update_digest_with_sessions` | REASONING | qwen3:14b | project_digest.py:148-218 |
| 13 | **Self-Reflection** | `reflect_on_response` | REASONING | qwen3:14b | agent_background.py:76 |
| 14 | **Plan Generation** | `generate_plan` | CODING/TOOL_CALLING | glm-5:cloud | planner.py:53 |

### Interactive Roles (multi-turn, foreground)

| # | Role | Prompt Function(s) | Task Type | Current Model | Call Site |
|---|------|-------------------|-----------|---------------|-----------|
| 15 | **Tool Calling** | (system prompt + chat loop) | TOOL_CALLING | glm-5:cloud | agent_chat.py:119 |
| 16 | **Coding** | `executor_system_prompt`, `dynamic_execution_prompt` | CODING | glm-5:cloud | executor.py:472 |

---

## 2. What "Best Model" Means Per Role

Each role gets a **composite score normalized to 0.0–1.0**. Every sub-metric is binary (pass/fail) or already 0–1. The composite is `passes / total_checks` unless otherwise noted. Speed is reported separately and not folded into the composite.

### Speed Targets

| Role Category | Target Latency |
|---------------|---------------|
| Pipeline (background, high-volume) | <0.5–2s per call |
| Tagging (nightly batch) | <1–3s per call |
| Synthesis (session-end) | <3–15s per call |
| Interactive (foreground) | <3–5s first token |

### Role 1: Summarization

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Parse OK** | Non-empty response, no exception | 1 if ok, 0 if not |
| **SKIP correctness** | Returns "SKIP" for filler/greetings, non-SKIP for substantive content | 1 if matches expected |
| **Word count** | ≤30 words | 1 if ≤30 |
| **No echo** | Summary ≠ input (Jaccard similarity <0.7) | 1 if distinct |
| **Third person** | No "I ", "my ", "you " at word boundaries | 1 if 3rd person |
| **Compression** | Output chars / input chars < 0.5 (for inputs >100 chars) | 1 if compressed |

**Composite**: `sum(checks) / 6` per test case, averaged across cases.
**Test data**: 8 curated messages (existing) + 20 real DB messages (stratified by rank).

### Role 2: Scoring (Rank + Importance + Classification)

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Parse OK** | All 3 values extracted (rank, importance, type) | 1 if all 3 parsed |
| **Rank accuracy** | \|predicted − expected\| ≤ 1 | 1 if within ±1 |
| **Importance accuracy** | \|predicted − expected\| ≤ 0.2 | 1 if within ±0.2 |
| **Type accuracy** | Matches expected memory_type | 1 if exact match |

**Composite**: `sum(checks) / 4` per case, averaged.
**Test data**: 20 real DB memories with known rank/importance/type (stratified by rank 1–5).

### Role 3: Deduplication

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Parse OK** | Valid action extracted (ADD/UPDATE/DELETE/NONE) | 1 if valid |
| **Decision correct** | Action matches expected for curated pair | 1 if correct |

**Composite**: `sum(checks) / 2` per case, averaged.
**Test data**: 10 curated pairs — 3 clearly unrelated (expect ADD), 3 near-duplicates (expect UPDATE/DELETE), 2 identical (expect NONE), 2 updates (expect UPDATE). **NEW — no existing curated set.**

### Role 4: Contradiction Detection

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Parse OK** | Response starts with YES or NO (not INVALID) | 1 if valid |
| **Accuracy** | Matches expected YES/NO | 1 if correct |

**Composite**: `sum(checks) / 2` per case, averaged.
**Test data**: 6 curated pairs (existing in benchmark_models.py) + 10 real core memory pairs from DB.

### Role 5: Lesson Extraction

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **SKIP correctness** | SKIP for filler, non-SKIP for substantive conversation | 1 if matches |
| **Non-empty** | For non-SKIP cases: response >20 chars | 1 if substantive |
| **Actionable** | Contains a verb (heuristic: matches `\b(use|avoid|prefer|always|never|try|consider|remember)\b`) | 1 if actionable |

**Composite**: `sum(checks) / 3` per case, averaged.
**Test data**: 5 curated conversations (existing) + 10 real session chunks from DB.

### Role 6: Entity Extraction

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Parse OK** | ≥1 valid triple extracted for substantive content, NONE for filler | 1 if appropriate |
| **Triple validity** | All triples have 5 non-empty pipe-delimited fields | per-triple binary, averaged |
| **Type validity** | subject_type and object_type are in known type set | per-triple binary, averaged |
| **Coverage** | Expected entities found in extracted subjects/objects | expected_found / expected_total |

**Composite**: average of the 4 metrics.
**Test data**: 5 curated summaries (existing) + 20 real DB summaries with manually verified entities.

### Role 7: Entity Resolution

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Parse OK** | Response starts with YES or NO | 1 if valid |
| **Accuracy** | Matches expected | 1 if correct |

**Composite**: `sum(checks) / 2` per case, averaged.
**Test data**: **NEW — 12 curated pairs needed**: 4 same-entity pairs (e.g., "Jim Buschman" / "Jim B."), 4 different-entity pairs, 4 tricky pairs (e.g., "Python" language vs "Python" snake).

### Role 8: Tag Discovery

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Parse OK** | ≥1 `tag_name: pattern` line extracted, or NONE for tagless content | 1 if valid |
| **Regex validity** | Each pattern compiles via `re.compile()` | per-pattern binary, averaged |
| **Tag novelty** | Tags not in the "existing tags" input list | per-tag binary, averaged |

**Composite**: average of the 3 metrics.
**Test data**: **NEW — 8 curated test sets needed**: 5 sets of poorly-tagged summaries + existing tag list (expect new patterns), 3 sets of well-tagged content (expect NONE).

### Role 9: Batch Tag Assignment

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Precision** | Correct assigned tags / total assigned tags | 0.0–1.0 |
| **Recall** | Ground truth tags found / total ground truth | 0.0–1.0 |
| **F1** | Harmonic mean of precision and recall | 0.0–1.0 |

**Composite**: F1 score.
**Test data**: 30 real memories with ≥3 ground-truth tags (existing in benchmark_tagger.py).

### Role 10: Session Reflection

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Sections filled** | Count of 5 sections present / 5 | 0.0–1.0 |
| **Valid effectiveness** | effectiveness ∈ {effective, partially_effective, ineffective, unclear} | 1 if valid |
| **Specificity** | Proportion of bullets with concrete detail markers (`.`, `(`, `:`, `=`, `/`, `` ` ``) | 0.0–1.0 |

**Composite**: `(sections_filled/5 + validity + specificity) / 3`.
**Test data**: 5 real sessions from DB (existing in benchmark_reflection.py).

### Role 11: Session Titling

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Length OK** | ≤10 words | 1 if ok |
| **No filler** | Does not start with "A ", "The ", "Session about" | 1 if clean |
| **Relevance** | Contains ≥1 keyword from the session content (top TF-IDF term) | 1 if relevant |

**Composite**: `sum(checks) / 3` per case, averaged.
**Test data**: **NEW — 10 real sessions with manually verified ideal titles needed.**

### Role 12: Project Digest

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Sections present** | Contains all 5 expected sections (Overview, Status, Decisions, Activity, Issues) | count/5 |
| **Length OK** | 300–500 words | 1 if in range |
| **Non-generic** | Mentions project name and ≥2 specific details from input | 1 if specific |

**Composite**: `(sections/5 + length_ok + specificity) / 3`.
**Test data**: **NEW — 3 projects with session history from DB.** Falls back to synthetic if DB unavailable.

### Role 13: Self-Reflection

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **NO_CHANGES correctness** | Returns NO_CHANGES for already-good responses | 1 if matches |
| **Improvement detection** | Returns improved text for flawed responses | 1 if improved |
| **No regression** | Improved response doesn't introduce new errors (heuristic: keeps correct parts) | 1 if no regression |

**Composite**: `sum(checks) / 3` per case, averaged.
**Test data**: **NEW — 8 curated cases needed**: 4 good responses (expect NO_CHANGES), 4 flawed responses with known errors (expect correction).

### Role 14: Plan Generation

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Parse OK** | `_parse_steps()` extracts ≥1 step | 1 if parsed |
| **Step count** | 1–5 steps | 1 if in range |
| **Tool hints** | Proportion of steps with valid tool_hint | 0.0–1.0 |
| **Relevance** | Steps reference the actual task (contains task keyword) | 1 if relevant |

**Composite**: `(parse + count + hints + relevance) / 4`.
**Test data**: **NEW — 6 curated task descriptions needed** (from simple file read to multi-step refactor).

### Role 15: Tool Calling (Interactive Chat)

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Tool selection** | Correct tool called for given scenario | 1 if correct |
| **No errors** | No exceptions during execution | 1 if clean |
| **Completion** | Response is non-empty and addresses the query | 1 if complete |
| **Efficiency** | tool_calls ≤ expected max | 1 if within budget |

**Composite**: `sum(checks) / 4` per test, averaged across tests.
**Test data**: 14 tests from existing `--simple-chat` suite + 4 existing from benchmark_reasoning.py tool_calling tests.

### Role 16: Coding (Executor)

| Metric | Definition | How Scored |
|--------|-----------|------------|
| **Task completion** | `task_complete` tool called or substantial text response | 1 if complete |
| **Check accuracy** | `checks_passed / checks_total` per task (existing verification checks) | 0.0–1.0 |
| **Efficiency** | tool_calls ≤ 2× expected (not grinding) | 1 if efficient |
| **No errors** | No FATAL errors | 1 if clean |

**Composite**: `(completion + accuracy + efficiency + no_errors) / 4` per task, averaged.
**Test data**: 8 real-world tasks from benchmark_coding.py + 65 tasks from `--stress` suite. Each task has verify_checks baked in.

> **Note on the stress test**: The 65-task stress suite tests **executor scaffolding quality** (tool call correctness, error recovery, budget wind-down, same-args dedup), not raw model capability. It is absorbed into the Coding role as test cases, not treated as a separate benchmark.

---

## 3. Test Data Sources and Coverage

### Test Mechanism Per Role

| Role | Mechanism | Data Source | Cases | New Data? |
|------|-----------|-------------|-------|-----------|
| Summarization | Single-prompt suite | Curated (8) + Real DB (20) | 28 | No |
| Scoring | Single-prompt suite | Real DB (20, stratified) | 20 | No |
| Deduplication | Single-prompt suite | **Curated (10)** | 10 | **YES** |
| Contradiction | Single-prompt suite | Curated (6) + Real DB (10) | 16 | No |
| Lesson Extraction | Single-prompt suite | Curated (5) + Real DB (10) | 15 | No |
| Entity Extraction | Single-prompt suite | Curated (5) + Real DB (20) | 25 | No |
| Entity Resolution | Single-prompt suite | **Curated (12)** | 12 | **YES** |
| Tag Discovery | Single-prompt suite | **Curated (8)** | 8 | **YES** |
| Batch Tag Assignment | Single-prompt suite | Real DB (30, with ground truth) | 30 | No |
| Session Reflection | Single-prompt suite | Real DB (5 sessions) | 5 | No |
| Session Titling | Single-prompt suite | **Real DB (10 sessions)** | 10 | **YES** |
| Project Digest | Single-prompt suite | **Real DB (3 projects)** or synthetic | 3 | **YES** |
| Self-Reflection | Single-prompt suite | **Curated (8)** | 8 | **YES** |
| Plan Generation | Single-prompt suite | **Curated (6)** | 6 | **YES** |
| Tool Calling | Multi-turn harness | Existing simple-chat (14) + reasoning (4) | 18 | No |
| Coding | Multi-turn harness | Existing coding (8) + stress (65) | 73 | No |

**Total: 297 test cases. 57 new cases to author across 7 roles.**

### Coverage Gaps

| Role | Current Coverage | Gap |
|------|-----------------|-----|
| **Entity Resolution** | Zero. Unit tests mock the LLM. | No prompt quality benchmark. |
| **Tag Discovery** | Zero. Only parser unit tests. | No prompt quality benchmark. |
| **Project Digest** | Zero. Code marked "NEEDS TESTING". | Never run on real data. |
| **Self-Reflection** | 1 synthetic case in benchmark_reasoning.py. | Effectively uncovered. |
| **Session Titling** | Zero. Only used in backfill script. | No quality benchmark. |
| **Plan Generation** | 1 synthetic case in benchmark_reasoning.py. | Effectively uncovered. |
| **Deduplication** | benchmark_pipeline_speed.py tests speed + action distribution only. | No decision accuracy metric. |
| **Lesson Extraction** | benchmark_models.py captures raw output. | No quality metric (just "did it respond?"). |

**Fully covered (existing benchmarks reusable as-is):** Summarization, Scoring, Contradiction, Entity Extraction, Batch Tag Assignment, Session Reflection, Tool Calling, Coding.

---

## 4. Unified Architecture

### 4.1 Entry Point

```bash
# Run everything on candidate models
python benchmark_unified.py \
  --models qwen3:14b,glm-5:cloud,gpt-oss:latest \
  --db data/blipshell.db \
  --sample 20 \
  --suite all

# Run a specific suite
python benchmark_unified.py --suite pipeline --models qwen3:14b,qwen2.5:14b

# Run stress tests with a config
python benchmark_unified.py --suite interactive --config config.yaml
```

### 4.2 Suite Grouping

| Suite | Roles Included | Est. Cases |
|-------|---------------|------------|
| `pipeline` | Summarization, Scoring, Dedup, Contradiction, Lesson Extraction | 89 |
| `extraction` | Entity Extraction, Entity Resolution, Tag Discovery, Batch Tags | 75 |
| `synthesis` | Session Reflection, Session Titling, Project Digest, Self-Reflection, Plan Generation | 42 |
| `interactive` | Tool Calling, Coding | 91 |
| `all` | Everything | 297 |

### 4.3 Execution Order

1. **Single-prompt suites** run first (`pipeline` → `extraction` → `synthesis`). Each role creates a temporary router pointed at the model under test, fires all test cases through the production prompt + parser, and scores results.
2. **Interactive suites** run last (`tool_calling` → `coding`). Each test spins up a full Agent, runs the task, captures the structured report, and scores via verify_checks.
3. Results merge into one JSON file and one Rich table printed to terminal.

### 4.4 Estimated Wall-Clock Time (Per Model)

| Suite | Cases | Local 14b | Cloud |
|-------|-------|-----------|-------|
| pipeline | 89 | ~6 min | ~2 min |
| extraction | 75 | ~5 min | ~1.5 min |
| synthesis | 42 | ~4 min | ~1 min |
| interactive | 91 | ~45 min | ~15 min |
| **Total** | **297** | **~60 min** | **~20 min** |

### 4.5 Output: Terminal Table

```
Model          │ summ │ score│ dedup│ contr│ lssn │ ent  │ e_res│ tag_d│ b_tag│ refl │ title│dgst │ s_rfl│ plan │ chat │ code │  AVG │ avg_s
───────────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼─────┼──────┼──────┼──────┼──────┼──────┼──────
qwen3:14b      │ 0.92 │ 0.88 │ 0.85 │ 0.95 │ 0.80 │ 0.78 │ 0.90 │ 0.75 │ 0.82 │ 0.88 │ 0.85 │0.80 │ 0.70 │ 0.85 │  —   │  —   │ 0.84 │  4.2
glm-5:cloud    │ 0.95 │ 0.90 │ 0.88 │ 0.92 │ 0.85 │ 0.82 │ 0.92 │ 0.80 │ 0.85 │ 0.90 │ 0.88 │0.85 │ 0.80 │ 0.90 │ 0.92 │ 0.85 │ 0.87 │  1.8
```

`—` = model not applicable for that role (e.g., small models skipped for interactive). `AVG` = mean of non-null role scores. `avg_s` = mean latency in seconds.

### 4.6 Output: JSON File

```json
{
  "meta": {
    "timestamp": "2026-03-03T...",
    "sample_size": 20,
    "db_path": "data/blipshell.db"
  },
  "models": {
    "qwen3:14b": {
      "summarization": { "score": 0.92, "avg_time": 3.1, "cases": 28, "details": [...] },
      "scoring":       { "score": 0.88, "avg_time": 2.4, "cases": 20, "details": [...] }
    }
  }
}
```

---

## 5. Migration: What Gets Absorbed from Existing Benchmarks

| Existing File | Absorbed Into | What Changes |
|--------------|---------------|--------------|
| benchmark_models.py | `pipeline` + `extraction` suites | Curated test cases reused, scoring normalized to 0–1 |
| benchmark_realdata.py | `pipeline` + `extraction` suites (real-data portion) | Sample loading reused, new scoring metrics applied |
| benchmark_pipeline_speed.py | `pipeline` suite (speed column) | Speed data captured alongside quality |
| benchmark_tagger.py | `extraction` suite (batch_tags role) | F1 scoring reused as-is |
| benchmark_reflection.py | `synthesis` suite (reflection role) | Section/specificity scoring reused, normalized |
| benchmark_reasoning.py | `synthesis` (plan, self-reflection) + `interactive` (tool_calling) | Tests absorbed, new scoring applied |
| benchmark_coding.py | `interactive` suite (coding role) | Verify checks reused, scoring normalized |
| test_executor.py --stress | `interactive` suite (coding role, expanded) | 65 stress tests become coding test cases |
| test_executor.py --simple-chat | `interactive` suite (tool_calling role) | 14 tests become tool_calling test cases |

**Old files remain runnable** for backward compatibility but are no longer the source of truth.

---

## 6. New Test Data Required

Author these **before** writing benchmark code.

| Role | Cases | What To Write |
|------|-------|---------------|
| Deduplication | 10 | Pairs of memory summaries with expected action (ADD/UPDATE/DELETE/NONE) |
| Entity Resolution | 12 | Pairs of entity names with expected YES/NO |
| Tag Discovery | 8 | Sets of poorly-tagged summaries + existing tag lists + expected behavior |
| Session Titling | 10 | Sessions with manually verified ideal titles (for keyword-relevance check) |
| Project Digest | 3 | Projects with session history — can generate from DB |
| Self-Reflection | 8 | (response, flawed?) pairs — 4 good, 4 with known errors |
| Plan Generation | 6 | Task descriptions with expected step count and tool hints |

**Total: 57 new test cases across 7 files.**

---

## 7. Models To Test

Default model list covering all production models + candidates:

| Model | Type | Relevant Roles |
|-------|------|---------------|
| qwen3:14b | Local | All pipeline + extraction + synthesis (current REASONING model) |
| qwen2.5:14b | Local | Scoring, batch tags (current RANKING models) |
| qwen2.5:7b | Local | Scoring fallback — pipeline + extraction |
| glm4:latest | Local | Summarization (current SUMMARIZATION model) |
| gpt-oss:latest | Local | All roles (current fallback for tool_calling/coding/reasoning) |
| glm-5:cloud | Cloud | All roles including interactive (current TOOL_CALLING/CODING model) |
| Groq llama-3.3-70b | Cloud | Pipeline roles (current RANKING_IMPORTANCE cloud model) |
| Groq gpt-oss-120b | Cloud | Pipeline roles (current SUMMARIZATION cloud model) |

User can override via `--models`. Interactive roles only run on models flagged as capable (configurable).

---

## 8. Model Selection Workflow

1. **Author test data** for the 7 gap roles (57 cases).
2. **Run full suite** on candidate models:
   ```bash
   python benchmark_unified.py --suite all --models qwen3:14b,qwen2.5:14b,glm4:latest,glm-5:cloud
   ```
3. **Review results table** — compare composite scores per role.
4. **Select per-role best**: fast + accurate for background tasks, capable for interactive tasks, balanced cost/speed for batch tasks.
5. **Update `config.yaml`**:
   ```yaml
   models:
     reasoning: "qwen3:14b"
     summarization: "glm4:latest"
     ranking_importance: "groq:llama-3.3-70b-versatile"
     tool_calling: "glm-5:cloud"
     coding: "glm-5:cloud"
   ```

---

## 9. Implementation Checklist

### Existing (Reusable As-Is)

- [x] `benchmark_models.py` — Pipeline curated tests (summarization, ranking, importance, entity, contradiction, dedup, lessons)
- [x] `benchmark_realdata.py` — Pipeline tests on real DB data
- [x] `benchmark_reasoning.py` — Plan generation, self-reflection, tool calling (synthetic)
- [x] `benchmark_tagger.py` — Batch tag assignment (30 ground-truth memories)
- [x] `benchmark_reflection.py` — Session reflection (5 real sessions)
- [x] `benchmark_embeds.py` — Embedding similarity
- [x] `test_executor.py --stress` — 65 functional executor tests
- [x] `test_executor.py --simple-chat` — 14 tool-calling functional tests

### New: Test Data (Author First)

- [ ] Dedup curated pairs (10 cases)
- [ ] Entity resolution curated pairs (12 cases)
- [ ] Tag discovery test sets (8 cases)
- [ ] Session titling verified titles (10 cases)
- [ ] Project digest test projects (3 cases)
- [ ] Self-reflection response pairs (8 cases)
- [ ] Plan generation task descriptions (6 cases)

### New: Benchmark Infrastructure

- [ ] **Unified runner**: `benchmark_unified.py` — Single CLI entry point with `--suite`, `--models`, `--db`, `--sample` flags
- [ ] **Suite registry**: Suite discovery and grouping (pipeline, extraction, synthesis, interactive, all)
- [ ] **Results aggregation**: Cross-suite merge into one JSON + one Rich terminal table
- [ ] **Per-role scorers**: Implement the composite scoring formulas from Section 2 for each role
- [ ] **Session Titling benchmark**: New scorer for title quality
- [ ] **Tag Discovery benchmark**: New scorer for pattern extraction quality
- [ ] **Entity Resolution benchmark**: New scorer for YES/NO accuracy
- [ ] **Project Digest benchmark**: New scorer for digest completeness
- [ ] **Self-Reflection benchmark**: New scorer for improvement detection
- [ ] **Plan Generation benchmark**: New scorer for step quality
- [ ] **Deduplication benchmark**: Upgrade from speed-only to decision accuracy

---

## 10. Summary

This spec provides:

1. **Complete role inventory** — all 16 LLM calls mapped with call sites, task types, and current models.
2. **Precise quality metrics** — binary/normalized scoring rubrics for every role, no subjective evaluation.
3. **Existing coverage audit** — 8 roles fully covered, 8 with gaps ranging from zero coverage to missing quality metrics.
4. **Unified architecture** — one command, one JSON output, one terminal table. Four suite groups with defined execution order.
5. **Migration path** — every existing benchmark file maps to a suite, old files stay runnable.
6. **Implementation roadmap** — 57 test cases to author, then infrastructure to build. Test data first, code second.
