# BlipShell model benchmark
_Generated 2026-09-25T14:22:30.451667+00:00_

## Which model to use where

One block per key in `config.yaml`. **Every job a key controls is listed together**, because a key is a single choice: a model that wins one of its jobs and loses another is not an upgrade. UNKNOWN means the incumbent has never been measured on its own job -- run the command shown and it resolves.

Model identifiers are endpoint-specific: `minimax/minimax-m3` (OpenRouter) and `minimax-m3:cloud` (Ollama cloud) are different serving stacks and are NOT treated as the same measurement. If a similar name appears in a table below but the incumbent still reads UNKNOWN, that is deliberate -- benchmark the identifier you actually route to.

### `models.coding` -> minimax/minimax-m3   [UNKNOWN]
Controls: project-mode coding (cli project path, background coding tasks)

| Model | coding_agentic | code_gen |
|---|---|---|
| devstral-small-2:24b | 0.190 | not measured |
| gemma4:31b-cloud | 0.320 | not measured |
| gpt-oss:latest | 0.190 | not measured |

**UNKNOWN** -- minimax/minimax-m3 has no score for coding_agentic, code_gen -- the key's own job(s).

```
blipshell benchmark run minimax/minimax-m3 --jobs coding,reasoning
```

### `models.embedding` -> qwen3-embedding:0.6b   [UNKNOWN]
Controls: all vector search (memories, lessons, core, entities, self-thoughts)

_Nothing measured for this key yet._

**UNKNOWN** -- qwen3-embedding:0.6b has no score for embedding -- the key's own job(s).

```
blipshell benchmark run qwen3-embedding:0.6b --jobs embedding
```

### `models.ranking_importance` -> openai/gpt-oss-120b   [UNKNOWN]
Controls: the live memory pipeline's combined rank+importance call

| Model | rank_importance |
|---|---|
| deepseek/deepseek-v4-flash | 0.979 |
| deepseek/deepseek-v4.1-flash | 0.978 |
| devstral-small-2:24b | 0.972 |
| gemma4:31b-cloud | 0.979 |
| gemma4:e4b | 0.962 |
| glm-4.7-flash:latest | 0.952 |
| gpt-oss:120b-cloud | 0.953 |
| gpt-oss:latest | 0.945 |
| lfm2.5:latest | 0.582 |
| minimax/minimax-m3 | 0.972 |
| minimax/minimax-m3:free | 0.985 |
| ministral-3:14b | 0.956 |
| ministral-3:8b | 0.965 |
| moonshotai/kimi-k2.5 | 0.964 |
| nemotron-3-nano:30b-cloud | 0.924 |
| nemotron-3-super:cloud | 0.974 |
| nemotron-3-ultra:cloud | 0.970 |
| openai/gpt-5.6-luna | 0.978 |
| openai/gpt-6-luna | 0.981 |
| phi4:14b | 0.963 |
| qwen2.5-coder:14b | 0.943 |
| qwen2.5:14b | 0.966 |
| qwen3-coder:30b | 0.960 |
| qwen3.5:4b | 0.945 |
| qwen3.5:9b | 0.956 |
| qwen3.6:27b | 0.975 |
| qwen3:14b | 0.961 |
| qwen3:8b | 0.941 |
| z-ai/glm-5.2 | 0.976 |

**UNKNOWN** -- openai/gpt-oss-120b has no score for rank_importance -- the key's own job(s).

```
blipshell benchmark run openai/gpt-oss-120b --jobs pipeline
```

### `models.reasoning` -> qwen3:14b   [UNKNOWN]
Controls: entity extraction + merge, contradiction checks, the write-time dedup verdict (processor._decide_and_apply_action), tag discovery, project digests, guardrail audits, self-thought relevance judge, context compaction

| Model | reasoning | entity | contradiction | dedup | dedup_structured |
|---|---|---|---|---|---|
| **qwen3:14b** (current) | not measured | 0.833 | 1.000 | 0.750 | 0.750 |
| deepseek/deepseek-v4-flash | not measured | 0.779 | 1.000 | not measured | not measured |
| deepseek/deepseek-v4.1-flash | not measured | 0.817 | 0.917 | 0.917 | 1.000 |
| devstral-small-2:24b | not measured | 0.753 | 1.000 | 0.750 | 0.611 |
| gemma4:31b-cloud | not measured | 0.811 | 1.000 | 1.000 | 1.000 |
| gemma4:e4b | not measured | 0.818 | 1.000 | 0.889 | 0.611 |
| glm-4.7-flash:latest | not measured | 0.808 | 1.000 | 0.583 | 0.611 |
| glm-5.2:cloud | not measured | 0.790 | 1.000 | not measured | not measured |
| gpt-oss:120b-cloud | not measured | 0.820 | 1.000 | 1.000 | 1.000 |
| gpt-oss:latest | not measured | 0.801 | 1.000 | 0.889 | not measured |
| kimi-k2.7-code:cloud | not measured | 0.798 | 1.000 | not measured | not measured |
| lfm2.5 | not measured | 0.356 | not measured | not measured | not measured |
| lfm2.5:latest | not measured | 0.341 | not measured | 0.555 | 0.167 |
| minimax-m3:cloud | not measured | 0.868 | 1.000 | not measured | not measured |
| minimax/minimax-m3 | not measured | 0.818 | 1.000 | 1.000 | 1.000 |
| minimax/minimax-m3:free | not measured | 0.811 | 1.000 | not measured | not measured |
| ministral-3:14b | not measured | 0.762 | 1.000 | 0.805 | 0.833 |
| ministral-3:8b | not measured | 0.844 | 0.972 | 0.722 | 0.833 |
| moonshotai/kimi-k2.5 | not measured | 0.811 | 1.000 | 1.000 | 1.000 |
| nemotron-3-nano:30b-cloud | not measured | 0.597 | 1.000 | 0.417 | 0.583 |
| nemotron-3-super:cloud | not measured | 0.723 | 0.750 | 0.583 | 0.545 |
| nemotron-3-ultra:cloud | not measured | 0.694 | 1.000 | not measured | not measured |
| openai/gpt-5.6-luna | not measured | 0.884 | 1.000 | 1.000 | 0.833 |
| openai/gpt-6-luna | not measured | 0.782 | 0.917 | 1.000 | 0.917 |
| phi4:14b | not measured | 0.789 | 1.000 | 0.472 | 0.472 |
| qwen2.5-coder:14b | not measured | 0.782 | 0.917 | 0.611 | 0.694 |
| qwen2.5:14b | not measured | 0.815 | 0.917 | 0.722 | 0.833 |
| qwen3-coder:30b | not measured | 0.760 | 1.000 | 0.555 | 0.611 |
| qwen3.5:4b | not measured | 0.834 | 0.778 | 0.389 | 0.528 |
| qwen3.5:9b | not measured | 0.887 | 0.972 | 0.750 | 0.833 |
| qwen3.6:27b | not measured | 0.837 | 1.000 | 0.750 | 0.917 |
| qwen3:8b | not measured | 0.780 | 0.917 | 0.528 | 0.750 |
| z-ai/glm-5.2 | not measured | 0.869 | 1.000 | 1.000 | 1.000 |

**UNKNOWN** -- qwen3:14b has no score for reasoning -- the key's own job(s).

```
blipshell benchmark run qwen3:14b --jobs reasoning
```

### `models.session_review` -> deepseek/deepseek-v4-flash   [UNKNOWN]
Controls: session reflections (incl. the chunk+merge path for oversized sessions), LESSON EXTRACTION, friction analysis

| Model | session_review | lessons | session_review_chunked |
|---|---|---|---|
| deepseek/deepseek-v4.1-flash | not measured | not measured | 1.000 |
| devstral-small-2:24b | not measured | not measured | 0.900 |
| gemma4:31b-cloud | not measured | not measured | 0.900 |
| gemma4:e4b | not measured | not measured | 0.900 |
| glm-4.7-flash:latest | not measured | not measured | 0.833 |
| gpt-oss:120b-cloud | not measured | not measured | 0.900 |
| gpt-oss:latest | not measured | not measured | 0.933 |
| lfm2.5:latest | not measured | not measured | 0.867 |
| minimax/minimax-m3 | not measured | not measured | 1.000 |
| ministral-3:14b | not measured | not measured | 0.900 |
| ministral-3:8b | not measured | not measured | 0.900 |
| moonshotai/kimi-k2.5 | not measured | not measured | 1.000 |
| nemotron-3-nano:30b-cloud | not measured | not measured | 0.900 |
| nemotron-3-super:cloud | not measured | not measured | 0.800 |
| openai/gpt-5.6-luna | not measured | not measured | 0.900 |
| openai/gpt-6-luna | not measured | not measured | 0.900 |
| phi4:14b | not measured | not measured | 0.867 |
| qwen2.5-coder:14b | not measured | not measured | 0.900 |
| qwen2.5:14b | not measured | not measured | 0.833 |
| qwen3-coder:30b | not measured | not measured | 0.933 |
| qwen3.5:4b | not measured | not measured | 0.867 |
| qwen3.5:9b | not measured | not measured | 0.900 |
| qwen3:14b | not measured | not measured | 0.867 |
| qwen3:8b | not measured | not measured | 0.833 |
| z-ai/glm-5.2 | not measured | not measured | 0.900 |

**UNKNOWN** -- deepseek/deepseek-v4-flash has no score for session_review, lessons, session_review_chunked -- the key's own job(s).

```
blipshell benchmark run deepseek/deepseek-v4-flash --jobs pipeline,session_review
```

### `models.summarization` -> openai/gpt-oss-120b   [UNKNOWN]
Controls: memory + session summaries, web-fetch summaries, imports

_Nothing measured for this key yet._

**UNKNOWN** -- openai/gpt-oss-120b has no score for summarization -- the key's own job(s).

```
blipshell benchmark run openai/gpt-oss-120b --jobs pipeline
```

### `models.tool_calling` -> deepseek/deepseek-v4-flash   [UNKNOWN]
Controls: interactive chat + executor tool loop (agent_chat, executor, planner)

| Model | tool_calling | reasoning | code_gen |
|---|---|---|---|
| **deepseek/deepseek-v4-flash** (current) | 0.973 | not measured | not measured |
| deepseek-v4-flash:cloud | 0.733 | not measured | not measured |
| deepseek/deepseek-v4.1-flash | 1.000 | not measured | not measured |
| devstral-small-2:24b | 1.000 | not measured | not measured |
| gemma4:31b-cloud | 1.000 | not measured | not measured |
| gemma4:e4b | 1.000 | not measured | not measured |
| glm-4.7-flash:latest | 0.978 | not measured | not measured |
| glm-5.1:cloud | 0.800 | not measured | not measured |
| glm-5.2:cloud | 0.800 | not measured | not measured |
| glm4:latest | 0.000 | not measured | not measured |
| gpt-oss:120b-cloud | 1.000 | not measured | not measured |
| gpt-oss:latest | 1.000 | not measured | not measured |
| kimi-k2.7-code:cloud | 0.800 | not measured | not measured |
| lfm2.5 | 0.933 | not measured | not measured |
| lfm2.5:latest | 0.956 | not measured | not measured |
| minimax-m2.7:cloud | 0.867 | not measured | not measured |
| minimax-m3:cloud | 0.667 | not measured | not measured |
| minimax/minimax-m3 | 0.933 | not measured | not measured |
| minimax/minimax-m3:free | 0.000 | not measured | not measured |
| ministral-3:14b | 0.667 | not measured | not measured |
| ministral-3:8b | 0.933 | not measured | not measured |
| moonshotai/kimi-k2.5 | 1.000 | not measured | not measured |
| nemotron-3-nano:30b-cloud | 1.000 | not measured | not measured |
| nemotron-3-super:cloud | 1.000 | not measured | not measured |
| nemotron-3-ultra:cloud | 0.000 | not measured | not measured |
| openai/gpt-5.6-luna | 1.000 | not measured | not measured |
| openai/gpt-6-luna | 1.000 | not measured | not measured |
| qwen2.5-coder:14b | 0.000 | not measured | not measured |
| qwen2.5-coder:7b | 0.000 | not measured | not measured |
| qwen2.5:14b | 0.978 | not measured | not measured |
| qwen2.5:7b | 0.867 | not measured | not measured |
| qwen3-coder:30b | 0.933 | not measured | not measured |
| qwen3.5:4b | 1.000 | not measured | not measured |
| qwen3.5:9b | 1.000 | not measured | not measured |
| qwen3.6:27b | 0.000 | not measured | not measured |
| qwen3:14b | 0.978 | not measured | not measured |
| qwen3:8b | 1.000 | not measured | not measured |
| z-ai/glm-5.2 | 0.933 | not measured | not measured |

**UNKNOWN** -- deepseek/deepseek-v4-flash has no score for reasoning, code_gen -- the key's own job(s).

```
blipshell benchmark run deepseek/deepseek-v4-flash --jobs reasoning
```

### `models.ranking` -> qwen3:14b   [CONSIDER]
Controls: batch tagger + import path only (not the live pipeline)

| Model | ranking |
|---|---|
| **qwen3:14b** (current) | 0.924 |
| deepseek/deepseek-v4-flash | 0.964 |
| deepseek/deepseek-v4.1-flash | 0.972 |
| devstral-small-2:24b | 0.925 |
| gemma4:31b-cloud | 0.957 |
| gemma4:e4b | 0.930 |
| glm-4.7-flash:latest | 0.874 |
| glm-5.2:cloud | 0.961 |
| gpt-oss:120b-cloud | 0.957 |
| gpt-oss:latest | 0.940 |
| kimi-k2.7-code:cloud | 0.953 |
| lfm2.5 | 0.515 |
| lfm2.5:latest | 0.515 |
| minimax-m3:cloud | 0.973 |
| minimax/minimax-m3 | 0.967 |
| minimax/minimax-m3:free | 0.967 |
| ministral-3:14b | 0.959 |
| ministral-3:8b | 0.956 |
| moonshotai/kimi-k2.5 | 0.937 |
| nemotron-3-nano:30b-cloud | 0.814 |
| nemotron-3-super:cloud | 0.958 |
| nemotron-3-ultra:cloud | 0.969 |
| openai/gpt-5.6-luna | 0.976 |
| openai/gpt-6-luna | 0.966 |
| phi4:14b | 0.912 |
| qwen2.5-coder:14b | 0.923 |
| qwen2.5:14b | 0.958 |
| qwen3-coder:30b | 0.889 |
| qwen3.5:4b | 0.936 |
| qwen3.5:9b | 0.904 |
| qwen3.6:27b | 0.926 |
| qwen3:8b | 0.869 |
| z-ai/glm-5.2 | 0.973 |

**CONSIDER** -- ministral-3:14b beats qwen3:14b on ranking (+0.035) with no regression on this key's other jobs (mean gain across all its jobs: +0.035).

### `models.importance` -> qwen3:14b   [KEEP]
Controls: import path only (live pipeline uses ranking_importance)

| Model | importance |
|---|---|
| **qwen3:14b** (current) | 0.929 |
| deepseek/deepseek-v4-flash | 0.933 |
| deepseek/deepseek-v4.1-flash | 0.977 |
| devstral-small-2:24b | 0.887 |
| gemma4:31b-cloud | 0.970 |
| gemma4:e4b | 0.923 |
| glm-4.7-flash:latest | 0.913 |
| glm-5.2:cloud | 0.948 |
| gpt-oss:120b-cloud | 0.884 |
| gpt-oss:latest | 0.883 |
| kimi-k2.7-code:cloud | 0.950 |
| lfm2.5 | 0.621 |
| lfm2.5:latest | 0.637 |
| minimax-m3:cloud | 0.954 |
| minimax/minimax-m3 | 0.960 |
| minimax/minimax-m3:free | 0.964 |
| ministral-3:14b | 0.926 |
| ministral-3:8b | 0.915 |
| moonshotai/kimi-k2.5 | 0.962 |
| nemotron-3-nano:30b-cloud | 0.816 |
| nemotron-3-super:cloud | 0.895 |
| nemotron-3-ultra:cloud | 0.939 |
| openai/gpt-5.6-luna | 0.945 |
| openai/gpt-6-luna | 0.944 |
| phi4:14b | 0.925 |
| qwen2.5-coder:14b | 0.921 |
| qwen2.5:14b | 0.929 |
| qwen3-coder:30b | 0.890 |
| qwen3.5:4b | 0.901 |
| qwen3.5:9b | 0.899 |
| qwen3.6:27b | 0.929 |
| qwen3:8b | 0.891 |
| z-ai/glm-5.2 | 0.949 |

**KEEP** -- No measured candidate beats qwen3:14b by more than the noise floor (0.067) on this key's jobs.

---

## How to read this
BlipShell routes a **mix of local and cloud models per job**, each with a fallback, chosen by endpoint priority and availability. Cloud is generally strongest but we do NOT need cloud for every job — the point of this benchmark is to find, per job, the cheapest model that's good enough. This report makes **no switch recommendation**: it lays out quality and speed (and cost/context for cloud models) per job so you can decide. Higher quality = better; lower latency = faster. Objective jobs are scored locally; open-ended jobs remain blank until a blinded external review is imported. COMPOSITE uses effective score (quality multiplied by completion rate).

Benchmark contract: **v2.0**, dataset **2026-09-23**. Required coverage is fixed at 15 production categories.

## Quality by job (higher is better)
| Job | deepseek-v3.2:cloud | deepseek-v4-flash:cloud | deepseek/deepseek-v4-flash | deepseek/deepseek-v4.1-flash | devstral-small-2:24b | gemini-3-flash-preview:cloud | gemma3n:e4b | gemma4:31b-cloud | gemma4:e4b | glm-4.7-flash:latest | glm-4.7:cloud | glm-5.1:cloud | glm-5.2:cloud | glm-5:cloud | glm4:latest | gpt-oss:120b-cloud | gpt-oss:latest | kimi-k2-thinking:cloud | kimi-k2.5:cloud | kimi-k2.7-code:cloud | kimi-k3:cloud | lfm2.5 | lfm2.5:latest | minimax-m2.5:cloud | minimax-m2.7:cloud | minimax-m3:cloud | minimax/minimax-m3 | minimax/minimax-m3:free | ministral-3:14b | ministral-3:8b | moonshotai/kimi-k2.5 | nemotron-3-nano:30b-cloud | nemotron-3-super:cloud | nemotron-3-ultra:cloud | openai/gpt-5.6-luna | openai/gpt-6-luna | phi4:14b | qwen2.5-coder:14b | qwen2.5-coder:7b | qwen2.5:14b | qwen2.5:7b | qwen3-coder-next:cloud | qwen3-coder:30b | qwen3.5:4b | qwen3.5:9b | qwen3.6:27b | qwen3:14b | qwen3:8b | z-ai/glm-5.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rank+Importance (live pipeline) | — | — | 0.979 | 0.978 | 0.972 | — | — | 0.979 | 0.962 +/-0.00 | 0.952 +/-0.01 | — | — | — | — | — | 0.953 | 0.945 +/-0.04 | — | — | — | — | — | 0.582 +/-0.10 | — | — | — | 0.972 | **0.985** | 0.956 +/-0.02 | 0.965 +/-0.00 | 0.964 | 0.924 | 0.974 | 0.970 | 0.978 | 0.981 | 0.963 +/-0.01 | 0.943 +/-0.02 | — | 0.966 | — | — | 0.960 +/-0.01 | 0.945 +/-0.03 | 0.956 +/-0.03 | 0.975 @ 33% complete | 0.961 +/-0.01 | 0.941 +/-0.01 | 0.976 |
| Ranking (import path) | — | — | 0.964 | 0.972 | 0.925 +/-0.02 | — | — | 0.957 | 0.930 +/-0.03 | 0.874 +/-0.04 | — | — | 0.961 | — | — | 0.957 | 0.940 +/-0.03 | — | — | 0.953 | — | 0.515 | 0.515 | — | — | 0.973 | 0.967 | 0.967 | 0.959 +/-0.01 | 0.956 +/-0.01 | 0.937 | 0.814 | 0.958 | 0.969 | **0.976** | 0.966 | 0.912 +/-0.04 | 0.923 +/-0.01 | — | 0.958 | — | — | 0.889 +/-0.02 | 0.936 +/-0.06 | 0.904 +/-0.06 | 0.926 @ 33% complete | 0.924 | 0.869 +/-0.00 | 0.973 |
| Importance (import path) | — | — | 0.933 | **0.977** | 0.887 +/-0.01 | — | — | 0.970 | 0.923 +/-0.02 | 0.913 +/-0.02 | — | — | 0.948 | — | — | 0.884 | 0.883 +/-0.11 | — | — | 0.950 | — | 0.621 | 0.637 +/-0.00 | — | — | 0.954 | 0.960 | 0.964 | 0.926 | 0.915 +/-0.01 | 0.962 | 0.816 | 0.895 | 0.939 | 0.945 | 0.944 | 0.925 +/-0.02 | 0.921 +/-0.08 | — | 0.929 +/-0.02 | — | — | 0.890 +/-0.02 | 0.901 +/-0.04 | 0.899 +/-0.03 | 0.929 @ 33% complete | 0.929 +/-0.02 | 0.891 +/-0.02 | 0.949 |
| Contradiction | — | — | **1.000** | 0.917 | 1.000 | — | — | 1.000 | 1.000 | 1.000 | — | — | 1.000 | — | — | 1.000 | 1.000 | — | — | 1.000 | — | — | — | — | — | 1.000 | 1.000 | 1.000 | 1.000 | 0.972 +/-0.08 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.917 | 1.000 | 0.917 +/-0.17 | — | 0.917 | — | — | 1.000 | 0.778 +/-0.08 | 0.972 +/-0.08 | 1.000 @ 33% complete | 1.000 | 0.917 | 1.000 |
| Dedup verdict (live pipeline) | — | — | — | 0.917 | 0.750 | — | — | **1.000** | 0.889 +/-0.08 | 0.583 +/-0.25 | — | — | — | — | — | 1.000 | 0.889 +/-0.17 | — | — | — | — | — | 0.555 +/-0.08 | — | — | — | 1.000 | — | 0.805 +/-0.08 | 0.722 +/-0.08 | 1.000 | 0.417 | 0.583 | — | 1.000 | 1.000 | 0.472 +/-0.08 | 0.611 +/-0.33 | — | 0.722 +/-0.08 | — | — | 0.555 +/-0.08 | 0.389 +/-0.25 | 0.750 | 0.750 @ 33% complete | 0.750 | 0.528 +/-0.08 | 1.000 |
| Dedup verdict (structured, experimental) | — | — | — | **1.000** | 0.611 +/-0.08 | — | — | 1.000 | 0.611 +/-0.08 | 0.611 +/-0.17 | — | — | — | — | — | 1.000 | — | — | — | — | — | — | 0.167 | — | — | — | 1.000 | — | 0.833 | 0.833 | 1.000 | 0.583 | 0.545 | — | 0.833 | 0.917 | 0.472 +/-0.08 | 0.694 +/-0.17 | — | 0.833 | — | — | 0.611 +/-0.08 | 0.528 +/-0.25 | 0.833 +/-0.17 | 0.917 | 0.750 | 0.750 | 1.000 |
| Entity extraction | — | — | 0.779 | 0.817 | 0.753 +/-0.05 | — | — | 0.811 | 0.818 +/-0.04 | 0.808 +/-0.08 | — | — | 0.790 | — | — | 0.820 | 0.801 +/-0.13 | — | — | 0.798 | — | 0.356 | 0.341 +/-0.18 | — | — | 0.868 | 0.818 | 0.811 | 0.762 +/-0.10 | 0.844 +/-0.05 | 0.811 | 0.597 | 0.723 | 0.694 | 0.884 | 0.782 | 0.789 +/-0.10 | 0.782 +/-0.03 | — | 0.815 +/-0.01 | — | — | 0.760 +/-0.08 | 0.834 +/-0.04 | **0.887** +/-0.01 | 0.837 @ 33% complete | 0.833 +/-0.02 | 0.780 +/-0.05 | 0.869 |
| Coding (agentic) | — | — | — | — | 0.190 | — | — | **0.320** +/-0.65 | — | — | — | — | — | — | — | — | 0.190 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Coding (legacy - ambiguous) | — | — | — | — | — | — | — | — | — | — | — | — | **0.905** | — | — | — | — | — | — | 0.886 | — | 0.253 | — | — | — | 0.861 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Tool calling | — | 0.733 | 0.973 +/-0.07 | **1.000** | 1.000 | — | — | 1.000 | 1.000 | 0.978 +/-0.07 | — | 0.800 | 0.800 | — | 0.000 | 1.000 | 1.000 | — | — | 0.800 | — | 0.933 | 0.956 +/-0.13 | — | 0.867 | 0.667 | 0.933 | 0.000 | 0.667 | 0.933 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | — | 0.000 | 0.000 | 0.978 +/-0.07 | 0.867 | — | 0.933 +/-0.13 | 1.000 | 1.000 | 0.000 @ 0% complete | 0.978 +/-0.07 | 1.000 | 0.933 |
| Session review (chunked, multi-call) | — | — | — | **1.000** | 0.900 | — | — | 0.900 | 0.900 | 0.833 +/-0.10 | — | — | — | — | — | 0.900 | 0.933 +/-0.10 | — | — | — | — | — | 0.867 +/-0.10 | — | — | — | 1.000 | — | 0.900 | 0.900 | 1.000 | 0.900 | 0.800 | — | 0.900 | 0.900 | 0.867 +/-0.10 | 0.900 +/-0.30 | — | 0.833 +/-0.10 | — | — | 0.933 +/-0.10 | 0.867 +/-0.20 | 0.900 | — | 0.867 +/-0.10 | 0.833 +/-0.10 | 0.900 |
| **COMPOSITE** | — | 0.733 (partial) | 0.938 (partial) | 0.947 (partial) | 0.820 (partial) | — | — | 0.882 (partial) | 0.928 (partial) | 0.868 (partial) | — | 0.800 (partial) | 0.900 (partial) | — | 0.000 (partial) | 0.939 (partial) | 0.842 (partial) | — | — | 0.900 (partial) | — | 0.606 (partial) | 0.636 (partial) | — | 0.867 (partial) | 0.892 (partial) | 0.956 (partial) | 0.788 (partial) | 0.872 (partial) | 0.901 (partial) | 0.959 (partial) | 0.808 (partial) | 0.835 (partial) | 0.762 (partial) | 0.960 (partial) | 0.936 (partial) | 0.847 (partial) | 0.750 (partial) | 0.000 (partial) | 0.890 (partial) | 0.867 (partial) | — | 0.865 (partial) | 0.831 (partial) | 0.909 (partial) | 0.258 (partial) | 0.905 (partial) | 0.845 (partial) | 0.950 (partial) |

**(partial) Incomplete coverage:** deepseek-v3.2:cloud (0/15 jobs), deepseek-v4-flash:cloud (1/15 jobs), deepseek/deepseek-v4-flash (6/15 jobs), deepseek/deepseek-v4.1-flash (8/15 jobs), devstral-small-2:24b (9/15 jobs), gemini-3-flash-preview:cloud (0/15 jobs), gemma3n:e4b (0/15 jobs), gemma4:31b-cloud (9/15 jobs), gemma4:e4b (8/15 jobs), glm-4.7-flash:latest (8/15 jobs), glm-4.7:cloud (0/15 jobs), glm-5.1:cloud (1/15 jobs), glm-5.2:cloud (5/15 jobs), glm-5:cloud (0/15 jobs), glm4:latest (1/15 jobs), gpt-oss:120b-cloud (8/15 jobs), gpt-oss:latest (9/15 jobs), kimi-k2-thinking:cloud (0/15 jobs), kimi-k2.5:cloud (0/15 jobs), kimi-k2.7-code:cloud (5/15 jobs), kimi-k3:cloud (0/15 jobs), lfm2.5 (4/15 jobs), lfm2.5:latest (7/15 jobs), minimax-m2.5:cloud (0/15 jobs), minimax-m2.7:cloud (1/15 jobs), minimax-m3:cloud (5/15 jobs), minimax/minimax-m3 (8/15 jobs), minimax/minimax-m3:free (6/15 jobs), ministral-3:14b (8/15 jobs), ministral-3:8b (8/15 jobs), moonshotai/kimi-k2.5 (8/15 jobs), nemotron-3-nano:30b-cloud (8/15 jobs), nemotron-3-super:cloud (8/15 jobs), nemotron-3-ultra:cloud (6/15 jobs), openai/gpt-5.6-luna (8/15 jobs), openai/gpt-6-luna (8/15 jobs), phi4:14b (7/15 jobs), qwen2.5-coder:14b (8/15 jobs), qwen2.5-coder:7b (1/15 jobs), qwen2.5:14b (8/15 jobs), qwen2.5:7b (1/15 jobs), qwen3-coder-next:cloud (0/15 jobs), qwen3-coder:30b (8/15 jobs), qwen3.5:4b (8/15 jobs), qwen3.5:9b (8/15 jobs), qwen3.6:27b (7/15 jobs), qwen3:14b (8/15 jobs), qwen3:8b (8/15 jobs), z-ai/glm-5.2 (8/15 jobs). A composite averaged over fewer jobs is NOT comparable to a full one and cannot win the row — a model that only ran its strongest job would otherwise top the table. Blank cells mean 'not measured', never 'scored zero'. Re-run those models across all jobs before comparing composites.

## Same-configuration comparison cohorts

| Cohort | Models |
|---|---|
| 1 | qwen3:14b, qwen3-coder:30b, qwen2.5-coder:14b, ministral-3:14b, gemma4:e4b, qwen2.5:14b, qwen3:8b, lfm2.5:latest, qwen3.5:4b, glm-4.7-flash:latest, ministral-3:8b, phi4:14b, qwen3.6:27b, qwen3.5:9b |
| 2 | gpt-oss:120b-cloud, nemotron-3-super:cloud, nemotron-3-nano:30b-cloud |
| 3 | openai/gpt-6-luna, deepseek/deepseek-v4.1-flash, minimax/minimax-m3, openai/gpt-5.6-luna, z-ai/glm-5.2, moonshotai/kimi-k2.5 |

## Adjacent system evidence
These suites are kept separate because they measure a different boundary, but are included here so a routing decision does not ignore agent behavior or continuity. Model identifiers are shown exactly as recorded; similarly named serving stacks are not silently merged.

### Multi-turn agent evaluation

| Model | Score | Silent | Turn limit | API errors | s/episode | Run |
|---|---|---|---|---|---|---|
| glm-5.2_cloud | 30/30 | 0 | 0 | 0 | 13.2 | 2026-08-19 |
| gemma4_cloud | 29/30 | 0 | 0 | 0 | 17.0 | 2026-08-19 |
| deepseek-v4-flash_cloud | 28/30 | 0 | 0 | 0 | 13.3 | 2026-08-19 |
| qwen3.5_397b-cloud | 28/30 | 0 | 0 | 0 | 10.0 | 2026-08-19 |
| gemma4_31b-cloud | 27/30 | 0 | 0 | 0 | 10.5 | 2026-08-19 |
| minimax-m2.7_cloud | 27/30 | 0 | 1 | 0 | 11.4 | 2026-08-19 |
| nemotron-3-ultra_cloud | 27/30 | 4 | 4 | 0 | 40.1 | 2026-08-19 |
| minimax-m3_cloud | 26/30 | 0 | 0 | 0 | 7.3 | 2026-08-19 |
| nemotron-3-super_cloud | 26/30 | 3 | 2 | 1 | 24.5 | 2026-08-19 |
| kimi-k2.7-code_cloud | 25/30 | 4 | 1 | 0 | 5.9 | 2026-08-19 |
| mistral-large-3_675b-cloud | 25/30 | 0 | 0 | 0 | 8.5 | 2026-08-19 |
| nemotron-3-nano_30b-cloud | 25/30 | 3 | 3 | 0 | 37.3 | 2026-08-19 |
| gpt-oss_120b-cloud | 24/30 | 4 | 4 | 0 | 7.8 | 2026-08-19 |
| qwen3.5_4b | 21/30 | 9 | 0 | 0 | 8.9 | 2026-08-19 |
| qwen3_4b | 21/30 | 0 | 0 | 0 | 67.6 | 2026-08-19 |
| gpt-oss_20b-cloud | 18/30 | 9 | 0 | 0 | 7.0 | 2026-08-19 |
| lfm2.5_latest | 18/30 | 0 | 0 | 0 | 5.4 | 2026-08-19 |
| qwen2.5_3b | 15/30 | 0 | 0 | 0 | 3.0 | 2026-08-19 |
| qwen2.5_1.5b | 14/30 | 0 | 0 | 0 | 2.7 | 2026-08-19 |
| qwen3_1.7b | 14/30 | 1 | 0 | 0 | 9.9 | 2026-08-19 |
| llama3.2_3b | 9/30 | 0 | 0 | 0 | 2.5 | 2026-08-19 |
| phi4-mini_latest | 7/30 | 0 | 0 | 0 | 2.3 | 2026-08-19 |
| gemma3_1b | 0/30 | 0 | 0 | 30 | 0.7 | 2026-08-19 |
| gemma3_4b | 0/30 | 0 | 0 | 30 | 0.5 | 2026-08-19 |

### Behavioral continuity

| Model | Passed | Failed | Scored |
|---|---|---|---|
| gpt-oss:latest | 12 | 3 | 15 |
| minimax/minimax-m3 | 65 | 25 | 90 |

## Awaiting external review
These open-ended jobs have no imported review score. A new v2 run creates the review sources needed by `blipshell benchmark review export`; review the packet in ChatGPT, Claude, or manually, then use `blipshell benchmark review import RESPONSE.json`.

| Model | Unscored open-ended jobs |
|---|---|
| deepseek-v3.2:cloud | code_gen, lessons, reasoning, session_review, summarization |
| deepseek-v4-flash:cloud | code_gen, lessons, reasoning, session_review, summarization |
| deepseek/deepseek-v4-flash | code_gen, lessons, reasoning, session_review, summarization |
| deepseek/deepseek-v4.1-flash | code_gen, lessons, reasoning, session_review, summarization |
| devstral-small-2:24b | code_gen, lessons, reasoning, session_review, summarization |
| gemini-3-flash-preview:cloud | code_gen, lessons, reasoning, session_review, summarization |
| gemma3n:e4b | code_gen, lessons, reasoning, session_review, summarization |
| gemma4:31b-cloud | code_gen, lessons, reasoning, session_review, summarization |
| gemma4:e4b | code_gen, lessons, reasoning, session_review, summarization |
| glm-4.7-flash:latest | code_gen, lessons, reasoning, session_review, summarization |
| glm-4.7:cloud | code_gen, lessons, reasoning, session_review, summarization |
| glm-5.1:cloud | code_gen, lessons, reasoning, session_review, summarization |
| glm-5.2:cloud | code_gen, lessons, reasoning, session_review, summarization |
| glm-5:cloud | code_gen, lessons, reasoning, session_review, summarization |
| glm4:latest | code_gen, lessons, reasoning, session_review, summarization |
| gpt-oss:120b-cloud | code_gen, lessons, reasoning, session_review, summarization |
| gpt-oss:latest | code_gen, lessons, reasoning, session_review, summarization |
| kimi-k2-thinking:cloud | code_gen, lessons, reasoning, session_review, summarization |
| kimi-k2.5:cloud | code_gen, lessons, reasoning, session_review, summarization |
| kimi-k2.7-code:cloud | code_gen, lessons, reasoning, session_review, summarization |
| kimi-k3:cloud | code_gen, lessons, reasoning, session_review, summarization |
| lfm2.5 | code_gen, lessons, reasoning, session_review, summarization |
| lfm2.5:latest | code_gen, lessons, reasoning, session_review, summarization |
| minimax-m2.5:cloud | code_gen, lessons, reasoning, session_review, summarization |
| minimax-m2.7:cloud | code_gen, lessons, reasoning, session_review, summarization |
| minimax-m3:cloud | code_gen, lessons, reasoning, session_review, summarization |
| minimax/minimax-m3 | code_gen, lessons, reasoning, session_review, summarization |
| minimax/minimax-m3:free | code_gen, lessons, reasoning, session_review, summarization |
| ministral-3:14b | code_gen, lessons, reasoning, session_review, summarization |
| ministral-3:8b | code_gen, lessons, reasoning, session_review, summarization |
| moonshotai/kimi-k2.5 | code_gen, lessons, reasoning, session_review, summarization |
| nemotron-3-nano:30b-cloud | code_gen, lessons, reasoning, session_review, summarization |
| nemotron-3-super:cloud | code_gen, lessons, reasoning, session_review, summarization |
| nemotron-3-ultra:cloud | code_gen, lessons, reasoning, session_review, summarization |
| openai/gpt-5.6-luna | code_gen, lessons, reasoning, session_review, summarization |
| openai/gpt-6-luna | code_gen, lessons, reasoning, session_review, summarization |
| phi4:14b | code_gen, lessons, reasoning, session_review, summarization |
| qwen2.5-coder:14b | code_gen, lessons, reasoning, session_review, summarization |
| qwen2.5-coder:7b | code_gen, lessons, reasoning, session_review, summarization |
| qwen2.5:14b | code_gen, lessons, reasoning, session_review, summarization |
| qwen2.5:7b | code_gen, lessons, reasoning, session_review, summarization |
| qwen3-coder-next:cloud | code_gen, lessons, reasoning, session_review, summarization |
| qwen3-coder:30b | code_gen, lessons, reasoning, session_review, summarization |
| qwen3.5:4b | code_gen, lessons, reasoning, session_review, summarization |
| qwen3.5:9b | code_gen, lessons, reasoning, session_review, summarization |
| qwen3.6:27b | code_gen, lessons, reasoning, session_review, summarization |
| qwen3:14b | code_gen, lessons, reasoning, session_review, summarization |
| qwen3:8b | code_gen, lessons, reasoning, session_review, summarization |
| z-ai/glm-5.2 | code_gen, lessons, reasoning, session_review, summarization |

## Latency by suite — mean seconds/call (lower is faster)
| Suite | deepseek-v3.2:cloud | deepseek-v4-flash:cloud | deepseek/deepseek-v4-flash | deepseek/deepseek-v4.1-flash | devstral-small-2:24b | gemini-3-flash-preview:cloud | gemma3n:e4b | gemma4:31b-cloud | gemma4:e4b | glm-4.7-flash:latest | glm-4.7:cloud | glm-5.1:cloud | glm-5.2:cloud | glm-5:cloud | glm4:latest | gpt-oss:120b-cloud | gpt-oss:latest | kimi-k2-thinking:cloud | kimi-k2.5:cloud | kimi-k2.7-code:cloud | kimi-k3:cloud | lfm2.5 | lfm2.5:latest | minimax-m2.5:cloud | minimax-m2.7:cloud | minimax-m3:cloud | minimax/minimax-m3 | minimax/minimax-m3:free | ministral-3:14b | ministral-3:8b | moonshotai/kimi-k2.5 | nemotron-3-nano:30b-cloud | nemotron-3-super:cloud | nemotron-3-ultra:cloud | openai/gpt-5.6-luna | openai/gpt-6-luna | phi4:14b | qwen2.5-coder:14b | qwen2.5-coder:7b | qwen2.5:14b | qwen2.5:7b | qwen3-coder-next:cloud | qwen3-coder:30b | qwen3.5:4b | qwen3.5:9b | qwen3.6:27b | qwen3:14b | qwen3:8b | z-ai/glm-5.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pipeline | — | — | 0.85 | 3.62 | 3.86 | — | — | 0.46 | 0.38 | 0.97 | — | — | 1.44 | — | — | 0.93 | 7.21 | — | — | 0.93 | — | 3.76 | 3.26 | — | — | 3.35 | 2.64 | 0.86 | 1.17 | 0.41 | 18.87 | 0.78 | 1.38 | 0.68 | 1.62 | 1.88 | 0.66 | 1.10 | — | 1.17 | — | — | 0.99 | 0.39 | 0.57 | 1.89 | 1.25 | **0.32** | 1.84 |
| reasoning_suite | 0.16 | 4.28 | 3.36 | 18.01 | 36.87 | **0.16** | 3.20 | 1.02 | 8.11 | 65.92 | 0.16 | 12.54 | 4.40 | 0.16 | 4.80 | 2.07 | 14.60 | 0.17 | 0.16 | 12.19 | 3.49 | 6.25 | 4.79 | 0.16 | 10.21 | 8.90 | 9.82 | 4.39 | 11.01 | 4.47 | 46.71 | 6.17 | 9.62 | 1.97 | 4.34 | 3.91 | 4.42 | 8.09 | 2.58 | 11.76 | 2.57 | 0.16 | 6.39 | 31.76 | 35.46 | 25.08 | 86.86 | 21.22 | 6.62 |
| session_review | — | — | 1.69 | 7.56 | 61.17 | — | — | 1.00 | 4.26 | 12.03 | — | — | 4.45 | — | — | 2.99 | 23.56 | — | — | 4.09 | — | 8.82 | 9.59 | — | — | 15.68 | 16.00 | 0.80 | 17.79 | 7.92 | 66.56 | 2.47 | 4.67 | **0.00** | 5.14 | 4.40 | 7.12 | 12.63 | — | 11.13 | — | — | 15.44 | 4.43 | 6.51 | 0.00 | 17.46 | 4.87 | 3.98 |
| coding | — | — | — | — | 4.66 | — | — | 11.13 | — | — | — | — | 86.64 | — | — | — | **4.66** | — | — | 48.88 | — | 37.69 | — | — | — | 100.15 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| realdata_suite | — | — | — | — | — | — | — | **0.25** | — | — | — | — | 1.47 | — | — | — | — | — | — | 1.03 | — | 2.92 | — | — | — | 3.58 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

_Latency is measured per suite, so jobs in the same suite share a number (pipeline = ranking/importance/contradiction/entity/summarization/lessons; reasoning_suite = reasoning/coding-gen/tool_calling)._

## Output length (mean words, externally reviewed jobs) — longer is NOT better
| Job | deepseek-v3.2:cloud | deepseek-v4-flash:cloud | deepseek/deepseek-v4-flash | deepseek/deepseek-v4.1-flash | devstral-small-2:24b | gemini-3-flash-preview:cloud | gemma3n:e4b | gemma4:31b-cloud | gemma4:e4b | glm-4.7-flash:latest | glm-4.7:cloud | glm-5.1:cloud | glm-5.2:cloud | glm-5:cloud | glm4:latest | gpt-oss:120b-cloud | gpt-oss:latest | kimi-k2-thinking:cloud | kimi-k2.5:cloud | kimi-k2.7-code:cloud | kimi-k3:cloud | lfm2.5 | lfm2.5:latest | minimax-m2.5:cloud | minimax-m2.7:cloud | minimax-m3:cloud | minimax/minimax-m3 | minimax/minimax-m3:free | ministral-3:14b | ministral-3:8b | moonshotai/kimi-k2.5 | nemotron-3-nano:30b-cloud | nemotron-3-super:cloud | nemotron-3-ultra:cloud | openai/gpt-5.6-luna | openai/gpt-6-luna | phi4:14b | qwen2.5-coder:14b | qwen2.5-coder:7b | qwen2.5:14b | qwen2.5:7b | qwen3-coder-next:cloud | qwen3-coder:30b | qwen3.5:4b | qwen3.5:9b | qwen3.6:27b | qwen3:14b | qwen3:8b | z-ai/glm-5.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Summarization | — | — | 8 | 7 | 5 | — | — | 8 | 9 | 9 | — | — | — | — | — | 8 | 9 | — | — | — | — | — | 204 | — | — | — | 10 | 7 | 6 | 9 | 7 | 10 | 10 | 8 | 9 | 8 | 7 | 8 | — | 8 | — | — | 7 | 12 | 10 | 8 | 8 | 7 | 8 |
| Lessons | — | — | 28 | 34 | 26 | — | — | 14 | 26 | 24 | — | — | — | — | — | 35 | 38 | — | — | — | — | — | 507 | — | — | — | 32 | 8 | 51 | 53 | 30 | 38 | 28 | 20 | 18 | 15 | 46 | 21 | — | 30 | — | — | 37 | 27 | 33 | 15 | 44 | 31 | 26 |
| Reasoning | — | 209 | 199 | 245 | 159 | — | 149 | 207 | 252 | 161 | — | 167 | 170 | — | 142 | 318 | 231 | — | — | 133 | — | — | 199 | — | 204 | 249 | 268 | 219 | 217 | 239 | 203 | 333 | 311 | 243 | 207 | 144 | 194 | 169 | 154 | 181 | 123 | — | 160 | 146 | 166 | — | 132 | 124 | 197 |
| Code generation | — | 76 | 66 | 105 | 68 | — | 81 | 88 | 122 | 60 | — | 64 | 90 | — | 93 | 230 | 135 | — | — | 71 | — | — | 73 | — | 91 | 137 | 144 | 125 | 83 | 99 | 70 | 134 | 147 | — | 76 | 62 | 123 | 61 | 76 | 80 | 100 | — | 74 | 75 | 63 | — | 61 | 72 | 65 |
| Session review | — | — | 175 | 231 | 123 | — | — | 94 | 116 | 147 | — | — | — | — | — | 209 | 188 | — | — | 172 | — | — | 1046 | — | — | — | 244 | 217 | 179 | 235 | 198 | 144 | 180 | — | 196 | 111 | 166 | 115 | — | 100 | — | — | 220 | 190 | 192 | — | 178 | 170 | 165 |

_External-review rubrics do not reward length. If a model wins while writing far more, inspect the packet and rationale before acting._

## Provenance — when each column was measured
| Model | Newest run (UTC) | Data spans | Runs | Code (git) | Benchmark | Dataset | Scorer | Source | Scope |
|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3.2:cloud | 2026-08-13 03:32 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 0/15 jobs |
| deepseek-v4-flash:cloud | 2026-08-13 03:30 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 1/15 jobs |
| deepseek/deepseek-v4-flash | 2026-09-02 17:28 | 2026-09-02 | 2 | 86da755 (mixed) | legacy | legacy | legacy | legacy | 6/15 jobs |
| deepseek/deepseek-v4.1-flash | 2026-09-25 11:58 | 2026-09-25 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| devstral-small-2:24b | 2026-09-25 02:09 | 2026-08-11..2026-09-25 | 3 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 9/15 jobs |
| gemini-3-flash-preview:cloud | 2026-08-13 04:07 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 0/15 jobs |
| gemma3n:e4b | 2026-08-13 05:29 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 0/15 jobs |
| gemma4:31b-cloud | 2026-09-25 11:20 | 2026-08-18..2026-09-25 | 3 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 9/15 jobs |
| gemma4:e4b | 2026-09-25 04:22 | 2026-08-11..2026-09-25 | 3 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| glm-4.7-flash:latest | 2026-09-24 17:02 | 2026-09-24 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| glm-4.7:cloud | 2026-08-13 03:29 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 0/15 jobs |
| glm-5.1:cloud | 2026-08-13 03:21 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 1/15 jobs |
| glm-5.2:cloud | 2026-08-13 03:19 | 2026-06-23..2026-08-13 | 2 | 39f4fa6 | legacy | legacy | legacy | legacy | 5/15 jobs |
| glm-5:cloud | 2026-08-13 03:29 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 0/15 jobs |
| glm4:latest | 2026-08-13 06:02 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 1/15 jobs |
| gpt-oss:120b-cloud | 2026-09-25 11:36 | 2026-08-18..2026-09-25 | 2 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| gpt-oss:latest | 2026-09-24 07:11 | 2026-08-10..2026-09-24 | 3 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 9/15 jobs |
| kimi-k2-thinking:cloud | 2026-08-13 03:40 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 0/15 jobs |
| kimi-k2.5:cloud | 2026-08-13 03:40 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 0/15 jobs |
| kimi-k2.7-code:cloud | 2026-08-13 03:33 | 2026-06-24..2026-08-13 | 3 | 39f4fa6 | legacy | legacy | legacy | legacy | 5/15 jobs |
| kimi-k3:cloud | 2026-08-13 03:16 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 0/15 jobs |
| lfm2.5 | 2026-06-24 11:44 | 2026-06-24 | 1 | pre-migration | legacy | legacy | legacy | legacy | 4/15 jobs |
| lfm2.5:latest | 2026-09-25 08:33 | 2026-08-13..2026-09-25 | 2 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 7/15 jobs |
| minimax-m2.5:cloud | 2026-08-13 03:47 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 0/15 jobs |
| minimax-m2.7:cloud | 2026-08-13 03:40 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 1/15 jobs |
| minimax-m3:cloud | 2026-08-13 03:11 | 2026-06-23..2026-08-13 | 2 | pre-migration | legacy | legacy | legacy | legacy | 5/15 jobs |
| minimax/minimax-m3 | 2026-09-25 12:19 | 2026-09-25 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| minimax/minimax-m3:free | 2026-09-02 15:13 | 2026-09-02 | 1 | e5aa214 | legacy | legacy | legacy | legacy | 6/15 jobs |
| ministral-3:14b | 2026-09-24 06:31 | 2026-08-11..2026-09-24 | 3 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| ministral-3:8b | 2026-09-24 21:39 | 2026-09-24 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| moonshotai/kimi-k2.5 | 2026-09-25 12:54 | 2026-09-25 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| nemotron-3-nano:30b-cloud | 2026-09-25 11:41 | 2026-09-25 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| nemotron-3-super:cloud | 2026-09-25 11:24 | 2026-09-25 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| nemotron-3-ultra:cloud | 2026-08-18 12:35 | 2026-08-18 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 6/15 jobs |
| openai/gpt-5.6-luna | 2026-09-25 12:35 | 2026-09-25 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| openai/gpt-6-luna | 2026-09-25 11:49 | 2026-09-25 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| phi4:14b | 2026-09-24 22:50 | 2026-09-24 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 7/15 jobs |
| qwen2.5-coder:14b | 2026-09-24 20:31 | 2026-08-11..2026-09-24 | 3 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| qwen2.5-coder:7b | 2026-08-13 06:09 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 1/15 jobs |
| qwen2.5:14b | 2026-09-24 21:02 | 2026-08-13..2026-09-24 | 2 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| qwen2.5:7b | 2026-08-13 06:11 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 1/15 jobs |
| qwen3-coder-next:cloud | 2026-08-13 03:47 | 2026-08-13 | 1 | 39f4fa6 | legacy | legacy | legacy | legacy | 0/15 jobs |
| qwen3-coder:30b | 2026-09-24 20:02 | 2026-08-11..2026-09-24 | 3 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| qwen3.5:4b | 2026-09-25 06:44 | 2026-08-13..2026-09-25 | 2 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| qwen3.5:9b | 2026-09-25 04:47 | 2026-09-25 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| qwen3.6:27b | 2026-09-24 23:09 | 2026-09-24 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 7/15 jobs |
| qwen3:14b | 2026-09-24 03:00 | 2026-08-03..2026-09-24 | 7 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| qwen3:8b | 2026-09-24 21:56 | 2026-08-13..2026-09-24 | 2 | d1e355a (mixed) | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |
| z-ai/glm-5.2 | 2026-09-25 12:44 | 2026-09-25 | 1 | d1e355a | 2.0 | 2026-09-23 | 2 | b821993265074d47 | 8/15 jobs |

_Results persist across machines and months. Columns measured at different dates or different commits are NOT strictly comparable — a scorer or prompt change between them moves scores on its own. When two rows disagree and their commits differ, re-run the older one before concluding anything._

_A model showing fewer than all jobs did not measure every category — usually a `--jobs`-scoped run. Its blank cells mean 'not run', not 'scored zero', and its COMPOSITE averages over fewer jobs than a full run's, so do not compare composites across differing scopes._

## Methodology — what each job measures and how it's scored
| Job | Measures | Scoring |
|---|---|---|
| Rank+Importance (live pipeline) | The combined rank+importance+type call every message goes through (rank_importance_and_classify, processor.py:185). This is models.ranking_importance. | Mean of rank-correlation and importance calibration vs gold. |
| Ranking (import path) | Standalone rank prompt used only by the bulk import path (rank_memory, import_common.py:598) — NOT the live pipeline. | Rank-correlation of predicted vs gold rank (order matters); flat output scores ~0. |
| Importance (import path) | Standalone importance prompt used only by the bulk import path (ask_importance, import_common.py:648) — NOT the live pipeline. | Average of correlation and calibration (1-MAE) vs gold importance. |
| Contradiction | Decides whether two memories contradict. | Exact YES/NO accuracy over a balanced set. |
| Dedup verdict (live pipeline) | Given a new memory and up to three similar ones, picks ADD / NONE / UPDATE n / DELETE n as a one-line verdict (processor._decide_and_apply_action). Parsed by the strict grammar production runs; an unparseable reply counts as WRONG here because in production it costs a re-ask and then defaults to keeping the memory. | Accuracy: action AND target must match gold. valid_rate (informational) = share of replies the strict grammar accepted. |
| Dedup verdict (structured, experimental) | Same cases, asked as a schema-constrained JSON object (memory.dedup.structured_output). NOT the production default: this row exists to decide whether it should be. Displayed, not in the composite. | Accuracy over schema-VALID replies; valid_rate = share of replies that validated. Enable the toggle only when valid_rate clears ~0.98 on the model that serves models.reasoning. |
| Entity extraction | Extracts entities/relationships from text. | F1 of extracted entities vs the expected entity set per item. |
| Coding (agentic) | Real multi-step coding tasks in a sandbox. | Fraction of verification checks passed (executes code, runs pytest). |
| Coding (legacy - ambiguous) | Runs before 2026-08-03 reported code generation AND agentic coding under one task_type, so this column is whichever one happened to be written last. | NOT comparable to the two rows above, or between models. Re-run to replace it. |
| Tool calling | Picks the right tool with the right arguments. | Exact tool name + required-argument match. |
| Session review (chunked, multi-call) | The path a session too big for the context window takes: chunk-scoped reflection per part, then merge_chunk_reflections (processor.py:553-577). Single-chunk cases never reach it. | 1.0 when the merged reflection does NOT report work that a later chunk resolved as 'never addressed' — the invented finding that flows to lessons. |

## Caveats
- Latency is a per-suite mean, not per individual job; cloud latency includes network.
- Open-ended quality is absent until an external review is imported. Review files name the reviewer and remain independently replaceable; candidate inference is never rerun.
- Compare models from the same benchmark version, dataset version, Git commit, tier, context window, and repeat count. Mixed or legacy columns are historical context only.
- Agentic coding executes real tasks; a model with weak tool-calling will score low there even if its raw code quality is fine — that's intended (it reflects real executor use).
- Embedding needs a labeled retrieval set in the DB; it's blank if unavailable.
