# Archived scripts

One-off scripts that have already done their job. They're kept because they
document what was done to the corpus — several describe a state the database
was in ("81% of summaries are third-person", "all 33K memories are
'conversation'") that no longer exists and can't be reconstructed from the
schema.

**None of these are wired into anything.** `nightly.py` and `cli.py` import
from `scripts/`, but never from here. Nothing in `scripts/archive/` should
acquire a caller — if one of these becomes useful again, move it back out and
bring it up to date first. Most reference APIs that have since changed
(ChromaDB, pre-sqlite-vec vector handling, the old scoring prompts).

| Script | Was for |
|---|---|
| `migrate_from_echo.py` | One-time migration from EchoFrontendV2 |
| `reprocess_scores_and_lessons.py` | Fixing two defects from the initial batch import |
| `reprocess_summaries.py` | The batch where 81% of summaries were third-person |
| `reclassify_memory_types.py` | When all 33K memories were still typed 'conversation' |
| `fix_entity_types.py` | Curated name→type corrections for one misclassification event |
| `verify_import.py` | ChatGPT-export completeness; `tests/test_import_chatgpt.py` covers it now |
| `extract_entities.py` | Draining the extraction backlog; nightly + the memory worker do this continuously |
| `capture_ollama_429.py` | Capturing 429 shapes "so we can write a correct classifier" — the classifier exists |
| `diagnose_fk_violations.py` | One specific FK incident on the Ollama PC (2026-05) |
| `fix_skipped_reflections.py` | One-shot repair of a bad-skip batch |
| `rebuild_chroma.py` | Superseded by `rebuild_vectors.py`; the name outlived ChromaDB itself |
| `diagnose_pipeline.py` | A specific pipeline incident |
| `probe_scoring_prompts.py` | A/B prompt bake-off against qwen2.5:14b; superseded by `blipshell/benchmark/` |
| `live_integration.py` | Live-Ollama integration probe; nothing imported it |

The last two were renamed on archiving: they began with `test_`, which matches
pytest's default discovery. They were safe only because `testpaths = ["tests"]`
confines collection — a bare `pytest .` from the repo root would have collected
them, and `live_integration.py` hits a real Ollama.
