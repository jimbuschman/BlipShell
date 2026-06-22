"""Unified model-benchmark harness.

Runs candidate LLMs through BlipShell's existing benchmark suites, normalizes
their results into one scoreboard DB, grades open-ended outputs with a neutral
cloud judge, and renders a switch-verdict vs the current production models.

This is a dev/eval tool invoked via `blipshell benchmark` — it sits on top of
the existing `tests/benchmark_*` suites rather than replacing them.
"""
