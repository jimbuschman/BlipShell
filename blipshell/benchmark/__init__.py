"""Unified model-benchmark harness.

Runs candidate LLMs through BlipShell's existing benchmark suites, scores
objective checks locally, exports open-ended outputs for blinded offline
review, and renders versioned assignment evidence for production models.

This is a dev/eval tool invoked via `blipshell benchmark` — it sits on top of
the existing `tests/benchmark_*` suites rather than replacing them.
"""
