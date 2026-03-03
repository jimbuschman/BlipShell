"""Unified benchmark system for BlipShell LLM model comparison.

Usage:
    blipshell benchmark                                  # all suites, default models
    blipshell benchmark --models qwen3.5:9b,qwen3:14b    # specific models
    blipshell benchmark --suite reflection               # one suite only
    blipshell benchmark --list                           # list available suites
"""

from __future__ import annotations

from blipshell.benchmark.models import BenchmarkResult, SuiteResult, TaskScore
from blipshell.benchmark.runner import run_benchmark
from blipshell.benchmark.suites import SUITES, list_suites

__all__ = [
    "BenchmarkResult",
    "SuiteResult",
    "TaskScore",
    "run_benchmark",
    "SUITES",
    "list_suites",
]
