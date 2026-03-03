"""Data classes for the unified benchmark system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskScore:
    """One score for one model on one task within a suite."""
    task_name: str          # e.g. "ranking_importance", "tag_f1"
    quality: float          # 0.0-1.0 normalized quality (for comparison table)
    speed_s: float          # average seconds per call
    detail: dict = field(default_factory=dict)   # suite-specific raw metrics
    samples: int = 0
    errors: int = 0


@dataclass
class SuiteResult:
    """Results from one suite for one model."""
    suite_name: str
    model: str
    scores: list[TaskScore] = field(default_factory=list)
    elapsed_s: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Full benchmark run results."""
    suite_results: list[SuiteResult] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    suites: list[str] = field(default_factory=list)
    timestamp: str = ""
    elapsed_s: float = 0.0
