"""Data models for task planning, execution, and background tasks."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PlanStatus(str, Enum):
    """Status of a task plan."""
    PLANNING = "planning"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Status of an individual step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskStep(BaseModel):
    """A single step in a task plan."""
    id: Optional[int] = None
    plan_id: Optional[int] = None
    step_number: int
    description: str
    status: StepStatus = StepStatus.PENDING
    tool_hint: Optional[str] = None
    output_result: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TaskPlan(BaseModel):
    """A multi-step task plan."""
    id: Optional[int] = None
    session_id: Optional[int] = None
    user_request: str
    status: PlanStatus = PlanStatus.PLANNING
    steps: list[TaskStep] = Field(default_factory=list)
    result_summary: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BackgroundTaskStatus(str, Enum):
    """Status of a background task."""
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundTaskType(str, Enum):
    """Types of background tasks."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    CODE_REVIEW = "code_review"
    CUSTOM = "custom"


class BackgroundTask(BaseModel):
    """A background/async task."""
    id: Optional[int] = None
    session_id: Optional[int] = None
    plan_id: Optional[int] = None
    title: str
    task_type: str = BackgroundTaskType.CUSTOM
    prompt: str = ""
    status: BackgroundTaskStatus = BackgroundTaskStatus.PENDING
    priority: int = 0
    progress_pct: float = 0.0
    progress_message: str = ""
    result: Optional[str] = None
    error_message: Optional[str] = None
    target_endpoint: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
