from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Literal


@dataclass
class StepResult:
    step: str
    output: str


@dataclass
class ReviewResult:
    approved: bool
    feedback: str


@dataclass
class WorkflowState:
    goal: str
    plan: list[str] = field(default_factory=list)
    step_index: int = 0
    results: Annotated[list[StepResult], add] = field(default_factory=list)
    status: Literal[
        "pending",
        "running",
        "retrying",
        "completed",
        "failed",
    ] = "pending"
    error: str | None = None
    review: ReviewResult | None = None

    # Number of retries for the current step.
    step_retry_count: int = 0
