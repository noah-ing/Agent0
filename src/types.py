"""Shared dataclasses for Agent0 pipeline orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class ExecutorFeedback:
    """Aggregated statistics from sampling the executor agent."""

    p_hat: float
    majority_answer: str
    tool_counts: List[int]
    traces: List[Any]
    answers: List[str]

    @property
    def avg_tool_calls(self) -> float:
        if not self.tool_counts:
            return 0.0
        return sum(self.tool_counts) / len(self.tool_counts)


@dataclass
class EvaluatedSample:
    """Curriculum sample paired with executor feedback."""

    sample: Any
    feedback: ExecutorFeedback