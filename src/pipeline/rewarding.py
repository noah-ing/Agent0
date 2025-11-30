"""Heuristics that translate curriculum samples into reward breakdowns."""
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..utils.telemetry import TelemetryLogger

QUESTION_PATTERNS = [r"\?", r"Let's think", r"explain why", r"prove", r"derive"]
TOOL_PATTERNS = [r"```python", r"code", r"calculator", r"write a program"]
UNCERTAINTY_TERMS = ["complex", "challenging", "multi-step", "hard", "open"]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _frequency_ratio(text: str) -> float:
    tokens = [tok for tok in re.split(r"\s+", text.strip()) if tok]
    if not tokens:
        return 1.0
    unique = len(set(tokens))
    return unique / len(tokens)


def _count_matches(patterns, text: str) -> int:
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, flags=re.IGNORECASE))
    return total


class CurriculumRewardModel:
    """Combines judge verdicts, executor feedback, and text heuristics into reward triples."""

    def __init__(self, judge: any, telemetry: TelemetryLogger | None = None, tool_cap: float = 4.0) -> None:
        self._judge = judge
        self._logger = telemetry
        self._tool_cap = tool_cap

    def score(self, sample: str, stats: Dict[str, float] | None = None) -> Dict[str, float]:
        verdict = self._judge.verify(sample)
        tokens = len(sample.split())
        question_hits = _count_matches(QUESTION_PATTERNS, sample)
        tool_hits = _count_matches(TOOL_PATTERNS, sample)
        novelty_ratio = _frequency_ratio(sample)
        uncertainty_boost = sum(term in sample.lower() for term in UNCERTAINTY_TERMS)

        if stats and "p_hat" in stats:
            p_hat = float(stats["p_hat"])
            uncertainty = _clip(1.0 - 2.0 * abs(p_hat - 0.5))
        else:
            complexity = _clip(tokens / 400)
            uncertainty = _clip(0.25 + 0.4 * complexity + 0.05 * question_hits + 0.05 * uncertainty_boost)
        if verdict.is_valid:
            uncertainty = _clip(uncertainty - 0.05)
        else:
            uncertainty = _clip(uncertainty + 0.05)

        if stats and "avg_tool_calls" in stats:
            avg_tools = max(0.0, float(stats["avg_tool_calls"]))
            tool_usage = _clip(avg_tools / max(self._tool_cap, 1e-3))
        else:
            tool_usage = _clip(0.1 + 0.2 * math.log1p(tool_hits) + (0.15 if "```python" in sample else 0.0))
        repetition = _clip(1.0 - novelty_ratio)

        rewards = {
            "uncertainty": uncertainty,
            "tool_usage": tool_usage,
            "repetition": repetition,
        }
        if self._logger:
            payload = {
                "curriculum/uncertainty": uncertainty,
                "curriculum/tool_usage": tool_usage,
                "curriculum/repetition": repetition,
                "curriculum/judge_pass": float(verdict.is_valid),
            }
            if stats and "p_hat" in stats:
                payload["curriculum/p_hat"] = float(stats["p_hat"])
            if stats and "avg_tool_calls" in stats:
                payload["executor/tool_calls_avg"] = float(stats["avg_tool_calls"])
            self._logger.log(payload)
            if verdict.feedback:
                self._logger.log_text("curriculum/judge_feedback", verdict.feedback)
        return rewards