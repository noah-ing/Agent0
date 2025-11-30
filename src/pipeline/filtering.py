"""Frontier filtering utilities (self-consistency + repetition guards)."""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..types import EvaluatedSample
    from ..utils.telemetry import TelemetryLogger


@dataclass
class FilterConfig:
    low: float
    high: float
    repetition_threshold: float
    max_history: int = 256


class FrontierFilter:
    """Selects tasks with desirable uncertainty and novelty."""

    def __init__(self, cfg: FilterConfig, judge: any, logger: "TelemetryLogger" | None = None) -> None:
        self.cfg = cfg
        self._judge = judge
        self._history: List[str] = []
        self._logger = logger

    def _consistency(self, feedback) -> float:
        return max(0.0, min(1.0, feedback.p_hat))

    def _is_repetitive(self, text: str) -> bool:
        for prev in self._history:
            ratio = difflib.SequenceMatcher(None, text, prev).ratio()
            if ratio >= self.cfg.repetition_threshold:
                return True
        return False

    def _remember(self, text: str) -> None:
        self._history.append(text)
        if len(self._history) > self.cfg.max_history:
            self._history = self._history[-self.cfg.max_history :]

    def _log(self, payload):
        if self._logger and payload:
            self._logger.log(payload)

    def build_frontier(self, evaluated: List["EvaluatedSample"]) -> List["EvaluatedSample"]:
        frontier: List["EvaluatedSample"] = []
        for record in evaluated:
            sample = record.sample
            feedback = record.feedback
            if self._is_repetitive(sample.raw_output):
                self._log({"frontier/rejected_repetition": 1})
                continue
            consistency = self._consistency(feedback)
            self._log({"frontier/consistency": consistency})
            if not (self.cfg.low <= consistency <= self.cfg.high):
                self._log({"frontier/rejected_consistency": consistency})
                continue
            verdict = self._judge.verify(sample.raw_output)
            self._log({"judge/is_valid": float(verdict.is_valid)})
            if self._logger and verdict.feedback:
                self._logger.log_text("judge/feedback", verdict.feedback)
            if verdict.is_valid:
                frontier.append(record)
                self._log({"frontier/accepted": 1})
                self._remember(sample.raw_output)
        return frontier
