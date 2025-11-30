"""Ambiguity-Dynamic Policy Optimization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AmbiguityStats:
    consistency: float
    advantage: float
    prompt: str
    response: str


class ADPOTrainer:
    """Rescales advantages by ambiguity and adjusts PPO clip bounds."""

    def __init__(
        self,
        lower_clip: float,
        base_upper_clip: float,
        scale: float,
        backend: Optional[Any] = None,
        trl_runner: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._lower = lower_clip
        self._base_upper = base_upper_clip
        self._scale = scale
        self._backend = backend
        self._trl_runner = trl_runner
        self._logger = logger

    def _dynamic_upper(self, consistency: float) -> float:
        # Consistency near 0.5 expands the clip range; confident samples stay tight.
        return self._base_upper + self._scale * abs(0.5 - consistency)

    def rescale(self, stats: AmbiguityStats) -> Dict[str, float]:
        upper = self._dynamic_upper(stats.consistency)
        weight = max(0.0, 1.0 - 2.0 * abs(stats.consistency - 0.5))
        return {
            "adv": stats.advantage * weight,
            "clip_low": self._lower,
            "clip_high": upper,
            "prompt": stats.prompt,
            "response": stats.response,
        }

    def _dispatch_trl(self, payload: List[Dict[str, float]]) -> None:
        if not self._trl_runner:
            return
        prompts = [item["prompt"] for item in payload]
        responses = [item["response"] for item in payload]
        rewards = [item["adv"] for item in payload]
        self._trl_runner.step(prompts, responses, rewards)

    def step(self, batch: List[AmbiguityStats]) -> List[Dict[str, float]]:
        payload = [self.rescale(item) for item in batch]
        if self._logger:
            for item in payload:
                self._logger.log({"executor/adv_scaled": item["adv"]})
        self._dispatch_trl(payload)
        if self._backend:
            self._backend.step(payload)
        return payload
