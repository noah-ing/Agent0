"""Generalized Reward Policy Optimization helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RewardBreakdown:
    uncertainty: float
    tool_usage: float
    repetition: float


class GRPOTrainer:
    """Computes composite rewards and (optionally) streams them to TRL."""

    def __init__(
        self,
        coeffs: Dict[str, float],
        backend: Optional[Any] = None,
        trl_runner: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._coeffs = coeffs
        self._backend = backend
        self._trl_runner = trl_runner
        self._logger = logger

    def compute_reward(self, breakdown: RewardBreakdown) -> float:
        reward = (
            self._coeffs.get("uncertainty", 0.0) * breakdown.uncertainty
            + self._coeffs.get("tool_usage", 0.0) * breakdown.tool_usage
            - self._coeffs.get("repetition", 0.0) * breakdown.repetition
        )
        return max(0.0, reward)

    def _dispatch_trl(self, experiences: List[Dict[str, Any]]) -> None:
        if not self._trl_runner:
            return
        prompts = [exp["prompt"] for exp in experiences]
        responses = [exp["response"] for exp in experiences]
        rewards = [exp["reward"] for exp in experiences]
        self._trl_runner.step(prompts, responses, rewards)

    def step(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        experiences: List[Dict[str, Any]] = []
        for item in batch:
            breakdown = RewardBreakdown(**item["reward_breakdown"])
            reward = self.compute_reward(breakdown)
            exp = {
                "prompt": item["prompt"],
                "response": item["response"],
                "reward": reward,
                "raw_breakdown": breakdown.__dict__,
            }
            experiences.append(exp)
            if self._logger:
                self._logger.log({"curriculum/reward": reward})
        self._dispatch_trl(experiences)
        if self._backend:
            self._backend.step(experiences)
        return experiences
