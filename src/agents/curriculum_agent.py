"""Curriculum agent generates frontier tasks per Agent0 spec."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .base_agent import AgentConfig, BaseAgent


@dataclass
class CurriculumSample:
    prompt: str
    raw_output: str


class CurriculumAgent(BaseAgent):
    """Adds reward-aware formatting checks atop the base agent."""

    def __init__(self, config: AgentConfig, client_factory: Any) -> None:
        client = client_factory(config)
        super().__init__(config, client)

    def generate_batch(self, batch_size: int) -> List[CurriculumSample]:
        samples: List[CurriculumSample] = []
        for _ in range(batch_size):
            # Placeholder prompt; real version will synthesize frontier-aware seeds.
            seed = "Create a novel math reasoning task that benefits from code execution."
            raw = self.invoke(seed)
            samples.append(CurriculumSample(prompt=seed, raw_output=raw))
        return samples
