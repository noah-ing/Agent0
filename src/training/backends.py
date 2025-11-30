"""Backends for GRPO/ADPO trainers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class FlexRLConfig:
    project: str
    run_name: str
    optimizer: str = "adamw"


class FlexRLBackend:
    """Placeholder wrapper for DeepSpeed FlexRL trainer hooks."""

    def __init__(self, cfg: FlexRLConfig) -> None:
        self.cfg = cfg

    def step(self, batch: List[Dict[str, Any]]) -> None:
        # Real integration will call into DeepSpeed runtime; placeholder logs size.
        print(f"[flexrl:{self.cfg.run_name}] processed batch of {len(batch)}")
