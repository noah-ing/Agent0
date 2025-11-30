"""Minimal telemetry helpers (local JSONL + optional Weights & Biases)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import json
import os
import time


@dataclass
class TelemetryConfig:
    project: str
    run_name: str
    log_path: Path
    enable_wandb: bool
    tags: Optional[str] = None


class TelemetryLogger:
    """Streams structured metrics to disk and (optionally) W&B."""

    def __init__(self, cfg: TelemetryConfig) -> None:
        self.cfg = cfg
        self._log_path = cfg.log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._wandb_run = None
        if cfg.enable_wandb:
            self._wandb_run = self._init_wandb()

    @classmethod
    def from_env(cls, run_name: str, default_project: str = "agent0-demo") -> "TelemetryLogger":
        enable_wandb = os.getenv("AGENT0_USE_WANDB", "0") == "1"
        project = os.getenv("AGENT0_WANDB_PROJECT", default_project)
        log_path = Path(os.getenv("AGENT0_TELEMETRY_PATH", "reports/telemetry.jsonl"))
        tags = os.getenv("AGENT0_WANDB_TAGS")
        cfg = TelemetryConfig(project=project, run_name=run_name, log_path=log_path, enable_wandb=enable_wandb, tags=tags)
        return cls(cfg)

    def _init_wandb(self):
        try:
            import wandb  # type: ignore
        except ModuleNotFoundError:
            print("[telemetry] wandb not installed; falling back to local logging only")
            return None
        try:
            run = wandb.init(
                project=self.cfg.project,
                name=self.cfg.run_name,
                config={"run_name": self.cfg.run_name},
                tags=(self.cfg.tags.split(",") if self.cfg.tags else None),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[telemetry] failed to init wandb: {exc}")
            return None
        return run

    def log(self, metrics: Dict[str, float | int | str], step: Optional[int] = None) -> None:
        if not metrics:
            return
        record = {"ts": time.time(), **metrics}
        if step is not None:
            record["step"] = step
        with self._log_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record) + "\n")
        if self._wandb_run:
            kwargs = {"commit": True}
            if step is not None:
                kwargs["step"] = step
            self._wandb_run.log(metrics, **kwargs)

    def log_rollout(self, payload: Dict[str, int | float | str]) -> None:
        flattened = {f"rollout/{k}": v for k, v in payload.items()}
        self.log(flattened)

    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        self.log({key: text}, step)

    def close(self) -> None:
        if self._wandb_run:
            try:
                self._wandb_run.finish()
            except Exception as exc:  # noqa: BLE001
                print(f"[telemetry] failed to close wandb: {exc}")
            finally:
                self._wandb_run = None