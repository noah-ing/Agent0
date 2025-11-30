"""Workspace-level settings helpers."""
from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None


def load_env(root: Path | None = None) -> None:
    """Load .env from repo root when python-dotenv is available."""
    if load_dotenv is None:
        return
    root = root or Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)