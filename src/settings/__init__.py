"""Settings utilities exposed at the package level."""
from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None  # type: ignore


def load_env(root: Path | None = None) -> None:
    """Load a repo-scoped .env file when python-dotenv is available."""
    if load_dotenv is None:
        return
    root = root or Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


__all__ = ["load_env"]
