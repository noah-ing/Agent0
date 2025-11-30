"""Lightweight YAML config loader helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
    raise RuntimeError("PyYAML is required to load Agent0 configs. Run `pip install pyyaml`." ) from exc


def _read_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found")
    return path.read_text(encoding="utf-8")


def load_yaml_config(path: Path | str) -> Dict[str, Any]:
    """Load YAML into a dictionary with helpful error context."""
    path = Path(path)
    try:
        payload = yaml.safe_load(_read_file(path))
    except yaml.YAMLError as exc:  # pragma: no cover - depends on file contents
        raise ValueError(f"Failed to parse YAML config {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} must define a mapping, got {type(payload)}")
    return payload


def dump_config(payload: Dict[str, Any]) -> str:
    """Return a deterministic string preview for logging/debugging."""
    return json.dumps(payload, indent=2, sort_keys=True)
