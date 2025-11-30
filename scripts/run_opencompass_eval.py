#!/usr/bin/env python3
"""Orchestrate OpenCompass benchmarks for Agent0 checkpoints."""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "configs" / "opencompass"
DEFAULT_WORK_DIR = REPO_ROOT / "outputs" / "opencompass"
MODEL_CONFIG = "agent0_vllm"

from src.settings import load_env

load_env(REPO_ROOT)

DATASET_SUITES = {
    "math-lite": [
        "gsm8k_gen_1d7fe4",
        "math_0shot_gen_393424",
    ],
    "math-heavy": [
        "gsm8k_gen_1d7fe4",
        "math_0shot_gen_393424",
        "bbh_gen_2879b0",
        "gpqa_gen_4baadb",
    ],
}


def _ensure_opencompass() -> None:
    try:
        importlib.import_module("opencompass")
    except ModuleNotFoundError as exc:  # noqa: PERF203
        raise SystemExit(
            "OpenCompass is not installed. Run `pip install opencompass==0.5.1` inside your env."
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenCompass evaluations against Agent0 endpoints")
    parser.add_argument(
        "--suite",
        choices=sorted(DATASET_SUITES.keys()),
        default="math-lite",
        help="Named dataset bundle defined in DATASET_SUITES",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Override suite with explicit dataset config names (e.g. gsm8k/gsm8k_gen_1d7fe4)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Directory for OpenCompass outputs",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "infer", "eval", "viz"],
        default="all",
        help="Pass-through for OpenCompass --mode flag",
    )
    parser.add_argument("--debug", action="store_true", help="Run OpenCompass in debug (single-process) mode")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching tasks")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Max concurrent workers forwarded to OpenCompass",
    )
    parser.add_argument(
        "--reuse",
        default=None,
        help="Reuse an existing OpenCompass work_dir timestamp (for resume)",
    )
    return parser.parse_args()


def _resolve_datasets(args: argparse.Namespace) -> List[str]:
    if args.datasets:
        return args.datasets
    return DATASET_SUITES[args.suite]


def main() -> None:
    _ensure_opencompass()
    args = _parse_args()
    datasets = _resolve_datasets(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "opencompass.cli.main",
        "--config-dir",
        str(CONFIG_DIR),
        "--models",
        MODEL_CONFIG,
        "--datasets",
        *datasets,
        "--work-dir",
        str(args.work_dir.resolve()),
        "--mode",
        args.mode,
        "--max-num-workers",
        str(args.max_workers),
    ]
    if args.debug:
        cmd.append("--debug")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.reuse:
        cmd.extend(["--reuse", args.reuse])

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", env.get("AGENT0_EVAL_API_KEY", "EMPTY"))

    print("[opencompass] launching:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
