#!/usr/bin/env python3
"""Render an iteration report by fusing telemetry stats + metadata."""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.telemetry import load_records, summarize
from src.settings import load_env
from src.settings.config_loader import load_yaml_config

load_env(ROOT)

DEFAULT_TEMPLATE = ROOT / "reports" / "templates" / "iteration_report.md"
DEFAULT_OUTPUT = ROOT / "reports" / "iter_000.md"
EXECUTOR_CONFIG_PATH = ROOT / "configs" / "executor.yaml"


def _default_git_sha() -> str:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    return sha or "unknown"


def _format_float(value: float | int | None, precision: int = 4, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return f"{numeric:.{precision}f}"


def _format_int(value: float | int | None, default: str = "0") -> str:
    if value is None:
        return default
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return str(numeric)


def _load_filter_band(path: Path) -> tuple[float, float]:
    if not path.exists():
        return (0.0, 1.0)
    cfg = load_yaml_config(path)
    filtering = cfg.get("filtering", {})
    band = filtering.get("self_consistency_band", {})

    def _coerce(value: object, default: float) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    low = _coerce(band.get("low"), 0.3)
    high = _coerce(band.get("high"), 0.8)
    if low > high:
        low, high = high, low
    return (low, high)


def _build_rejection_summary(stats) -> str:
    repetition = _format_int(stats.totals.get("frontier/rejected_repetition"))
    band = _format_int(stats.counts.get("frontier/rejected_consistency"))
    return f"repetition={repetition}, out-of-band={band}"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = pct / 100 * (len(values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    frac = rank - lower
    return values[lower] * (1 - frac) + values[upper] * frac


def _format_consistency_bands(stats, low: float, high: float) -> str:
    values = stats.series.get("curriculum/p_hat") if hasattr(stats, "series") else None
    if not values:
        return "N/A"
    total = len(values)
    low_ct = sum(1 for v in values if v < low)
    mid_ct = sum(1 for v in values if low <= v <= high)
    high_ct = total - low_ct - mid_ct

    def _pct(count: int) -> float:
        return (count / total) * 100 if total else 0.0

    return (
        f"low < {low:.2f}: {low_ct}/{total} ({_pct(low_ct):.1f}%) | "
        f"band [{low:.2f},{high:.2f}]: {mid_ct}/{total} ({_pct(mid_ct):.1f}%) | "
        f"high > {high:.2f}: {high_ct}/{total} ({_pct(high_ct):.1f}%)"
    )


def _format_tool_usage(stats) -> str:
    values = stats.series.get("executor/tool_calls_avg") if hasattr(stats, "series") else None
    if not values:
        return "N/A"
    sorted_vals = sorted(values)
    mean = sum(sorted_vals) / len(sorted_vals)
    p50 = _percentile(sorted_vals, 50)
    p90 = _percentile(sorted_vals, 90)
    return f"avg={mean:.2f}, p50={p50:.2f}, p90={p90:.2f} (n={len(sorted_vals)})"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill an Agent0 iteration report from telemetry logs")
    parser.add_argument("--telemetry", type=Path, required=True, help="Path to telemetry JSONL file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination markdown file")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="Report template path")
    parser.add_argument("--run-name", default=os.getenv("AGENT0_RUN_NAME", "iter_000"))
    parser.add_argument("--date", default=_dt.date.today().isoformat())
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--curriculum-batch", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", default=None, help="Notable curriculum seeds (ordered)")
    parser.add_argument("--trace-file", default="", help="Relative path to exemplar rollout JSON")
    parser.add_argument("--trl-status", default="not configured")
    parser.add_argument("--eval-suite", default="math-lite")
    parser.add_argument("--gsm8k-acc", type=float, default=None)
    parser.add_argument("--math-acc", type=float, default=None)
    parser.add_argument("--bbh-acc", type=float, default=None)
    parser.add_argument("--wins", default="TBD")
    parser.add_argument("--issues", default="TBD")
    parser.add_argument("--next-step", default="TBD")
    parser.add_argument("--executor-config", type=Path, default=EXECUTOR_CONFIG_PATH)
    return parser.parse_args()


def _load_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template {path} not found")
    return path.read_text(encoding="utf-8")


def _build_placeholder_map(args: argparse.Namespace, stats) -> Dict[str, str]:
    git_sha = args.git_sha or _default_git_sha()
    datasets = ", ".join(args.datasets) if args.datasets else "N/A"
    tasks = args.tasks or []
    task_1 = tasks[0] if len(tasks) > 0 else "(add example)"
    task_2 = tasks[1] if len(tasks) > 1 else "(add example)"
    frontier_total = stats.counts.get("frontier/consistency") or 0
    frontier_accepted = stats.totals.get("frontier/accepted") or 0
    grpo_mean = stats.means.get("curriculum/reward")
    grpo_std = stats.stds.get("curriculum/reward")
    adpo_mean = stats.means.get("executor/adv_scaled")
    adpo_std = stats.stds.get("executor/adv_scaled")
    judge_pass = stats.derived.get("judge/pass_rate")
    mean_turns = stats.means.get("rollout/turns")
    tool_percentiles = stats.percentiles.get("rollout/tool_events", {})
    band_low_val, band_high_val = _load_filter_band(args.executor_config)
    band_low = _format_float(band_low_val, precision=2)
    band_high = _format_float(band_high_val, precision=2)
    consistency_bands = _format_consistency_bands(stats, band_low_val, band_high_val)
    tool_usage = _format_tool_usage(stats)
    replacements: Dict[str, str] = {
        "DATE": args.date,
        "RUN_NAME": args.run_name,
        "TELEMETRY_PATH": str(args.telemetry),
        "GIT_SHA": git_sha,
        "DATASETS": datasets,
        "CURRICULUM_BATCH": _format_int(args.curriculum_batch or stats.counts.get("curriculum/reward")),
        "MEAN_REWARD": _format_float(grpo_mean),
        "TASK_1": task_1,
        "TASK_2": task_2,
        "FRONTIER_ACCEPTED": _format_int(frontier_accepted),
        "FRONTIER_TOTAL": _format_int(frontier_total),
        "FILTER_LOW": band_low,
        "FILTER_HIGH": band_high,
        "CONSISTENCY_BANDS": consistency_bands,
        "TOOL_USAGE": tool_usage,
        "JUDGE_PASS_RATE": _format_float(judge_pass),
        "REJECTIONS": _build_rejection_summary(stats),
        "MEAN_TURNS": _format_float(mean_turns, precision=2),
        "P50_TOOLS": _format_float(tool_percentiles.get("p50"), precision=2),
        "P90_TOOLS": _format_float(tool_percentiles.get("p90"), precision=2),
        "TRACE_FILE": args.trace_file or "(attach rollout file)",
        "GRPO_MEAN": _format_float(grpo_mean),
        "GRPO_STD": _format_float(grpo_std),
        "ADPO_MEAN": _format_float(adpo_mean),
        "ADPO_STD": _format_float(adpo_std),
        "TRL_STATUS": args.trl_status,
        "EVAL_SUITE": args.eval_suite,
        "GSM8K_ACC": _format_float(args.gsm8k_acc, precision=2),
        "MATH_ACC": _format_float(args.math_acc, precision=2),
        "BBH_ACC": _format_float(args.bbh_acc, precision=2),
        "WINS": args.wins,
        "ISSUES": args.issues,
        "NEXT_STEP": args.next_step,
    }
    return replacements


def _render_report(template: str, replacements: Dict[str, str]) -> str:
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(f"{{{{{token}}}}}", value)
    return rendered


def main() -> None:
    args = _parse_args()
    records = list(load_records(args.telemetry))
    if not records:
        raise SystemExit(f"No telemetry records found at {args.telemetry}")
    stats = summarize(records)
    template = _load_template(args.template)
    replacements = _build_placeholder_map(args, stats)
    content = _render_report(template, replacements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"[report] wrote {args.output}")


if __name__ == "__main__":
    main()
