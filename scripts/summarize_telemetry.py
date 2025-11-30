#!/usr/bin/env python3
"""Summarize telemetry JSONL logs into report-ready stats."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.settings import load_env
from src.settings.config_loader import load_yaml_config
from src.reporting.telemetry import TelemetryStats, load_records, summarize

load_env(ROOT)

EXECUTOR_CONFIG_PATH = ROOT / "configs" / "executor.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute aggregates from telemetry JSONL")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("reports/telemetry_local.jsonl"),
        help="Path to telemetry JSONL file",
    )
    parser.add_argument(
        "--executor-config",
        type=Path,
        default=EXECUTOR_CONFIG_PATH,
        help="Executor config used to derive self-consistency bands",
    )
    return parser.parse_args()


def print_summary(stats: TelemetryStats, band_summary: Optional[str] = None, tool_summary: Optional[str] = None) -> None:
    print("=== Telemetry Summary ===")
    for key in sorted(stats.means):
        mean = stats.means[key]
        std = stats.stds.get(key)
        if std is not None:
            print(f"{key:30s}: {mean:.4f} ± {std:.4f} (n={stats.counts.get(key, 0)})")
        else:
            print(f"{key:30s}: {mean:.4f} (n={stats.counts.get(key, 0)})")
    if stats.percentiles:
        print("\n=== Percentiles ===")
        for key, payload in stats.percentiles.items():
            print(
                f"{key:30s}: mean={payload['mean']:.2f} p50={payload['p50']:.2f} p90={payload['p90']:.2f}"
            )
    if stats.derived:
        print("\n=== Derived Metrics ===")
        for key, value in stats.derived.items():
            print(f"{key:30s}: {value:.4f}")
    if band_summary:
        print("\n=== Self-Consistency Bands ===")
        print(band_summary)
    if tool_summary:
        print("\n=== Executor Tool Usage ===")
        print(tool_summary)


def _load_band_thresholds(path: Path) -> tuple[float, float]:
    if not path.exists():
        return (0.3, 0.8)
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


def _format_band_summary(values: Optional[List[float]], low: float, high: float) -> Optional[str]:
    if not values:
        return None
    total = len(values)
    low_ct = sum(1 for val in values if val < low)
    mid_ct = sum(1 for val in values if low <= val <= high)
    high_ct = total - low_ct - mid_ct

    def _pct(count: int) -> float:
        return (count / total) * 100 if total else 0.0

    return (
        f"low < {low:.2f}: {low_ct} ({_pct(low_ct):.1f}%) | "
        f"band [{low:.2f},{high:.2f}]: {mid_ct} ({_pct(mid_ct):.1f}%) | "
        f"high > {high:.2f}: {high_ct} ({_pct(high_ct):.1f}%)"
    )


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = pct / 100 * (len(values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    frac = rank - lower
    return values[lower] * (1 - frac) + values[upper] * frac


def _format_tool_summary(values: Optional[List[float]]) -> Optional[str]:
    if not values:
        return None
    sorted_vals = sorted(values)
    mean = sum(sorted_vals) / len(sorted_vals)
    p50 = _percentile(sorted_vals, 50)
    p90 = _percentile(sorted_vals, 90)
    return f"avg={mean:.2f}, p50={p50:.2f}, p90={p90:.2f}, n={len(sorted_vals)}"


def main() -> None:
    args = parse_args()
    records = list(load_records(args.log))
    if not records:
        raise SystemExit("No telemetry records found; run `make demo` first.")
    stats = summarize(records)
    band_low, band_high = _load_band_thresholds(args.executor_config)
    band_summary = _format_band_summary(stats.series.get("curriculum/p_hat"), band_low, band_high)
    tool_summary = _format_tool_summary(stats.series.get("executor/tool_calls_avg"))
    print_summary(stats, band_summary, tool_summary)


if __name__ == "__main__":
    main()
