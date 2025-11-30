"""Telemetry parsing and summary helpers."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

ROLLUP_KEYS = (
    "curriculum/",
    "executor/",
    "frontier/",
    "judge/",
    "rollout/",
    "loop/",
)
PERCENTILE_KEYS = {"rollout/tool_events", "rollout/turns"}
SERIES_KEYS = {"curriculum/p_hat", "executor/tool_calls_avg"}


@dataclass
class TelemetryStats:
    means: Dict[str, float]
    stds: Dict[str, float]
    derived: Dict[str, float]
    percentiles: Dict[str, Dict[str, float]]
    counts: Dict[str, int]
    totals: Dict[str, float]
    series: Dict[str, List[float]]


def load_records(path: Path) -> Iterable[Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Telemetry log {path} not found")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def summarize(records: Iterable[Dict[str, float]]) -> TelemetryStats:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    sq_sums: Dict[str, float] = defaultdict(float)
    judge_events = 0
    judge_pass = 0
    buckets: Dict[str, List[float]] = defaultdict(list)
    series_data: Dict[str, List[float]] = {key: [] for key in SERIES_KEYS}

    for record in records:
        for key, value in record.items():
            if key == "judge/is_valid":
                judge_events += 1
                try:
                    judge_pass += float(value)
                except (TypeError, ValueError):
                    pass
            if not isinstance(value, (int, float)):
                continue
            if key.startswith(ROLLUP_KEYS):
                value_f = float(value)
                sums[key] += value_f
                counts[key] += 1
                sq_sums[key] += value_f * value_f
                if key in SERIES_KEYS:
                    series_data[key].append(value_f)
            if key in PERCENTILE_KEYS:
                buckets[key].append(float(value))

    means = {key: sums[key] / counts[key] for key in counts}
    stds: Dict[str, float] = {}
    for key, total in sums.items():
        n = counts.get(key, 0)
        if n <= 1:
            continue
        mean = means[key]
        variance = max(0.0, (sq_sums[key] / n) - mean**2)
        stds[key] = variance**0.5
    derived: Dict[str, float] = {}
    if judge_events:
        derived["judge/pass_rate"] = judge_pass / judge_events

    percentile_stats: Dict[str, Dict[str, float]] = {}
    for key, values in buckets.items():
        if not values:
            continue
        sorted_vals = sorted(values)
        percentile_stats[key] = {
            "p50": _percentile(sorted_vals, 50),
            "p90": _percentile(sorted_vals, 90),
            "mean": sum(sorted_vals) / len(sorted_vals),
        }

    pruned_series = {key: values for key, values in series_data.items() if values}

    return TelemetryStats(
        means=means,
        stds=stds,
        derived=derived,
        percentiles=percentile_stats,
        counts=dict(counts),
        totals=dict(sums),
        series=pruned_series,
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