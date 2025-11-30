#!/usr/bin/env python3
"""Simple Rich-based dashboard for monitoring Agent0 telemetry in real time."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Optional

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.telemetry import TelemetryStats, load_records, summarize
from src.settings import load_env

load_env(ROOT)

DEFAULT_LOG = ROOT / "reports" / "telemetry.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live dashboard for Agent0 telemetry logs")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Telemetry JSONL to monitor")
    parser.add_argument("--refresh", type=float, default=5.0, help="Refresh interval (seconds)")
    parser.add_argument(
        "--curriculum-metrics",
        nargs="*",
        default=["curriculum/reward", "curriculum/uncertainty", "curriculum/tool_usage"],
        help="Curriculum metrics to display",
    )
    parser.add_argument(
        "--executor-metrics",
        nargs="*",
        default=["executor/adv_scaled", "rollout/turns"],
        help="Executor metrics to display",
    )
    return parser.parse_args()


def _format_mean_std(stats: TelemetryStats, key: str) -> tuple[str, str]:
    mean = stats.means.get(key)
    std = stats.stds.get(key)
    cnt = stats.counts.get(key, 0)
    mean_str = f"{mean:.4f}" if mean is not None else "-"
    if std is not None and cnt > 1:
        std_str = f"±{std:.4f}"
    else:
        std_str = ""
    return mean_str, std_str


def _build_metric_table(title: str, keys: Iterable[str], stats: TelemetryStats) -> Table:
    table = Table(title=title, box=box.SIMPLE)
    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Count", justify="right")
    for key in keys:
        mean, std = _format_mean_std(stats, key)
        count = stats.counts.get(key, 0)
        table.add_row(key, mean, std, str(count))
    return table


def _build_percentile_table(stats: TelemetryStats) -> Optional[Table]:
    if not stats.percentiles:
        return None
    table = Table(title="Tool Events", box=box.SIMPLE)
    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p90", justify="right")
    for key, payload in stats.percentiles.items():
        table.add_row(
            key,
            f"{payload['mean']:.2f}",
            f"{payload['p50']:.2f}",
            f"{payload['p90']:.2f}",
        )
    return table


def _build_judge_panel(stats: TelemetryStats) -> Panel:
    rate = stats.derived.get("judge/pass_rate")
    text = Text()
    if rate is None:
        text.append("Judge pass rate: -\n", style="bold yellow")
    else:
        pct = rate * 100
        color = "green" if pct >= 60 else "yellow" if pct >= 40 else "red"
        text.append(f"Judge pass rate: {pct:.1f}%\n", style=f"bold {color}")
    accepted = int(stats.totals.get("frontier/accepted", 0))
    total = int(stats.counts.get("frontier/consistency", 0))
    text.append(f"Frontier accepted: {accepted} / {max(total, 1)}\n")
    rejected_rep = int(stats.totals.get("frontier/rejected_repetition", 0))
    text.append(f"Rejections (repeat): {rejected_rep}\n")
    return Panel(text, title="Frontier & Judge", box=box.SIMPLE)


def _render_layout(stats: Optional[TelemetryStats], log_path: Path, args: argparse.Namespace) -> Panel:
    if not stats:
        return Panel(f"Waiting for telemetry at {log_path}", title="Agent0 Dashboard", box=box.SIMPLE)
    tables = [
        _build_metric_table("Curriculum", args.curriculum_metrics, stats),
        _build_metric_table("Executor", args.executor_metrics, stats),
    ]
    pct_table = _build_percentile_table(stats)
    if pct_table:
        tables.append(pct_table)
    tables.append(_build_judge_panel(stats))
    group = Group(*tables)
    return Panel(group, title=f"Agent0 Dashboard :: {log_path}", box=box.ROUNDED)


def _load_stats(log_path: Path) -> Optional[TelemetryStats]:
    try:
        records = list(load_records(log_path))
    except FileNotFoundError:
        return None
    if not records:
        return None
    return summarize(records)


def main() -> None:
    args = _parse_args()
    console = Console()
    with Live(console=console, refresh_per_second=4) as live:
        while True:
            stats = _load_stats(args.log)
            panel = _render_layout(stats, args.log, args)
            live.update(panel, refresh=True)
            time.sleep(max(args.refresh, 0.5))


if __name__ == "__main__":
    main()
