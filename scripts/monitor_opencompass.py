#!/usr/bin/env python3
"""Rich-based live dashboard for monitoring OpenCompass evaluation runs."""
from __future__ import annotations

import argparse
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\[([^\]]+)<([^\]]+),\s*([^\]]+)\]")
TIMESTAMP_RE = re.compile(r"(\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})")


@dataclass
class DatasetStatus:
    name: str
    completed: int = 0
    total: int = 0
    elapsed: str = ""
    remaining: str = ""
    rate: str = ""
    last_update: float = field(default_factory=time.time)
    finished: bool = False


def _parse_log_tail(log_path: Path, last_pos: int) -> tuple[int, Optional[DatasetStatus]]:
    """Read new lines from log, return updated position and parsed status."""
    try:
        size = log_path.stat().st_size
        if size <= last_pos:
            return last_pos, None
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(last_pos)
            chunk = fh.read()
            new_pos = fh.tell()
    except (FileNotFoundError, OSError):
        return last_pos, None

    status: Optional[DatasetStatus] = None
    for line in reversed(chunk.splitlines()):
        match = PROGRESS_RE.search(line)
        if match:
            completed, total, elapsed, remaining, rate = match.groups()
            status = DatasetStatus(
                name=log_path.stem,
                completed=int(completed),
                total=int(total),
                elapsed=elapsed.strip(),
                remaining=remaining.strip(),
                rate=rate.strip(),
                last_update=time.time(),
                finished=(int(completed) >= int(total)),
            )
            break
    return new_pos, status


def _build_table(statuses: Dict[str, DatasetStatus], stall_threshold: float) -> Table:
    table = Table(title="OpenCompass Evaluation Progress", box=box.ROUNDED)
    table.add_column("Dataset", style="bold cyan")
    table.add_column("Progress", justify="right")
    table.add_column("Elapsed", justify="right")
    table.add_column("ETA", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Status", justify="center")

    now = time.time()
    for name in sorted(statuses.keys()):
        s = statuses[name]
        pct = (s.completed / s.total * 100) if s.total else 0
        progress_str = f"{s.completed}/{s.total} ({pct:.1f}%)"

        idle_sec = now - s.last_update
        if s.finished:
            status_str = Text("✓ done", style="bold green")
        elif idle_sec > stall_threshold:
            status_str = Text(f"⚠ stall {int(idle_sec)}s", style="bold red")
        else:
            status_str = Text("● active", style="bold yellow")

        table.add_row(name, progress_str, s.elapsed, s.remaining, s.rate, status_str)

    return table


def _build_panel(
    statuses: Dict[str, DatasetStatus],
    work_dir: Path,
    stall_threshold: float,
    start_time: datetime,
) -> Panel:
    table = _build_table(statuses, stall_threshold)
    elapsed = datetime.now() - start_time
    footer = Text(f"\nMonitoring: {work_dir}\nSession elapsed: {str(elapsed).split('.')[0]}", style="dim")
    return Panel(table, subtitle=str(footer), box=box.DOUBLE)


def monitor(work_dir: Path, refresh: float = 5.0, stall_threshold: float = 300.0) -> None:
    console = Console()
    log_root = work_dir / "logs" / "infer"
    positions: Dict[Path, int] = {}
    statuses: Dict[str, DatasetStatus] = {}
    start_time = datetime.now()

    console.print(f"[bold]Waiting for logs under {log_root}...[/bold]")
    while not log_root.exists():
        time.sleep(2)

    with Live(console=console, refresh_per_second=2) as live:
        while True:
            for log_file in sorted(log_root.rglob("*.out")):
                last_pos = positions.get(log_file, 0)
                new_pos, status = _parse_log_tail(log_file, last_pos)
                positions[log_file] = new_pos
                if status:
                    statuses[status.name] = status

            panel = _build_panel(statuses, work_dir, stall_threshold, start_time)
            live.update(panel)

            # Check if all datasets finished
            if statuses and all(s.finished for s in statuses.values()):
                console.print("[bold green]All datasets complete![/bold green]")
                break

            time.sleep(refresh)


def _find_latest_run(base_dir: Path) -> Optional[Path]:
    """Find the most recent timestamped run directory."""
    candidates = sorted(base_dir.glob("**/logs/infer"), key=lambda p: p.parent.name, reverse=True)
    if candidates:
        return candidates[0].parent
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor OpenCompass evaluation progress")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="OpenCompass work directory (auto-detects latest if omitted)",
    )
    parser.add_argument("--refresh", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument(
        "--stall-threshold",
        type=float,
        default=300.0,
        help="Seconds of inactivity before flagging a stall",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "opencompass",
        help="Base directory to search for runs if --work-dir not provided",
    )
    args = parser.parse_args()

    if args.work_dir:
        work_dir = args.work_dir
    else:
        work_dir = _find_latest_run(args.base_dir)
        if not work_dir:
            print(f"No runs found under {args.base_dir}. Start an evaluation first or specify --work-dir.")
            return

    monitor(work_dir, refresh=args.refresh, stall_threshold=args.stall_threshold)


if __name__ == "__main__":
    main()
