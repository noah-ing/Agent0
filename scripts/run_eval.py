#!/usr/bin/env python3
"""Convenience wrapper for running OpenCompass evaluations."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
OPENCOMPASS_SCRIPT = ROOT / "scripts" / "run_opencompass_eval.py"
MONITOR_SCRIPT = ROOT / "scripts" / "monitor_opencompass.py"
PROMOTE_SCRIPT = ROOT / "scripts" / "promote_eval_results.py"
EXP_FOLDER_RE = re.compile(r"Current exp folder:\s*(.+?)\s*$")
PROGRESS_RE = re.compile(r"(\d+)/(\d+)")
LOG_POLL_SECONDS = 15


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agent0 OpenCompass harness with sensible defaults")
    parser.add_argument("--suite", choices=["math-lite", "math-heavy"], default="math-lite")
    parser.add_argument("--datasets", nargs="*", default=None, help="Override suite with explicit dataset names")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs" / "opencompass")
    parser.add_argument("--mode", choices=["all", "infer", "eval", "viz"], default="all")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--reuse", default=None, help="Reuse an existing OpenCompass timestamp")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--env-key",
        default=None,
        help="Override API key env var forwarded to OpenCompass (defaults to AGENT0_EVAL_API_KEY or OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Additional arguments passed verbatim to run_opencompass_eval.py after '--'",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Launch Rich dashboard in a separate process for live progress",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="After completion, promote results to reports/ and update README",
    )
    return parser.parse_args()


def _build_command(args: argparse.Namespace) -> List[str]:
    if not OPENCOMPASS_SCRIPT.exists():
        raise SystemExit(f"Expected helper script at {OPENCOMPASS_SCRIPT}")
    cmd: List[str] = [sys.executable, str(OPENCOMPASS_SCRIPT)]
    if args.datasets:
        cmd.extend(["--datasets", *args.datasets])
    else:
        cmd.extend(["--suite", args.suite])
    cmd.extend(["--work-dir", str(args.work_dir)])
    cmd.extend(["--mode", args.mode])
    cmd.extend(["--max-workers", str(args.max_workers)])
    if args.debug:
        cmd.append("--debug")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.reuse:
        cmd.extend(["--reuse", args.reuse])
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _summarize_lines(lines: Iterable[str]) -> Optional[str]:
    interesting: List[str] = []
    for raw in lines:
        stripped = _strip_ansi(raw).strip()
        if not stripped:
            continue
        if "%|" in stripped or "Inferencing" in stripped or "examples/s" in stripped:
            interesting.append(stripped)
    if interesting:
        return interesting[-1]
    return None


def _should_emit(summary: str) -> bool:
    match = PROGRESS_RE.search(summary)
    if match:
        completed, total = match.groups()
        if completed == "0" and total == "1":
            return False
    return True


def _monitor_progress(exp_folder: Path, stop_event: threading.Event) -> None:
    log_root = exp_folder / "logs" / "infer"
    positions: Dict[Path, int] = {}
    last_summary: Dict[Path, str] = {}
    notified = False
    while not stop_event.is_set():
        if log_root.exists():
            if not notified:
                print(f"[progress] tracking OpenCompass logs under {log_root}")
                notified = True
            for log_file in sorted(log_root.rglob("*.out")):
                try:
                    pos = positions.get(log_file)
                    with log_file.open("r", encoding="utf-8", errors="ignore") as fh:
                        if pos is None:
                            fh.seek(0, os.SEEK_END)
                            positions[log_file] = fh.tell()
                            continue
                        fh.seek(pos)
                        chunk = fh.read()
                        if chunk:
                            positions[log_file] = fh.tell()
                            summary = _summarize_lines(chunk.splitlines())
                            if summary and _should_emit(summary):
                                if summary == last_summary.get(log_file):
                                    continue
                                last_summary[log_file] = summary
                                print(f"[progress:{log_file.stem}] {summary}")
                except FileNotFoundError:
                    positions.pop(log_file, None)
                    last_summary.pop(log_file, None)
        stop_event.wait(LOG_POLL_SECONDS)


def _stream_command(cmd: List[str], env: dict) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    stop_event = threading.Event()
    monitor_thread: Optional[threading.Thread] = None
    try:
        for line in proc.stdout:
            print(line, end="")
            if monitor_thread is None:
                match = EXP_FOLDER_RE.search(line)
                if match:
                    candidate = Path(match.group(1).strip())
                    if candidate.exists():
                        monitor_thread = threading.Thread(
                            target=_monitor_progress,
                            args=(candidate, stop_event),
                            daemon=True,
                        )
                        monitor_thread.start()
        returncode = proc.wait()
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def main() -> None:
    args = _parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_command(args)
    env = os.environ.copy()
    api_key_var = args.env_key or "AGENT0_EVAL_API_KEY"
    eval_key = env.get(api_key_var) or env.get("OPENAI_API_KEY")
    if eval_key:
        env["AGENT0_EVAL_API_KEY"] = eval_key
    print("[eval] launching:", " ".join(cmd))
    if args.dry_run:
        return

    # Optionally launch Rich dashboard monitor in background
    monitor_proc: Optional[subprocess.Popen] = None
    if args.monitor and MONITOR_SCRIPT.exists():
        monitor_proc = subprocess.Popen(
            [sys.executable, str(MONITOR_SCRIPT), "--base-dir", str(args.work_dir)],
            env=env,
        )
        print(f"[eval] started monitor (PID {monitor_proc.pid})")

    exp_folder: Optional[Path] = None
    try:
        # Stream the command and capture exp folder path
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        stop_event = threading.Event()
        monitor_thread: Optional[threading.Thread] = None
        for line in proc.stdout:
            print(line, end="")
            if exp_folder is None:
                match = EXP_FOLDER_RE.search(line)
                if match:
                    exp_folder = Path(match.group(1).strip())
            if monitor_thread is None and exp_folder and exp_folder.exists():
                monitor_thread = threading.Thread(
                    target=_monitor_progress,
                    args=(exp_folder, stop_event),
                    daemon=True,
                )
                monitor_thread.start()
        returncode = proc.wait()
        stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=1)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
    finally:
        if monitor_proc:
            monitor_proc.terminate()

    # Optionally promote results
    if args.promote and exp_folder and PROMOTE_SCRIPT.exists():
        print(f"[eval] promoting results from {exp_folder}")
        subprocess.run([sys.executable, str(PROMOTE_SCRIPT), str(exp_folder)], check=True)


if __name__ == "__main__":
    main()
