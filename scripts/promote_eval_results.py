#!/usr/bin/env python3
"""Promote OpenCompass evaluation results to reports/ and update README."""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_EVALS = ROOT / "reports" / "evals"
README = ROOT / "README.md"

BENCHMARK_TABLE_MARKER = "### Latest Benchmark Snapshot"


def _extract_summary(work_dir: Path) -> tuple[str, Path]:
    """Find summary markdown and return contents + path."""
    summary_dir = work_dir / "summary"
    mds = list(summary_dir.glob("*.md"))
    if not mds:
        raise FileNotFoundError(f"No summary markdown found in {summary_dir}")
    md_path = mds[0]
    return md_path.read_text(), md_path


def _parse_scores(md_content: str) -> dict[str, dict]:
    """Parse markdown table into {dataset: {metric, mode, score}}."""
    scores = {}
    for line in md_content.splitlines():
        if line.startswith("|") and "dataset" not in line.lower() and "---" not in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 5:
                dataset, version, metric, mode, score = parts[:5]
                scores[dataset] = {"version": version, "metric": metric, "mode": mode, "score": score}
    return scores


def _format_readme_table(scores: dict[str, dict], run_date: str) -> str:
    lines = [
        f"### Latest Benchmark Snapshot ({run_date})",
        "| Dataset | Config | Metric | Mode | Score |",
        "| --- | --- | --- | --- | --- |",
    ]
    for ds, info in sorted(scores.items()):
        lines.append(f"| {ds.upper()} | `{ds}_gen_{info['version']}` | {info['metric']} | `{info['mode']}` | **{info['score']}** |")
    return "\n".join(lines)


def _update_readme(new_table: str) -> None:
    content = README.read_text()
    # Find and replace existing benchmark table section
    pattern = re.compile(
        rf"({re.escape(BENCHMARK_TABLE_MARKER)}.*?)(?=\n##|\n### [^L]|\Z)",
        re.DOTALL,
    )
    match = pattern.search(content)
    if match:
        content = content[: match.start()] + new_table + "\n" + content[match.end() :]
    else:
        # Append after Evaluation Harness section header
        insert_marker = "## Evaluation Harness (OpenCompass)"
        idx = content.find(insert_marker)
        if idx != -1:
            # Find end of that section (next ##)
            next_section = content.find("\n## ", idx + len(insert_marker))
            if next_section == -1:
                next_section = len(content)
            content = content[:next_section] + "\n" + new_table + "\n" + content[next_section:]
        else:
            content += "\n" + new_table + "\n"
    README.write_text(content)


def promote(work_dir: Path, run_name: str | None = None) -> None:
    md_content, md_path = _extract_summary(work_dir)
    scores = _parse_scores(md_content)

    # Determine run date from work_dir name or fallback to today
    match = re.search(r"(\d{8})_\d{6}", work_dir.name)
    if match:
        run_date = datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")
    else:
        run_date = datetime.now().strftime("%Y-%m-%d")

    if not run_name:
        run_name = work_dir.name

    # Copy summary to reports/evals
    REPORTS_EVALS.mkdir(parents=True, exist_ok=True)
    dest = REPORTS_EVALS / f"{run_name}.md"
    shutil.copy(md_path, dest)
    print(f"Copied summary to {dest}")

    # Update README table
    new_table = _format_readme_table(scores, run_date)
    _update_readme(new_table)
    print(f"Updated {README} with new benchmark snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote OpenCompass results to reports and README")
    parser.add_argument("work_dir", type=Path, help="OpenCompass run directory containing summary/")
    parser.add_argument("--run-name", default=None, help="Override name for the report file")
    args = parser.parse_args()

    promote(args.work_dir, args.run_name)


if __name__ == "__main__":
    main()
