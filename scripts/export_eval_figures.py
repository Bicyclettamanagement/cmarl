#!/usr/bin/env python3
"""Export PNG figures from an rliable JSON report (no Jupyter required)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tyro

from evaluation.rliable_utils import build_score_matrix, discover_run_dirs
from evaluation.visualize import load_report, save_figure_bundle


@dataclass
class Args:
    report_path: str = "reports/rliable_matd3_pistonball.json"
    runs_root: str = "runs"
    output_dir: str = "reports/figures"


def main(args: Args) -> None:
    report_path = Path(args.report_path)
    report = load_report(report_path)
    run_dirs = discover_run_dirs(Path(args.runs_root), report["exp_name"], report.get("seeds"))
    matrix, _, _ = build_score_matrix(run_dirs) if run_dirs else (None, [], [])
    save_figure_bundle(report_path, Path(args.output_dir), score_matrix=matrix)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
