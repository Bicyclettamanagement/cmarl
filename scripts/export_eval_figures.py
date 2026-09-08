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
from evaluation.visualize import (
    load_report,
    plot_combined_performance_profile,
    plot_generalization_contexts,
    plot_method_comparison_bars,
    plot_per_context_comparison,
    save_figure_bundle,
)


@dataclass
class Args:
    report_path: str = "reports/rliable_baseline5_shared_randdrop.json"
    runs_root: str = "runs"
    output_dir: str = "reports/figures"


def main(args: Args) -> None:
    report_path = Path(args.report_path)
    payload = load_report(report_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Comparison report (context vs baseline)
    if "primary" in payload and "challenger" in payload:
        primary = payload["primary"]
        challenger = payload["challenger"]
        run_dirs_a = discover_run_dirs(Path(args.runs_root), primary["exp_name"], primary.get("seeds"))
        run_dirs_b = discover_run_dirs(Path(args.runs_root), challenger["exp_name"], challenger.get("seeds"))
        matrix_a, _, _ = build_score_matrix(run_dirs_a)
        matrix_b, _, _ = build_score_matrix(run_dirs_b)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_combined_performance_profile(payload["rliable_combined"], ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "compare_performance_profile.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_method_comparison_bars(primary, challenger, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "compare_iqm_bars.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4))
        plot_per_context_comparison(primary, matrix_a, challenger, matrix_b, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "compare_contexts.png", dpi=150)
        plt.close(fig)
        print(f"Comparison figures written to {output_dir}")
        return

    run_dirs = discover_run_dirs(Path(args.runs_root), payload["exp_name"], payload.get("seeds"))
    matrix, _, _ = build_score_matrix(run_dirs) if run_dirs else (None, [], [])
    save_figure_bundle(report_path, output_dir, score_matrix=matrix)
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
