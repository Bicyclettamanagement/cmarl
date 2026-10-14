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

from evaluation.rliable_utils import build_score_matrix, discover_run_dirs_relaxed
from evaluation.visualize import (
    load_report,
    plot_combined_performance_profile,
    plot_context_factor_slices_many,
    plot_generalization_contexts,
    plot_method_comparison_bars,
    plot_per_context_comparison_many,
    plot_sample_efficiency_overlay,
    save_figure_bundle,
    unpack_eval_payload,
)


@dataclass
class Args:
    report_path: str = "reports/rliable_matrix.json"
    runs_root: str = "runs"
    output_dir: str = "reports/figures"


def _matrices_for_reports(reports: list[dict], runs_root: Path) -> list:
    matrices = []
    for report in reports:
        run_dirs = discover_run_dirs_relaxed(
            runs_root,
            report["exp_name"],
            report.get("seeds"),
            method_tag=report.get("method_tag"),
        )
        matrix, _, _ = build_score_matrix(
            run_dirs,
            scores_filename=report.get("scores_filename") or "rliable_scores.json",
        )
        matrices.append(matrix)
    return matrices


def main(args: Args) -> None:
    report_path = Path(args.report_path)
    payload = load_report(report_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unpacked = unpack_eval_payload(payload)
    reports = unpacked["reports"]

    if unpacked["kind"] != "single" and unpacked.get("rliable_combined"):
        import matplotlib.pyplot as plt

        matrices = _matrices_for_reports(reports, Path(args.runs_root))
        entries = list(zip(reports, matrices))

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_combined_performance_profile(unpacked["rliable_combined"], ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "compare_performance_profile.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(reports) + 2.0), 4))
        plot_method_comparison_bars(*reports, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "compare_iqm_bars.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4.2))
        plot_per_context_comparison_many(entries, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "compare_contexts.png", dpi=150)
        plt.close(fig)

        if any(report.get("context_meta") for report in reports):
            slice_entries = [
                (report, matrix)
                for report, matrix in entries
                if report.get("context_meta") is not None
            ]
            if slice_entries:
                axes = plot_context_factor_slices_many(slice_entries)
                fig = axes[0].figure
                fig.tight_layout()
                fig.savefig(output_dir / "compare_context_slices.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

        ax = plot_sample_efficiency_overlay(reports)
        if ax is not None:
            fig = ax.figure
            fig.tight_layout()
            fig.savefig(output_dir / "compare_sample_efficiency.png", dpi=150)
            plt.close(fig)

        print(f"Comparison figures written to {output_dir}")
        return

    report = reports[0]
    run_dirs = discover_run_dirs_relaxed(
        Path(args.runs_root),
        report["exp_name"],
        report.get("seeds"),
        method_tag=report.get("method_tag"),
    )
    matrix, _, _ = (
        build_score_matrix(
            run_dirs,
            scores_filename=report.get("scores_filename") or "rliable_scores.json",
        )
        if run_dirs
        else (None, [], [])
    )
    save_figure_bundle(report_path, output_dir, score_matrix=matrix)
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
