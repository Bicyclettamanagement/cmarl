"""Readable matplotlib figures for rliable JSON reports (notebook + cluster post-processing)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

METRIC_LABELS = {
    "median": "Median return",
    "iqm": "IQM return",
    "mean": "Mean return",
    "optimality_gap": "Optimality gap",
}


def load_report(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def aggregate_metrics_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten rliable aggregate metrics into rows for display (e.g. pandas)."""
    rows: list[dict[str, Any]] = []
    for algo, block in report.get("rliable", {}).get("algorithms", {}).items():
        for metric, vals in block.get("metrics", {}).items():
            rows.append(
                {
                    "algorithm": algo,
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "point": vals["point"],
                    "ci_low": vals["ci_low"],
                    "ci_high": vals["ci_high"],
                }
            )
    return rows


def format_aggregate_summary(report: dict[str, Any]) -> str:
    """Human-readable summary for notebook markdown cells."""
    lines = [
        f"Experiment: **{report.get('exp_name', '?')}**",
        f"Seeds: {report.get('seeds', [])}",
        f"Contexts: {report.get('context_order', [])}",
        "",
    ]
    gen = report.get("generalization", {})
    lines.append(
        f"Generalization (mean return): ID={gen.get('id_return_mean', float('nan')):.2f}, "
        f"OOD={gen.get('ood_return_mean', float('nan')):.2f}, "
        f"gap={gen.get('generalization_gap', float('nan')):.2f}"
    )
    med = report.get("sample_efficiency", {}).get("median_steps_to_threshold", {})
    if med.get("median_steps_to_threshold") is not None:
        lines.append(
            f"Median steps to threshold: {med['median_steps_to_threshold']:.0f} "
            f"({med.get('n_seeds_reached', 0)} seeds)"
        )
    lines.append("")
    for row in aggregate_metrics_table(report):
        lines.append(
            f"- {row['label']}: {row['point']:.2f} "
            f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        )
    if "comparison" in report:
        poi = report["comparison"]["probability_of_improvement"]
        lines.append("")
        lines.append(
            f"P({report['comparison']['baseline']} > {report['comparison']['challenger']}): "
            f"{poi['probability_of_improvement']:.2f} "
            f"[{poi['ci_low']:.2f}, {poi['ci_high']:.2f}]"
        )
    return "\n".join(lines)


def plot_performance_profile(
    report: dict[str, Any],
    *,
    algorithm: str | None = None,
    ax: plt.Axes | None = None,
    title: str = "Performance profile (rliable)",
) -> plt.Axes:
    """Plot P(return ≥ τ) with bootstrap CI bands from a saved report."""
    profiles = report["rliable"]["performance_profiles"]
    algo = algorithm or report["exp_name"]
    if algo not in profiles:
        raise KeyError(f"Algorithm {algo!r} not in report performance profiles.")
    data = profiles[algo]
    thresholds = np.asarray(data["thresholds"], dtype=float)
    profile = np.asarray(data["profile"], dtype=float)
    ci_low = np.asarray(data["ci_low"], dtype=float)
    ci_high = np.asarray(data["ci_high"], dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, profile, linewidth=2.5, label=algo)
    ax.fill_between(thresholds, ci_low, ci_high, alpha=0.25)
    ax.set_xlabel("Team return threshold τ")
    ax.set_ylabel("Fraction of runs × contexts with return ≥ τ")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title(title)
    return ax


def plot_aggregate_intervals(
    report: dict[str, Any],
    *,
    algorithm: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar-style interval plot for IQM / median / mean / optimality gap."""
    algo = algorithm or report["exp_name"]
    metrics = report["rliable"]["algorithms"][algo]["metrics"]
    order = ["iqm", "median", "mean", "optimality_gap"]
    labels = [METRIC_LABELS[m] for m in order]

    points = [metrics[m]["point"] for m in order]
    lows = [metrics[m]["ci_low"] for m in order]
    highs = [metrics[m]["ci_high"] for m in order]
    yerr = np.array([np.array(points) - np.array(lows), np.array(highs) - np.array(points)])

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(order))
    ax.bar(x, points, yerr=yerr, capsize=6, color="#4C72B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(f"Aggregate metrics with 95% stratified bootstrap CI — {algo}")
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_generalization_contexts(
    report: dict[str, Any],
    score_matrix: np.ndarray,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Mean return ± seed std for each evaluation context (ID highlighted)."""
    contexts = report["context_order"]
    means = np.mean(score_matrix, axis=0)
    stds = np.std(score_matrix, axis=0)
    colors = ["#55A868" if c == "train" else "#C44E52" for c in contexts]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(contexts))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(contexts, rotation=35, ha="right")
    ax.axhline(means[contexts.index("train")] if "train" in contexts else 0, color="gray", ls="--", lw=1)
    ax.set_ylabel("Mean team return (across seeds)")
    ax.set_title("Zero-shot performance by context (green = in-distribution train)")
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_sample_efficiency_curve(
    report: dict[str, Any],
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes | None:
    """IQM in-distribution eval return vs training step (if present in report)."""
    curve = report.get("sample_efficiency", {}).get("eval_iqm_curve", {})
    if "error" in curve or "global_steps" not in curve:
        return None
    steps = np.asarray(curve["global_steps"])
    iqm = np.asarray(curve["iqm"])
    ci_low = np.asarray(curve["ci_low"])
    ci_high = np.asarray(curve["ci_high"])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, iqm, lw=2)
    ax.fill_between(steps, ci_low, ci_high, alpha=0.25)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("IQM team return (in-distribution eval)")
    ax.set_title("Sample efficiency (rliable IQM across seeds)")
    ax.grid(True, alpha=0.3)
    return ax


def save_figure_bundle(report_path: Path, output_dir: Path, score_matrix: np.ndarray | None = None) -> None:
    """Write standard evaluation figures next to the JSON report."""
    report = load_report(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    exp = report["exp_name"]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_performance_profile(report, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{exp}_performance_profile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_aggregate_intervals(report, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{exp}_aggregate_metrics.png", dpi=150)
    plt.close(fig)

    if score_matrix is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_generalization_contexts(report, score_matrix, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{exp}_contexts.png", dpi=150)
        plt.close(fig)

    ax = plot_sample_efficiency_curve(report)
    if ax is not None:
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(output_dir / f"{exp}_sample_efficiency.png", dpi=150)
        plt.close(fig)
