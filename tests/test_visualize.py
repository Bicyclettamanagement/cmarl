"""Tests for evaluation visualization helpers."""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np

from evaluation.visualize import (
    aggregate_metrics_table,
    format_aggregate_summary,
    load_report,
    plot_aggregate_intervals,
    plot_generalization_contexts,
    plot_performance_profile,
    save_figure_bundle,
)


def _minimal_report():
    thresholds = np.linspace(-10, 50, 11).tolist()
    profile = np.linspace(1.0, 0.0, 11).tolist()
    return {
        "exp_name": "test_exp",
        "seeds": [1, 2],
        "context_order": ["train", "ood"],
        "generalization": {
            "id_return_mean": 20.0,
            "ood_return_mean": 10.0,
            "generalization_gap": 10.0,
        },
        "sample_efficiency": {"median_steps_to_threshold": {"median_steps_to_threshold": None}},
        "rliable": {
            "algorithms": {
                "test_exp": {
                    "metrics": {
                        "iqm": {"point": 15.0, "ci_low": 12.0, "ci_high": 18.0},
                        "median": {"point": 14.0, "ci_low": 11.0, "ci_high": 17.0},
                        "mean": {"point": 15.5, "ci_low": 12.5, "ci_high": 18.5},
                        "optimality_gap": {"point": 5.0, "ci_low": 4.0, "ci_high": 6.0},
                    }
                }
            },
            "performance_profiles": {
                "test_exp": {
                    "thresholds": thresholds,
                    "profile": profile,
                    "ci_low": profile,
                    "ci_high": profile,
                }
            },
        },
    }


def test_aggregate_metrics_table():
    rows = aggregate_metrics_table(_minimal_report())
    assert len(rows) == 4
    assert rows[0]["metric"] == "iqm"


def test_format_summary_contains_generalization():
    text = format_aggregate_summary(_minimal_report())
    assert "Generalization" in text
    assert "IQM" in text


def test_plot_performance_profile_runs():
    report = _minimal_report()
    ax = plot_performance_profile(report)
    assert ax.lines


def test_plot_aggregate_intervals_runs():
    ax = plot_aggregate_intervals(_minimal_report())
    assert len(ax.patches) == 4


def test_plot_generalization_contexts():
    report = _minimal_report()
    matrix = np.array([[20.0, 10.0], [18.0, 8.0]])
    ax = plot_generalization_contexts(report, matrix)
    assert len(ax.patches) == 2


def test_save_figure_bundle(tmp_path):
    report = _minimal_report()
    report_path = tmp_path / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f)
    matrix = np.array([[20.0, 10.0], [18.0, 8.0]])
    save_figure_bundle(report_path, tmp_path / "figs", score_matrix=matrix)
    assert (tmp_path / "figs" / "test_exp_performance_profile.png").exists()
    loaded = load_report(report_path)
    assert loaded["exp_name"] == "test_exp"
