"""Tests for evaluation visualization helpers."""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np

from evaluation.visualize import (
    aggregate_metrics_table,
    clone_trapezoid_rows,
    env_switch_rows,
    format_aggregate_summary,
    format_env_switch_advice,
    format_pairwise_poi,
    load_report,
    methodology_dashboard_rows,
    plot_aggregate_intervals,
    plot_context_factor_slices,
    plot_context_factor_slices_many,
    plot_generalization_contexts,
    plot_method_comparison_bars,
    plot_per_context_comparison_many,
    plot_performance_profile,
    plot_seed_health,
    save_figure_bundle,
    seed_health_rows,
    slice_params_from_report,
    unpack_eval_payload,
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


def test_plot_context_factor_slices():
    report = {
        "exp_name": "test_exp",
        "context_order": ["default", "ball_mass_x0.5", "ball_mass_x2"],
        "context_meta": {
            "default": {"split": "id", "varied_param": None, "factor": 1.0},
            "ball_mass_x0.5": {"split": "ood", "varied_param": "ball_mass", "factor": 0.5},
            "ball_mass_x2": {"split": "ood", "varied_param": "ball_mass", "factor": 2.0},
        },
        "id_factor_low": 0.75,
        "id_factor_high": 1.5,
    }
    matrix = np.array([[20.0, 8.0, 9.0], [18.0, 7.0, 10.0]])
    axes = plot_context_factor_slices(report, matrix, params=("ball_mass",))
    assert axes[0].lines


def test_unpack_eval_payload_matrix():
    payload = {
        "experiments": [_minimal_report(), {**_minimal_report(), "label": "other", "exp_name": "other"}],
        "pairwise_poi": [
            {
                "a": "test_exp",
                "b": "other",
                "probability_a_beats_b": {
                    "probability_of_improvement": 0.6,
                    "ci_low": 0.4,
                    "ci_high": 0.8,
                },
            }
        ],
        "rliable_combined": {"performance_profiles": {}},
    }
    unpacked = unpack_eval_payload(payload)
    assert unpacked["kind"] == "compare"
    assert len(unpacked["reports"]) == 2
    assert "P(test_exp > other)" in format_pairwise_poi(unpacked["pairwise_poi"])


def test_plot_method_comparison_bars_many():
    a = _minimal_report()
    a["label"] = "test_exp"
    b = _minimal_report()
    b["exp_name"] = "other"
    b["label"] = "other"
    b["rliable"]["algorithms"]["other"] = b["rliable"]["algorithms"].pop("test_exp")
    ax = plot_method_comparison_bars(a, b)
    assert len(ax.patches) == 2


def test_plot_per_context_and_slices_many():
    report_a = {
        "exp_name": "a",
        "label": "A",
        "context_order": ["default", "ball_mass_x0.5", "ball_mass_x2"],
        "context_meta": {
            "default": {"split": "id", "varied_param": None, "factor": 1.0},
            "ball_mass_x0.5": {"split": "ood", "varied_param": "ball_mass", "factor": 0.5},
            "ball_mass_x2": {"split": "ood", "varied_param": "ball_mass", "factor": 2.0},
        },
        "id_factor_low": 0.75,
        "id_factor_high": 1.5,
    }
    report_b = {**report_a, "exp_name": "b", "label": "B"}
    matrix_a = np.array([[20.0, 8.0, 9.0], [18.0, 7.0, 10.0]])
    matrix_b = np.array([[22.0, 12.0, 11.0], [19.0, 10.0, 13.0]])
    ax = plot_per_context_comparison_many([(report_a, matrix_a), (report_b, matrix_b)])
    assert len(ax.patches) == 6
    axes = plot_context_factor_slices_many(
        [(report_a, matrix_a), (report_b, matrix_b)],
        params=("ball_mass",),
    )
    assert axes[0].lines


def test_plot_per_context_comparison_mismatched_grids():
    """JSON context_order can be shorter than a later eval matrix (e.g. 25 vs 29)."""
    report_a = {
        "exp_name": "a",
        "label": "A",
        "context_order": ["default", "ball_mass_x0.5"],
    }
    report_b = {
        "exp_name": "b",
        "label": "B",
        "context_order": ["default", "ball_mass_x0.5", "ball_mass_x2", "gravity_x1.5"],
    }
    matrix_a = np.array([[20.0, 8.0, 1.0, 2.0], [18.0, 7.0, 1.0, 2.0]])
    matrix_b = np.array([[22.0, 12.0, 11.0, 5.0], [19.0, 10.0, 13.0, 6.0]])
    ax = plot_per_context_comparison_many([(report_a, matrix_a), (report_b, matrix_b)])
    assert len(ax.patches) == 8
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "default",
        "ball_mass_x0.5",
        "ball_mass_x2",
        "gravity_x1.5",
    ]


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


def test_methodology_dashboard_and_flat_annotation():
    report = {
        "exp_name": "test_exp",
        "label": "Hidden context (varying)",
        "n_seeds": 2,
        "seeds": [1, 2],
        "context_order": ["default", "ball_mass_x0.5", "ball_mass_x2"],
        "context_meta": {
            "default": {"split": "id", "varied_param": None, "factor": 1.0},
            "ball_mass_x0.5": {"split": "ood", "varied_param": "ball_mass", "factor": 0.5},
            "ball_mass_x2": {"split": "ood", "varied_param": "ball_mass", "factor": 2.0},
        },
        "id_factor_low": 0.75,
        "id_factor_high": 1.5,
        "eval_factor_grid": [0.5, 1.0, 2.0],
        "generalization": {
            "id_return_iqm": 20.0,
            "ood_return_iqm": 19.5,
            "generalization_gap_iqm": 0.5,
        },
        "sample_efficiency": {
            "median_steps_to_threshold": {"median_steps_to_threshold": 50000, "n_seeds_reached": 2}
        },
        "rliable": {
            "algorithms": {
                "Hidden context (varying)": {
                    "metrics": {
                        "iqm": {"point": 20.0, "ci_low": 18.0, "ci_high": 22.0},
                        "mean": {"point": 19.0, "ci_low": 17.0, "ci_high": 21.0},
                    }
                }
            }
        },
    }
    matrix = np.array([[80.0, 80.0, 80.0], [82.0, 82.0, 82.0]])
    assert slice_params_from_report(report) == ("ball_mass",)
    dash = methodology_dashboard_rows([report], [matrix])
    assert dash[0]["cloned axes"] == "ball_mass"
    assert "ball_mass" in dash[0]["flat C.2 axes"]
    trap = clone_trapezoid_rows([report], [matrix])
    assert trap[0]["cloned"] is True
    health = seed_health_rows([report], [matrix])
    assert len(health) == 2
    axes = plot_context_factor_slices(report, matrix, params=("ball_mass",))
    assert "FLAT" in axes[0].get_title()
    advice = format_env_switch_advice(env_switch_rows([report], [matrix]))
    assert "re_eval_hard_grid_with_gravity" in advice
    ax = plot_seed_health([report], [matrix])
    assert ax is not None
