"""Tests for rliable evaluation utilities."""

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.rliable_utils import (
    aggregate_with_rliable,
    build_score_matrix,
    compute_generalization_gap,
    save_rliable_scores,
    summarize_episode_scores,
)


def test_summarize_episode_scores_includes_iqm():
    stats = summarize_episode_scores([1.0, 2.0, 3.0, 100.0], reference_return=100.0)
    assert "return_iqm" in stats
    assert stats["return_mean"] == pytest.approx(26.5)
    assert stats["optimality_gap"] >= 0.0


def test_save_and_build_score_matrix(tmp_path):
    for seed in (1, 2):
        save_rliable_scores(
            tmp_path / f"run_{seed}",
            algorithm="matd3_pistonball",
            method_tag="hidden_context",
            seed=seed,
            global_step=1000,
            context_order=["train", "ctx_a"],
            scores_by_context={"train": [10.0, 12.0], "ctx_a": [5.0, 7.0]},
            eval_split="zero_shot_transfer",
            reference_return=100.0,
        )
    run_dirs = [tmp_path / "run_1", tmp_path / "run_2"]
    matrix, order, seeds = build_score_matrix(run_dirs)
    assert order == ["train", "ctx_a"]
    assert seeds == [1, 2]
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(11.0)


def test_aggregate_with_rliable_bootstrap(tmp_path):
    save_rliable_scores(
        tmp_path / "run_1",
        algorithm="matd3_pistonball",
        method_tag="hidden_context",
        seed=1,
        global_step=1000,
        context_order=["train", "ood"],
        scores_by_context={"train": [80.0, 90.0], "ood": [40.0, 50.0]},
        eval_split="zero_shot_transfer",
        reference_return=100.0,
    )
    save_rliable_scores(
        tmp_path / "run_2",
        algorithm="matd3_pistonball",
        method_tag="hidden_context",
        seed=2,
        global_step=1000,
        context_order=["train", "ood"],
        scores_by_context={"train": [70.0, 85.0], "ood": [30.0, 45.0]},
        eval_split="zero_shot_transfer",
        reference_return=100.0,
    )
    matrix, order, _ = build_score_matrix([tmp_path / "run_1", tmp_path / "run_2"])
    gap = compute_generalization_gap(matrix, order)
    assert gap["generalization_gap"] > 0
    report = aggregate_with_rliable({"matd3_pistonball": matrix}, reference_return=100.0, bootstrap_reps=200)
    iqm = report["algorithms"]["matd3_pistonball"]["metrics"]["iqm"]
    assert iqm["ci_low"] <= iqm["point"] <= iqm["ci_high"]
