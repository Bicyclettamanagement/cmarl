"""Clone detection, C.2 trapezoid checks, seed health, and env-switch advice."""

from __future__ import annotations

import numpy as np

from evaluation.diagnostics import (
    diagnose_method,
    detect_clone_groups,
    recommend_env_switch,
    seed_health,
    slice_trapezoid_stats,
)


def _mass_friction_meta() -> tuple[list[str], dict]:
    order = [
        "default",
        "ball_mass_x0.5",
        "ball_mass_x2",
        "ball_friction_x0.5",
        "ball_friction_x2",
        "gravity_x0.25",
        "gravity_x4",
    ]
    meta = {
        "default": {"split": "id", "varied_param": None, "factor": 1.0},
        "ball_mass_x0.5": {"split": "ood", "varied_param": "ball_mass", "factor": 0.5},
        "ball_mass_x2": {"split": "ood", "varied_param": "ball_mass", "factor": 2.0},
        "ball_friction_x0.5": {"split": "ood", "varied_param": "ball_friction", "factor": 0.5},
        "ball_friction_x2": {"split": "ood", "varied_param": "ball_friction", "factor": 2.0},
        "gravity_x0.25": {"split": "ood", "varied_param": "gravity", "factor": 0.25},
        "gravity_x4": {"split": "ood", "varied_param": "gravity", "factor": 4.0},
    }
    return order, meta


def test_detect_clone_groups_flags_identical_mass_slice():
    order, meta = _mass_friction_meta()
    # mass cloned per seed; friction and gravity vary
    matrix = np.array(
        [
            [80.0, 10.0, 10.0, 50.0, 40.0, 70.0, 20.0],
            [70.0, 11.0, 11.0, 55.0, 35.0, 65.0, 25.0],
        ]
    )
    groups = {g["param"]: g for g in detect_clone_groups(matrix, order, meta)}
    assert groups["ball_mass"]["cloned"] is True
    assert groups["ball_friction"]["cloned"] is False
    assert groups["gravity"]["cloned"] is False


def test_slice_trapezoid_stats_marks_flat_mass():
    order, meta = _mass_friction_meta()
    matrix = np.array(
        [
            [80.0, 79.0, 78.0, 50.0, 40.0, 70.0, 20.0],
            [82.0, 81.0, 80.0, 52.0, 38.0, 68.0, 22.0],
        ]
    )
    rows = {r["param"]: r for r in slice_trapezoid_stats(matrix, order, meta)}
    assert rows["ball_mass"]["flat"] is True
    assert rows["ball_friction"]["flat"] is False
    assert rows["gravity"]["flat"] is False
    assert rows["gravity"]["ood_drop"] > 5.0


def test_seed_health_flags_collapse():
    order, meta = _mass_friction_meta()
    matrix = np.array(
        [
            [80.0, 10.0, 10.0, 50.0, 40.0, 70.0, 20.0],
            [20.0, 18.0, 19.0, 15.0, 16.0, 12.0, 11.0],
        ]
    )
    health = seed_health(matrix, order, seeds=[1, 2])
    assert health[0]["collapsed"] is False
    assert health[1]["collapsed"] is True
    assert health[1]["default_return"] == 20.0


def test_recommend_env_switch_sensitive_axis_stays():
    slices = [{"param": "gravity", "flat": False, "ood_drop": 40.0}]
    advice = recommend_env_switch(slices, [], has_gravity_slice=True, hard_grid=True)
    assert advice["switch_env"] is False
    assert advice["action"] == "keep_pistonball_train_on_sensitive_axes"


def test_recommend_env_switch_without_gravity_asks_hard_eval():
    slices = [{"param": "ball_mass", "flat": True, "ood_drop": 0.1}]
    clones = [{"param": "ball_mass", "cloned": True}]
    advice = recommend_env_switch(slices, clones, has_gravity_slice=False, hard_grid=False)
    assert advice["switch_env"] is False
    assert advice["action"] == "re_eval_hard_grid_with_gravity"


def test_recommend_env_switch_hard_gravity_flat():
    slices = [
        {"param": "ball_mass", "flat": True, "ood_drop": 0.2},
        {"param": "gravity", "flat": True, "ood_drop": 1.0},
    ]
    advice = recommend_env_switch(slices, [], has_gravity_slice=True, hard_grid=True)
    assert advice["switch_env"] is True
    assert advice["action"] == "switch_to_multiwalker"
    assert advice["candidate"]["env"] == "pettingzoo.sisl.multiwalker_v9"


def test_diagnose_method_bundles_fields():
    order, meta = _mass_friction_meta()
    matrix = np.array(
        [
            [80.0, 10.0, 10.0, 50.0, 40.0, 70.0, 20.0],
            [70.0, 11.0, 11.0, 55.0, 35.0, 65.0, 25.0],
        ]
    )
    report = {
        "context_order": order,
        "context_meta": meta,
        "seeds": [1, 2],
        "eval_factor_grid": [0.25, 0.5, 1.0, 2.0, 4.0],
    }
    diag = diagnose_method(matrix, report)
    assert "clones" in diag and "slices" in diag and "seed_health" in diag
    assert diag["env_switch"]["switch_env"] is False
    assert "gravity" in {r["param"] for r in diag["slices"]}
