"""Clone detection, C.2 trapezoid checks, seed health, and env-switch advice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PHYSICS_SLICE_PARAMS: tuple[str, ...] = (
    "ball_mass",
    "ball_friction",
    "ball_elasticity",
    "gravity",
    "package_mass",
    "terrain_friction",
)

# Return drop below this (ID mean − OOD mean) counts as a flat slice.
FLAT_DROP_THRESHOLD = 5.0
# Per-seed return below this on the default context is a collapsed run.
COLLAPSE_RETURN_THRESHOLD = 45.0
# Max unique returns along a 1D slice (across factors, per seed) to call it a clone.
CLONE_UNIQUE_MAX = 1


def detect_clone_groups(
    matrix: np.ndarray,
    context_order: list[str],
    context_meta: dict[str, dict] | None = None,
    *,
    atol: float = 1e-4,
) -> list[dict[str, Any]]:
    """Find 1D slices where every factor yields the same per-seed return vector.

    This is the mass/elasticity pattern: ``ball_mass_x0.5`` … ``ball_mass_x2``
    identical for each seed, which makes ID vs OOD gaps uninformative.
    """
    meta = context_meta or {}
    by_param: dict[str, list[int]] = {}
    for i, name in enumerate(context_order):
        param = (meta.get(name) or {}).get("varied_param")
        if not param:
            continue
        by_param.setdefault(str(param), []).append(i)

    groups: list[dict[str, Any]] = []
    for param, idxs in by_param.items():
        if len(idxs) < 2:
            continue
        block = matrix[:, idxs]
        # A slice is cloned if every seed has (almost) one unique value across factors.
        per_seed_unique = [int(np.unique(np.round(block[s], decimals=4)).size) for s in range(block.shape[0])]
        cloned_seeds = sum(1 for n in per_seed_unique if n <= CLONE_UNIQUE_MAX)
        max_pair_diff = float(np.max(np.abs(block - block[:, :1])))
        cloned = cloned_seeds == block.shape[0] or max_pair_diff <= atol
        groups.append(
            {
                "param": param,
                "contexts": [context_order[i] for i in idxs],
                "cloned": bool(cloned),
                "cloned_seed_count": int(cloned_seeds),
                "n_seeds": int(block.shape[0]),
                "n_factors": int(len(idxs)),
                "max_abs_diff": max_pair_diff,
                "per_seed_unique_returns": per_seed_unique,
            }
        )
    return groups


def slice_trapezoid_stats(
    matrix: np.ndarray,
    context_order: list[str],
    context_meta: dict[str, dict] | None = None,
    *,
    flat_drop: float = FLAT_DROP_THRESHOLD,
) -> list[dict[str, Any]]:
    """Per-parameter ID vs OOD drop used to verify a C.2 trapezoid."""
    meta = context_meta or {}
    by_param: dict[str, list[int]] = {}
    for i, name in enumerate(context_order):
        info = meta.get(name) or {}
        param = info.get("varied_param")
        if not param:
            continue
        by_param.setdefault(str(param), []).append(i)

    rows: list[dict[str, Any]] = []
    means = np.mean(matrix, axis=0)
    default_idx = next((i for i, n in enumerate(context_order) if n in ("default", "train")), None)
    default_is_id = False
    if default_idx is not None:
        default_split = str((meta.get(context_order[default_idx]) or {}).get("split") or "id")
        default_is_id = default_split == "id"

    for param, idxs in by_param.items():
        id_idx = [i for i in idxs if str((meta.get(context_order[i]) or {}).get("split")) == "id"]
        if default_idx is not None and default_is_id and default_idx not in id_idx:
            id_idx = [default_idx, *id_idx]
        ood_idx = [i for i in idxs if str((meta.get(context_order[i]) or {}).get("split")) == "ood"]
        id_mean = float(np.mean(means[id_idx])) if id_idx else float("nan")
        ood_mean = float(np.mean(means[ood_idx])) if ood_idx else float("nan")
        drop = id_mean - ood_mean
        factor_idx = list(idxs)
        if default_idx is not None and default_idx not in factor_idx:
            factor_idx = [default_idx, *factor_idx]
        factors = [float((meta.get(context_order[i]) or {}).get("factor", 1.0)) for i in factor_idx]
        returns = [float(means[i]) for i in factor_idx]
        # Expected trapezoid: return falls as |log factor| grows.
        logdev = np.abs(np.log(np.clip(np.asarray(factors, dtype=float), 1e-6, None)))
        corr = float("nan")
        if np.std(logdev) > 1e-9 and np.std(returns) > 1e-9:
            corr = float(np.corrcoef(logdev, returns)[0, 1])
        rows.append(
            {
                "param": param,
                "id_return_mean": id_mean,
                "ood_return_mean": ood_mean,
                "ood_drop": drop,
                "flat": bool(ood_idx) and (not np.isfinite(drop) or abs(drop) < flat_drop),
                "logdev_return_corr": corr,
                "n_id": len(id_idx),
                "n_ood": len(ood_idx),
            }
        )
    return rows


def seed_health(
    matrix: np.ndarray,
    context_order: list[str],
    seeds: list[int] | None = None,
    *,
    collapse_threshold: float = COLLAPSE_RETURN_THRESHOLD,
) -> list[dict[str, Any]]:
    """Per-seed mean return and collapse flag (failed training run)."""
    default_idx = next((i for i, n in enumerate(context_order) if n in ("default", "train")), 0)
    ids = list(seeds) if seeds is not None else list(range(1, matrix.shape[0] + 1))
    rows: list[dict[str, Any]] = []
    for s, seed in enumerate(ids[: matrix.shape[0]]):
        mean_all = float(np.mean(matrix[s]))
        default_ret = float(matrix[s, default_idx])
        rows.append(
            {
                "seed": int(seed),
                "default_return": default_ret,
                "mean_return_all_contexts": mean_all,
                "min_return": float(np.min(matrix[s])),
                "max_return": float(np.max(matrix[s])),
                "collapsed": default_ret < collapse_threshold,
            }
        )
    return rows


def collect_coordination(
    run_dirs: list[Path],
    results_filename: str = "evaluation_results.json",
) -> dict[str, float]:
    """Mean per-agent reward variance / success from evaluation_results.json if present."""
    vars_: list[float] = []
    success: list[float] = []
    lengths: list[float] = []
    for run_dir in run_dirs:
        path = Path(run_dir) / results_filename
        if not path.exists():
            continue
        with open(path) as f:
            payload = json.load(f)
        for stats in (payload.get("contexts") or {}).values():
            if "per_agent_return_variance_mean" in stats:
                vars_.append(float(stats["per_agent_return_variance_mean"]))
            if "success_rate" in stats:
                success.append(float(stats["success_rate"]))
            if "length_mean" in stats:
                lengths.append(float(stats["length_mean"]))
    out: dict[str, float] = {}
    if vars_:
        out["per_agent_return_variance_mean"] = float(np.mean(vars_))
    if success:
        out["success_rate_mean"] = float(np.mean(success))
    if lengths:
        out["episode_length_mean"] = float(np.mean(lengths))
    return out


def recommend_env_switch(
    slice_rows: list[dict[str, Any]],
    clone_groups: list[dict[str, Any]],
    *,
    has_gravity_slice: bool,
    hard_grid: bool,
) -> dict[str, Any]:
    """What to do if C.2 slices stay flat (step 4)."""
    cloned = {g["param"] for g in clone_groups if g.get("cloned")}
    flat = {r["param"] for r in slice_rows if r.get("flat")}
    sensitive = [
        r["param"] for r in slice_rows if not r.get("flat") and np.isfinite(r.get("ood_drop", np.nan))
    ]
    if sensitive:
        return {
            "switch_env": False,
            "reason": "at_least_one_physics_axis_is_sensitive",
            "sensitive_params": sensitive,
            "action": "keep_pistonball_train_on_sensitive_axes",
        }
    if not has_gravity_slice:
        return {
            "switch_env": False,
            "reason": "mass_elasticity_likely_cloned_or_flat",
            "cloned_params": sorted(cloned),
            "flat_params": sorted(flat),
            "action": "re_eval_hard_grid_with_gravity",
        }
    if not hard_grid:
        return {
            "switch_env": False,
            "reason": "gravity_or_standard_grid_still_easy",
            "action": "re_eval_hard_factor_grid",
        }
    return {
        "switch_env": True,
        "reason": "pistonball_physics_ood_still_flat",
        "cloned_params": sorted(cloned),
        "flat_params": sorted(flat),
        "action": "switch_to_multiwalker",
        "candidate": {
            "env": "pettingzoo.sisl.multiwalker_v9",
            "why": "continuous MARL with explicit body physics; gravity/friction/package mass change the task",
            "cost": "vector observations (not pixels); new actor/critic input shape",
        },
    }


def diagnose_method(
    matrix: np.ndarray,
    report: dict[str, Any],
    run_dirs: list[Path] | None = None,
    results_filename: str = "evaluation_results.json",
) -> dict[str, Any]:
    """Bundle clone / trapezoid / seed / coordination diagnostics for one method."""
    order = list(report.get("context_order") or [])
    meta = report.get("context_meta") or {}
    seeds = report.get("seeds")
    clones = detect_clone_groups(matrix, order, meta)
    slices = slice_trapezoid_stats(matrix, order, meta)
    health = seed_health(matrix, order, seeds)
    grid = [float(x) for x in (report.get("eval_factor_grid") or [])]
    hard_grid = any(f <= 0.3 or f >= 3.0 for f in grid) if grid else False
    has_gravity = any(r["param"] == "gravity" for r in slices)
    coord = collect_coordination(run_dirs or [], results_filename=results_filename) if run_dirs else {}
    return {
        "clones": clones,
        "slices": slices,
        "seed_health": health,
        "n_collapsed_seeds": sum(1 for r in health if r["collapsed"]),
        "coordination": coord,
        "env_switch": recommend_env_switch(
            slices, clones, has_gravity_slice=has_gravity, hard_grid=hard_grid
        ),
    }
