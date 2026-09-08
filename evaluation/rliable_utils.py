"""rliable-based robust evaluation utilities for contextual-MARL experiments.

Aligns with the evaluation protocol in extended_abstract.tex (performance,
zero-shot generalization, sample efficiency) and with rliable recommendations:
IQM + stratified bootstrap CIs, optimality gap, probability of improvement,
and performance profiles.

Score matrices are shaped ``(num_runs, num_tasks)`` where *runs* are independent
training seeds and *tasks* are evaluation contexts (in-distribution + OOD).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from rliable import library as rly
    from rliable import metrics as rly_metrics
except ImportError as e:  # pragma: no cover - optional at train time
    rly = None
    rly_metrics = None
    _RLIABLE_IMPORT_ERROR = e
else:
    _RLIABLE_IMPORT_ERROR = None

RLIABLE_SCORES_FILENAME = "rliable_scores.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"

# Common N=5 experiment presets (2×2: static/varying × hidden/explicit context).
EXPERIMENT_PRESETS: dict[str, dict[str, str]] = {
    "baseline5_static": {
        "exp_name": "baseline5_static_randdrop",
        "method_tag": "hidden_context_static",
        "label": "Hidden context (static)",
        "algorithm": "matd3_pistonball",
    },
    "baseline5_varying": {
        "exp_name": "baseline5_varying_randdrop",
        "method_tag": "hidden_context_varying",
        "label": "Hidden context (varying)",
        "algorithm": "matd3_pistonball",
    },
    "context5_static": {
        "exp_name": "context5_concat_static_randdrop",
        "method_tag": "context_concat_static",
        "label": "Context concat (static)",
        "algorithm": "matd3_pistonball_context",
    },
    "context5_varying": {
        "exp_name": "context5_concat_varying_randdrop",
        "method_tag": "context_concat_varying",
        "label": "Context concat (varying)",
        "algorithm": "matd3_pistonball_context",
    },
    # Legacy aliases (constant-context training runs)
    "baseline5": {
        "exp_name": "baseline5_shared_randdrop",
        "method_tag": "hidden_context",
        "label": "Hidden context",
        "algorithm": "matd3_pistonball",
    },
    "context5": {
        "exp_name": "context5_concat_randdrop",
        "method_tag": "context_concat",
        "label": "Context concat",
        "algorithm": "matd3_pistonball_context",
    },
}


def require_rliable() -> None:
    if rly is None or rly_metrics is None:
        raise ImportError(
            "rliable is required for robust aggregation. Install with: pip install rliable"
        ) from _RLIABLE_IMPORT_ERROR


def interquartile_mean(scores: np.ndarray) -> float:
    """IQM over a 1D array (e.g. eval episodes within one run)."""
    require_rliable()
    return float(rly_metrics.aggregate_iqm(np.asarray(scores, dtype=np.float64)))


def summarize_episode_scores(
    episode_returns: list[float],
    *,
    reference_return: float | None = None,
) -> dict[str, float]:
    """Summarize raw eval episodes for one (seed, context) pair."""
    scores = np.asarray(episode_returns, dtype=np.float64)
    if scores.size == 0:
        return {
            "return_mean": 0.0,
            "return_std": 0.0,
            "return_median": 0.0,
            "return_iqm": 0.0,
            "optimality_gap": 0.0,
            "n_episodes": 0.0,
        }
    out = {
        "return_mean": float(np.mean(scores)),
        "return_std": float(np.std(scores)),
        "return_median": float(np.median(scores)),
        "n_episodes": float(scores.size),
    }
    if rly_metrics is not None:
        out["return_iqm"] = float(rly_metrics.aggregate_iqm(scores))
        gamma = reference_return if reference_return is not None else float(np.max(scores))
        out["optimality_gap"] = float(rly_metrics.aggregate_optimality_gap(scores, gamma=gamma))
    else:
        out["return_iqm"] = out["return_mean"]
        out["optimality_gap"] = 0.0
    return out


def save_rliable_scores(
    run_dir: Path,
    *,
    algorithm: str,
    method_tag: str,
    seed: int,
    global_step: int,
    context_order: list[str],
    scores_by_context: dict[str, list[float]],
    eval_split: str,
    reference_return: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist per-episode returns for cross-seed rliable aggregation."""
    per_context = {}
    for name in context_order:
        episodes = scores_by_context.get(name, [])
        per_context[name] = {
            "episode_returns": episodes,
            **summarize_episode_scores(episodes, reference_return=reference_return),
        }
    payload = {
        "algorithm": algorithm,
        "method_tag": method_tag,
        "seed": seed,
        "global_step": global_step,
        "eval_split": eval_split,
        "context_order": context_order,
        "reference_return": reference_return,
        "scores_by_context": per_context,
        **(extra or {}),
    }
    path = run_dir / RLIABLE_SCORES_FILENAME
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_rliable_scores(run_dir: Path) -> dict | None:
    path = run_dir / RLIABLE_SCORES_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def parse_seed_from_run_name(run_name: str) -> int | None:
    """Extract training seed from a run folder name."""
    for part in run_name.split("__"):
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    # Legacy layout: pistonball__<exp>__<seed>__<timestamp>
    parts = run_name.split("__")
    if len(parts) >= 4 and parts[0] == "pistonball":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def _run_matches_experiment(run_name: str, exp_name: str) -> bool:
    """Match current and legacy run directory naming schemes."""
    return run_name.startswith(f"{exp_name}__") or run_name.startswith(f"pistonball__{exp_name}__")


def discover_run_dirs(
    runs_root: Path,
    exp_name: str,
    seeds: list[int] | None = None,
    *,
    method_tag: str | None = None,
    require_finished: bool = False,
) -> list[Path]:
    """Find run directories for an experiment (newest per seed if duplicates).

    Supports folder names like ``baseline5_shared_randdrop__shared__...__seed3__<ts>``
    and legacy ``pistonball__matd3_pistonball__3__<ts>``.
    """
    if not runs_root.is_dir():
        return []
    candidates: dict[int, Path] = {}
    for child in runs_root.iterdir():
        if not child.is_dir() or not _run_matches_experiment(child.name, exp_name):
            continue
        seed = parse_seed_from_run_name(child.name)
        if seed is None:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if method_tag is not None:
            data = load_rliable_scores(child)
            if data is None or data.get("method_tag") != method_tag:
                continue
        if require_finished:
            manifest_path = child / RUN_MANIFEST_FILENAME
            if not manifest_path.exists():
                continue
            with open(manifest_path) as f:
                manifest = json.load(f)
            if not manifest.get("finished", False):
                continue
        prev = candidates.get(seed)
        if prev is None or child.name > prev.name:
            candidates[seed] = child
    return [candidates[s] for s in sorted(candidates)]


def build_score_matrix(
    run_dirs: list[Path],
    *,
    score_key: str = "episode_returns",
    use_iqm_per_run: bool = False,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Build ``(num_runs, num_tasks)`` matrix from saved rliable score files."""
    rows: list[list[float]] = []
    context_order: list[str] | None = None
    seeds: list[int] = []
    for run_dir in run_dirs:
        data = load_rliable_scores(run_dir)
        if data is None:
            continue
        if context_order is None:
            context_order = list(data["context_order"])
        seeds.append(int(data["seed"]))
        row = []
        for ctx in context_order:
            episodes = data["scores_by_context"][ctx][score_key]
            if use_iqm_per_run and rly_metrics is not None and len(episodes) > 0:
                row.append(float(rly_metrics.aggregate_iqm(np.asarray(episodes, dtype=np.float64))))
            else:
                row.append(float(np.mean(episodes)) if episodes else float("nan"))
        rows.append(row)
    if context_order is None or not rows:
        raise ValueError("No rliable_scores.json files found in the given run directories.")
    return np.asarray(rows, dtype=np.float64), context_order, seeds


def split_id_ood(context_order: list[str]) -> tuple[list[int], list[int]]:
    id_idx = [i for i, name in enumerate(context_order) if name == "train"]
    ood_idx = [i for i, name in enumerate(context_order) if name != "train"]
    return id_idx, ood_idx


def compute_generalization_gap(matrix: np.ndarray, context_order: list[str]) -> dict[str, float]:
    """In-distribution vs OOD gap (mean over runs), per abstract §Evaluation."""
    id_idx, ood_idx = split_id_ood(context_order)
    if not id_idx or not ood_idx:
        return {
            "id_return_mean": float("nan"),
            "ood_return_mean": float("nan"),
            "generalization_gap": float("nan"),
            "id_return_iqm": float("nan"),
            "ood_return_iqm": float("nan"),
            "generalization_gap_iqm": float("nan"),
        }
    id_col = matrix[:, id_idx]
    ood_cols = matrix[:, ood_idx]
    id_mean = float(np.mean(id_col))
    ood_mean = float(np.mean(ood_cols))
    out = {
        "id_return_mean": id_mean,
        "ood_return_mean": ood_mean,
        "generalization_gap": id_mean - ood_mean,
    }
    if rly_metrics is not None:
        id_iqm = float(rly_metrics.aggregate_iqm(id_col.reshape(-1)))
        ood_iqm = float(rly_metrics.aggregate_iqm(ood_cols.reshape(-1)))
        out["id_return_iqm"] = id_iqm
        out["ood_return_iqm"] = ood_iqm
        out["generalization_gap_iqm"] = id_iqm - ood_iqm
    return out


def compute_per_context_stats(
    matrix: np.ndarray,
    context_order: list[str],
) -> dict[str, dict[str, float]]:
    """Per-context summary across seeds (mean return + IQM over seeds)."""
    stats: dict[str, dict[str, float]] = {}
    for idx, ctx in enumerate(context_order):
        col = matrix[:, idx]
        entry = {
            "return_mean": float(np.mean(col)),
            "return_std": float(np.std(col)),
            "return_median": float(np.median(col)),
            "n_seeds": float(col.size),
        }
        if rly_metrics is not None:
            entry["return_iqm"] = float(rly_metrics.aggregate_iqm(col))
        stats[ctx] = entry
    return stats


def subset_score_matrix(
    matrix: np.ndarray,
    context_order: list[str],
    context_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Select columns by context name."""
    indices = [context_order.index(name) for name in context_names]
    return matrix[:, indices], context_names


def resolve_experiment_config(
    *,
    preset: str | None,
    exp_name: str | None,
    method_tag: str | None,
    label: str | None,
) -> dict[str, str]:
    """Merge CLI preset overrides with explicit flags."""
    if preset is not None:
        if preset not in EXPERIMENT_PRESETS:
            known = ", ".join(sorted(EXPERIMENT_PRESETS))
            raise ValueError(f"Unknown preset {preset!r}; choose from: {known}")
        base = dict(EXPERIMENT_PRESETS[preset])
    else:
        if not exp_name:
            raise ValueError("Provide --exp-name or --preset.")
        base = {
            "exp_name": exp_name,
            "method_tag": method_tag or "",
            "label": label or exp_name,
            "algorithm": exp_name,
        }
    if exp_name:
        base["exp_name"] = exp_name
    if method_tag:
        base["method_tag"] = method_tag
    if label:
        base["label"] = label
    if not base.get("method_tag"):
        base["method_tag"] = ""
    if not base.get("label"):
        base["label"] = base["exp_name"]
    return base


def aggregate_with_rliable(
    scores_dict: dict[str, np.ndarray],
    *,
    reference_return: float | None = None,
    bootstrap_reps: int = 5000,
    profile_thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute rliable aggregate metrics + stratified bootstrap CIs for each algorithm."""
    require_rliable()
    aggregate_func = lambda x: np.array(
        [
            rly_metrics.aggregate_median(x),
            rly_metrics.aggregate_iqm(x),
            rly_metrics.aggregate_mean(x),
            rly_metrics.aggregate_optimality_gap(
                x, gamma=reference_return if reference_return is not None else np.max(x)
            ),
        ]
    )
    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        scores_dict, aggregate_func, reps=bootstrap_reps
    )
    metric_names = ["median", "iqm", "mean", "optimality_gap"]
    report: dict[str, Any] = {"algorithms": {}, "bootstrap_reps": bootstrap_reps}
    for algo in scores_dict:
        vals = aggregate_scores[algo]
        cis = aggregate_cis[algo]
        report["algorithms"][algo] = {
            "score_matrix_shape": list(scores_dict[algo].shape),
            "metrics": {
                name: {
                    "point": float(vals[i]),
                    "ci_low": float(cis[0, i]),
                    "ci_high": float(cis[1, i]),
                }
                for i, name in enumerate(metric_names)
            },
        }
    if profile_thresholds is not None:
        profiles, profile_cis = rly.create_performance_profile(scores_dict, profile_thresholds)
        report["performance_profiles"] = {
            algo: {
                "thresholds": profile_thresholds.tolist(),
                "profile": profiles[algo].tolist(),
                "ci_low": profile_cis[algo][0].tolist(),
                "ci_high": profile_cis[algo][1].tolist(),
            }
            for algo in scores_dict
        }
    return report


def compare_probability_of_improvement(
    scores_x: np.ndarray,
    scores_y: np.ndarray,
    *,
    bootstrap_reps: int = 2000,
) -> dict[str, float]:
    """P(algorithm X > algorithm Y) with stratified bootstrap CI (rliable)."""
    require_rliable()
    pairs = {"x_vs_y": (scores_x, scores_y)}
    prob, prob_cis = rly.get_interval_estimates(pairs, rly_metrics.probability_of_improvement, reps=bootstrap_reps)
    return {
        "probability_of_improvement": float(prob["x_vs_y"]),
        "ci_low": float(prob_cis["x_vs_y"][0, 0]),
        "ci_high": float(prob_cis["x_vs_y"][1, 0]),
    }


def load_sample_efficiency_matrix(
    run_dirs: list[Path],
    *,
    metric: str = "return_mean",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build ``(num_runs, num_checkpoints)`` from eval_history.jsonl (train ID eval only)."""
    rows: list[list[float]] = []
    steps_ref: list[int] | None = None
    seeds: list[int] = []
    for run_dir in run_dirs:
        manifest_path = run_dir / RUN_MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        seeds.append(int(manifest["args"]["seed"]))
        history_path = run_dir / "eval_history.jsonl"
        if not history_path.exists():
            continue
        by_step: dict[int, float] = {}
        with open(history_path) as f:
            for line in f:
                rec = json.loads(line)
                by_step[int(rec["global_step"])] = float(rec[metric])
        steps = sorted(by_step)
        if steps_ref is None:
            steps_ref = steps
        row = [by_step.get(s, float("nan")) for s in steps_ref]
        rows.append(row)
    if steps_ref is None or not rows:
        raise ValueError("No eval_history.jsonl found for sample-efficiency aggregation.")
    return np.asarray(rows, dtype=np.float64), np.asarray(steps_ref, dtype=np.int64), seeds


def aggregate_sample_efficiency_iqm(
    run_dirs: list[Path],
    *,
    bootstrap_reps: int = 5000,
) -> dict[str, Any]:
    """IQM learning curves with bootstrap CIs across seeds (rliable)."""
    require_rliable()
    matrix, steps, seeds = load_sample_efficiency_matrix(run_dirs)
    # Wrap as pseudo-algorithm dict; each column is a "task" = checkpoint (stratified over runs×tasks).
    scores_dict = {"policy": matrix}

    def iqm_over_checkpoints(scores: np.ndarray) -> np.ndarray:
        return np.array([rly_metrics.aggregate_iqm(scores[:, i]) for i in range(scores.shape[1])])

    iqm_curve, iqm_cis = rly.get_interval_estimates(scores_dict, iqm_over_checkpoints, reps=bootstrap_reps)
    return {
        "seeds": seeds,
        "global_steps": steps.tolist(),
        "iqm": iqm_curve["policy"].tolist(),
        "ci_low": iqm_cis["policy"][0].tolist(),
        "ci_high": iqm_cis["policy"][1].tolist(),
    }


def collect_seed_stability(run_dirs: list[Path], context_order: list[str]) -> dict[str, float]:
    """Convergence variability across seeds (final transfer ID return)."""
    matrix, _, _ = build_score_matrix(run_dirs)
    id_idx, _ = split_id_ood(context_order)
    if not id_idx:
        return {"seed_return_std": float("nan"), "seed_return_iqm": float("nan")}
    id_col = matrix[:, id_idx[0]]
    out = {"seed_return_std": float(np.std(id_col)), "seed_return_mean": float(np.mean(id_col))}
    if rly_metrics is not None:
        out["seed_return_iqm"] = float(rly_metrics.aggregate_iqm(id_col))
    return out


def median_steps_to_threshold(run_dirs: list[Path]) -> dict[str, float | None]:
    """Median steps-to-threshold across seeds (abstract: sample efficiency)."""
    values = []
    for run_dir in run_dirs:
        manifest_path = run_dir / RUN_MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        val = manifest.get("steps_to_threshold")
        if val is not None:
            values.append(float(val))
    if not values:
        return {"median_steps_to_threshold": None, "n_seeds_reached": 0}
    return {
        "median_steps_to_threshold": float(np.median(values)),
        "n_seeds_reached": len(values),
    }
