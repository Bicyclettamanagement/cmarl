"""Offline evaluation of saved MATD3 checkpoints on the ID/OOD context grid.

Training only runs a cheap in-distribution KPI eval. This module loads
``checkpoints/best.pt`` (falling back to ``latest.pt``) and rolls out the
frozen policy on the 1D context slice used for C.2-style plots and rliable
aggregation.

The underlying Pistonball env is created **once** and mutated in-place, matching
training, so a dense sweep does not leak host RAM.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from evaluation.context_sampling import (
    SWEEP_SEED_OFFSET,
    apply_context_to_env,
    context_spec_from_args,
    eval_points_1d,
    resolve_eval_grid,
    resolve_eval_vary_params,
    summarize_context_sweep,
)
from evaluation.rliable_utils import RUN_MANIFEST_FILENAME, save_rliable_scores

EVALUATION_RESULTS_FILENAME = "evaluation_results.json"
CONTEXT_CLIP_RESULTS_FILENAME = "evaluation_results_context_clip.json"
CONTEXT_CLIP_SCORES_FILENAME = "rliable_scores_context_clip.json"


def _tuple_fields(cls) -> set[str]:
    names: set[str] = set()
    for f in fields(cls):
        anno = f.type
        if anno is tuple or getattr(anno, "__origin__", None) is tuple:
            names.add(f.name)
        # PEP 604 / postponed annotations: ``tuple[float, ...]`` stored as string
        if isinstance(anno, str) and anno.startswith("tuple"):
            names.add(f.name)
    return names


def instantiate_args(cls, raw: dict) -> Any:
    """Rebuild an ``Args`` dataclass from a checkpoint / manifest dict."""
    valid = {f.name for f in fields(cls)}
    tuples = _tuple_fields(cls)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in valid:
            continue
        if key in tuples and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def detect_family(args_dict: dict) -> str:
    """Return ``multiwalker``, ``pistonball_context``, or ``pistonball``."""
    env = str(args_dict.get("env_name") or "")
    exp = str(args_dict.get("exp_name") or "")
    experiment = str(args_dict.get("experiment") or "")
    blob = f"{env} {exp} {experiment}".lower()
    if env == "multiwalker" or "multiwalker" in blob or experiment.startswith("mw"):
        return "multiwalker"
    if "context_mode" in args_dict or "pistonball_context" in exp:
        return "pistonball_context"
    return "pistonball"


def detect_algorithm(args_dict: dict) -> str:
    """Return ``context`` if the run used explicit context networks, else ``hidden``."""
    family = detect_family(args_dict)
    if family == "multiwalker":
        return "context" if str(args_dict.get("context_mode", "hidden")) == "concat" else "hidden"
    if family == "pistonball_context":
        return "context"
    exp = str(args_dict.get("exp_name", ""))
    if "pistonball_context" in exp:
        return "context"
    return "hidden"


def load_algorithm(args_dict: dict):
    family = detect_family(args_dict)
    if family == "multiwalker":
        import matd3_multiwalker as algo
    elif family == "pistonball_context":
        import matd3_pistonball_context as algo
    else:
        import matd3_pistonball as algo
    return algo


def _episode_returns_from_eval_history(run_dir: Path, global_step: int | None) -> list[float] | None:
    """Reuse train-time ID eval episodes recorded at the selected checkpoint."""
    history_path = run_dir / "eval_history.jsonl"
    if not history_path.exists() or global_step is None:
        return None
    matched: list[float] | None = None
    with open(history_path) as f:
        for line in f:
            rec = json.loads(line)
            if int(rec.get("global_step", -1)) != int(global_step):
                continue
            rets = rec.get("episode_returns")
            if isinstance(rets, list) and rets:
                matched = [float(x) for x in rets]
    return matched


def evaluate_run(
    run_dir: Path,
    *,
    eval_episodes: int | None = None,
    device: torch.device | None = None,
    checkpoint: str = "best",
    merge_train_id_eval: bool = False,
    eval_factor_grid: tuple[float, ...] | None = None,
    eval_grid_preset: str | None = None,
    eval_vary_params: tuple[str, ...] | None = None,
    clip_context_to_id: bool = False,
) -> dict:
    """Run the 1D ID/OOD sweep for one training run and persist artifacts.

    ``merge_train_id_eval`` is off by default: mixing the 3 training KPI
    episodes into ``default`` made C.2 x=1.0 incomparable to the other factors.
    """
    import matd3_pistonball as base

    run_dir = Path(run_dir)
    with open(run_dir / RUN_MANIFEST_FILENAME) as f:
        manifest = json.load(f)
    args_dict = dict(manifest.get("args") or {})
    family = detect_family(args_dict)
    algo = load_algorithm(args_dict)
    args = instantiate_args(algo.Args, args_dict)
    if clip_context_to_id and (
        family != "multiwalker" or getattr(args, "context_mode", None) != "concat"
    ):
        raise ValueError(
            "clip_context_to_id only applies to Multiwalker concat checkpoints "
            "(the simulator stays at the true physics; only the context vector is clipped)"
        )
    if eval_episodes is not None:
        args.eval_episodes = int(eval_episodes)
    if eval_factor_grid or eval_grid_preset:
        args.eval_factor_grid = resolve_eval_grid(preset=eval_grid_preset, factors=eval_factor_grid)
    vary_override = resolve_eval_vary_params(
        preset=eval_grid_preset,
        params=eval_vary_params,
        env_name=getattr(args, "env_name", None) or ("multiwalker" if family == "multiwalker" else None),
    )
    if vary_override:
        args.eval_vary_params = vary_override

    ckpt_dir = run_dir / "checkpoints"
    ckpt_path = ckpt_dir / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        fallback = ckpt_dir / "best.pt" if checkpoint != "best" else ckpt_dir / "latest.pt"
        if fallback.exists():
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"no checkpoint {ckpt_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() and getattr(args, "cuda", True) else "cpu")

    nets = algo.build_agents(args, device)
    ckpt = base.load_checkpoint_into_nets(ckpt_path, nets, device)
    ckpt_step = int(
        ckpt.get("global_step")
        or (manifest.get("best_eval") or {}).get("step")
        or 0
    )

    spec = context_spec_from_args(args)
    points = eval_points_1d(args)
    agents = nets["agents"]
    actors = nets["actors"]
    reference = spec.default
    is_contextual = detect_algorithm(args_dict) == "context"
    n_episodes = int(args.eval_episodes)
    eval_seed = int(args.seed) + SWEEP_SEED_OFFSET

    if family == "multiwalker":
        from envs.multiwalker import apply_context_to_env as apply_ctx
        from envs.multiwalker import make_env as make_eval_env
    else:
        apply_ctx = apply_context_to_env
        make_eval_env = base.make_env

    env = make_eval_env(args, reference)
    per_context: dict[str, dict] = {}
    scores_by_context: dict[str, list[float]] = {}
    try:
        for point in points:
            apply_ctx(env, point.context)
            if family == "multiwalker":
                stats = algo.evaluate(
                    actors,
                    args,
                    point.context,
                    agents,
                    device,
                    seed=eval_seed,
                    n_episodes=n_episodes,
                    env=env,
                    reference_context=reference,
                    clip_context_to_id=clip_context_to_id,
                )
            elif family == "pistonball_context":
                stats = algo.evaluate(
                    actors,
                    args,
                    point.context,
                    reference,
                    agents,
                    device,
                    seed=eval_seed,
                    n_episodes=n_episodes,
                    env=env,
                )
            else:
                stats = algo.evaluate(
                    actors,
                    args,
                    point.context,
                    agents,
                    device,
                    seed=eval_seed,
                    n_episodes=n_episodes,
                    env=env,
                )
            stats = dict(stats)
            stats["context"] = point.context
            stats.update(point.meta())
            per_context[point.name] = stats
            scores_by_context[point.name] = list(stats["episode_returns"])
    finally:
        env.close()

    if merge_train_id_eval:
        extra = _episode_returns_from_eval_history(run_dir, ckpt_step)
        if extra and "default" in scores_by_context:
            merged = extra + scores_by_context["default"]
            scores_by_context["default"] = merged
            from evaluation.rliable_utils import summarize_episode_scores

            summary = summarize_episode_scores(merged, reference_return=args.reference_return)
            per_context["default"].update(summary)
            per_context["default"]["episode_returns"] = merged
            per_context["default"]["n_episodes"] = float(len(merged))

    summary = summarize_context_sweep(per_context, points)
    method_tag = args.method_tag or args_dict.get("method_tag") or "unknown"
    context_order = [p.name for p in points]
    context_meta = spec.context_meta(points)

    results = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt_step,
        "algorithm": args.exp_name,
        "method_tag": method_tag,
        "seed": int(args.seed),
        "eval_episodes": n_episodes,
        "id_factor_low": spec.id_factor_low,
        "id_factor_high": spec.id_factor_high,
        "eval_factor_grid": list(spec.eval_factor_grid),
        "eval_vary_params": list(spec.eval_params),
        "eval_grid_preset": eval_grid_preset,
        "merge_train_id_eval": bool(merge_train_id_eval),
        "context_order": context_order,
        "context_meta": context_meta,
        "contexts": per_context,
        "physics_probes": {
            name: stats.get("physics_probe") for name, stats in per_context.items()
        },
        **summary,
    }
    if is_contextual:
        results["context_mode"] = getattr(args, "context_mode", None)
    if hasattr(args, "context_kind"):
        results["context_kind"] = args.context_kind
    results["env_name"] = getattr(args, "env_name", "pistonball")
    results["family"] = family
    results["context_clip_to_id"] = bool(clip_context_to_id)
    if clip_context_to_id:
        results["context_clip_low"] = spec.id_factor_low
        results["context_clip_high"] = spec.id_factor_high

    results_name = CONTEXT_CLIP_RESULTS_FILENAME if clip_context_to_id else EVALUATION_RESULTS_FILENAME
    scores_name = CONTEXT_CLIP_SCORES_FILENAME if clip_context_to_id else "rliable_scores.json"
    with open(run_dir / results_name, "w") as f:
        json.dump(results, f, indent=2)

    save_rliable_scores(
        run_dir,
        algorithm=args.exp_name,
        method_tag=method_tag,
        seed=int(args.seed),
        global_step=ckpt_step,
        context_order=context_order,
        scores_by_context=scores_by_context,
        eval_split="id_ood_1d_slice",
        reference_return=args.reference_return,
        extra={
            "context_meta": context_meta,
            "id_factor_low": spec.id_factor_low,
            "id_factor_high": spec.id_factor_high,
            "eval_factor_grid": list(spec.eval_factor_grid),
            "eval_vary_params": list(spec.eval_params),
            "eval_grid_preset": eval_grid_preset,
            "generalization_gap_mean": summary["generalization_gap_mean"],
            "generalization_gap_iqm": summary["generalization_gap_iqm"],
            "checkpoint": str(ckpt_path),
            "family": family,
            "env_name": getattr(args, "env_name", "pistonball"),
            "context_clip_to_id": bool(clip_context_to_id),
            "context_clip_low": spec.id_factor_low if clip_context_to_id else None,
            "context_clip_high": spec.id_factor_high if clip_context_to_id else None,
        },
        scores_filename=scores_name,
    )
    return results


def evaluate_experiment(
    runs_root: Path,
    exp_name: str,
    *,
    eval_episodes: int | None = None,
    seeds: list[int] | None = None,
    device: torch.device | None = None,
    checkpoint: str = "best",
    merge_train_id_eval: bool = False,
    eval_factor_grid: tuple[float, ...] | None = None,
    eval_grid_preset: str | None = None,
    eval_vary_params: tuple[str, ...] | None = None,
    clip_context_to_id: bool = False,
) -> list[dict]:
    """Evaluate every discovered run for ``exp_name``."""
    from evaluation.rliable_utils import discover_run_dirs

    run_dirs = discover_run_dirs(Path(runs_root), exp_name, seeds=seeds, require_finished=True)
    if not run_dirs:
        # Finished flag may be true even without post-hoc eval; also accept unfinished
        run_dirs = discover_run_dirs(Path(runs_root), exp_name, seeds=seeds, require_finished=False)
    if not run_dirs:
        raise FileNotFoundError(f"no runs for exp_name={exp_name!r} under {runs_root}")
    return [
        evaluate_run(
            d,
            eval_episodes=eval_episodes,
            device=device,
            checkpoint=checkpoint,
            merge_train_id_eval=merge_train_id_eval,
            eval_factor_grid=eval_factor_grid,
            eval_grid_preset=eval_grid_preset,
            eval_vary_params=eval_vary_params,
            clip_context_to_id=clip_context_to_id,
        )
        for d in run_dirs
    ]
