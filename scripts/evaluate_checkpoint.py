#!/usr/bin/env python3
"""Run the dense 1D ID/OOD evaluation sweep on trained checkpoints.

Training only records a cheap in-distribution KPI. After a run (or a seed sweep)
finishes, call this to produce ``evaluation_results.json`` and
``rliable_scores.json`` for aggregation / C.2-style plots.

Examples
--------
    python scripts/evaluate_checkpoint.py --run-dir runs/pistonball/n5/<run>
    python scripts/evaluate_checkpoint.py --exp-name baseline5_varying_randdrop --eval-episodes 10
    python scripts/evaluate_checkpoint.py --exp-name baseline5_varying_randdrop --grid-preset hard
    python scripts/evaluate_checkpoint.py --exp-name mw3_hidden_varying --grid-preset hard
    # ``--runs-root runs`` still finds nested dirs; archive is separate: --runs-root old_runs
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import tyro

from evaluation.checkpoint_eval import evaluate_experiment, evaluate_run


@dataclass
class Args:
    run_dir: str | None = None
    """Single run directory containing checkpoints/ and run_manifest.json"""

    runs_root: str = "runs"
    """Root used with --exp-name"""

    exp_name: str | None = None
    """Evaluate every matching run (uses finished manifests when present)"""

    seeds: tuple[int, ...] | None = None
    """Optional seed subset when using --exp-name"""

    eval_episodes: int = 10
    """Episodes per evaluation context (advanced eval; more than training KPIs)"""

    checkpoint: str = "best"
    """Checkpoint stem: best or latest"""

    grid_preset: str | None = None
    """Eval factor grid shortcut: ``standard`` (0.5–2×) or ``hard`` (0.25–4× + gravity)"""

    eval_factor_grid: tuple[float, ...] | None = None
    """Override 1D eval factors (takes precedence over --grid-preset)"""

    eval_vary_params: tuple[str, ...] | None = None
    """Physics axes to sweep. ``hard`` defaults to mass/friction/elasticity/gravity."""

    merge_train_id_eval: bool = False
    """If True, prepend training KPI episodes into the ``default`` context scores"""

    clip_context_to_id: bool = False
    """Multiwalker concat only: clamp the context vector to the ID factor window.
    The simulator still uses the true OOD physics. Writes ``*_context_clip.json``
    next to the existing scores and does not overwrite them."""

    cuda: bool = True


def _eval_kwargs(args: Args) -> dict:
    return {
        "eval_episodes": args.eval_episodes,
        "checkpoint": args.checkpoint,
        "merge_train_id_eval": args.merge_train_id_eval,
        "eval_factor_grid": args.eval_factor_grid,
        "eval_grid_preset": args.grid_preset,
        "eval_vary_params": args.eval_vary_params,
        "clip_context_to_id": args.clip_context_to_id,
    }


def main(args: Args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    kwargs = _eval_kwargs(args)
    if args.run_dir:
        results = evaluate_run(Path(args.run_dir), device=device, **kwargs)
        print(
            f"{results['run_dir']}: ID={results['id_return_mean']:.2f} "
            f"OOD={results['ood_return_mean']:.2f} gap={results['generalization_gap']:.2f} "
            f"grid={results.get('eval_factor_grid')} axes={results.get('eval_vary_params')}"
        )
        return
    if not args.exp_name:
        raise SystemExit("Provide --run-dir or --exp-name")
    all_results = evaluate_experiment(
        Path(args.runs_root),
        args.exp_name,
        seeds=list(args.seeds) if args.seeds else None,
        device=device,
        **kwargs,
    )
    for results in all_results:
        print(
            f"seed={results['seed']}: ID={results['id_return_mean']:.2f} "
            f"OOD={results['ood_return_mean']:.2f} gap={results['generalization_gap']:.2f}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
