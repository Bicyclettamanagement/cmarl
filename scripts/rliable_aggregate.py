"""Aggregate multi-seed experiment outputs with rliable (robust evaluation)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import tyro

from evaluation.rliable_utils import (
    aggregate_sample_efficiency_iqm,
    aggregate_with_rliable,
    build_score_matrix,
    collect_seed_stability,
    compare_probability_of_improvement,
    compute_generalization_gap,
    discover_run_dirs,
    median_steps_to_threshold,
)


@dataclass
class Args:
    runs_root: str = "runs"
    """Root directory containing training runs"""
    exp_name: str = "matd3_pistonball"
    """Experiment name embedded in run folder names"""
    seeds: tuple[int, ...] | None = None
    """Optional subset of seeds; default = all found"""
    reference_return: float | None = 100.0
    """Optimality-gap reference (typical successful Pistonball return scale)"""
    bootstrap_reps: int = 5000
    """Stratified bootstrap repetitions for aggregate metrics"""
    compare_exp_name: str | None = None
    """If set, compute probability of improvement vs this second experiment"""
    output_dir: str = "reports"
    """Directory for JSON reports"""
    profile_min: float = -50.0
    """Performance profile threshold range (return)"""
    profile_max: float = 100.0
    profile_steps: int = 31


def main(args: Args) -> Path:
    runs_root = Path(args.runs_root)
    seeds = list(args.seeds) if args.seeds is not None else None
    run_dirs = discover_run_dirs(runs_root, args.exp_name, seeds)
    if not run_dirs:
        raise SystemExit(f"No runs found for exp_name={args.exp_name!r} under {runs_root}")

    matrix, context_order, found_seeds = build_score_matrix(run_dirs)
    scores_dict = {args.exp_name: matrix}
    thresholds = np.linspace(args.profile_min, args.profile_max, args.profile_steps)

    report = {
        "exp_name": args.exp_name,
        "seeds": found_seeds,
        "context_order": context_order,
        "generalization": compute_generalization_gap(matrix, context_order),
        "seed_stability": collect_seed_stability(run_dirs, context_order),
        "sample_efficiency": {
            "median_steps_to_threshold": median_steps_to_threshold(run_dirs),
        },
    }
    try:
        report["sample_efficiency"]["eval_iqm_curve"] = aggregate_sample_efficiency_iqm(
            run_dirs, bootstrap_reps=args.bootstrap_reps
        )
    except ValueError as exc:
        report["sample_efficiency"]["eval_iqm_curve"] = {"error": str(exc)}

    report["rliable"] = aggregate_with_rliable(
        scores_dict,
        reference_return=args.reference_return,
        bootstrap_reps=args.bootstrap_reps,
        profile_thresholds=thresholds,
    )

    if args.compare_exp_name is not None:
        compare_dirs = discover_run_dirs(runs_root, args.compare_exp_name, seeds)
        matrix_b, context_b, seeds_b = build_score_matrix(compare_dirs)
        if context_b != context_order:
            raise ValueError("Context order mismatch between experiments; cannot compare.")
        report["comparison"] = {
            "baseline": args.exp_name,
            "challenger": args.compare_exp_name,
            "probability_of_improvement": compare_probability_of_improvement(
                matrix, matrix_b, bootstrap_reps=min(args.bootstrap_reps, 2000)
            ),
            "challenger_seeds": seeds_b,
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rliable_{args.exp_name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")
    iqm = report["rliable"]["algorithms"][args.exp_name]["metrics"]["iqm"]
    print(
        f"IQM return [{iqm['ci_low']:.2f}, {iqm['ci_high']:.2f}] "
        f"(point {iqm['point']:.2f}) over {matrix.shape[0]} seeds × {matrix.shape[1]} contexts"
    )
    gap = report["generalization"]
    print(f"Generalization gap (ID-OOD): {gap['generalization_gap']:.3f}")
    return out_path


if __name__ == "__main__":
    main(tyro.cli(Args))
