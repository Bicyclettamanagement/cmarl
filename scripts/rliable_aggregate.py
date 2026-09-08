"""Aggregate multi-seed experiment outputs with rliable (robust evaluation).

Designed for the contextual-MARL Pistonball study:
  - hidden context baseline (`baseline5` preset)
  - explicit context concat (`context5` preset)
  - head-to-head comparison with probability of improvement

Follows rliable recommendations: IQM as primary metric, stratified bootstrap CIs,
performance profiles, and optimality gap vs a reference return.
"""

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
    EXPERIMENT_PRESETS,
    aggregate_sample_efficiency_iqm,
    aggregate_with_rliable,
    build_score_matrix,
    collect_seed_stability,
    compare_probability_of_improvement,
    compute_generalization_gap,
    compute_per_context_stats,
    discover_run_dirs,
    median_steps_to_threshold,
    resolve_experiment_config,
    split_id_ood,
    subset_score_matrix,
)


@dataclass
class Args:
    runs_root: str = "runs"
    """Root directory containing training runs"""

    preset: str | None = None
    """Shortcut: ``baseline5`` or ``context5`` (sets exp_name + method_tag)"""

    exp_name: str | None = None
    """Experiment tag embedded in run folder names (overrides preset)"""

    method_tag: str | None = None
    """Filter runs by ``rliable_scores.json`` method_tag (e.g. hidden_context)"""

    label: str | None = None
    """Human-readable name in reports / plots (default: preset label or exp_name)"""

    seeds: tuple[int, ...] | None = None
    """Optional subset of seeds; default = all found"""

    require_finished: bool = False
    """Only include runs whose manifest has finished=true"""

    reference_return: float | None = 100.0
    """Optimality-gap reference (typical successful Pistonball return scale)"""

    bootstrap_reps: int = 5000
    """Stratified bootstrap repetitions for aggregate metrics"""

    compare_preset: str | None = None
    """Compare against another preset (e.g. --preset context5 --compare-preset baseline5)"""

    compare_exp_name: str | None = None
    """Compare against a second experiment (manual, overrides compare_preset)"""

    compare_method_tag: str | None = None
    """method_tag filter for the comparison experiment"""

    compare_label: str | None = None
    """Display label for the comparison experiment"""

    output_dir: str = "reports"
    """Directory for JSON reports"""

    profile_min: float = -50.0
    """Performance profile threshold range (return)"""

    profile_max: float = 100.0

    profile_steps: int = 31


def _aggregate_one_experiment(
    runs_root: Path,
    config: dict[str, str],
    *,
    seeds: list[int] | None,
    require_finished: bool,
    reference_return: float | None,
    bootstrap_reps: int,
    thresholds: np.ndarray,
) -> dict:
    method_tag = config["method_tag"] or None
    run_dirs = discover_run_dirs(
        runs_root,
        config["exp_name"],
        seeds,
        method_tag=method_tag,
        require_finished=require_finished,
    )
    if not run_dirs:
        tag_hint = f" method_tag={method_tag!r}" if method_tag else ""
        raise SystemExit(
            f"No runs found for exp_name={config['exp_name']!r}{tag_hint} under {runs_root}"
        )

    matrix, context_order, found_seeds = build_score_matrix(run_dirs)
    report_label = config["label"]
    id_idx, ood_idx = split_id_ood(context_order)
    id_names = [context_order[i] for i in id_idx]
    ood_names = [context_order[i] for i in ood_idx]

    scores_dict = {report_label: matrix}
    rliable_all = aggregate_with_rliable(
        scores_dict,
        reference_return=reference_return,
        bootstrap_reps=bootstrap_reps,
        profile_thresholds=thresholds,
    )

    rliable_splits: dict[str, dict] = {"all_contexts": rliable_all}
    if id_names:
        id_matrix, _ = subset_score_matrix(matrix, context_order, id_names)
        rliable_splits["in_distribution"] = aggregate_with_rliable(
            {report_label: id_matrix},
            reference_return=reference_return,
            bootstrap_reps=bootstrap_reps,
        )
    if ood_names:
        ood_matrix, _ = subset_score_matrix(matrix, context_order, ood_names)
        rliable_splits["out_of_distribution"] = aggregate_with_rliable(
            {report_label: ood_matrix},
            reference_return=reference_return,
            bootstrap_reps=bootstrap_reps,
        )

    report = {
        "exp_name": config["exp_name"],
        "method_tag": method_tag or None,
        "label": report_label,
        "algorithm": config.get("algorithm", config["exp_name"]),
        "preset": config.get("preset"),
        "seeds": found_seeds,
        "n_seeds": len(found_seeds),
        "run_dirs": [str(p) for p in run_dirs],
        "context_order": context_order,
        "generalization": compute_generalization_gap(matrix, context_order),
        "per_context": compute_per_context_stats(matrix, context_order),
        "seed_stability": collect_seed_stability(run_dirs, context_order),
        "sample_efficiency": {
            "median_steps_to_threshold": median_steps_to_threshold(run_dirs),
        },
        "rliable": rliable_all,
        "rliable_by_split": rliable_splits,
    }
    try:
        report["sample_efficiency"]["eval_iqm_curve"] = aggregate_sample_efficiency_iqm(
            run_dirs, bootstrap_reps=bootstrap_reps
        )
    except ValueError as exc:
        report["sample_efficiency"]["eval_iqm_curve"] = {"error": str(exc)}
    return report, matrix


def _compare_reports(
    primary: dict,
    primary_matrix: np.ndarray,
    challenger: dict,
    challenger_matrix: np.ndarray,
    *,
    bootstrap_reps: int,
) -> dict:
    if primary["context_order"] != challenger["context_order"]:
        raise ValueError("Context order mismatch between experiments; cannot compare.")
    primary_label = primary["label"]
    challenger_label = challenger["label"]
    return {
        "primary": primary_label,
        "challenger": challenger_label,
        "primary_exp_name": primary["exp_name"],
        "challenger_exp_name": challenger["exp_name"],
        "shared_seeds": sorted(set(primary["seeds"]) & set(challenger["seeds"])),
        "probability_primary_beats_challenger": compare_probability_of_improvement(
            primary_matrix,
            challenger_matrix,
            bootstrap_reps=min(bootstrap_reps, 2000),
        ),
        "probability_challenger_beats_primary": compare_probability_of_improvement(
            challenger_matrix,
            primary_matrix,
            bootstrap_reps=min(bootstrap_reps, 2000),
        ),
    }


def _print_summary(report: dict, matrix: np.ndarray) -> None:
    label = report["label"]
    iqm = report["rliable"]["algorithms"][label]["metrics"]["iqm"]
    print(
        f"[{label}] IQM return [{iqm['ci_low']:.2f}, {iqm['ci_high']:.2f}] "
        f"(point {iqm['point']:.2f}) over {matrix.shape[0]} seeds × {matrix.shape[1]} contexts"
    )
    gap = report["generalization"]
    print(
        f"  Generalization gap (ID−OOD mean): {gap['generalization_gap']:.3f} "
        f"(ID={gap['id_return_mean']:.2f}, OOD={gap['ood_return_mean']:.2f})"
    )
    if "generalization_gap_iqm" in gap:
        print(
            f"  Generalization gap (ID−OOD IQM): {gap['generalization_gap_iqm']:.3f} "
            f"(ID IQM={gap['id_return_iqm']:.2f}, OOD IQM={gap['ood_return_iqm']:.2f})"
        )
    ood = report.get("rliable_by_split", {}).get("out_of_distribution")
    if ood:
        ood_iqm = ood["algorithms"][label]["metrics"]["iqm"]
        print(
            f"  OOD-only IQM [{ood_iqm['ci_low']:.2f}, {ood_iqm['ci_high']:.2f}] "
            f"(point {ood_iqm['point']:.2f})"
        )


def main(args: Args) -> Path:
    runs_root = Path(args.runs_root)
    seeds = list(args.seeds) if args.seeds is not None else None
    thresholds = np.linspace(args.profile_min, args.profile_max, args.profile_steps)

    primary_config = resolve_experiment_config(
        preset=args.preset,
        exp_name=args.exp_name,
        method_tag=args.method_tag,
        label=args.label,
    )
    if args.preset:
        primary_config["preset"] = args.preset

    primary_report, primary_matrix = _aggregate_one_experiment(
        runs_root,
        primary_config,
        seeds=seeds,
        require_finished=args.require_finished,
        reference_return=args.reference_return,
        bootstrap_reps=args.bootstrap_reps,
        thresholds=thresholds,
    )

    compare_config: dict[str, str] | None = None
    if args.compare_preset is not None:
        compare_config = resolve_experiment_config(
            preset=args.compare_preset,
            exp_name=args.compare_exp_name,
            method_tag=args.compare_method_tag,
            label=args.compare_label,
        )
        compare_config["preset"] = args.compare_preset
    elif args.compare_exp_name is not None:
        compare_config = resolve_experiment_config(
            preset=None,
            exp_name=args.compare_exp_name,
            method_tag=args.compare_method_tag,
            label=args.compare_label,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if compare_config is not None:
        compare_report, compare_matrix = _aggregate_one_experiment(
            runs_root,
            compare_config,
            seeds=seeds,
            require_finished=args.require_finished,
            reference_return=args.reference_return,
            bootstrap_reps=args.bootstrap_reps,
            thresholds=thresholds,
        )
        comparison = _compare_reports(
            primary_report,
            primary_matrix,
            compare_report,
            compare_matrix,
            bootstrap_reps=args.bootstrap_reps,
        )
        primary_report["comparison"] = comparison
        primary_report["compare_report"] = compare_report

        # Combined rliable block for overlay performance profiles in the notebook.
        combined_scores = {
            primary_report["label"]: primary_matrix,
            compare_report["label"]: compare_matrix,
        }
        primary_report["rliable_combined"] = aggregate_with_rliable(
            combined_scores,
            reference_return=args.reference_return,
            bootstrap_reps=args.bootstrap_reps,
            profile_thresholds=thresholds,
        )

        compare_slug = compare_report["exp_name"].replace("/", "_")
        out_path = out_dir / f"rliable_compare_{primary_report['exp_name']}_vs_{compare_slug}.json"
        payload = {
            "primary": primary_report,
            "challenger": compare_report,
            "comparison": comparison,
            "rliable_combined": primary_report["rliable_combined"],
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote comparison report {out_path}")
        _print_summary(primary_report, primary_matrix)
        _print_summary(compare_report, compare_matrix)
        poi = comparison["probability_challenger_beats_primary"]
        print(
            f"P({compare_report['label']} > {primary_report['label']}): "
            f"{poi['probability_of_improvement']:.3f} "
            f"[{poi['ci_low']:.3f}, {poi['ci_high']:.3f}]"
        )
        return out_path

    out_path = out_dir / f"rliable_{primary_report['exp_name']}.json"
    with open(out_path, "w") as f:
        json.dump(primary_report, f, indent=2)
    print(f"Wrote {out_path}")
    _print_summary(primary_report, primary_matrix)
    return out_path


if __name__ == "__main__":
    main(tyro.cli(Args))
