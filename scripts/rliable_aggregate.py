"""Aggregate multi-seed experiment outputs with rliable (robust evaluation).

Designed for the contextual-MARL study:
  - hidden context baseline vs explicit context concat
  - static vs varying training physics (2×2 Pistonball matrix)
  - Multiwalker hidden vs concat (``--compare-multiwalker``)
  - Pistonball context-kind ablation (``--compare-kinds``)

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
    KIND_PRESETS,
    MATRIX_PRESETS,
    MULTIWALKER_PRESETS,
    aggregate_sample_efficiency_iqm,
    aggregate_with_rliable,
    build_score_matrix,
    collect_seed_stability,
    compare_probability_of_improvement,
    compute_generalization_gap,
    compute_per_context_stats,
    discover_run_dirs_relaxed,
    load_context_meta,
    load_rliable_scores,
    median_steps_to_threshold,
    resolve_experiment_config,
    split_id_ood,
    subset_score_matrix,
)
from evaluation.diagnostics import diagnose_method


@dataclass
class Args:
    runs_root: str = "runs"
    """Root directory containing training runs"""

    preset: str | None = None
    """Single experiment shortcut (e.g. baseline5_varying). Prefer --compare-all for the 2×2."""

    presets: str | None = None
    """Comma-separated presets to aggregate together (overrides --preset)."""

    compare_all: bool = False
    """Aggregate the 2×2 matrix (static/varying × hidden/concat) into one report."""

    compare_multiwalker: bool = False
    """Aggregate Multiwalker hidden vs concat (varying). Default reference return 300."""

    compare_multiwalker_clip: bool = False
    """Hidden vs concat vs concat with the context vector clipped to the ID window."""

    compare_kinds: bool = False
    """Aggregate Pistonball concat physics vs agent vs both (varying)."""

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
    """Legacy pairwise compare (e.g. --preset context5 --compare-preset baseline5)"""

    compare_exp_name: str | None = None
    """Compare against a second experiment (manual, overrides compare_preset)"""

    compare_method_tag: str | None = None
    """method_tag filter for the comparison experiment"""

    compare_label: str | None = None
    """Display label for the comparison experiment"""

    output_dir: str = "reports"
    """Directory for JSON reports"""

    output_name: str | None = None
    """Optional output filename (default: rliable_matrix.json for multi-method)"""

    profile_min: float = -50.0
    """Performance profile threshold range (return)"""

    profile_max: float = 100.0

    profile_steps: int = 31


def _parse_preset_list(raw: str) -> list[str]:
    names = [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]
    unknown = [name for name in names if name not in EXPERIMENT_PRESETS]
    if unknown:
        known = ", ".join(sorted(EXPERIMENT_PRESETS))
        raise SystemExit(f"Unknown preset(s) {unknown}; choose from: {known}")
    if not names:
        raise SystemExit("--presets was empty")
    return names


def _aggregate_one_experiment(
    runs_root: Path,
    config: dict[str, str],
    *,
    seeds: list[int] | None,
    require_finished: bool,
    reference_return: float | None,
    bootstrap_reps: int,
    thresholds: np.ndarray,
) -> tuple[dict, np.ndarray]:
    requested_tag = config["method_tag"] or None
    run_dirs = discover_run_dirs_relaxed(
        runs_root,
        config["exp_name"],
        seeds,
        method_tag=requested_tag,
        require_finished=require_finished,
    )
    if not run_dirs:
        tag_hint = f" method_tag={requested_tag!r}" if requested_tag else ""
        raise SystemExit(
            f"No runs found for exp_name={config['exp_name']!r}{tag_hint} under {runs_root}"
        )

    from evaluation.rliable_utils import RLIABLE_SCORES_FILENAME

    scores_filename = config.get("scores_filename") or RLIABLE_SCORES_FILENAME
    results_filename = config.get("results_filename") or "evaluation_results.json"
    matrix, context_order, found_seeds = build_score_matrix(run_dirs, scores_filename=scores_filename)
    context_meta = load_context_meta(run_dirs, scores_filename=scores_filename)
    report_label = config["label"]
    id_idx, ood_idx = split_id_ood(context_order, context_meta)
    id_names = [context_order[i] for i in id_idx]
    ood_names = [context_order[i] for i in ood_idx]
    scores_file = load_rliable_scores(run_dirs[0], scores_filename) or {}
    actual_tag = scores_file.get("method_tag") or requested_tag

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
        "method_tag": actual_tag or None,
        "label": report_label,
        "algorithm": config.get("algorithm", config["exp_name"]),
        "preset": config.get("preset"),
        "seeds": found_seeds,
        "n_seeds": len(found_seeds),
        "run_dirs": [str(p) for p in run_dirs],
        "context_order": context_order,
        "context_meta": context_meta,
        "generalization": compute_generalization_gap(matrix, context_order, context_meta),
        "per_context": compute_per_context_stats(matrix, context_order),
        "scores_filename": scores_filename,
        "results_filename": results_filename,
        "context_clip_to_id": bool(scores_file.get("context_clip_to_id")),
        "seed_stability": collect_seed_stability(run_dirs, context_order, scores_filename=scores_filename),
        "sample_efficiency": {
            "median_steps_to_threshold": median_steps_to_threshold(run_dirs),
        },
        "rliable": rliable_all,
        "rliable_by_split": rliable_splits,
        "id_factor_low": scores_file.get("id_factor_low"),
        "id_factor_high": scores_file.get("id_factor_high"),
        "eval_factor_grid": scores_file.get("eval_factor_grid"),
        "eval_vary_params": scores_file.get("eval_vary_params"),
        "eval_grid_preset": scores_file.get("eval_grid_preset"),
        "diagnostics": diagnose_method(matrix, {
            "context_order": context_order,
            "context_meta": context_meta,
            "seeds": found_seeds,
            "eval_factor_grid": scores_file.get("eval_factor_grid"),
        }, run_dirs, results_filename=results_filename),
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


def _pairwise_poi(
    reports: list[dict],
    matrices: list[np.ndarray],
    *,
    bootstrap_reps: int,
) -> list[dict]:
    pairs: list[dict] = []
    poi_reps = min(bootstrap_reps, 2000)
    for i, (report_a, matrix_a) in enumerate(zip(reports, matrices)):
        for report_b, matrix_b in zip(reports[i + 1 :], matrices[i + 1 :]):
            pairs.append(
                {
                    "a": report_a["label"],
                    "b": report_b["label"],
                    "a_exp_name": report_a["exp_name"],
                    "b_exp_name": report_b["exp_name"],
                    "probability_a_beats_b": compare_probability_of_improvement(
                        matrix_a, matrix_b, bootstrap_reps=poi_reps
                    ),
                    "probability_b_beats_a": compare_probability_of_improvement(
                        matrix_b, matrix_a, bootstrap_reps=poi_reps
                    ),
                }
            )
    return pairs


def _combined_split_scores(
    reports: list[dict],
    matrices: list[np.ndarray],
    *,
    split: str,
) -> dict[str, np.ndarray] | None:
    context_order = reports[0]["context_order"]
    context_meta = reports[0].get("context_meta")
    id_idx, ood_idx = split_id_ood(context_order, context_meta)
    names = [context_order[i] for i in (id_idx if split == "in_distribution" else ood_idx)]
    if not names:
        return None
    scores: dict[str, np.ndarray] = {}
    for report, matrix in zip(reports, matrices):
        subset, _ = subset_score_matrix(matrix, context_order, names)
        scores[report["label"]] = subset
    return scores


def _combine_experiments(
    reports: list[dict],
    matrices: list[np.ndarray],
    *,
    reference_return: float | None,
    bootstrap_reps: int,
    thresholds: np.ndarray,
) -> dict:
    context_order = reports[0]["context_order"]
    for report in reports[1:]:
        if report["context_order"] != context_order:
            raise ValueError(
                "Context order mismatch between experiments; cannot combine. "
                f"{reports[0]['exp_name']}={context_order} vs "
                f"{report['exp_name']}={report['context_order']}"
            )
    combined_scores = {report["label"]: matrix for report, matrix in zip(reports, matrices)}
    rliable_combined = aggregate_with_rliable(
        combined_scores,
        reference_return=reference_return,
        bootstrap_reps=bootstrap_reps,
        profile_thresholds=thresholds,
    )
    rliable_combined_by_split: dict[str, dict] = {"all_contexts": rliable_combined}
    for split in ("in_distribution", "out_of_distribution"):
        split_scores = _combined_split_scores(reports, matrices, split=split)
        if split_scores is None:
            continue
        rliable_combined_by_split[split] = aggregate_with_rliable(
            split_scores,
            reference_return=reference_return,
            bootstrap_reps=bootstrap_reps,
        )
    return {
        "experiments": reports,
        "rliable_combined": rliable_combined,
        "rliable_combined_by_split": rliable_combined_by_split,
        "pairwise_poi": _pairwise_poi(reports, matrices, bootstrap_reps=bootstrap_reps),
        "context_order": context_order,
        "context_meta": reports[0].get("context_meta"),
        "presets": [report.get("preset") for report in reports],
        "exp_names": [report["exp_name"] for report in reports],
        "labels": [report["label"] for report in reports],
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
    med = report.get("sample_efficiency", {}).get("median_steps_to_threshold") or {}
    if med.get("median_steps_to_threshold") is not None:
        print(
            f"  Median steps to threshold: {med['median_steps_to_threshold']:.0f} "
            f"({med.get('n_seeds_reached', 0)}/{report.get('n_seeds', '?')} seeds reached)"
        )
    diag = report.get("diagnostics") or {}
    clones = [g["param"] for g in diag.get("clones", []) if g.get("cloned")]
    flats = [s["param"] for s in diag.get("slices", []) if s.get("flat")]
    n_coll = diag.get("n_collapsed_seeds")
    if clones or flats or n_coll:
        print(
            f"  Diagnostics: cloned={clones or '—'} flat={flats or '—'} "
            f"collapsed={n_coll}/{report.get('n_seeds', '?')}"
        )
    env = (diag.get("env_switch") or {}).get("action")
    if env:
        print(f"  Env advice: {env}")
    coord = diag.get("coordination") or {}
    if coord.get("per_agent_return_variance_mean") is not None:
        print(
            f"  Coordination: per-agent return var="
            f"{coord['per_agent_return_variance_mean']:.3f}"
        )


def _print_pairwise_poi(pairs: list[dict]) -> None:
    if not pairs:
        return
    print("Probability of improvement:")
    for pair in pairs:
        poi = pair["probability_a_beats_b"]
        print(
            f"  P({pair['a']} > {pair['b']}): {poi['probability_of_improvement']:.3f} "
            f"[{poi['ci_low']:.3f}, {poi['ci_high']:.3f}]"
        )


def _config_from_preset(preset: str, args: Args, *, is_compare: bool = False) -> dict[str, str]:
    if is_compare:
        config = resolve_experiment_config(
            preset=preset,
            exp_name=args.compare_exp_name,
            method_tag=args.compare_method_tag,
            label=args.compare_label,
        )
    else:
        config = resolve_experiment_config(
            preset=preset,
            exp_name=args.exp_name if args.preset == preset else None,
            method_tag=args.method_tag if args.preset == preset else None,
            label=args.label if args.preset == preset else None,
        )
    config["preset"] = preset
    return config


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _aggregate_named_configs(
    args: Args,
    configs: list[dict[str, str]],
    *,
    seeds: list[int] | None,
    thresholds: np.ndarray,
) -> tuple[list[dict], list[np.ndarray]]:
    reports: list[dict] = []
    matrices: list[np.ndarray] = []
    for config in configs:
        report, matrix = _aggregate_one_experiment(
            Path(args.runs_root),
            config,
            seeds=seeds,
            require_finished=args.require_finished,
            reference_return=args.reference_return,
            bootstrap_reps=args.bootstrap_reps,
            thresholds=thresholds,
        )
        reports.append(report)
        matrices.append(matrix)
        _print_summary(report, matrix)
    return reports, matrices


def main(args: Args) -> Path:
    seeds = list(args.seeds) if args.seeds is not None else None
    thresholds = np.linspace(args.profile_min, args.profile_max, args.profile_steps)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preset_names: list[str] = []
    if args.compare_all:
        preset_names = list(MATRIX_PRESETS)
    elif args.compare_multiwalker or args.compare_multiwalker_clip:
        preset_names = list(MULTIWALKER_PRESETS)
        if args.reference_return == 100.0:
            args.reference_return = 300.0
        if args.profile_max == 100.0:
            args.profile_max = 300.0
        if args.output_name is None:
            args.output_name = (
                "rliable_multiwalker_context_clip.json"
                if args.compare_multiwalker_clip
                else "rliable_multiwalker.json"
            )
        thresholds = np.linspace(args.profile_min, args.profile_max, args.profile_steps)
    elif args.compare_kinds:
        preset_names = list(KIND_PRESETS)
        if args.output_name is None:
            args.output_name = "rliable_context_kinds.json"
    elif args.presets:
        preset_names = _parse_preset_list(args.presets)

    # Multi-method path: --compare-all or --presets with 2+ names.
    if preset_names:
        if len(preset_names) == 1 and not args.compare_all and args.compare_preset is None:
            args.preset = preset_names[0]
            preset_names = []
        else:
            configs = [_config_from_preset(name, args) for name in preset_names]
            if args.compare_multiwalker_clip:
                from evaluation.checkpoint_eval import (
                    CONTEXT_CLIP_RESULTS_FILENAME,
                    CONTEXT_CLIP_SCORES_FILENAME,
                )

                clip_cfg = dict(configs[-1])
                clip_cfg["label"] = "Multiwalker concat, context clipped to ID"
                clip_cfg["preset"] = "mw3_concat_varying_fwd5_context_clip"
                clip_cfg["scores_filename"] = CONTEXT_CLIP_SCORES_FILENAME
                clip_cfg["results_filename"] = CONTEXT_CLIP_RESULTS_FILENAME
                configs.append(clip_cfg)
            reports, matrices = _aggregate_named_configs(
                args, configs, seeds=seeds, thresholds=thresholds
            )
            payload = _combine_experiments(
                reports,
                matrices,
                reference_return=args.reference_return,
                bootstrap_reps=args.bootstrap_reps,
                thresholds=thresholds,
            )
            filename = args.output_name or "rliable_matrix.json"
            out_path = _write_json(out_dir / filename, payload)
            print(f"Wrote comparison report {out_path}")
            _print_pairwise_poi(payload["pairwise_poi"])
            return out_path

    primary_config = resolve_experiment_config(
        preset=args.preset,
        exp_name=args.exp_name,
        method_tag=args.method_tag,
        label=args.label,
    )
    if args.preset:
        primary_config["preset"] = args.preset

    primary_report, primary_matrix = _aggregate_one_experiment(
        Path(args.runs_root),
        primary_config,
        seeds=seeds,
        require_finished=args.require_finished,
        reference_return=args.reference_return,
        bootstrap_reps=args.bootstrap_reps,
        thresholds=thresholds,
    )

    compare_config: dict[str, str] | None = None
    if args.compare_preset is not None:
        compare_config = _config_from_preset(args.compare_preset, args, is_compare=True)
    elif args.compare_exp_name is not None:
        compare_config = resolve_experiment_config(
            preset=None,
            exp_name=args.compare_exp_name,
            method_tag=args.compare_method_tag,
            label=args.compare_label,
        )

    if compare_config is not None:
        compare_report, compare_matrix = _aggregate_one_experiment(
            Path(args.runs_root),
            compare_config,
            seeds=seeds,
            require_finished=args.require_finished,
            reference_return=args.reference_return,
            bootstrap_reps=args.bootstrap_reps,
            thresholds=thresholds,
        )
        payload = _combine_experiments(
            [primary_report, compare_report],
            [primary_matrix, compare_matrix],
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
        # Keep the historical pairwise JSON keys for existing notebook paths.
        pairwise_payload = {
            "primary": primary_report,
            "challenger": compare_report,
            "comparison": comparison,
            "rliable_combined": payload["rliable_combined"],
            "experiments": payload["experiments"],
            "rliable_combined_by_split": payload["rliable_combined_by_split"],
            "pairwise_poi": payload["pairwise_poi"],
            "context_order": payload["context_order"],
            "context_meta": payload["context_meta"],
        }
        compare_slug = compare_report["exp_name"].replace("/", "_")
        filename = args.output_name or (
            f"rliable_compare_{primary_report['exp_name']}_vs_{compare_slug}.json"
        )
        out_path = _write_json(out_dir / filename, pairwise_payload)
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

    out_path = _write_json(out_dir / f"rliable_{primary_report['exp_name']}.json", primary_report)
    print(f"Wrote {out_path}")
    _print_summary(primary_report, primary_matrix)
    return out_path


if __name__ == "__main__":
    main(tyro.cli(Args))
