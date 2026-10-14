"""Readable matplotlib figures for rliable JSON reports (notebook + cluster post-processing)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

METRIC_LABELS = {
    "median": "Median return",
    "iqm": "IQM return",
    "mean": "Mean return",
    "optimality_gap": "Optimality gap",
}

METHOD_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B3", "#CCB974", "#64B5CD"]


def load_report(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def unpack_eval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize single / pairwise / matrix JSON into a list of experiment reports."""
    if "experiments" in payload and payload["experiments"]:
        reports = list(payload["experiments"])
        kind = "matrix" if len(reports) > 2 else "compare"
        return {
            "kind": kind,
            "reports": reports,
            "rliable_combined": payload.get("rliable_combined"),
            "rliable_combined_by_split": payload.get("rliable_combined_by_split"),
            "pairwise_poi": payload.get("pairwise_poi") or [],
            "comparison": payload.get("comparison"),
            "context_order": payload.get("context_order") or reports[0].get("context_order"),
            "context_meta": payload.get("context_meta") or reports[0].get("context_meta"),
        }
    if "primary" in payload and "challenger" in payload:
        reports = [payload["primary"], payload["challenger"]]
        return {
            "kind": "compare",
            "reports": reports,
            "rliable_combined": payload.get("rliable_combined"),
            "rliable_combined_by_split": payload.get("rliable_combined_by_split"),
            "pairwise_poi": payload.get("pairwise_poi") or [],
            "comparison": payload.get("comparison"),
            "context_order": payload.get("context_order") or reports[0].get("context_order"),
            "context_meta": payload.get("context_meta") or reports[0].get("context_meta"),
        }
    return {
        "kind": "single",
        "reports": [payload],
        "rliable_combined": None,
        "rliable_combined_by_split": None,
        "pairwise_poi": [],
        "comparison": None,
        "context_order": payload.get("context_order"),
        "context_meta": payload.get("context_meta"),
    }


def aggregate_metrics_table(report: dict[str, Any], *, algorithm: str | None = None) -> list[dict[str, Any]]:
    """Flatten rliable aggregate metrics into rows for display (e.g. pandas)."""
    rows: list[dict[str, Any]] = []
    algo = algorithm or report.get("label") or report.get("exp_name")
    for algo_name, block in report.get("rliable", {}).get("algorithms", {}).items():
        if algo is not None and algo_name != algo:
            continue
        for metric, vals in block.get("metrics", {}).items():
            rows.append(
                {
                    "algorithm": algo_name,
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "point": vals["point"],
                    "ci_low": vals["ci_low"],
                    "ci_high": vals["ci_high"],
                }
            )
    return rows


def format_aggregate_summary(report: dict[str, Any]) -> str:
    """Human-readable summary for notebook markdown cells."""
    label = report.get("label") or report.get("exp_name", "?")
    lines = [
        f"Experiment: **{report.get('exp_name', '?')}** ({label})",
        f"Method tag: {report.get('method_tag', '—')}",
        f"Seeds ({report.get('n_seeds', len(report.get('seeds', [])))}): {report.get('seeds', [])}",
        f"Contexts: {report.get('context_order', [])}",
        "",
    ]
    gen = report.get("generalization", {})
    lines.append(
        f"Generalization (mean return): ID={gen.get('id_return_mean', float('nan')):.2f}, "
        f"OOD={gen.get('ood_return_mean', float('nan')):.2f}, "
        f"gap={gen.get('generalization_gap', float('nan')):.2f}"
    )
    if "generalization_gap_iqm" in gen:
        lines.append(
            f"Generalization (IQM): ID={gen.get('id_return_iqm', float('nan')):.2f}, "
            f"OOD={gen.get('ood_return_iqm', float('nan')):.2f}, "
            f"gap={gen.get('generalization_gap_iqm', float('nan')):.2f}"
        )
    med = report.get("sample_efficiency", {}).get("median_steps_to_threshold", {})
    if med.get("median_steps_to_threshold") is not None:
        lines.append(
            f"Median steps to threshold: {med['median_steps_to_threshold']:.0f} "
            f"({med.get('n_seeds_reached', 0)} seeds)"
        )
    lines.append("")
    algo = label
    for row in aggregate_metrics_table(report, algorithm=algo):
        lines.append(
            f"- {row['label']}: {row['point']:.2f} "
            f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        )
    ood_block = report.get("rliable_by_split", {}).get("out_of_distribution")
    if ood_block and algo in ood_block.get("algorithms", {}):
        ood_iqm = ood_block["algorithms"][algo]["metrics"]["iqm"]
        lines.append(
            f"- OOD-only IQM return: {ood_iqm['point']:.2f} "
            f"[{ood_iqm['ci_low']:.2f}, {ood_iqm['ci_high']:.2f}]"
        )
    if "comparison" in report:
        cmp = report["comparison"]
        poi = cmp.get("probability_challenger_beats_primary") or cmp.get("probability_of_improvement")
        if poi:
            lines.append("")
            lines.append(
                f"P({cmp.get('challenger', '?')} > {cmp.get('primary', '?')}): "
                f"{poi['probability_of_improvement']:.2f} "
                f"[{poi['ci_low']:.2f}, {poi['ci_high']:.2f}]"
            )
    return "\n".join(lines)


def format_pairwise_poi(pairs: list[dict[str, Any]]) -> str:
    """Human-readable probability-of-improvement table."""
    if not pairs:
        return ""
    lines = ["Probability of improvement:"]
    for pair in pairs:
        poi = pair.get("probability_a_beats_b") or {}
        lines.append(
            f"- P({pair.get('a', '?')} > {pair.get('b', '?')}): "
            f"{poi.get('probability_of_improvement', float('nan')):.3f} "
            f"[{poi.get('ci_low', float('nan')):.3f}, {poi.get('ci_high', float('nan')):.3f}]"
        )
    return "\n".join(lines)


def format_matrix_summary(payload: dict[str, Any]) -> str:
    """Concatenate per-method summaries plus pairwise PoI for a combined report."""
    unpacked = unpack_eval_payload(payload)
    blocks = [format_aggregate_summary(report) for report in unpacked["reports"]]
    poi = format_pairwise_poi(unpacked.get("pairwise_poi") or [])
    if poi:
        blocks.append(poi)
    return "\n\n".join(blocks)


def _report_label(report: dict[str, Any]) -> str:
    return str(report.get("label") or report.get("exp_name") or "?")


def _rliable_metric(report: dict[str, Any], metric: str, *, split: str | None = None) -> dict[str, float]:
    label = _report_label(report)
    if split:
        block = (report.get("rliable_by_split") or {}).get(split) or {}
        algo = (block.get("algorithms") or {}).get(label) or {}
    else:
        algo = (report.get("rliable") or {}).get("algorithms", {}).get(label) or {}
    return dict((algo.get("metrics") or {}).get(metric) or {})


def slice_params_from_report(report: dict[str, Any]) -> tuple[str, ...]:
    """Physics axes present in ``context_meta`` (includes gravity when evaluated)."""
    meta = report.get("context_meta") or {}
    order = list(report.get("context_order") or [])
    params: list[str] = []
    seen: set[str] = set()
    for name in order:
        param = (meta.get(name) or {}).get("varied_param")
        if param and param not in seen:
            seen.add(str(param))
            params.append(str(param))
    return tuple(params) or ("ball_mass", "ball_friction", "ball_elasticity")


def ensure_diagnostics(
    report: dict[str, Any],
    score_matrix: np.ndarray | None = None,
    run_dirs: list[Path] | list[str] | None = None,
) -> dict[str, Any]:
    """Return clone/trapezoid/seed diagnostics, computing them if the JSON is older."""
    if score_matrix is None:
        return dict(report.get("diagnostics") or {})
    from evaluation.diagnostics import diagnose_method

    dirs = run_dirs if run_dirs is not None else report.get("run_dirs") or []
    diag = diagnose_method(
        np.asarray(score_matrix, dtype=float),
        report,
        [Path(p) for p in dirs],
    )
    report["diagnostics"] = diag
    return diag


def _poi_vs_reference(
    label: str,
    pairwise_poi: list[dict[str, Any]] | None,
    reference: str | None,
) -> dict[str, float] | None:
    if not pairwise_poi or not reference or label == reference:
        return None
    for pair in pairwise_poi:
        if pair.get("a") == label and pair.get("b") == reference:
            return pair.get("probability_a_beats_b")
        if pair.get("b") == label and pair.get("a") == reference:
            return pair.get("probability_b_beats_a")
    return None


def methodology_dashboard_rows(
    reports: list[dict[str, Any]],
    matrices: list[np.ndarray] | None = None,
    pairwise_poi: list[dict[str, Any]] | None = None,
    *,
    reference_label: str | None = None,
) -> list[dict[str, Any]]:
    """One row per method: IQM, ID/OOD gap, PoI, steps-to-threshold, collapse, flatness."""
    labels = [_report_label(r) for r in reports]
    if reference_label is None:
        for candidate in ("Hidden context (varying)", "hidden_context_varying"):
            if candidate in labels:
                reference_label = candidate
                break
        if reference_label is None and len(labels) > 1:
            reference_label = labels[0]

    rows: list[dict[str, Any]] = []
    for i, report in enumerate(reports):
        label = _report_label(report)
        matrix = None if matrices is None else matrices[i]
        diag = ensure_diagnostics(report, matrix, report.get("run_dirs"))
        iqm = _rliable_metric(report, "iqm")
        mean = _rliable_metric(report, "mean")
        gen = report.get("generalization") or {}
        med = (report.get("sample_efficiency") or {}).get("median_steps_to_threshold") or {}
        clones = [g["param"] for g in diag.get("clones", []) if g.get("cloned")]
        flats = [s["param"] for s in diag.get("slices", []) if s.get("flat")]
        poi = _poi_vs_reference(label, pairwise_poi, reference_label)
        coord = diag.get("coordination") or {}
        env = diag.get("env_switch") or {}
        rows.append(
            {
                "method": label,
                "IQM": iqm.get("point"),
                "IQM CI low": iqm.get("ci_low"),
                "IQM CI high": iqm.get("ci_high"),
                "mean": mean.get("point"),
                "ID IQM": gen.get("id_return_iqm"),
                "OOD IQM": gen.get("ood_return_iqm"),
                "ID→OOD IQM gap": gen.get("generalization_gap_iqm"),
                f"PoI vs {reference_label}" if reference_label else "PoI": (
                    None if poi is None else poi.get("probability_of_improvement")
                ),
                "median steps to threshold": med.get("median_steps_to_threshold"),
                "seeds reached threshold": med.get("n_seeds_reached"),
                "collapsed seeds": diag.get("n_collapsed_seeds"),
                "n seeds": report.get("n_seeds") or len(report.get("seeds") or []),
                "cloned axes": ", ".join(clones) if clones else "—",
                "flat C.2 axes": ", ".join(flats) if flats else "—",
                "per-agent return var": coord.get("per_agent_return_variance_mean"),
                "success rate": coord.get("success_rate_mean"),
                "env advice": env.get("action"),
            }
        )
    return rows


def seed_health_rows(
    reports: list[dict[str, Any]],
    matrices: list[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, report in enumerate(reports):
        matrix = None if matrices is None else matrices[i]
        diag = ensure_diagnostics(report, matrix, report.get("run_dirs"))
        label = _report_label(report)
        for health in diag.get("seed_health", []):
            rows.append({"method": label, **health})
    return rows


def clone_trapezoid_rows(
    reports: list[dict[str, Any]],
    matrices: list[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, report in enumerate(reports):
        matrix = None if matrices is None else matrices[i]
        diag = ensure_diagnostics(report, matrix, report.get("run_dirs"))
        label = _report_label(report)
        clones = {g["param"]: g for g in diag.get("clones", [])}
        for slc in diag.get("slices", []):
            clone = clones.get(slc["param"]) or {}
            rows.append(
                {
                    "method": label,
                    "param": slc["param"],
                    "ID return": slc.get("id_return_mean"),
                    "OOD return": slc.get("ood_return_mean"),
                    "OOD drop": slc.get("ood_drop"),
                    "flat": slc.get("flat"),
                    "logdev–return corr": slc.get("logdev_return_corr"),
                    "cloned": bool(clone.get("cloned")),
                    "max |Δ| across factors": clone.get("max_abs_diff"),
                    "cloned seeds": clone.get("cloned_seed_count"),
                }
            )
    return rows


def coordination_rows(
    reports: list[dict[str, Any]],
    matrices: list[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, report in enumerate(reports):
        matrix = None if matrices is None else matrices[i]
        diag = ensure_diagnostics(report, matrix, report.get("run_dirs"))
        coord = diag.get("coordination") or {}
        if not coord:
            continue
        rows.append({"method": _report_label(report), **coord})
    return rows


def env_switch_rows(
    reports: list[dict[str, Any]],
    matrices: list[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, report in enumerate(reports):
        matrix = None if matrices is None else matrices[i]
        diag = ensure_diagnostics(report, matrix, report.get("run_dirs"))
        env = dict(diag.get("env_switch") or {})
        env["method"] = _report_label(report)
        rows.append(env)
    return rows


def format_env_switch_advice(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No env-switch diagnostics yet. Re-run aggregation or hard eval."
    lines = ["Env-switch advice (step 4):"]
    switch = any(r.get("switch_env") for r in rows)
    for row in rows:
        lines.append(
            f"- **{row.get('method', '?')}**: {row.get('action', '—')} "
            f"({row.get('reason', '')})"
        )
    if switch:
        cand = next((r.get("candidate") for r in rows if r.get("candidate")), None) or {}
        lines.append("")
        lines.append(
            "Hard + gravity slices are still flat → switch environment. "
            f"Candidate: `{cand.get('env', 'pettingzoo.sisl.multiwalker_v9')}` "
            f"({cand.get('why', 'explicit body physics')})."
        )
    else:
        lines.append("")
        lines.append(
            "Stay on Pistonball until a hard+gravity re-eval is also flat. "
            "Do not add gravity to concat `CONTEXT_KEYS` (that would invalidate 4-D checkpoints)."
        )
    return "\n".join(lines)


def plot_performance_profile(
    report: dict[str, Any],
    *,
    algorithm: str | None = None,
    ax: plt.Axes | None = None,
    title: str = "Performance profile (rliable)",
    color: str | None = None,
) -> plt.Axes:
    """Plot P(return ≥ τ) with bootstrap CI bands from a saved report."""
    profiles = report["rliable"]["performance_profiles"]
    algo = algorithm or report.get("label") or report["exp_name"]
    if algo not in profiles:
        raise KeyError(f"Algorithm {algo!r} not in report performance profiles.")
    data = profiles[algo]
    thresholds = np.asarray(data["thresholds"], dtype=float)
    profile = np.asarray(data["profile"], dtype=float)
    ci_low = np.asarray(data["ci_low"], dtype=float)
    ci_high = np.asarray(data["ci_high"], dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    line_color = color or None
    ax.plot(thresholds, profile, linewidth=2.5, label=algo, color=line_color)
    ax.fill_between(thresholds, ci_low, ci_high, alpha=0.25, color=line_color)
    ax.set_xlabel("Team return threshold τ")
    ax.set_ylabel("Fraction of runs × contexts with return ≥ τ")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title(title)
    return ax


def plot_combined_performance_profile(
    combined_report: dict[str, Any],
    *,
    ax: plt.Axes | None = None,
    title: str = "Method comparison — performance profiles",
) -> plt.Axes:
    """Overlay performance profiles from an ``rliable_combined`` block."""
    profiles = combined_report["performance_profiles"]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    colors = METHOD_COLORS
    for idx, (algo, data) in enumerate(profiles.items()):
        color = colors[idx % len(colors)]
        thresholds = np.asarray(data["thresholds"], dtype=float)
        profile = np.asarray(data["profile"], dtype=float)
        ci_low = np.asarray(data["ci_low"], dtype=float)
        ci_high = np.asarray(data["ci_high"], dtype=float)
        ax.plot(thresholds, profile, linewidth=2.5, label=algo, color=color)
        ax.fill_between(thresholds, ci_low, ci_high, alpha=0.2, color=color)
    ax.set_xlabel("Team return threshold τ")
    ax.set_ylabel("Fraction of runs × contexts with return ≥ τ")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title(title)
    return ax


def plot_aggregate_intervals(
    report: dict[str, Any],
    *,
    algorithm: str | None = None,
    ax: plt.Axes | None = None,
    color: str = "#4C72B0",
) -> plt.Axes:
    """Bar-style interval plot for IQM / median / mean / optimality gap."""
    algo = algorithm or report.get("label") or report["exp_name"]
    metrics = report["rliable"]["algorithms"][algo]["metrics"]
    order = ["iqm", "median", "mean", "optimality_gap"]
    labels = [METRIC_LABELS[m] for m in order]

    points = [metrics[m]["point"] for m in order]
    lows = [metrics[m]["ci_low"] for m in order]
    highs = [metrics[m]["ci_high"] for m in order]
    yerr = np.array([np.array(points) - np.array(lows), np.array(highs) - np.array(points)])

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(order))
    ax.bar(x, points, yerr=yerr, capsize=6, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(f"Aggregate metrics with 95% stratified bootstrap CI — {algo}")
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_method_comparison_bars(
    *reports: dict[str, Any] | list[dict[str, Any]],
    metric: str = "iqm",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Side-by-side bootstrap CI bars for one metric (default: IQM)."""
    if len(reports) == 1 and isinstance(reports[0], (list, tuple)):
        report_list = list(reports[0])
    else:
        report_list = [r for r in reports if isinstance(r, dict)]
    if len(report_list) < 1:
        raise ValueError("Need at least one experiment report.")
    labels: list[str] = []
    points: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for report in report_list:
        label = report.get("label") or report["exp_name"]
        vals = report["rliable"]["algorithms"][label]["metrics"][metric]
        labels.append(label)
        points.append(vals["point"])
        lows.append(vals["ci_low"])
        highs.append(vals["ci_high"])

    if ax is None:
        width = max(6.0, 1.6 * len(report_list) + 2.0)
        _, ax = plt.subplots(figsize=(width, 4))
    yerr = np.array([np.array(points) - np.array(lows), np.array(highs) - np.array(points)])
    x = np.arange(len(report_list))
    colors = [METHOD_COLORS[i % len(METHOD_COLORS)] for i in range(len(report_list))]
    ax.bar(x, points, yerr=yerr, capsize=6, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"Method comparison — {METRIC_LABELS.get(metric, metric)} (all contexts)")
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def _mean_std_by_context(
    report: dict[str, Any],
    score_matrix: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Map context name → (mean, std). Extra matrix columns without names are ignored."""
    order = list(report.get("context_order") or [])
    arr = np.asarray(score_matrix, dtype=float)
    if arr.size == 0:
        return {}
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    n = min(len(order), int(means.shape[0]))
    return {order[i]: (float(means[i]), float(stds[i])) for i in range(n)}


def plot_per_context_comparison(
    report_a: dict[str, Any],
    matrix_a: np.ndarray,
    report_b: dict[str, Any] | None = None,
    matrix_b: np.ndarray | None = None,
    *,
    ax: plt.Axes | None = None,
    entries: list[tuple[dict[str, Any], np.ndarray]] | None = None,
) -> plt.Axes:
    """Grouped bars: mean return per context for two or more methods."""
    if entries is None:
        entries = [(report_a, matrix_a)]
        if report_b is not None and matrix_b is not None:
            entries.append((report_b, matrix_b))
    return plot_per_context_comparison_many(entries, ax=ax)


def plot_per_context_comparison_many(
    entries: list[tuple[dict[str, Any], np.ndarray]],
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Grouped bars: mean return per context for N methods.

    Aligns methods by context *name* so reports/matrices with different grids
    (e.g. 25 vs 29 eval contexts) can be plotted together.
    """
    if not entries:
        raise ValueError("Need at least one (report, score_matrix) pair.")
    stats = [_mean_std_by_context(report, matrix) for report, matrix in entries]
    contexts = list(dict.fromkeys(name for mapping in stats for name in mapping))
    if not contexts:
        raise ValueError("No contexts to plot.")
    n = len(entries)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(11.0, 1.1 * len(contexts) + 2.0), 4.2))
    x = np.arange(len(contexts))
    width = 0.8 / n
    for idx, ((report, _matrix), mapping) in enumerate(zip(entries, stats)):
        label = report.get("label") or report["exp_name"]
        means = np.array([mapping[c][0] if c in mapping else np.nan for c in contexts], dtype=float)
        stds = np.array([mapping[c][1] if c in mapping else 0.0 for c in contexts], dtype=float)
        stds = np.where(np.isfinite(means), stds, 0.0)
        offset = (idx - (n - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=label,
            color=METHOD_COLORS[idx % len(METHOD_COLORS)],
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(contexts, rotation=35, ha="right")
    ax.set_ylabel("Mean team return (across seeds)")
    ax.set_title("Zero-shot transfer by context — method comparison")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def per_context_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for per-context ID/OOD breakdown."""
    meta = report.get("context_meta") or {}
    rows = []
    for ctx, stats in report.get("per_context", {}).items():
        split = meta.get(ctx, {}).get("split")
        if split is None:
            split = "ID" if ctx in ("train", "default") else "OOD"
        else:
            split = split.upper()
        rows.append({"context": ctx, "split": split, **stats})
    return rows


def _id_window(report: dict[str, Any]) -> tuple[float, float]:
    id_low = report.get("id_factor_low")
    id_high = report.get("id_factor_high")
    if id_low is not None and id_high is not None:
        return float(id_low), float(id_high)
    meta = report.get("context_meta") or {}
    id_factors = [
        float(meta[n]["factor"])
        for n in report.get("context_order", [])
        if meta.get(n, {}).get("split") == "id" and meta.get(n, {}).get("factor") is not None
    ]
    return (
        min(id_factors) if id_factors else 0.75,
        max(id_factors) if id_factors else 1.5,
    )


def _slice_xy(
    report: dict[str, Any],
    score_matrix: np.ndarray,
    param: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context_order = list(report["context_order"])
    meta = report.get("context_meta") or {}
    means = np.mean(score_matrix, axis=0)
    stds = np.std(score_matrix, axis=0)
    by_name = {name: (means[i], stds[i]) for i, name in enumerate(context_order)}
    default_mean, default_std = by_name.get("default", by_name.get("train", (np.nan, 0.0)))
    xs = [1.0]
    ys = [default_mean]
    yerr = [default_std]
    for name, (m, s) in by_name.items():
        info = meta.get(name) or {}
        if info.get("varied_param") != param:
            continue
        xs.append(float(info.get("factor", np.nan)))
        ys.append(m)
        yerr.append(s)
    order = np.argsort(xs)
    return (
        np.asarray(xs, dtype=float)[order],
        np.asarray(ys, dtype=float)[order],
        np.asarray(yerr, dtype=float)[order],
    )


def plot_context_factor_slices(
    report: dict[str, Any],
    score_matrix: np.ndarray,
    *,
    axes: np.ndarray | None = None,
    params: tuple[str, ...] = ("ball_mass", "ball_friction", "ball_elasticity"),
) -> np.ndarray:
    """C.2-style curves: return vs single-parameter factor with the ID window shaded.

    ``score_matrix`` is ``(n_seeds, n_contexts)`` aligned with ``report['context_order']``.
    """
    return plot_context_factor_slices_many(
        [(report, score_matrix)],
        axes=axes,
        params=params,
    )


def plot_context_factor_slices_many(
    entries: list[tuple[dict[str, Any], np.ndarray]],
    *,
    axes: np.ndarray | None = None,
    params: tuple[str, ...] = ("ball_mass", "ball_friction", "ball_elasticity"),
) -> np.ndarray:
    """Overlay C.2-style factor slices for several methods."""
    if not entries:
        raise ValueError("Need at least one (report, score_matrix) pair.")
    if axes is None:
        _, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 3.6), sharey=True)
    if len(params) == 1:
        axes = np.array([axes])

    id_low, id_high = _id_window(entries[0][0])
    from evaluation.diagnostics import slice_trapezoid_stats

    for ax, param in zip(axes, params):
        ax.axvspan(id_low, id_high, color="#55A868", alpha=0.18, label="ID window")
        flat_methods: list[str] = []
        for idx, (report, matrix) in enumerate(entries):
            color = METHOD_COLORS[idx % len(METHOD_COLORS)]
            label = report.get("label") or report["exp_name"]
            xs_a, ys_a, ye_a = _slice_xy(report, matrix, param)
            ax.plot(xs_a, ys_a, marker="o", color=color, lw=2, label=label)
            ax.fill_between(xs_a, ys_a - ye_a, ys_a + ye_a, color=color, alpha=0.15)
            slice_rows = slice_trapezoid_stats(
                np.asarray(matrix, dtype=float),
                list(report["context_order"]),
                report.get("context_meta"),
            )
            row = next((r for r in slice_rows if r["param"] == param), None)
            if row and row.get("flat"):
                flat_methods.append(str(label))
        ax.set_xlabel(f"{param} factor (× default)")
        title = param
        if flat_methods:
            title = f"{param}  [FLAT]"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean team return (across seeds)")
    axes[0].legend(loc="best", fontsize=8)
    fig = axes[0].figure
    fig.suptitle("Single-parameter generalization (shaded = in-distribution)", y=1.02)
    return axes


def plot_generalization_contexts(
    report: dict[str, Any],
    score_matrix: np.ndarray,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Mean return ± seed std for each evaluation context (ID highlighted)."""
    mapping = _mean_std_by_context(report, score_matrix)
    contexts = [c for c in report["context_order"] if c in mapping]
    means = np.array([mapping[c][0] for c in contexts], dtype=float)
    stds = np.array([mapping[c][1] for c in contexts], dtype=float)
    meta = report.get("context_meta") or {}

    def _is_id(name: str) -> bool:
        if name in meta:
            return meta[name].get("split") == "id"
        return name in ("train", "default")

    colors = ["#55A868" if _is_id(c) else "#C44E52" for c in contexts]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(contexts))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(contexts, rotation=35, ha="right")
    id_name = next((c for c in ("default", "train") if c in contexts), None)
    if id_name is not None:
        ax.axhline(means[contexts.index(id_name)], color="gray", ls="--", lw=1)
    ax.set_ylabel("Mean team return (across seeds)")
    ax.set_title("Zero-shot performance by context (green = in-distribution)")
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_sample_efficiency_curve(
    report: dict[str, Any],
    *,
    ax: plt.Axes | None = None,
    color: str | None = None,
    label: str | None = None,
) -> plt.Axes | None:
    """IQM in-distribution eval return vs training step (if present in report)."""
    curve = report.get("sample_efficiency", {}).get("eval_iqm_curve", {})
    if "error" in curve or "global_steps" not in curve:
        return None
    steps = np.asarray(curve["global_steps"])
    iqm = np.asarray(curve["iqm"])
    ci_low = np.asarray(curve["ci_low"])
    ci_high = np.asarray(curve["ci_high"])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    line_label = label or report.get("label") or report.get("exp_name")
    ax.plot(steps, iqm, lw=2, color=color, label=line_label)
    ax.fill_between(steps, ci_low, ci_high, alpha=0.2, color=color)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("IQM team return (in-distribution eval)")
    ax.set_title("Sample efficiency (rliable IQM across seeds)")
    ax.grid(True, alpha=0.3)
    return ax


def plot_sample_efficiency_overlay(
    reports: list[dict[str, Any]],
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes | None:
    """Overlay sample-efficiency IQM curves for several methods."""
    drawn = None
    for idx, report in enumerate(reports):
        drawn = plot_sample_efficiency_curve(
            report,
            ax=ax if drawn is None else drawn,
            color=METHOD_COLORS[idx % len(METHOD_COLORS)],
            label=report.get("label") or report.get("exp_name"),
        ) or drawn
        if drawn is not None:
            ax = drawn
    if drawn is not None:
        drawn.legend(loc="best")
    return drawn


def plot_seed_health(
    reports: list[dict[str, Any]],
    matrices: list[np.ndarray],
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes | None:
    """Default-context return per seed; collapsed seeds are hatched."""
    rows = seed_health_rows(reports, matrices)
    if not rows:
        return None
    methods = list(dict.fromkeys(r["method"] for r in rows))
    seeds = list(dict.fromkeys(int(r["seed"]) for r in rows))
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(seeds) + 2.0), 4.0))
    n = len(methods)
    x = np.arange(len(seeds))
    width = 0.8 / max(n, 1)
    for idx, method in enumerate(methods):
        vals = []
        collapsed = []
        for seed in seeds:
            match = next((r for r in rows if r["method"] == method and int(r["seed"]) == seed), None)
            vals.append(float("nan") if match is None else float(match["default_return"]))
            collapsed.append(bool(match and match.get("collapsed")))
        offset = (idx - (n - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=method,
            color=METHOD_COLORS[idx % len(METHOD_COLORS)],
            alpha=0.85,
        )
        for bar, is_collapsed in zip(bars, collapsed):
            if is_collapsed:
                bar.set_hatch("//")
                bar.set_edgecolor("#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Default-context return")
    ax.set_title("Seed health (hatched = collapsed)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def save_figure_bundle(report_path: Path, output_dir: Path, score_matrix: np.ndarray | None = None) -> None:
    """Write standard evaluation figures next to the JSON report."""
    report = load_report(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    exp = report["exp_name"]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_performance_profile(report, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{exp}_performance_profile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_aggregate_intervals(report, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{exp}_aggregate_metrics.png", dpi=150)
    plt.close(fig)

    if score_matrix is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_generalization_contexts(report, score_matrix, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{exp}_contexts.png", dpi=150)
        plt.close(fig)
        if report.get("context_meta"):
            axes = plot_context_factor_slices(report, score_matrix)
            fig = axes[0].figure
            fig.tight_layout()
            fig.savefig(output_dir / f"{exp}_context_slices.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    ax = plot_sample_efficiency_curve(report)
    if ax is not None:
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(output_dir / f"{exp}_sample_efficiency.png", dpi=150)
        plt.close(fig)
