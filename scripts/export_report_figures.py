#!/usr/bin/env python3
"""Report figures from the JSON already under reports/.

No environment and no checkpoint reload. Each figure is one entry in FIGURES.
Pass ``--only`` with a comma-separated list of those ids to redraw a subset.
Outputs land in ``reports/figures/report/`` and do not replace the diagnostic
PNGs already under ``reports/figures/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS = _ROOT / "reports"

# Place in the report, then why the panel exists. Written out as FIGURES.md.
CATALOG = [
    {
        "id": "pb_iqm_id_ood",
        "file": "pb_iqm_id_ood.png",
        "place": "Results, Pistonball (H1 supported)",
        "why": (
            "The 2×2: varying training beats static for both methods, and "
            "in-distribution and out-of-distribution IQM match. Concat does not beat hidden. "
            "This is the null result before any slice diagnosis."
        ),
        "source": "reports/rliable_matrix_hard.json",
    },
    {
        "id": "pb_axis_drops",
        "file": "pb_axis_drops.png",
        "place": "Results, Pistonball, next to the 2×2",
        "why": (
            "Per-axis ID−OOD drop with FLAT and CLONE marks. This is why the null is a "
            "validity failure: the factors do not move return, so concatenation has nothing to condition on."
        ),
        "source": "reports/rliable_matrix_hard.json",
    },
    {
        "id": "pb_slices",
        "file": "pb_slices_varying.png",
        "place": "Results, Pistonball, or the H1 paragraph in the Discussion",
        "why": (
            "Return against factor for the varying runs. Flat lines are the same claim as the drop table, "
            "in the form a reader can see. Static runs are omitted so the panel stays the sensitive-axis check."
        ),
        "source": "reports/rliable_matrix_hard.json",
    },
    {
        "id": "pb_learning",
        "file": "pb_learning.png",
        "place": "Results, sample efficiency, one short paragraph",
        "why": (
            "In-distribution eval IQM during training. The curve that pulls away is the varying regime, "
            "for hidden and concat alike. Secondary to the return gap; do not use it as evidence for context."
        ),
        "source": "reports/rliable_matrix_hard.json",
    },
    {
        "id": "mw_id_ood",
        "file": "mw_id_ood.png",
        "place": "Results, Multiwalker (H1 rejected, H2 rejected)",
        "why": (
            "The split the overall IQM hides: concat is far above hidden in distribution and below it "
            "out of distribution. Lead with the right-hand panel."
        ),
        "source": "reports/rliable_multiwalker.json",
    },
    {
        "id": "mw_slices",
        "file": "mw_slices.png",
        "place": "Results, Multiwalker, after the ID/OOD bars",
        "why": (
            "Gravity, package mass, and terrain friction all move return, so H1 fails here. "
            "Concat's curves fall off outside the shaded training window."
        ),
        "source": "reports/rliable_multiwalker.json",
    },
    {
        "id": "clip_id_ood",
        "file": "clip_id_ood.png",
        "place": "Results, clip probe (feature-extrapolation follow-up)",
        "why": (
            "Same concat policies with the context vector clamped and the physics left unchanged. "
            "In-distribution IQM stays put; out-of-distribution IQM recovers past the hidden baseline. "
            "The existing overall-IQM bar chart does not show this split."
        ),
        "source": "reports/rliable_multiwalker_context_clip.json",
    },
    {
        "id": "clip_axis_drops",
        "file": "clip_axis_drops.png",
        "place": "Results, clip probe, or the extrapolation paragraph in the Discussion",
        "why": (
            "Mass and friction drops shrink to about the hidden baseline. Gravity shrinks and still "
            "costs a large penalty. That is the two-failure-mode reading of the probe."
        ),
        "source": "reports/rliable_multiwalker_context_clip.json",
    },
    {
        "id": "clip_slices",
        "file": "clip_slices.png",
        "place": "Results, clip probe, if a third panel fits; otherwise the drop bars are enough",
        "why": (
            "Factor curves for hidden, concat, and clipped concat. "
            "Do not use reports/figures/multiwalker_context_clip/compare_context_slices.png: "
            "that file was drawn with the Pistonball axis list and does not show these factors."
        ),
        "source": "reports/rliable_multiwalker_context_clip.json",
    },
    {
        "id": "n10_eval",
        "file": "n10_eval.png",
        "place": "Appendix, N=10 curriculum pilot. One sentence in the Discussion points here",
        "why": (
            "Native-task eval during the seed-1 curriculum. The curve is not a context comparison: "
            "one seed, and the stage advances were mostly the episode cap. "
            "Built from reports/n10_curriculum_pilot.json (scripts/collect_report_series.py)."
        ),
        "source": "reports/n10_curriculum_pilot.json",
    },
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _experiments(path: Path) -> list[dict]:
    payload = _load(path)
    if "experiments" not in payload:
        raise SystemExit(f"{path} has no 'experiments' list")
    return payload["experiments"]


def _iqm(block: dict) -> dict:
    alg = next(iter(block["algorithms"]))
    return block["algorithms"][alg]["metrics"]["iqm"]


def _short(label: str) -> str:
    text = label.replace(" context", "").replace("Context ", "")
    text = text.replace("Multiwalker ", "")
    text = text.replace(" (varying, fwd×5)", "")
    text = text.replace(", context clipped to ID", ", clipped")
    return text


def _params(exp: dict) -> list[str]:
    seen: list[str] = []
    for name in exp["context_order"]:
        param = (exp["context_meta"].get(name) or {}).get("varied_param")
        if param and param not in seen:
            seen.append(param)
    return seen


def _slice_points(exp: dict, param: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = exp["context_meta"]
    per = exp["per_context"]
    xs, ys, ye = [1.0], [], []
    default = per.get("default") or per.get("train")
    if default is None:
        xs = []
    else:
        ys.append(float(default["return_mean"]))
        ye.append(float(default.get("return_std") or 0.0))
    for name, info in meta.items():
        if info.get("varied_param") != param or name not in per:
            continue
        xs.append(float(info["factor"]))
        ys.append(float(per[name]["return_mean"]))
        ye.append(float(per[name].get("return_std") or 0.0))
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order], np.asarray(ye)[order]


def _colors(n: int) -> list[str]:
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    return [palette[i % len(palette)] for i in range(n)]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(_ROOT)}")


def plot_iqm_id_ood(exps: list[dict], path: Path, title: str) -> None:
    labels = [_short(e["label"]) for e in exps]
    colors = _colors(len(exps))
    overall = [_iqm(e["rliable"]) for e in exps]
    id_iqm = [_iqm(e["rliable_by_split"]["in_distribution"]) for e in exps]
    ood_iqm = [_iqm(e["rliable_by_split"]["out_of_distribution"]) for e in exps]
    fig, axes = plt.subplots(1, 2, figsize=(max(8.5, 1.5 * len(exps) + 4), 4.2))
    x = np.arange(len(exps))
    yerr = np.vstack(
        [
            [p["point"] - p["ci_low"] for p in overall],
            [p["ci_high"] - p["point"] for p in overall],
        ]
    )
    axes[0].bar(x, [p["point"] for p in overall], yerr=yerr, capsize=3, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("IQM return")
    axes[0].set_title("Overall IQM")
    axes[0].axhline(0, color="black", lw=0.6)
    width = 0.36
    axes[1].bar(
        x - width / 2,
        [p["point"] for p in id_iqm],
        width,
        label="In distribution",
        color="#55A868",
        yerr=np.vstack(
            [
                [p["point"] - p["ci_low"] for p in id_iqm],
                [p["ci_high"] - p["point"] for p in id_iqm],
            ]
        ),
        capsize=3,
    )
    axes[1].bar(
        x + width / 2,
        [p["point"] for p in ood_iqm],
        width,
        label="Out of distribution",
        color="#C44E52",
        yerr=np.vstack(
            [
                [p["point"] - p["ci_low"] for p in ood_iqm],
                [p["ci_high"] - p["point"] for p in ood_iqm],
            ]
        ),
        capsize=3,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_title("In-distribution vs out-of-distribution")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].legend(frameon=False)
    fig.suptitle(title)
    _save(fig, path)


def plot_drop_heatmap(exps: list[dict], path: Path, title: str) -> None:
    """Cell chart for near-zero drops, where grouped bars bury the FLAT marks."""
    params: list[str] = []
    for exp in exps:
        for param in _params(exp):
            if param not in params:
                params.append(param)
    drop = np.zeros((len(exps), len(params)))
    marks = [["" for _ in params] for _ in exps]
    for i, exp in enumerate(exps):
        by = {s["param"]: s for s in exp["diagnostics"]["slices"]}
        clones = {c["param"]: c for c in exp["diagnostics"].get("clones") or []}
        for j, param in enumerate(params):
            info = by.get(param) or {}
            drop[i, j] = float(info.get("ood_drop") or 0.0)
            tags = []
            if info.get("flat"):
                tags.append("FLAT")
            if clones.get(param, {}).get("cloned"):
                tags.append("CLONE")
            marks[i][j] = "\n".join(tags)
    limit = 5.0 if np.nanmax(np.abs(drop)) < 8 else float(np.nanmax(np.abs(drop)))
    fig, ax = plt.subplots(figsize=(1.6 * len(params) + 3, 0.7 * len(exps) + 2))
    image = ax.imshow(drop, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=15, ha="right")
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels([_short(e["label"]) for e in exps])
    for i in range(drop.shape[0]):
        for j in range(drop.shape[1]):
            ax.text(j, i, f"{drop[i, j]:+.1f}\n{marks[i][j]}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="In-distribution − out-of-distribution mean return")
    ax.set_title(title)
    _save(fig, path)


def plot_drops(exps: list[dict], path: Path, title: str) -> None:
    params: list[str] = []
    for exp in exps:
        for param in _params(exp):
            if param not in params:
                params.append(param)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(params) + 3), 4.2))
    x = np.arange(len(params))
    width = 0.8 / max(len(exps), 1)
    for i, exp in enumerate(exps):
        by = {s["param"]: s for s in exp["diagnostics"]["slices"]}
        clones = {c["param"]: c for c in exp["diagnostics"].get("clones") or []}
        drops = [float(by[p]["ood_drop"]) if p in by else np.nan for p in params]
        bars = ax.bar(
            x + (i - (len(exps) - 1) / 2) * width,
            drops,
            width,
            label=_short(exp["label"]),
            color=_colors(len(exps))[i],
        )
        for bar, param, drop in zip(bars, params, drops):
            marks = []
            if param in by and by[param].get("flat"):
                marks.append("FLAT")
            if clones.get(param, {}).get("cloned"):
                marks.append("CLONE")
            if marks:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    "\n".join(marks),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=15, ha="right")
    ax.set_ylabel("In-distribution − out-of-distribution mean return")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_slices(exps: list[dict], path: Path, title: str) -> None:
    params = _params(exps[0])
    fig, axes = plt.subplots(1, len(params), figsize=(4.0 * len(params), 3.8), sharey=True)
    if len(params) == 1:
        axes = [axes]
    low = float(exps[0].get("id_factor_low") or 0.75)
    high = float(exps[0].get("id_factor_high") or 1.5)
    for ax, param in zip(axes, params):
        ax.axvspan(low, high, color="#55A868", alpha=0.15, label="Training window")
        for i, exp in enumerate(exps):
            xs, ys, ye = _slice_points(exp, param)
            color = _colors(len(exps))[i]
            ax.plot(xs, ys, "o-", color=color, lw=2, label=_short(exp["label"]))
            ax.fill_between(xs, ys - ye, ys + ye, color=color, alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel(f"{param} factor")
        ax.set_title(param)
        ax.axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("Mean return across seeds")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    _save(fig, path)


def plot_learning(exps: list[dict], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    drawn = False
    for i, exp in enumerate(exps):
        curve = (exp.get("sample_efficiency") or {}).get("eval_iqm_curve") or {}
        steps = curve.get("global_steps")
        if not steps:
            continue
        color = _colors(len(exps))[i]
        ax.plot(steps, curve["iqm"], lw=2, color=color, label=_short(exp["label"]))
        if "ci_low" in curve and "ci_high" in curve:
            ax.fill_between(steps, curve["ci_low"], curve["ci_high"], color=color, alpha=0.15)
        drawn = True
    if not drawn:
        plt.close(fig)
        print(f"skip {path.name}: no eval_iqm_curve in the report")
        return
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("IQM return (in-distribution eval)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_n10(pilot_path: Path, path: Path) -> bool:
    if not pilot_path.exists():
        print(f"skip {path.name}: {pilot_path} is missing. Run scripts/collect_report_series.py")
        return False
    payload = _load(pilot_path)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for i, run in enumerate(payload["runs"]):
        rows = run["in_training_eval"]
        if not rows:
            continue
        color = _colors(len(payload["runs"]))[i]
        steps = [r["global_step"] for r in rows]
        axes[0].plot(steps, [r["return_mean"] for r in rows], "o-", color=color, lw=2, label=run["method"])
        axes[1].plot(steps, [r["success_rate"] for r in rows], "o-", color=color, lw=2, label=run["method"])
        if run.get("graduation_step"):
            for ax in axes:
                ax.axvline(run["graduation_step"], color=color, ls="--", lw=1, alpha=0.7)
    axes[0].set_ylabel("Native-task eval return")
    axes[1].set_ylabel("Native-task success rate")
    for ax in axes:
        ax.set_xlabel("Training environment steps")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("N=10 curriculum, seed 1 (dashed = graduation). Not a context result.")
    _save(fig, path)
    return True


def _varying_only(exps: list[dict]) -> list[dict]:
    chosen = [e for e in exps if "varying" in e["label"].lower()]
    return chosen or exps


def write_catalog(output_dir: Path, written: set[str]) -> None:
    lines = [
        "# Report figures",
        "",
        "Generated by `scripts/export_report_figures.py`. "
        "The diagnostic PNGs under `reports/figures/diagnostic_day/`, "
        "`multiwalker_matrix/`, and `multiwalker_context_clip/` stay as working plots. "
        "Use this folder in the report.",
        "",
        "Do not use `reports/figures/multiwalker_context_clip/compare_context_slices.png`. "
        "It was drawn with the Pistonball parameter list.",
        "",
        "Do not use `compare_sample_efficiency.png` in that same folder as a clip result. "
        "Clipping is an eval-time change, so that curve is the concat training run drawn twice.",
        "",
    ]
    for item in CATALOG:
        on_disk = (output_dir / item["file"]).exists()
        status = "written" if item["id"] in written or on_disk else "not written"
        lines.append(f"## `{item['file']}` ({status})")
        lines.append("")
        lines.append(f"- Place: {item['place']}")
        lines.append(f"- Why: {item['why']}")
        lines.append(f"- Source: `{item['source']}`")
        lines.append("")
    path = output_dir / "FIGURES.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path.relative_to(_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPORTS / "figures" / "report")
    parser.add_argument("--pistonball", type=Path, default=REPORTS / "rliable_matrix_hard.json")
    parser.add_argument("--multiwalker", type=Path, default=REPORTS / "rliable_multiwalker.json")
    parser.add_argument("--clip", type=Path, default=REPORTS / "rliable_multiwalker_context_clip.json")
    parser.add_argument("--n10", type=Path, default=REPORTS / "n10_curriculum_pilot.json")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated figure ids. Default: all of them.",
    )
    args = parser.parse_args()
    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    known = {item["id"] for item in CATALOG}
    unknown = wanted - known
    if unknown:
        raise SystemExit(f"Unknown figure id(s): {', '.join(sorted(unknown))}. Known: {', '.join(sorted(known))}")

    def selected(figure_id: str) -> bool:
        return not wanted or figure_id in wanted

    out = args.output_dir
    written: set[str] = set()

    def run(figure_id: str, fn) -> None:
        if not selected(figure_id):
            return
        fn()
        written.add(figure_id)

    if args.pistonball.exists():
        pb = _experiments(args.pistonball)
        run("pb_iqm_id_ood", lambda: plot_iqm_id_ood(pb, out / "pb_iqm_id_ood.png", "Pistonball, hard grid"))
        run(
            "pb_axis_drops",
            lambda: plot_drop_heatmap(
                pb,
                out / "pb_axis_drops.png",
                "Pistonball axis drop. |drop| < 5 is FLAT",
            ),
        )
        run(
            "pb_slices",
            lambda: plot_slices(
                _varying_only(pb),
                out / "pb_slices_varying.png",
                "Pistonball varying runs, one factor at a time",
            ),
        )
        run("pb_learning", lambda: plot_learning(pb, out / "pb_learning.png", "Pistonball in-distribution eval during training"))
    elif any(selected(i) for i in ("pb_iqm_id_ood", "pb_axis_drops", "pb_slices", "pb_learning")):
        print(f"skip Pistonball figures: {args.pistonball} is missing")

    if args.multiwalker.exists():
        mw = _experiments(args.multiwalker)
        run("mw_id_ood", lambda: plot_iqm_id_ood(mw, out / "mw_id_ood.png", "Multiwalker, hard grid"))
        run("mw_slices", lambda: plot_slices(mw, out / "mw_slices.png", "Multiwalker, one factor at a time"))
    elif any(selected(i) for i in ("mw_id_ood", "mw_slices")):
        print(f"skip Multiwalker figures: {args.multiwalker} is missing")

    if args.clip.exists():
        clip = _experiments(args.clip)
        run("clip_id_ood", lambda: plot_iqm_id_ood(clip, out / "clip_id_ood.png", "Multiwalker clip probe"))
        run(
            "clip_axis_drops",
            lambda: plot_drops(clip, out / "clip_axis_drops.png", "Drop after clamping the context vector"),
        )
        run("clip_slices", lambda: plot_slices(clip, out / "clip_slices.png", "Clip probe, one factor at a time"))
    elif any(selected(i) for i in ("clip_id_ood", "clip_axis_drops", "clip_slices")):
        print(f"skip clip figures: {args.clip} is missing")

    if selected("n10_eval") and plot_n10(args.n10, out / "n10_eval.png"):
        written.add("n10_eval")

    write_catalog(out, written)


if __name__ == "__main__":
    main()
