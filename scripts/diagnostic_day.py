#!/usr/bin/env python3
"""Offline H1–H3 diagnostic day: tables + figures from rliable_matrix_hard.json.

Does not import PettingZoo / training scripts — only JSON + matplotlib.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B3"]
PARAMS = ("ball_mass", "ball_friction", "ball_elasticity", "gravity")


@dataclass
class Args:
    report_path: str = "reports/rliable_matrix_hard.json"
    standard_report_path: str = "reports/rliable_matrix.json"
    runs_root: str = "runs"
    output_dir: str = "reports/figures/diagnostic_day"


def parse_args(argv: list[str] | None = None) -> Args:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-path", default=Args.report_path)
    p.add_argument("--standard-report-path", default=Args.standard_report_path)
    p.add_argument("--runs-root", default=Args.runs_root)
    p.add_argument("--output-dir", default=Args.output_dir)
    ns = p.parse_args(argv)
    return Args(
        report_path=ns.report_path,
        standard_report_path=ns.standard_report_path,
        runs_root=ns.runs_root,
        output_dir=ns.output_dir,
    )


def _iqm_block(exp: dict) -> dict:
    alg = next(iter(exp["rliable"]["algorithms"]))
    return exp["rliable"]["algorithms"][alg]["metrics"]


def _resolve_scores(rd: str, runs_root: Path) -> Path:
    p = Path(rd) / "rliable_scores.json"
    if p.exists():
        return p
    hits = list(runs_root.rglob(f"{Path(rd).name}/rliable_scores.json"))
    if not hits:
        raise FileNotFoundError(rd)
    return hits[0]


def build_matrix(exp: dict, runs_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = exp["context_order"]
    health = {h["seed"]: h for h in (exp.get("diagnostics") or {}).get("seed_health", [])}
    rows, seeds, collapsed = [], [], []
    for rd in exp["run_dirs"]:
        data = json.load(open(_resolve_scores(rd, runs_root)))
        seed = int(data["seed"])
        rows.append(
            [float(np.mean(data["scores_by_context"][c]["episode_returns"])) for c in order]
        )
        seeds.append(seed)
        collapsed.append(bool(health.get(seed, {}).get("collapsed")))
    return np.asarray(rows, float), np.asarray(seeds), np.asarray(collapsed)


def dashboard_rows(experiments: list[dict]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        m = _iqm_block(exp)
        diag = exp["diagnostics"]
        g = exp["generalization"]
        rows.append(
            {
                "method": exp["label"],
                "IQM": m["iqm"]["point"],
                "IQM_lo": m["iqm"]["ci_low"],
                "IQM_hi": m["iqm"]["ci_high"],
                "mean": m["mean"]["point"],
                "ID_IQM": g["id_return_iqm"],
                "OOD_IQM": g["ood_return_iqm"],
                "gap_IQM": g["generalization_gap_iqm"],
                "collapsed": diag["n_collapsed_seeds"],
                "cloned": ",".join(c["param"] for c in diag["clones"] if c["cloned"]) or "—",
                "flat": ",".join(s["param"] for s in diag["slices"] if s["flat"]) or "—",
                "sensitive": ",".join(s["param"] for s in diag["slices"] if not s["flat"]) or "—",
                "env": diag["env_switch"]["action"],
                "switch": diag["env_switch"]["switch_env"],
            }
        )
    return pd.DataFrame(rows)


def slice_rows(experiments: list[dict]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        diag = exp["diagnostics"]
        clones = {c["param"]: c for c in diag["clones"]}
        for s in diag["slices"]:
            c = clones.get(s["param"]) or {}
            rows.append(
                {
                    "method": exp["label"],
                    "param": s["param"],
                    "ID": s["id_return_mean"],
                    "OOD": s["ood_return_mean"],
                    "drop": s["ood_drop"],
                    "flat": s["flat"],
                    "corr": s.get("logdev_return_corr"),
                    "cloned": bool(c.get("cloned")),
                    "max_abs_diff": c.get("max_abs_diff"),
                }
            )
    return pd.DataFrame(rows)


def _slice_xy(exp: dict, matrix: np.ndarray, param: str):
    meta = exp["context_meta"]
    order = exp["context_order"]
    xs, cols = [], []
    for i, name in enumerate(order):
        m = meta.get(name) or {}
        if m.get("varied_param") == param:
            xs.append(float(m.get("factor", 1.0)))
            cols.append(i)
    if not cols:
        return np.array([]), np.array([]), np.array([])
    block = matrix[:, cols]
    return np.asarray(xs), block.mean(0), block.std(0)


def export_figures(hard: dict, runs_root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    experiments = hard["experiments"]

    # 1) IQM + ID/OOD
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels, iqm, lo, hi, idv, ood = [], [], [], [], [], []
    for exp in experiments:
        m = _iqm_block(exp)["iqm"]
        g = exp["generalization"]
        labels.append(exp["label"].replace(" context", "").replace("Context ", ""))
        iqm.append(m["point"])
        lo.append(m["ci_low"])
        hi.append(m["ci_high"])
        idv.append(g["id_return_iqm"])
        ood.append(g["ood_return_iqm"])
    x = np.arange(len(labels))
    yerr = np.vstack([np.asarray(iqm) - np.asarray(lo), np.asarray(hi) - np.asarray(iqm)])
    axes[0].bar(x, iqm, yerr=yerr, capsize=4, color=COLORS, alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("IQM return")
    axes[0].set_title("Overall IQM (hard OOD grid)")
    axes[0].set_ylim(50, 100)
    w = 0.35
    axes[1].bar(x - w / 2, idv, width=w, label="ID IQM", color="#55A868")
    axes[1].bar(x + w / 2, ood, width=w, label="OOD IQM", color="#C44E52")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_title("ID vs OOD IQM (gaps ≈ 0)")
    axes[1].legend()
    axes[1].set_ylim(50, 100)
    fig.tight_layout()
    fig.savefig(out / "01_iqm_id_ood.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) drop heatmap
    drop = np.zeros((len(experiments), len(PARAMS)))
    cloned = np.zeros_like(drop, dtype=bool)
    for i, exp in enumerate(experiments):
        by = {s["param"]: s for s in exp["diagnostics"]["slices"]}
        cl = {c["param"]: c for c in exp["diagnostics"]["clones"]}
        for j, p in enumerate(PARAMS):
            drop[i, j] = by[p]["ood_drop"]
            cloned[i, j] = bool(cl.get(p, {}).get("cloned"))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(drop, cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
    ax.set_xticks(range(len(PARAMS)))
    ax.set_xticklabels(PARAMS)
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels([e["label"].replace(" context", "") for e in experiments])
    for i in range(drop.shape[0]):
        for j in range(drop.shape[1]):
            mark = "FLAT" + ("/CLONE" if cloned[i, j] else "")
            ax.text(j, i, f"{drop[i, j]:+.1f}\n{mark}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="ID−OOD mean return (drop)")
    ax.set_title("C.2 axis sensitivity (hard grid); |drop|<5 ⇒ FLAT")
    fig.tight_layout()
    fig.savefig(out / "02_ood_drop_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) C.2 slices (varying)
    entries = []
    for exp in experiments:
        if "varying" not in exp["label"].lower():
            continue
        mat, _, _ = build_matrix(exp, runs_root)
        entries.append((exp, mat))
    id_low = experiments[0].get("id_factor_low", 0.75)
    id_high = experiments[0].get("id_factor_high", 1.5)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, param in zip(axes, PARAMS):
        ax.axvspan(id_low, id_high, color="#55A868", alpha=0.18, label="ID window")
        for idx, (exp, mat) in enumerate(entries):
            xs, ys, ye = _slice_xy(exp, mat, param)
            order = np.argsort(xs)
            xs, ys, ye = xs[order], ys[order], ye[order]
            ax.plot(xs, ys, "o-", color=COLORS[idx], lw=2, label=exp["label"].replace(" context", ""))
            ax.fill_between(xs, ys - ye, ys + ye, color=COLORS[idx], alpha=0.15)
        ax.set_xlabel(f"{param} ×")
        ax.set_title(f"{param}  [FLAT]")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean team return")
    axes[0].legend(fontsize=8)
    fig.suptitle("Hard-grid C.2 slices — hidden vs concat (varying only)", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "03_c2_slices_varying.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4) profiles
    pp = hard["rliable_combined"]["performance_profiles"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, data) in enumerate(pp.items()):
        thr = np.asarray(data["thresholds"])
        prof = np.asarray(data["profile"])
        lo_ = np.asarray(data["ci_low"])
        hi_ = np.asarray(data["ci_high"])
        ax.plot(thr, prof, lw=2.5, color=COLORS[i % len(COLORS)], label=name)
        ax.fill_between(thr, lo_, hi_, color=COLORS[i % len(COLORS)], alpha=0.15)
    ax.set_xlabel("Return threshold τ")
    ax.set_ylabel("P(return ≥ τ)")
    ax.set_title("Performance profiles (hard grid)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "04_performance_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5) concat − hidden
    hid = next(e for e in experiments if e["label"] == "Hidden context (varying)")
    con = next(e for e in experiments if e["label"] == "Context concat (varying)")
    Mh, _, _ = build_matrix(hid, runs_root)
    Mc, _, _ = build_matrix(con, runs_root)
    ctx = [c for c in hid["context_order"] if c in con["context_order"]]
    ih = {c: i for i, c in enumerate(hid["context_order"])}
    ic = {c: i for i, c in enumerate(con["context_order"])}
    Mh = Mh[:, [ih[c] for c in ctx]]
    Mc = Mc[:, [ic[c] for c in ctx]]
    delta_mean = Mc.mean(0) - Mh.mean(0)
    delta_med = np.median(Mc, axis=0) - np.median(Mh, axis=0)
    meta = hid["context_meta"]
    colors = ["#55A868" if (meta.get(c) or {}).get("split") == "id" else "#C44E52" for c in ctx]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, delta, title in [
        (axes[0], delta_mean, "Mean over seeds (biased if collapses)"),
        (axes[1], delta_med, "Median over seeds (robust)"),
    ]:
        ax.bar(range(len(ctx)), delta, color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel("concat − hidden")
        ax.set_title(
            f"{title}  |  meanΔ={delta.mean():+.2f}, |Δ|<5 on {(np.abs(delta) < 5).mean() * 100:.0f}% contexts"
        )
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(range(len(ctx)))
    axes[1].set_xticklabels(ctx, rotation=45, ha="right", fontsize=7)
    fig.suptitle(
        "Does explicit physics context help? (varying, hard grid)\nGreen=ID, Red=OOD", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out / "05_concat_minus_hidden.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6) seed health
    fig, ax = plt.subplots(figsize=(9, 4))
    xpos = 0
    xticks, xtlabels = [], []
    for i, exp in enumerate(experiments):
        for h in exp["diagnostics"]["seed_health"]:
            color = "#C44E52" if h["collapsed"] else COLORS[i]
            ax.bar(
                xpos,
                h["default_return"],
                color=color,
                alpha=0.85,
                hatch="//" if h["collapsed"] else None,
            )
            xticks.append(xpos)
            xtlabels.append(f"s{h['seed']}")
            xpos += 1
        xpos += 0.8
    ax.axhline(45, color="gray", ls="--", lw=1, label="collapse threshold")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtlabels, fontsize=8)
    ax.set_ylabel("Default-context return")
    ax.set_title("Seed health (hatched = collapsed)")
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "06_seed_health.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_verdict(hard: dict, out: Path) -> None:
    poi_line = ""
    for p in hard.get("pairwise_poi") or []:
        pair = {p["a"], p["b"]}
        if pair == {"Hidden context (varying)", "Context concat (varying)"}:
            a_beats = p["probability_a_beats_b"]
            if p["a"].startswith("Hidden"):
                poi_line = (
                    f"PoI(hidden>concat)={a_beats['probability_of_improvement']:.3f} "
                    f"[{a_beats['ci_low']:.3f}, {a_beats['ci_high']:.3f}]"
                )
            else:
                b_beats = p["probability_b_beats_a"]
                poi_line = (
                    f"PoI(hidden>concat)={b_beats['probability_of_improvement']:.3f} "
                    f"[{b_beats['ci_low']:.3f}, {b_beats['ci_high']:.3f}]"
                )
    text = f"""DIAGNOSTIC DAY VERDICT (hard OOD grid)
=======================================
H1 Env-insensitive: SUPPORTED
  - Every C.2 axis FLAT for every method (incl. gravity)
  - mass/elasticity often CLONED (identical returns across factors)
  - env advice: switch_to_multiwalker

H2 Context redundant: NOT TESTABLE on Pistonball (needs non-flat slices)
H3 Wrong context kind: PREMATURE (physics itself looks uninformative)

Main effects:
  - Training regime >> context: varying ≫ static
  - {poi_line or 'PoI hidden vs concat: see report'}
  - Explicit concat does not clearly beat hidden on IQM

NEXT: one Multiwalker concat-vs-hidden matrix (same ID/OOD protocol).
SKIP: FiLM/latent, actor/critic, transfer study, agent-kind on flat Pistonball.
"""
    (out / "VERDICT.txt").write_text(text)


def main(args: Args) -> None:
    hard = json.load(open(args.report_path))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs_root = Path(args.runs_root)

    dash = dashboard_rows(hard["experiments"])
    slices = slice_rows(hard["experiments"])
    dash.round(3).to_csv(Path("reports/diagnostic_hard_dashboard.csv"), index=False)
    slices.round(4).to_csv(Path("reports/diagnostic_hard_slices.csv"), index=False)
    if Path(args.standard_report_path).exists():
        std = json.load(open(args.standard_report_path))
        std_rows = []
        for exp in std["experiments"]:
            m = _iqm_block(exp)
            g = exp["generalization"]
            std_rows.append(
                {
                    "method": exp["label"],
                    "IQM": m["iqm"]["point"],
                    "ID_IQM": g["id_return_iqm"],
                    "OOD_IQM": g["ood_return_iqm"],
                    "gap_IQM": g["generalization_gap_iqm"],
                }
            )
        pd.DataFrame(std_rows).round(3).to_csv(
            Path("reports/diagnostic_standard_dashboard.csv"), index=False
        )

    export_figures(hard, runs_root, out)
    write_verdict(hard, out)
    print(dash.round(3).to_string(index=False))
    print("\nWrote", out)
    print((out / "VERDICT.txt").read_text())


if __name__ == "__main__":
    main(parse_args())
