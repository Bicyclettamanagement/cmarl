#!/usr/bin/env python3
"""Collect report inputs that are not already an rliable JSON.

The hard-grid and Multiwalker reports already contain the returns used in the
write-up. This script does two further jobs:

- Build ``reports/n10_curriculum_pilot.json`` from the seed-1 curriculum logs
  (in-training eval, graduation, offline ID/OOD). No new rollouts.
- Check that the figure script's other inputs exist. With ``--aggregate-missing``
  it rebuilds a missing report from ``rliable_scores.json`` files already in the
  run directories. With ``--run-eval`` it launches ``evaluate_checkpoint.py``
  only when those score files themselves are absent.

``--run-eval`` is off by default. The clip probe and the N=5 / Multiwalker grids
are already evaluated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPORTS = _ROOT / "reports"

# Figure inputs. ``scores_hint`` is how to tell whether a missing report can be
# rebuilt locally (score files already on disk) or still needs a rollout.
REQUIRED_REPORTS = {
    "pistonball_hard": REPORTS / "rliable_matrix_hard.json",
    "multiwalker": REPORTS / "rliable_multiwalker.json",
    "multiwalker_clip": REPORTS / "rliable_multiwalker_context_clip.json",
}

AGGREGATE_COMMANDS = {
    "pistonball_hard": [
        sys.executable,
        "scripts/rliable_aggregate.py",
        "--presets",
        "baseline5_static_randdrop,baseline5_varying_randdrop,context5_concat_static_randdrop,context5_concat_varying_randdrop",
        "--output-name",
        "rliable_matrix_hard.json",
    ],
    "multiwalker": [sys.executable, "scripts/rliable_aggregate.py", "--compare-multiwalker"],
    "multiwalker_clip": [
        sys.executable,
        "scripts/rliable_aggregate.py",
        "--compare-multiwalker-clip",
    ],
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _curriculum_runs(runs_root: Path) -> list[Path]:
    root = runs_root / "pistonball" / "n10"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and "curriculum" in p.name)


def _method_label(run_dir: Path) -> str:
    name = run_dir.name
    if "concat" in name:
        return "concat"
    if "hidden" in name:
        return "hidden"
    return name.split("__")[0]


def _summarize_curriculum_run(run_dir: Path) -> dict:
    eval_rows = _read_jsonl(run_dir / "eval_history.jsonl")
    episodes = _read_jsonl(run_dir / "training_episodes.jsonl")
    offline_path = run_dir / "evaluation_results.json"
    offline = json.loads(offline_path.read_text()) if offline_path.exists() else None

    graduation_step = None
    n_advances = 0
    n_forced = 0
    post_success: list[float] = []
    for row in episodes:
        ctx = row.get("context") or {}
        if ctx.get("just_graduated") and graduation_step is None:
            graduation_step = int(row["global_step"])
        if ctx.get("advanced"):
            n_advances += 1
            if ctx.get("forced"):
                n_forced += 1
        if ctx.get("graduated") and not ctx.get("just_graduated"):
            post_success.append(1.0 if row.get("success") else 0.0)

    default = None
    if offline and "contexts" in offline:
        default = (offline["contexts"].get("default") or {})

    return {
        "run_dir": str(run_dir),
        "method": _method_label(run_dir),
        "seed": int((offline or {}).get("seed") or 1),
        "graduation_step": graduation_step,
        "n_advances": n_advances,
        "n_forced_advances": n_forced,
        "post_graduation_success_rate": (
            float(sum(post_success) / len(post_success)) if post_success else None
        ),
        "post_graduation_episodes": len(post_success),
        "in_training_eval": [
            {
                "global_step": int(row["global_step"]),
                "return_mean": float(row["return_mean"]),
                "success_rate": float(row.get("success_rate") or 0.0),
            }
            for row in eval_rows
        ],
        "offline": None
        if offline is None
        else {
            "id_return_mean": offline.get("id_return_mean"),
            "ood_return_mean": offline.get("ood_return_mean"),
            "generalization_gap": offline.get("generalization_gap"),
            "default_return_mean": None if default is None else default.get("return_mean"),
            "default_success_rate": None if default is None else default.get("success_rate"),
        },
    }


def write_n10_pilot(runs_root: Path, output: Path) -> Path | None:
    runs = _curriculum_runs(runs_root)
    if not runs:
        print(f"No curriculum runs under {runs_root / 'pistonball' / 'n10'}")
        return None
    payload = {
        "note": (
            "Seed-1 pilot. In-training eval is the native far-spawn task, not the "
            "short curriculum court. Do not put this in the main results table."
        ),
        "runs": [_summarize_curriculum_run(path) for path in runs],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output}")
    return output


def _score_files_present(kind: str, runs_root: Path) -> bool:
    """Cheap existence check. Aggregation itself discovers the real run set."""
    if kind == "pistonball_hard":
        root = runs_root / "pistonball"
    else:
        root = runs_root / "multiwalker"
    if not root.is_dir():
        return False
    name = "rliable_scores_context_clip.json" if kind == "multiwalker_clip" else "rliable_scores.json"
    return any(root.rglob(name))


def check_reports(
    *,
    aggregate_missing: bool,
    run_eval: bool,
    runs_root: Path,
) -> int:
    missing_scores = False
    for kind, path in REQUIRED_REPORTS.items():
        if path.exists():
            print(f"present  {path.relative_to(_ROOT)}")
            continue
        print(f"missing  {path.relative_to(_ROOT)}")
        if _score_files_present(kind, runs_root):
            cmd = AGGREGATE_COMMANDS[kind]
            print("  score files exist; rebuild with:")
            print("   ", " ".join(cmd))
            if aggregate_missing:
                subprocess.run(cmd, cwd=_ROOT, check=True)
        else:
            missing_scores = True
            print("  score files are also absent, so this needs evaluate_checkpoint.py")
            if kind == "multiwalker_clip":
                print(
                    "   ",
                    "python scripts/evaluate_checkpoint.py --exp-name mw3_concat_varying_fwd5 "
                    "--grid-preset hard --eval-episodes 10 --clip-context-to-id",
                )
            elif run_eval:
                print("  --run-eval has no generic launcher for this report; run the command above.")
    if missing_scores and run_eval:
        print("--run-eval was set, but only the clip command is specific enough to print.")
        print("Refusing to start a rollout from a partial guess.")
        return 2
    return 1 if missing_scores else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=_ROOT / "runs")
    parser.add_argument(
        "--n10-output",
        type=Path,
        default=REPORTS / "n10_curriculum_pilot.json",
    )
    parser.add_argument(
        "--aggregate-missing",
        action="store_true",
        help="Rebuild a missing rliable JSON from score files already on disk.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Allow a checkpoint eval when score files are absent. Off by default.",
    )
    args = parser.parse_args()
    write_n10_pilot(args.runs_root, args.n10_output)
    status = check_reports(
        aggregate_missing=args.aggregate_missing,
        run_eval=args.run_eval,
        runs_root=args.runs_root,
    )
    if status:
        print("Some figure inputs are still missing. See the commands above.")
    else:
        print("Figure inputs are on disk. Next: python scripts/export_report_figures.py")


if __name__ == "__main__":
    main()
