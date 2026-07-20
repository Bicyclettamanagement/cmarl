"""Integration tests for rliable_aggregate CLI."""

import json
import sys
from pathlib import Path

import pytest

from evaluation.rliable_utils import save_rliable_scores
from scripts.rliable_aggregate import Args, main


def _write_fake_run(run_dir, seed: int):
    save_rliable_scores(
        run_dir,
        algorithm="matd3_pistonball",
        method_tag="hidden_context",
        seed=seed,
        global_step=1000,
        context_order=["train", "ball_mass_x2"],
        scores_by_context={"train": [50.0, 60.0], "ball_mass_x2": [30.0, 40.0]},
        eval_split="zero_shot_transfer",
        reference_return=100.0,
    )
    manifest = {
        "args": {"seed": seed, "performance_threshold": 10.0},
        "steps_to_threshold": 5000 + seed,
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f)
    with open(run_dir / "eval_history.jsonl", "w") as f:
        f.write(json.dumps({"global_step": 100, "return_mean": 10.0}) + "\n")
        f.write(json.dumps({"global_step": 200, "return_mean": 20.0 + seed}) + "\n")


def test_rliable_aggregate_script_entrypoint(tmp_path, monkeypatch):
    """Ensure CLI works when invoked as a script (cluster usage)."""
    import os
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    runs = tmp_path / "runs"
    for seed in (3,):
        run_dir = runs / f"pistonball__matd3_pistonball__{seed}__2000"
        run_dir.mkdir(parents=True)
        _write_fake_run(run_dir, seed)

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "rliable_aggregate.py"),
            "--runs-root",
            str(runs),
            "--output-dir",
            str(tmp_path / "reports"),
            "--bootstrap-reps",
            "50",
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "reports" / "rliable_matd3_pistonball.json").exists()


def test_rliable_aggregate_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    for seed in (1, 2):
        run_dir = runs / f"pistonball__matd3_pistonball__{seed}__1000"
        run_dir.mkdir(parents=True)
        _write_fake_run(run_dir, seed)

    out = main(
        Args(
            runs_root="runs",
            exp_name="matd3_pistonball",
            bootstrap_reps=100,
            profile_steps=11,
        )
    )
    assert out.exists()
    report = json.loads(out.read_text())
    assert "rliable" in report
    assert report["seeds"] == [1, 2]
    assert "performance_profiles" in report["rliable"]
    iqm = report["rliable"]["algorithms"]["matd3_pistonball"]["metrics"]["iqm"]
    assert iqm["ci_low"] <= iqm["point"] <= iqm["ci_high"]
