"""Integration tests for rliable_aggregate CLI."""

import json
import sys
from pathlib import Path

import pytest

from evaluation.rliable_utils import discover_run_dirs, save_rliable_scores
from scripts.rliable_aggregate import Args, main


def _write_fake_run(run_dir: Path, seed: int, *, exp_tag: str = "baseline5_shared_randdrop", method_tag: str = "hidden_context"):
    save_rliable_scores(
        run_dir,
        algorithm="matd3_pistonball",
        method_tag=method_tag,
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
        "finished": True,
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f)
    with open(run_dir / "eval_history.jsonl", "w") as f:
        f.write(json.dumps({"global_step": 100, "return_mean": 10.0}) + "\n")
        f.write(json.dumps({"global_step": 200, "return_mean": 20.0 + seed}) + "\n")


def test_discover_run_dirs_new_naming(tmp_path):
    run_dir = tmp_path / "context5_concat_randdrop__concat__shared__randdrop__n5__seed3__9999"
    run_dir.mkdir(parents=True)
    _write_fake_run(run_dir, 3, exp_tag="context5_concat_randdrop", method_tag="context_concat")
    found = discover_run_dirs(tmp_path, "context5_concat_randdrop", method_tag="context_concat")
    assert found == [run_dir]


def test_rliable_aggregate_script_entrypoint(tmp_path, monkeypatch):
    """Ensure CLI works when invoked as a script (cluster usage)."""
    import os
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    runs = tmp_path / "runs"
    for seed in (3,):
        run_dir = runs / f"baseline5_shared_randdrop__static__shared__randdrop__n5__seed{seed}__2000"
        run_dir.mkdir(parents=True)
        _write_fake_run(run_dir, seed)

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "rliable_aggregate.py"),
            "--preset",
            "baseline5",
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
    assert (tmp_path / "reports" / "rliable_baseline5_shared_randdrop.json").exists()


def test_rliable_aggregate_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    for seed in (1, 2):
        run_dir = runs / f"baseline5_shared_randdrop__static__shared__randdrop__n5__seed{seed}__1000"
        run_dir.mkdir(parents=True)
        _write_fake_run(run_dir, seed)

    out = main(
        Args(
            preset="baseline5",
            runs_root="runs",
            bootstrap_reps=100,
            profile_steps=11,
        )
    )
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["label"] == "Hidden context"
    assert report["seeds"] == [1, 2]
    assert "performance_profiles" in report["rliable"]
    assert "per_context" in report
    iqm = report["rliable"]["algorithms"]["Hidden context"]["metrics"]["iqm"]
    assert iqm["ci_low"] <= iqm["point"] <= iqm["ci_high"]


def test_rliable_aggregate_compare_presets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    for seed in (1, 2):
        base_dir = runs / f"baseline5_shared_randdrop__shared__randdrop__n5__seed{seed}__1000"
        base_dir.mkdir(parents=True)
        _write_fake_run(base_dir, seed, method_tag="hidden_context")
        ctx_dir = runs / f"context5_concat_randdrop__concat__varying__shared__randdrop__n5__seed{seed}__2000"
        ctx_dir.mkdir(parents=True)
        _write_fake_run(
            ctx_dir,
            seed,
            exp_tag="context5_concat_randdrop",
            method_tag="context_concat",
        )
        save_rliable_scores(
            ctx_dir,
            algorithm="matd3_pistonball_context",
            method_tag="context_concat",
            seed=seed,
            global_step=1000,
            context_order=["train", "ball_mass_x2"],
            scores_by_context={"train": [70.0, 80.0], "ball_mass_x2": [55.0, 65.0]},
            eval_split="zero_shot_transfer",
            reference_return=100.0,
        )

    out = main(
        Args(
            preset="context5",
            compare_preset="baseline5",
            runs_root="runs",
            bootstrap_reps=100,
            profile_steps=11,
        )
    )
    payload = json.loads(out.read_text())
    assert "comparison" in payload
    assert "rliable_combined" in payload
    assert payload["primary"]["label"] == "Context concat"
    assert payload["challenger"]["label"] == "Hidden context"
