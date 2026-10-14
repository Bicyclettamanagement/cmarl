"""Offline checkpoint evaluation on the 1D ID/OOD grid."""

from __future__ import annotations

import json

import pytest
import torch

from evaluation.checkpoint_eval import detect_family, evaluate_run
from matd3_pistonball import Args, CheckpointManager, RunManifest, build_agents, build_checkpoint_payload


@pytest.fixture
def args() -> Args:
    return Args(
        n_pistons=3,
        max_cycles=8,
        frame_size=32,
        frame_stack=2,
        feature_dim=16,
        hidden_dim=32,
        eval_episodes=1,
        share_actors=True,
        cuda=False,
        seed=0,
        eval_factor_grid=(0.5, 1.0, 2.0),
        id_factor_low=0.75,
        id_factor_high=1.5,
        experiment="unit_eval",
        method_tag="hidden_context",
    )


def test_evaluate_run_writes_id_ood_artifacts(tmp_path, args):
    device = torch.device("cpu")
    nets = build_agents(args, device)
    run_dir = tmp_path / "run"
    manifest = RunManifest(run_name="unit_eval", run_dir=run_dir, args=vars(args))
    manifest.finished = True
    manifest.write()
    manager = CheckpointManager(run_dir, metric_key="return_mean")
    manager.save(build_checkpoint_payload(nets, args), global_step=50, eval_stats={"return_mean": 1.0})

    results = evaluate_run(run_dir, eval_episodes=1, device=device, merge_train_id_eval=False)
    assert (run_dir / "evaluation_results.json").exists()
    assert (run_dir / "rliable_scores.json").exists()
    assert "default" in results["contexts"]
    assert results["context_meta"]["ball_mass_x2"]["split"] == "ood"
    assert results["context_meta"]["default"]["split"] == "id"
    scores = json.loads((run_dir / "rliable_scores.json").read_text())
    assert scores["eval_split"] == "id_ood_1d_slice"
    assert "context_meta" in scores
    probe = results["contexts"]["default"].get("physics_probe")
    assert probe is not None
    assert "ball_mass" in probe
    assert "gravity" in probe
    assert results["merge_train_id_eval"] is False
    # Training KPI merge is off, so default scores are the eval episodes only.
    assert len(results["contexts"]["default"]["episode_returns"]) == 1
    assert detect_family(vars(args)) == "pistonball"
    assert results["family"] == "pistonball"


def test_evaluate_run_hard_grid_adds_gravity(tmp_path, args):
    device = torch.device("cpu")
    nets = build_agents(args, device)
    run_dir = tmp_path / "run_hard"
    manifest = RunManifest(run_name="unit_eval_hard", run_dir=run_dir, args=vars(args))
    manifest.finished = True
    manifest.write()
    manager = CheckpointManager(run_dir, metric_key="return_mean")
    manager.save(build_checkpoint_payload(nets, args), global_step=50, eval_stats={"return_mean": 1.0})

    results = evaluate_run(
        run_dir,
        eval_episodes=1,
        device=device,
        eval_grid_preset="hard",
        eval_vary_params=("gravity",),
    )
    assert 4.0 in results["eval_factor_grid"]
    assert "gravity_x4" in results["contexts"]
    assert "gravity_x0.25" in results["contexts"]
    assert results["physics_probes"]["gravity_x4"]["gravity"] == pytest.approx(args.gravity * 4.0)
