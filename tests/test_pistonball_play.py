"""Headless checks for the mixed human/policy Pistonball player."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.pistonball_play import (
    apply_drop_flags,
    apply_overrides,
    near_obstacle_policy_mean,
    parse_piston_spec,
    prepare_pair,
    run_scripted_episode,
    scripted_scalar,
)
from evaluation.rollout_viz import load_policy
from matd3_pistonball import Args, CheckpointManager, RunManifest, build_agents, build_checkpoint_payload


def test_parse_piston_spec():
    assert parse_piston_spec("center", 5) == [2]
    assert parse_piston_spec("left", 5) == [0]
    assert parse_piston_spec("right", 5) == [4]
    assert parse_piston_spec("1, piston_3", 5) == [1, 3]
    assert parse_piston_spec("none", 5) == []
    with pytest.raises(ValueError):
        parse_piston_spec("9", 5)


def test_apply_overrides_replaces_only_claimed_pistons():
    policy = np.zeros((5, 1), dtype=np.float32)
    mixed = apply_overrides(policy, {2: 1.0, 4: -1.0})
    assert mixed.shape == (5, 1)
    assert mixed[2, 0] == pytest.approx(1.0)
    assert mixed[4, 0] == pytest.approx(-1.0)
    assert np.all(mixed[[0, 1, 3]] == 0.0)
    assert np.all(policy == 0.0)


def test_near_obstacle_mean_uses_neighbors_only():
    policy = np.array(
        [
            [0.0, 0.0, 9.0, 1.0, 1.0],
            [0.0, 0.0, 9.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    # Ball sits on manual piston 2 for both steps. Neighbors are 0,1,3,4 but
    # the mask is "non-manual", so the 9s on the manual piston are excluded.
    mean = near_obstacle_policy_mean(policy, np.array([2, 2]), {2})
    assert mean == pytest.approx(0.0)
    assert near_obstacle_policy_mean(policy, np.array([2, 2]), set()) is None


def test_assist_raises_when_ball_is_to_the_right():
    class _Body:
        def __init__(self, x: float):
            self.position = (x, 0.0)

    class _Raw:
        piston_width = 40.0
        ball = _Body(100.0)
        pistonList = [_Body(40.0), _Body(200.0)]

    raw = _Raw()
    assert scripted_scalar("assist", raw, 0) == 1.0
    assert scripted_scalar("assist", raw, 1) == -1.0
    assert scripted_scalar("block", raw, 0) == 1.0
    assert scripted_scalar("pit", raw, 0) == -1.0
    assert scripted_scalar("freeze", raw, 0) == 0.0


def _write_run(tmp_path: Path) -> Path:
    args = Args(
        n_pistons=5,
        max_cycles=6,
        frame_size=32,
        frame_stack=2,
        feature_dim=16,
        hidden_dim=32,
        share_actors=True,
        cuda=False,
        seed=0,
        experiment="unit_play",
        exp_name="baseline5_varying_randdrop",
        method_tag="hidden_context",
        ball_mass=0.75,
    )
    device = torch.device("cpu")
    nets = build_agents(args, device)
    run_dir = tmp_path / "unit_play__seed0__1"
    manifest = RunManifest(run_name=run_dir.name, run_dir=run_dir, args=vars(args))
    manifest.finished = True
    manifest.write()
    manager = CheckpointManager(run_dir, metric_key="return_mean")
    manager.save(build_checkpoint_payload(nets, args), global_step=1, eval_stats={"return_mean": 0.0})
    return run_dir


def test_scripted_block_overrides_policy_and_fixed_drop(tmp_path, monkeypatch):
    run_dir = _write_run(tmp_path)
    policy = load_policy(run_dir, device=torch.device("cpu"))
    zeros = np.zeros((5, 1), dtype=np.float32)

    def _zeros(policy_obj, obs, env, context):
        assert context["ball_mass"] == pytest.approx(policy_obj.args.ball_mass)
        return zeros.copy()

    monkeypatch.setattr("evaluation.pistonball_play._act", _zeros)
    env_ctx, fed_ctx, env_name, fed_name = prepare_pair(
        policy, "ball_mass_x4", fixed_drop=True, lie_context="default"
    )
    assert fed_ctx["ball_mass"] == pytest.approx(policy.args.ball_mass)
    assert env_ctx["ball_mass"] == pytest.approx(policy.args.ball_mass * 4)
    assert env_ctx["random_drop"] is False

    from evaluation.pistonball_play import make_play_env

    env = make_play_env(policy, env_ctx)
    try:
        result, frames = run_scripted_episode(
            policy,
            env,
            env_ctx,
            fed_ctx,
            context_name=env_name,
            fed_context_name=fed_name,
            seed=0,
            manual={2},
            script="block",
            max_steps=3,
            collect_frames=True,
        )
    finally:
        env.close()

    assert result.length == 3
    assert result.random_drop is False
    assert result.physics["ball_mass"] == pytest.approx(policy.args.ball_mass * 4, rel=1e-3)
    assert result.fed_context_name == "default"
    assert np.allclose(result.applied_actions[:, 2], 1.0)
    assert np.allclose(result.applied_actions[:, [0, 1, 3, 4]], 0.0)
    assert np.allclose(result.policy_actions, 0.0)
    assert frames and frames[0].ndim == 3
    text = result.format_line()
    assert "near-obstacle policy mean" in text
    assert "mass=" in text


def test_policy_script_ignores_manual_pistons(tmp_path, monkeypatch):
    run_dir = _write_run(tmp_path)
    policy = load_policy(run_dir, device=torch.device("cpu"))
    ones = np.ones((5, 1), dtype=np.float32)
    monkeypatch.setattr("evaluation.pistonball_play._act", lambda *a, **k: ones.copy())
    env_ctx, fed_ctx, env_name, fed_name = prepare_pair(
        policy, "default", fixed_drop=True, lie_context=None
    )
    from evaluation.pistonball_play import make_play_env

    env = make_play_env(policy, env_ctx)
    try:
        result, _ = run_scripted_episode(
            policy,
            env,
            env_ctx,
            fed_ctx,
            context_name=env_name,
            fed_context_name=fed_name,
            seed=0,
            manual={2},
            script="policy",
            max_steps=2,
        )
    finally:
        env.close()
    assert result.manual == []
    assert np.allclose(result.applied_actions, 1.0)


def test_apply_drop_flags_leaves_random_when_disabled():
    ctx = apply_drop_flags({"ball_mass": 1.0, "random_drop": True}, fixed_drop=False)
    assert ctx["random_drop"] is True
    pinned = apply_drop_flags({"ball_mass": 1.0, "random_drop": True}, fixed_drop=True)
    assert pinned["random_drop"] is False
    assert pinned["random_rotate"] is False
