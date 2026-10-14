"""RGB rollout helpers used by the agent-visualization notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.rollout_viz import (
    clips_to_html,
    crop_empty_top,
    encode_gif_bytes,
    list_available_runs,
    list_context_names,
    load_policies,
    load_policy,
    prepare_display_frames,
    resolve_context,
    resolve_run_dir,
    resolve_run_dirs,
    rollout_episode,
    short_context_name,
    suggested_contexts,
    watch_many_policies_one_env,
    watch_one_policy_many_envs,
    watch_specs,
)
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
        experiment="unit_viz",
        exp_name="baseline5_varying_randdrop",
        method_tag="hidden_context",
    )


def _write_run(tmp_path, args, seed: int = 0) -> Path:
    args = Args(**{**vars(args), "seed": seed})
    device = torch.device("cpu")
    nets = build_agents(args, device)
    run_dir = tmp_path / f"baseline5_varying_randdrop__shared__randdrop__n3__seed{seed}__1"
    manifest = RunManifest(run_name=run_dir.name, run_dir=run_dir, args=vars(args))
    manifest.finished = True
    manifest.best_eval = {"step": 50, "metric": "return_mean", "value": 1.0, "stats": {"return_mean": 12.0}}
    manifest.write()
    manager = CheckpointManager(run_dir, metric_key="return_mean")
    manager.save(build_checkpoint_payload(nets, args), global_step=50, eval_stats={"return_mean": 1.0})
    return run_dir


def test_list_and_resolve_runs(tmp_path, args):
    run_dir = _write_run(tmp_path, args)
    rows = list_available_runs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["seed"] == 0
    assert rows[0]["checkpoint"] == "best"
    assert rows[0]["preset"] == "baseline5_varying"
    resolved = resolve_run_dir(tmp_path, "baseline5_varying", seed=0)
    assert resolved == run_dir
    _write_run(tmp_path, args, seed=1)
    all_dirs = resolve_run_dirs(tmp_path, "baseline5_varying")
    assert [d.name for d in all_dirs] == [
        "baseline5_varying_randdrop__shared__randdrop__n3__seed0__1",
        "baseline5_varying_randdrop__shared__randdrop__n3__seed1__1",
    ]
    loaded = load_policies(tmp_path, ["baseline5_varying"], device=torch.device("cpu"))
    assert [p.seed for p in loaded] == [0, 1]
    assert loaded[0].short_label == "baseline5_varying"
    assert loaded[0].train_return == pytest.approx(12.0)


def test_resolve_context_names(tmp_path, args):
    policy = load_policy(_write_run(tmp_path, args), device=torch.device("cpu"))
    names = list_context_names(policy)
    assert "default" in names
    assert "ball_mass_x2" in names
    default = resolve_context(policy, "default")
    assert default.split == "id"
    ood = resolve_context(policy, "ball_mass_x2")
    assert ood.split == "ood"
    assert ood.context["ball_mass"] == pytest.approx(args.ball_mass * 2.0)
    custom = resolve_context(policy, {"ball_mass": args.ball_mass * 3.0})
    assert custom.split == "custom"
    assert "ball_mass" in custom.name


def test_encode_gif_and_html():
    pytest.importorskip("PIL")
    frames = np.zeros((4, 16, 24, 3), dtype=np.uint8)
    frames[1:, :, :, 0] = 200
    gif = encode_gif_bytes(frames, fps=8, max_width=20)
    assert gif[:6] in (b"GIF87a", b"GIF89a")

    clip = type("Clip", (), {})()
    clip.frames = frames
    clip.title = "Hidden · default"
    clip.caption = "return 1.0  ·  4 steps"
    html = clips_to_html([clip], fps=8, max_width=20, overlay=False, crop=False)
    assert "data:image/gif;base64," in html
    assert "Hidden" in html
    assert "return 1.0" in html


def test_crop_empty_top_keeps_lower_half():
    frame = np.full((20, 12, 3), 255, dtype=np.uint8)
    frame[14:, :, :] = 10
    cropped = crop_empty_top(frame, keep_min=0.5)
    assert cropped.shape[0] < frame.shape[0]
    assert cropped[-1, 0, 0] == 10
    assert cropped.shape[0] >= 10


def test_overlay_and_short_context():
    pytest.importorskip("PIL")
    from evaluation.rollout_viz import RolloutClip

    assert short_context_name("ball_mass_x2") == "mass×2"
    assert short_context_name("default") == "default"
    frames = np.full((3, 40, 80, 3), 180, dtype=np.uint8)
    clip = RolloutClip(
        frames=frames,
        episode_return=12.0,
        length=3,
        success=0.0,
        context_name="ball_mass_x2",
        context={},
        policy_label="Hidden context (varying) (seed 1)",
        short_label="baseline5_varying",
        seed=1,
        env_kind="pistonball",
        split="ood",
        eval_iqm=94.0,
        returns_at_frame=np.array([0.0, 4.0, 12.0], dtype=np.float32),
    )
    prepared = prepare_display_frames(clip, max_width=80, crop=False, overlay=True)
    assert prepared.shape[0] == 3
    assert prepared.shape[-1] == 3
    assert prepared.shape[2] <= 80
    assert int(prepared[0, 0, 0, 0]) < 40


def test_watch_specs_all_seeds(tmp_path, args):
    _write_run(tmp_path, args, seed=0)
    _write_run(tmp_path, args, seed=1)
    clips = watch_specs(
        tmp_path,
        ["baseline5_varying"],
        ["default"],
        device=torch.device("cpu"),
        episode_seed=0,
    )
    assert len(clips) == 2
    assert {c.seed for c in clips} == {0, 1}
    assert all(c.short_label == "baseline5_varying" for c in clips)
    html = clips_to_html(clips, fps=8, max_width=48, overlay=True, crop=True)
    assert "repeat(2,max-content)" in html


def test_rollout_and_side_by_side(tmp_path, args):
    run_dir = _write_run(tmp_path, args)
    policy = load_policy(run_dir, device=torch.device("cpu"))
    clip = rollout_episode(policy, "default", seed=0)
    assert clip.frames.ndim == 4
    assert clip.frames.shape[-1] == 3
    assert clip.length >= 1
    assert clip.frames.shape[0] >= 1
    assert np.isfinite(clip.episode_return)

    many_envs = watch_one_policy_many_envs(policy, ["default", "ball_mass_x2"], seed=0)
    assert len(many_envs) == 2
    assert many_envs[0].context_name == "default"
    assert many_envs[1].context_name == "ball_mass_x2"
    assert many_envs[1].split == "ood"

    many_policies = watch_many_policies_one_env([policy, policy], "default", seed=0)
    assert len(many_policies) == 2
    assert {c.context_name for c in many_policies} == {"default"}


def test_suggested_contexts():
    assert "ball_mass_x2" in suggested_contexts("pistonball")
    assert "gravity_x2" in suggested_contexts("multiwalker")


def test_mix_env_families_rejected(tmp_path, args):
    policy = load_policy(_write_run(tmp_path, args), device=torch.device("cpu"))
    other = load_policy(_write_run(tmp_path / "other", args), device=torch.device("cpu"))
    other.family = "multiwalker"
    with pytest.raises(ValueError, match="cannot mix"):
        watch_many_policies_one_env([policy, other], "default")
