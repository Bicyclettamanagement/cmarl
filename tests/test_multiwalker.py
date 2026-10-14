"""Multiwalker physics context, trainer plumbing, and checkpoint-eval dispatch."""

from __future__ import annotations

import json

import pytest
import torch

from evaluation.checkpoint_eval import detect_algorithm, detect_family, evaluate_run
from evaluation.rliable_utils import EXPERIMENT_PRESETS, KIND_PRESETS, MULTIWALKER_PRESETS


def test_rliable_presets_cover_next_experiments():
    for name in (*MULTIWALKER_PRESETS, *KIND_PRESETS):
        assert name in EXPERIMENT_PRESETS
    assert EXPERIMENT_PRESETS["mw3_hidden_varying"]["exp_name"] == "mw3_hidden_varying"
    assert EXPERIMENT_PRESETS["mw3_hidden_varying_fwd5"]["exp_name"] == "mw3_hidden_varying_fwd5"
    assert list(MULTIWALKER_PRESETS) == [
        "mw3_hidden_varying_fwd5",
        "mw3_concat_varying_fwd5",
    ]
    assert EXPERIMENT_PRESETS["context5_agent_varying"]["method_tag"] == "context_concat_agent"


def test_detect_family_multiwalker_vs_pistonball():
    assert detect_family({"env_name": "multiwalker", "exp_name": "matd3_multiwalker"}) == "multiwalker"
    assert detect_family({"experiment": "mw3_hidden_varying", "exp_name": "matd3_multiwalker"}) == "multiwalker"
    assert detect_family({"context_mode": "concat", "exp_name": "matd3_pistonball_context"}) == "pistonball_context"
    assert detect_family({"exp_name": "matd3_pistonball"}) == "pistonball"
    assert detect_algorithm({"env_name": "multiwalker", "context_mode": "concat"}) == "context"
    assert detect_algorithm({"env_name": "multiwalker", "context_mode": "hidden"}) == "hidden"


def _mw_args(**overrides):
    pytest.importorskip("Box2D")
    from matd3_multiwalker import Args

    kwargs = dict(
        n_walkers=3,
        max_cycles=40,
        terrain_length=20,
        total_timesteps=80,
        learning_starts=20,
        buffer_size=64,
        batch_size=8,
        eval_frequency=40,
        eval_episodes=1,
        checkpoint_eval=True,
        transfer_eval=False,
        cuda=False,
        seed=0,
        training_regime="varying",
        context_mode="hidden",
        experiment="unit_mw",
        eval_factor_grid=(0.5, 1.0, 2.0),
        performance_threshold=-500.0,
    )
    kwargs.update(overrides)
    return Args(**kwargs)


def test_eval_points_count_multiwalker():
    from evaluation.context_sampling import eval_points_1d
    from envs.multiwalker import MULTIWALKER_PHYSICS_KEYS

    args = _mw_args()
    points = eval_points_1d(args)
    assert len(points) == 1 + len(MULTIWALKER_PHYSICS_KEYS) * (len(args.eval_factor_grid) - 1)
    names = {p.name for p in points}
    assert "default" in names
    assert "gravity_x2" in names
    assert "package_mass_x0.5" in names
    assert "terrain_friction_x2" in names


def test_apply_context_after_reset_changes_live_physics():
    from envs.multiwalker import apply_context_after_reset, default_context, make_env

    args = _mw_args()
    env = make_env(args, default_context(args))
    try:
        env.reset(seed=0)
        ref = apply_context_after_reset(env, default_context(args))
        heavy = dict(default_context(args))
        heavy["gravity"] = args.gravity * 2.0
        heavy["package_mass"] = args.package_mass * 3.0
        heavy["terrain_friction"] = args.terrain_friction * 0.5
        env.reset(seed=0)
        probe = apply_context_after_reset(env, heavy)
        assert probe["gravity"] == pytest.approx(heavy["gravity"])
        assert probe["package_mass"] != pytest.approx(ref["package_mass"])
        assert probe["terrain_friction"] == pytest.approx(heavy["terrain_friction"])

        env.reset(seed=1)
        restored = apply_context_after_reset(env, default_context(args))
        assert restored["gravity"] == pytest.approx(args.gravity)
        assert restored["terrain_friction"] == pytest.approx(args.terrain_friction)
    finally:
        env.close()


def test_time_penalty_shifts_step_reward():
    from envs.multiwalker import TimePenaltyWrapper, default_context, make_env

    base = _mw_args(time_penalty=0.0)
    pen = _mw_args(time_penalty=-0.1)
    env0 = make_env(base, default_context(base))
    env1 = make_env(pen, default_context(pen))
    try:
        assert isinstance(env1, TimePenaltyWrapper)
        env0.reset(seed=0)
        env1.reset(seed=0)
        zeros = {a: env0.action_space(a).sample() * 0 for a in env0.agents}
        _, r0, *_ = env0.step(zeros)
        _, r1, *_ = env1.step(zeros)
        a = list(r0.keys())[0]
        assert float(r1[a]) - float(r0[a]) == pytest.approx(-0.1)
    finally:
        env0.close()
        env1.close()


def test_sequential_eval_points_not_stuck():
    from envs.multiwalker import (
        apply_context_after_reset,
        apply_context_to_env,
        default_context,
        make_env,
    )

    args = _mw_args()
    env = make_env(args, default_context(args))
    try:
        gravities = []
        masses = []
        for factor in (0.5, 2.0, 1.0):
            ctx = dict(default_context(args))
            ctx["gravity"] = args.gravity * factor
            ctx["package_mass"] = args.package_mass * factor
            apply_context_to_env(env, ctx)
            env.reset(seed=0)
            probe = apply_context_after_reset(env, ctx)
            gravities.append(probe["gravity"])
            masses.append(probe["package_mass"])
        assert gravities[0] != pytest.approx(gravities[1])
        assert masses[0] != pytest.approx(masses[1])
        assert gravities[2] == pytest.approx(args.gravity)
    finally:
        env.close()


def test_concat_context_dim():
    from envs.multiwalker import MULTIWALKER_PHYSICS_KEYS
    from matd3_multiwalker import build_agents, context_feature_dim

    hidden = _mw_args(context_mode="hidden")
    concat = _mw_args(context_mode="concat")
    assert context_feature_dim(hidden) == 0
    assert context_feature_dim(concat) == len(MULTIWALKER_PHYSICS_KEYS)
    nets = build_agents(concat, torch.device("cpu"))
    assert nets["context_dim"] == len(MULTIWALKER_PHYSICS_KEYS)
    assert nets["actors"][0].context_dim == len(MULTIWALKER_PHYSICS_KEYS)


def test_multiwalker_train_smoke(tmp_path, monkeypatch):
    from matd3_multiwalker import train

    monkeypatch.chdir(tmp_path)
    args = _mw_args()
    run_name = train(args)
    run_dir = tmp_path / "runs" / "multiwalker" / run_name
    assert "__hidden__varying__" in run_name
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["finished"] is True
    assert manifest["args"]["env_name"] == "multiwalker"
    assert manifest["args"]["performance_threshold"] == -500.0
    assert (run_dir / "checkpoints" / "latest.pt").exists()


def test_checkpoint_eval_dispatches_multiwalker(tmp_path):
    from matd3_multiwalker import build_agents
    from matd3_pistonball import CheckpointManager, RunManifest, build_checkpoint_payload

    args = _mw_args()
    device = torch.device("cpu")
    nets = build_agents(args, device)
    run_dir = tmp_path / "run_mw"
    manifest = RunManifest(run_name="unit_mw", run_dir=run_dir, args=vars(args))
    manifest.finished = True
    manifest.write()
    manager = CheckpointManager(run_dir, metric_key="return_mean")
    manager.save(build_checkpoint_payload(nets, args), global_step=50, eval_stats={"return_mean": 1.0})

    results = evaluate_run(run_dir, eval_episodes=1, device=device, eval_grid_preset="hard")
    assert results["family"] == "multiwalker"
    assert results["env_name"] == "multiwalker"
    assert "package_mass" in results["eval_vary_params"]
    assert "terrain_friction" in results["eval_vary_params"]
    assert 4.0 in results["eval_factor_grid"]
    assert "gravity_x4" in results["contexts"]
    probe = results["physics_probes"]["gravity_x4"]
    assert probe["gravity"] == pytest.approx(args.gravity * 4.0)


def test_clip_context_vector_uses_the_training_factor_window():
    from matd3_multiwalker import clip_context_vector, context_to_vector

    args = _mw_args()
    ref = {
        "gravity": args.gravity,
        "package_mass": args.package_mass,
        "terrain_friction": args.terrain_friction,
    }
    far = dict(ref)
    far["gravity"] = args.gravity * 4.0
    near = dict(ref)
    near["package_mass"] = args.package_mass * 0.25
    far_vec = context_to_vector(far, ref)
    near_vec = context_to_vector(near, ref)
    # Gravity is divided by abs(reference), so a 4x downward pull is -4.
    assert far_vec[0] == pytest.approx(-4.0)
    assert near_vec[1] == pytest.approx(0.25)
    clipped_far = clip_context_vector(far_vec, args.id_factor_low, args.id_factor_high)
    clipped_near = clip_context_vector(near_vec, args.id_factor_low, args.id_factor_high)
    assert clipped_far[0] == pytest.approx(-args.id_factor_high)
    assert clipped_near[1] == pytest.approx(args.id_factor_low)
    identity = clip_context_vector(context_to_vector(ref, ref), args.id_factor_low, args.id_factor_high)
    assert identity.tolist() == pytest.approx([-1.0, 1.0, 1.0])


def test_context_clip_eval_keeps_physics_and_does_not_overwrite_scores(tmp_path):
    from matd3_multiwalker import build_agents
    from matd3_pistonball import CheckpointManager, RunManifest, build_checkpoint_payload

    args = _mw_args(context_mode="concat", experiment="unit_mw_concat")
    device = torch.device("cpu")
    nets = build_agents(args, device)
    run_dir = tmp_path / "run_mw_clip"
    manifest = RunManifest(run_name="unit_mw_clip", run_dir=run_dir, args=vars(args))
    manifest.finished = True
    manifest.write()
    manager = CheckpointManager(run_dir, metric_key="return_mean")
    manager.save(build_checkpoint_payload(nets, args), global_step=50, eval_stats={"return_mean": 1.0})
    original_scores = run_dir / "rliable_scores.json"
    original_scores.write_text('{"untouched": true}')

    results = evaluate_run(
        run_dir,
        eval_episodes=1,
        device=device,
        eval_factor_grid=(4.0,),
        eval_vary_params=("gravity",),
        clip_context_to_id=True,
    )
    assert results["context_clip_to_id"] is True
    point = results["contexts"]["gravity_x4"]
    assert point["physics_probe"]["gravity"] == pytest.approx(args.gravity * 4.0)
    assert point["context_vector"][0] == pytest.approx(-4.0)
    assert point["context_vector_clipped"][0] == pytest.approx(-args.id_factor_high)
    assert max(abs(v) for v in point["context_vector_clipped"]) <= args.id_factor_high + 1e-6
    default = results["contexts"]["default"]
    assert default["context_vector_clipped"][0] == pytest.approx(default["context_vector"][0])
    assert json.loads(original_scores.read_text()) == {"untouched": True}
    assert (run_dir / "rliable_scores_context_clip.json").exists()
    assert (run_dir / "evaluation_results_context_clip.json").exists()
    assert not (run_dir / "evaluation_results.json").exists()
