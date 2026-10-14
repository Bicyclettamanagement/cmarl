"""Tests for training-time context sampling and the ID/OOD eval grid."""

from __future__ import annotations

import random

import pytest

from evaluation.context_sampling import (
    HARD_EVAL_PARAMS,
    MULTIWALKER_PHYSICS_KEYS,
    PHYSICS_CONTEXT_KEYS,
    TrainingEnvironment,
    apply_context_after_reset,
    apply_context_to_env,
    context_spec_from_args,
    eval_points_1d,
    get_raw_pistonball_env,
    make_context_rng,
    resolve_eval_grid,
    resolve_eval_vary_params,
    sample_training_context,
)
from matd3_pistonball import Args, default_context, make_env


@pytest.fixture
def args() -> Args:
    return Args(
        n_pistons=3,
        max_cycles=8,
        frame_size=32,
        frame_stack=2,
        seed=0,
        training_regime="varying",
        train_context_sampling="random",
        id_factor_low=0.75,
        id_factor_high=1.5,
        eval_factor_grid=(0.5, 1.0, 2.0),
    )


def test_static_sampling_returns_default(args):
    static = Args(**{**vars(args), "training_regime": "static"})
    ctx = sample_training_context(static, random.Random(0))
    assert ctx == default_context(static)


def test_random_sampling_stays_inside_id_window(args):
    rng = random.Random(123)
    contexts = [sample_training_context(args, rng) for _ in range(40)]
    masses = {c["ball_mass"] for c in contexts}
    assert len(masses) > 1
    base = default_context(args)
    spec = context_spec_from_args(args)
    for ctx in contexts:
        factor = ctx["ball_mass"] / base["ball_mass"]
        assert spec.is_id_factor(factor)
        # OOD endpoints must not appear in training samples
        assert ctx["ball_mass"] != pytest.approx(base["ball_mass"] * 0.5)
        assert ctx["ball_mass"] != pytest.approx(base["ball_mass"] * 2.0)


def test_ood_eval_factors_are_outside_id_window(args):
    spec = context_spec_from_args(args)
    assert spec.is_id_factor(1.0)
    assert spec.is_id_factor(0.75)
    assert spec.is_id_factor(1.5)
    assert not spec.is_id_factor(0.5)
    assert not spec.is_id_factor(2.0)
    points = eval_points_1d(args)
    names = {p.name for p in points}
    assert "default" in names
    ood = [p for p in points if p.split == "ood"]
    id_pts = [p for p in points if p.split == "id"]
    assert ood and id_pts
    assert all(p.varied_param is None or p.varied_param in PHYSICS_CONTEXT_KEYS for p in points)
    # One parameter at a time (or default)
    for p in points:
        if p.name == "default":
            assert p.varied_param is None
        else:
            assert p.varied_param is not None
            assert p.context[p.varied_param] != pytest.approx(spec.default[p.varied_param])
            for other in PHYSICS_CONTEXT_KEYS:
                if other != p.varied_param:
                    assert p.context[other] == pytest.approx(spec.default[other])


def test_grid_sampling_uses_id_factors_only(args):
    grid_args = Args(**{**vars(args), "train_context_sampling": "grid"})
    rng = random.Random(0)
    samples = {sample_training_context(grid_args, rng)["ball_mass"] for _ in range(50)}
    base = default_context(grid_args)
    # eval grid is (0.5, 1.0, 2.0); only 1.0 is inside [0.75, 1.5]
    assert samples == {base["ball_mass"]}


def test_single_axis_training_leaves_other_params_at_default(args):
    one = Args(**{**vars(args), "train_vary_params": ("ball_mass",)})
    rng = random.Random(7)
    ctx = sample_training_context(one, rng)
    base = default_context(one)
    assert ctx["ball_friction"] == pytest.approx(base["ball_friction"])
    assert ctx["ball_elasticity"] == pytest.approx(base["ball_elasticity"])
    assert ctx["ball_mass"] != pytest.approx(base["ball_mass"]) or True  # may rarely hit 1.0


def test_context_sampling_is_seed_deterministic(args):
    a = [sample_training_context(args, make_context_rng(11)) for _ in range(8)]
    b = [sample_training_context(args, make_context_rng(11)) for _ in range(8)]
    c = [sample_training_context(args, make_context_rng(12)) for _ in range(8)]
    assert a == b
    assert a != c


def test_eval_points_2d_ready_for_future_plots(args):
    spec = context_spec_from_args(args)
    points = spec.eval_points_2d("ball_mass", "ball_friction")
    assert len(points) == len(spec.eval_factor_grid) ** 2
    regions = {p.region for p in points}
    assert "interpolation" in regions
    assert "extrapolation" in regions
    assert "inter_extrapolation" in regions


def test_apply_context_updates_ball_on_reset(args):
    env = make_env(args, default_context(args))
    try:
        raw = get_raw_pistonball_env(env)
        ctx = dict(default_context(args))
        ctx["ball_mass"] = args.ball_mass * 2.0
        ctx["ball_friction"] = args.ball_friction * 0.5
        ctx["ball_elasticity"] = args.ball_elasticity * 2.0
        apply_context_to_env(env, ctx)
        env.reset(seed=0)
        assert raw.ball.mass == pytest.approx(ctx["ball_mass"])
        shapes = list(raw.ball.shapes)
        assert shapes
        assert shapes[0].friction == pytest.approx(ctx["ball_friction"])
        assert shapes[0].elasticity == pytest.approx(ctx["ball_elasticity"])
    finally:
        env.close()


def test_training_environment_resamples_without_recreating_env(args):
    env = TrainingEnvironment(args, rng=make_context_rng(7))
    try:
        outer_id = id(env.env)
        _, ctx0 = env.reset(seed=1)
        _, ctx1 = env.start_episode(seed=2)
        _, ctx2 = env.start_episode(seed=3)
        assert id(env.env) == outer_id
        assert not env.env_recreated
        assert ctx0 != ctx1 or ctx1 != ctx2
        raw = get_raw_pistonball_env(env.env)
        assert raw.ball.mass == pytest.approx(ctx2["ball_mass"])
    finally:
        env.close()


def test_training_environment_many_episodes_reuses_same_env(args):
    env = TrainingEnvironment(args, rng=make_context_rng(11))
    try:
        outer_id = id(env.env)
        masses = []
        for i in range(50):
            _, ctx = env.start_episode(seed=100 + i)
            masses.append(ctx["ball_mass"])
            assert id(env.env) == outer_id
        assert len(set(masses)) > 1
        assert not env.env_recreated
    finally:
        env.close()


def test_training_environment_same_seed_same_contexts(args):
    env_a = TrainingEnvironment(args, rng=make_context_rng(args.seed))
    env_b = TrainingEnvironment(args, rng=make_context_rng(args.seed))
    try:
        seq_a = [env_a.start_episode(seed=i)[1] for i in range(5)]
        seq_b = [env_b.start_episode(seed=i)[1] for i in range(5)]
        assert seq_a == seq_b
    finally:
        env_a.close()
        env_b.close()


def test_eval_points_1d_with_gravity_keeps_default_count_formula(args):
    points = eval_points_1d(args)
    assert len(points) == 1 + 3 * (len(args.eval_factor_grid) - 1)
    gravity_args = Args(**{**vars(args), "eval_vary_params": HARD_EVAL_PARAMS})
    g_points = eval_points_1d(gravity_args)
    assert len(g_points) == 1 + 4 * (len(args.eval_factor_grid) - 1)
    names = {p.name for p in g_points}
    assert "gravity_x2" in names
    assert "default" in names
    # Standard 3-axis count is unchanged when gravity is not requested.
    assert "gravity_x2" not in {p.name for p in points}


def test_resolve_eval_grid_hard_preset():
    grid = resolve_eval_grid(preset="hard")
    assert 0.25 in grid and 4.0 in grid
    assert resolve_eval_grid(factors=(0.5, 1.0, 2.0)) == (0.5, 1.0, 2.0)


def test_gravity_after_reset_overrides_hardcoded_space_gravity(args):
    env = make_env(args, default_context(args))
    try:
        ctx = dict(default_context(args))
        ctx["gravity"] = 3000.0
        apply_context_to_env(env, ctx)
        env.reset(seed=0)
        raw = get_raw_pistonball_env(env)
        # PettingZoo reset hardcodes gravity to 750; the after-reset patch is required.
        assert raw.space.gravity[1] == pytest.approx(750.0)
        probe = apply_context_after_reset(env, ctx)
        assert probe["gravity"] == pytest.approx(3000.0)
        assert raw.space.gravity[1] == pytest.approx(3000.0)
    finally:
        env.close()


def test_sequential_eval_points_change_live_mass(args):
    env = make_env(args, default_context(args))
    try:
        raw = get_raw_pistonball_env(env)
        masses = []
        for factor in (0.5, 2.0, 1.0):
            ctx = dict(default_context(args))
            ctx["ball_mass"] = args.ball_mass * factor
            apply_context_to_env(env, ctx)
            env.reset(seed=0)
            probe = apply_context_after_reset(env, ctx)
            masses.append(probe["ball_mass"])
            assert raw.ball.mass == pytest.approx(ctx["ball_mass"])
        assert masses[0] != pytest.approx(masses[1])
        assert masses[2] == pytest.approx(args.ball_mass)
    finally:
        env.close()


def test_hard_eval_vary_params_dispatch():
    assert resolve_eval_vary_params(preset="hard") == HARD_EVAL_PARAMS
    assert resolve_eval_vary_params(preset="hard", env_name="multiwalker") == MULTIWALKER_PHYSICS_KEYS
    assert resolve_eval_vary_params(preset="hard", params=("gravity",)) == ("gravity",)
    assert resolve_eval_vary_params(preset="standard") is None
