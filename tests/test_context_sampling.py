"""Tests for training-time context sampling."""

from __future__ import annotations

import random

import pytest

from evaluation.context_sampling import (
    TrainingEnvironment,
    apply_context_to_env,
    get_raw_pistonball_env,
    resolve_train_context_factors,
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
        transfer_factors=(0.5, 2.0),
    )


def test_static_sampling_returns_default(args):
    static = Args(**{**vars(args), "training_regime": "static"})
    ctx = sample_training_context(static, random.Random(0))
    assert ctx == default_context(static)


def test_random_sampling_varies_physics(args):
    rng = random.Random(123)
    contexts = [sample_training_context(args, rng) for _ in range(20)]
    masses = {c["ball_mass"] for c in contexts}
    frictions = {c["ball_friction"] for c in contexts}
    assert len(masses) > 1
    assert len(frictions) > 1
    base = default_context(args)
    for ctx in contexts:
        assert 0.5 * base["ball_mass"] <= ctx["ball_mass"] <= 2.0 * base["ball_mass"]


def test_grid_sampling_uses_transfer_contexts(args):
    grid_args = Args(**{**vars(args), "train_context_sampling": "grid"})
    rng = random.Random(0)
    samples = {sample_training_context(grid_args, rng)["ball_mass"] for _ in range(50)}
    base = default_context(grid_args)
    expected = {base["ball_mass"], base["ball_mass"] * 0.5, base["ball_mass"] * 2.0}
    assert samples.issubset(expected)


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
    env = TrainingEnvironment(args, rng=random.Random(7))
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
    """Regression: previously recreated pymunk envs every episode and leaked RAM."""
    env = TrainingEnvironment(args, rng=random.Random(11))
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


def test_resolve_train_context_factors_defaults_to_transfer(args):
    args.train_context_factors = ()
    assert resolve_train_context_factors(args) == (0.5, 2.0)
