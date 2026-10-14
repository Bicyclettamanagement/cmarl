"""Per-step Pistonball agent-context features for the kind ablation."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.agent_context import AGENT_FEATURE_DIM, pistonball_agent_features
from matd3_pistonball import Args, default_context, make_env


@pytest.fixture
def args() -> Args:
    return Args(
        n_pistons=3,
        max_cycles=8,
        frame_size=32,
        frame_stack=2,
        cuda=False,
        seed=0,
    )


def test_agent_features_shape_and_range(args):
    env = make_env(args, default_context(args))
    try:
        env.reset(seed=0)
        agents = list(env.possible_agents)
        feats = pistonball_agent_features(env, agents)
        assert feats.shape == (len(agents), AGENT_FEATURE_DIM)
        assert feats.dtype == np.float32
        assert np.all(np.isfinite(feats))
        assert np.all(np.abs(feats) < 5.0)
    finally:
        env.close()


def test_agent_features_change_after_steps(args):
    env = make_env(args, default_context(args))
    try:
        env.reset(seed=1)
        agents = list(env.possible_agents)
        before = pistonball_agent_features(env, agents).copy()
        for _ in range(4):
            if not env.agents:
                break
            action = {a: env.action_space(a).sample() for a in env.agents}
            env.step(action)
        after = pistonball_agent_features(env, agents)
        assert after.shape == before.shape
        assert not np.allclose(before, after)
    finally:
        env.close()
