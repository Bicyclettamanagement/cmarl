"""Tests for varying-context replay storage in contextual MATD3."""

from __future__ import annotations

import numpy as np
import pytest

from matd3_pistonball import default_context
from matd3_pistonball_context import (
    Args,
    ContextualReplayBuffer,
    context_to_vector,
)


@pytest.fixture
def args() -> Args:
    return Args(
        n_pistons=3,
        max_cycles=8,
        frame_size=32,
        frame_stack=2,
        feature_dim=16,
        hidden_dim=32,
        batch_size=8,
        buffer_size=64,
        cuda=False,
        training_regime="varying",
    )


def test_replay_buffer_stores_distinct_context_vectors(args):
    import torch

    device = torch.device("cpu")
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    ref = default_context(args)
    rb = ContextualReplayBuffer(args.buffer_size, n, (c, h, w), 1, 4, device)

    ctx_a = context_to_vector(ref, ref)
    ood = dict(ref)
    ood["ball_mass"] = ref["ball_mass"] * 2.0
    ctx_b = context_to_vector(ood, ref)

    for step in range(5):
        obs = np.zeros((n, c, h, w), dtype=np.uint8)
        rb.add(obs, obs, np.zeros((n, 1), np.float32), 1.0, 0.0, context=ctx_a)
    for step in range(5):
        obs = np.ones((n, c, h, w), dtype=np.uint8)
        rb.add(obs, obs, np.zeros((n, 1), np.float32), 1.0, 0.0, context=ctx_b)

    _, _, _, _, _, contexts, _agent = rb.sample(args.batch_size)
    unique_rows = {tuple(row.tolist()) for row in contexts.cpu().numpy()}
    assert len(unique_rows) >= 2
