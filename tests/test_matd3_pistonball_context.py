"""Tests for context-aware MATD3 (concat context inclusion)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from matd3_pistonball import default_context
from matd3_pistonball_context import (
    Args,
    CONTEXT_KEYS,
    ContextualActor,
    ContextualCritic,
    ContextualReplayBuffer,
    build_agents,
    build_run_name,
    context_feature_dim,
    context_to_vector,
    evaluate,
    joint_policy_actions,
    train,
    transfer_sweep,
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
        batch_size=6,
        buffer_size=64,
        learning_starts=0,
        eval_episodes=2,
        share_actors=True,
        critic_shared_obs=True,
        context_mode="concat",
        context_to_actor=True,
        context_to_critic=True,
        method_tag="context_concat",
        experiment="unit_ctx",
        cuda=False,
        seed=0,
    )


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_context_to_vector_normalization(args):
    ref = default_context(args)
    train_vec = context_to_vector(ref, ref)
    assert train_vec.shape == (len(CONTEXT_KEYS),)
    for i, key in enumerate(CONTEXT_KEYS):
        expected = ref[key] / (abs(ref[key]) if abs(ref[key]) > 1e-8 else 1.0)
        assert train_vec[i] == pytest.approx(expected)

    ood = dict(ref)
    ood["ball_mass"] = ref["ball_mass"] * 2.0
    ood_vec = context_to_vector(ood, ref)
    mass_idx = CONTEXT_KEYS.index("ball_mass")
    assert ood_vec[mass_idx] == pytest.approx(2.0)


def test_context_feature_dim_modes(args):
    assert context_feature_dim(args) == len(CONTEXT_KEYS)
    none_args = Args(**{**vars(args), "context_mode": "none"})
    assert context_feature_dim(none_args) == 0


def test_build_run_name_includes_context_mode(args):
    name = build_run_name(args)
    assert "__concat__static__" in name
    assert name.startswith("unit_ctx__")
    varying = Args(**{**vars(args), "training_regime": "varying"})
    assert "__concat__varying__" in build_run_name(varying)


def test_contextual_buffer_roundtrip(args, device):
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    ctx_dim = context_feature_dim(args)
    rb = ContextualReplayBuffer(args.buffer_size, n, (c, h, w), 1, ctx_dim, device)
    ref = default_context(args)
    ctx = context_to_vector(ref, ref)
    for step in range(10):
        obs = np.full((n, c, h, w), step % 256, dtype=np.uint8)
        rb.add(obs, obs, np.zeros((n, 1), np.float32), reward=1.0, done=0.0, context=ctx)
    assert len(rb) == 10
    obs, next_obs, actions, rewards, dones, contexts = rb.sample(args.batch_size)
    assert contexts.shape == (args.batch_size, ctx_dim)
    assert torch.allclose(contexts, torch.as_tensor(ctx).expand_as(contexts))


def test_actor_critic_shapes_with_context(args, device):
    nets = build_agents(args, device)
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    b = 4
    obs = torch.rand(b, n, c, h, w)
    ctx = torch.rand(b, nets["context_dim"])
    actions = joint_policy_actions(nets["actors"], obs, context=ctx)
    assert actions.shape == (b, n, nets["act_dim"])
    assert float(actions.min()) >= nets["action_low"] - 1e-5
    assert float(actions.max()) <= nets["action_high"] + 1e-5

    q = nets["qf1"](obs, actions, context=ctx)
    assert q.shape == (b, 1)
    assert isinstance(nets["actors"][0], ContextualActor)
    assert isinstance(nets["qf1"], ContextualCritic)
    assert nets["actors"][0].context_to_actor is True
    assert nets["qf1"].context_to_critic is True


def test_context_changes_actor_output(args, device):
    """Different context vectors must change the shared actor's actions."""
    nets = build_agents(args, device)
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    obs = torch.rand(2, n, c, h, w)
    ctx_a = torch.zeros(2, nets["context_dim"])
    ctx_b = torch.ones(2, nets["context_dim"])
    a1 = joint_policy_actions(nets["actors"], obs, context=ctx_a)
    a2 = joint_policy_actions(nets["actors"], obs, context=ctx_b)
    assert not torch.allclose(a1, a2)


def test_none_mode_ignores_context_dim(args, device):
    none_args = Args(**{**vars(args), "context_mode": "none", "method_tag": "hidden_context"})
    nets = build_agents(none_args, device)
    assert nets["context_dim"] == 0
    n, c, h, w = none_args.n_pistons, none_args.frame_stack, none_args.frame_size, none_args.frame_size
    obs = torch.rand(2, n, c, h, w)
    actions = joint_policy_actions(nets["actors"], obs, context=None)
    assert actions.shape == (2, n, nets["act_dim"])
    q = nets["qf1"](obs, actions, context=None)
    assert q.shape == (2, 1)


def test_training_update_changes_parameters(args, device):
    torch.manual_seed(0)
    np.random.seed(0)
    nets = build_agents(args, device)
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    ctx_dim = nets["context_dim"]
    rb = ContextualReplayBuffer(args.buffer_size, n, (c, h, w), nets["act_dim"], ctx_dim, device)
    ref = default_context(args)
    ctx = context_to_vector(ref, ref)
    for _ in range(20):
        obs = np.random.randint(0, 256, (n, c, h, w), dtype=np.uint8)
        act = np.random.uniform(-1, 1, (n, nets["act_dim"])).astype(np.float32)
        rb.add(obs, obs, act, reward=float(np.random.randn()), done=0.0, context=ctx)

    qf1 = nets["qf1"]
    actors = nets["actors"]
    q_before = [p.clone() for p in qf1.parameters()]
    a_before = [p.clone() for p in actors.parameters()]

    b_obs, b_next_obs, b_actions, b_rewards, b_dones, b_ctx = rb.sample(args.batch_size)
    with torch.no_grad():
        next_actions = joint_policy_actions(nets["target_actors"], b_next_obs, context=b_ctx)
        noise = (
            (torch.randn_like(next_actions) * args.policy_noise)
            .clamp(-args.noise_clip, args.noise_clip)
            * actors[0].action_scale
        )
        next_actions = (next_actions + noise).clamp(nets["action_low"], nets["action_high"])
        min_q = torch.min(
            nets["qf1_target"](b_next_obs, next_actions, context=b_ctx),
            nets["qf2_target"](b_next_obs, next_actions, context=b_ctx),
        )
        target = b_rewards + (1 - b_dones) * args.gamma * min_q

    q_loss = torch.nn.functional.mse_loss(qf1(b_obs, b_actions, context=b_ctx), target)
    q_loss = q_loss + torch.nn.functional.mse_loss(
        nets["qf2"](b_obs, b_actions, context=b_ctx), target
    )
    nets["q_optimizer"].zero_grad()
    q_loss.backward()
    nets["q_optimizer"].step()

    actor_loss = -qf1(b_obs, joint_policy_actions(actors, b_obs, context=b_ctx), context=b_ctx).mean()
    nets["actor_optimizer"].zero_grad()
    actor_loss.backward()
    nets["actor_optimizer"].step()

    assert any(not torch.allclose(b, p) for b, p in zip(q_before, qf1.parameters()))
    assert any(not torch.allclose(b, p) for b, p in zip(a_before, actors.parameters()))
    assert torch.isfinite(q_loss) and torch.isfinite(actor_loss)


def test_evaluate_and_transfer_smoke(args, device):
    nets = build_agents(args, device)
    ref = default_context(args)
    stats = evaluate(
        nets["actors"], args, ref, ref, nets["agents"], device, seed=0, n_episodes=2
    )
    assert "return_mean" in stats
    assert "action_saturation_mean" in stats
    assert 0.0 <= stats["success_rate"] <= 1.0

    results = transfer_sweep(nets["actors"], args, nets["agents"], device, ref)
    assert "train" in results["contexts"]
    assert results["context_mode"] == "concat"
    assert "generalization_gap" in results


def test_train_emits_context_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = Args(
        experiment="unit_ctx_train",
        n_pistons=3,
        max_cycles=8,
        frame_size=32,
        frame_stack=2,
        feature_dim=16,
        hidden_dim=32,
        total_timesteps=220,
        learning_starts=80,
        buffer_size=200,
        batch_size=16,
        eval_frequency=100,
        eval_episodes=2,
        checkpoint_eval=True,
        transfer_eval=True,
        performance_threshold=-500.0,
        share_actors=True,
        context_mode="concat",
        training_regime="varying",
        train_context_sampling="grid",
        cuda=False,
        seed=7,
    )
    run_name = train(args)
    run_dir = tmp_path / "runs" / run_name
    assert "__concat__varying__" in run_name
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "transfer_results.json").exists()
    assert (run_dir / "rliable_scores.json").exists()
    assert (run_dir / "checkpoints" / "latest.pt").exists()

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["finished"] is True
    assert manifest["args"]["context_mode"] == "concat"
    assert manifest["args"]["training_regime"] == "varying"
    assert manifest["args"]["method_tag"] == "context_concat"

    rliable = json.loads((run_dir / "rliable_scores.json").read_text())
    assert rliable["method_tag"] == "context_concat"

    transfer = json.loads((run_dir / "transfer_results.json").read_text())
    assert transfer["context_mode"] == "concat"
    assert "train" in transfer["contexts"]
