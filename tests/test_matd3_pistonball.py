"""Correctness tests for the MATD3 Pistonball baseline.

These run on CPU with a tiny environment (few pistons, small frames, short episodes)
so the whole suite finishes in seconds. They check the pieces that are easy to get
wrong when going from single-agent TD3 to multi-agent CTDE: tensor shapes, the
uint8 replay buffer, action bounds, centralized-critic joint encoding, target-network
initialization, the physics-context plumbing, and that a full update step actually
moves the parameters.
"""

import json

import numpy as np
import pytest
import torch

from matd3_pistonball import (
    Args,
    CentralizedCritic,
    CheckpointManager,
    MultiAgentReplayBuffer,
    RunManifest,
    action_saturation_fraction,
    append_eval_record,
    bootstrap_done,
    build_agents,
    build_checkpoint_payload,
    build_run_name,
    build_transfer_contexts,
    clip_gradients,
    current_exploration_noise,
    default_context,
    estimate_buffer_gb,
    evaluate,
    gradient_norm,
    joint_policy_actions,
    make_env,
    polyak_update,
    sanitize_experiment_tag,
    stack_obs,
    transfer_sweep,
)


@pytest.fixture
def args() -> Args:
    """Tiny, fast configuration used across the tests (shared actors by default)."""
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
        cuda=False,
        seed=0,
        eval_factor_grid=(0.5, 1.0, 2.0),
        id_factor_low=0.75,
        id_factor_high=1.5,
    )


@pytest.fixture
def indep_args(args) -> Args:
    return Args(**{**vars(args), "share_actors": False, "critic_shared_obs": False})


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_estimate_buffer_gb(args):
    gb = estimate_buffer_gb(args.n_pistons, (args.frame_stack, args.frame_size, args.frame_size), 1000)
    assert 0 < gb < 1.0


def test_env_shapes(args):
    env = make_env(args, default_context(args))
    obs_dict, _ = env.reset(seed=0)
    agents = list(env.possible_agents)
    assert len(agents) == args.n_pistons
    obs = stack_obs(obs_dict, agents)
    assert obs.shape == (args.n_pistons, args.frame_stack, args.frame_size, args.frame_size)
    assert obs.dtype == np.uint8
    act_space = env.action_space(agents[0])
    assert act_space.shape == (1,)
    assert np.allclose([act_space.low[0], act_space.high[0]], [-1.0, 1.0])
    env.close()


def test_context_changes_environment(args):
    """The physics context must actually reach the underlying environment."""
    ctx = dict(default_context(args))
    ctx["ball_mass"] = 3.14
    env = make_env(args, ctx)
    env.reset(seed=0)
    # Reach through the SuperSuit wrappers to the raw pistonball environment.
    raw = env
    while hasattr(raw, "unwrapped") and raw.unwrapped is not raw:
        raw = raw.unwrapped
    assert raw.ball_mass == pytest.approx(3.14)
    env.close()


def test_transfer_contexts_structure(args):
    contexts = build_transfer_contexts(args)
    assert "default" in contexts
    # default + one point per (param, non-1.0 factor)
    n_expected = 1 + 3 * (len(args.eval_factor_grid) - 1)
    assert len(contexts) == n_expected
    assert contexts["ball_mass_x2"]["ball_mass"] == pytest.approx(args.ball_mass * 2.0)
    base = contexts["default"]
    assert contexts["ball_mass_x2"]["ball_friction"] == base["ball_friction"]
    assert contexts["ball_mass_x0.5"]["ball_mass"] == pytest.approx(args.ball_mass * 0.5)


def test_replay_buffer_roundtrip(args, device):
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    rb = MultiAgentReplayBuffer(args.buffer_size, n, (c, h, w), 1, device)
    for step in range(10):
        obs = np.full((n, c, h, w), step % 256, dtype=np.uint8)
        rb.add(obs, obs, np.zeros((n, 1), np.float32), reward=1.0, done=0.0)
    assert len(rb) == 10
    obs, next_obs, actions, rewards, dones = rb.sample(args.batch_size)
    assert obs.shape == (args.batch_size, n, c, h, w)
    assert actions.shape == (args.batch_size, n, 1)
    assert rewards.shape == (args.batch_size, 1)
    assert dones.shape == (args.batch_size, 1)
    # Observations are normalized to [0, 1] float on sampling.
    assert obs.dtype == torch.float32
    assert 0.0 <= float(obs.min()) and float(obs.max()) <= 1.0


def test_replay_buffer_wraps_around(args, device):
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    rb = MultiAgentReplayBuffer(4, n, (c, h, w), 1, device)
    for _ in range(6):
        obs = np.zeros((n, c, h, w), dtype=np.uint8)
        rb.add(obs, obs, np.zeros((n, 1), np.float32), 0.0, 0.0)
    assert len(rb) == 4  # capped at capacity
    assert rb.full


def test_actor_output_shape_and_bounds(args, device):
    nets = build_agents(args, device)
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    obs = torch.rand(5, n, c, h, w)
    actions = joint_policy_actions(nets["actors"], obs)
    assert actions.shape == (5, n, nets["act_dim"])
    assert float(actions.min()) >= nets["action_low"] - 1e-5
    assert float(actions.max()) <= nets["action_high"] + 1e-5


def test_actors_are_independent(indep_args, device):
    """Independent mode: multiple actors must have distinct parameters."""
    nets = build_agents(indep_args, device)
    actors = nets["actors"]
    assert len(actors) == indep_args.n_pistons
    assert not actors[0].use_agent_id
    p0 = dict(actors[0].named_parameters())
    p1 = dict(actors[1].named_parameters())
    assert p0["fc_mu.weight"].data_ptr() != p1["fc_mu.weight"].data_ptr()
    assert not torch.allclose(p0["fc_mu.weight"], p1["fc_mu.weight"])


def test_shared_actor_uses_agent_id(args, device):
    nets = build_agents(args, device)
    actors = nets["actors"]
    assert len(actors) == 1
    assert actors[0].use_agent_id
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    obs = torch.rand(4, n, c, h, w)
    actions = joint_policy_actions(actors, obs)
    assert actions.shape == (4, n, nets["act_dim"])
    # Same image, different agent IDs -> generally different actions after random init.
    assert not torch.allclose(actions[:, 0], actions[:, 1])


def test_centralized_critic_shape(args, device):
    nets = build_agents(args, device)
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    obs = torch.rand(5, n, c, h, w)
    actions = torch.rand(5, n, nets["act_dim"]) * 2 - 1
    q = nets["qf1"](obs, actions)
    assert q.shape == (5, 1)
    assert nets["qf1"].shared_obs is True


def test_single_centralized_critic_instances(args, device):
    """Exactly one centralized critic (as twin-Q); it consumes *all* agents."""
    nets = build_agents(args, device)
    assert isinstance(nets["qf1"], CentralizedCritic)
    assert isinstance(nets["qf2"], CentralizedCritic)
    assert nets["qf1"].n_agents == args.n_pistons


def test_target_networks_initialized_equal(args, device):
    nets = build_agents(args, device)
    for p, tp in zip(nets["actors"].parameters(), nets["target_actors"].parameters()):
        assert torch.allclose(p, tp)
    for p, tp in zip(nets["qf1"].parameters(), nets["qf1_target"].parameters()):
        assert torch.allclose(p, tp)


def test_polyak_update_moves_target(args, device):
    nets = build_agents(args, device)
    src, tgt = nets["qf1"], nets["qf1_target"]
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)
    before = [p.clone() for p in tgt.parameters()]
    polyak_update(src, tgt, tau=0.5)
    for b, a in zip(before, tgt.parameters()):
        assert not torch.allclose(b, a)


def test_evaluate_returns_metrics(args, device):
    nets = build_agents(args, device)
    stats = evaluate(
        nets["actors"], args, default_context(args), nets["agents"], device,
        seed=0, n_episodes=2,
    )
    assert set(stats) == {
        "return_mean",
        "return_std",
        "return_median",
        "return_iqm",
        "optimality_gap",
        "length_mean",
        "success_rate",
        "action_std_mean",
        "action_saturation_mean",
        "per_agent_return_variance_mean",
        "physics_probe",
        "episode_returns",
        "episode_lengths",
        "episode_successes",
        "n_episodes",
    }
    assert 0.0 <= stats["success_rate"] <= 1.0
    assert 0.0 <= stats["action_saturation_mean"] <= 1.0
    assert stats["length_mean"] > 0


def test_transfer_sweep_reports_gap(args, device):
    nets = build_agents(args, device)
    results = transfer_sweep(nets["actors"], args, nets["agents"], device)
    assert "default" in results["contexts"]
    assert "generalization_gap" in results
    assert "generalization_gap_iqm" in results
    assert results["context_meta"]["ball_mass_x2"]["split"] == "ood"
    assert results["context_meta"]["default"]["split"] == "id"
    expected = results["id_return_mean"] - results["ood_return_mean"]
    assert results["generalization_gap"] == pytest.approx(expected)


def test_training_update_changes_parameters(args, device):
    """A critic + actor update must run end-to-end and change parameters."""
    torch.manual_seed(0)
    np.random.seed(0)
    nets = build_agents(args, device)
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    rb = MultiAgentReplayBuffer(args.buffer_size, n, (c, h, w), nets["act_dim"], device)
    for _ in range(20):
        obs = np.random.randint(0, 256, (n, c, h, w), dtype=np.uint8)
        act = np.random.uniform(-1, 1, (n, nets["act_dim"])).astype(np.float32)
        rb.add(obs, obs, act, reward=np.random.randn(), done=0.0)

    qf1, qf2 = nets["qf1"], nets["qf2"]
    actors = nets["actors"]
    action_scale = actors[0].action_scale
    q_params_before = [p.clone() for p in qf1.parameters()]
    actor_params_before = [p.clone() for p in actors.parameters()]

    b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(args.batch_size)
    with torch.no_grad():
        next_actions = joint_policy_actions(nets["target_actors"], b_next_obs)
        noise = (
            (torch.randn_like(next_actions) * args.policy_noise)
            .clamp(-args.noise_clip, args.noise_clip)
            * action_scale
        )
        next_actions = (next_actions + noise).clamp(nets["action_low"], nets["action_high"])
        min_q_next = torch.min(
            nets["qf1_target"](b_next_obs, next_actions),
            nets["qf2_target"](b_next_obs, next_actions),
        )
        target_q = b_rewards + (1 - b_dones) * args.gamma * min_q_next

    qf_loss = torch.nn.functional.mse_loss(qf1(b_obs, b_actions), target_q) + \
        torch.nn.functional.mse_loss(qf2(b_obs, b_actions), target_q)
    nets["q_optimizer"].zero_grad()
    qf_loss.backward()
    nets["q_optimizer"].step()

    actor_loss = -qf1(b_obs, joint_policy_actions(actors, b_obs)).mean()
    nets["actor_optimizer"].zero_grad()
    actor_loss.backward()
    nets["actor_optimizer"].step()

    assert any(not torch.allclose(b, p) for b, p in zip(q_params_before, qf1.parameters()))
    assert any(not torch.allclose(b, p) for b, p in zip(actor_params_before, actors.parameters()))
    assert torch.isfinite(qf_loss) and torch.isfinite(actor_loss)


def test_checkpoint_manager_keeps_latest_and_best(tmp_path, args, device):
    nets = build_agents(args, device)
    manager = CheckpointManager(tmp_path, metric_key="return_mean")
    payload = build_checkpoint_payload(nets, args)

    manager.save(payload, global_step=10, eval_stats={"return_mean": 1.0})
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / "best.pt").exists()

    manager.save(payload, global_step=20, eval_stats={"return_mean": 0.5})
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    # best should still be step 10
    best = torch.load(tmp_path / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert best["global_step"] == 10
    assert best["checkpoint_metric_value"] == 1.0
    # only two checkpoint files should exist
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == ["best.pt", "latest.pt"]


def test_run_manifest_written(tmp_path, args):
    manifest = RunManifest(
        run_name="test_run",
        run_dir=tmp_path,
        args=vars(args),
        environment={"python": "test"},
    )
    manifest.write()
    data = json.loads((tmp_path / "run_manifest.json").read_text())
    assert data["run_name"] == "test_run"
    assert data["args"]["n_pistons"] == args.n_pistons
    assert data["finished"] is False


def test_append_eval_record(tmp_path):
    append_eval_record(tmp_path, 100, {"return_mean": 12.3, "success_rate": 0.5}, 42.0)
    lines = (tmp_path / "eval_history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["global_step"] == 100
    assert record["return_mean"] == 12.3
    assert record["wallclock_seconds"] == 42.0


def test_gradient_norm_reports_nonzero_after_backward(args, device):
    nets = build_agents(args, device)
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    obs = torch.rand(2, n, c, h, w)
    actions = torch.rand(2, n, nets["act_dim"])
    loss = nets["qf1"](obs, actions).mean()
    loss.backward()
    norm = gradient_norm(nets["qf1"].parameters())
    assert norm > 0.0


def test_bootstrap_done_stops_on_timeout_and_success():
    """Pistonball max_cycles truncations must stop TD bootstrapping."""
    assert bootstrap_done(terminated=False, truncated=False) == 0.0
    assert bootstrap_done(terminated=True, truncated=False) == 1.0
    assert bootstrap_done(terminated=False, truncated=True) == 1.0
    assert bootstrap_done(terminated=True, truncated=True) == 1.0


def test_clip_gradients_reduces_norm(args, device):
    nets = build_agents(args, device)
    n, c, h, w = args.n_pistons, args.frame_stack, args.frame_size, args.frame_size
    obs = torch.rand(2, n, c, h, w)
    actions = torch.rand(2, n, nets["act_dim"])
    loss = nets["qf1"](obs, actions).mean() * 1000.0
    loss.backward()
    before = gradient_norm(nets["qf1"].parameters())
    clip_gradients(nets["qf1"].parameters(), max_grad_norm=1e-3)
    after = gradient_norm(nets["qf1"].parameters())
    assert before > after
    assert after <= 1e-3 + 1e-6


def test_checkpoint_manager_tracks_best_stats(tmp_path, args, device):
    nets = build_agents(args, device)
    manager = CheckpointManager(tmp_path, metric_key="return_mean")
    payload = build_checkpoint_payload(nets, args)
    manager.save(payload, global_step=10, eval_stats={"return_mean": 1.0, "success_rate": 0.2})
    manager.save(payload, global_step=20, eval_stats={"return_mean": 0.5, "success_rate": 0.0})
    assert manager.best_step == 10
    assert manager.best_stats["success_rate"] == 0.2


def test_maybe_restore_best_checkpoint(tmp_path, args, device):
    from matd3_pistonball import maybe_restore_best_checkpoint

    nets = build_agents(args, device)
    manager = CheckpointManager(tmp_path, metric_key="return_mean")
    # Mutate actors, save as "best", mutate again, then restore.
    with torch.no_grad():
        for p in nets["actors"].parameters():
            p.fill_(0.0)
    manager.save(build_checkpoint_payload(nets, args), global_step=5, eval_stats={"return_mean": 2.0})
    with torch.no_grad():
        for p in nets["actors"].parameters():
            p.fill_(1.0)
    step = maybe_restore_best_checkpoint(manager, nets, device)
    assert step == 5
    assert all(torch.allclose(p, torch.zeros_like(p)) for p in nets["actors"].parameters())


def test_action_saturation_fraction():
    actions = np.array([[1.0], [-0.99], [0.0], [0.5]], dtype=np.float32)
    assert action_saturation_fraction(actions, threshold=0.95, action_bound=1.0) == pytest.approx(0.5)


def test_exploration_noise_decays():
    args = Args(learning_starts=10, exploration_noise=0.2, exploration_noise_end=0.05, exploration_noise_decay_steps=100)
    assert current_exploration_noise(args, 0) == pytest.approx(0.2)
    assert current_exploration_noise(args, 10) == pytest.approx(0.2)
    assert current_exploration_noise(args, 60) == pytest.approx(0.125)
    assert current_exploration_noise(args, 110) == pytest.approx(0.05)


def test_build_run_name_includes_experiment_tag():
    args = Args(experiment="sanity5 shared!", share_actors=True, random_drop=False, n_pistons=5, seed=2)
    name = build_run_name(args)
    assert name.startswith("sanity5_shared__static__shared__nodrop__n5__seed2__")
    assert sanitize_experiment_tag("a/b c") == "a_b_c"
