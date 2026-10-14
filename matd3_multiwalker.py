"""MATD3 on PettingZoo Multiwalker (vector obs) for contextual MARL.

Pistonball physics-OOD is too easy for a visual policy. Multiwalker is a continuous
cooperative locomotion task where gravity, package mass, and terrain friction
change the dynamics. Observations are 31-D vectors (not pixels).

``--context-mode hidden``: physics is sampled/swept but never given to the nets.
``--context-mode concat``: normalized physics vector is concatenated (oracle at eval).
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import matd3_pistonball as base
from envs.multiwalker import (
    MULTIWALKER_PHYSICS_KEYS,
    TrainingEnvironment,
    apply_context_after_reset,
    apply_context_to_env,
    default_context,
    make_env,
    stack_obs,
)
from evaluation.context_sampling import (
    EVAL_SEED_OFFSET,
    SWEEP_SEED_OFFSET,
    eval_points_1d,
    make_context_rng,
    summarize_context_sweep,
)
from evaluation.rliable_utils import save_rliable_scores, summarize_episode_scores

Context = dict
ContextMode = Literal["hidden", "concat"]
TrainingRegime = Literal["static", "varying"]
TrainContextSampling = Literal["random", "grid"]
CONTEXT_KEYS: tuple[str, ...] = MULTIWALKER_PHYSICS_KEYS


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    experiment: str = "mw3"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "contextual-marl"
    wandb_entity: str | None = None
    save_model: bool = False
    checkpoint_eval: bool = True
    checkpoint_metric: str = "return_mean"
    performance_threshold: float | None = 50.0
    rolling_return_window: int = 10
    log_gradient_norms: bool = False
    method_tag: str = "hidden_context"
    reference_return: float = 300.0
    env_name: str = "multiwalker"

    n_walkers: int = 3
    max_cycles: int = 500
    terrain_length: int = 75
    position_noise: float = 1e-3
    angle_noise: float = 1e-3
    forward_reward: float = 1.0
    terminate_reward: float = -100.0
    fall_reward: float = -10.0
    time_penalty: float = 0.0
    """per-step reward added after each env.step (use negative to punish standing)"""
    shared_reward: bool = True
    terminate_on_fall: bool = True
    remove_on_fall: bool = True

    gravity: float = -10.0
    package_mass: float = 1.0
    terrain_friction: float = 2.5

    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    actor_lr: float = 1e-4
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    policy_noise: float = 0.2
    exploration_noise: float = 0.2
    exploration_noise_end: float = 0.05
    exploration_noise_decay_steps: int = 100_000
    learning_starts: int = 10_000
    policy_frequency: int = 2
    noise_clip: float = 0.5
    max_grad_norm: float = 10.0
    share_actors: bool = True
    action_saturation_threshold: float = 0.95
    hidden_dim: int = 256

    eval_frequency: int = 25_000
    eval_episodes: int = 3
    transfer_eval: bool = False
    id_factor_low: float = 0.75
    id_factor_high: float = 1.5
    eval_factor_grid: tuple[float, ...] = (0.5, 0.6, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0)
    train_vary_params: tuple[str, ...] = MULTIWALKER_PHYSICS_KEYS
    eval_vary_params: tuple[str, ...] = MULTIWALKER_PHYSICS_KEYS
    training_regime: TrainingRegime = "varying"
    train_context_sampling: TrainContextSampling = "random"

    context_mode: ContextMode = "hidden"
    context_to_actor: bool = True
    context_to_critic: bool = True


def context_feature_dim(args: Args) -> int:
    return 0 if args.context_mode != "concat" else len(CONTEXT_KEYS)


def context_to_vector(context: Context, reference: Context) -> np.ndarray:
    vec = np.empty(len(CONTEXT_KEYS), dtype=np.float32)
    for i, key in enumerate(CONTEXT_KEYS):
        ref = float(reference[key])
        scale = abs(ref) if abs(ref) > 1e-8 else 1.0
        vec[i] = float(context[key]) / scale
    return vec


def clip_context_vector(vec: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clamp each feature's magnitude into the training factor window.

    ``context_to_vector`` divides by ``abs(reference)``, so mass and friction
    are positive factors and gravity is a negative factor (nominal gravity is
    ``-1``). Training only sees magnitudes inside ``[low, high]`` (default
    0.75–1.5). Out-of-distribution eval can feed magnitudes 0.25 or 4. Clipping
    the magnitude, and keeping the sign, tests whether the OOD drop is the
    network extrapolating the vector. The simulator keeps the true physics.
    In-distribution points are unchanged.
    """
    arr = np.asarray(vec, dtype=np.float32)
    sign = np.sign(arr)
    sign = np.where(sign == 0.0, 1.0, sign).astype(np.float32)
    magnitude = np.clip(np.abs(arr), float(low), float(high))
    return (sign * magnitude).astype(np.float32)


def build_run_name(args: Args) -> str:
    tag = base.sanitize_experiment_tag(args.experiment)
    share = "shared" if args.share_actors else "indep"
    return (
        f"{tag}__{args.context_mode}__{args.training_regime}__{share}__n{args.n_walkers}"
        f"__seed{args.seed}__{int(time.time())}"
    )


class VectorReplayBuffer:
    """Float32 vector-obs replay (shared reward). Optional per-transition context."""

    def __init__(self, capacity, n_agents, obs_dim, act_dim, context_dim, device, rng=None):
        self.capacity = capacity
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pos = 0
        self.full = False
        self.context_dim = context_dim
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.contexts = np.zeros((capacity, max(context_dim, 1)), dtype=np.float32)

    def add(self, obs, next_obs, actions, reward, done, context=None):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = actions
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        if self.context_dim > 0:
            if context is None:
                raise ValueError("context vector required when context_dim > 0")
            self.contexts[self.pos, : self.context_dim] = context
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = 0
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample(self, batch_size):
        idx = self.rng.integers(0, len(self), size=batch_size)
        to = lambda a: torch.as_tensor(a, device=self.device)
        ctx = to(self.contexts[idx, : self.context_dim]) if self.context_dim > 0 else to(np.zeros((batch_size, 0), dtype=np.float32))
        return to(self.obs[idx]), to(self.next_obs[idx]), to(self.actions[idx]), to(self.rewards[idx]), to(self.dones[idx]), ctx

    def memory_gb(self) -> float:
        arrays = (self.obs, self.next_obs, self.actions, self.rewards, self.dones, self.contexts)
        return sum(a.nbytes for a in arrays) / 1e9


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, n_agents, use_agent_id, context_dim, action_low, action_high, context_to_actor):
        super().__init__()
        self.use_agent_id = use_agent_id
        self.n_agents = n_agents
        self.context_to_actor = context_to_actor and context_dim > 0
        self.context_dim = context_dim if self.context_to_actor else 0
        in_dim = obs_dim + (n_agents if use_agent_id else 0) + self.context_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, act_dim)
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0))

    def forward(self, obs, agent_index=None, context=None):
        parts = [obs]
        if self.use_agent_id:
            if agent_index is None:
                raise ValueError("shared actor requires agent_index")
            if isinstance(agent_index, int):
                idx = torch.full((obs.shape[0],), agent_index, device=obs.device, dtype=torch.long)
            else:
                idx = agent_index
            parts.append(F.one_hot(idx, self.n_agents).float())
        if self.context_to_actor:
            if context is None:
                raise ValueError("contextual actor requires context")
            parts.append(context)
        x = F.relu(self.fc1(torch.cat(parts, dim=-1)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x)) * self.action_scale + self.action_bias


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, n_agents, act_dim, hidden_dim, context_dim, context_to_critic):
        super().__init__()
        self.context_to_critic = context_to_critic and context_dim > 0
        self.context_dim = context_dim if self.context_to_critic else 0
        in_dim = n_agents * (obs_dim + act_dim) + self.context_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, actions, context=None):
        b = obs.shape[0]
        parts = [obs.reshape(b, -1), actions.reshape(b, -1)]
        if self.context_to_critic:
            if context is None:
                raise ValueError("contextual critic requires context")
            parts.append(context)
        x = F.relu(self.fc1(torch.cat(parts, dim=1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def joint_policy_actions(actors, obs, context=None):
    b, n = obs.shape[:2]
    actor0 = actors[0]
    if actor0.use_agent_id:
        flat_obs = obs.reshape(b * n, *obs.shape[2:])
        agent_index = torch.arange(n, device=obs.device).view(1, n).expand(b, n).reshape(b * n)
        flat_ctx = None
        if context is not None and getattr(actor0, "context_to_actor", False):
            flat_ctx = context.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)
        return actor0(flat_obs, agent_index=agent_index, context=flat_ctx).reshape(b, n, -1)
    return torch.stack([actor(obs[:, i], context=context) for i, actor in enumerate(actors)], dim=1)


@torch.no_grad()
def evaluate(
    actors,
    args,
    context,
    agents,
    device,
    seed,
    n_episodes,
    env=None,
    reference_context=None,
    render_mode=None,
    clip_context_to_id: bool = False,
):
    owns_env = env is None
    if owns_env:
        env = make_env(args, context, render_mode=render_mode)
    apply_context_to_env(env, context)
    ctx_dim = context_feature_dim(args)
    ref = reference_context or default_context(args)
    raw_vec = context_to_vector(context, ref) if ctx_dim > 0 else None
    ctx_vec = raw_vec
    if clip_context_to_id and raw_vec is not None:
        ctx_vec = clip_context_vector(
            raw_vec,
            float(getattr(args, "id_factor_low", 0.75)),
            float(getattr(args, "id_factor_high", 1.5)),
        )
    episode_returns, episode_lengths, episode_successes = [], [], []
    episode_action_stds, episode_per_agent_var, episode_saturations = [], [], []
    physics_probe = None
    try:
        for ep in range(n_episodes):
            obs_dict, _ = env.reset(seed=seed + ep)
            probe = apply_context_after_reset(env, context)
            if physics_probe is None:
                physics_probe = probe
            ep_return, ep_len, success = 0.0, 0, 0.0
            ep_action_stds, ep_per_agent_vars, ep_saturations = [], [], []
            while env.agents:
                obs = torch.as_tensor(stack_obs(obs_dict, agents), device=device)
                ctx_t = torch.as_tensor(ctx_vec, device=device).float().unsqueeze(0) if ctx_vec is not None else None
                actions = joint_policy_actions(actors, obs.unsqueeze(0), context=ctx_t).squeeze(0)
                actions_np = actions.cpu().numpy()
                ep_action_stds.append(float(np.std(actions_np)))
                ep_saturations.append(base.action_saturation_fraction(
                    actions_np, threshold=args.action_saturation_threshold,
                    action_bound=float(actors[0].action_scale.item() + abs(actors[0].action_bias.item())),
                ))
                action_dict = {a: actions_np[i] for i, a in enumerate(agents)}
                obs_dict, rewards, terms, truncs, _ = env.step(action_dict)
                rew_vals = list(rewards.values())
                ep_return += float(np.mean(rew_vals))
                ep_per_agent_vars.append(float(np.var(rew_vals)))
                ep_len += 1
                if any(terms.values()):
                    success = 1.0
            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)
            episode_successes.append(success)
            episode_action_stds.append(float(np.mean(ep_action_stds)) if ep_action_stds else 0.0)
            episode_per_agent_var.append(float(np.mean(ep_per_agent_vars)) if ep_per_agent_vars else 0.0)
            episode_saturations.append(float(np.mean(ep_saturations)) if ep_saturations else 0.0)
    finally:
        if owns_env:
            env.close()
    summary = summarize_episode_scores(episode_returns, reference_return=args.reference_return)
    summary.update({
        "length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "action_std_mean": float(np.mean(episode_action_stds)) if episode_action_stds else 0.0,
        "action_saturation_mean": float(np.mean(episode_saturations)) if episode_saturations else 0.0,
        "per_agent_return_variance_mean": float(np.mean(episode_per_agent_var)) if episode_per_agent_var else 0.0,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes,
        "n_episodes": float(len(episode_returns)),
        "physics_probe": physics_probe,
    })
    if raw_vec is not None:
        summary["context_vector"] = [float(x) for x in raw_vec]
        if clip_context_to_id:
            summary["context_vector_clipped"] = [float(x) for x in ctx_vec]
            summary["context_clip_to_id"] = True
    return summary


def build_agents(args: Args, device: torch.device):
    probe = make_env(args, default_context(args))
    probe.reset(seed=args.seed)
    agents = list(probe.possible_agents)
    n_agents = len(agents)
    act_space = probe.action_space(agents[0])
    obs_space = probe.observation_space(agents[0])
    act_dim = int(np.prod(act_space.shape))
    obs_dim = int(np.prod(obs_space.shape))
    action_low = float(act_space.low[0])
    action_high = float(act_space.high[0])
    probe.close()
    ctx_dim = context_feature_dim(args)
    to_actor = args.context_to_actor and ctx_dim > 0
    to_critic = args.context_to_critic and ctx_dim > 0

    def make_actor():
        return MLPActor(obs_dim, act_dim, args.hidden_dim, n_agents, args.share_actors, ctx_dim, action_low, action_high, to_actor).to(device)

    if args.share_actors:
        actors = nn.ModuleList([make_actor()])
        target_actors = nn.ModuleList([make_actor()])
    else:
        actors = nn.ModuleList([make_actor() for _ in range(n_agents)])
        target_actors = nn.ModuleList([make_actor() for _ in range(n_agents)])

    def make_critic():
        return MLPCritic(obs_dim, n_agents, act_dim, args.hidden_dim, ctx_dim, to_critic).to(device)

    qf1, qf2 = make_critic(), make_critic()
    qf1_target, qf2_target = make_critic(), make_critic()
    target_actors.load_state_dict(actors.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(actors.parameters(), lr=args.actor_lr)
    return {
        "agents": agents, "n_agents": n_agents, "act_dim": act_dim, "obs_dim": obs_dim,
        "action_low": action_low, "action_high": action_high, "context_dim": ctx_dim,
        "actors": actors, "target_actors": target_actors,
        "qf1": qf1, "qf2": qf2, "qf1_target": qf1_target, "qf2_target": qf2_target,
        "q_optimizer": q_optimizer, "actor_optimizer": actor_optimizer,
    }


def train(args: Args) -> str:
    if args.context_mode == "concat":
        args.method_tag = args.method_tag if args.method_tag != "hidden_context" else "context_concat"
    elif args.context_mode != "hidden":
        raise ValueError(f"unsupported context_mode={args.context_mode!r}")

    run_name = build_run_name(args)
    from evaluation.run_paths import resolve_run_dir

    run_dir = resolve_run_dir(args, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = base.RunManifest(run_name=run_name, run_dir=run_dir, args=vars(args), environment=base.collect_environment_info())
    manifest.write()
    checkpoint_manager = base.CheckpointManager(run_dir, metric_key=args.checkpoint_metric) if args.checkpoint_eval else None
    writer = SummaryWriter(str(run_dir))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])))
    replay_rng, rollout_rng = base.seed_run(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    reference_context = default_context(args)
    nets = build_agents(args, device)
    agents = nets["agents"]
    actors, target_actors = nets["actors"], nets["target_actors"]
    qf1, qf2 = nets["qf1"], nets["qf2"]
    qf1_target, qf2_target = nets["qf1_target"], nets["qf2_target"]
    q_optimizer, actor_optimizer = nets["q_optimizer"], nets["actor_optimizer"]
    action_low, action_high = nets["action_low"], nets["action_high"]
    action_scale = actors[0].action_scale
    ctx_dim = nets["context_dim"]
    rb = VectorReplayBuffer(args.buffer_size, nets["n_agents"], nets["obs_dim"], nets["act_dim"], ctx_dim, device, rng=replay_rng)
    if args.learning_starts >= args.buffer_size:
        raise ValueError("learning_starts must be < buffer_size")
    print(f"env=multiwalker n={args.n_walkers} mode={args.context_mode} regime={args.training_regime} ctx_dim={ctx_dim} buffer={rb.memory_gb():.3f} GB")
    training_env = TrainingEnvironment(args, rng=make_context_rng(args.seed))
    obs_dict, episode_context = training_env.reset(seed=args.seed)
    episode_ctx_vec = context_to_vector(episode_context, reference_context) if ctx_dim > 0 else None
    obs = stack_obs(obs_dict, agents)
    ep_return, ep_len = 0.0, 0
    ep_per_agent_var_sum, ep_var_steps = 0.0, 0
    start_time = time.time()
    recent_returns: deque[float] = deque(maxlen=args.rolling_return_window)
    steps_to_threshold = None
    actor_loss = torch.tensor(0.0)

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = rollout_rng.uniform(action_low, action_high, size=(nets["n_agents"], nets["act_dim"])).astype(np.float32)
            explore_std = args.exploration_noise
        else:
            explore_std = base.current_exploration_noise(args, global_step)
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                ctx_t = torch.as_tensor(episode_ctx_vec, device=device).float().unsqueeze(0) if episode_ctx_vec is not None else None
                actions = joint_policy_actions(actors, obs_t, context=ctx_t).squeeze(0)
                actions += torch.normal(0, action_scale * explore_std, size=actions.shape, device=device)
                actions = actions.cpu().numpy().clip(action_low, action_high).astype(np.float32)

        next_obs_dict, rewards, terms, truncs, _ = training_env.step({a: actions[i] for i, a in enumerate(agents)})
        reward = float(np.mean(list(rewards.values())))
        rew_vals = list(rewards.values())
        ep_per_agent_var_sum += float(np.var(rew_vals))
        ep_var_steps += 1
        terminated, truncated = any(terms.values()), any(truncs.values())
        next_obs = stack_obs(next_obs_dict, agents)
        ep_return += reward
        ep_len += 1
        rb.add(obs, next_obs, actions, reward, base.bootstrap_done(terminated, truncated), context=episode_ctx_vec if ctx_dim > 0 else None)
        obs = next_obs

        if terminated or truncated:
            ep_var_mean = ep_per_agent_var_sum / max(ep_var_steps, 1)
            writer.add_scalar("performance/team_episodic_return", ep_return, global_step)
            writer.add_scalar("performance/episodic_length", ep_len, global_step)
            writer.add_scalar("performance/success", float(terminated), global_step)
            base.append_training_episode_record(run_dir, global_step, ep_return, ep_len, bool(terminated), ep_var_mean, context=episode_context)
            recent_returns.append(ep_return)
            if len(recent_returns) == recent_returns.maxlen:
                rolling_mean = float(np.mean(recent_returns))
                writer.add_scalar("sample_efficiency/rolling_return_mean", rolling_mean, global_step)
                if args.performance_threshold is not None and steps_to_threshold is None and rolling_mean >= args.performance_threshold:
                    steps_to_threshold = global_step
                    writer.add_scalar("sample_efficiency/steps_to_threshold", steps_to_threshold, global_step)
                    manifest.steps_to_threshold = steps_to_threshold
            print(f"global_step={global_step}, episodic_return={ep_return:.2f}, length={ep_len}")
            obs_dict, episode_context = training_env.start_episode(seed=args.seed + global_step)
            episode_ctx_vec = context_to_vector(episode_context, reference_context) if ctx_dim > 0 else None
            obs = stack_obs(obs_dict, agents)
            ep_return, ep_len = 0.0, 0
            ep_per_agent_var_sum, ep_var_steps = 0.0, 0

        if global_step > args.learning_starts:
            b_obs, b_next_obs, b_actions, b_rewards, b_dones, b_ctx = rb.sample(args.batch_size)
            with torch.no_grad():
                next_actions = joint_policy_actions(target_actors, b_next_obs, context=b_ctx)
                noise = (torch.randn_like(next_actions) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip) * action_scale
                next_actions = (next_actions + noise).clamp(action_low, action_high)
                min_q_next = torch.min(qf1_target(b_next_obs, next_actions, context=b_ctx), qf2_target(b_next_obs, next_actions, context=b_ctx))
                next_q_value = b_rewards + (1 - b_dones) * args.gamma * min_q_next
            qf_loss = F.mse_loss(qf1(b_obs, b_actions, context=b_ctx), next_q_value) + F.mse_loss(qf2(b_obs, b_actions, context=b_ctx), next_q_value)
            q_params = list(qf1.parameters()) + list(qf2.parameters())
            q_optimizer.zero_grad()
            qf_loss.backward()
            base.clip_gradients(q_params, args.max_grad_norm)
            q_optimizer.step()
            if global_step % args.policy_frequency == 0:
                actor_actions = joint_policy_actions(actors, b_obs, context=b_ctx)
                actor_loss = -qf1(b_obs, actor_actions, context=b_ctx).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                base.clip_gradients(actors.parameters(), args.max_grad_norm)
                actor_optimizer.step()
                base.polyak_update(actors, target_actors, args.tau)
                base.polyak_update(qf1, qf1_target, args.tau)
                base.polyak_update(qf2, qf2_target, args.tau)
            if global_step % 100 == 0:
                base.log_timing_metrics(writer, global_step, start_time, args.total_timesteps)
                base.log_resource_metrics(writer, global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("exploration/noise_std", explore_std, global_step)

        if args.eval_frequency > 0 and global_step > args.learning_starts and global_step % args.eval_frequency == 0:
            stats = evaluate(actors, args, reference_context, agents, device, seed=args.seed + EVAL_SEED_OFFSET, n_episodes=args.eval_episodes, reference_context=reference_context)
            stats["split"] = "id"
            base.log_eval_metrics(writer, "eval", stats, global_step)
            base.append_eval_record(run_dir, global_step, stats, time.time() - start_time)
            if checkpoint_manager is not None:
                saved = checkpoint_manager.save(base.build_checkpoint_payload(nets, args), global_step, stats)
                if "best" in saved:
                    manifest.best_eval = {"step": checkpoint_manager.best_step, "metric": checkpoint_manager.metric_key, "value": checkpoint_manager.best_value, "stats": checkpoint_manager.best_stats}
            manifest.global_step = global_step
            manifest.wallclock_seconds = time.time() - start_time
            manifest.write()

    training_env.close()
    if checkpoint_manager is not None and checkpoint_manager.best_step < 0:
        checkpoint_manager.save(base.build_checkpoint_payload(nets, args), args.total_timesteps, {"return_mean": ep_return})
    if args.transfer_eval:
        base.maybe_restore_best_checkpoint(checkpoint_manager, nets, device)
        points = eval_points_1d(args)
        env = make_env(args, reference_context)
        per_context, scores_by_context = {}, {}
        try:
            for point in points:
                stats = dict(evaluate(actors, args, point.context, agents, device, seed=args.seed + SWEEP_SEED_OFFSET, n_episodes=args.eval_episodes, env=env, reference_context=reference_context))
                stats["context"] = point.context
                stats.update(point.meta())
                per_context[point.name] = stats
                scores_by_context[point.name] = list(stats["episode_returns"])
        finally:
            env.close()
        save_rliable_scores(
            run_dir, algorithm=args.exp_name, method_tag=args.method_tag, seed=args.seed,
            global_step=args.total_timesteps, context_order=[p.name for p in points],
            scores_by_context=scores_by_context, eval_split="id_ood_1d_slice",
            reference_return=args.reference_return, extra={"context_meta": {p.name: p.meta() for p in points}},
        )
        summarize_context_sweep(per_context, points)
    manifest.finished = True
    manifest.global_step = args.total_timesteps
    manifest.wallclock_seconds = time.time() - start_time
    manifest.steps_to_threshold = steps_to_threshold
    manifest.write()
    writer.close()
    return run_name


if __name__ == "__main__":
    train(tyro.cli(Args))
